import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from training.managers.utils.advantage_scaler import scale_advantage
from abc import abstractmethod
from nn.roles.actor_network import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import compute_gae, get_discounted_rewards
from utils.structure.trajectory_handler  import BatchTrajectory


    
class BaseTrainer(TrainingManager, StrategyManager):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
        TrainingManager.__init__(self, optimization_params, networks, target_networks)
        StrategyManager.__init__(self, env_config, algorithm_params, exploration_params, normalization_params, device)
        
        self.env_config: EnvConfig = env_config
        
        self.trainer_name = trainer_name
        self.discount_factor = algorithm_params.discount_factor
        self.batch_size = training_params.batch_size
        self.reward_scale = normalization_params.reward_scale
        self.advantage_scaler = normalization_params.advantage_scaler
        
        self.use_gae_advantage = algorithm_params.use_gae_advantage
        self.use_sequence_batch = algorithm_params.use_sequence_batch
        self.samples_per_step  = env_config.samples_per_step 

        self.num_td_steps = algorithm_params.num_td_steps
        self.buffer_type = memory_params.buffer_type

        self.device = device
        
    def calculate_advantage(self, estimated_value, expected_value):
        with torch.no_grad():
            advantage = (expected_value - estimated_value)
            advantage = scale_advantage(advantage, self.advantage_scaler)
        return advantage

    def calculate_value_loss(self, estimated_value, expected_value):
        return F.mse_loss(estimated_value, expected_value)
    
    def get_discount_factor(self):
        return self.discount_factor 
    
    def get_discount_factors(self):
        return (self.discount_factor ** torch.arange(self.num_td_steps).float()).to(self.device)

    def compute_gae_advantage(self, states, rewards, next_states, dones):
        critic = self.get_networks()[0]
        trajectory_states = torch.cat([states, next_states[:, -1:]], dim=1)
        trajectory_values = critic(trajectory_states)  # Assuming critic outputs values for each state in the trajectory
        advantages = compute_gae(trajectory_values, rewards, dones).detach()
        return advantages
    
    def select_first_transitions(self, *tensor_sequences: torch.Tensor):
        results = tuple(tensor[:, 0, :].unsqueeze(1) for tensor in tensor_sequences)
    
        # If only one tensor is passed, return the tensor directly instead of a tuple    
        if len(results) == 1:
            return results[0]
        return results
    
    def select_trajectory_segment(self, trajectory: BatchTrajectory):
        """Extract and process the trajectory based on the use_sequence_batch flag."""

        if self.use_sequence_batch or self.use_gae_advantage:
            return trajectory
        else:
            states = trajectory.state[:, 0, :].unsqueeze(1)
            actions = trajectory.action[:, 0, :].unsqueeze(1)
            rewards = trajectory.reward[:, 0, :].unsqueeze(1)
            next_states = trajectory.next_state[:, 0, :].unsqueeze(1)
            dones = trajectory.done[:, 0, :].unsqueeze(1)

            return BatchTrajectory(states, actions, rewards, next_states, dones)


        
    def calculate_expected_value(self, rewards, next_states, dones):
        """
        Computes the expected return for transitions.
        Considers immediate rewards and, based on configuration, 
        uses either the entire sequence or just the first transition for future rewards estimation. 
        Potential future rewards are considered only if the episode terminates at the last step.
        """
        batch_size = rewards.shape[0]
        discount_factors = self.get_discount_factors()
        # Reversing discount factors before shaping
        discount_factors = discount_factors.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        discounted_rewards = get_discounted_rewards(rewards, discount_factors)
        # Deriving reversed discounted rewards
        reversed_discount_factors = discount_factors.flip(dims=[1])
        
        # If done is flagged at the end of the trajectory, the mask should be all zeros, otherwise, it should be all ones.
        masks = (1 - dones[:, -1:]).expand_as(dones)
        
        if self.use_sequence_batch:
            # Shape discount_factors for batch computation
            future_values = self.trainer_calculate_future_value(next_states)
            expected_value = discounted_rewards + reversed_discount_factors * masks * future_values
        else:
            # Only consider the first transition
            discounted_reward, reversed_discount_factor, mask, next_state = self.select_first_transitions(discounted_rewards, reversed_discount_factors, masks, next_states)
            future_value = self.trainer_calculate_future_value(next_state)
            expected_value = discounted_reward + reversed_discount_factor * mask * future_value
        return expected_value

    
    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    def compute_values(self, states: torch.Tensor, rewards: torch.Tensor, 
                                             next_states: torch.Tensor, dones: torch.Tensor, 
                                             estimated_value: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute the advantage and expected value."""
        if self.use_gae_advantage:
            with torch.no_grad():
                advantage = self.compute_gae_advantage(states, rewards, next_states, dones)
                expected_value = advantage + estimated_value
        else:
            with torch.no_grad():
                expected_value = self.calculate_expected_value(rewards, next_states, dones)
                advantage = self.calculate_advantage(estimated_value, expected_value)
        return estimated_value, expected_value, advantage 
                    
    @abstractmethod
    def trainer_calculate_future_value(self, next_state):
        pass

    @abstractmethod
    def get_action(self, state, training):
        pass
    
