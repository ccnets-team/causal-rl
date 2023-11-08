import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectory_handler  import BatchTrajectory
from .trainer_utils import compute_gae, compute_end_step, calculate_accumulative_rewards, get_end_future_value, compute_discounted_future_value, masked_mean

class BaseTrainer(TrainingManager, StrategyManager):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
        TrainingManager.__init__(self, optimization_params, networks, target_networks)
        StrategyManager.__init__(self, env_config, exploration_params, normalization_params, device)
        
        self.device = device
        self.curiosity_factor = algorithm_params.curiosity_factor
        self.discount_factor = algorithm_params.discount_factor
        self.use_gae_advantage = algorithm_params.use_gae_advantage
        
    def calculate_advantage(self, estimated_value, expected_value):
        with torch.no_grad():
            advantage = (expected_value - estimated_value)
        return advantage

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        loss = (estimated_value - expected_value).square()
        if mask is not None:
            loss = masked_mean(loss, mask)
        else:
            loss = loss.mean(dim = 0)
        return loss

    def compute_gae_advantage(self, states, rewards, next_states, dones):
        critic = self.get_networks()[0]
        trajectory_states = torch.cat([states, next_states[:, -1:]], dim=1)

        # Calculate cumulative dones to mask out values and rewards after episode ends
        cumulative_dones = torch.cumsum(dones, dim=1)
        trajectory_values = critic(trajectory_states)  # Assuming critic outputs values for each state in the trajectory
        # Zero-out rewards and values after the end of the episode
        advantages = compute_gae(trajectory_values, rewards, cumulative_dones, self.discount_factor)
        return advantages
    
    def select_first_transitions(self, *tensor_sequences: torch.Tensor):
        results = tuple(tensor[:, 0, :].unsqueeze(1) for tensor in tensor_sequences)
    
        # If only one tensor is passed, return the tensor directly instead of a tuple    
        if len(results) == 1:
            return results[0]
        return results

    def calculate_curiosity_rewards(self, intrinsic_value):
        with torch.no_grad():
            curiosity_reward = self.curiosity_factor * intrinsic_value.square()
        return curiosity_reward
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, intrinsic_value: None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory
        rewards += 0 if intrinsic_value is None else self.calculate_curiosity_rewards(intrinsic_value)

        with torch.no_grad():
            if self.use_gae_advantage:
                advantage = self.compute_gae_advantage(states, rewards, next_states, dones)
                expected_value = advantage + estimated_value
            else:
                expected_value = self.calculate_expected_value(rewards, next_states, dones)
                advantage = self.calculate_advantage(estimated_value, expected_value)
                
        return expected_value, advantage

    def calculate_expected_value(self, rewards, next_states, dones):
        # Future values calculated from the trainer's future value function
        future_values = self.trainer_calculate_future_value(next_states) # This function needs to be defined elsewhere
        
        # Compute the end step from the dones tensor
        end_step = compute_end_step(dones)

        # Get the future value at the end step
        future_value_at_end_step = get_end_future_value(future_values, end_step)

        # Compute the sequence length from rewards
        seq_len = rewards.size(1)
        discount_factors = compute_discounted_future_value(end_step, self.discount_factor, seq_len)
        
        # Calculate the discount factors for each transition
        accumulative_rewards = calculate_accumulative_rewards(rewards, end_step, self.discount_factor)
        
        # Calculate the expected values
        sequence_dones = dones.any(dim=1, keepdim=True).expand_as(dones).type(dones.dtype)
        expected_values = accumulative_rewards + (1 - sequence_dones) * discount_factors * future_value_at_end_step
        
        return expected_values


        
    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
