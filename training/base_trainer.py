import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectory_handler  import BatchTrajectory
from .trainer_utils import compute_gae, calculate_accumulative_rewards, compute_discounted_future_value, create_padding_mask_before_dones, calculate_advantage

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
        num_td_steps = algorithm_params.num_td_steps
        self.discount_factors = compute_discounted_future_value(self.discount_factor, num_td_steps).to(self.device)

    def calculate_curiosity_rewards(self, intrinsic_value):
        with torch.no_grad():
            curiosity_reward = self.curiosity_factor * intrinsic_value.square()
        return curiosity_reward

    def calculate_gae_advantage(self, trajectory_states, rewards, trajectory_mask):

        # Calculate cumulative dones to mask out values and rewards after episode ends
        trajectory_values = self.trainer_calculate_future_value(trajectory_states, trajectory_mask, use_target=False)  # Assuming critic outputs values for each state in the trajectory
        # Zero-out rewards and values after the end of the episode
        mask = trajectory_mask[:,:-1]
        
        advantages = compute_gae(trajectory_values, rewards, mask, self.discount_factor)
        return advantages

    def calculate_expected_value(self, next_states, rewards, mask):
        # Compute the end step from the dones tensor
        
        # Future values calculated from the trainer's future value function
        future_values = self.trainer_calculate_future_value(next_states, mask, use_target=True) # This function needs to be defined elsewhere

        # # Get the future value at the end step
        future_value_at_end_step = future_values[:,-1:] 

        # Calculate the discount factors for each transition
        accumulative_rewards = calculate_accumulative_rewards(rewards, self.discount_factor, mask)

        # Compute the sequence length from rewardsa
        seq_len = rewards.size(1)
        discount_factors = self.discount_factors[:,-seq_len:]
        
        # Calculate the expected values
        sequence_dones = 1 - mask.all(dim=1, keepdim=True).expand_as(mask).type(mask.dtype)
        expected_values = accumulative_rewards + (1 - sequence_dones) * discount_factors * future_value_at_end_step
        return expected_values
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, intrinsic_value: torch.Tensor = None):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory
        rewards += 0 if intrinsic_value is None else self.calculate_curiosity_rewards(intrinsic_value)

        with torch.no_grad():
            
            if self.use_gae_advantage:
                trajectory_states = torch.cat([states, next_states[:, -1:]], dim=1)
                mask = create_padding_mask_before_dones(dones)
                trajectory_mask = torch.cat([mask, mask[:, -1:]], dim=1)
                advantage = self.calculate_gae_advantage(trajectory_states, rewards, trajectory_mask)
                expected_value = advantage + estimated_value
            else:
                expected_value = self.calculate_expected_value(next_states, rewards, mask)
                advantage = calculate_advantage(estimated_value, expected_value)
                
        return expected_value, advantage

    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None, use_target = False):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
