import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.exploration_manager import ExplorationUtils 
from training.managers.normalization_manager import NormalizationUtils 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectories  import BatchTrajectory
from .trainer_utils import calculate_gae_returns, calculate_lambda_returns, compute_discounted_future_value, create_padding_mask_before_dones, calculate_advantage

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
        TrainingManager.__init__(self, network_params, optimization_params, networks, target_networks)
        NormalizationUtils.__init__(self, env_config, normalization_params, device)
        ExplorationUtils.__init__(self, exploration_params)
        
        self.device = device
        self.curiosity_factor = algorithm_params.curiosity_factor
        self.discount_factor = algorithm_params.discount_factor
        self.use_gae_advantage = algorithm_params.use_gae_advantage
        self.num_td_steps = algorithm_params.num_td_steps
        self.use_target_network = algorithm_params.use_target_network
        
        self.advantage_lambda = algorithm_params.advantage_lambda
            
        self.discount_factors = compute_discounted_future_value(self.discount_factor, self.num_td_steps).to(self.device)

    def calculate_curiosity_rewards(self, intrinsic_value):
        with torch.no_grad():
            curiosity_reward = self.curiosity_factor * intrinsic_value.square()
        return curiosity_reward
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, mask: torch.Tensor, intrinsic_value: torch.Tensor = None):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory
        rewards += 0 if intrinsic_value is None else self.calculate_curiosity_rewards(intrinsic_value)

        discount_factor = self.discount_factor
        with torch.no_grad():
            trajectory_state = torch.cat([states, next_states[:, -1:]], dim=1)
            trajectory_mask = torch.cat([mask, torch.ones_like(mask[:, -1:])], dim=1)
            if self.use_target_network:
                trajectory_value = self.trainer_calculate_future_value(trajectory_state, trajectory_mask, use_target=self.use_target_network)
                trajectory_value[:,:-1] = estimated_value
            else:
                trajectory_value = self.trainer_calculate_future_value(trajectory_state, trajectory_mask, use_target=self.use_target_network)
            
            if self.use_gae_advantage:
                gae_lambda = self.advantage_lambda
                advantage = calculate_gae_returns(trajectory_value, rewards, dones, discount_factor, gae_lambda)
                expected_value = (advantage + estimated_value)
            else:
                td_lambda = self.advantage_lambda
                expected_value = calculate_lambda_returns(trajectory_value, rewards, dones, discount_factor, td_lambda)
                advantage = (expected_value - estimated_value)
                
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
    
