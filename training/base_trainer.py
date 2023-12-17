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
from .trainer_utils import calculate_gae_returns, calculate_lambda_returns, compute_discounted_future_value

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
            
        total_iterations = exploration_params.max_steps//training_params.train_intervel
        training_start_step = training_params.early_training_start_step
        if training_start_step is None:
            training_start_step = memory_params.buffer_size//int(training_params.batch_size/training_params.replay_ratio) 
        total_iterations = max(exploration_params.max_steps - training_start_step, 0)
            
        TrainingManager.__init__(self, optimization_params, total_iterations, networks, target_networks)
        NormalizationUtils.__init__(self, env_config, normalization_params, device)
        ExplorationUtils.__init__(self, exploration_params)
        
        self.device = device
        self.use_gae_advantage = algorithm_params.use_gae_advantage
        self.num_td_steps = algorithm_params.num_td_steps
        self.use_target_network = network_params.use_target_network
        # normalization_params.
        self.advantage_lambda = algorithm_params.advantage_lambda
        self.discount_factor = algorithm_params.discount_factor
        self.discount_factors = compute_discounted_future_value(self.discount_factor, self.num_td_steps).to(self.device)
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, mask: torch.Tensor):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory

        gamma = self.discount_factor
        lambd = self.advantage_lambda
        with torch.no_grad():
            future_values = self.trainer_calculate_future_value(next_states, mask, use_target=self.use_target_network)
            trajectory_values = torch.cat([torch.zeros_like(estimated_value[:,:1]),  future_values], dim = 1)
            
            if self.use_gae_advantage:
                advantage = calculate_gae_returns(trajectory_values, rewards, dones, gamma, lambd)
                expected_value = (advantage + estimated_value)
            else:
                expected_value = calculate_lambda_returns(trajectory_values, rewards, dones, gamma, lambd)
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
    
