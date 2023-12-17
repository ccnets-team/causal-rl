import torch
from training.managers.training_manager import TrainingManager 
from training.managers.exploration_manager import ExplorationUtils 
from training.managers.normalization_manager import NormalizationUtils 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectories  import BatchTrajectory
from .trainer_utils import calculate_gae_returns, calculate_lambda_returns, compute_discounted_future_value, scale_advantage

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils()
        self._init_trainer_specific_params()

    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.exploration_params, 
         self.memory_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        training_start_step = self._compute_training_start_step()
        total_iterations = max(self.exploration_params.max_steps - training_start_step, 0)
        TrainingManager.__init__(self, networks, target_networks, self.optimization_params.lr, 
                                 self.optimization_params.clip_grad_range, self.network_params.tau, total_iterations)
        self.device = device
        self.discount_factors = compute_discounted_future_value(self.discount_factor, self.num_td_steps).to(self.device)

    def _init_normalization_utils(self, env_config, device):
        NormalizationUtils.__init__(self, env_config, self.normalization_params, device)

    def _init_exploration_utils(self):
        ExplorationUtils.__init__(self, self.exploration_params)

    def _init_trainer_specific_params(self):
        self.use_gae_advantage = self.algorithm_params.use_gae_advantage
        self.num_td_steps = self.algorithm_params.num_td_steps
        self.use_target_network = self.network_params.use_target_network
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor
        self.advantage_normalizer = self.normalization_params.advantage_normalizer
        self.advantage_threshold = self.normalization_params.advantage_threshold 

    def _compute_training_start_step(self):
        training_start_step = self.training_params.early_training_start_step
        if training_start_step is None:
            batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
            training_start_step = self.memory_params.buffer_size // int(batch_size_ratio)
        return training_start_step
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, mask: torch.Tensor):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory

        gamma = self.discount_factor
        lambd = self.advantage_lambda
        with torch.no_grad():
            trajectory_states = torch.cat([states, next_states[:,-1:]], dim = 1)
            trajectory_mask = torch.cat([mask, mask[:,-1:]], dim = 1)
            trajectory_values = self.trainer_calculate_future_value(trajectory_states, trajectory_mask, use_target=self.use_target_network)
            
            if self.use_gae_advantage:
                advantage = calculate_gae_returns(trajectory_values, rewards, dones, gamma, lambd)
                expected_value = (advantage + estimated_value)
            else:
                expected_value = calculate_lambda_returns(trajectory_values, rewards, dones, gamma, lambd)
                advantage = (expected_value - estimated_value)
                
        expected_value, advantage = scale_advantage(expected_value, advantage, self.advantage_normalizer, self.advantage_threshold)
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
    
