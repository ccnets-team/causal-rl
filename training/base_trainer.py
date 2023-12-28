import torch
from training.managers.training_manager import TrainingManager 
from training.managers.exploration_manager import ExplorationUtils 
from training.managers.normalization_manager import NormalizationUtils 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectories  import BatchTrajectory
from .trainer_utils import calculate_gae_returns, calculate_lambda_returns, compute_discounted_future_value 
from .trainer_utils import adaptive_masked_tensor_reduction, masked_tensor_reduction, apply_seq_mask
from .trainer_utils import create_padding_mask_before_dones, convert_trajectory_data

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils()
        self.discount_factors = compute_discounted_future_value(self.discount_factor, self.num_td_steps).to(self.device)
        self.sum_discounted_gammas = torch.sum(self.discount_factors)
        self.reduction_type = 'cross'

    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.exploration_params, 
         self.memory_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        training_start_step = self._compute_training_start_step()
        total_iterations = max(self.exploration_params.max_steps - training_start_step, 0)
        TrainingManager.__init__(self, networks, target_networks, self.optimization_params.lr, self.optimization_params.lr_decay_ratio, 
                                 self.optimization_params.clip_grad_range, self.network_params.tau, total_iterations)
        self.device = device

    def _init_normalization_utils(self, env_config, device):
        NormalizationUtils.__init__(self, env_config, self.normalization_params, self.model_seq_length, device)

    def _init_exploration_utils(self):
        ExplorationUtils.__init__(self, self.exploration_params)

    def _init_trainer_specific_params(self):
        self.use_gae_advantage = self.algorithm_params.use_gae_advantage
        self.num_td_steps = self.algorithm_params.num_td_steps 
        self.model_seq_length = self.algorithm_params.model_seq_length
        self.use_target_network = self.network_params.use_target_network
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor

    def _compute_training_start_step(self):
        training_start_step = self.training_params.early_training_start_step
        if training_start_step is None:
            batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
            training_start_step = self.memory_params.buffer_size // int(batch_size_ratio)
        return training_start_step

    def scale_seq_rewards(self, rewards):
        # Compute the scaling factors for each trajectory
        if self.discount_factor == 0:
            return rewards
        scaling_factors = 1 / self.sum_discounted_gammas
        scaled_rewards = scaling_factors * rewards 

        return scaled_rewards
    
    def select_tensor_reduction(self, tensor, mask=None):
        """
        Applies either masked_tensor_reduction or adaptive_masked_tensor_reduction 
        based on the specified reduction type.

        :param tensor: The input tensor to be reduced.
        :param mask: The mask tensor indicating the elements to consider in the reduction.
        :reduction_type: Type of reduction to apply ('adaptive', 'batch', 'seq', 'all', or 'cross').
        :length_weight_exponent: The exponent used in adaptive reduction.
        :return: The reduced tensor.
        """
        if mask is not None:
            if self.reduction_type == 'adaptive':
                return adaptive_masked_tensor_reduction(tensor, mask, length_weight_exponent = 2)
            elif self.reduction_type in ['batch', 'seq', 'all']:
                return masked_tensor_reduction(tensor, mask, reduction=self.reduction_type)
            elif self.reduction_type == 'cross':
                batch_dim_reduction = masked_tensor_reduction(tensor, mask, reduction="batch")/2
                seq_dim_reduction = masked_tensor_reduction(tensor, mask, reduction="seq")/2
                return torch.concat([batch_dim_reduction, seq_dim_reduction], dim=0)
            else:
                raise ValueError("Invalid reduction type. Choose 'adaptive', 'batch', 'seq', 'all', or 'cross'.")
        else:
            return tensor.mean()

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        squared_error = (estimated_value - expected_value).square()
        reduced_loss = self.select_tensor_reduction(squared_error, mask)
        return reduced_loss
    
    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, model_seq_mask: torch.Tensor):
        """Compute the advantage and expected value."""
        gamma = self.discount_factor
        lambd = self.advantage_lambda
        
        states, actions, rewards, next_states, dones = trajectory 

        full_padding_mask = create_padding_mask_before_dones(dones)
        trajectory_states, trajectory_mask = convert_trajectory_data(states, next_states, full_padding_mask)
        scaled_rewards = self.scale_seq_rewards(rewards)
        
        with torch.no_grad():
            trajectory_values = self.trainer_calculate_future_value(trajectory_states, trajectory_mask)
            if self.use_gae_advantage:
                _advantage = calculate_gae_returns(trajectory_values, scaled_rewards, dones, gamma, lambd)
                _expected_value = (_advantage + estimated_value)
            else:
                _expected_value = calculate_lambda_returns(trajectory_values, scaled_rewards, dones, gamma, lambd)
            
            expected_value = apply_seq_mask(_expected_value, model_seq_mask, self.model_seq_length)
            advantage = (expected_value - estimated_value)
            
            advantage = self.normalize_advantage(advantage)
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
    
