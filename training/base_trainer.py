import torch
from training.managers.training_manager import TrainingManager 
from training.managers.normalization_manager import NormalizationUtils 
from abc import abstractmethod
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.trajectories  import BatchTrajectory
from .trainer_utils import calculate_lambda_returns, get_discount_sequence, masked_tensor_reduction, create_padding_mask_before_dones

class BaseTrainer(TrainingManager, NormalizationUtils):
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self.gammas = get_discount_sequence(self.discount_factor, self.gpt_seq_length)
        self.lambdas = get_discount_sequence(self.advantage_lambda, self.gpt_seq_length)
        # Calculate the scaling factors by summing the product of gammas and lambdas across the sequence.
        # The scaling factors are divided by 2 to scale the sum of normalized rewards to the midpoint of the GPT sequence length.
        # This ensures that the cumulative effect of discounted rewards is centered around the average position in the sequence,
        # effectively balancing the influence of rewards from the start to the end of the sequence.
        # This scaling approach enables the value loss to be maintained around 0.5, preventing it from increasing or decreasing easily during training.
        # Such stabilization is crucial for maintaining a consistent training process and achieving convergence.
        accumulative_factors = (self.gammas * self.lambdas).to(self.device)
        scaling_cumsum = torch.cumsum(accumulative_factors, dim=1)
        self.scaling_factors = scaling_cumsum.mean()
        
    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.exploration_params, 
         self.memory_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        training_start_step = self._compute_training_start_step()
        total_iterations = max(self.exploration_params.max_steps - training_start_step, 0)//self.training_params.train_interval
        TrainingManager.__init__(self, networks, target_networks, self.optimization_params.lr, self.optimization_params.min_lr, 
                                 self.optimization_params.clip_grad_range, self.optimization_params.tau, total_iterations, self.optimization_params.scheduler_type)
        self.device = device

    def _init_normalization_utils(self, env_config, device):
        NormalizationUtils.__init__(self, env_config.state_size, self.normalization_params, self.gpt_seq_length, device)

    def _init_trainer_specific_params(self):
        self.gpt_seq_length = self.algorithm_params.gpt_seq_length 
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor
        self.reduction_type = 'batch'

    def _compute_training_start_step(self):
        batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
        training_start_step = self.memory_params.buffer_size // int(batch_size_ratio)
        return training_start_step
    
    def scale_rewards(self, rewards):
        """Scales the given rewards by the scaling factors."""
        return rewards / self.scaling_factors
    
    def select_tensor_reduction(self, tensor, mask=None):
        """
        Applies either masked_tensor_reduction or adaptive_masked_tensor_reduction 
        based on the specified reduction type.

        :param tensor: The input tensor to be reduced.
        :param mask: The mask tensor indicating the elements to consider in the reduction.
        :reduction_type: Type of reduction to apply ('batch', 'seq', or 'all').
        :length_weight_exponent: The exponent used in adaptive reduction.
        :return: The reduced tensor.
        """
        if mask is not None:
            if self.reduction_type in ['batch', 'seq', 'all']:
                return masked_tensor_reduction(tensor, mask, reduction=self.reduction_type)
            elif self.reduction_type == 'none':
                return tensor[mask>0].flatten()
            else:
                raise ValueError("Invalid reduction type. Choose 'batch', 'seq', or 'all'.")
        else:
            if self.reduction_type == 'none':
                return tensor
            else:
                return tensor.mean()

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        squared_error = (estimated_value - expected_value).square()
        reduced_loss = self.select_tensor_reduction(squared_error, mask)
        return reduced_loss

    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, padding_mask: torch.Tensor):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory 
        scaled_rewards = self.scale_rewards(rewards)
        
        with torch.no_grad():
            future_values = self.trainer_calculate_future_value(next_states, padding_mask)
            trajectory_values = torch.cat([torch.zeros_like(future_values[:, :1]), future_values], dim=1)
            expected_value = calculate_lambda_returns(trajectory_values, scaled_rewards, dones, self.discount_factor, self.advantage_lambda)
            
            advantage = (expected_value - estimated_value)
            normalized_advantage = self.normalize_advantage(advantage)
    
        return expected_value, normalized_advantage

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
