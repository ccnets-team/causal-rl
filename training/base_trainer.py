import torch
from training.managers.training_manager import TrainingManager 
from training.managers.normalization_manager import NormalizationUtils 
from training.managers.exploration_manager import ExplorationUtils 
from abc import abstractmethod
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.structure.data_structures  import BatchTrajectory
from .trainer_utils import calculate_lambda_returns, masked_tensor_reduction, calculate_sum_reward_weights, create_sequence_weights

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils(rl_params.max_steps)
        self.base_sequence_weights = create_sequence_weights(self.gpt_seq_length, self.device)
        self.train_iter = 0

    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        training_start_step = self._compute_training_start_step()
        total_iterations = max(self.training_params.max_steps - training_start_step, 0)//self.training_params.train_interval
        TrainingManager.__init__(self, networks, target_networks, self.optimization_params.lr, self.optimization_params.min_lr, 
                                 self.optimization_params.clip_grad_range, self.optimization_params.max_grad_norm, self.optimization_params.tau, total_iterations, self.optimization_params.scheduler_type)
        self.device = device

    def _init_normalization_utils(self, env_config, device):
        NormalizationUtils.__init__(self, env_config.state_size, self.normalization_params, self.gpt_seq_length, device)

    def _init_exploration_utils(self, max_steps):
        self.decay_mode = self.optimization_params.scheduler_type
        ExplorationUtils.__init__(self, max_steps, self.device)

    def _init_trainer_specific_params(self):
        self.gpt_seq_length = self.algorithm_params.gpt_seq_length 
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor
        self.use_deterministic = self.algorithm_params.use_deterministic
        self.use_masked_exploration = self.algorithm_params.use_masked_exploration 
        self.reduction_type = 'batch'

    def _compute_training_start_step(self):
        batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
        training_start_step = self.training_params.buffer_size // int(batch_size_ratio)
        return training_start_step

    def get_dynamic_lambda(self) -> float:
        """
        Calculates a dynamic lambda value by integrating the current exploration rate with the agent's base advantage lambda. 
        This adjustment dynamically balances immediate and future rewards in the learning process, based on the agent's 
        current exploration rate. As exploration decreases and exploitation increases, the lambda value shifts to emphasize 
        longer-term rewards, reflecting a strategy shift from short-term to long-term reward maximization as the agent 
        becomes more confident in its policy.

        :return: The dynamically adjusted lambda value, tailored to the current exploration behavior and focusing on the balance 
                between immediate and future rewards.
        """
        dynamic_lambda = (1 - self.exploration_rate) + self.advantage_lambda * self.exploration_rate
        return dynamic_lambda
    
    def apply_tensor_weights(self, tensor, mask=None):
        """
        Applies the calculated sequence weights to a tensor, optionally using a mask to focus on specific elements.
        This method is particularly useful for models that process sequential data, enhancing the impact of elements based on their position in the sequence.

        :param tensor: The input tensor to be weighted. Shape is assumed to be [batch_size, seq_length, feature_dim].
        :param mask: Optional mask tensor indicating the elements to consider in the operation.
        :return: A tensor with applied sequence weights, emphasizing later elements in the sequence.
        """
        dynamic_sequence_weights = torch.pow(self.base_sequence_weights, self.exploration_rate)
        dynamic_sequence_weights = dynamic_sequence_weights/dynamic_sequence_weights.mean(dim = 1).clamp_min(1e-8)

        if mask is not None:
            # Apply the mask to the sequence weights, focusing on unmasked (valid) parts of the tensor
            weights = mask * dynamic_sequence_weights
            # Normalize the weights to ensure that their sum matches the number of valid positions in the mask
            adjusted_weights = (weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)) * mask.sum(dim=1, keepdim=True).clamp_min(1)
            # Apply the normalized, adjusted weights to the tensor
            weighted_tensor = tensor * adjusted_weights
        else:
            # If no mask is provided, apply the sequence weights directly to the tensor
            weighted_tensor = tensor * dynamic_sequence_weights
        
        return weighted_tensor
        
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
        weighted_tensor = self.apply_tensor_weights(tensor, mask)
        
        if mask is not None:
            if self.reduction_type in ['batch', 'seq', 'all']:
                return masked_tensor_reduction(weighted_tensor, mask, reduction=self.reduction_type)
            elif self.reduction_type == 'cross':
                batch_wise_reduction = masked_tensor_reduction(weighted_tensor, mask, reduction='batch')
                seq_wise_reduction = masked_tensor_reduction(weighted_tensor, mask, reduction='seq')
                return torch.cat([batch_wise_reduction, seq_wise_reduction], dim = 0)
            elif self.reduction_type == 'none':
                return weighted_tensor[mask>0].flatten()
            else:
                raise ValueError("Invalid reduction type. Choose 'batch', 'seq', or 'all'.")
        else:
            if self.reduction_type == 'none':
                return weighted_tensor
            else:
                return weighted_tensor.mean()

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        """
        Calculates the value loss as the squared difference between estimated and expected values.
        This loss reflects the accuracy of the model's value predictions against the target values,
        where a lower loss indicates better prediction accuracy.

        :param estimated_value: The value estimated by the model.
        :param expected_value: The target value, typically computed from rewards and future values.
        :param mask: Optional mask to exclude certain elements from loss calculation, e.g., padded elements.
        :return: The reduced value loss, after applying optional masking and reduction over the tensor.
        """
        # Compute squared error, a common choice for regression tasks, providing a smooth gradient for optimization.
        squared_error = (estimated_value - expected_value).square()

        # Reduce the squared error tensor to a scalar loss value, applying optional masking to focus loss calculation.
        reduced_loss = self.select_tensor_reduction(squared_error, mask)

        return reduced_loss

    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, padding_mask: torch.Tensor):
        """
        Computes the expected value and advantage for each state in a given trajectory,
        based on future values and actual rewards. The advantage calculation helps to quantify
        the relative benefit of each action compared to the average, guiding the policy improvement.

        :param trajectory: A batch of trajectories containing states, actions, rewards, next_states, and dones.
        :param estimated_value: The value estimates produced by the model for each state.
        :param padding_mask: A mask indicating valid elements in the trajectory to exclude padding.
        :return: The estimated value, the corrected expected value, and the computed advantage for the trajectory.
        """
        # Extract components from the trajectory
        states, actions, rewards, next_states, dones = trajectory 

        gamma = self.discount_factor  # Define the discount factor for future rewards, determining their present value.
        lambd = self.get_dynamic_lambda()  # Calculate the dynamic lambda value, adjusting the balance between immediate and future rewards.
        
        with torch.no_grad():
            # Calculate future values from the model, used to estimate the expected return from each state.
            future_values = self.trainer_calculate_future_value(next_states, padding_mask)
            
            # Prepare trajectory values for lambda returns calculation, initializing the sequence with zeros.
            trajectory_values = torch.cat([torch.zeros_like(future_values[:, :1]), future_values], dim=1)
            
            # Compute the weighting factors for summing rewards over the trajectory, based on sequence length, gamma, and lambda.
            sum_reward_weights = calculate_sum_reward_weights(self.gpt_seq_length, gamma, lambd, self.device)  
            
            # Calculate expected values and sum of rewards using lambda returns method,
            # which blends immediate rewards with future values for a balanced estimate.
            expected_value, sum_rewards = calculate_lambda_returns(trajectory_values, rewards, dones, gamma, lambd)
            
            # Normalize the sum of rewards, applying the mask to handle valid sequence lengths.
            normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, padding_mask) * sum_reward_weights
            
            # Correct the expected value by adjusting with the normalized sum of rewards.
            expected_value = expected_value - sum_rewards + normalized_sum_rewards
                
        with torch.no_grad():
            # Calculate the advantage as the difference between corrected expected values and model estimates.
            # This measures how much better (or worse) an action is compared to the policy's average action.
            advantage = (expected_value - estimated_value)
            
            # Normalize the advantage, improving training stability by adjusting the scale of gradient updates.
            advantage = self.normalize_advantage(advantage, padding_mask)

        return estimated_value, expected_value, advantage

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
