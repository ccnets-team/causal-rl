import torch
from training.managers.training_manager import TrainingManager 
from training.managers.normalization_manager import NormalizationUtils 
from training.managers.exploration_manager import ExplorationUtils 
from abc import abstractmethod
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import calculate_lambda_returns, masked_tensor_reduction, calculate_sum_reward_weights, create_padding_mask_before_dones, create_train_sequence_mask, apply_sequence_mask


class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils(self.gpt_seq_length, rl_params.max_steps)
            
        self.gammas = torch.tensor(self.discount_factor, device=self.device).expand(self.gpt_seq_length).unsqueeze(0).unsqueeze(-1)
        self.lambdas = torch.tensor(self.advantage_lambda, device=self.device).expand(self.gpt_seq_length).unsqueeze(0).unsqueeze(-1)
        self.sum_reward_weights = calculate_sum_reward_weights(self.gpt_seq_length, self.gammas, self.lambdas, self.device)  
        
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

    def _init_exploration_utils(self, gpt_seq_length, max_steps):
        self.decay_mode = self.optimization_params.scheduler_type
        ExplorationUtils.__init__(self, gpt_seq_length, max_steps, self.device)

    def _init_trainer_specific_params(self):
        self.gpt_seq_length = self.algorithm_params.gpt_seq_length 
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor
        self.use_deterministic = self.algorithm_params.use_deterministic
        self.use_masked_exploration = self.algorithm_params.use_masked_exploration 
        self.reduction_type = 'cross'

    def _compute_training_start_step(self):
        batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
        training_start_step = self.training_params.buffer_size // int(batch_size_ratio)
        return training_start_step
    
    def init_train(self):
        self.set_train(training=True)
        
        self.exploration_rate = self.get_exploration_rate()
        dynamic_lambda = (1 - self.exploration_rate) + self.advantage_lambda * self.exploration_rate
        self.lambdas.fill_(dynamic_lambda)
        self.sum_reward_weights = calculate_sum_reward_weights(self.gpt_seq_length, self.gammas, self.lambdas, self.device)  
        
    def select_tensor_reduction(self, tensor, mask=None):
        """
        Applies either masked_tensor_reduction or adaptive_masked_tensor_reduction 
        based on the specified reduction type.

        :param tensor: The input tensor to be reduced.
        :param mask: The mask tensor indicating the elements to consider in the reduction.
        :reduction_type: Type of reduction to apply ('batch', 'seq', or 'all').
        :return: The reduced tensor.
        """
        
        if mask is not None:
            return masked_tensor_reduction(tensor, mask, reduction=self.reduction_type)
        else:
            if self.reduction_type == 'none':
                return tensor
            else:
                return tensor.mean()

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

    def select_train_sequence(self, trajectory):
        """
        Selects a segment from the trajectory for training, tailored to the model's input requirements. This method is 
        crucial for handling trajectories of length 'gpt_td_seq_length', which is intentionally longer than 'gpt_seq_length' 
        to provide additional context for TD calculations. It identifies the valid portion of these extended trajectories, 
        starting from the last non-padded point and moving backward, and selects a segment of size 'gpt_seq_length'. If 
        the valid portion is shorter than 'gpt_seq_length', the selection includes padding towards the beginning.

        This strategy ensures consistent input size while maximizing the use of relevant experience and accommodating 
        the extended context provided by 'gpt_td_seq_length'. The method effectively balances the need for fixed-length 
        inputs and the desire to capture as much meaningful data as possible within each training sequence.

        :param trajectory: A tuple containing the trajectory's components (states, actions, rewards, next_states, dones).
                        The trajectory is expected to be of 'gpt_td_seq_length', providing a larger context for selection.
        :param padding_mask: A tensor indicating valid (1) and padded (0) parts of the trajectory, shape [B, S, 1],
                            where B is the batch size, and S is the sequence length ('gpt_td_seq_length').
        :return: A tuple of selected and masked trajectory components (states, actions, rewards, next_states, dones, 
                padding_mask), each trimmed to 'gpt_seq_length' to fit the model's input specifications.
        """
        states, actions, rewards, next_states, dones = trajectory
        padding_mask = create_padding_mask_before_dones(dones)

        train_seq_length = self.gpt_seq_length
        train_seq_mask, selection_end_idx = create_train_sequence_mask(padding_mask, train_seq_length)

        # Apply the mask to each trajectory component
        sel_states = apply_sequence_mask(states, train_seq_mask, train_seq_length)
        sel_actions = apply_sequence_mask(actions, train_seq_mask, train_seq_length)
        sel_rewards = apply_sequence_mask(rewards, train_seq_mask, train_seq_length)
        sel_next_states = apply_sequence_mask(next_states, train_seq_mask, train_seq_length)
        sel_dones = apply_sequence_mask(dones, train_seq_mask, train_seq_length)
        sel_padding_mask = apply_sequence_mask(padding_mask, train_seq_mask, train_seq_length)
        
        end_value = self.calculate_sequence_end_value(trajectory, selection_end_idx)
        return sel_states, sel_actions, sel_rewards, sel_next_states, sel_dones, sel_padding_mask, end_value

    def calculate_sequence_end_value(self, trajectory, selection_end_idx):
        """
        Computes the future value for TD calculations from the terminal segment of a trajectory sampled at 'gpt_td_seq_length'.
        This extended trajectory length allows for the division of the sequence into two parts: the front part, used for training,
        and the rear part, of the same length as 'gpt_seq_length', designated for TD computation. The sequence is considered valid
        from the right side (end of the episode) towards the left, ensuring the inclusion of meaningful end-of-episode data.

        This function focuses on the rear part of the trajectory, calculating the expected future return from the final state(s) 
        in the context of TD updates. The selection of the sequence for this computation is based on the valid portion extending 
        leftward from the end of the episode, incorporating any necessary padding to meet the 'gpt_seq_length' if the valid 
        portion is shorter.

        :param trajectory: A tuple containing the trajectory's components (states, actions, rewards, next_states, dones),
                        sampled to 'gpt_td_seq_length' to provide sufficient context for both training and TD computation.
        :param selection_end_idx: The index marking the division between the training and TD computation segments within
                                the 'gpt_td_seq_length', facilitating precise extraction of the terminal segment for value calculation.
        :return: A tensor of the discounted future value from the terminal sequence segment,
                critical for computing expected values and advantages in `compute_values`.
        """
        _, _, rewards, next_states, dones = trajectory
        
        # Extract the last segment of the trajectory based on the training sequence length
        last_segment = slice(-self.gpt_seq_length, None)
        last_next_states = next_states[:, last_segment]
        last_dones = dones[:, last_segment]
        last_rewards = rewards[:, last_segment]

        # Calculate future values for the last states and construct trajectory values for lambda returns calculation
        last_padding_mask = create_padding_mask_before_dones(last_dones)
        future_values = self.trainer_calculate_future_value(last_next_states, last_padding_mask)
        trajectory_values = torch.cat([torch.zeros_like(future_values[:, :1]), future_values], dim=1)

        # Calculate expected value from the trajectory, considering rewards and dones
        expected_value, sum_rewards = calculate_lambda_returns(
            trajectory_values, last_rewards, last_dones, self.gammas, self.lambdas)

        # Normalize the sum of rewards, applying the mask to handle valid sequence lengths.
        normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, last_padding_mask) * self.sum_reward_weights
        
        # Correct the expected value by adjusting with the normalized sum of rewards.
        expected_value = expected_value - sum_rewards + normalized_sum_rewards
        
        # Select the appropriate future value using end_select_idx
        target_idx = (self.gpt_seq_length - next_states.size(1) + selection_end_idx - 1).flatten()
        batch_idx = torch.arange(0, next_states.size(0), device=self.device)
        discounted_future_value = expected_value[batch_idx, target_idx].unsqueeze(-1)
        
        return discounted_future_value
        
    def compute_expected_value(self, states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, padding_mask: torch.Tensor, future_value: torch.Tensor):
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

        with torch.no_grad():
            # Calculate future values from the model, used to estimate the expected return from each state.
            target_values = self.trainer_calculate_future_value(states, padding_mask)
            
            # Prepare trajectory values for lambda returns calculation, initializing the sequence with zeros.
            trajectory_values = torch.cat([target_values, future_value], dim=1)
            
            # Calculate expected values and sum of rewards using lambda returns method,
            # which blends immediate rewards with future values for a balanced estimate.
            expected_value, sum_rewards = calculate_lambda_returns(trajectory_values, rewards, dones, self.gammas, self.lambdas)
            
            # Normalize the sum of rewards, applying the mask to handle valid sequence lengths.
            normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, padding_mask) * self.sum_reward_weights
            
            # Correct the expected value by adjusting with the normalized sum of rewards.
            expected_value = expected_value - sum_rewards + normalized_sum_rewards
                
    def compute_advantage(self, estimated_value: torch.Tensor, expected_value: torch.Tensor, padding_mask: torch.Tensor):
        
        with torch.no_grad():
            # Calculate the advantage as the difference between corrected expected values and model estimates.
            # This measures how much better (or worse) an action is compared to the policy's average action.
            advantage = (expected_value - estimated_value)
            
            # Normalize the advantage, improving training stability by adjusting the scale of gradient updates.
            advantage = self.normalize_advantage(advantage, padding_mask)

        return estimated_value, expected_value, advantage

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass
