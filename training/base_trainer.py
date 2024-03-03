import torch
from training.managers.training_manager import TrainingManager 
from training.managers.normalization_manager import NormalizationUtils 
from training.managers.exploration_manager import ExplorationUtils 
from abc import abstractmethod
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import masked_tensor_reduction, create_padding_mask_before_dones, create_train_sequence_mask, apply_sequence_mask, GradScaler
from .learnable_td import LearnableTD
UPDATE_LEARNABLE_TD_INTERVAL = 10

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()

        self.learnable_td = LearnableTD(self.gpt_seq_length, self.discount_factor, self.advantage_lambda, device)
        
        # Initializing the training manager with the networks involved in the learning process
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils(self.gpt_seq_length)
  
        self.train_iter = 0

    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        # Use 'learning_networks' to emphasize the networks' role in the learning process
        learning_networks = networks + [self.learnable_td]
        # Constructing learning parameters for each network, including the learnable_td with adjusted parameters
        learning_param_list = [
            {'lr': self.optimization_params.lr, 'decay_rate_100k': self.optimization_params.decay_rate_100k,
             'scheduler_type': self.optimization_params.scheduler_type, 
             'clip_grad_range': self.optimization_params.clip_grad_range, 
             'max_grad_norm': self.optimization_params.max_grad_norm}
            for _ in networks
        ]
        # Adjusting parameters for learnable_td
        learnable_td_params = {
            'lr': self.optimization_params.lr * UPDATE_LEARNABLE_TD_INTERVAL, 
            'decay_rate_100k': self.optimization_params.decay_rate_100k, 
            'scheduler_type': self.optimization_params.scheduler_type, 
            'clip_grad_range': self.optimization_params.clip_grad_range, 
            'max_grad_norm': self.optimization_params.max_grad_norm
        }
        learning_param_list.append(learnable_td_params)
        
        TrainingManager.__init__(self, learning_networks, target_networks, learning_param_list, self.optimization_params.tau, self.total_iterations)
        self.device = device

    def _init_normalization_utils(self, env_config, device):
        NormalizationUtils.__init__(self, env_config.state_size, self.normalization_params, self.gpt_seq_length, device)

    def _init_exploration_utils(self, gpt_seq_length):
        self.decay_mode = self.optimization_params.scheduler_type
        ExplorationUtils.__init__(self, gpt_seq_length, self.total_iterations, self.device)

    def _init_trainer_specific_params(self):
        self.gpt_seq_length = self.algorithm_params.gpt_seq_length 
        self.td_seq_length = self.algorithm_params.td_seq_length 
        self.advantage_lambda = self.algorithm_params.advantage_lambda
        self.discount_factor = self.algorithm_params.discount_factor
        self.use_deterministic = self.algorithm_params.use_deterministic
        self.use_masked_exploration = self.algorithm_params.use_masked_exploration 
        self.reduction_type = 'cross'

        self.training_start_step = self._compute_training_start_step()
        self.total_iterations = max(self.training_params.max_steps - self.training_start_step, 0)//self.training_params.train_interval

    def _compute_training_start_step(self):
        batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
        training_start_step = self.training_params.buffer_size // int(batch_size_ratio)
        return training_start_step
    
    def init_train(self):
        self.set_train(training=True)
        
        with torch.no_grad():
            self.learnable_td.update_sum_reward_weights()
  
    def update_step(self):
        """
        Executes a series of update operations essential for the training iteration.

        This function orchestrates the training process by sequentially:
        1. Clipping gradients to prevent exploding gradient issues, ensuring stable training.
        2. Updating the model's optimizers, applying the computed gradients to adjust model parameters.
        3. Updating target networks, which are used in certain RL algorithms to provide stable targets for learning.
        4. Adjusting learning rates through schedulers, optimizing the training efficiency over time.
        5. Modifying the exploration rate, crucial for balancing exploration and exploitation in RL.
        6. Incrementing the training iteration counter, tracking the progress of the training process.

        Each step plays a vital role in maintaining the health and effectiveness of the training loop, 
        contributing to the overall performance and convergence of the learning algorithm.
        """
        self.clip_gradients()         # Prevents exploding gradients.
        self.update_optimizers()      # Applies gradients to adjust model parameters.
        self.update_target_networks() # Updates target networks for stable learning targets.
        self.update_schedulers()      # Adjusts learning rates for optimal training.
        self.update_exploration_rate()# Modifies exploration rate for effective exploration-exploitation balance.
        self.train_iter += 1          # Tracks training progress.
                
    def should_update_learnable_td(self):
        """
        Checks if learnable TD parameters should be updated based on the current training iteration.
        Returns:
            bool: True if it's time to update learnable TD parameters; False otherwise.
        """
        return self.train_iter % UPDATE_LEARNABLE_TD_INTERVAL == 0
        
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
        if self.should_update_learnable_td():
            squared_error = (estimated_value - expected_value).square()
        else:
            squared_error = (estimated_value - expected_value.detach()).square()
            
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
        gpt_seq_length = self.gpt_seq_length
        td_seq_length = self.td_seq_length
        diff_seq_length = td_seq_length - gpt_seq_length
        start_seq_idx = gpt_seq_length - diff_seq_length
        end_seq_idx = gpt_seq_length
        
        # Extract the last segment of the trajectory based on the training sequence length
        last_segment = slice(-gpt_seq_length, None)
        last_next_states = next_states[:, last_segment]
        last_dones = dones[:, last_segment]
        last_rewards = rewards[:, last_segment]

        # Calculate future values for the last states and construct trajectory values for lambda returns calculation
        last_padding_mask = create_padding_mask_before_dones(last_dones)
        future_values = self.trainer_calculate_future_value(last_next_states, last_padding_mask)
        trajectory_values = torch.cat([torch.zeros_like(future_values[:, :1]), future_values], dim=1)

        # Optimize trajectory adjustments for TD calculations by focusing on the differential segment (`td_seq_length` - `gpt_seq_length`). 
        adjusted_trajectory_values = trajectory_values[:, -(diff_seq_length + 1):]
        adjusted_last_rewards = last_rewards[:, -diff_seq_length:]
        adjusted_last_dones = last_dones[:, -diff_seq_length:]
        adjusted_padding_mask = last_padding_mask[:, -diff_seq_length:]
        
        # Calculate expected value and sum of rewards for the terminal segment of the trajectory
        expected_value, sum_rewards = self.learnable_td.calculate_lambda_returns(
            adjusted_trajectory_values, adjusted_last_rewards, adjusted_last_dones, seq_range=(start_seq_idx, end_seq_idx))
        
        with torch.no_grad():
            # Retrieve normalized sum reward weights for the given sequence range
            sum_reward_weights = self.learnable_td.get_sum_reward_weights(seq_range=(start_seq_idx, end_seq_idx))

            # Normalize sum rewards with the adjusted mask and weights
            normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, adjusted_padding_mask, seq_range=(start_seq_idx, end_seq_idx)) * sum_reward_weights

        # Correct the expected value to align with the length of sum_rewards and normalized_sum_rewards
        expected_value[:, :-1, :] = expected_value[:, :-1, :] - sum_rewards + normalized_sum_rewards
    
        # Select the appropriate future value using end_select_idx
        target_idx = (selection_end_idx - gpt_seq_length).flatten()
        batch_idx = torch.arange(0, next_states.size(0), device=self.device)
        discounted_future_value = expected_value[batch_idx, target_idx].unsqueeze(-1)
        
        return discounted_future_value
        
    def compute_expected_value(self, states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, padding_mask: torch.Tensor, end_value: torch.Tensor):
        """
        Computes the expected value for each state in a batch, utilizing future values and rewards.
        This function prepares the groundwork for advantage calculation by providing a baseline of expected returns,
        which are essential for evaluating the relative benefit of actions taken in each state.
        The inputs are 3D tensors with dimensions [Batch Size, Sequence Length, Feature Dimension], where Batch Size
        is the number of sequences, Sequence Length is the number of steps in each sequence, and Feature Dimension
        represents the dimensionality of states, rewards, and dones, except for the end_value which has dimensions [B, 1, Value Dim].

        :param states: A 3D tensor of states from the environment [B, Seq, State Dim].
        :param rewards: A 3D tensor of rewards obtained after executing actions from the states [B, Seq, 1].
        :param dones: A 3D tensor indicating whether a state is terminal (1 if true, 0 otherwise) [B, Seq, 1].
        :param padding_mask: A 3D mask tensor indicating valid states to consider, used to exclude padding in batched operations [B, Seq, 1].
        :param end_value: A 3D tensor representing the last future value predicted by the model for each sequence, used for calculating the return from the final state [B, 1, Value Dim].
        :return: The expected value for each state, adjusted with the normalized sum of rewards [B, Seq, Value Dim].
        """
    
        gpt_seq_length = self.gpt_seq_length
        # Calculate future values from the model, used to estimate the expected return from each state.
        target_values = self.trainer_calculate_future_value(states, padding_mask)
        
        # Prepare trajectory values for lambda returns calculation, initializing the sequence with zeros.
        trajectory_values = torch.cat([target_values, end_value], dim=1)
        
        # Calculate expected values and sum of rewards using lambda returns method,
        # which blends immediate rewards with future values for a balanced estimate.
        expected_value, sum_rewards = self.learnable_td.calculate_lambda_returns(trajectory_values, rewards, dones, seq_range = (0, gpt_seq_length))
        # Trim the last timestep from expected_value to match the length of sum_rewards.
        # This step ensures both tensors have consistent dimensions for subsequent calculations,
        # as lambda_returns method produces an n+1 length for expected_value, while sum_rewards is n.
        expected_value = expected_value[:, :-1, :]
        
        with torch.no_grad():
            sum_reward_weights = self.learnable_td.get_sum_reward_weights(seq_range = (0, gpt_seq_length))
            # Normalize the sum of rewards, applying the mask to handle valid sequence lengths.
            normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, padding_mask, seq_range = (0, gpt_seq_length)) * sum_reward_weights
        
        # Correct the expected value by adjusting with the normalized sum of rewards.
        expected_value = expected_value - sum_rewards + normalized_sum_rewards

        return expected_value
                
    def compute_advantage(self, estimated_value: torch.Tensor, expected_value: torch.Tensor, padding_mask: torch.Tensor):
        """
        Calculates the advantage, which measures the benefit of taking specific actions from given states over the policy's average prediction for those states.
        This is determined by comparing the expected value of actions, as computed from actual outcomes and future predictions, against the value estimated by the model.

        :param estimated_value: A 3D tensor of value estimates for each state as predicted by the model. This tensor has dimensions [B, Seq, Value Dim],
                                where Value Dim typically equals 1, representing the predicted value of each state within the sequence.
        :param expected_value: A 3D tensor of expected values for each state, computed considering future rewards and values, with dimensions [B, Seq, Value Dim].
                            This tensor provides the baseline against which the estimated value is compared to calculate the advantage.
        :param padding_mask: A 3D mask tensor used to identify valid states and exclude padding when processing batches, with dimensions [B, Seq, 1].
                            The mask ensures that only valid sequence elements contribute to the advantage calculation, improving the accuracy of training signals.
        :return: The normalized advantage for each state-action pair within the sequences, indicating the relative effectiveness of actions taken. 
                The output is a 3D tensor with dimensions [B, Seq, Value Dim], where each element reflects the calculated advantage for the corresponding state.
        """
        # Determine the computation mode based on the update status of learnable TD parameters.
        # If TD parameters are being updated, detach only the estimated value to emphasize the learning signal from actual outcomes.
        # Otherwise, detach both to stabilize the learning against a static baseline.
        if self.should_update_learnable_td():
            advantage = (expected_value - estimated_value.detach())
        else:
            advantage = (expected_value.detach() - estimated_value.detach())
            
        # Normalize the advantage to enhance training stability, ensuring consistent gradient scales.
        advantage = self.normalize_advantage(advantage, padding_mask)

        if self.should_update_learnable_td():
            # When TD parameters are being updated, apply gradient scaling to the advantage calculation.
            advantage = GradScaler.apply(advantage, -1)
        else:
            # Proceed without gradient scaling when TD parameters are not being updated.
            advantage = advantage  # This line is effectively a no-op and could be omitted for clarity.

        return advantage

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass
