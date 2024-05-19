import torch
from abc import abstractmethod
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from training.managers.training_manager import TrainingManager 
from training.managers.normalization_manager import NormalizationManager 
from training.managers.exploration_manager import ExplorationManager 
from training.learners.sequence_length_learner import SequenceLengthLearner
from training.learners.gamma_lambda_learner import GammaLambdaLearner
from training.utils.tensor_util import masked_tensor_reduction, create_transformation_matrix, shift_left_padding_mask
from training.utils.sequence_util import create_padding_mask_before_dones, select_train_sequence, apply_sequence_mask, select_sequence_range
from training.utils.sequence_util import LEARNABLE_TD_UPDATE_INTERVAL, SEQUENCE_LENGTH_UPDATE_INTERVAL

class BaseTrainer(TrainingManager, NormalizationManager, ExplorationManager):
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        
        self.gamma_lambda_learner = GammaLambdaLearner(rl_params.gamma, rl_params.lambd, self.max_seq_len, device)
        self.sequence_length_learner = SequenceLengthLearner(self.gamma_lambda_learner, self.max_seq_len)
        
        # Initializing the training manager with the networks involved in the learning process
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_manager(env_config, self.max_seq_len, device)
        self._init_exploration_manager()

        state_size = env_config.state_size
        value_size = env_config.value_size
        self.error_transformation_matrix = create_transformation_matrix(state_size, value_size).to(device)
        self.error_to_state_size_ratio = value_size/state_size
        self.train_iter = 0
        
    def _unpack_rl_params(self, rl_params): 
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        # Use 'learning_networks' to emphasize the networks' role in the learning process
        learning_networks = networks + [self.gamma_lambda_learner]
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
            'lr': self.optimization_params.lr,
            'decay_rate_100k': self.optimization_params.decay_rate_100k,
            'scheduler_type': self.optimization_params.scheduler_type,
            'clip_grad_range': self.optimization_params.clip_grad_range, 
            'max_grad_norm': self.optimization_params.max_grad_norm # Explicitly set to None if not using max_grad_norm for learnable_td_params
        }            

        learning_param_list.append(learnable_td_params)
        
        TrainingManager.__init__(self, learning_networks, target_networks, learning_param_list, self.optimization_params.tau, self.total_iterations)
        self.device = device

    def _init_normalization_manager(self, env_config, max_seq_len, device):
        NormalizationManager.__init__(self, env_config.state_size, env_config.value_size, self.normalization_params, max_seq_len, device)


    def _init_exploration_manager(self):
        ExplorationManager.__init__(self, self.gamma_lambda_learner, self.total_iterations, self.device)

    def _init_trainer_specific_params(self):
        self.max_seq_len = self.algorithm_params.max_seq_len 
        self.use_deterministic = self.algorithm_params.use_deterministic 
        self.reduction_type = 'cross'

        self.training_start_step = self._compute_training_start_step()
        self.total_iterations = max(self.training_params.max_steps - self.training_start_step, 0)//self.training_params.train_interval

    def _compute_training_start_step(self):
        batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
        training_start_step = self.training_params.buffer_size // int(batch_size_ratio)
        return training_start_step

    def should_update_learnable_td(self):
        return self.train_iter % LEARNABLE_TD_UPDATE_INTERVAL == 0
    
    def get_input_seq_len(self):
        return self.sequence_length_learner.get_input_seq_len()

    def get_total_seq_len(self):
        return self.sequence_length_learner.get_total_seq_len()

    def get_max_seq_len(self):
        return self.sequence_length_learner.get_max_seq_len()

    def get_min_seq_len(self):
        return self.sequence_length_learner.get_min_seq_len()

    def get_td_extension_steps(self):
        return self.sequence_length_learner.get_td_extension_steps()
    
    def get_max_td_extension_steps(self):
        return self.sequence_length_learner.get_max_td_extension_steps()

    def get_gamma(self):
        return self.get_gamma()

    def get_lambdas(self, seq_range):
        return self.gamma_lambda_learner.get_lambdas(seq_range = seq_range)

    def get_gamma_lambda_learner(self):
        return self.gamma_lambda_learner

    def get_sequence_length_learner(self):
        return self.sequence_length_learner

    def init_train(self):
        self.set_train(training=True)
        input_seq_len = self.get_input_seq_len()
        td_extension_steps = self.get_td_extension_steps()
        self.gamma_lambda_learner.update_sum_reward_weights(input_seq_len, td_extension_steps)
        
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
        if self.train_iter % SEQUENCE_LENGTH_UPDATE_INTERVAL == 0:
            self.sequence_length_learner.update_sequence_length() # Updates the learnable sequence length based on learnable_td parameters.
        self.train_iter += 1          # Tracks training progress.
    
    def select_tensor_reduction(self, tensor, mask=None):
        """
        Applies either masked_tensor_reduction or adaptive_masked_tensor_reduction 
        based on the specified reduction type.

        :param tensor: The input tensor to be reduced.
        :param mask: The mask tensor indicating the elements to consider in the reduction.
        :reduction_type: Type of reduction to apply ('batch', 'seq', or 'all').
        :return: The reduced tensor.
        """
        return masked_tensor_reduction(tensor, mask, reduction=self.reduction_type)

    def select_sequence(self, trajectory):
        """
        Selects a segment from the trajectory for training, tailored to the model's input requirements. This method is 
        crucial for handling trajectories of length 'gpt_td_seq_length', which is intentionally longer than 'input_seq_len' 
        to provide additional context for TD calculations. It identifies the valid portion of these extended trajectories, 
        starting from the last non-padded point and moving backward, and selects a segment of size 'input_seq_len'. If 
        the valid portion is shorter than 'input_seq_len', the selection includes padding towards the beginning.

        This strategy ensures consistent input size while maximizing the use of relevant experience and accommodating 
        the extended context provided by 'gpt_td_seq_length'. The method effectively balances the need for fixed-length 
        inputs and the desire to capture as much meaningful data as possible within each training sequence.

        :param trajectory: A tuple containing the trajectory's components (states, actions, rewards, next_states, dones).
                        The trajectory is expected to be of 'gpt_td_seq_length', providing a larger context for selection.
        :param padding_mask: A tensor indicating valid (1) and padded (0) parts of the trajectory, shape [B, S, 1],
                            where B is the batch size, and S is the sequence length ('gpt_td_seq_length').
        :return: A tuple of selected and masked trajectory components (states, actions, rewards, next_states, dones, 
                padding_mask), each trimmed to 'input_seq_len' to fit the model's input specifications.
        """
        trajectory_states, actions, rewards, dones = trajectory
        states = trajectory_states[:,:-1]
        next_states = trajectory_states[:,1:]
        
        padding_mask = create_padding_mask_before_dones(dones)
        
        input_seq_len = self.get_input_seq_len()
        total_seq_len = self.get_total_seq_len()
        
        assert padding_mask.size(1) == total_seq_len, "The input sequence length must match the total sequence length."
        
        train_seq_mask, selection_end_indices = select_train_sequence(padding_mask, input_seq_len)
        
        end_value = self.calculate_sequence_end_value(rewards, next_states, dones, selection_end_indices)
    
        # Apply the mask to each trajectory component
        sel_states, sel_actions, sel_rewards, sel_next_states, sel_dones, sel_padding_mask = \
            apply_sequence_mask(train_seq_mask, input_seq_len, states, actions, rewards, next_states, dones, padding_mask)
        
        return sel_states, sel_actions, sel_rewards, sel_next_states, sel_dones, sel_padding_mask, end_value

    def calculate_normalized_lambda_returns(self, trajectory_values, rewards, dones, padding_mask, seq_range, use_td_extension_steps = False):
        # Calculate expected value and sum of rewards for the terminal segment of the trajectory
        expected_value, sum_rewards = self.gamma_lambda_learner.calculate_lambda_returns(
            trajectory_values, rewards, dones, seq_range=seq_range)

        normalized_sum_rewards = self.normalize_sum_rewards(sum_rewards, padding_mask, seq_range=seq_range).detach()
        
        sum_reward_weights = self.gamma_lambda_learner.get_sum_reward_weights(use_td_extension_steps = use_td_extension_steps)

        # Correct the expected value to align with the length of sum_rewards and normalized_sum_rewards
        expected_value[:, :-1, :] = expected_value[:, :-1, :] - sum_rewards + sum_reward_weights * normalized_sum_rewards
        
        return expected_value
    
    def calculate_sequence_end_value(self, rewards, next_states, dones, selection_end_indices):
        """
        Computes the future value for TD calculations from the terminal segment of a trajectory sampled at 'gpt_td_seq_length'.
        This extended trajectory length allows for the division of the sequence into two parts: the front part, used for training,
        and the rear part, of the same length as 'input_seq_len', designated for TD computation. The sequence is considered valid
        from the right side (end of the episode) towards the left, ensuring the inclusion of meaningful end-of-episode data.

        This function focuses on the rear part of the trajectory, calculating the expected future return from the final state(s) 
        in the context of TD updates. The selection of the sequence for this computation is based on the valid portion extending 
        leftward from the end of the episode, incorporating any necessary padding to meet the 'input_seq_len' if the valid 
        portion is shorter.

        :param trajectory: A tuple containing the trajectory's components (states, actions, rewards, next_states, dones),
                        sampled to 'gpt_td_seq_length' to provide sufficient context for both training and TD computation.
        :param selection_end_idx: The index marking the division between the training and TD computation segments within
                                the 'gpt_td_seq_length', facilitating precise extraction of the terminal segment for value calculation.
        :return: A tensor of the discounted future value from the terminal sequence segment,
                critical for computing expected values and advantages in `compute_values`.
        """
        input_seq_len = self.get_input_seq_len()
        td_extension_steps = self.get_td_extension_steps()
        local_end_indices = selection_end_indices - input_seq_len
        
        # Extract the last segment of the trajectory based on the training sequence length
        td_next_states, td_rewards, td_dones = select_sequence_range(slice(-input_seq_len, None), next_states, rewards, dones)
        # Calculate future values for the last states and construct trajectory values for lambda returns calculation
        td_padding_mask = create_padding_mask_before_dones(td_dones)
        
        td_next_padding_mask = shift_left_padding_mask(td_padding_mask)
        future_values = self.trainer_calculate_future_value(td_next_states, td_next_padding_mask)
        trajectory_values = torch.cat([torch.zeros_like(future_values[:, :1]), future_values], dim=1)
        
        trajectory_values = select_sequence_range(slice(-(td_extension_steps + 1), None), trajectory_values)
        selected_rewards, selected_dones, selected_padding_mask = select_sequence_range(slice(-td_extension_steps, None), td_rewards, td_dones, td_padding_mask)
        expected_value = self.calculate_normalized_lambda_returns(trajectory_values, selected_rewards, selected_dones, selected_padding_mask, seq_range = (-td_extension_steps, None), use_td_extension_steps = True)
        
        # Select the appropriate future value using end_select_idx
        target_idx = local_end_indices.flatten()
        batch_idx = torch.arange(0, next_states.size(0), device=self.device)
        discounted_future_value = expected_value[batch_idx, target_idx].unsqueeze(1)
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
    
        input_seq_len = self.get_input_seq_len()
        # Calculate future values from the model, used to estimate the expected return from each state.
        future_values = self.trainer_calculate_future_value(states, padding_mask)
        trajectory_values = torch.cat([future_values, end_value], dim=1)

        expected_value = self.calculate_normalized_lambda_returns(trajectory_values, rewards, dones, padding_mask, seq_range = (-input_seq_len, None))
        expected_value = expected_value[:, :-1, :]

        return expected_value

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
        squared_error = (expected_value.detach() - estimated_value).square()
            
        # Reduce the squared error tensor to a scalar loss value, applying optional masking to focus loss calculation.
        reduced_loss = self.select_tensor_reduction(squared_error, mask)

        return reduced_loss

    def calculate_bipolar_advantage_loss(self, estimated_value: torch.Tensor, expected_value: torch.Tensor,
                                padding_mask: torch.Tensor, polar_point: float = 1.0):
        """
        This function calculates a loss that aims to balance short-term and long-term rewards by adopting a bipolar 
        interpretation of advantage, which is the difference between the estimated and expected values. The bipolar 
        advantage loss function dynamically adjusts gamma and lambda parameters, promoting actions that may increase 
        or decrease returns. This adaptive behavior facilitates learning with an optimal scale of advantage, encouraging 
        a deeper exploration of the action-value space.

        The loss calculation incorporates a fourth-degree polynomial error component to achieve balance at specified 
        polar points, thus promoting a bipolar distribution of advantage values. This method aims for a nuanced adjustment 
        of advantage values to reach a desired distribution that supports effective learning strategies. Additionally, 
        by including a second-degree polynomial error component, the loss further incentivizes actions that lead to 
        an increase in the expected value, aligning with the objective of maximizing long-term returns.

        :param estimated_value: Tensor representing the model's estimated value for given states/actions.
        :param expected_value: Tensor representing the target or expected value for given states/actions.
        :param padding_mask: Tensor indicating valid positions (1) and padding (0) for loss calculation.
        :param polar_point: Float specifying the magnitude of the polar points around which the bipolar distribution of 
        advantages is centered. The function uses both positive and negative values of this parameter (+polar_point and 
        -polar_point) within a fourth-degree polynomial error component to foster a bipolar distribution of advantage 
        values. This dual-point consideration allows for a nuanced balancing of advantage values, promoting learning 
        that effectively navigates between maximizing and minimizing expected returns based on the situational context.
        :return: The computed bipolar advantage loss, or None if an update to the learnable parameters is not deemed necessary.
        """
        if self.should_update_learnable_td():
            advantage = expected_value - estimated_value.detach()
            
            # Compute the value_incentive component of the bipolar TD error, scaled by the advantage.
            # This component aims to encourage actions that increase the expected value.
            second_degree_polynomial_error = (advantage - polar_point).square()
            
            # Calculate the fourth-degree polynomial error component, focusing on the squared difference
            # between squared advantage and squared polar distance.
            fourth_degree_polynomial_error = (advantage.square() + (polar_point)**2).square()

            # Form the total bipolar TD error by combining the polynomial error with the
            # value_incentive component, modulating the TD error to guide expected value adjustments.
            bipolar_advantage_error = fourth_degree_polynomial_error + second_degree_polynomial_error
            
            # Aggregate the scaled bipolar TD error to produce a consolidated loss value,
            # considering padding in the input tensors for accurate error computation.
            bipolar_advantage_loss = self.select_tensor_reduction(bipolar_advantage_error, padding_mask)
        else:
            bipolar_advantage_loss = None
            
        return bipolar_advantage_loss
        
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
        advantage = (expected_value.detach() - estimated_value.detach())
            
        # # Normalize the advantage to enhance training stability, ensuring consistent gradient scales.
        advantage = self.normalize_advantage(advantage, padding_mask)

        return advantage

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass
