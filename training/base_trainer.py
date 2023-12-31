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
from .trainer_utils import adaptive_masked_tensor_reduction, masked_tensor_reduction, apply_seq_mask, create_model_seq_mask
from .trainer_utils import create_padding_mask_before_dones, convert_trajectory_data

class BaseTrainer(TrainingManager, NormalizationUtils, ExplorationUtils):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_params: RLParameters, networks, target_networks, device):
        self._unpack_rl_params(rl_params)
        self._init_trainer_specific_params()
        self._init_training_manager(networks, target_networks, device)
        self._init_normalization_utils(env_config, device)
        self._init_exploration_utils()
        self.discount_factors = compute_discounted_future_value(self.discount_factor, self.num_td_steps).to(self.device)

    def _unpack_rl_params(self, rl_params):
        (self.training_params, self.algorithm_params, self.network_params, 
         self.optimization_params, self.exploration_params, 
         self.memory_params, self.normalization_params) = rl_params

    def _init_training_manager(self, networks, target_networks, device):
        training_start_step = self._compute_training_start_step()
        total_iterations = max(self.exploration_params.max_steps - training_start_step, 0)//self.training_params.train_interval
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
        self.reduction_type = self.algorithm_params.reduction_type

    def _compute_training_start_step(self):
        training_start_step = self.training_params.early_training_start_step
        if training_start_step is None:
            batch_size_ratio = self.training_params.batch_size / self.training_params.replay_ratio
            training_start_step = self.memory_params.buffer_size // int(batch_size_ratio)
        return training_start_step
    
    def select_model_seq_length(self, trajectory):
        states, actions, rewards, next_states, dones = trajectory
        
        padding_mask = create_padding_mask_before_dones(dones)

        model_seq_length = self.model_seq_length 
        model_seq_mask = create_model_seq_mask(padding_mask, model_seq_length)

        # Apply the mask to each trajectory component
        sel_states = apply_seq_mask(states, model_seq_mask, model_seq_length)
        sel_actions = apply_seq_mask(actions, model_seq_mask, model_seq_length)
        sel_rewards = apply_seq_mask(rewards, model_seq_mask, model_seq_length)
        sel_next_states = apply_seq_mask(next_states, model_seq_mask, model_seq_length)
        sel_dones = apply_seq_mask(dones, model_seq_mask, model_seq_length)

        return sel_states, sel_actions, sel_rewards, sel_next_states, sel_dones, model_seq_mask

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
            elif self.reduction_type == 'none':
                return tensor[mask>0].flatten()
            else:
                raise ValueError("Invalid reduction type. Choose 'adaptive', 'batch', 'seq', 'all', or 'cross'.")
        else:
            if self.reduction_type == 'none':
                return tensor
            else:
                return tensor.mean()

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        squared_error = (estimated_value - expected_value).square()
        reduced_loss = self.select_tensor_reduction(squared_error, mask)
        return reduced_loss

    def compute_td_errors(self, trajectory: BatchTrajectory):
        states, actions, rewards, next_states, dones = trajectory 
        
        padding_mask = create_padding_mask_before_dones(dones)
        trajectory_states, trajectory_mask = convert_trajectory_data(states, next_states, mask=padding_mask)

        mode_seq_sum_discounted_gammas = sum(self.discount_factors[:,-rewards.shape[1]:])
        scaled_rewards = rewards/mode_seq_sum_discounted_gammas
        
        with torch.no_grad():
            estimated_value = self.trainer_calculate_value_estimate(states, mask=padding_mask)
            trajectory_values = self.trainer_calculate_future_value(trajectory_states, trajectory_mask)
            
            if self.use_gae_advantage:
                _advantage = calculate_gae_returns(trajectory_values, scaled_rewards, dones, self.discount_factor, self.advantage_lambda)
                expected_value = (_advantage + estimated_value)
            else:
                expected_value = calculate_lambda_returns(trajectory_values, scaled_rewards, dones, self.discount_factor, self.advantage_lambda)
                
            _advantage = (expected_value - estimated_value)
            advantage = self.normalize_advantage(_advantage)
            td_errors = advantage.abs()
            
        trajectory.push_td_errors(td_errors, padding_mask)

    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor, model_seq_mask: torch.Tensor):
        """Compute the advantage and expected value."""
        states, actions, rewards, next_states, dones = trajectory 

        padding_mask = create_padding_mask_before_dones(dones)
        trajectory_states, trajectory_mask = convert_trajectory_data(states, next_states, mask=padding_mask)

        td_steps_sum_discounted_gammas = sum(self.discount_factors[:,-rewards.shape[1]:])
        scaled_rewards = rewards/td_steps_sum_discounted_gammas
        
        with torch.no_grad():
            trajectory_values = self.trainer_calculate_future_value(trajectory_states, trajectory_mask)
            
            if self.use_gae_advantage:
                _advantage = calculate_gae_returns(trajectory_values, scaled_rewards, dones, self.discount_factor, self.advantage_lambda)
                _expected_value = (_advantage + estimated_value)
            else:
                _expected_value = calculate_lambda_returns(trajectory_values, scaled_rewards, dones, self.discount_factor, self.advantage_lambda)
            
            model_seq_length = estimated_value.shape[1]
            expected_value = apply_seq_mask(_expected_value, model_seq_mask, model_seq_length)
            _advantage = (expected_value - estimated_value)
            
            advantage = self.normalize_advantage(_advantage)
            self.update_advantage(_advantage)
            
        return expected_value, advantage

    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None, use_target = False):
        pass

    @abstractmethod
    def trainer_calculate_value_estimate(self, states, mask = None):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
