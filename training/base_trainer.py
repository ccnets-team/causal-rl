import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import compute_gae, get_discounted_rewards, get_end_next_state, get_termination_step
from utils.structure.trajectory_handler  import BatchTrajectory
from nn.gpt import GPT2

class BaseTrainer(TrainingManager, StrategyManager):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
        TrainingManager.__init__(self, optimization_params, networks, target_networks)
        StrategyManager.__init__(self, env_config, exploration_params, normalization_params, device)
        
        self.env_config: EnvConfig = env_config
        
        self.trainer_name = trainer_name
        self.discount_factor = algorithm_params.discount_factor
        self.batch_size = training_params.batch_size
        self.reward_scale = normalization_params.reward_scale
        
        self.use_gae_advantage = algorithm_params.use_gae_advantage

        self.use_sequence_batch = False
        for network_role in networks:
            if isinstance(network_role.net, GPT2):
                self.use_sequence_batch = True
                break  # Optional: If you know there's only one instance of TransformerEncoder, you can exit the loop early

        self.samples_per_step  = env_config.samples_per_step 

        self.num_td_steps = algorithm_params.num_td_steps
        self.buffer_type = memory_params.buffer_type

        self.device = device
        
    def calculate_advantage(self, estimated_value, expected_value):
        with torch.no_grad():
            advantage = (expected_value - estimated_value)
        return advantage

    def calculate_value_loss(self, estimated_value, expected_value, mask=None):
        loss = (estimated_value - expected_value).square()
        
        if mask is not None:
            loss = loss[mask > 0].flatten()
            
        return loss.mean()
        
    def get_discount_factor(self):
        return self.discount_factor 

    def compute_gae_advantage(self, states, rewards, next_states, dones):
        critic = self.get_networks()[0]
        trajectory_states = torch.cat([states, next_states[:, -1:]], dim=1)
        trajectory_values = critic(trajectory_states)  # Assuming critic outputs values for each state in the trajectory
        advantages = compute_gae(trajectory_values, rewards, dones).detach()
        if self.use_sequence_batch:
            return advantages
        return advantages[:,0,:].unsqueeze(1)
    
    def select_first_transitions(self, *tensor_sequences: torch.Tensor):
        results = tuple(tensor[:, 0, :].unsqueeze(1) for tensor in tensor_sequences)
    
        # If only one tensor is passed, return the tensor directly instead of a tuple    
        if len(results) == 1:
            return results[0]
        return results
    
    def select_trajectory_segment(self, trajectory: BatchTrajectory):
        """Extract and process the trajectory based on the use_sequence_batch flag."""

        if self.use_sequence_batch:
            return trajectory
        else:
            states = trajectory.state[:, 0, :].unsqueeze(1)
            actions = trajectory.action[:, 0, :].unsqueeze(1)
            rewards = trajectory.reward[:, 0, :].unsqueeze(1)
            next_states = trajectory.next_state[:, 0, :].unsqueeze(1)
            dones = trajectory.done[:, 0, :].unsqueeze(1)

            return BatchTrajectory(states, actions, rewards, next_states, dones)

    def zero_out_values_post_done(self, dones: torch.Tensor, *tensors: torch.Tensor):
        """
        Shifts the 'dones' tensor and zeros out the values in the provided tensors wherever the shifted mask is True.
        
        Args:
        - dones (torch.Tensor): A tensor indicating terminal states in a sequence.
        - *tensors (torch.Tensor): Any number of tensors whose values should be zeroed out based on the shifted 'dones' tensor.

        Returns:
        - Tuple of tensors with updated values.
        """
        mask = torch.zeros_like(dones)
        mask[:, 1:, :] = dones[:, :-1, :]
        
        zeroed_tensors = tuple(tensor.masked_fill_(mask > 0, 0) for tensor in tensors)
        
        # If only one tensor is passed, return the tensor directly instead of a tuple    
        if len(zeroed_tensors) == 1:
            return zeroed_tensors[0]
        return zeroed_tensors

    def compute_values(self, trajectory: BatchTrajectory, estimated_value: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute the advantage and expected value."""
        
        states, actions, rewards, next_states, dones = trajectory
        
        with torch.no_grad():
            if self.use_gae_advantage:
                advantage = self.compute_gae_advantage(states, rewards, next_states, dones)
                expected_value = advantage + estimated_value
            else:
                expected_value = self.calculate_expected_value(rewards, next_states, dones)
                advantage = self.calculate_advantage(estimated_value, expected_value)
        
        return expected_value, advantage
            
    def calculate_expected_value(self, rewards, next_states, dones):
        """
        Computes the expected return for transitions.
        Considers immediate rewards and potential future rewards 
        based on the described mechanism.
        """
        batch_size, seq_len, _ = rewards.shape
        discount_factors = self._expand_discount_factors(batch_size, seq_len)
        discounted_rewards = get_discounted_rewards(rewards, discount_factors)
        
        done, end_step = get_termination_step(dones)
        next_state = get_end_next_state(next_states, end_step)
        
        future_value = self.trainer_calculate_future_value(next_state)
        
        if self.use_sequence_batch:
            expected_value = discounted_rewards + self._calculate_future_values_discounted(batch_size, seq_len, done, future_value)
        else:
            expected_value = self._calculate_first_sequence_expected_value(discounted_rewards, done, end_step, future_value)
        
        return expected_value

    def _expand_discount_factors(self, batch_size, seq_len):
        discount_factors = (self.discount_factor ** torch.arange(seq_len).float()).to(self.device)
        return discount_factors.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    def _calculate_future_values_discounted(self, batch_size, seq_len, done, future_value):
        gamma = self.get_discount_factor()
        flip_discount_factors = gamma * self._expand_discount_factors(batch_size, seq_len).flip(dims=[1])
        return flip_discount_factors * ((1 - done) * future_value).unsqueeze(dim=1).expand(-1, seq_len, -1)

    def _calculate_first_sequence_expected_value(self, discounted_rewards, done, end_step, future_value):
        gamma = self.get_discount_factor()
        future_value_discounted = (1 - done) * (gamma ** end_step) * future_value
        return (discounted_rewards[:, 0, :] + future_value_discounted).unsqueeze(dim=1)

    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    @abstractmethod
    def trainer_calculate_future_value(self, next_state):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
