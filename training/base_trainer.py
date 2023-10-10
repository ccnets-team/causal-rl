import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from training.managers.utils.advantage_scaler import scale_advantage
from abc import abstractmethod
from nn.roles.actor_network import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import compute_gae, get_discounted_rewards, get_termination_step, get_end_next_state

class BaseTrainer(TrainingManager, StrategyManager):
    def __init__(self, trainer_name, env_config: EnvConfig, rl_parmas: RLParameters, networks, target_networks, device):  
        training_params, algorithm_params, network_params, \
            optimization_params, exploration_params, memory_params, normalization_params = rl_parmas
        TrainingManager.__init__(self, optimization_params, networks, target_networks)
        StrategyManager.__init__(self, env_config, algorithm_params, exploration_params, normalization_params, device)
        
        self.env_config: EnvConfig = env_config
        
        self.trainer_name = trainer_name
        self.discount_factor = algorithm_params.discount_factor
        self.batch_size = training_params.batch_size
        self.reward_scale = normalization_params.reward_scale
        self.advantage_scaler = normalization_params.advantage_scaler
        
        self.samples_per_step  = env_config.samples_per_step 

        self.num_td_steps = algorithm_params.num_td_steps
        self.use_gae_advantage = algorithm_params.use_gae_advantage
        self.buffer_type = memory_params.buffer_type

        self.device = device
        
    def calculate_advantage(self, estimated_value, expected_value):
        with torch.no_grad():
            advantage = (expected_value - estimated_value)
            advantage = scale_advantage(advantage, self.advantage_scaler)
        return advantage

    def calculate_value_loss(self, estimated_value, expected_value):
        return F.mse_loss(estimated_value, expected_value)
    
    def get_discount_factor(self):
        return self.discount_factor 
    
    def get_discount_factors(self):
        return (self.discount_factor ** torch.arange(self.num_td_steps).float()).to(self.device)

    def compute_gae_advantage(self, states, rewards, next_states, dones):
        critic = self.get_networks()[0]
        trajectory_states = torch.cat([states, next_states[:, -1:]], dim=1)
        trajectory_values = critic(trajectory_states)  # Assuming critic outputs values for each state in the trajectory
        advantages = compute_gae(trajectory_values, rewards, dones).detach()
        return advantages

    def select_first_transitions(self, *tensor_sequences: torch.Tensor):
        """Extract the appropriate input for each tensor based on the GAE flag."""
        if self.use_gae_advantage:
            results = tuple(tensor for tensor in tensor_sequences)
        else:
            results = tuple(tensor[:, 0, :] for tensor in tensor_sequences)

        # If only one tensor is passed, return the tensor directly instead of a tuple
        if len(results) == 1:
            return results[0]
        return results
        
    def select_last_transitions(self, dones, *tensor_sequences: torch.Tensor):
        """Extract the appropriate input for each tensor based on the GAE flag."""
        if self.use_gae_advantage:
            results = tuple(tensor for tensor in tensor_sequences)
        else:
            _, end_step = get_termination_step(dones)
            indices = (end_step - 1).squeeze(1)
            results = tuple(tensor[torch.arange(tensor.shape[0]), indices] for tensor in tensor_sequences)
        
        # If only one tensor is passed, return the tensor directly instead of a tuple
        if len(results) == 1:
            return results[0]
        return results

    def calculate_expected_value(self, rewards, next_states, dones):
        discount_factors = self.get_discount_factors()
        discounted_rewards = get_discounted_rewards(rewards, dones, discount_factors)
        done, end_step = get_termination_step(dones)
        next_state = get_end_next_state(next_states, end_step)
        gamma = self.get_discount_factor()
        
        future_value = self.trainer_calculate_future_value(gamma, end_step, next_state)
        expected_value = discounted_rewards + (1 - done) * future_value
        return expected_value

    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    def compute_values(self, states: torch.Tensor, rewards: torch.Tensor, 
                                             next_states: torch.Tensor, dones: torch.Tensor, 
                                             estimated_value: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute the advantage and expected value."""
        with torch.no_grad():
            if self.use_gae_advantage:
                advantage = self.compute_gae_advantage(states, rewards, next_states, dones)
                expected_value = advantage + estimated_value
            else:
                expected_value = self.calculate_expected_value(rewards, next_states, dones)
                advantage = self.calculate_advantage(estimated_value, expected_value)
        return expected_value, advantage 
                    
    @abstractmethod
    def trainer_calculate_future_value(self, gamma, end_step, next_state):
        pass

    @abstractmethod
    def get_action(self, state, training):
        pass
    
