import torch
from torch.functional import F
from training.managers.training_manager import TrainingManager 
from training.managers.strategy_manager import StrategyManager 
from abc import abstractmethod
from nn.roles.actor import _BaseActor
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from .trainer_utils import compute_gae
from utils.structure.trajectory_handler  import BatchTrajectory

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
        return advantages
    
    def select_first_transitions(self, *tensor_sequences: torch.Tensor):
        results = tuple(tensor[:, 0, :].unsqueeze(1) for tensor in tensor_sequences)
    
        # If only one tensor is passed, return the tensor directly instead of a tuple    
        if len(results) == 1:
            return results[0]
        return results

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
        future_values = self.trainer_calculate_future_value(next_states)
        expected_value = rewards + (1 - dones)*self.discount_factor*future_values
        return expected_value

    def reset_actor_noise(self, reset_noise):
        for actor in self.get_networks():
            if isinstance(actor, _BaseActor):
                actor.reset_noise(reset_noise)

    @abstractmethod
    def trainer_calculate_future_value(self, next_state, mask = None):
        pass

    @abstractmethod
    def get_action(self, state, mask = None, training: bool = False):
        pass
    
