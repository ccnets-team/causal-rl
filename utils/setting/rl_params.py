import numpy as np
from nn.gpt import GPT
from nn.super_net import SuperNet

DEFAULT_TRAINING_START_STEP = 1000

class TrainingParameters:
    def __init__(self, replay_ratio=3, train_frequency=5, batch_size=512):
        self.replay_ratio = replay_ratio
        self.train_frequency = train_frequency
        self.batch_size = batch_size
        self.training_start_step = DEFAULT_TRAINING_START_STEP
        
    def minimum_samples_per_step(self):
        samples_per_step = int(max(1, np.ceil(self.batch_size/(self.replay_ratio))))
        return samples_per_step        
    
class AlgorithmParameters:
    def __init__(self, num_td_steps=10, discount_factor=0.99, curiosity_factor = 0.0, use_gae_advantage = True):
        self.num_td_steps = num_td_steps
        self.discount_factor = discount_factor
        self.use_gae_advantage = use_gae_advantage
        self.curiosity_factor = curiosity_factor
            
class NetworkParameters:
    def __init__(self, num_layer=4, hidden_size=128, dropout = 0.0):
        self.critic_network = GPT
        self.actor_network = GPT
        self.reverse_env_network = GPT
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout = dropout

class OptimizationParameters:
    def __init__(self, beta1=0.9, lr_gamma=0.9998, step_size=32, lr=3e-4, tau=5e-3):
        self.beta1 = beta1
        self.lr_gamma = lr_gamma
        self.step_size = step_size
        self.lr = lr
        self.tau = tau

class ExplorationParameters:
    def __init__(self, noise_type = None, initial_exploration = 1.0, min_exploration = 0.01, decay_percentage = 0.8, decay_mode = 'linear', \
        max_steps=1000000):
        self.noise_type = noise_type
        self.initial_exploration = initial_exploration
        self.min_exploration = min_exploration
        self.decay_percentage = decay_percentage
        self.decay_mode = decay_mode
        self.max_steps = max_steps

class MemoryParameters:
    def __init__(self, buffer_type = 'standard', buffer_size = 1000000):
        self.buffer_type = buffer_type
        self.buffer_size = int(buffer_size)
        
class NormalizationParameters:
    def __init__(self, reward_scale=1, state_normalizer='running_z_standardizer'):
        self.reward_scale = reward_scale
        self.state_normalizer = state_normalizer

class RLParameters:
    def __init__(self,
                 training: TrainingParameters = None,
                 algorithm: AlgorithmParameters = None,
                 network: NetworkParameters = None,
                 optimization: OptimizationParameters = None,
                 exploration: ExplorationParameters = None,
                 memory: MemoryParameters = None,
                 normalization: NormalizationParameters = None):
        
        self.training = TrainingParameters() if training is None else training
        self.algorithm = AlgorithmParameters() if algorithm is None else algorithm
        self.network = NetworkParameters() if network is None else network
        self.optimization = OptimizationParameters() if optimization is None else optimization
        self.exploration = ExplorationParameters() if exploration is None else exploration
        self.memory = MemoryParameters() if memory is None else memory
        self.normalization = NormalizationParameters() if normalization is None else normalization

    def on_policy_iterations(self):
        return int(self.training.replay_ratio * self.memory.buffer_size / self.training.batch_size)
            
    def __iter__(self):
        yield self.training
        yield self.algorithm
        yield self.network
        yield self.optimization
        yield self.exploration
        yield self.memory
        yield self.normalization
        
        
        
        
        

