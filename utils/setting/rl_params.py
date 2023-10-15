import numpy as np
from nn.super_net import SuperNet
from nn.transformer import TransformerEncoder 
DEFAULT_TRAINING_START_STEP = 1000

class TrainingParameters:
    def __init__(self, replay_ratio=4, train_frequency=1, batch_size=512):
        self.replay_ratio = replay_ratio
        self.train_frequency = train_frequency
        self.batch_size = batch_size
        self.training_start_step = DEFAULT_TRAINING_START_STEP
        
    def minimum_samples_per_step(self):
        samples_per_step = int(max(1, np.ceil(self.batch_size/(self.train_frequency*self.replay_ratio))))
        return samples_per_step        
    
class AlgorithmParameters:
    def __init__(self, discount_factor=0.99, num_td_steps=1, use_sequence_batch = False, use_gae_advantage = False, use_curiosity = False):
        self.discount_factor = discount_factor
        self.num_td_steps = num_td_steps
        self.curiosity_factor = 0.1
        self.use_gae_advantage = use_gae_advantage
        self.use_sequence_batch = use_sequence_batch
        self.use_curiosity = use_curiosity
            
class NetworkParameters:
    def __init__(self,  num_layer=4, hidden_size=128, value_network = SuperNet, policy_network = SuperNet, reverse_env_network = TransformerEncoder,\
                critic_joint_type = 'cat', actor_joint_type = 'cat', rev_env_joint_type = 'cat'):
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.value_network = value_network
        self.policy_network = policy_network
        self.reverse_env_network = reverse_env_network
        self.critic_joint_type = critic_joint_type
        self.actor_joint_type = actor_joint_type
        self.rev_env_joint_type = rev_env_joint_type

class OptimizationParameters:
    def __init__(self, beta1=0.9, lr_gamma=0.9998, step_size=32, lr=3e-4, tau=5e-3):
        self.beta1 = beta1
        self.lr_gamma = lr_gamma
        self.step_size = step_size
        self.lr = lr
        self.tau = tau

class ExplorationParameters:
    def __init__(self, noise_type = None, initial_exploration = 1.0, min_exploration = 0.01, decay_percentage = 0.8, decay_mode = 'linear', \
        max_steps=1000000, use_deterministic_policy = False):
        self.noise_type = noise_type
        self.initial_exploration = initial_exploration
        self.min_exploration = min_exploration
        self.decay_percentage = decay_percentage
        self.decay_mode = decay_mode
        self.max_steps = max_steps
        self.use_deterministic_policy = use_deterministic_policy

class MemoryParameters:
    def __init__(self, buffer_type = 'standard', buffer_size = 1000000):
        self.buffer_type = buffer_type
        self.buffer_size = int(buffer_size)
        
class NormalizationParameters:
    def __init__(self, reward_scale=1, reward_shift=0, state_normalizer='running_z_standardizer', reward_normalizer='none', advantage_scaler='none'):
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer
        self.advantage_scaler = advantage_scaler

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
        
        
        
        
        

