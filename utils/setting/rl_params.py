import numpy as np
from nn.gpt import GPT
import os

# Start training after this number of steps
DEFAULT_TRAINING_START_STEP = 1000


class TrainingParameters:
    # Initialize training parameters
    def __init__(self, batch_size=512, replay_ratio=2.0, train_intervel = 8):
        self.batch_size = batch_size  # Batch size for training
        self.replay_ratio = replay_ratio  # How often past experiences are reused in training (batch size / samples per step)
        self.train_intervel  = train_intervel  # Determines how frequently training updates occur based on the number of explorations before each update
        self.training_start_step = DEFAULT_TRAINING_START_STEP  
        
    def minimum_samples_per_step(self):
        # Calculate minimum samples per step
        samples_per_step = int(max(1, np.ceil(self.batch_size/(self.replay_ratio))))
        return samples_per_step        
    
class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, num_td_steps=16, discount_factor=0.995, curiosity_factor=0.0, use_gae_advantage=False):
        self.num_td_steps = num_td_steps  # Number of TD steps for multi-step returns
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.curiosity_factor = curiosity_factor  # influences the agent's desire to explore new things and learn through intrinsic rewards
        self.use_gae_advantage = use_gae_advantage  # Whether to use Generalized Advantage Estimation
            
class NetworkParameters:
    # Initialize network parameters
    def __init__(self, num_layer=4, hidden_size=128, dropout = 0.1):
        self.critic_network = GPT  # Critic network architecture (GPT in this case)
        self.actor_network = GPT  # Actor network architecture (GPT in this case)
        self.reverse_env_network = GPT  # Reverse environment network architecture (GPT in this case)
        self.num_layer = num_layer  # Number of layers in the networks
        self.hidden_size = hidden_size  # Demension of model 
        self.dropout = dropout  # Dropout rate
        
class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, beta1=0.9, lr_gamma=0.9998, step_size=4, lr=3e-4, tau=5e-3):
        self.beta1 = beta1  # Beta1 parameter for Adam optimizer
        self.lr_gamma = lr_gamma  # Learning rate decay factor
        self.step_size = step_size  # Step size for learning rate scheduling
        self.lr = lr  # Initial learning rate
        self.tau = tau  # Target network update rate
        
class ExplorationParameters:
    # Initialize exploration parameters
    def __init__(self, noise_type='none', initial_exploration=1.0, min_exploration=0.01, decay_percentage=0.8, decay_mode='linear',
                 max_steps=1000000):
        self.noise_type = noise_type  # Type of exploration noise ('none' for no noise)
        self.initial_exploration = initial_exploration  # Initial exploration rate
        self.min_exploration = min_exploration  # Minimum exploration rate
        self.decay_percentage = decay_percentage  # Percentage of total steps for exploration decay
        self.decay_mode = decay_mode  # Mode of exploration decay ('linear' for linear decay)
        self.max_steps = max_steps  # Maximum number of training steps
        
class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_type='standard', buffer_size=1000000):
        self.buffer_type = buffer_type  # Type of replay buffer ('standard' for standard buffer)
        self.buffer_size = int(buffer_size)  # Size of the replay buffer
        
class NormalizationParameters:
    # Initialize normalization parameters
    def __init__(self, reward_scale=1, state_normalizer='running_z_standardizer'):
        self.reward_scale = reward_scale  # Scaling factor for rewards
        self.state_normalizer = state_normalizer  # State normalization method (e.g., 'running_z_standardizer')

class RLParameters:
    def __init__(self,
                 training: TrainingParameters = None,
                 algorithm: AlgorithmParameters = None,
                 network: NetworkParameters = None,
                 optimization: OptimizationParameters = None,
                 exploration: ExplorationParameters = None,
                 memory: MemoryParameters = None,
                 normalization: NormalizationParameters = None):
        
        # Initialize RL parameters
        self.training = TrainingParameters() if training is None else training
        self.algorithm = AlgorithmParameters() if algorithm is None else algorithm
        self.network = NetworkParameters() if network is None else network
        self.optimization = OptimizationParameters() if optimization is None else optimization
        self.exploration = ExplorationParameters() if exploration is None else exploration
        self.memory = MemoryParameters() if memory is None else memory
        self.normalization = NormalizationParameters() if normalization is None else normalization

    def on_policy_iterations(self):
        # Calculate on-policy iterations, which is the number of iterations needed for a complete cycle
        # through the replay buffer based on the batch size and replay ratio.
        return int(self.training.replay_ratio * self.memory.buffer_size / self.training.batch_size)
            
    def __iter__(self):
        yield self.training
        yield self.algorithm 
        yield self.network
        yield self.optimization
        yield self.exploration
        yield self.memory
        yield self.normalization
        
class RLParamsLoader:
    # Initialize RLParamsLoader with environment name and trainer name
    def __init__(self, env_name, trainer_name): 
        self.env_name = env_name
        self.trainer_name = trainer_name

    @staticmethod
    def parse_value(value_str):
        # Parse a value from a string
        value_str = value_str.strip()
        if value_str.isdigit():
            return int(value_str)
        elif "." in value_str:
            try:
                return float(value_str)
            except ValueError:
                pass
        elif value_str == 'True':
            return True
        elif value_str == 'False':
            return False
        else:
            return value_str

    @classmethod
    def load_params(cls, env_name: str, trainer_name: str, train_time=None) -> dict:
        # Load RL parameters from a txt file
        rl_params = {}

        if train_time is None:
            # Get a list of subdirectories inside the trainer_name directory
            trainer_subdirs = [d for d in os.listdir(os.path.join('./log', env_name, trainer_name)) if os.path.isdir(os.path.join('./log', env_name, trainer_name, d))]
            trainer_subdirs.sort()
            latest_subdir = trainer_subdirs[-1]
            directory_path = os.path.join('./log', env_name, trainer_name, latest_subdir)
        else:
            directory_path = os.path.join('./log', env_name, trainer_name, train_time)

        for filename in os.listdir(directory_path):
            if filename.startswith('parameters'):
                file_path = os.path.join(directory_path, filename)
                break
        else:
            raise ValueError(f"No 'parameters' file found in {directory_path}")

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value_str = line.split(": ")
                    value = cls.parse_value(value_str)
                    rl_params[key] = value

        return rl_params
    
def set_parameters(rl_params, batch_size, num_td_steps, train_frequency, noise_type, network_type, replay_ratio):
    rl_params.training.batch_size = batch_size
    rl_params.algorithm.num_td_steps = num_td_steps
    rl_params.training.train_frequency = train_frequency
    rl_params.exploration.noise_type = noise_type
    rl_params.network.critic_network = network_type
    rl_params.network.actor_network = network_type
    rl_params.network.reverse_env_network = network_type
    rl_params.training.replay_ratio = replay_ratio
