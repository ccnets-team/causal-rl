from nn.gpt import GPT
import torch
# Start training after this number of steps

class TrainingParameters:
    # Initialize training parameters
    def __init__(self, batch_size=64, replay_ratio=4, train_intervel = 1):
        self.batch_size = batch_size  # Batch size for training
        self.replay_ratio = replay_ratio  # How often past experiences are reused in training (batch size / samples per step)
        self.train_intervel  = train_intervel  # Determines how frequently training updates occur based on the number of explorations before each update
        self.early_training_start_step = "none"  # Training starts when the replay buffer is full. Set to a specific step count to start training earlier.
                
class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, num_td_steps=16, discount_factor=0.995, advantage_lambda = 0.95, use_gae_advantage=False):
        self.num_td_steps = num_td_steps  # Number of TD steps for multi-step returns
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.advantage_lambda = advantage_lambda # TD or GAE lambda parameter for weighting n-step returns.
        # Flag to enable dynamic adjustment of TD steps based on exploration-exploitation balance.
        # When True, the number of TD steps is automatically adjusted during training, potentially 
        # increasing as the exploration rate decreases, to balance exploration and exploitation.
        self.use_gae_advantage = use_gae_advantage  # Whether to use Generalized Advantage Estimation

class GPTParams:
    def __init__(self, d_model, num_layers, dropout):
        """
        Initialize a GPT network.	
        Args:
        - d_model (int): Dimension of the model.
        - num_layers (int): Number of layers in the network.
        - dropout (float): Dropout rate.
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        # Initialize other GPT specific configurations here

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.05, use_target_network=True):
        self.critic_network = GPT
        self.actor_network = GPT
        self.reverse_env_network = GPT
        self.critic_params = GPTParams(d_model = d_model, num_layers = num_layers, dropout = dropout)
        self.actor_params = GPTParams(d_model = d_model, num_layers = num_layers, dropout = dropout)
        self.rev_env_params = GPTParams(d_model = d_model, num_layers = num_layers, dropout = dropout)
        self.use_target_network = use_target_network
                   
class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, lr=3e-5, tau=5e-3, clip_grad_range=None):
        self.lr = lr  # Initial learning rate
        self.tau = tau  # Target network update rate
        self.clip_grad_range = clip_grad_range  
        
class ExplorationParameters:
    # Initialize exploration parameters
    def __init__(self, noise_type='none', 
                 initial_exploration=1.0, min_exploration=0.01, decay_percentage=0.8, decay_mode='linear',
                 max_steps=100000):
        self.noise_type = noise_type  # Type of exploration noise ('none' for no noise)
        self.initial_exploration = initial_exploration  # Initial exploration rate
        self.min_exploration = min_exploration  # Minimum exploration rate
        self.decay_percentage = decay_percentage  # Percentage of total steps for exploration decay
        self.decay_mode = decay_mode  # Mode of exploration decay ('linear' for linear decay)
        self.max_steps = max_steps 
        
class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_type='standard', buffer_size=64000):
        self.buffer_type = buffer_type  # Type of replay buffer ('standard' for standard buffer)
        self.buffer_size = int(buffer_size)  # Size of the replay buffer
        
class NormalizationParameters:
    # Initialize normalization parameters
    def __init__(self, reward_scale = 0.05, clip_norm_range = 10, window_size = 20, 
                 reward_normalizer='running_abs_mean', state_normalizer='running_mean_std'):
        self.reward_scale = reward_scale  # Scaling factor for rewards
        self.clip_norm_range = clip_norm_range  
        self.window_size = window_size  
        self.reward_normalizer = reward_normalizer  # reward normalization method (e.g., 'running_mean_std, hybrid_moving_mean_var')
        self.state_normalizer = state_normalizer  # State normalization method (e.g., 'running_mean_std, hybrid_moving_mean_var')

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
            
    def __iter__(self):
        yield self.training
        yield self.algorithm 
        yield self.network
        yield self.optimization
        yield self.exploration
        yield self.memory
        yield self.normalization
        
