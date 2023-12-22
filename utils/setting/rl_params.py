from nn.gpt import GPT, GPTParams

class TrainingParameters:
    # Initialize training parameters
    def __init__(self, batch_size=64, replay_ratio=4, train_intervel = 1):
        self.batch_size = batch_size  # Batch size for training
        self.replay_ratio = replay_ratio  # How often past experiences are reused in training (batch size / samples per step)
        self.train_intervel  = train_intervel  # Determines how frequently training updates occur based on the number of explorations before each update
        self.early_training_start_step = None  # Training starts when the replay buffer is full. Set to a specific step count to start training earlier.
                
class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, num_td_steps=16, discount_factor=0.995, advantage_lambda = 0.99, use_gae_advantage=False):
        self.num_td_steps = num_td_steps  # Number of TD steps for multi-step returns
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.advantage_lambda = advantage_lambda # TD or GAE lambda parameter for weight    ing n-step returns.
        self.use_gae_advantage = use_gae_advantage  # Whether to use Generalized Advantage Estimation

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.05, 
                 tau=1e-1, use_target_network=True):
        self.critic_network = GPT  # GPT-based network used for the critic.
        self.actor_network = GPT  # GPT-based network used for the actor.
        self.reverse_env_network = GPT  # GPT-based network for reverse environment modeling.
        self.critic_params = GPTParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the critic network.
        self.actor_params = GPTParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the actor network.
        self.rev_env_params = GPTParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the reverse environment network.
        self.tau = tau  # Target network update rate, used in algorithms with target networks.
        self.use_target_network = use_target_network  # Flag to determine whether to use target networks for stability.

class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, lr=2e-5, lr_decay_ratio=5e-3, clip_grad_range=None):
        self.lr = lr  # Learning rate for optimization algorithms, crucial for convergence.
        self.lr_decay_ratio = lr_decay_ratio  # Ratio for learning rate decay over the course of training.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.

class ExplorationParameters:
    # Initialize exploration parameters
    def __init__(self, noise_type='none', 
                 initial_exploration=0.0, min_exploration=0.0, decay_percentage=0.0, decay_mode=None,
                 max_steps=100000):
        self.noise_type = noise_type  # Type of exploration noise used to encourage exploration in the agent.
        self.initial_exploration = initial_exploration  # Initial rate of exploration, determining initial randomness in actions.
        self.min_exploration = min_exploration  # Minimum exploration rate, ensuring some level of exploration throughout training.
        self.decay_percentage = decay_percentage  # Defines how quickly the exploration rate decays.
        self.decay_mode = decay_mode  # Determines the method of decay for exploration rate (e.g., 'linear').
        self.max_steps = max_steps  # Maximum number of steps for the exploration phase.

class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_type='standard', buffer_size=64000):
        self.buffer_type = buffer_type  # Determines the type of memory buffer used for storing experiences.
        self.buffer_size = int(buffer_size)  # Total size of the memory buffer, impacting how many past experiences can be stored.

class NormalizationParameters:
    def __init__(self, reward_scale=1, 
                 reward_normalizer='running_mean_std', state_normalizer='running_mean_std', 
                 advantage_normalizer=None, min_threshold=None, max_threshold=None):
        self.reward_scale = reward_scale  # Scaling factor for rewards, used to adjust the magnitude of rewards appplies after reward normalization.
        self.reward_normalizer = reward_normalizer  # Specifies the method for normalizing rewards, such as 'running_mean_std' or 'running_abs_mean'.
        self.state_normalizer = state_normalizer  # Defines the method for normalizing state values, using approaches like 'running_mean_std'.
        self.advantage_normalizer = advantage_normalizer  # Determines the normalization technique for advantage values, for example, 'L1_norm'.
        self.min_threshold = min_threshold  # Sets the lower threshold for scaling when normalizing advantages; scaling is applied if the mean is below this value.
        self.max_threshold = max_threshold  # Sets the upper threshold for scaling when normalizing advantages; scaling is applied if the mean is above this value.

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