from nn.gpt import GPT
from nn.utils.network_init import ModelParams

class TrainingParameters:
    # Initialize training parameters for a reinforcement learning model.
    def __init__(self, batch_size=8, replay_ratio=1, train_interval=1):
        self.batch_size = batch_size  # Number of samples processed before model update; larger batch size can lead to more stable but slower training.
        self.replay_ratio = replay_ratio  # Ratio for how often past experiences are reused in training (batch size / samples per step).
        self.train_interval = train_interval  # Frequency of training updates, based on the number of explorations before each update.

class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, gpt_seq_length=16, discount_factor=0.99, advantage_lambda=0.98):
        self.gpt_seq_length = gpt_seq_length  # Maximum sequence length for training and exploration. In training, it defines the length of sequences used for calculating TD steps. In exploration, it sets the upper limit for sequence length.
        self.discount_factor = discount_factor  # Discount factor for future rewards.
        self.advantage_lambda = advantage_lambda # TD (Temporal Difference) or GAE (Generalized Advantage Estimation) lambda parameter for weighting advantages in policy optimization.

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.01, network_type=GPT):
        self.critic_network = network_type  # Selected model-based network used for the critic.
        self.actor_network = network_type  # Selected model-based network used for the actor.
        self.rev_env_network = network_type  # Selected model-based network for reverse environment modeling.
        self.critic_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the critic network.
        self.actor_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the actor network.
        self.rev_env_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the reverse environment network.

class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, lr=5e-5, lr_decay_ratio=2e-1, scheduler_type='cyclic', tau=1e-1, clip_grad_range=None): 
        self.lr = lr  # Learning rate for optimization algorithms, crucial for convergence.
        self.lr_decay_ratio = lr_decay_ratio  # Ratio for learning rate decay over the course of training. In 'cyclic', it's used to determine the base_lr.
        self.scheduler_type = scheduler_type  # Type of learning rate scheduler: 'linear', 'exponential', or 'cyclic'.
        self.tau = tau  # Target network update rate, used in algorithms with target networks.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.

class ExplorationParameters:
    def __init__(self, max_steps=100000):
        self.max_steps = max_steps  # Maximum number of steps for the exploration phase. This defines the period over which the exploration strategy is applied.
        
class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_size=16000):
        self.buffer_size = int(buffer_size)  # Total size of the memory buffer, impacting how many past experiences can be stored.
        self.early_training_start_step = None  # Optional step count to start training earlier than when replay buffer is full.
        
class NormalizationParameters:
    def __init__(self, state_normalizer='exponential_moving_mean_var', reward_normalizer='exponential_moving_mean_var', 
                 exponential_moving_alpha = 1e-4, clip_norm_range = 10.0):
        self.state_normalizer = state_normalizer  # Defines the method for normalizing state values, using approaches like 'running_mean_std'.
        self.reward_normalizer = reward_normalizer  # Specifies the method for normalizing rewards, such as 'running_mean_std' or 'running_abs_mean'.
        self.exponential_moving_alpha = exponential_moving_alpha
        self.clip_norm_range = clip_norm_range

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

    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of RLParameters' direct attributes
        if name in ["training", "algorithm", "network", "optimization", "exploration", "memory", "normalization"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of RLParameters
            super().__setattr__(name, value)

    def __iter__(self):
        yield from [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]
