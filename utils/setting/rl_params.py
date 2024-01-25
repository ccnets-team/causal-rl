from nn.gpt import GPT
from nn.utils.network_init import ModelParams

class TrainingParameters:
    # Initialize training parameters for a reinforcement learning model.
    def __init__(self, trainer_name = 'causal_rl', trainer_variant = 'classic', batch_size=64, replay_ratio=1, train_interval=1):
        self.trainer_name = trainer_name  # Specifies the type of trainer algorithm to be used (e.g., 'causal_rl', 'ddpg', 'a2c', etc.). Determines the learning strategy and underlying mechanics of the model.
        self.trainer_variant = trainer_variant 
        self.batch_size = batch_size  # Number of samples processed before model update; larger batch size can lead to more stable but slower training.
        self.replay_ratio = replay_ratio  # Ratio for how often past experiences are reused in training (batch size / samples per step).
        self.train_interval = train_interval  # Frequency of training updates, based on the number of explorations before each update.

class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, max_seq_length=16, discount_factor=0.99, advantage_lambda=0.99):
        self.min_seq_length = 1  # Minimum sequence length during exploration. Determines the lower bound for the number of consecutive states the model considers while exploring.
        self.max_seq_length = max_seq_length  # Maximum sequence length for training and exploration. In training, it defines the length of sequences used for calculating TD steps. In exploration, it sets the upper limit for sequence length.
        self.discount_factor = discount_factor  # Discount factor for future rewards.
        self.advantage_lambda = advantage_lambda # TD (Temporal Difference) or GAE (Generalized Advantage Estimation) lambda parameter for weighting advantages in policy optimization.
        self.use_on_policy = None # This is set automatically in rl_trainer.py based on whether the trainer is off-policy (False) or on-policy (True).
        self.use_gae_advantage = None # This is set automatically in rl_trainer.py based on whether the policy is off-policy (False) or on-policy (True).

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
    def __init__(self, lr=5e-5, min_lr=5e-6, scheduler_type='exponential', tau=1e-1, use_target_network=True, clip_grad_range=None): 
        self.lr = lr  # Learning rate for optimization algorithms, crucial for convergence.
        self.min_lr = min_lr  # Minimum learning rate to which the lr will decay.
        self.scheduler_type = scheduler_type  # Type of learning rate scheduler: 'linear', 'exponential', or 'cyclic'.
        self.tau = tau  # Target network update rate, used in algorithms with target networks.
        self.use_target_network = use_target_network  # Flag to determine whether to use target networks for stability.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.

class ExplorationParameters:
    # Initialize exploration parameters
    def __init__(self, noise_type=None, max_steps=100000):
        self.noise_type = noise_type  # Type of exploration noise used to encourage exploration in the agent. This could be any noise algorithm like epsilon-greedy, OU noise strategy, etc.
        self.max_steps = max_steps  # Maximum number of steps for the exploration phase. This defines the period over which the exploration strategy is applied.

class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_size=256000):
        self.buffer_size = int(buffer_size)  # Total size of the memory buffer, impacting how many past experiences can be stored.
        # Note: Training begins only after the replay buffer is filled to its full capacity.

class NormalizationParameters:
    def __init__(self, state_normalizer='running_mean_std', reward_normalizer='running_mean_std', advantage_normalizer=None):
        self.state_normalizer = state_normalizer  # Defines the method for normalizing state values, using approaches like 'running_mean_std' or 'exponential_moving_mean_var'.
        self.reward_normalizer = reward_normalizer  # Defines the method for normalizing reward values, using approaches like 'running_mean_std' or 'exponential_moving_mean_var'.
        self.advantage_normalizer = advantage_normalizer  # Defines the method for normalizing advantage values, using approaches like 'running_mean_std' or 'exponential_moving_mean_var'.

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
