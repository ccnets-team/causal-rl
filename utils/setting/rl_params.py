from nn.gpt import GPT
from nn.super_net import SuperNet
from nn.utils.network_init import ModelParams

class TrainingParameters:
    # Initialize training parameters for a reinforcement learning model.
    def __init__(self, trainer_name='causal_rl', use_on_policy=False, batch_size=64, replay_ratio=1, train_interval=1):
        self.trainer_name = trainer_name  # Specifies the type of trainer algorithm to be used (e.g., 'causal_rl', 'ddpg', 'a2c', etc.). Determines the learning strategy and underlying mechanics of the model.
        self.use_on_policy = use_on_policy  # Indicates if training uses on-policy (True) or off-policy (False) methods.
        self.batch_size = batch_size  # Number of samples processed before model update; larger batch size can lead to more stable but slower training.
        self.replay_ratio = replay_ratio  # Ratio for how often past experiences are reused in training (batch size / samples per step).
        self.train_interval = train_interval  # Frequency of training updates, based on the number of explorations before each update.
        self.early_training_start_step = None  # Optional step count to start training earlier than when replay buffer is full.

class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, num_td_steps=16, train_seq_length=16, explore_seq_length=12, discount_factor=0.99, advantage_lambda=0.99, use_gae_advantage=False):
        self.num_td_steps = num_td_steps  # The number of TD steps for multi-step returns. Determines the temporal horizon used for learning.
        self.train_seq_length = train_seq_length  # Sequence length for training. Represents the number of consecutive states used in each training update.
        self.explore_seq_length = explore_seq_length  # Sequence length during exploration. Impacts how the model interacts with and perceives its environment.
        self.discount_factor = discount_factor  # Discount factor for future rewards.
        self.advantage_lambda = advantage_lambda # TD or GAE lambda parameter for weighting
        self.use_gae_advantage = use_gae_advantage  # Whether to use Generalized Advantage Estimation
            
        assert self.num_td_steps >= self.train_seq_length, "num_td_steps must be greater than or equal to train_seq_length"

        # Explanation:
        # num_td_steps: Represents the horizon over which Temporal Difference updates are calculated.
        # train_seq_length: Defines the window of consecutive data points used for each training update. 
        # It is crucial that num_td_steps is not less than train_seq_length, as it would mean the model tries to learn from a future beyond its TD update horizon.

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.01, 
                 tau=1e-1, use_target_network=True, network_type=GPT):
        self.critic_network = network_type  # Selected model-based network used for the critic.
        self.actor_network = network_type  # Selected model-based network used for the actor.
        self.rev_env_network = network_type  # Selected model-based network for reverse environment modeling.
        self.critic_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the critic network.
        self.actor_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the actor network.
        self.rev_env_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the reverse environment network.
        self.tau = tau  # Target network update rate, used in algorithms with target networks.
        self.use_target_network = use_target_network  # Flag to determine whether to use target networks for stability.

class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, lr=2e-5, lr_decay_ratio=1e-1, clip_grad_range=None):
        self.lr = lr  # Learning rate for optimization algorithms, crucial for convergence.
        self.lr_decay_ratio = lr_decay_ratio  # Ratio for learning rate decay over the course of training.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.

class ExplorationParameters:
    # Initialize exploration parameters
    # The exploration rate starts at 1 (maximum exploration) and decays over time towards a minimum value (e.g., 0.01) during the training process. 
    # The specific decay behavior (e.g., linear, exponential) and the rate of decay are managed by the ExplorationUtils class.
    # By default, the exploration rate decays linearly from 1 to 0.01 over 80% of the max_steps.
    def __init__(self, noise_type=None, max_steps=100000):
        self.noise_type = noise_type  # Type of exploration noise used to encourage exploration in the agent. This could be any noise algorithm like epsilon-greedy, OU noise strategy, etc.
        self.max_steps = max_steps  # Maximum number of steps for the exploration phase. This defines the period over which the exploration strategy is applied.
        
class MemoryParameters:
    # Initialize memory parameters
    def __init__(self, buffer_priority=0.0, buffer_size=256000):
        self.buffer_priority = buffer_priority  # Adjusts the prioritization in the memory buffer. A value of 0 indicates a standard buffer.
        self.buffer_size = int(buffer_size)  # Total size of the memory buffer, impacting how many past experiences can be stored.
        
class NormalizationParameters:
    def __init__(self, state_normalizer='running_mean_std', reward_normalizer=None, value_normalizer='running_mean_std'):
        self.state_normalizer = state_normalizer  # Defines the method for normalizing state values, using approaches like 'running_mean_std'.
        self.reward_normalizer = reward_normalizer  # Specifies the method for normalizing rewards, such as 'running_mean_std' or 'running_abs_mean'.
        self.value_normalizer = value_normalizer

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
