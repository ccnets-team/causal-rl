from nn.gpt import GPT
from nn.network_utils import ModelParams

class TrainingParameters:
    # Initialize training parameters for a reinforcement learning model.
    def __init__(self, batch_size=64, replay_ratio=1, train_interval=1, max_steps=200000, buffer_size=256000):
        self.batch_size = batch_size  # Number of samples processed before model update; larger batch size can lead to more stable but slower training.
        self.replay_ratio = replay_ratio  # Ratio for how often past experiences are reused in training (batch size / samples per step).
        self.train_interval = train_interval  # Frequency of training updates, based on the number of explorations before each update.
        self.max_steps = max_steps  # Maximum number of steps for the exploration phase. This defines the period over which the exploration strategy is applied.
        self.buffer_size = buffer_size  # Total size of the memory buffer, impacting how many past experiences can be stored.
        # Note: Training begins only after the replay buffer is filled to its full capacity.

class AlgorithmParameters:
    # Initialize algorithm parameters
    def __init__(self, gpt_seq_length=16, discount_factor=0.99, advantage_lambda=0.99):
        self.gpt_seq_length = gpt_seq_length  # Maximum sequence length for training and exploration. In training, it defines the length of sequences used for calculating TD steps. In exploration, it sets the upper limit for sequence length.
        self.discount_factor = discount_factor  # Discount factor for future rewards.
        self.advantage_lambda = advantage_lambda # TD (Temporal Difference) lambda parameter for weighting advantages in policy optimization.

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.02, network_type=GPT):
        self.critic_network = network_type  # Selected model-based network used for the critic.
        self.actor_network = network_type  # Selected model-based network used for the actor.
        self.rev_env_network = network_type  # Selected model-based network for reverse environment modeling.
        self.critic_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the critic network.
        self.actor_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the actor network.
        self.rev_env_params = ModelParams(d_model=d_model, num_layers=num_layers, dropout=dropout)  # Parameters for the reverse environment network.

class OptimizationParameters:
    # Initialize optimization parameters
    def __init__(self, lr=1e-4, min_lr=1e-6, scheduler_type='exponential', tau=1e-1, use_target_network=True, clip_grad_range=None, max_grad_norm=0.5): 
        self.lr = lr  # Learning rate for optimization algorithms, crucial for convergence.
        self.min_lr = min_lr  # Minimum learning rate to which the lr will decay.
        self.scheduler_type = scheduler_type  # Type of learning rate scheduler: 'linear', 'exponential', or 'cyclic'.
        self.tau = tau  # Target network update rate, used in algorithms with target networks.
        self.use_target_network = use_target_network  # Flag to determine whether to use target networks for stability.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.
        self.max_grad_norm = max_grad_norm  # L2 norm threshold for gradient clipping to prevent exploding gradients.

class NormalizationParameters:
    def __init__(self, state_normalizer='running_mean_std', reward_normalizer='running_mean_std', sum_reward_normalizer='running_mean_std', advantage_normalizer='running_abs_mean'):
        self.state_normalizer = state_normalizer  # Defines the method for normalizing state values, using approaches like 'running_mean_std' or 'None'.
        self.reward_normalizer = reward_normalizer  # Defines the method for normalizing reward values, using approaches like 'running_mean_std' or 'None'.
        self.sum_reward_normalizer = sum_reward_normalizer  # Defines the method for normalizing reward values, using approaches like 'running_mean_std' or 'None'.
        self.advantage_normalizer = advantage_normalizer  # Defines the method for normalizing advantage values, using approaches like 'running_abs_mean', 'L1_norm', 'batch_norm', or 'None'.