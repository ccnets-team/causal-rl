from tqdm.notebook import tqdm
from utils.structure.env_config import EnvConfig
from utils.setting.rl_tune_helper import RLTuneHelper

class RLTune:
    """
    Class for tuning and training reinforcement learning models using a specified Trainer.
    """
    def __init__(self, env_config: EnvConfig, trainer, device, use_graphics=False, use_print=False):
        """Initialize an instance of RLTune.
        Args:
            env_config (EnvConfig): Configuration for the environment.
            trainer (Trainer): Trainer object for the reinforcement learning algorithm.
            device (Device): Computational device (e.g., CPU, GPU).
            use_graphics (bool, optional): Whether to use graphics during training/testing. Default is False.
            use_print (bool, optional): Whether to print training/testing logs. Default is False.
        """
        
        self.trainer = trainer.initialize(env_config, device)
        self.device = device
        self.max_steps = trainer.rl_params.exploration.max_steps
        self.train_env, self.test_env, self.memory = None, None, None
        self.helper = RLTuneHelper(self, trainer.trainer_name, env_config, trainer.rl_params, use_graphics, use_print)

    # Context Management
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close_environments()

    def _close_environments(self):
        self._end_environment(self.train_env)
        self._end_environment(self.test_env)

    def _end_environment(self, env):
        if env:
            env.end()

    # Main Public Methods
    def train(self, on_policy: bool = False, resume_training: bool = False) -> None:
        """
        Train the model based on the provided policy.
        """
        self.helper.setup(training=True)
        
        if resume_training:
            self.helper.load_model()

        training_method = self.train_on_policy if on_policy else self.train_off_policy
        for step in tqdm(range(self.max_steps)):
            self._train_step_logic(step, training_method)

    def test(self, max_episodes: int = 100) -> None:
        self.helper.setup(training=False)
        self.helper.load_model()
        
        step = 0
        for episode in tqdm(range(max_episodes), desc="Testing"):
            while True:  # This loop will run until broken by a condition inside
                if self.helper.get_test_episode_counts() > episode:
                    break
                if step >= self.max_steps:
                    break
                self._test_step_logic(step)
                step += 1  # Increment the global step count after processing
            if step >= self.max_steps:
                break

    # Training Logic Methods
    def _train_step_logic(self, step: int, training_method) -> None:
        """Unified training logic for each step."""
        self.helper.init_step(step)
        self.process_test_environment(self.test_env)
        training_method(step)
        self.helper.end_step(step)

    def _test_step_logic(self, step: int) -> None:
        """Unified testing logic for each step."""
        self.helper.init_step(step)
        self.process_test_environment(self.test_env)
        self.helper.end_step(step)

    def train_on_policy(self, step: int) -> None:
        """Train the model with on-policy algorithms."""
        self.process_train_environment(self.train_env)
        self.trainer.update_exploration_rate()

        if self.helper.should_update_strategy(step):
            self._update_strategy_from_samples()

        if self.helper.should_reset_memory():
            self.reset_memory_and_train()

    def train_off_policy(self, step: int) -> None:
        """Train the model with off-policy algorithms."""
        self.process_train_environment(self.train_env)
        self.trainer.update_exploration_rate()
        
        if self.helper.should_update_strategy(step):
            self._update_strategy_from_samples()
        
        if self.helper.should_train_step(step):
            self.train_step()

    # Helpers for Training
    def _update_strategy_from_samples(self) -> None:
        """Fetch samples and update strategy."""
        samples = self.memory.get_trajectory_data()
        if samples is not None:
            self.trainer.update_strategy(samples)

    def reset_memory_and_train(self) -> None:
        """Reset the memory and train the model again."""
        for _ in range(self.helper.total_on_policy_iterations):
            self.train_step()
        self.memory.reset_buffers()
        
    def train_step(self) -> None:
        """Single step of training."""
        transition = self.memory.sample_trajectory_data()
        if transition is not None:
            self.trainer.transform_transition(transition)
            train_data = self.trainer.train_model(transition)
            self.helper.add_train_metrics(train_data)

    # Environment Interaction
    def interact_environment(self, env, training=True):
        """Unified method to interact with environments."""
        env.step_env()
        multi_env_trajectories = env.fetch_env()
        self.helper.record(multi_env_trajectories, training=training)

        if training:
            self.helper.push_trajectories(multi_env_trajectories)

        env.explore_env(self.trainer, training=training)
        
    def process_train_environment(self, train_env):
        self.interact_environment(train_env, training=True)

    def process_test_environment(self, test_env):
        self.interact_environment(test_env, training=False)
