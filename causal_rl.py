from tqdm.notebook import tqdm
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from utils.setting.causal_rl_helper import CausalRLHelper
from utils.wandb_logger import wandb_end
from training.causal_trainer import CausalTrainer
from utils.printer import print_rl_params

class CausalRL:
    """
    Class for tuning and training reinforcement learning models using a specified Trainer.
    """
    def __init__(self, env_config: EnvConfig, rl_params: RLParameters, device, use_print=False, use_wandb=False):
        """Initialize an instance of RLTune.
        Args:
            env_config (EnvConfig): Configuration for the environment.
            trainer (Trainer): Trainer object for the reinforcement learning algorithm.
            device (Device): Computational device (e.g., CPU, GPU).
            use_graphics (bool, optional): Whether to use graphics during training/testing. Default is False.
            use_print (bool, optional): Whether to print training/testing logs. Default is False.
        """
        
        print_rl_params('causal_rl', rl_params)
        self.device = device
        self.max_steps = rl_params.max_steps
        self.train_env, self.eval_env, self.test_env, self.memory = None, None, None, None
        self.trainer = CausalTrainer(env_config, rl_params, device)
        self.helper = CausalRLHelper(self, env_config, rl_params, use_print, use_wandb)

    # Context Management
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.helper.use_wandb:
            wandb_end()
        self._close_environments()

    def _close_environments(self):
        self._end_environment(self.train_env)
        self._end_environment(self.eval_env)
        self._end_environment(self.test_env)

    def _end_environment(self, env):
        if env:
            env.end()

    # Main Public Methods
    def train(self, resume_training: bool = False, use_graphics=False) -> None:
        """
        Train the model based on the provided policy.
        """
        self.helper.setup(use_graphics, training=True)
        
        if resume_training:
            self.helper.load_model()

        for step in tqdm(range(self.max_steps)):
            # Training Logic Methods
            """Unified training logic for each step."""
            self.helper.init_step(step)
            self.interact_environment(self.eval_env, training=False)            
            self.interact_environment(self.train_env, training=True)
            self.train_off_policy(step)
            self.helper.end_step(step)  
            
        self.helper.save_model()
        self._end_environment(self.train_env)
        self._end_environment(self.eval_env)
        
    def train_off_policy(self, step: int) -> None:
        """Train the model with off-policy algorithms."""
        samples = self.memory.sample_batch_trajectory()

        if self.helper.should_update_strategy(samples, step):
            """Fetch samples and update strategy."""
            self.trainer.update_normalizer(samples)
        
        if self.helper.should_train_step(samples, step):
            """Single step of training."""
            self.trainer.normalize_trajectories(samples)
            train_data = self.trainer.train_model(samples)
            self.helper.add_train_metrics(train_data)
            
        self.trainer.update_exploration_rate()
        
    # Environment Interaction
    def interact_environment(self, env, training=True):
        """Unified method to interact with environments."""
        env.step_environments()
        transitions = env.fetch_transitions()
        self.helper.record_data(transitions, training=training)

        if training:
            self.memory.push_transitions(transitions)

        env.explore_environments(self.trainer, training=training)

    def test(self, max_episodes: int = 100, use_graphics=True) -> None:
        self.helper.setup(use_graphics, training=False)
        self.helper.load_model()
        
        step = 0
        for episode in tqdm(range(max_episodes), desc="Testing"):
            while True:  # This loop will run until broken by a condition inside
                if self.helper.get_test_episode_counts() > episode:
                    break
                """Unified testing logic for each step."""
                self.helper.init_step(step)
                self.interact_environment(self.test_env, training=False)
                self.helper.end_step(step)
                step += 1  # Increment the global step count after processing