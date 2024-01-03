
import time
from environments.environment_pool import EnvironmentPool
from utils.init import set_seed
from utils.printer import print_step, print_metrics, print_scores
from utils.logger import log_data
from utils.wandb_logger import wandb_log_data, wandb_init
from utils.structure.trajectories  import MultiEnvTrajectories
from memory.experience_memory import ExperienceMemory
from utils.loader import save_trainer, load_trainer
from training.managers.record_manager import RecordManager
import logging

DEFAULT_PRINT_INTERVAL = 100
DEFAULT_SAVE_INTERVAL = 1000

class RLTuneHelper:
    def __init__(self, parent, trainer_name, env_config, rl_params, use_graphics, use_print, use_wandb):
        self.recorder = RecordManager(trainer_name, env_config, rl_params)

        self.parent = parent
        self.use_graphics, self.use_print = use_graphics, use_print
        self.env_config, self.rl_params = env_config, rl_params
        if use_wandb:
            wandb_init(trainer_name, env_config, rl_params)
        self.use_wandb = use_wandb
        
        self.use_normalizer = (rl_params.normalization.reward_normalizer) is not None or (rl_params.normalization.state_normalizer is not None) 
        
        self.print_interval = DEFAULT_PRINT_INTERVAL
        self.save_interval = DEFAULT_SAVE_INTERVAL
        self.logger = logging.getLogger(__name__)

        self._initialize_training_parameters()

    # Setup Methods
    def setup(self, training: bool):
        """Configures environments and other necessary parameters for training/testing."""
        self.use_training = training
        if training:
            self._setup_training()
        else:
            self._setup_testing()

    def load_model(self):
        """Loads the RL model."""
        load_trainer(self.parent.trainer, self.recorder.save_path)

    def get_test_episode_counts(self):
        return self.recorder.test_tracker.get_episode_counts()

    def add_train_metrics(self, train_data):
        """Add training metrics to the recorder."""
        self.recorder.metrics_tracker.add_step(train_data)

    def init_step(self, step):
        """Initializes the training/testing step."""        
        if self.recorder.pivot_time is None:
            self.recorder.pivot_time = time.time()
        set_seed(step)

    def end_step(self, step):
        """Finalizes the step and performs save or log operations if conditions are met."""
        if self.use_training:
            self._save_rl_model_conditional(step)

        self._log_step_info_conditional(step)

    def record(self, multi_env_trajectories, training):
        """Records environment transitions."""        
        self.recorder.record_trajectories(multi_env_trajectories, training=training)

    def push_trajectories(self, multi_env_trajectories: MultiEnvTrajectories):
        """Pushes trajectories to memory."""
        exploratoin_rate = self.parent.trainer.get_exploration_rate()
        self.parent.memory.push_trajectory_data(multi_env_trajectories, exploratoin_rate)

    def should_update_strategy(self, step: int) -> bool:
        """Checks if the strategy should be updated."""
        return (step % self.train_interval == 0) and self.use_normalizer
    
    def should_reset_memory(self) -> bool:
        """Checks if the memory should be reset."""
        return self.parent.memory.get_total_data_points() >= self.buffer_size

    def should_train_step(self, step: int) -> bool:
        """Checks if the model should be trained on the current step."""
        return (step % self.train_interval == 0) and (step >= self.training_start_step) and (len(self.parent.memory) >= self.batch_size)

    # Private Helpers
    def _initialize_training_parameters(self):
        training_params = self.rl_params.training
        exploration_params = self.rl_params.exploration
        memory_params = self.rl_params.memory

        self.num_td_steps = self.rl_params.algorithm.num_td_steps
        self.model_seq_length = self.rl_params.algorithm.model_seq_length
        self.use_dynamic_steps = self.rl_params.algorithm.use_dynamic_steps
        
        self.max_steps = exploration_params.max_steps
        self.buffer_size = memory_params.buffer_size
        self.batch_size = training_params.batch_size
        self.replay_ratio = training_params.replay_ratio
        self.train_interval = training_params.train_interval

        self.samples_per_step = training_params.batch_size//training_params.replay_ratio
        self.training_start_step = self.buffer_size//int(self.batch_size/training_params.replay_ratio) if training_params.early_training_start_step is None else training_params.early_training_start_step
        self.total_on_policy_iterations = int((self.buffer_size * self.replay_ratio) // (self.train_interval*self.batch_size))

    def _setup_training(self):
        """Configures the environment and other parameters for training."""
        self._ensure_train_environment_exists()
        self._ensure_test_environment_exists()
        self._ensure_memory_exists()

    def _setup_testing(self):
        """Configures the environment for testing."""
        self._ensure_test_environment_exists()

    def _ensure_train_environment_exists(self):
        if not self.parent.train_env:
            self.parent.train_env = EnvironmentPool.create_train_environments(self.env_config, self.model_seq_length, self.use_dynamic_steps, self.parent.device)
    def _ensure_test_environment_exists(self):
        if not self.parent.test_env:
            self.parent.test_env = EnvironmentPool.create_test_environments(self.env_config, self.model_seq_length, self.use_dynamic_steps, self.parent.device, self.use_graphics)

    def _ensure_memory_exists(self):
        if not self.parent.memory:
            compute_td_errors = self.parent.trainer.compute_td_errors
            self.parent.memory = ExperienceMemory(self.env_config, self.rl_params.training, self.rl_params.algorithm, self.rl_params.memory, compute_td_errors, self.parent.device)

    def _save_rl_model_conditional(self, step: int):
        if step % self.save_interval == 0:
            save_path = self.recorder.save_path if self.recorder.is_best_period() else self.recorder.temp_save_path
            try:
                save_trainer(self.parent.trainer, save_path)
            except Exception as e:
                self.logger.error(f"Failed to save trainer at step {step}: {e}")

    def _log_step_info_conditional(self, step: int):
        """Logs step info if conditions are met."""
        if step > 0 and step % self.print_interval == 0:
            self._log_step_info(step)

    def _log_step_info(self, step: int):
        """Logs information related to the given step."""        
        self.recorder.compute_records()
        metrics = self.recorder.get_records()
        log_data(self.parent.trainer, self.recorder.tensor_board_logger, *metrics, step, time.time() - self.recorder.pivot_time)
        if self.use_wandb:
            wandb_log_data(self.parent.trainer, *metrics, step, time.time() - self.recorder.pivot_time)
        if self.use_print:
            self._print_step_details(step, metrics)
        self.recorder.pivot_time = None
        
    def _print_step_details(self, step, metrics):
        """Prints detailed information about the current step."""
        print_step(self.parent.trainer, self.parent.memory, step, time.time() - self.recorder.pivot_time)
        print_metrics(metrics[4])
        print_scores(*metrics[:4])
        
    def _should_print(self, step: int) -> bool:
       """Determines if logs should be printed for a given step."""        
       return step > 0 and step % self.print_interval == 0
   

        
