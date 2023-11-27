
import time
from environments.environment_pool import EnvironmentPool
from utils.init import set_seed
from utils.printer import print_step, print_metrics, print_scores
from utils.logger import log_data
from utils.structure.trajectory_handler  import MultiEnvTrajectories
from memory.replay_buffer import ExperienceMemory
from utils.loader import save_trainer, load_trainer
from training.managers.record_manager import RecordManager
import logging

PRINT_INTERVAL = 100
DEFAULT_SAVE_INTERVAL = 1000

class RLTuneHelper:
    def __init__(self, parent, trainer_name, env_config, rl_params, use_graphics, use_print):
        self.recorder = RecordManager(trainer_name, env_config, rl_params)

        self.parent = parent
        self.use_graphics, self.use_print = use_graphics, use_print
        self.env_config, self.rl_params = env_config, rl_params
        
        self.print_interval = PRINT_INTERVAL
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
        self.recorder.record_transitions(multi_env_trajectories, training=training)

    def push_trajectories(self, multi_env_trajectories: MultiEnvTrajectories):
        """Pushes trajectories to memory."""
        self.parent.memory.push_env_trajectories(multi_env_trajectories)

    def should_update_strategy(self, step: int) -> bool:
        """Checks if the strategy should be updated."""
        return len(self.parent.memory) >= self.batch_size
    
    def should_reset_memory(self) -> bool:
        """Checks if the memory should be reset."""
        return self.parent.memory.get_buffer_size() >= self.buffer_size

    def should_train_step(self, step: int) -> bool:
        """Checks if the model should be trained on the current step."""
        return (step % self.train_intervel == 0) and (step >= self.training_start_step)

    # Private Helpers
    def _initialize_training_parameters(self):
        training_params = self.rl_params.training
        exploration_params = self.rl_params.exploration
        memory_params = self.rl_params.memory
        
        self.max_steps = exploration_params.max_steps
        self.training_start_step = training_params.training_start_step
        self.batch_size = training_params.batch_size
        self.replay_ratio = training_params.replay_ratio
        self.train_intervel = training_params.train_intervel
        self.buffer_size = memory_params.buffer_size
        self.total_on_policy_iterations = (self.buffer_size * self.replay_ratio) // self.batch_size

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
            self.parent.train_env = EnvironmentPool.create_train_environments(self.env_config, self.rl_params.algorithm.num_td_steps, self.parent.device)

    def _ensure_test_environment_exists(self):
        if not self.parent.test_env:
            self.parent.test_env = EnvironmentPool.create_test_environments(self.env_config, self.rl_params.algorithm.num_td_steps, self.parent.device, self.use_graphics)

    def _ensure_memory_exists(self):
        if not self.parent.memory:
            self.parent.memory = ExperienceMemory(self.env_config, self.rl_params.training, self.rl_params.algorithm, self.rl_params.memory, self.parent.device)

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
        
        if self.use_print:
            self._print_step_details(step, metrics)
        
    def _print_step_details(self, step, metrics):
        """Prints detailed information about the current step."""
        print_step(self.parent.trainer, self.parent.memory, step, time.time() - self.recorder.pivot_time)
        print_metrics(metrics[4])
        print_scores(*metrics[:4])
        self.recorder.pivot_time = None
        
    def _should_print(self, step: int) -> bool:
       """Determines if logs should be printed for a given step."""        
       return step > 0 and step % self.print_interval == 0
        
