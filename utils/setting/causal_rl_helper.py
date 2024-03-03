import time
from environments.environment_pool import EnvironmentPool
from utils.init import set_seed
from utils.printer import print_step, print_metrics, print_train_scores, print_test_scores
from utils.logger import log_train_data, log_test_data
from utils.wandb_logger import wandb_init, wandb_log_train_data, wandb_log_test_data 
from memory.experience_memory import ExperienceMemory
from utils.loader import save_trainer, load_trainer
from training.managers.record_manager import RecordManager
import logging

DEFAULT_PRINT_INTERVAL = 100
DEFAULT_SAVE_INTERVAL = 1000

class CausalRLHelper:
    def __init__(self, parent, rl_params, use_print, use_wandb):
        trainer_name = 'causal_rl'
        env_config = rl_params.env_config
        self.recorder = RecordManager(trainer_name, env_config, rl_params)
        self.parent = parent
        self.use_print = use_print
        self.env_config, self.rl_params = env_config, rl_params
        if use_wandb:
            wandb_init(env_config, rl_params)
        self.use_wandb = use_wandb
        
        self.print_interval = DEFAULT_PRINT_INTERVAL
        self.save_interval = DEFAULT_SAVE_INTERVAL
        self.logger = logging.getLogger(__name__)

        self._initialize_training_parameters()

    # Setup Methods
    def setup(self, setup_train = True, use_graphics = False, use_eval = False):
        """Configures environments and other necessary parameters for training/testing."""
        self.setup_train = setup_train
        self.use_eval = use_eval
        
        if setup_train:
            """Configures the environment and other parameters for training."""
            self._ensure_train_environment_exists()
            if use_eval:
                self._ensure_eval_environment_exists(use_graphics)
            self._ensure_memory_exists()
        else:
            """Configures the environment for testing."""
            self._ensure_test_environment_exists(use_graphics)

    def save_model(self):
        """Saves the RL model."""
        save_trainer(self.parent.trainer, self.recorder.save_path)

    def load_model(self):
        """Loads the RL model."""
        load_trainer(self.parent.trainer, self.recorder.save_path)

    def get_test_episode_counts(self):
        return self.recorder.test_reward_tracker.get_episode_counts()

    def add_train_metrics(self, train_data):
        """Add training metrics to the recorder."""
        self.recorder.metrics_tracker.add_metrics(train_data)

    def init_step(self, step):
        """Initializes the training/testing step."""        
        if self.recorder.pivot_time is None:
            self.recorder.pivot_time = time.time()
        set_seed(step)

    def end_step(self, step):
        """Finalizes the step and performs save or log operations if conditions are met."""
        if self.setup_train:
            self._save_rl_model_conditional(step)

        self._log_step_info_conditional(step)

    def record_data(self, multi_trajectories, training):
        """Records environment transitions."""      
        self.recorder.record_trajectories(multi_trajectories, self.setup_train, training=training)

    def should_update_strategy(self, samples, step: int) -> bool:
        """Checks if the strategy should be updated."""
        return samples is not None and (step % self.train_interval == 0) and self.use_normalizer

    def should_train_step(self, samples, step: int) -> bool:
        """Checks if the model should be trained on the current step."""
        return samples is not None and (step % self.train_interval == 0) and (step >= self.training_start_step) and (len(self.parent.memory) >= self.batch_size)
        
    # Private Helpers
    def _initialize_training_parameters(self):
        self.gpt_seq_length = self.rl_params.algorithm.gpt_seq_length
        self.max_steps = self.rl_params.max_steps
        self.buffer_size = self.rl_params.buffer_size
        self.batch_size = self.rl_params.batch_size
        self.replay_ratio = self.rl_params.replay_ratio
        self.train_interval = self.rl_params.train_interval
        self.use_normalizer = (self.rl_params.normalization.reward_normalizer) is not None or (self.rl_params.normalization.state_normalizer is not None) 
        
        self.samples_per_step = self.batch_size//self.replay_ratio
        self.training_start_step = self.buffer_size//int(self.batch_size/self.replay_ratio) 
        self.total_on_policy_iterations = int((self.buffer_size * self.replay_ratio) // (self.train_interval*self.batch_size))
        self.td_seq_length = self.rl_params.td_seq_length
        self.num_save_models = 0
        
    def _ensure_train_environment_exists(self):
        if not self.parent.train_env:
            self.parent.train_env = EnvironmentPool.create_train_environments(self.env_config, self.parent.device)
            
    def _ensure_eval_environment_exists(self, use_graphics):
        if not self.parent.eval_env:
            self.parent.eval_env = EnvironmentPool.create_eval_environments(self.env_config, self.parent.device, use_graphics)

    def _ensure_test_environment_exists(self, use_graphics):
        if not self.parent.test_env:
            self.parent.test_env = EnvironmentPool.create_test_environments(self.env_config, self.parent.device, use_graphics)

    def _ensure_memory_exists(self):
        if not self.parent.memory:
            self.parent.memory = ExperienceMemory(self.env_config, self.td_seq_length, self.batch_size, self.buffer_size, self.parent.device)
    
    def should_save_model_at_train_end(self) -> bool:
        """Determines if the model should be saved at the training end."""
        return self.num_save_models < 1
    
    def _save_rl_model_conditional(self, step: int):
        if step % self.save_interval == 0:
            if self.recorder.is_best_period():
                save_path = self.recorder.save_path
                self.num_save_models += 1
            else:
                save_path = self.recorder.temp_save_path
            try:
                save_trainer(self.parent.trainer, save_path)
            except Exception as e:
                self.logger.error(f"Failed to save trainer at step {step}: {e}")

    def _log_step_info_conditional(self, step: int):
        """Logs step info if conditions are met."""
        if step > 0 and step % self.print_interval == 0:
            """Logs information related to the given step."""    
            if self.setup_train:    
                self.recorder.compute_train_records()
                metrics = self.recorder.get_train_records()
                log_train_data(self.parent.trainer, self.recorder.tensor_board_logger, *metrics, step, time.time() - self.recorder.pivot_time)

                if self.use_wandb:
                    wandb_log_train_data(self.parent.trainer, *metrics, step, time.time() - self.recorder.pivot_time)
            else:    
                self.recorder.compute_test_records()
                metrics = self.recorder.get_test_records()
                log_test_data(self.parent.trainer, self.recorder.tensor_board_logger, *metrics, step, time.time() - self.recorder.pivot_time)

                if self.use_wandb:
                    wandb_log_test_data(self.parent.trainer, *metrics, step, time.time() - self.recorder.pivot_time)

            if self.use_print:
                self._print_step_details(step, metrics)
            self.recorder.pivot_time = None
        
    def _print_step_details(self, step, metrics):
        """Prints detailed information about the current step."""
        print_step(self.parent.trainer, self.parent.memory, step, time.time() - self.recorder.pivot_time, self.max_steps)
        if self.setup_train:
            print_metrics(metrics[-1])
            print_train_scores(*metrics[:-1])
        else:
            print_test_scores(*metrics[:-1])
        
    def _should_print(self, step: int) -> bool:
       """Determines if logs should be printed for a given step."""        
       return step > 0 and step % self.print_interval == 0
   

        
