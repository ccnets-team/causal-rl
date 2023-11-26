import numpy as np
from utils.logger import get_log_name
from pathlib import Path
import os
from utils.structure.metrics_recorder import MetricsTracker, RewardTracker
from utils.structure.trajectory_handler import MultiEnvTrajectories
from utils.printer import save_training_parameters_to_file
from torch.utils.tensorboard import SummaryWriter

class RecordManager:
    def __init__(self, trainer_name, env_config, rl_params):
        env_name = env_config.env_name
        self.tensor_board_logger = None
        
        save_path = "./saved"
        log_dir = f"./log/{env_name}/{trainer_name}"
        log_path = get_log_name(log_dir)
        
        self.save_path = self._ensure_directory_exists(f"{save_path}/" + f"{env_name}/{trainer_name}/")
        self.temp_save_path = self._ensure_directory_exists(f"{save_path}/temp/" + f"{env_name}/{trainer_name}/")
        self.log_path = self._ensure_directory_exists(log_path)
        save_training_parameters_to_file(self.log_path, trainer_name= trainer_name, rl_params=rl_params)

        self.train_reward_per_step  = -np.inf
        self.test_reward_per_step = -np.inf
        self.train_accumulative_rewards = -np.inf
        self.test_accumulative_rewards = -np.inf

        self.tracking_interval = int(rl_params.memory.buffer_size/env_config.samples_per_step) 
        self.pivot_time = None
        self.time_cost = None
        self.max_steps = rl_params.exploration.max_steps

        self.metrics_tracker = MetricsTracker(self.tracking_interval)
        self.train_tracker = RewardTracker(self.tracking_interval)
        self.test_tracker = RewardTracker(self.tracking_interval)

    def init_logger(self):
        if self.tensor_board_logger is None:
            self.tensor_board_logger = SummaryWriter(log_dir=self.log_path)
    
    def _ensure_directory_exists(self, path):
        if not Path(path).exists(): 
            os.makedirs(path, exist_ok=True)
        return path

    def compute_records(self):
        self.init_logger()
        
        self.test_reward_per_step = self.test_tracker.compute_average()
        self.train_reward_per_step = self.train_tracker.compute_average()
        
        self.test_accumulative_rewards = self.test_tracker.compute_accumulative_rewards()
        self.train_accumulative_rewards = self.train_tracker.compute_accumulative_rewards()

        self.avg_metrics = self.metrics_tracker.compute_average()
        
    def get_records(self):
        return self.train_reward_per_step , self.test_reward_per_step, self.train_accumulative_rewards, self.test_accumulative_rewards, self.avg_metrics
        
    def record_transitions(self, transitions: MultiEnvTrajectories, training: bool):        
        env_ids = transitions.env_ids
        agent_ids = transitions.agent_ids
        rewards = transitions.rewards
        dones_terminated = transitions.dones_terminated
        dones_truncated = transitions.dones_truncated
        if training:
            self.train_tracker._add_rewards(env_ids, agent_ids, rewards, dones_terminated, dones_truncated)
        else:
            self.test_tracker._add_rewards(env_ids, agent_ids, rewards, dones_terminated, dones_truncated)

    def is_best_period(self):
        return self.test_tracker.is_best_record_period()
