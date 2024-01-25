import numpy as np
from utils.logger import get_log_name
from pathlib import Path
import os
from utils.structure.metrics_recorder import MetricsTracker, RewardTracker
from utils.structure.trajectories import MultiEnvTrajectories
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
        self.train_reward_tracker = RewardTracker(self.tracking_interval)
        self.test_reward_tracker = RewardTracker(self.tracking_interval)

    def init_logger(self):
        if self.tensor_board_logger is None:
            self.tensor_board_logger = SummaryWriter(log_dir=self.log_path)
    
    def _ensure_directory_exists(self, path):
        if not Path(path).exists(): 
            os.makedirs(path, exist_ok=True)
        return path

    def compute_records(self):
        self.init_logger()
        self.test_reward_per_step = self.test_reward_tracker.compute_average_reward()
        self.train_reward_per_step = self.train_reward_tracker.compute_average_reward()
        
        self.test_accumulative_rewards = self.test_reward_tracker.compute_accumulative_rewards()
        self.train_accumulative_rewards = self.train_reward_tracker.compute_accumulative_rewards()

        self.avg_metrics = self.metrics_tracker.compute_average_metrics()
        
    def get_records(self):
        return self.train_reward_per_step , self.test_reward_per_step, self.train_accumulative_rewards, self.test_accumulative_rewards, self.avg_metrics
        
    def record_trajectories(self, trajectories: MultiEnvTrajectories, training: bool):        
        env_ids = trajectories.env_ids
        agent_ids = trajectories.agent_ids
        rewards = trajectories.rewards
        dones_terminated = trajectories.dones_terminated
        dones_truncated = trajectories.dones_truncated
        if training:
            self.train_reward_tracker._add_rewards(env_ids, agent_ids, rewards, dones_terminated, dones_truncated)
        else:
            self.test_reward_tracker._add_rewards(env_ids, agent_ids, rewards, dones_terminated, dones_truncated)

    def is_best_period(self):
        return self.test_reward_tracker.is_best_record_period()
