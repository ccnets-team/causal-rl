import time
from tqdm.notebook import tqdm
from environments.generator import EnvGenerator
from utils.init import set_seed
from utils.printer import print_step, print_metrics, print_scores
from utils.logger import log_data
from utils.loader import save_trainer, load_trainer
from utils.structure.env_config import EnvConfig
from utils.structure.trajectory_handler  import MultiEnvTrajectories
from memory.replay_buffer import ExperienceMemory
from training.managers.record_manager import RecordManager
import logging
PRINT_INTERVAL = 100
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_TRAINING_START_STEP = 1000

class RLTune(RecordManager):
    def __init__(self, env_config: EnvConfig, trainer, device, use_graphics = False, use_print = False):
        super().__init__(trainer.trainer_name, env_config, trainer.rl_params)
        
        self.trainer = trainer.initialize(env_config, device)
        
        num_trainer_env = env_config.num_test_env + env_config.num_environments
        self.train_env = EnvGenerator.create_train_environments(env_config, device, num_trainer_env) 
        self.test_env = EnvGenerator.create_test_env(env_config, device, use_graphics, num_trainer_env) 
        
        rl_params = trainer.rl_params
        
        memory_params = rl_params.memory
        algorithm_params = rl_params.algorithm
        training_params = rl_params.training
        exploration_params = rl_params.exploration
        self.memory = ExperienceMemory(env_config, algorithm_params, memory_params, device)
        
        self.use_print = use_print
        self.max_steps = exploration_params.max_steps
        self.batch_size = training_params.batch_size
        self.train_frequency   = training_params.train_frequency  
        self.replay_ratio = training_params.replay_ratio
        self.buffer_size = memory_params.buffer_size
        
        self.print_interval = PRINT_INTERVAL
        self.save_interval = DEFAULT_SAVE_INTERVAL
        self.training_start_step = DEFAULT_TRAINING_START_STEP
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.train_env.end()
        self.test_env.end()        
        
    def push_trajectories(self, multi_env_trajectories: MultiEnvTrajectories):
        self.memory.push_env_trajectories(multi_env_trajectories)
        
    def save_rl_tune(self, step: int) -> None:
        """
        Save the model based on conditions evaluated against step.
        
        :param step: Current step to evaluate against for saving the model.
        """
        ...
        try:
            if step % (self.save_interval) != 0:
                return
            save_path = self.save_path if self.is_best_period() else self.temp_save_path
            save_trainer(self.trainer, save_path)             
        except Exception as e:
            self.logger.error(f"Failed to save trainer: {e}")        


    def train(self, on_policy: bool = False) -> None:
        """
        Train the model based on the provided policy.
        
        :param on_policy: Specifies whether to train on policy or not. Defaults to False.
        """
        train_env = self.train_env
        test_env = self.test_env

        for step in tqdm(range(self.max_steps)):
            self.init_step(step)
            self.handle_test_env(test_env)
            self.train_based_on_policy(on_policy, train_env, step)
            self.save_rl_tune(step)
            self.end_step()
            if self.should_print(step):
                self.print_step_info(step)  
            
    def train_based_on_policy(self, on_policy: bool, train_env, step: int) -> None:
        """
        Train the model based on whether the policy is on-policy or off-policy.

        :param on_policy: Specifies whether to train on policy or not.
        :param train_env: The training environment.
        :param step: Current training step.
        """
        if on_policy:
            self.train_on_policy(train_env, step)
        else:
            self.train_off_policy(train_env, step)

    def train_on_policy(self, trainum_environments, step: int) -> None:
        """
        Train the model with on-policy algorithms.
        
        :param trainum_environments: Training environments.
        :param step: Current training step.
        """
        if self.should_update_strategy(step):
            samples = self.memory.get_agent_samples(self.batch_size)
            self.trainer.update_strategy(samples)

        if self.should_reset_memory():
            self.reset_memory_and_train()
        else:
            self.collect_train_data(trainum_environments)
        
    def train_off_policy(self, trainum_environments, step):
        self.collect_train_data(trainum_environments)
        if self.should_update_strategy(step):
            samples = self.memory.get_agent_samples(self.batch_size)
            self.trainer.update_strategy(samples)
        
        if self.should_train_step(step):
            self.train_step()
            
    def test(self):
        load_trainer(self.trainer, self.save_path)
        test_env = self.test_env
        for step in tqdm(range(self.max_steps)):
            self.init_step(step)
            self.handle_test_env(test_env)
            self.end_step()
            if self.should_print(step):
                self.print_step_info(step)  

    def should_print(self, step: int) -> bool:
        return step > 0 and step % self.print_interval == 0
    
    def should_update_strategy(self, step: int) -> bool:
        return step % self.train_frequency == 0 and len(self.memory) >= self.batch_size
    
    def should_reset_memory(self) -> bool:
        return len(self.memory) >= self.buffer_size
    
    def reset_memory_and_train(self) -> None:
        for _ in range(self.total_on_policy_iterations):
            self.train_step()
        self.memory.reset_buffers()

    def should_train_step(self, step: int) -> bool:
        return step >= self.training_start_step
    
    def train_step(self):
        transition = self.memory.sample_agent_transition(self.batch_size)
        if transition is None:
            return
        self.trainer.transform_transition(transition)
        train_data = self.trainer.train_model(transition)
        # If each trainer has its own metrics tracker, adjust this part
        self.metrics_tracker.add_step(train_data)

    def init_step(self, step):
        if self.pivot_time is None:
            self.pivot_time = time.time()
        set_seed(step)
        
    def end_step(self):
        cur_time = time.time()
        self.time_cost = (cur_time - self.pivot_time)

    def handle_test_env(self, test_env):
        trainer = self.trainer
        if test_env is None:
            return
        test_env.step_env()   
        test_multi_env_trajectories = test_env.fetch_env()
        self.record_transitions(test_multi_env_trajectories, training=False)
        test_env.explore_env(trainer, training=False)

    def collect_train_data(self, train_env) -> None:
        train_env.step_env()   
        multi_env_trajectories = train_env.fetch_env()
        self.record_transitions(multi_env_trajectories, training=True)
        self.push_trajectories(multi_env_trajectories)
        train_env.explore_env(self.trainer, training=True)
        return 

    def print_step_info(self, step) -> None:
        self.compute_records()
        train_reward_per_step , test_reward_per_step, train_accumulative_rewards, test_accumulative_rewards, metrics = self.get_records()
        log_data(self.trainer, self.tensor_board_logger, train_reward_per_step , test_reward_per_step, train_accumulative_rewards, test_accumulative_rewards, metrics, step, self.time_cost)
        
        if self.use_print:  
            print_step(self.trainer, self.memory, step, self.time_cost)
            print_metrics(metrics)
            print_scores(train_reward_per_step , test_reward_per_step, train_accumulative_rewards, test_accumulative_rewards)
        self.pivot_time = None