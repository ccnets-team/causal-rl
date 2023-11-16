import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.trajectory_handler  import MultiEnvTrajectories
import torch
import random

class EnvironmentPool: 
    def __init__(self, env_config, num_td_steps, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device
        self.num_td_steps = num_td_steps

        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, num_td_steps, test_env, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, num_td_steps, test_env, use_graphics = use_graphics, \
                worker_id = int(w_id + i), seed= int(w_id + i)) \
                for i in range(worker_num)]
            
    def reset(self):
        for it in self.env_list:
            it.reset_env()  

    def end(self):
        for it in self.env_list:
            it.env.close()

    def fetch_env(self):
        combined_transition = MultiEnvTrajectories()

        for env_idx, env in enumerate(self.env_list):
            agent_ids, obs, action, reward, next_obs, done = env.output_transitions()
            combined_transition.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done)
        return combined_transition

    def step_env(self):
        for env in self.env_list:
            env.step_environment()

    def sample_td_steps(self, exploration_rate):
        """
        Samples the number of TD steps using a Gaussian distribution, with standard deviation adjusted based on exploration rate.
        The mean is set to self.num_td_steps to ensure that the maximum number of TD steps is the most commonly sampled value.
        
        Args:
            exploration_rate (float): The current exploration rate, ranging from 1 (high exploration) to 0 (low exploration).
        """
        mean = self.num_td_steps  # Set mean to the maximum length

        # Adjust standard deviation directly based on exploration rate
        # For a standard deviation (std) of 0.5, the probability increases to about 95.45%.
        std_dev = 0.5*exploration_rate

        # Sample and ensure it falls within the valid range
        random_td_steps = int(round(max(1, min(np.random.normal(mean, std_dev), self.num_td_steps))))
        return random_td_steps

    
    def explore_env(self, trainer, training):
        trainer.set_train(training = training)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_mask = np.concatenate([env.observations.mask for env in self.env_list], axis=0)
        np_reset = np.concatenate([env.agent_reset for env in self.env_list], axis=0)

        reset_tensor = torch.from_numpy(np_reset).to(self.device)
        state_tensor = torch.from_numpy(np_state).to(self.device)
        mask_tensor = torch.from_numpy(np_mask).to(self.device)

        if training:
            # Use the new method to sample the TD steps
            exploration_rate = trainer.get_exploration_rate()
            sampled_td_steps = self.sample_td_steps(exploration_rate)

            # Select the sequence accordingly
            state_tensor = state_tensor[:, -sampled_td_steps:]
            mask_tensor = mask_tensor[:, -sampled_td_steps:]
                            
        state_tensor = trainer.normalize_state(state_tensor)
        action_tensor = trainer.get_action(state_tensor, mask_tensor, training=training)
        if training:
            trainer.reset_actor_noise(reset_noise=reset_tensor)

        for env in self.env_list:
            env.agent_reset.fill(False)
            
        np_action = action_tensor.cpu().numpy()
        start_idx = 0
        for env in self.env_list:
            end_idx = start_idx + len(env.agent_dec)
            valid_action = np_action[start_idx:end_idx][env.agent_dec]
            select_valid_action = valid_action[:,-1,:]
            env.update(select_valid_action)
            start_idx = end_idx

    @staticmethod
    def create_train_environments(env_config, num_td_steps, device):
        return EnvironmentPool(env_config, num_td_steps, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_test_environments(env_config, num_td_steps, device, use_graphics):
        return EnvironmentPool(env_config, num_td_steps, device, test_env=True, use_graphics = use_graphics)

