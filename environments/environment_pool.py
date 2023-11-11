import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.trajectory_handler  import MultiEnvTrajectories
import torch
import random

class EnvironmentPool: 
    def __init__(self, env_config, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device
        self.num_td_steps = env_config.num_td_steps

        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, test_env, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, test_env, use_graphics = use_graphics, \
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
            
    def roll_left(self, tensor, mask):
        mask_sum = torch.sum(mask, dim=1)
        adjusted_tensor = tensor.clone()
        adjusted_mask = mask.clone()

        # Tensor to keep track of roll amounts
        roll_lengths = torch.zeros(mask_sum.size(0), dtype=torch.int64)

        for i in range(tensor.size(1)):
            move_len = tensor.size(1) - i
            if move_len == 0 or move_len == tensor.size(1):
                continue
            indices = (mask_sum == i).squeeze()
            if indices.sum() < 1:
                continue
            adjusted_tensor[indices] = torch.roll(tensor[indices], -move_len, dims=1)
            adjusted_mask[indices] = torch.roll(adjusted_mask[indices], -move_len, dims=1)

            # Record the roll amount for each row
            roll_lengths[indices] = move_len

        return adjusted_tensor, adjusted_mask, roll_lengths
    
    def explore_env(self, trainer, training):
        trainer.set_train(training = training)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_mask = np.concatenate([env.observations.mask for env in self.env_list], axis=0)
        np_reset = np.concatenate([env.agent_reset for env in self.env_list], axis=0)

        reset_tensor = torch.from_numpy(np_reset).to(self.device)
        state_tensor = torch.from_numpy(np_state).to(self.device)
        mask_tensor = torch.from_numpy(np_mask).to(self.device)
        
        state_tensor, mask_tensor, roll_lengths = self.roll_left(state_tensor, mask_tensor)

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
            valid_lengths = roll_lengths[start_idx:end_idx][env.agent_dec]

            if len(valid_action.shape) > 2:
                select_valid_action = valid_action[:,-1,:]
                for i in range(len(valid_action)):
                    # Get the index for the second dimension based on roll length
                    select_idx = valid_action.shape[1] - valid_lengths[i] - 1
                    
                    # Select the specific action
                    select_valid_action[i] = valid_action[i, select_idx, :]
            env.update(select_valid_action)
            start_idx = end_idx

    @staticmethod
    def create_train_environments(env_config, device):
        return EnvironmentPool(env_config, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_test_environments(env_config, device, use_graphics):
        return EnvironmentPool(env_config, device, test_env=True, use_graphics = use_graphics)

