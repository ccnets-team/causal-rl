import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.data_structures  import AgentTransitions
import torch
import random

def get_sampled_seq_len(max_seq_len, sigma_factor = 4):
    """
    Sample a sequence length using a half-normal distribution, focusing on longer lengths.
    
    Parameters:
        max_seq_len (int): The maximum length of the sequence.
        sigma_factor (float): A factor to adjust sigma, resulting in a focus on longer lengths.
        
    Returns:
        int: A sampled sequence length, ensuring it's at least 2 and no more than max_seq_len.
    """
    sigma = max_seq_len / sigma_factor  # Dynamically calculate sigma based on some factor
    # Sample using a normal distribution and mirror it to focus on longer lengths
    sample = abs(np.random.normal(loc=max_seq_len, scale=sigma))
    sampled_length = int(min(max_seq_len, max(2, sample)))
    return sampled_length

class EnvironmentPool: 
    def __init__(self, env_config, max_seq_len, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device        
        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, max_seq_len, test_env, device, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, max_seq_len, test_env, device, use_graphics = use_graphics, \
                worker_id = int(w_id + i), seed= int(w_id + i)) \
                for i in range(worker_num)]
            
    def reset(self):
        for it in self.env_list:
            it.reset_environment()  

    def end(self):
        for it in self.env_list:
            it.env.close()

    def step_environments(self):
        for env in self.env_list:
            env.step()
            
    def fetch_transitions(self):
        transitions = AgentTransitions()
        for env_idx, env in enumerate(self.env_list):
            agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated = env.output_transitions()
            transitions.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated)
        return transitions
        
    def explore_environments(self, trainer, training):
        trainer.set_train(training = training)
        state_tensor = torch.cat([env.observations.get_obs() for env in self.env_list], dim=0)
        padding_mask = torch.cat([env.observations.mask for env in self.env_list], dim=0)
        
        max_seq_len = trainer.get_input_seq_len()
        state_tensor = trainer.normalize_states(state_tensor)

        if training:
            input_seq_len = get_sampled_seq_len(max_seq_len)  # Clean up and introduce variability
        else:
            input_seq_len = max_seq_len
            
        selected_state_tensor = state_tensor[:, -input_seq_len:]
        selected_padding_mask = padding_mask[:, -input_seq_len:]
                    
        action_tensor = trainer.get_action(selected_state_tensor, selected_padding_mask, training=training)
        
        # Apply actions to environments
        self.apply_actions_to_envs(action_tensor)

    def apply_actions_to_envs(self, action_tensor):
        np_action = action_tensor.cpu().numpy()
        start_idx = 0
        for env in self.env_list:
            end_idx = start_idx + len(env.agent_dec)
            valid_action = np_action[start_idx:end_idx][env.agent_dec]

            select_valid_action = valid_action[:, -1, :]
            env.update(select_valid_action)
            start_idx = end_idx
            
    @staticmethod
    def create_train_environments(env_config, seq_len, device):
        return EnvironmentPool(env_config, seq_len, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_eval_environments(env_config, seq_len, device, use_graphics):
        return EnvironmentPool(env_config, seq_len, device, test_env=True, use_graphics = use_graphics)

    @staticmethod
    def create_test_environments(env_config, seq_len, device, use_graphics):
        return EnvironmentPool(env_config, seq_len, device, test_env=True, use_graphics = use_graphics)
