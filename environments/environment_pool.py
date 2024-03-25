import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.data_structures  import AgentTransitions
import torch

class EnvironmentPool: 
    def __init__(self, env_config, max_seq_len, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device        
        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, max_seq_len, test_env, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, max_seq_len, test_env, use_graphics = use_graphics, \
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
            agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, np_content_length = env.output_transitions()
            transitions.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, np_content_length)
        return transitions
        
    def explore_environments(self, trainer, training):
        trainer.set_train(training = training)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_mask = np.concatenate([env.observations.mask for env in self.env_list], axis=0)
        
        _state_tensor = torch.from_numpy(np_state).to(self.device)
        _padding_mask = torch.from_numpy(np_mask).to(self.device)
        
        # In your training loop or function
        input_seq_len = trainer.get_input_seq_len()
        state_tensor = _state_tensor[:, -input_seq_len:]
        padding_mask = _padding_mask[:, -input_seq_len:]
        
        if training:
            padding_mask, content_lengths = trainer.apply_sequence_masking(padding_mask)
        else:
            content_lengths = trainer.get_optimal_content_lengths(input_seq_len)
            content_lengths = content_lengths.expand(padding_mask.size(0))
        
        state_tensor = trainer.normalize_states(state_tensor)
        action_tensor = trainer.get_action(state_tensor, padding_mask, training=training)
        
        # Apply actions to environments
        self.apply_actions_to_envs(action_tensor, content_lengths)

    def apply_actions_to_envs(self, action_tensor, content_lengths):
        np_action = action_tensor.cpu().numpy()
        np_content_lengths = content_lengths.cpu().numpy()
        start_idx = 0
        for env in self.env_list:
            end_idx = start_idx + len(env.agent_dec)
            valid_action = np_action[start_idx:end_idx][env.agent_dec]
            valid_content_lengths = np_content_lengths[start_idx:end_idx][env.agent_dec]

            select_valid_action = valid_action[:, -1, :]
            env.update(select_valid_action, valid_content_lengths)
            start_idx = end_idx
            
    @staticmethod
    def create_train_environments(env_config, max_seq_len, device):
        return EnvironmentPool(env_config, max_seq_len, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_eval_environments(env_config, max_seq_len, device, use_graphics):
        return EnvironmentPool(env_config, max_seq_len, device, test_env=True, use_graphics = use_graphics)

    @staticmethod
    def create_test_environments(env_config, max_seq_len, device, use_graphics):
        return EnvironmentPool(env_config, max_seq_len, device, test_env=True, use_graphics = use_graphics)
