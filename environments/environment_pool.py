import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.trajectories  import MultiTrajectories
import torch

class EnvironmentPool: 
    def __init__(self, env_config, gpt_seq_length, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device        
        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, gpt_seq_length, test_env, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, gpt_seq_length, test_env, use_graphics = use_graphics, \
                worker_id = int(w_id + i), seed= int(w_id + i)) \
                for i in range(worker_num)]
            
    def reset(self):
        for it in self.env_list:
            it.reset_environment()  

    def end(self):
        for it in self.env_list:
            it.env.close()

    def step_env(self):
        for env in self.env_list:
            env.step_environment()
            
    def fetch_env(self):
        combined_transition = MultiTrajectories()

        for env_idx, env in enumerate(self.env_list):
            agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated = env.output_transitions()
            combined_transition.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated)
        return combined_transition
        
    def explore_env(self, trainer, training):
        trainer.set_train(training = training)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_mask = np.concatenate([env.observations.mask for env in self.env_list], axis=0)

        state_tensor = torch.from_numpy(np_state).to(self.device)
        padding_mask = torch.from_numpy(np_mask).to(self.device)

        # In your training loop or function
        if training:
            padding_mask = trainer.apply_exploration_masking(padding_mask)
                        
        state_tensor = trainer.normalize_states(state_tensor)
        action_tensor = trainer.get_action(state_tensor, padding_mask, training=training)
            
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
    def create_train_environments(env_config, gpt_seq_length, device):
        return EnvironmentPool(env_config, gpt_seq_length, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_eval_environments(env_config, gpt_seq_length, device, use_graphics):
        return EnvironmentPool(env_config, gpt_seq_length, device, test_env=True, use_graphics = use_graphics)

    @staticmethod
    def create_test_environments(env_config, gpt_seq_length, device, use_graphics):
        return EnvironmentPool(env_config, gpt_seq_length, device, test_env=True, use_graphics = use_graphics)
