import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.trajectory_handler  import MultiEnvTrajectories
import torch

class EnvGenerator: 
    def __init__(self, env_config, device, test_env, use_graphics, start_env_idx):
        super(EnvGenerator, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += (start_env_idx + 100)
        self.device = device

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

    def fetch_rewards(self):
        np_reward = []
        for env in self.env_list:
            env.step_environment()
        for env in self.env_list:
            reward = env.output_rewards()
            np_reward.extend(reward)
        return np_reward

    def fetch_env(self):
        combined_transition = MultiEnvTrajectories()

        for env_idx, env in enumerate(self.env_list):
            agent_ids, obs, action, reward, next_obs, done = env.output_transitions()
            combined_transition.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done)
        return combined_transition

    def step_env(self):
        for env in self.env_list:
            env.step_environment()

    def explore_env(self, trainer, training):
        trainer.set_train(training = training)
        # np_state = np.concatenate([env.states for env in self.env_list], axis=0)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_life = np.concatenate([env.agent_life for env in self.env_list], axis=0)

        state_tensor = torch.from_numpy(np_state).to(self.device)
        life_tensor = torch.from_numpy(np_life).to(self.device)

        state_tensor = trainer.normalize_state(state_tensor)
        action_tensor = trainer.get_action(state_tensor, training=training)

        trainer.reset_actor_noise(reset_noise=~life_tensor)
        np_action = action_tensor.cpu().numpy()

        start_idx = 0
        for env in self.env_list:
            end_idx = start_idx + len(env.agent_dec)
            valid_action = np_action[start_idx:end_idx][env.agent_dec]
            env.update(valid_action)
            start_idx = end_idx
    
    @staticmethod
    def create_train_environments(env_config, device, start_env_idx= 0):
        return EnvGenerator(env_config, device, test_env=False, use_graphics = False, start_env_idx = start_env_idx)
    
    @staticmethod
    def create_test_env(env_config, device, use_graphics, start_env_idx = 0):
        return EnvGenerator(env_config, device, test_env=True, use_graphics = use_graphics, start_env_idx = start_env_idx)

