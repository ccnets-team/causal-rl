import gymnasium as gym
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from environments.settings.gym_rewards import get_ongoing_rewards_from_info, get_final_rewards_from_info, get_final_observations_from_info
from utils.structure.env_observations import EnvObservations

class GymEnvWrapper(AgentExperienceCollector):
    MAX_RANDOM_SEED = 1000  # class constant
    def __init__(self, env_config, test_env, use_graphics=False, seed=0):
        num_agents = 1 if test_env or use_graphics else env_config.num_agents
        super(GymEnvWrapper, self).__init__(num_agents, env_config)
        self.num_agents = num_agents
        self.use_discrete = env_config.use_discrete

        if not self.use_discrete:
            self.action_low = env_config.action_low
            self.action_high = env_config.action_high
        
        self.prev_value = np.zeros(self.num_agents)
        
        if use_graphics:
            self.time_scale = 1.5

        env_name = env_config.env_name
        self.env_name = env_name
        self.test_env = test_env
        self.seed = seed
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types

        self.agents = np.arange(self.num_agents)
        self.observations = EnvObservations(self.obs_shapes, self.obs_types, self.num_agents)
        self.next_observations = EnvObservations(self.obs_shapes, self.obs_types, self.num_agents)

        self.agent_dec = np.ones(self.num_agents, dtype=bool)
        self.agent_life = np.zeros(self.num_agents, dtype=bool)

        self.agent_reset = np.zeros(self.num_agents, dtype=bool)

        self.env = gym.make(env_name, render_mode='human') if use_graphics else gym.make_vec(env_name, num_envs=self.num_agents) 
        self.use_graphics = use_graphics
        self.reset_env()

    def convert_observation_spec(self, raw_observations, observations):
        offset = 0
        # We will create a dictionary to store slices of data for each observation type
        sliced_data = {}
        for key in observations.obs_types:
            shape = observations.data[key].shape[1:]  # Exclude num_agents dimension
            end_offset = offset + np.prod(shape)

            # Reshape slice to match observation shape
            if self.use_graphics:
                slice_data = raw_observations[offset:end_offset].reshape((-1, *shape))
                sliced_data[key] = np.array(slice_data, dtype=np.float32)
                # sliced_data[key] = np.array([slice_data], dtype=np.float32)
            else:
                slice_data = raw_observations[:, offset:end_offset].reshape((-1, *shape))
                sliced_data[key] = np.array(slice_data, dtype=np.float32)

            offset = end_offset

        # Assign sliced data to all agents in the observations
        all_agent_indices = list(range(observations.num_agents))
        observations[all_agent_indices] = sliced_data
        
    def reset_env(self):
        self.running_cnt = 0
        random_num = np.random.randint(0, self.MAX_RANDOM_SEED)
        obs, _ = self.env.reset(seed=random_num)
        self.convert_observation_spec(obs, self.observations)

    def step_environment(self):
        return False
    
    def update(self, action):
        self.running_cnt += 1
        action_input = self._get_action_input(action)

        next_obs, reward, terminated, truncated, info = self.env.step(action_input)
        ongoing_terminated = np.array(terminated, np.bool8)
        ongoing_truncated = np.array(truncated, np.bool8)
        ongoing_next_obs = np.array(next_obs, np.float32)
        ongoing_reward = np.array(reward, np.float32)

        done = np.logical_or(ongoing_terminated, ongoing_truncated)
        self.agent_life[~done] = True 
        self.agent_life[done] = False
        self.agent_reset[done] = True 

        if not self.test_env:
            self.update_for_training(done, ongoing_terminated, info, action, ongoing_next_obs)
        else:
            self.update_for_test(done, action, ongoing_next_obs, ongoing_reward)
        
        return False

    def update_for_training(self, done, ongoing_terminated, info, action, ongoing_next_obs):
        ongoing_immediate_reward, ongoing_future_reward = get_ongoing_rewards_from_info(info, self.num_agents)
        final_immediate_reward, final_future_reward = get_final_rewards_from_info(ongoing_terminated, info, self.num_agents)

        final_next_observation = np.zeros_like(ongoing_next_obs)
        final_next_observation = get_final_observations_from_info(info, final_next_observation)
        
        immediate_reward = np.where(done, final_immediate_reward, ongoing_immediate_reward)
        future_reward = np.where(done, final_future_reward, ongoing_future_reward)
        next_obs = np.where(done[:, np.newaxis], final_next_observation, ongoing_next_obs)
        self.convert_observation_spec(next_obs, self.next_observations)

        value_diff = future_reward - self.prev_value
        reward = immediate_reward + value_diff

        self.update_agent_data(self.agents, self.observations.to_vector(), action, reward, self.next_observations.to_vector(), done)
        self.prev_value = ongoing_future_reward.copy()
        self.convert_observation_spec(ongoing_next_obs, self.observations)

    def update_for_test(self, done, action, ongoing_next_obs, ongoing_reward):
        next_obs = ongoing_next_obs
        reward = ongoing_reward
        self.convert_observation_spec(next_obs, self.next_observations)
        if self.use_graphics:
            self.append_agent_transition(0, self.observations.to_vector(), action, reward, self.next_observations.to_vector(), done)
        else:
            self.append_agent_transition(0, self.observations.to_vector()[0], action[0], reward[0], self.next_observations.to_vector()[0], done[0])
        self.observations = self.next_observations.copy()

        if done.any():
            self.reset_env()

    def _get_action_input(self, action):
        if self.use_graphics:
            action_input = action[0]
        else:
            action_input = action[:]        
        if self.use_discrete:
            return np.argmax(action_input, axis=-1)
        action0 = np.tanh(action_input)
        return self.action_low + (action0 + 1.0) * 0.5 * (self.action_high - self.action_low)
