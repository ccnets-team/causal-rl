import numpy as np

class AgentExperienceCollector:
    def __init__(self, env_config):
        super(AgentExperienceCollector, self).__init__()
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.num_agents = env_config.num_agents
        
        # Assuming observation shapes are uniform for simplification
        obs_size = env_config.state_size
        action_size = env_config.action_size  # Placeholder, define properly based on your env_config

        # Pre-allocate NumPy arrays for agent data
        self.agent_num = np.zeros(self.num_agents, dtype=int)
        self.agent_obs = np.zeros((self.num_agents, obs_size), dtype=np.float32)
        self.agent_action = np.zeros((self.num_agents, action_size), dtype=np.float32)
        self.agent_reward = np.zeros((self.num_agents), dtype=np.float32)
        self.agent_next_obs = np.zeros((self.num_agents, obs_size), dtype=np.float32)
        self.agent_done_terminated = np.zeros((self.num_agents), dtype=bool)
        self.agent_done_truncated = np.zeros((self.num_agents), dtype=bool)
        self.agent_padding_length = np.zeros((self.num_agents), dtype=int)
        self.agent_data_hold = np.zeros((self.num_agents), dtype=int)

    def init_observation(self, observations):
        struct_observations = {}
        for i, (obs, shape) in enumerate(zip(observations, self.obs_shapes)):
            if len(shape) == 1:
                struct_observations[f'vector_{i}'] = obs
            elif len(shape) == 3:
                struct_observations[f'image_{i}'] = obs
            else:
                raise ValueError(f"Unsupported observation shape: {shape}")
        return struct_observations

    def filter_data(self, target_ids, exclude_ids, reward, observations):
        mask = ~np.isin(target_ids, exclude_ids)
        return target_ids[mask], reward[mask], observations[mask]

    def filter_agents(self, agent_ids, next_agent_ids, reward, next_obs):
        next_agent_check = np.isin(next_agent_ids, agent_ids)
        return next_agent_ids[next_agent_check], reward[next_agent_check], next_obs[next_agent_check]

    def find_indices(self, agent_ids, selected_next_agent_ids):
        agent_ids_ls = agent_ids.tolist()
        return [agent_ids_ls.index(selected_id) for selected_id in selected_next_agent_ids]

    def select_data(self, agent_ids, selected_agent_indices, obs, action, reward, terminated, trucated, padding_length):
        selected_agent_ids = agent_ids[selected_agent_indices]
        selected_obs = obs[selected_agent_indices]
        action = action[selected_agent_indices]
        padding_length = padding_length[selected_agent_indices]
        terminated = np.ones_like(reward, dtype=np.bool8) if terminated else np.zeros_like(reward, dtype=np.bool8)
        trucated = np.ones_like(reward, dtype=np.bool8) if trucated else np.zeros_like(reward, dtype=np.bool8)
        return selected_agent_ids, selected_obs, action, reward, terminated, trucated, padding_length
        
    def select_observations(self, struct_observations):
        vector_obervations = []
        for idx, (key, observation) in enumerate(struct_observations.items()):
            if 'vector' in key:
                vector_obervations.append(observation)
            elif 'image' in key:
                raise ValueError(f"Unsupported image observations: {key}")
            else:
                raise ValueError(f"Unsupported key in observations: {key}")
        cat_obervation = np.concatenate(vector_obervations, axis=1)
        return cat_obervation

    def append_transitions(self, agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, padding_length):
        # Ensure all inputs are NumPy arrays and properly reshaped or flattened as needed
        # This example assumes obs and next_obs need flattening to match the pre-allocated array shapes
        obs_flat = obs.reshape(len(agent_ids), -1)  # Reshape obs to ensure it fits into the agent_obs array
        next_obs_flat = next_obs.reshape(len(agent_ids), -1)  # Same for next_obs
        # Directly update the arrays for each agent using advanced indexing
        self.agent_obs[agent_ids] = obs_flat
        self.agent_action[agent_ids] = action  # Assuming action is already correctly shaped
        self.agent_reward[agent_ids] = reward
        self.agent_next_obs[agent_ids] = next_obs_flat
        self.agent_done_terminated[agent_ids] = done_terminated
        self.agent_done_truncated[agent_ids] = done_truncated
        self.agent_padding_length[agent_ids] = padding_length
        self.agent_data_hold[agent_ids] = True

    def push_transitions(self, agent_ids, obs, action, next_agent_ids, reward, next_obs, done_terminated, done_truncated, padding_length):
        if len(next_agent_ids) == 0: return
        if done_truncated is None:
            done_truncated = [False] * len(agent_ids)
        
        selected_next_agent_ids, reward, next_obs = self.filter_agents(agent_ids, next_agent_ids, reward, next_obs)
        
        selected_agent_indices = self.find_indices(agent_ids, selected_next_agent_ids)
        selected_agent_ids, obs, action, reward, done_terminated, done_truncated, padding_length = self.select_data(agent_ids, selected_agent_indices, obs, action, reward, done_terminated, done_truncated, padding_length)
        self.append_transitions(selected_agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, padding_length)

    def add_transition(self, agent_id, obs, action, reward, next_obs, done_terminated, done_truncated, padding_length):
        self.agent_obs[agent_id] = obs
        self.agent_action[agent_id] = action
        self.agent_reward[agent_id] = reward
        self.agent_next_obs[agent_id] = next_obs
        self.agent_done_terminated[agent_id] = done_terminated
        self.agent_done_truncated[agent_id] = done_truncated
        self.agent_padding_length[agent_id] = padding_length
        self.agent_data_hold[agent_id] = True

    def output_transitions(self):
        # select self.agent_data_hold positive boolean indices
        agent_ids = np.where(self.agent_data_hold)[0]
        self.agent_data_hold.fill(False)
        return agent_ids, self.agent_obs[agent_ids], self.agent_action[agent_ids], self.agent_reward[agent_ids], \
            self.agent_next_obs[agent_ids], self.agent_done_terminated[agent_ids], self.agent_done_truncated[agent_ids], self.agent_padding_length[agent_ids]
