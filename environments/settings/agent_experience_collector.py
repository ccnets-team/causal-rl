import numpy as np

class AgentExperienceCollector:
    def __init__(self, num_agents, env_config):
        super(AgentExperienceCollector, self).__init__()
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.num_agents = num_agents
        self.agent_num = np.zeros((self.num_agents), dtype=int)
        self.agent_obs = [[] for i in range(self.num_agents)]
        self.agent_action = [[] for i in range(self.num_agents)]
        self.agent_reward = [[] for i in range(self.num_agents)]
        self.agent_next_obs = [[] for i in range(self.num_agents)]
        self.agent_done_terminated = [[] for i in range(self.num_agents)]
        self.agent_done_truncated = [[] for i in range(self.num_agents)]

    def reset_agents(self):
        for i in range(self.num_agents):
            self.agent_obs[i].clear()
            self.agent_action[i].clear()
            self.agent_reward[i].clear()
            self.agent_next_obs[i].clear()
            self.agent_done_terminated[i].clear()  
            self.agent_done_truncated[i].clear()  
        self.agent_num.fill(int(0))
        
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

    def filter_data(self, target_ids, exclude_ids, reward, observations):
        mask = ~np.isin(target_ids, exclude_ids)
        return target_ids[mask], reward[mask], observations[mask]

    def filter_agents(self, agent_ids, next_agent_ids, reward, next_obs):
        next_agent_check = np.isin(next_agent_ids, agent_ids)
        return next_agent_ids[next_agent_check], reward[next_agent_check], next_obs[next_agent_check]

    def find_indices(self, agent_ids, selected_next_agent_ids):
        agent_ids_ls = agent_ids.tolist()
        return [agent_ids_ls.index(selected_id) for selected_id in selected_next_agent_ids]

    def select_data(self, agent_ids, selected_agent_indices, obs, action, reward, term):
        selected_agent_ids = agent_ids[selected_agent_indices]
        selected_obs = obs[selected_agent_indices]
        action = action[selected_agent_indices]
        done = np.ones_like(reward, dtype=np.bool8) if term else np.zeros_like(reward, dtype=np.bool8)

        return selected_agent_ids, selected_obs, action, reward, done

    def append_agent_transition(self, agent_id, obs, action, reward, next_obs, done_terminated, done_truncated):
        self.agent_obs[agent_id].append(obs)
        self.agent_action[agent_id].append(action)
        self.agent_reward[agent_id].append(reward)
        self.agent_next_obs[agent_id].append(next_obs)
        self.agent_done_terminated[agent_id].append(done_terminated)
        self.agent_done_truncated[agent_id].append(done_truncated)
        self.agent_num[agent_id] += 1

    def update_agent_data(self, agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated = None):
        for agent_idx, agent_id in enumerate(agent_ids):
            truncated = False  if done_truncated is None else done_truncated[agent_idx]
            self.append_agent_transition(agent_id, obs[agent_idx], action[agent_idx], reward[agent_idx], next_obs[agent_idx], done_terminated[agent_idx], truncated)

    def push_transitions(self, agent_ids, obs, action, next_agent_ids, reward, next_obs, term=False):
        if len(next_agent_ids) == 0: return
        
        selected_next_agent_ids, reward, next_obs = self.filter_agents(agent_ids, next_agent_ids, reward, next_obs)
        
        selected_agent_indices = self.find_indices(agent_ids, selected_next_agent_ids)
        selected_agent_ids, obs, action, reward, done = self.select_data(agent_ids, selected_agent_indices, obs, action, reward, term)
        self.update_agent_data(selected_agent_ids, obs, action, reward, next_obs, done)

    def output_transitions(self):
        np_obs = [item for sublist in self.agent_obs[:self.num_agents] for item in sublist]
        np_action = [item for sublist in self.agent_action[:self.num_agents] for item in sublist]
        np_reward = [item for sublist in self.agent_reward[:self.num_agents] for item in sublist]
        np_next_obs = [item for sublist in self.agent_next_obs[:self.num_agents] for item in sublist]
        np_done_terminated = [item for sublist in self.agent_done_terminated[:self.num_agents] for item in sublist]
        np_done_truncated = [item for sublist in self.agent_done_truncated[:self.num_agents] for item in sublist]
        np_agent_id = [i for i in range(self.num_agents) for _ in range(len(self.agent_obs[i]))]
    
        self.reset_agents()
        return np_agent_id, np_obs, np_action, np_reward, np_next_obs, np_done_terminated, np_done_truncated
