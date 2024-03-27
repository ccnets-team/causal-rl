import numpy as np
import torch

class AgentExperienceCollector:
    def __init__(self, env_config, device):
        super(AgentExperienceCollector, self).__init__()
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.num_agents = env_config.num_agents
        self.device = device
        
        # Assuming observation shapes are uniform for simplification
        state_size = env_config.state_size
        action_size = env_config.action_size  # Placeholder, define properly based on your env_config

        # Pre-allocate NumPy arrays for agent data
        self.agent_num = torch.zeros(self.num_agents, dtype=torch.int, device=device)
        self.agent_obs = torch.zeros((self.num_agents, state_size), dtype=torch.float32, device=device)
        self.agent_action = torch.zeros((self.num_agents, action_size), dtype=torch.float32, device=device)
        self.agent_reward = torch.zeros((self.num_agents), dtype=torch.float32, device=device)
        self.agent_next_obs = torch.zeros((self.num_agents, state_size), dtype=torch.float32, device=device)
        self.agent_done_terminated = torch.zeros((self.num_agents), dtype=torch.bool, device=device)
        self.agent_done_truncated = torch.zeros((self.num_agents), dtype=torch.bool, device=device)
        self.agent_content_lengths = torch.zeros((self.num_agents), dtype=torch.int, device=device)
        self.agent_data_check = torch.zeros((self.num_agents), dtype=torch.bool, device=device)

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
        vector_observations = []
        for key, observation in struct_observations.items():
            if 'vector' in key:
                # Ensure observation is a PyTorch tensor
                if not isinstance(observation, torch.Tensor):
                    # Convert to tensor. Specify the device and dtype as needed.
                    observation = torch.tensor(observation, device=self.device)
                vector_observations.append(observation)
            elif 'image' in key:
                # Handle differently or raise an error based on your requirements
                raise ValueError(f"Unsupported image observations: {key}")
        if vector_observations:
            # Concatenate along the specified dimension if vector observations exist
            cat_observation = torch.cat(vector_observations, dim=1)
        else:
            # Handle the case where there are no vector observations
            cat_observation = torch.tensor([], device=self.device)
        return cat_observation
    
    def to_torch(self, data, dtype=None):
        """
        Checks if the data is a PyTorch tensor, and converts it to one if not.
        Sets the tensor to the specified device.
        
        Parameters:
        - data: The input data to be converted.
        - dtype: The desired data type of the tensor.
        
        Returns:
        - A PyTorch tensor of the specified data type on the correct device.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device, dtype=dtype if dtype else torch.float)
        else:
            data = data.to(self.device)
        return data
    
    def filter_data(self, target_ids, exclude_ids, reward, observations):
        # Create a mask for target_ids not in exclude_ids
        if isinstance(target_ids, torch.Tensor):
            mask = ~torch.isin(target_ids, exclude_ids)
        else:
            mask = ~np.isin(target_ids, exclude_ids)
        return target_ids[mask], reward[mask], observations[mask]

    def filter_agents(self, agent_ids, next_agent_ids, reward, next_obs):
        if isinstance(next_agent_ids, torch.Tensor):
            next_agent_check = torch.isin(next_agent_ids, agent_ids)
        else:
            next_agent_check = np.isin(next_agent_ids, agent_ids)
        return next_agent_ids[next_agent_check], reward[next_agent_check], next_obs[next_agent_check]

    def find_indices(self, agent_ids, selected_next_agent_ids):
        agent_ids_ls = agent_ids.tolist()
        return [agent_ids_ls.index(selected_id) for selected_id in selected_next_agent_ids]

    def select_data(self, agent_ids, selected_agent_indices, obs, action, reward, terminated, truncated, content_lengths):
        selected_agent_ids = agent_ids[selected_agent_indices]
        selected_obs = obs[selected_agent_indices]
        action = action[selected_agent_indices]
        content_lengths = content_lengths[selected_agent_indices]
        
        if isinstance(reward, torch.Tensor):
            # Handling booleans in PyTorch
            terminated = torch.ones_like(reward, dtype=torch.bool) if terminated else torch.zeros_like(reward, dtype=torch.bool)
            truncated = torch.ones_like(reward, dtype=torch.bool) if truncated else torch.zeros_like(reward, dtype=torch.bool)
        else:
            terminated = np.ones_like(reward, dtype=bool) if terminated else np.zeros_like(reward, dtype=bool)
            truncated = np.ones_like(reward, dtype=bool) if truncated else np.zeros_like(reward, dtype=bool)
            
        return selected_agent_ids, selected_obs, action, reward, terminated, truncated, content_lengths

    def append_transitions(self, agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, content_lengths):
        self.agent_obs[agent_ids] = self.to_torch(obs, dtype=torch.float).detach()
        self.agent_action[agent_ids] = self.to_torch(action, dtype=torch.float)
        self.agent_reward[agent_ids] = self.to_torch(reward, dtype=torch.float)
        self.agent_next_obs[agent_ids] = self.to_torch(next_obs.reshape(len(agent_ids), -1), dtype=torch.float)
        self.agent_done_terminated[agent_ids] = self.to_torch(done_terminated, dtype=torch.bool)
        self.agent_done_truncated[agent_ids] = self.to_torch(done_truncated, dtype=torch.bool)
        self.agent_content_lengths[agent_ids] = self.to_torch(content_lengths, dtype=torch.int)
        self.agent_data_check[agent_ids] = True

    def push_transitions(self, agent_ids, obs, action, next_agent_ids, reward, next_obs, done_terminated, done_truncated, content_lengths):
        if len(next_agent_ids) == 0: return
        if done_truncated is None:
            done_truncated = [False] * len(agent_ids)

        # Assume filter_agents and find_indices are other methods that handle specific logic
        selected_next_agent_ids, reward, next_obs = self.filter_agents(agent_ids, next_agent_ids, reward, next_obs)
        selected_agent_indices = self.find_indices(agent_ids, selected_next_agent_ids)
        selected_agent_ids, obs, action, reward, done_terminated, done_truncated, content_lengths = self.select_data(agent_ids, selected_agent_indices, obs, action, reward, done_terminated, done_truncated, content_lengths)
        
        self.append_transitions(selected_agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated, content_lengths)
                
    def add_transition(self, agent_id, obs_tensor, action, reward, next_obs, done_terminated, done_truncated, content_lengths):
        self.agent_obs[agent_id] = obs_tensor.to(self.device)
        self.agent_action[agent_id] = torch.tensor(action, dtype = torch.float, device = self.device)
        self.agent_reward[agent_id] = torch.tensor(reward, dtype = torch.float, device = self.device)
        self.agent_next_obs[agent_id] = torch.tensor(next_obs, dtype = torch.float, device = self.device)
        self.agent_done_terminated[agent_id] = torch.tensor(done_terminated, dtype = torch.float, device = self.device)
        self.agent_done_truncated[agent_id] = torch.tensor(done_truncated, dtype = torch.float, device = self.device)
        self.agent_content_lengths[agent_id] = torch.tensor(content_lengths, dtype = torch.float, device = self.device)
        self.agent_data_check[agent_id] = True

    def output_transitions(self):
        # Select self.agent_data_check positive boolean indices
        agent_ids = torch.nonzero(self.agent_data_check, as_tuple=False).squeeze(-1)
        self.agent_data_check.fill_(False)  # Reset the data check tensor to False
        return agent_ids, self.agent_obs[agent_ids], self.agent_action[agent_ids], self.agent_reward[agent_ids], \
            self.agent_next_obs[agent_ids], self.agent_done_terminated[agent_ids], self.agent_done_truncated[agent_ids], self.agent_content_lengths[agent_ids]
