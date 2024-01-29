from collections import defaultdict
import numpy as np
import random
import torch
from utils.structure.trajectories import BatchTrajectory, MultiEnvTrajectories
from memory.standard_buffer import StandardBuffer

class ExperienceMemory:
    def __init__(self, env_config, training_params, algorithm_params, device):
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.gpt_seq_length = algorithm_params.gpt_seq_length
        
        self.batch_size = training_params.batch_size

        # Capacity calculation now in a separate method for clarity
        self.capacity_per_agent = self._calculate_capacity_per_agent(training_params.buffer_size)

        # Buffer initialization now in a separate method for clarity
        self.multi_buffers = self._initialize_buffers()

    def _calculate_capacity_per_agent(self, buffer_size):
        # Capacity calculation logic separated for clarity
        return int(buffer_size) // (self.num_environments * self.num_agents) + self.gpt_seq_length

    def _initialize_buffers(self):
        # Buffer initialization logic separated for clarity
        return [[StandardBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.gpt_seq_length) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
            
    def __len__(self):
        return sum(len(buf) for env in self.multi_buffers for buf in env)

    def get_total_data_points(self):
        return sum(buf.size for env in self.multi_buffers for buf in env)

    def reset_memory(self):
        return [buf.reset_buffer() for env in self.multi_buffers for buf in env]

    def sample_trajectory_from_buffer(self, env_id, agent_id, indices, td_steps):
        return self.multi_buffers[env_id][agent_id].sample_trajectories(indices, td_steps)

    def _create_batch_trajectory_components(self, samples):
        components = [np.stack([b[i] for b in samples], axis=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        return states, actions, rewards, next_states, dones

    def _get_env_agent_ids(self, buffer_id):
        # Retrieve environment and agent IDs from the global index
        env_id = buffer_id // self.num_agents
        agent_id = buffer_id % self.num_agents
        return env_id, agent_id
            
    def push_trajectory_data(self, multi_env_trajectories: MultiEnvTrajectories):
        if multi_env_trajectories.env_ids is None:
            return

        for data in zip(multi_env_trajectories.env_ids, multi_env_trajectories.agent_ids,
                        multi_env_trajectories.states, multi_env_trajectories.actions,
                        multi_env_trajectories.rewards, multi_env_trajectories.next_states,
                        multi_env_trajectories.dones_terminated, multi_env_trajectories.dones_truncated):
            env_id, agent_id = data[:2]
            buffer = self.multi_buffers[env_id][agent_id]
            buffer.add_transition(*data[2:])
        
    def sample_batch_trajectory(self):
        samples, buffer_indices = self.sample_trajectory_data()
        if samples is None:
            return None

        states, actions, rewards, next_states, dones = self._create_batch_trajectory_components(samples)
        return BatchTrajectory(states, actions, rewards, next_states, dones)
    
    def sample_trajectory_data(self):
        sample_sz = self.batch_size
        td_steps = self.gpt_seq_length 
        
        # Cumulative size calculation now a separate method for clarity
        cumulative_sizes, total_buffer_size = self._calculate_cumulative_sizes()
        if sample_sz > total_buffer_size:
            return None, None
        samples, buffer_indices = self.balanced_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        return samples, buffer_indices  
    
    def balanced_sample_trajectory_data(self, sample_size, td_steps, cumulative_sizes, total_buffer_size):
        sampled_indices = random.sample(range(total_buffer_size), sample_size)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)

            previous_cumulative_size = cumulative_sizes[buffer_id - 1] if buffer_id > 0 else 0
            local_idx = idx - previous_cumulative_size

            buffer_indices[buffer_id].append(local_idx)
        
        samples = self._fetch_samples(buffer_indices, td_steps)
        return samples, buffer_indices

    def _calculate_cumulative_sizes(self):
        cumulative_sizes = []
        total_buffer_size = 0
        for env_id in range(self.num_environments):
            for agent_id in range(self.num_agents):
                total_buffer_size += len(self.multi_buffers[env_id][agent_id])
                cumulative_sizes.append(total_buffer_size)
        return cumulative_sizes, total_buffer_size

    def _fetch_samples(self, buffer_indices, td_steps):
        samples = []
        for buffer_id, local_indices in buffer_indices.items():
            env_id, agent_id = self._get_env_agent_ids(buffer_id)
            # Fetch the experience using local indices
            batch = self.sample_trajectory_from_buffer(env_id, agent_id, local_indices, td_steps)
            samples.extend(batch)

        return samples
