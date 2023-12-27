from collections import defaultdict
import numpy as np
import random
import torch
from utils.structure.trajectories import BatchTrajectory, MultiEnvTrajectories
from memory.standard_buffer import StandardBuffer
from memory.priority_buffer import PriorityBuffer
from numpy.random import choice

class ExperienceMemory:
    def __init__(self, env_config, training_params, algorithm_params, memory_params, reward_normalizer, device):
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.num_td_steps = algorithm_params.num_td_steps
        self.batch_size = training_params.batch_size
        self.buffer_type = memory_params.buffer_type
        self.gamma = algorithm_params.discount_factor
        self.reward_normalizer = reward_normalizer
        self.use_priority = False

        # Capacity calculation now in a separate method for clarity
        self.capacity_per_agent = self._calculate_capacity_per_agent(memory_params.buffer_size)

        # Buffer initialization now in a separate method for clarity
        self.multi_buffers = self._initialize_buffers()

    def _calculate_capacity_per_agent(self, buffer_size):
        # Capacity calculation logic separated for clarity
        return int(buffer_size) // (self.num_environments * self.num_agents) + self.num_td_steps

    def _initialize_buffers(self):
        # Buffer initialization logic separated for clarity
        if self.buffer_type == "standard": 
            self.use_priority = False
            return [[StandardBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.num_td_steps, self.gamma) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
        else:
            self.use_priority = True
            return [[PriorityBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.num_td_steps, self.gamma) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
            
    def __len__(self):
        return sum(len(buf) for env in self.multi_buffers for buf in env)

    def get_total_data_points(self):
        return sum(buf.size for env in self.multi_buffers for buf in env)

    def reset_memory(self):
        return [buf._reset_buffer() for env in self.multi_buffers for buf in env]

    def sample_trajectory_from_buffer(self, env_id, agent_id, indices, td_steps):
        return self.multi_buffers[env_id][agent_id].sample_trajectories(indices, td_steps)
    
    def push_trajectory_data(self, multi_env_trajectories: MultiEnvTrajectories):
        if multi_env_trajectories.env_ids is None:
            return
        
        normalized_rewards = self.reward_normalizer.normalize(torch.tensor(multi_env_trajectories.rewards).to(self.device))
        normalized_rewards = normalized_rewards.cpu().numpy()
        for data in zip(multi_env_trajectories.env_ids, multi_env_trajectories.agent_ids, 
                        multi_env_trajectories.states, multi_env_trajectories.actions, 
                        multi_env_trajectories.rewards, multi_env_trajectories.next_states, 
                        multi_env_trajectories.dones_terminated, multi_env_trajectories.dones_truncated, 
                        multi_env_trajectories.values, normalized_rewards):
            
            self.multi_buffers[data[0]][data[1]].add_transition(*data[2:])

    def sample_batch_trajectory(self, use_sampling_normalizer_update = False):
        samples = self.sample_trajectory_data(use_sampling_normalizer_update)
        if samples is None:
            return None

        # Simplify stack operations
        components = [np.stack([b[i] for b in samples], axis=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        
        return BatchTrajectory(states, actions, rewards, next_states, dones)

    def sample_trajectory_data(self, use_sampling_normalizer_update = True):
        sample_sz = self.batch_size
        td_steps = self.num_td_steps 

        # Cumulative size calculation now a separate method for clarity
        cumulative_sizes, total_buffer_size = self._calculate_cumulative_sizes()
        if sample_sz > total_buffer_size:
            return None
        if use_sampling_normalizer_update or not self.use_priority:
            samples = self.balanced_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        else:
            samples = self.priority_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        return samples
    
    def balanced_sample_trajectory_data(self, sample_size, sample_td_step, cumulative_sizes, total_buffer_size):
        sampled_indices = random.sample(range(total_buffer_size), sample_size)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)
            buffer_indices[buffer_id].append(idx)
        
        return self._fetch_samples(buffer_indices, cumulative_sizes, sample_td_step)

    def priority_sample_trajectory_data(self, sample_size, td_steps, cumulative_sizes, total_buffer_size):

        # Step 1: Calculate sampling probabilities based on TD errors
        sampling_probabilities = self._calculate_sampling_probabilities()

        # Step 2: Sample indices based on the calculated probabilities
        sampled_indices = choice(range(total_buffer_size), size=sample_size, p=sampling_probabilities)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)
            buffer_indices[buffer_id].append(idx)

        return self._fetch_samples(buffer_indices, cumulative_sizes, td_steps)

    def _calculate_cumulative_sizes(self):
        cumulative_sizes = []
        total_buffer_size = 0
        for env_id in range(self.num_environments):
            for agent_id in range(self.num_agents):
                total_buffer_size += len(self.multi_buffers[env_id][agent_id])
                cumulative_sizes.append(total_buffer_size)
        return cumulative_sizes, total_buffer_size

    def _fetch_samples(self, buffer_indices, cumulative_sizes, num_td_steps):
        samples = []
        for buffer_id, global_indices in buffer_indices.items():
            # Map buffer_id back to env_id and agent_id
            env_id = buffer_id // self.num_agents
            agent_id = buffer_id % self.num_agents

            # Calculate the starting index for the current buffer
            cumulative_start_index = cumulative_sizes[buffer_id - 1] if buffer_id > 0 else 0

            # Convert global indices to local indices for the specific buffer
            local_indices = [idx - cumulative_start_index for idx in global_indices]

            # Fetch the experience using local indices
            batch = self.sample_trajectory_from_buffer(env_id, agent_id, local_indices, num_td_steps)
            samples.extend(batch)

        return samples

    def _calculate_sampling_probabilities(self, alpha = 0.6):
        td_errors = []
        for env in self.multi_buffers:
            for buf in env:
                td_errors.extend(buf.td_errors[buf.valid_indices])

        td_errors = np.array(td_errors)

        # Adjust TD errors by alpha
        if alpha != 0:
            adjusted_td_errors = np.power(td_errors, alpha)
        else:
            # If alpha is 0, use uniform distribution
            adjusted_td_errors = np.ones_like(td_errors)

        probabilities = adjusted_td_errors / adjusted_td_errors.sum()
        return probabilities