from collections import defaultdict
import numpy as np
import random
import torch
from utils.structure.trajectories import BatchTrajectory, MultiEnvTrajectories
from memory.standard_buffer import StandardBuffer
from memory.priority_buffer import PriorityBuffer
from numpy.random import choice

class ExperienceMemory:
    def __init__(self, env_config, training_params, algorithm_params, memory_params, compute_td_errors, device):
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.num_td_steps = algorithm_params.num_td_steps
        self.model_seq_length = algorithm_params.model_seq_length
        
        self.batch_size = training_params.batch_size
        self.buffer_type = memory_params.buffer_type
        self.priority_alpha = memory_params.priority_alpha
        self.gamma = algorithm_params.discount_factor
        self.compute_td_errors = compute_td_errors
        self.use_priority = False
        self.td_error_update_counter = 0

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
            return [[StandardBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.num_td_steps) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
        else:
            self.use_priority = True
            return [[PriorityBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.num_td_steps) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
            
    def __len__(self):
        return sum(len(buf) for env in self.multi_buffers for buf in env)

    def get_total_data_points(self):
        return sum(buf.size for env in self.multi_buffers for buf in env)

    def reset_memory(self):
        return [buf._reset_buffer() for env in self.multi_buffers for buf in env]

    def sample_trajectory_from_buffer(self, env_id, agent_id, indices, td_steps):
        return self.multi_buffers[env_id][agent_id]._fetch_trajectory_slices(indices, td_steps)

    def _create_batch_trajectory_components(self, samples):
        components = [np.stack([b[i] for b in samples], axis=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        return states, actions, rewards, next_states, dones

    def _get_env_agent_ids(self, buffer_id):
        # Retrieve environment and agent IDs from the global index
        env_id = buffer_id // self.num_agents
        agent_id = buffer_id % self.num_agents
        return env_id, agent_id
            
    def push_trajectory_data(self, multi_env_trajectories: MultiEnvTrajectories, exploration_rate):
        if multi_env_trajectories.env_ids is None:
            return

        buffer_candidates = defaultdict(list)
        for data in zip(multi_env_trajectories.env_ids, multi_env_trajectories.agent_ids,
                        multi_env_trajectories.states, multi_env_trajectories.actions,
                        multi_env_trajectories.rewards, multi_env_trajectories.next_states,
                        multi_env_trajectories.dones_terminated, multi_env_trajectories.dones_truncated):
            env_id, agent_id = data[:2]
            buffer = self.multi_buffers[env_id][agent_id]
            buffer_id = int(env_id * self.num_agents + agent_id)
            buffer_candidates[buffer_id].append(buffer.index)
            buffer.add_transition(*data[2:])
        
        self.td_error_update_counter += 1

        if not self.use_priority:
            return 
        
        if self.td_error_update_counter % self.model_seq_length != 0:
            return
        
        buffer_indices = defaultdict(list)
        samples = []
        for buffer_id, indices in buffer_candidates.items():
            env_id, agent_id = self._get_env_agent_ids(buffer_id)
            buffer = self.multi_buffers[env_id][agent_id]
            trajectory = buffer._fetch_trajectory_slices(indices, self.model_seq_length)
            samples.extend(trajectory)
        
        if samples is None or len(samples) == 0:
            return
        states, actions, rewards, next_states, dones = self._create_batch_trajectory_components(samples)
        batch_trajectory = BatchTrajectory(states, actions, rewards, next_states, dones, buffer_indices)
        self.compute_td_errors(batch_trajectory)
        self.update_td_errors(batch_trajectory)

    def sample_batch_trajectory(self, use_sampling_normalizer_update=False):
        samples, buffer_indices = self.sample_trajectory_data(use_sampling_normalizer_update)
        if samples is None:
            return None

        states, actions, rewards, next_states, dones = self._create_batch_trajectory_components(samples)
        return BatchTrajectory(states, actions, rewards, next_states, dones, buffer_indices)
    
    def update_td_errors(self, trajectory: BatchTrajectory):
        if not self.use_priority:
            return 
        
        td_errors = trajectory.td_errors.cpu().numpy()
        mask = trajectory.padding_mask.cpu().numpy()
        buffer_indices = trajectory.buffer_indices
        start_index = 0

        for buffer_id, indices in buffer_indices.items():
            # Map buffer_id back to env_id and agent_id
            env_id, agent_id = self._get_env_agent_ids(buffer_id)
            buffer = self.multi_buffers[env_id][agent_id]

            end_index = start_index + len(indices)
            # Extract corresponding local normalized rewards and values
            local_td_errors = td_errors[start_index:end_index]
            local_mask = mask[start_index:end_index]

            # Update TD errors for the buffer
            buffer.update_td_errors_for_sampled(indices, local_td_errors, local_mask)
            
            start_index = end_index  # Update start_index for next iteration

    def sample_trajectory_data(self, use_sampling_normalizer_update = True):
        sample_sz = self.batch_size
        td_steps = self.num_td_steps
            
        # Cumulative size calculation now a separate method for clarity
        cumulative_sizes, total_buffer_size = self._calculate_cumulative_sizes()
        if sample_sz > total_buffer_size:
            return None, None
        if use_sampling_normalizer_update or not self.use_priority:
            samples, buffer_indices = self.balanced_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        else:
            samples, buffer_indices = self.priority_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        return samples, buffer_indices  
    
    def balanced_sample_trajectory_data(self, sample_size, td_step, cumulative_sizes, total_buffer_size):
        sampled_indices = random.sample(range(total_buffer_size), sample_size)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)

            previous_cumulative_size = cumulative_sizes[buffer_id - 1] if buffer_id > 0 else 0
            local_idx = idx - previous_cumulative_size

            buffer_indices[buffer_id].append(local_idx)
        
        samples = self._fetch_samples(buffer_indices, td_step, use_actual_indices=False)
        return samples, buffer_indices

    def priority_sample_trajectory_data(self, sample_size, td_steps, cumulative_sizes, total_buffer_size):

        # Step 1: Calculate sampling probabilities based on TD errors
        sampling_probabilities = self._calculate_sampling_probabilities()

        # Step 2: Sample indices based on the calculated probabilities
        sampled_indices = choice(range(total_buffer_size), size=sample_size, p=sampling_probabilities)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)

            previous_cumulative_size = cumulative_sizes[buffer_id - 1] if buffer_id > 0 else 0
            local_idx = idx - previous_cumulative_size

            buffer_indices[buffer_id].append(local_idx)

        samples = self._fetch_samples(buffer_indices, td_steps, use_actual_indices=False)
        return samples, buffer_indices

    def _calculate_cumulative_sizes(self):
        cumulative_sizes = []
        total_buffer_size = 0
        for env_id in range(self.num_environments):
            for agent_id in range(self.num_agents):
                total_buffer_size += len(self.multi_buffers[env_id][agent_id])
                cumulative_sizes.append(total_buffer_size)
        return cumulative_sizes, total_buffer_size

    def _fetch_samples(self, buffer_indices, num_td_steps, use_actual_indices):
        samples = []
        for buffer_id, local_indices in buffer_indices.items():
            env_id, agent_id = self._get_env_agent_ids(buffer_id)
            # Fetch the experience using local indices
            buffer = self.multi_buffers[env_id][agent_id]
            batch = buffer._fetch_trajectory_slices(local_indices, num_td_steps)
            samples.extend(batch)

        return samples

    def _calculate_sampling_probabilities(self):
        alpha=self.priority_alpha
        # Accumulate TD errors using NumPy arrays for memory efficiency
        td_errors = np.concatenate([buf.td_errors[:buf.size] for env in self.multi_buffers for buf in env])

        # Adjust TD errors by alpha
        if alpha != 0:
            adjusted_td_errors = np.power(td_errors, alpha)
        else:
            # If alpha is 0, use uniform distribution
            adjusted_td_errors = np.ones_like(td_errors)

        probabilities = adjusted_td_errors / adjusted_td_errors.sum()
        return probabilities
