from collections import defaultdict
import numpy as np
import random
import torch
from utils.structure.trajectories import BatchTrajectory, MultiEnvTrajectories
from memory.standard_buffer import StandardBuffer
from memory.priority_buffer import PriorityBuffer
from numpy.random import choice

def create_padding_mask_before_dones(dones: torch.Tensor) -> torch.Tensor:
    """
    Creates a padding mask for a trajectory by sampling from the end of the sequence. The mask is set to 0 
    (masked) for elements occurring before the 'done' signal when viewed from the end of the trajectory. 
    This includes padding elements that are positioned on the left side of the first 'done' signal in the 
    reversed sequence. The elements from the 'done' signal to the end of the trajectory (rightmost end) 
    are unmasked (set to 1).

    This function is useful for trajectories where sampling starts from the end and padding occurs before 
    the 'done' signal in the reversed order.

    Args:
    - dones (torch.Tensor): The tensor representing the 'done' signals in the trajectory.

    Returns:
    - mask (torch.Tensor): The resultant padding mask tensor. In this mask, elements occurring before the 
      'done' signal in the reversed sequence are masked (set to 0), while the elements from the 'done' 
      signal to the end of the trajectory are unmasked (set to 1).
    """
    mask = torch.ones_like(dones)

    if mask.size(1) > 1:
        # Reverse 'dones' along the specified axis (axis=1)
        reversed_dones = torch.flip(dones, dims=[1])

        # Perform cumulative sum on the reversed tensor
        cumulative_dones_reversed = torch.cumsum(reversed_dones[:,1:], dim=1)

        # Reverse the result back to get the cumulative sum in the original order
        cumulative_dones = torch.flip(cumulative_dones_reversed, dims=[1])
        
        mask[:, :-1, :] = 1 - cumulative_dones
    
    return mask

class ExperienceMemory:
    def __init__(self, env_config, training_params, algorithm_params, memory_params, reward_normalizer, value_function, device):
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.num_td_steps = algorithm_params.num_td_steps
        self.batch_size = training_params.batch_size
        self.buffer_type = memory_params.buffer_type
        self.priority_alpha = memory_params.priority_alpha
        self.gamma = algorithm_params.discount_factor
        self.value_function = value_function
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
        samples, buffer_indices, cumulative_sizes = self.sample_trajectory_data(use_sampling_normalizer_update)
        if samples is None:
            return None

        # Simplify stack operations
        components = [np.stack([b[i] for b in samples], axis=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        batch_trajectory = BatchTrajectory(states, actions, rewards, next_states, dones)

        if self.use_priority:
            mask = create_padding_mask_before_dones(dones)
            # Recompute normalized rewards and values
            normalized_rewards = self.reward_normalizer.normalize(rewards).cpu().numpy()
            values = self.value_function(states, mask = mask).detach().cpu().numpy()
            mask = mask.cpu().numpy()
            # Update TD errors for sampled trajectories
            self.update_td_errors(buffer_indices, cumulative_sizes, normalized_rewards, values, mask)
        
        return batch_trajectory

    def update_td_errors(self, buffer_indices, cumulative_sizes, normalized_rewards, values, mask):
        # Iterate over buffers and update TD errors
        for buffer_id, _ in buffer_indices.items():
            # Map buffer_id back to env_id and agent_id
            env_id = buffer_id // self.num_agents
            agent_id = buffer_id % self.num_agents
            buffer = self.multi_buffers[env_id][agent_id]

            # Determine the start and end indices for the current buffer in the global array
            start_index = cumulative_sizes[buffer_id - 1] if buffer_id > 0 else 0
            end_index = cumulative_sizes[buffer_id]

            # Extract corresponding local normalized rewards and values
            local_normalized_rewards = normalized_rewards[start_index:end_index]
            local_values = values[start_index:end_index]
            local_mask = mask[start_index:end_index]

            # Determine the local indices within the buffer
            local_indices = list(range(start_index, end_index))

            # Update TD errors for the buffer
            buffer.update_td_errors_for_sampled(local_indices, local_normalized_rewards, local_values, local_mask)

    def sample_trajectory_data(self, use_sampling_normalizer_update = True):
        sample_sz = self.batch_size
        td_steps = self.num_td_steps 

        # Cumulative size calculation now a separate method for clarity
        cumulative_sizes, total_buffer_size = self._calculate_cumulative_sizes()
        if sample_sz > total_buffer_size:
            return None, None, None
        if use_sampling_normalizer_update or not self.use_priority:
            samples, buffer_indices = self.balanced_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        else:
            samples, buffer_indices = self.priority_sample_trajectory_data(sample_sz, td_steps, cumulative_sizes, total_buffer_size)
        return samples, buffer_indices, cumulative_sizes  
    
    def balanced_sample_trajectory_data(self, sample_size, sample_td_step, cumulative_sizes, total_buffer_size):
        sampled_indices = random.sample(range(total_buffer_size), sample_size)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)
            buffer_indices[buffer_id].append(idx)
        
        samples = self._fetch_samples(buffer_indices, cumulative_sizes, sample_td_step)
        return samples, buffer_indices

    def priority_sample_trajectory_data(self, sample_size, td_steps, cumulative_sizes, total_buffer_size):

        # Step 1: Calculate sampling probabilities based on TD errors
        sampling_probabilities = self._calculate_sampling_probabilities()

        # Step 2: Sample indices based on the calculated probabilities
        sampled_indices = choice(range(total_buffer_size), size=sample_size, p=sampling_probabilities)
        buffer_indices = defaultdict(list)
        for idx in sampled_indices:
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)
            buffer_indices[buffer_id].append(idx)

        samples = self._fetch_samples(buffer_indices, cumulative_sizes, td_steps)
        return samples, buffer_indices

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

    def _calculate_sampling_probabilities(self):
        alpha=self.priority_alpha
        # Accumulate TD errors using NumPy arrays for memory efficiency
        td_errors = np.concatenate([buf.td_errors[buf.valid_indices] for env in self.multi_buffers for buf in env])

        # Adjust TD errors by alpha
        if alpha != 0:
            adjusted_td_errors = np.power(td_errors, alpha)
        else:
            # If alpha is 0, use uniform distribution
            adjusted_td_errors = np.ones_like(td_errors)

        probabilities = adjusted_td_errors / adjusted_td_errors.sum()
        return probabilities
