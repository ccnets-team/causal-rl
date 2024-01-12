'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np

class BaseBuffer:
    def __init__(self, buffer_type, capacity, state_size, action_size, num_td_steps):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.num_td_steps = num_td_steps
        self.state_size = state_size
        self.action_size = action_size
        self._reset_buffer()

    def _reset_buffer(self):
        self.size = 0  
        self.index = 0
        self.states = np.empty((self.capacity, self.state_size))
        self.actions = np.empty((self.capacity, self.action_size))
        self.rewards = np.empty(self.capacity)
        self.next_states = np.empty((self.capacity, self.state_size))
        self.terminated = np.empty(self.capacity)       
        self.truncated = np.empty(self.capacity)       
        self.td_errors = np.empty(self.capacity)  # Store TD errors for each transition

    def __len__(self):
        return max(self.size - self.num_td_steps + 1, 0)

    def _fetch_trajectory_slices(self, actual_indices, td_steps):
        batch_size = len(actual_indices)
        buffer_size = self.capacity
        # Expand indices for num_td_steps steps and wrap around using modulo operation
        expanded_indices = np.array([range(buffer_size + i -  td_steps + 1, buffer_size + i + 1) for i in actual_indices]) % buffer_size
        expanded_indices = expanded_indices.reshape(batch_size, td_steps)

        # Fetch the slices using advanced indexing
        states_slices = self.states[expanded_indices]
        actions_slices = self.actions[expanded_indices]
        rewards_slices = self.rewards[expanded_indices]
        next_states_slices = self.next_states[expanded_indices]
        terminated_slices = self.terminated[expanded_indices]
        truncated_slices = self.truncated[expanded_indices]

        if self.size < self.capacity:
            # Create a mask to identify valid indices within the current buffer size
            valid_mask = expanded_indices < self.size 
                    
            # Zero out the data at invalid indices
            states_slices[~valid_mask] = 0
            actions_slices[~valid_mask] = 0
            rewards_slices[~valid_mask] = 0
            next_states_slices[~valid_mask] = 0
            terminated_slices[~valid_mask] = True
            truncated_slices[~valid_mask] = True
        
        states_slices = states_slices.reshape(batch_size, td_steps, -1)
        actions_slices = actions_slices.reshape(batch_size, td_steps, -1)
        rewards_slices = rewards_slices.reshape(batch_size, td_steps, -1)
        next_states_slices = next_states_slices.reshape(batch_size, td_steps, -1)
        terminated_slices = terminated_slices.reshape(batch_size, td_steps, -1)
        truncated_slices = truncated_slices.reshape(batch_size, td_steps, -1)
        
        # Ensure the last element of truncated_slices is False to prevent truncation
        truncated_slices[:, -1] = False
        dones_slices = np.logical_or(terminated_slices, truncated_slices)
        
        transitions = list(zip(states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices))
        return transitions

    def _reindex_indices(self, indices):
        np_indices = np.array(indices, dtype=np.int32)

        # Calculate the start and end of the excluded range
        excluded_range_start = self.index
        excluded_range_end = (self.index + self.num_td_steps - 1) % self.capacity

        # Check if the range wraps around
        if excluded_range_end < excluded_range_start:
            # Identify indices that are either after the start or before the end of the range
            need_reindexing = (np_indices >= excluded_range_start) | (np_indices <= excluded_range_end)
        else:
            # Identify indices that are within the continuous range
            need_reindexing = (np_indices >= excluded_range_start) & (np_indices <= excluded_range_end)

        # Reindex these indices
        np_indices[need_reindexing] += (self.num_td_steps - 1)

        # Ensure indices are within bounds
        actual_indices = np_indices % self.capacity

        return actual_indices
                
    def sample_trajectories(self, indices, td_steps, use_actual_indices=False):
        if use_actual_indices:
            actual_indices = indices
        else:
            actual_indices = self._reindex_indices(indices)
            
        # Retrieve trajectories for the given actual indices
        samples = self._fetch_trajectory_slices(actual_indices, td_steps)
            
        return samples