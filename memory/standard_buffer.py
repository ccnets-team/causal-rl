'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
from .base_buffer import BaseBuffer
import numpy as np

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)

    def __len__(self):
        return max(self.size - self.num_td_steps + 1, 0)

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
        
    def add_transition(self, state, action, reward, next_state, terminated, truncated):

        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated
        
        self._exclude_from_sampling(self.index)

        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer
        
    def sample_trajectories(self, indices, td_steps):
        actual_indices = self._reindex_indices(indices)
            
        # Retrieve trajectories for the given actual indices
        samples = self._fetch_trajectory_slices(actual_indices, td_steps)
            
        return samples