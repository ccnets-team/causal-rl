'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
from .base_buffer import BaseBuffer
import numpy as np

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, seq_len):
        super().__init__("standard", capacity, state_size, action_size, seq_len)
        self.reset_buffer()

    def reset_buffer(self):
        self._reset_buffer()
    
    def reindex_indices(self, indices):
        ordered_valid_set = list(self.valid_set)
        
        actual_indices = [ordered_valid_set[idx] for idx in indices]
        
        return actual_indices
        
    def add_transition(self, state, action, reward, next_state, terminated, truncated):
        
        self._exclude_from_sampling(self.index)
        
        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated
        
        self._include_for_sampling(self.index, terminated or truncated)
        
        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer
        
    def sample_trajectories(self, indices, td_steps):
        actual_indices = self.reindex_indices(indices)
            
        # Retrieve trajectories for the given actual indices
        samples = self._fetch_trajectory_slices(actual_indices, td_steps)
            
        return samples