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
    
    def add_transition(self, state, action, reward, next_state, terminated, truncated, padding_length):        
        self._reset_sample_prob(self.index)
        
        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated
        self.padding_length[self.index] = padding_length
        
        self._assign_sample_prob(self.index)
        
        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer
        
    def sample_trajectories(self, indices, td_steps):
        # Retrieve trajectories for the given actual indices
        samples = self._fetch_trajectory_slices(indices, td_steps)
        return samples