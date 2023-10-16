from memory.base_buffer import BaseBuffer 
import random
import numpy as np

from collections import deque

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)
        self.trajectories = np.empty(shape=(capacity, 2), dtype=np.int32)
        self.start_idx = 0  # Index for the start of the trajectories
        self.end_idx = 0  # Index for the end of the trajectories
        self.use_done = False


    def add(self, state, action, reward, next_state, done, td_error=None):
        # Before adding, check if the buffer is full
                
        # Add the new experience
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        if self.start_idx == self.end_idx or self.use_done:
            self.trajectories[self.end_idx] = (self.index, 1)
            self.end_idx = (self.end_idx + 1) % self.capacity  
            self.use_done = False
        else:
            first_start_idx, first_length = self.trajectories[self.start_idx]
            if first_start_idx == self.index:
                self.trajectories[self.start_idx] = ((first_start_idx + 1) % self.capacity, first_length - 1)
                
            last_start_idx, last_length = self.trajectories[self.end_idx - 1]    
            self.trajectories[self.end_idx - 1] = (last_start_idx, last_length + 1)  # Reverted back to tuple assignment
        
        start_idx, length = self.trajectories[self.start_idx]
        if length <= 0:
            self.start_idx = (self.start_idx + 1) % self.capacity
        
        if done:
            self.use_done = True
        
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def __len__(self):
        # Determine the range of indices to include based on start_idx and end_idx
        if self.start_idx <= self.end_idx:
            # No wrap-around
            index_range = np.arange(self.start_idx, self.end_idx)
        else:
            # Wrap-around case
            index_range = np.concatenate((np.arange(self.start_idx, self.capacity), np.arange(0, self.end_idx)))

        # Now compute the total length using only the selected range of trajectories
        total_len = np.sum(np.maximum(self.trajectories[index_range, 1] - self.num_td_steps + 1, 0))
        return total_len                    
    
    def sample(self, sample_size):
        valid_indices = self.get_trajectory_indicies()
        
        indices = random.sample(valid_indices, sample_size)        
        samples = self.get_trajectories(indices, self.num_td_steps)
        assert(len(samples) == sample_size)
        return samples
