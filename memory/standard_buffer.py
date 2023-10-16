from memory.base_buffer import BaseBuffer 
import random
import numpy as np

from collections import deque

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)
        self.trajectories = deque() # queue to store (starting_idx, length) of each trajectory
        self.use_done = False

    def add(self, state, action, reward, next_state, done, td_error=None):
        # Before adding, check if the buffer is full
                
        # Add the new experience
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def __len__(self):
        return len(self.get_trajectory_indicies())
    
    def sample(self, sample_size):
        valid_indices = self.get_trajectory_indicies()
        
        indices = random.sample(valid_indices, sample_size)        
        samples = self.get_trajectories(indices, self.num_td_steps)
        assert(len(samples) == sample_size)
        return samples
