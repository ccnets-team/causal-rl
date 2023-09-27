from memory.base_buffer import BaseBuffer 
import random
import numpy as np

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)

    def add(self, state, action, reward, next_state, done, td_error = None):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity
        
        if self.size < self.capacity:
            self.size += 1
            
    def sample(self, sample_size):
        valid_indices = self.get_trajectory_indicies()
        indices = random.sample(valid_indices, sample_size)
        samples = self.get_trajectories(indices, self.num_td_steps)
        assert(len(samples) == sample_size)
        return samples
            