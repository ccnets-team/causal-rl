from memory.base_buffer import BaseBuffer 
import random
import numpy as np

from collections import deque

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)
        self.trajectories = deque() # queue to store (starting_idx, length) of each trajectory

    def add(self, state, action, reward, next_state, done, td_error=None):
        # Before adding, check if the buffer is full
                
        # Add the new experience
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        if self.size < self.capacity:
            self.size += 1

        # Check if the experience being overwritten is the start of an old trajectory
        if not self.trajectories:
            self.trajectories.append((self.index, 1))  # Initialized with 0 length since it will start with the next experience
        elif self.trajectories[0][0] == self.index:
            start, next_length = self.trajectories[0]
            self.trajectories[0] = (start + 1, next_length - 1)
            if next_length - 1 <= 0:
                self.trajectories.popleft()  # remove old trajectory

        # Increase the length of the last trajectory by 1
        start, length = self.trajectories[-1]
        self.trajectories[-1] = (start, length + 1)

        # If this experience is a `done`, the next experience should start a new trajectory
        self.index = (self.index + 1) % self.capacity
        if done:
            self.trajectories.append((self.index, 0))  # Initialized with 0 length since it will start with the next experience
            
    def __len__(self):
        total_len = sum(max(length - self.num_td_steps + 1, 0) for _, length in self.trajectories)
        return total_len
    
    def sample(self, sample_size):
        valid_indices = self.get_trajectory_indicies()
        
        indices = random.sample(valid_indices, sample_size)        
        samples = self.get_trajectories(indices, self.num_td_steps)
        assert(len(samples) == sample_size)
        return samples
