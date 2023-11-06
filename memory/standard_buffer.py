'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np
import random

class BaseBuffer:
    def __init__(self, buffer_type, capacity, state_size, action_size, num_td_steps):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.num_td_steps = num_td_steps
        self.state_size = state_size
        self.action_size = action_size
        # Initialize the buffer as ndarrays with given shapes
        self._reset()

    def _reset(self):
        self.size = 0  
        self.index = 0
        self.states = np.empty((self.capacity, self.state_size))
        self.actions = np.empty((self.capacity, self.action_size))
        self.rewards = np.empty(self.capacity)
        self.next_states = np.empty((self.capacity, self.state_size))
        self.dones = np.empty(self.capacity)        

    def __len__(self):
        return max(self.size - self.num_td_steps + 1, 0)

    def get_trajectories(self, indices, td_steps):
        batch_size = len(indices)
        
        # Expand indices for num_td_steps steps and wrap around using modulo operation
        expanded_indices = np.array([range(i, i + td_steps) for i in indices]) % self.capacity
        expanded_indices = expanded_indices.reshape(batch_size, td_steps)
        
        # Fetch the slices
        states_slices = self.states[expanded_indices].reshape(batch_size, td_steps, -1)
        actions_slices = self.actions[expanded_indices].reshape(batch_size, td_steps, -1)
        rewards_slices = self.rewards[expanded_indices].reshape(batch_size, td_steps, -1)
        next_states_slices = self.next_states[expanded_indices].reshape(batch_size, td_steps, -1)
        dones_slices = self.dones[expanded_indices].reshape(batch_size, td_steps, -1)
        
        transitions = list(zip(states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices))
        return transitions
        
    def get_trajectory_indicies(self, td_steps):
        buffer_len = self.size
        if buffer_len == self.capacity:  # Buffer is full
            end_valid_idx = self.index - td_steps
            if end_valid_idx < 0:  # Check for wrap around
                possible_indices = np.arange(end_valid_idx + td_steps, end_valid_idx + buffer_len + 1) % self.capacity
            else:
                first_range = np.arange(end_valid_idx + td_steps, buffer_len)
                second_range = np.arange(0, end_valid_idx + 1)
                possible_indices = np.concatenate([first_range, second_range]) % self.capacity
        else:
            possible_indices = np.arange(0, self.index - td_steps + 1)

        return possible_indices.tolist()
    
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
            
    def sample(self, sample_size, td_steps):
        valid_indices = self.get_trajectory_indicies(td_steps)
        
        indices = random.sample(valid_indices, sample_size)        
        samples = self.get_trajectories(indices, td_steps)
        assert(len(samples) == sample_size)
        return samples
    