'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np

class BaseBuffer:
    def __init__(self, buffer_type, capacity, state_size, action_size, train_seq_length):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.train_seq_length = train_seq_length
        self.state_size = state_size
        self.action_size = action_size

    def _reset_buffer(self):
        self.size = 0  
        self.index = 0
        self.states = np.empty((self.capacity, self.state_size))
        self.actions = np.empty((self.capacity, self.action_size))
        self.rewards = np.empty(self.capacity)
        self.next_states = np.empty((self.capacity, self.state_size))
        self.terminated = np.empty(self.capacity)       
        self.truncated = np.empty(self.capacity)       

    def _fetch_trajectory_slices(self, indices, td_steps):
        batch_size = len(indices)
        buffer_size = self.capacity
        # Expand indices for train_seq_length steps and wrap around using modulo operation
        expanded_indices = np.array([range(buffer_size + i -  td_steps + 1, buffer_size + i + 1) for i in indices]) % buffer_size
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
