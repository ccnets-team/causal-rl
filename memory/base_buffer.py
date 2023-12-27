'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np

class BaseBuffer:
    def __init__(self, buffer_type, capacity, state_size, action_size, num_td_steps, gamma):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.num_td_steps = num_td_steps
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.valid_indices = np.zeros(capacity, dtype=bool)  # boolean ndarray
        self.valid_set = set()
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
        self.values = np.empty(self.capacity)  # Store TD errors for each transition
        self.td_errors = np.empty(self.capacity)  # Store TD errors for each transition
        self.valid_indices.fill(False)  # Reset all indices to invalid
        self.valid_set.clear() 

    def __len__(self):
        return len(self.valid_set)

    def _include_for_sampling(self, index, terminated, truncated):
        if self.size >= self.num_td_steps:
            start_idx = (self.capacity + index - self.num_td_steps + 1)
            end_idx = self.capacity + index
            if terminated or truncated:
                if not self.valid_indices[index]:
                    self.valid_indices[index] = True
                    self.valid_set.add(index)                
            else:
                is_fail: bool = False
                for idx in range(start_idx, end_idx):
                    current_idx = idx % self.capacity
                    if (self.terminated[current_idx] or self.truncated[current_idx]):
                        is_fail = True
                        break
                if not is_fail:
                    if not self.valid_indices[index]:
                        self.valid_indices[index] = True
                        self.valid_set.add(index)                

    def _exclude_from_sampling(self, index):
        end_idx = index + self.num_td_steps
        for idx in range(index, end_idx):
            current_idx = idx % self.capacity
            if self.valid_indices[current_idx]:
                self.valid_indices[current_idx] = False
                self.valid_set.remove(current_idx)

    def _fetch_trajectory_slices(self, indices, td_steps):
        batch_size = len(indices)
        buffer_size = self.capacity
        # Expand indices for num_td_steps steps and wrap around using modulo operation
        expanded_indices = np.array([range(buffer_size + i -  td_steps + 1, buffer_size + i + 1) for i in indices]) % buffer_size
        expanded_indices = expanded_indices.reshape(batch_size, td_steps)
        
        # Fetch the slices
        states_slices = self.states[expanded_indices].reshape(batch_size, td_steps, -1)
        actions_slices = self.actions[expanded_indices].reshape(batch_size, td_steps, -1)
        rewards_slices = self.rewards[expanded_indices].reshape(batch_size, td_steps, -1)
        next_states_slices = self.next_states[expanded_indices].reshape(batch_size, td_steps, -1)
        terminated_slices = self.terminated[expanded_indices].reshape(batch_size, td_steps, -1)
        truncated_slices = self.truncated[expanded_indices].reshape(batch_size, td_steps, -1)
        truncated_slices[:, -1] = False  # Set the last element to False to prevent truncation
        dones_slices = np.logical_or(terminated_slices, truncated_slices)
        
        transitions = list(zip(states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices))
        return transitions
