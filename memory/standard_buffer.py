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
        self.valid_indices = np.zeros(capacity, dtype=bool)  # boolean ndarray
        self.valid_set = set()
        self._reset()
        
    def _reset(self):
        self.size = 0  
        self.index = 0
        self.states = np.empty((self.capacity, self.state_size))
        self.actions = np.empty((self.capacity, self.action_size))
        self.rewards = np.empty(self.capacity)
        self.next_states = np.empty((self.capacity, self.state_size))
        self.dones = np.empty(self.capacity)       
        self.valid_indices.fill(False)  # Reset all indices to invalid
        self.valid_set.clear() 

        
    def __len__(self):
        return len(self.valid_set)
    
    def add_valid_index(self, index, terminated, truncated):
        buffer_len = self.size
        start_idx = (index - self.num_td_steps + 1) % self.capacity

        # Invalidate the trajectory if the current index is marked as done
        find_index: int = -1
        if terminated:
            # Iterate over the range to find the first index which is not done
            for idx in range(start_idx, start_idx + self.num_td_steps - 1):
                current_idx = idx % self.capacity
                if not self.dones[current_idx]:
                    find_index = current_idx
                    break
        elif not truncated:
            if buffer_len == self.capacity:
                if start_idx < index:
                    if not np.any(self.dones[start_idx:index]):
                        find_index = start_idx
                else:
                    if not np.any(np.concatenate((self.dones[index + 1:], self.dones[:start_idx + 1]))):
                        find_index = start_idx
            elif start_idx < index:
                if not np.any(self.dones[start_idx:index]):
                    find_index = start_idx
        if find_index >= 0:
            if not self.valid_indices[find_index]:
                self.valid_indices[find_index] = True
                self.valid_set.add(find_index)

    def remove_invalid_indices(self, index):
        if self.valid_indices[index]:
            self.valid_set.remove(index)
            self.valid_indices[index] = False

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
                    

class StandardBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("standard", capacity, state_size, action_size, num_td_steps)

    def add(self, state, action, reward, next_state, terminated, truncated, td_error=None):
        # Remove the current index from valid_indices if it's present
        self.remove_invalid_indices(self.index)

        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = terminated

        # Check if adding this data creates a valid trajectory
        self.add_valid_index(self.index, terminated, truncated)

        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer

    def sample(self, sample_size, td_steps):
        # Check if there are enough valid indices to sample from
        if len(self.valid_set) < sample_size:
            raise ValueError("Not enough valid samples in the buffer to draw the requested sample size.")

        # Randomly select 'sample_size' indices from the set of valid indices
        selected_indices = random.sample(self.valid_set, sample_size)
        
        # Retrieve trajectories for the selected indices
        samples = self.get_trajectories(selected_indices, td_steps)
        
        assert(len(samples) == sample_size)
        return samples