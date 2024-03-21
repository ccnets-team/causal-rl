'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np

def mark_padded_as_done(dones_slices, padding_length_slices):
    # Vectorized operation to update 'dones' based on padding length
    # Generate an array of indices for each trajectory
    batch_size, td_steps, _ = dones_slices.shape
    trajectory_indices = np.arange(td_steps)
    
    # Broadcast to match shapes (batch_size, td_steps)
    broadcasted_indices = trajectory_indices + np.zeros((batch_size, td_steps), dtype=int)
    
    # Calculate the last valid index for each trajectory
    last_valid_indices = td_steps - padding_length_slices[:, -1, 0]  # Assuming padding length is uniform across td_steps
    
    # Create a mask where each element is True if its index is >= the last valid index for its trajectory
    mask = broadcasted_indices >= (last_valid_indices[:, None] - 1)  # Subtract 1 because indices are 0-based
    
    # Apply the mask to the 'dones' slices
    dones_slices[mask] = True
    return dones_slices
    
class BaseBuffer:
    def __init__(self, buffer_type, capacity, state_size, action_size, seq_len):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.seq_len = seq_len
        self.state_size = state_size
        self.action_size = action_size
        self.valid_indices = np.zeros(capacity, dtype=bool)
        
    def __len__(self):
        if self.size >= self.capacity:
            return self.size
        return max(self.size - self.seq_len + 1, 0)
    
    def _assign_sample_prob(self, index):
        if self.size < self.seq_len - 1:
            return  # Early exit if buffer size is less than required sequence length.
        self.valid_indices[index] = True

    def _reset_sample_prob(self, index):
        end_idx = (index + self.seq_len - 1) % self.capacity
        self.valid_indices[end_idx] = False
                
    def _reset_buffer(self):
        self.size = 0  
        self.index = 0
        self.states = np.empty((self.capacity, self.state_size))
        self.actions = np.empty((self.capacity, self.action_size))
        self.rewards = np.empty(self.capacity)
        self.next_states = np.empty((self.capacity, self.state_size))
        self.terminated = np.empty(self.capacity)       
        self.truncated = np.empty(self.capacity)       
        self.padding_length = np.empty(self.capacity)       
        self.valid_indices.fill(False)  # Reset all indices to invalid



    def _fetch_trajectory_slices(self, indices, td_steps):
        batch_size = len(indices)
        buffer_size = self.capacity
        # Expand indices for num_td_steps steps and wrap around using modulo operation
        expanded_indices = np.array([range(buffer_size + i -  td_steps + 1, buffer_size + i + 1) for i in indices]) % buffer_size
        expanded_indices = expanded_indices.reshape(batch_size, td_steps)

        # Fetch the slices using advanced indexing
        states_slices = self.states[expanded_indices]
        actions_slices = self.actions[expanded_indices]
        rewards_slices = self.rewards[expanded_indices]
        next_states_slices = self.next_states[expanded_indices]
        terminated_slices = self.terminated[expanded_indices]
        truncated_slices = self.truncated[expanded_indices]
        padding_length_slices = self.padding_length[expanded_indices]

        states_slices = states_slices.reshape(batch_size, td_steps, -1)
        actions_slices = actions_slices.reshape(batch_size, td_steps, -1)
        rewards_slices = rewards_slices.reshape(batch_size, td_steps, -1)
        next_states_slices = next_states_slices.reshape(batch_size, td_steps, -1)
        terminated_slices = terminated_slices.reshape(batch_size, td_steps, -1)
        truncated_slices = truncated_slices.reshape(batch_size, td_steps, -1)
        padding_length_slices = padding_length_slices.reshape(batch_size, td_steps, -1)

        # Ensure the last element of truncated_slices is False to prevent truncation
        truncated_slices[:, -1] = False
        dones_slices = np.logical_or(terminated_slices, truncated_slices)

        dones_slices = mark_padded_as_done(dones_slices, padding_length_slices)
                
        transitions = list(zip(states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices))
        return transitions
    