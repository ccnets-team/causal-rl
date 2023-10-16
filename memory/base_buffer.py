'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import numpy as np

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

    def get_trajectories(self, indices, num_td_steps):
        batch_size = len(indices)
        
        # Expand indices for num_td_steps steps and wrap around using modulo operation
        expanded_indices = np.array([range(i, i + num_td_steps) for i in indices]) % self.capacity
        expanded_indices = expanded_indices.reshape(batch_size, num_td_steps)
        
        # Fetch the slices
        states_slices = self.states[expanded_indices].reshape(batch_size, num_td_steps, -1)
        actions_slices = self.actions[expanded_indices].reshape(batch_size, num_td_steps, -1)
        rewards_slices = self.rewards[expanded_indices].reshape(batch_size, num_td_steps, -1)
        next_states_slices = self.next_states[expanded_indices].reshape(batch_size, num_td_steps, -1)
        dones_slices = self.dones[expanded_indices].reshape(batch_size, num_td_steps, -1)
        
        # Create the done_mask
        cumulative_dones = np.cumsum(dones_slices, axis=1)
        shifted_dones = np.roll(cumulative_dones, shift=1, axis=1)
        shifted_dones[:, 0] = 0  # Set the first column to 0 after the roll
        done_mask = shifted_dones >= 1

        # Zero out elements after a done signal using done_mask
        states_slices[done_mask.repeat(self.state_size, axis=-1)] = 0
        actions_slices[done_mask.repeat(self.action_size, axis=-1)] = 0
        rewards_slices[done_mask] = 0
        next_states_slices[done_mask.repeat(self.state_size, axis=-1)] = 0
        dones_slices[done_mask] = 1
        # Note: We don't need to modify dones_slices since the mask itself is derived from it

        transitions = list(zip(states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices))
        return transitions
            
    def get_trajectory_indicies(self):
        if self.start_idx <= self.end_idx:
            trajectory_slices = self.trajectories[self.start_idx:self.end_idx]
        else:
            trajectory_slices = np.concatenate((self.trajectories[self.start_idx:], self.trajectories[:self.end_idx]), axis=0)

        # Get the start and length values from the slices
        starts, lengths = trajectory_slices[:, 0], trajectory_slices[:, 1]

        # Compute the end indices for each trajectory, ensuring they are within the buffer capacity
        ends = (starts + np.maximum(lengths - self.num_td_steps + 1, 0)) % self.capacity

        # Create an array to hold the valid indices
        valid_indices = np.empty(0, dtype=np.int32)  # Initialize an empty array

        # Iterate through the starts and ends, expanding the indices and appending them to valid_indices
        for start, end in zip(starts, ends):
            new_indices = np.arange(start, end) % self.capacity  # Compute the indices for this trajectory
            valid_indices = np.concatenate((valid_indices, new_indices))  # Append the new indices

        return valid_indices.tolist()  # Convert back to a list before returning, if needed