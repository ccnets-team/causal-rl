'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
from .base_buffer import BaseBuffer
import numpy as np

MIN_TD_ERROR = 1e-4

class PriorityBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps):
        super().__init__("priority", capacity, state_size, action_size, num_td_steps)

    def add_transition(self, state, action, reward, next_state, terminated, truncated):
        # Remove the current index from valid_indices if it's present
        self._exclude_from_sampling(self.index)

        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated
        self.td_errors[self.index] = MIN_TD_ERROR

        # Check if adding this data creates a valid trajectory
        self._include_for_sampling(self.index, terminated, truncated)

        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer

    def sample_trajectories(self, indices, td_steps, use_actual_indices=False):

        if use_actual_indices:
            actual_indices = indices
        else:
            # Convert valid_set to a list to maintain order
            ordered_valid_set = list(self.valid_set)

            # Check if the indices are more than the available samples
            if len(indices) > len(ordered_valid_set):
                raise ValueError("Not enough valid samples in the buffer to draw the requested sample size.")

            # Map the requested indices to actual indices in the valid set
            actual_indices = [ordered_valid_set[idx] for idx in indices]

        # Retrieve trajectories for the given actual indices
        samples = self._fetch_trajectory_slices(actual_indices, td_steps)
            
        # Ensure the number of samples matches the number of requested indices
        if len(samples) != len(indices):
            raise ValueError("Mismatch in the number of samples fetched and the number of requested indices.")

        return samples

    def update_td_errors_for_sampled(self, indices, td_errors, mask, use_actual_indices=False):
        """
        Updates the TD errors for sampled experiences.

        :param indices: Indices of the last element of the sampled experiences.
        :param td_errors: Temporal Difference errors for the sampled experiences.
        :param mask: Mask array indicating which steps to update.
        :param use_actual_indices: Flag to indicate if indices are actual indices or need conversion.
        """
        
        if use_actual_indices:
            actual_indices = indices
        else:
            # Convert valid_set to a list to maintain order
            ordered_valid_set = list(self.valid_set)

            # Check if the indices are more than the available samples
            if len(indices) > len(ordered_valid_set):
                raise ValueError("Not enough valid samples in the buffer to draw the requested sample size.")

            # Map the requested indices to actual indices in the valid set
            actual_indices = np.array([ordered_valid_set[idx] for idx in indices])

        # Calculate the range of indices for each trajectory
        seq_len = mask.shape[1]
        range_indices = seq_len - 1 - np.arange(seq_len)
        all_indices = (self.capacity + actual_indices.reshape(-1, 1) - range_indices) % self.capacity

        # Flatten the mask and indices array for advanced indexing
        update_mask = mask.ravel().astype(bool)
        all_indices_flat = all_indices.ravel()
        td_errors_flat = td_errors.ravel()

        # Perform the update using advanced indexing
        self.td_errors[all_indices_flat[update_mask]] = td_errors_flat[update_mask]

