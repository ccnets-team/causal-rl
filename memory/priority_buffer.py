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

        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated
        self.td_errors[self.index] = MIN_TD_ERROR

        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer

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
            actual_indices = self._reindex_indices(indices)

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

