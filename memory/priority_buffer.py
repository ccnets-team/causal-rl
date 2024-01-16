'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
from .base_buffer import BaseBuffer
import numpy as np

MIN_TD_ERROR = 1e-4

class PriorityBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, seq_len):
        super().__init__("priority", capacity, state_size, action_size, seq_len)
        self.reset_buffer()

    def __len__(self):
        return self.size

    def _exclude_from_sampling(self, index):
        end_idx = (index + self.seq_len - 1) % self.capacity
        self.td_errors[end_idx] = 0.0

    def reset_buffer(self):
        self._reset_buffer()
        self.td_errors = np.empty(self.capacity)  # Store TD errors for each transition

    def add_transition(self, state, action, reward, next_state, terminated, truncated):

        self._exclude_from_sampling(self.index)

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

    def sample_trajectories(self, indices, td_steps):
        return self._fetch_trajectory_slices(indices, td_steps)