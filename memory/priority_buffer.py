'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
from .base_buffer import BaseBuffer

MIN_TD_ERROR = 1e-6  # Minimum threshold for TD errors to prevent zero values
class PriorityBuffer(BaseBuffer):
    def __init__(self, capacity, state_size, action_size, num_td_steps, gamma):
        super().__init__("priority", capacity, state_size, action_size, num_td_steps, gamma)

    def _update_td_errors(self, idx, normalized_reward, value):
        """
        Updates the TD errors for the current and previous transitions.

        :param value: The estimated value for the current state.
        :param normalized_reward: Normalized reward for the current state.
        """
        self.values[idx] = value

        previous_idx = (idx - 1 + self.capacity) % self.capacity
        if self.valid_indices[previous_idx] and not self.terminated[previous_idx] and not self.truncated[previous_idx]:
            # Calculate and update TD error for the previous index
            prev_td_error = normalized_reward + self.gamma * value - self.values[previous_idx]
            self.td_errors[previous_idx] = max(abs(prev_td_error), MIN_TD_ERROR)

        if self.valid_indices[idx]:
            if self.terminated[idx] or self.truncated[idx]:
                # Calculate and update TD error for a terminal or truncated current state
                cur_td_error = normalized_reward - value
                self.td_errors[idx] = max(abs(cur_td_error), MIN_TD_ERROR)
            else:
                # Set TD error to minimum for non-terminal, non-truncated states
                self.td_errors[idx] = MIN_TD_ERROR
            
    def add_transition(self, state, action, reward, next_state, terminated, truncated, value, normalized_reward):
        # Remove the current index from valid_indices if it's present
        self._exclude_from_sampling(self.index)

        # Update the buffer with the new transition data
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated
        self.truncated[self.index] = truncated

        # Check if adding this data creates a valid trajectory
        self._include_for_sampling(self.index, terminated, truncated)

        if value is not None:
            self._update_td_errors(self.index, normalized_reward, value)

        # Increment the buffer index and wrap around if necessary
        self.index = (self.index + 1) % self.capacity

        # Increment the size of the buffer if it's not full
        if self.size < self.capacity:
            self.size += 1
        # Remove invalid indices caused by the circular nature of the buffer

    def sample_trajectories(self, indices, td_steps):
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

    def update_td_errors_for_sampled(self, indices, normalized_rewards, values):
        """
        Updates the TD errors for sampled experiences.

        :param indices: Indices of the sampled experiences.
        :param normalized_rewards: Normalized rewards for the sampled experiences.
        :param values: Estimated values for the sampled experiences.
        """
        for idx, norm_reward, val in zip(indices, normalized_rewards, values):
            # Calculate previous and current indices in the buffer
            previous_idx = (idx - 1 + self.capacity) % self.capacity
            current_idx = idx

            # Update TD errors for previous and current indices
            self._update_td_errors(previous_idx, norm_reward[0], val[0])
            self._update_td_errors(current_idx, norm_reward[1], val[1])
            