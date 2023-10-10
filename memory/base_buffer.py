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

    def __len__(self):
        return max(self.size - self.num_td_steps + 1, 0)

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
        
        # If a step is done, the subsequent steps are also flagged as done
        done_mask = np.cumsum(dones_slices, axis=1) > 0
        dones_slices[done_mask] = True

        # Identify termination push_transitionsstep
        done_steps = np.argmax(dones_slices, axis=1)
        non_terminated_indices = np.where(dones_slices.sum(axis=1) == 0)[0]
        done_steps[non_terminated_indices] = num_td_steps - 1  # If not terminated, set to the last step

        # Using direct indexing to extract the required transitions
        valid_states = states_slices
        valid_actions = actions_slices
        valid_rewards = rewards_slices
        valid_next_states = next_states_slices
        valid_dones = done_mask.astype(np.float32)  # Convert to float for easy processing with tensors later on
        
        transitions = list(zip(valid_states, valid_actions, valid_rewards, valid_next_states, valid_dones))
        return transitions
        
    def get_trajectory_indicies(self):
        buffer_len = self.size
        if buffer_len == self.capacity:  # Buffer is full
            end_valid_idx = self.index - self.num_td_steps
            if end_valid_idx < 0:  # Check for wrap around
                possible_indices = np.arange(end_valid_idx + self.num_td_steps, end_valid_idx + buffer_len + 1) % self.capacity
            else:
                first_range = np.arange(end_valid_idx + self.num_td_steps, buffer_len)
                second_range = np.arange(0, end_valid_idx + 1)
                possible_indices = np.concatenate([first_range, second_range]) % self.capacity
        else:
            possible_indices = np.arange(0, self.index - self.num_td_steps + 1)

        # Construct expanded indices
        expanded_indices = (np.arange(self.num_td_steps).reshape(-1, 1) + possible_indices) % self.capacity
        valid_mask = (~self.dones[expanded_indices[:-1]].any(axis=0))

        return possible_indices[valid_mask].tolist()