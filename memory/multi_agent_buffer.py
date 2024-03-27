import torch

class MultiAgentBuffer:
    def __init__(self, buffer_type, capacity, num_agents, state_size, action_size, seq_len, device='cpu'):
        self.buffer_type = buffer_type
        self.capacity = capacity
        self.num_agents = num_agents
        self.seq_len = seq_len
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self._reset_buffer()

    def __len__(self):
        multi_agent_buffer_length = 0

        # Indices where buffer sizes are equal to or exceed capacity
        non_negative_agent_indices = self.buffer_sizes >= self.capacity
        if non_negative_agent_indices.any():
            multi_agent_buffer_length += self.buffer_sizes[non_negative_agent_indices].sum().item()

        # Indices where buffer sizes are below capacity
        negative_agent_indices = self.buffer_sizes < self.capacity
        if negative_agent_indices.any():
            # Calculate effective sizes considering sequence length and ensure non-negativity
            effective_sizes = (self.buffer_sizes[negative_agent_indices] - self.seq_len + 1).clamp_min(0).sum().item()
            multi_agent_buffer_length += effective_sizes

        return multi_agent_buffer_length
        
    def _assign_sample_prob(self, agent_ids, buffer_indices):
        # Calculate condition for each agent if their buffer size is sufficient for a sequence
        condition_list = self.buffer_sizes[agent_ids] >= self.seq_len - 1
        # Use torch.where to filter indices based on the condition
        filtered_agent_ids, filtered_buffer_indices = agent_ids[condition_list], buffer_indices[condition_list]
        if len(filtered_agent_ids) > 0 or len(filtered_buffer_indices) > 0:
            self.valid_indices[filtered_agent_ids, filtered_buffer_indices] = True

    def _reset_sample_prob(self, agent_ids, indices):
        end_indices = (indices + self.seq_len - 1) % self.capacity
        self.valid_indices[agent_ids, end_indices] = False

    def _reset_buffer(self):
        self.buffer_sizes = torch.zeros(self.num_agents, dtype=torch.int, device=self.device)
        self.agent_indices = torch.zeros(self.num_agents, dtype=torch.int, device=self.device)
        self.states = torch.empty((self.num_agents, self.capacity, self.state_size), dtype=torch.float, device=self.device)
        self.actions = torch.empty((self.num_agents, self.capacity, self.action_size), dtype=torch.float, device=self.device)
        self.rewards = torch.empty((self.num_agents, self.capacity), dtype=torch.float, device=self.device)
        self.next_states = torch.empty((self.num_agents, self.capacity, self.state_size), dtype=torch.float, device=self.device)
        self.terminated = torch.zeros((self.num_agents, self.capacity), dtype=torch.bool, device=self.device)
        self.truncated = torch.empty((self.num_agents, self.capacity), dtype=torch.bool, device=self.device)
        self.valid_indices = torch.zeros((self.num_agents, self.capacity), dtype=torch.bool, device=self.device)

    def add_transitions(self, agent_ids, states, actions, rewards, next_states, terminateds, truncateds):
        buffer_indices = self.agent_indices[agent_ids]
        
        self._reset_sample_prob(agent_ids, buffer_indices)
        
        self.states[agent_ids, buffer_indices] = states
        self.actions[agent_ids, buffer_indices] = actions
        self.rewards[agent_ids, buffer_indices] = rewards
        self.next_states[agent_ids, buffer_indices] = next_states
        self.terminated[agent_ids, buffer_indices] = terminateds
        self.truncated[agent_ids, buffer_indices] = truncateds

        self._assign_sample_prob(agent_ids, buffer_indices)

        # Increment agent-specific index and size
        self.agent_indices[agent_ids] = (buffer_indices + 1) % self.capacity
        self.buffer_sizes[agent_ids] += 1
        self.buffer_sizes[agent_ids].clamp_(max=self.capacity)

    def sample_trajectories(self, agent_ids, buffer_indices, sample_seq_len):
        # Ensure all tensors are on the correct device
        agent_ids = agent_ids.to(self.device)
        buffer_indices = buffer_indices.to(self.device)
        end_sequence_idx = buffer_indices
        start_sequence_idx = (end_sequence_idx - sample_seq_len + 1)

        # Identify agents with buffer size less than capacity
        negative_agent_ids = self.buffer_sizes[agent_ids] < self.capacity
        
        # If any agent does not have enough data to sample from, return None
        not_enough_data = torch.logical_and(negative_agent_ids, start_sequence_idx < 0)
        if not_enough_data.any():
            return None

        # Generate sequence indices and apply modulo for wrapping
        sequence_indices = (buffer_indices.unsqueeze(1) + torch.arange(-sample_seq_len + 1 + self.capacity, 1 + self.capacity, device=self.device))
        sequence_indices %= self.capacity  # Assuming self.capacity is the per-agent capacity
   
        # Verify indices are within bounds
        assert torch.all(sequence_indices < self.capacity), "Index out of bound"
        
        def slices(data):
            # Utilizing advanced indexing to gather the desired slices
            return data[agent_ids.unsqueeze(1), sequence_indices]

        # Slicing each component of the trajectories
        states_sliced = slices(self.states)
        actions_sliced = slices(self.actions)
        rewards_sliced = slices(self.rewards).unsqueeze(-1)  # Adding an extra dimension for consistency
        next_states_sliced = slices(self.next_states)

        # Handling terminated and truncated flags to create the done signal
        terminated_sliced = slices(self.terminated)
        truncated_sliced = slices(self.truncated)
        truncated_sliced[:, -1] = False  # Ensuring the last state of the sequence is not marked as truncated
        done_sliced = torch.logical_or(terminated_sliced, truncated_sliced).unsqueeze(-1)  # Marking the sequence as done if either flag is True

        trajectory_states = torch.concat([states_sliced, next_states_sliced[:, -1:]], dim = 1)
        trajectories = (trajectory_states, actions_sliced, rewards_sliced, done_sliced)
        return trajectories