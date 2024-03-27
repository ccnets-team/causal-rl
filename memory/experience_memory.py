import torch
from utils.structure.data_structures import BatchTrajectory, AgentTransitions
from memory.multi_agent_buffer import MultiAgentBuffer

class ExperienceMemory:
    def __init__(self, env_config, seq_len, batch_size, buffer_size, device):
        """
        Initializes an experience memory buffer for storing and managing agent experiences.

        Args:
            env_config: Configuration containing environment details like number of agents, environments,
                        state size, and action size.
            tot_seq_len: Total sequence length for GPT and TD computations.
            batch_size: Size of batches to sample from the buffer.
            buffer_size: Total size of the buffer across all agents and environments.
            device: The computational device (CPU or GPU) where the data will be stored.
        """
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.total_agents = self.num_agents * self.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.seq_length = seq_len
        self.batch_size = batch_size

        # Capacity calculation now accounts for all agents across all environments
        self.capacity_for_agent = buffer_size//self.total_agents

        # Single buffer initialization
        self.buffer = MultiAgentBuffer('Experience', self.capacity_for_agent, self.total_agents, self.state_size, self.action_size, self.seq_length, 'cpu')

    def __len__(self):
        return len(self.buffer)

    def get_total_data_points(self):
        return self.buffer.buffer_sizes.sum().item()

    def reset_memory(self):
        self.buffer._reset_buffer()
        
    def _create_batch_trajectory_components(self, samples):
        components = [torch.stack([b[i] for b in samples], dim=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        return states, actions, rewards, next_states, dones

    def push_transitions(self, transitions: AgentTransitions):
        if transitions.env_ids is None:
            return
        agent_ids = transitions.env_ids * self.num_agents + transitions.agent_ids
        self.buffer.add_transitions(agent_ids, transitions.states, transitions.actions, transitions.rewards, transitions.next_states, \
            transitions.dones_terminated, transitions.dones_truncated, transitions.content_length)
        
    def _map_to_flat_agent_id(self, env_id, agent_id):
        # Map the (env_id, agent_id) tuple to a unique flat_agent_id assuming agent_id is unique within each environment
        return env_id * self.total_agents + agent_id

    def sample_batch_trajectory(self, sample_seq_len):
        sample_size = self.batch_size
        total_sample_list = self.buffer.valid_indices.to(self.device).flatten()
        # Filter indices that are marked True (valid)
        valid_indices = torch.nonzero(total_sample_list, as_tuple=False).squeeze()

        if len(valid_indices) < sample_size:
            return None

        # Generate a random permutation of indices up to the length of valid_indices and select the first sample_size elements
        permuted_indices = torch.randperm(len(valid_indices), device=self.device)[:sample_size]

        # Select the actual indices based on the permuted indices
        agent_ids = valid_indices[permuted_indices] // self.capacity_for_agent
        buffer_indices = valid_indices[permuted_indices] % self.capacity_for_agent
        
        # Sample trajectories from the buffer
        samples = self.buffer.sample_trajectories(agent_ids, buffer_indices, sample_seq_len)
        if samples is None:
            return None
        
        # Convert samples to the desired device and type all at once
        trajectory_states, actions, rewards, dones = (x.to(self.device).float() for x in samples)
        
        # Construct and return the BatchTrajectory
        return BatchTrajectory(trajectory_states, actions, rewards, dones)