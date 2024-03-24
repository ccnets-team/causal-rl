from collections import defaultdict
import numpy as np
import random
import torch
from utils.structure.data_structures import BatchTrajectory, AgentTransitions
from memory.standard_buffer import StandardBuffer
from training.managers.sequence_manager import TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS

class ExperienceMemory:
    def __init__(self, env_config, input_seq_len, batch_size, buffer_size, device):
        """
        Initializes an experience memory buffer for storing and managing agent experiences.

        Args:
            env_config: Configuration containing environment details like number of agents, environments,
                        state size, and action size.
            training_params: Training parameters including batch size and buffer size.
            algorithm_params: Algorithm-specific parameters, used here to determine the combined sequence
                              length for GPT and TD computations.
            device: The computational device (CPU or GPU) where the data will be stored.

        The buffer's total capacity and structure are tailored to support efficient experience replay,
        enabling the model to learn from past interactions. The `gpt_td_seq_length` is set to 1.5 times
        the GPT sequence length to account for sequence values computed for both the front and rear parts
        of the sequence, optimizing for computational efficiency and learning effectiveness.
        """
        self.device = device
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.total_seq_length = input_seq_len + max(input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.batch_size = batch_size

        # Capacity calculation now in a separate method for clarity
        self.capacity_per_agent = self._calculate_capacity_per_agent(buffer_size)

        # Buffer initialization now in a separate method for clarity
        self.multi_buffers = self._initialize_buffers()

    def _calculate_capacity_per_agent(self, buffer_size):
        # Capacity calculation logic separated for clarity
        return int(buffer_size) // (self.num_environments * self.num_agents) + self.total_seq_length

    def _initialize_buffers(self):
        # Buffer initialization logic separated for clarity
        return [[StandardBuffer(self.capacity_per_agent, self.state_size, self.action_size, self.total_seq_length) for _ in range(self.num_agents)] for _ in range(self.num_environments)]
            
    def __len__(self):
        return sum(len(buf) for env in self.multi_buffers for buf in env)

    def get_total_data_points(self):
        return sum(buf.size for env in self.multi_buffers for buf in env)

    def reset_memory(self):
        return [buf.reset_buffer() for env in self.multi_buffers for buf in env]

    def sample_trajectory_from_buffer(self, env_id, agent_id, indices, td_steps):
        return self.multi_buffers[env_id][agent_id].sample_trajectories(indices, td_steps)

    def _create_batch_trajectory_components(self, samples):
        components = [np.stack([b[i] for b in samples], axis=0) for i in range(5)]
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), components)
        return states, actions, rewards, next_states, dones

    def _get_env_agent_ids(self, buffer_id):
        # Retrieve environment and agent IDs from the global index
        env_id = buffer_id // self.num_agents
        agent_id = buffer_id % self.num_agents
        return env_id, agent_id
            
    def push_transitions(self, agent_transitions: AgentTransitions):
        if agent_transitions.env_ids is None:
            return

        for data in zip(agent_transitions.env_ids, agent_transitions.agent_ids,
                        agent_transitions.states, agent_transitions.actions,
                        agent_transitions.rewards, agent_transitions.next_states,
                        agent_transitions.dones_terminated, agent_transitions.dones_truncated,
                        agent_transitions.content_length):
            env_id, agent_id = data[:2]
            buffer = self.multi_buffers[env_id][agent_id]
            buffer.add_transition(*data[2:])
        
    def sample_batch_trajectory(self, sample_seq_len):
        sample_sz = self.batch_size
        samples = self.balanced_sample_trajectory_data(sample_sz, sample_seq_len)
        if samples is None:
            return None

        states, actions, rewards, next_states, dones = self._create_batch_trajectory_components(samples)
        return BatchTrajectory(states, actions, rewards, next_states, dones)
        
    def balanced_sample_trajectory_data(self, sample_size, sample_seq_len):
        total_sample_list = self._get_sample_list()
        
        # Filter indices that are marked True (valid)
        valid_indices = torch.nonzero(total_sample_list, as_tuple=False).squeeze()
        
        if len(valid_indices) < sample_size:
            return None

        # Generate a random permutation of indices up to the length of valid_indices and select the first sample_size elements
        permuted_indices = torch.randperm(len(valid_indices), device=self.device)[:sample_size]
        
        # Select the actual indices based on the permuted indices
        buffer_ids = valid_indices[permuted_indices] // self.capacity_per_agent
        local_indices = valid_indices[permuted_indices] % self.capacity_per_agent
        
        buffer_ids = buffer_ids.detach().cpu().numpy()
        local_indices = local_indices.detach().cpu().numpy()
        
        # Initialize a dictionary to hold arrays of local indices for each buffer
        buffer_local_indices = defaultdict(list)

        # Accumulate local indices for each buffer
        for buffer_id, local_idx in zip(buffer_ids, local_indices):
            buffer_local_indices[buffer_id].append(local_idx)

        samples = self._fetch_samples(buffer_local_indices, sample_seq_len)
        return samples
        
    def _fetch_samples(self, buffer_local_indices, td_steps):
        samples = []
        for buffer_id, local_indices_array in buffer_local_indices.items():
            env_id, agent_id = self._get_env_agent_ids(buffer_id)
            # Assuming sample_trajectory_from_buffer can accept an array of local_indices
            batch = self.sample_trajectory_from_buffer(env_id, agent_id, local_indices_array, td_steps)
            samples.extend(batch)
        return samples
    
    def _get_sample_list(self):
        buffer_sample_indices = torch.cat([
            torch.tensor(buf.valid_indices, dtype=torch.bool).to(self.device) 
            for env in self.multi_buffers for buf in env
        ], dim = 0)
        return buffer_sample_indices