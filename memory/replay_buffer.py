import random
import torch
import numpy as np
from collections import defaultdict
from utils.structure.trajectories import BatchTrajectory, MultiEnvTrajectories
from memory.standard_buffer import StandardBuffer

def initialize_buffers(buffer_type, num_environments, num_agents, capacity_per_agent, state_size, action_size, num_td_steps):
    if buffer_type == "standard":
        return [[StandardBuffer(capacity_per_agent, state_size, action_size, num_td_steps) for _ in range(num_agents)] for _ in range(num_environments)]
    elif buffer_type == "priority":
        return NotImplementedError
    else:
        raise NotImplementedError

class ExperienceMemory:
    def __init__(self, env_config, training_params, algorithm_params, memory_params, device):
        self.device = device
        num_td_steps = algorithm_params.num_td_steps
        
        self.num_agents = env_config.num_agents
        self.num_environments = env_config.num_environments
        self.capacity_per_agent = int(memory_params.buffer_size)//(env_config.num_environments*env_config.num_agents) + num_td_steps 
        
        state_size, action_size = env_config.state_size, env_config.action_size 
        self.batch_size = training_params.batch_size 
        self.num_td_steps = num_td_steps
        self.buffer_type  = memory_params.buffer_type
        self.multi_buffers = initialize_buffers(self.buffer_type, self.num_environments, self.num_agents, self.capacity_per_agent, state_size, action_size, num_td_steps)

    def __len__(self):
        return sum(len(buf) for env in self.multi_buffers for buf in env)

    def get_total_data_points(self):
        return sum(buf.size for env in self.multi_buffers for buf in env)

    def reset_buffers(self):
        return [buf._reset_buffer() for env in self.multi_buffers for buf in env]

    def sample_trajectory_from_buffer(self, env_id, agent_id, sample_size, td_steps):
        return self.multi_buffers[env_id][agent_id].sample(sample_size, td_steps)

    def push_trajectory_data(self, multi_env_trajectories: MultiEnvTrajectories):
        tr = multi_env_trajectories
        if tr.env_ids is None:
            return
        
        # Initialize td_errors with None if it doesn't have values
        td_errors = tr.td_errors if tr.td_errors is not None else [None] * len(tr.env_ids)
        
        # zip all the common attributes along with td_errors
        attributes = zip(tr.env_ids, tr.agent_ids, tr.states, tr.actions, tr.rewards, tr.next_states, tr.dones_terminated, tr.dones_truncated, td_errors)
        
        for env_id, agent_id, state, action, reward, next_state, done_terminated, done_truncated, td_error in attributes:
            self.multi_buffers[env_id][agent_id].add(state, action, reward, next_state, done_terminated, done_truncated, td_error)

    def sample_trajectory_data(self):
        batch_size = self.batch_size
        num_td_steps = self.num_td_steps
        samples = self._sample_balanced_trajectory_data(batch_size, num_td_steps) 
        if samples is None:
            return None
        states      = np.stack([b[0] for b in samples], axis=0)
        actions     = np.stack([b[1] for b in samples], axis=0)
        rewards     = np.stack([b[2] for b in samples], axis=0)
        next_states = np.stack([b[3] for b in samples], axis=0)
        dones       = np.stack([b[4] for b in samples], axis=0)
        
        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(x).to(self.device), 
                                                    [states, actions, rewards, next_states, dones])
        return BatchTrajectory(states, actions, rewards, next_states, dones)
    
    def _sample_balanced_trajectory_data(self, sample_size = None, num_td_steps = None):
        _num_td_steps = self.num_td_steps if num_td_steps is None else num_td_steps
        _sample_size = self.batch_size if sample_size is None else sample_size
        
        # Step 1: Compute cumulative sizes
        cumulative_sizes = []
        total_buffer_size = 0
        for env_id in range(self.num_environments):
            for agent_id in range(self.num_agents):
                total_buffer_size += len(self.multi_buffers[env_id][agent_id])
                cumulative_sizes.append(total_buffer_size)

        if _sample_size > total_buffer_size:
            return None
            # raise ValueError("Sample size exceeds the total number of experiences available")

        # Step 2: Randomly sample indices
        sampled_indices = random.sample(range(total_buffer_size), _sample_size)

        # Step 3: Count the number of samples needed from each buffer
        buffer_sample_counts = defaultdict(int)
        for idx in sampled_indices:
            # Find the buffer this index belongs to
            buffer_id = next(i for i, cum_size in enumerate(cumulative_sizes) if idx < cum_size)
            buffer_sample_counts[buffer_id] += 1

        # Step 4: Fetch the experiences using sample_from_buffer
        samples = []
        for buffer_id, _sample_size in buffer_sample_counts.items():
            # Map buffer_id back to env_id and agent_id
            env_id = buffer_id // self.num_agents
            agent_id = buffer_id % self.num_agents

            # Fetch the experience
            batch = self.sample_trajectory_from_buffer(env_id, agent_id, _sample_size, _num_td_steps)
            samples.extend(batch)
        return samples
        

