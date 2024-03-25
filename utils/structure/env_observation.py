import numpy as np
import torch

class EnvObservation:
    def __init__(self, obs_shapes, obs_types, num_agents, max_seq_len, device):
        assert len(obs_shapes) == len(obs_types), "The length of obs_shapes and obs_types must be the same."
        self.obs_shapes = obs_shapes
        self.obs_types = obs_types
        self.num_agents = num_agents
        self.max_seq_len = max_seq_len
        self.device = device
        self.data = self._create_empty_data()
        self.mask = torch.zeros((self.num_agents, self.max_seq_len), dtype=torch.float, device=self.device)

        # Compute the shape of concatenated vector data
        self.vector_data_shape = self._compute_vector_data_shape()

    def _create_empty_data(self):
        observations = {}
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            observations[obs_type] = torch.zeros((self.num_agents, self.max_seq_len, *shape), dtype=torch.float, device=self.device)
        return observations
    
    def _compute_vector_data_shape(self):
        # Compute the total size of the concatenated vector observations
        total_size = 0
        for shape in self.obs_shapes:
            if len(shape) == 1:  # Only include 1D observations
                total_size += shape[0]
        
        # Return the shape as (num_agents, total_size) assuming concatenation along the last dimension
        return (self.num_agents, total_size)
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            agent_indices, td_indices = key
        elif isinstance(key, slice) or isinstance(key, list) or isinstance(key, np.ndarray):
            agent_indices = key
            td_indices = None
        else:
            raise TypeError("Invalid key type. Must be a tuple, slice, list, or numpy array.")

        new_agent_indices = None
        if isinstance(agent_indices, int):
            new_agent_indices = np.array([agent_indices])
        elif isinstance(agent_indices, slice):
            new_agent_indices = np.arange(self.num_agents)[agent_indices]
        elif isinstance(agent_indices, list) or isinstance(agent_indices, np.ndarray):
            new_agent_indices = np.array(agent_indices)
        else:
            raise TypeError("Invalid type for agent_indices. Must be int, slice, list, or numpy array.")

        new_td_indices = None
        if td_indices is None:
            new_td_indices = np.arange(self.max_seq_len)
        elif isinstance(td_indices, int):
            new_td_indices = np.array([td_indices])
        elif isinstance(td_indices, slice):
            new_td_indices = np.arange(self.max_seq_len)[td_indices]
        elif isinstance(td_indices, list) or isinstance(td_indices, np.ndarray):
            new_td_indices = np.array(td_indices)
        else:
            raise TypeError("Invalid type for td_indices. Must be int, slice, list, numpy array, or None.")

        new_num_agents = len(new_agent_indices)
        new_model_seq_length = len(new_td_indices)

        new_observation = EnvObservation(self.obs_shapes, self.obs_types, num_agents=new_num_agents, max_seq_len=new_model_seq_length, device = self.device)
        for obs_type in self.obs_types:
            new_observation.data[obs_type] = self.data[obs_type][new_agent_indices, new_td_indices]
        new_observation.mask = self.mask[new_agent_indices, new_td_indices]
    
        return new_observation
        
    def __setitem__(self, agent_indices, values):
        for obs_type in self.obs_types:
            # Check if the input value is already a PyTorch tensor
            if not isinstance(values[obs_type], torch.Tensor):
                # Convert to a PyTorch tensor, ensuring it's of the correct type and on the correct device
                value_tensor = torch.tensor(values[obs_type], dtype=torch.float32, device=self.device)
            else:
                # Ensure the tensor is on the correct device (in case it is not)
                value_tensor = values[obs_type].to(device=self.device)
            
            # Assign the tensor to the specified indices for this observation type
            self.data[obs_type][agent_indices] = value_tensor

    def reset(self):
        self.mask.fill_(0.0)
        self.data = self._create_empty_data()
                                            
    def shift(self, term_agents, dec_agents):
        # Convert term_agents and dec_agents to torch tensors if they are not already
        if not isinstance(term_agents, torch.Tensor):
            term_agents = torch.tensor(term_agents, device=self.device)
        if not isinstance(dec_agents, torch.Tensor):
            dec_agents = torch.tensor(dec_agents, device=self.device)
            
        # Roll data and mask for 'dec_agents'
        for key in self.data:
            self.data[key][dec_agents] = torch.roll(self.data[key][dec_agents], shifts=-1, dims=1)
        self.mask[dec_agents] = torch.roll(self.mask[dec_agents], shifts=-1, dims=1)

        # Mask out all previous data for 'term_agents' and indicate the start of a new episode
        self.mask[term_agents, :] = 0  

        # Set the last time dimension to 0 for data of 'dec_agents'
        self.mask[dec_agents, -1] = 1

    def to_vector(self):
        """
        Convert the observations to a vector format.
        Only takes into account obs_types that are vectors after the num_agents dimension.
        Resulting shape will be (num_agents, concatenated_data)
        This method selects the data from the first time step for each agent.
        """
        # Filter the data for 1D observations
        filtered_data = {k: v for k, v in self.data.items() if len(v.size()[2:]) == 1}

        # Use advanced indexing to collect the tensors
        vectors = [v for v in filtered_data.values()]
        
        # Check if vectors is empty
        if not vectors:
            # Handle the empty case. You could return an empty tensor with a shape that fits your scenario.
            # For example, return a tensor of shape (num_agents, 0) to indicate no concatenated data.
            return torch.empty(self.vector_data_shape, device=self.device)
            
        concatenated_data = torch.cat(vectors, dim=-1)
        return concatenated_data
