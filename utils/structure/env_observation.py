import numpy as np

class EnvObservation:
    def __init__(self, obs_shapes, obs_types, num_agents, gpt_seq_length):
        assert len(obs_shapes) == len(obs_types), "The length of obs_shapes and obs_types must be the same."
        self.obs_shapes = obs_shapes
        self.obs_types = obs_types
        self.num_agents = num_agents
        self.gpt_seq_length = gpt_seq_length
        self.data = self._create_empty_data()
        self.mask = np.zeros((self.num_agents, self.gpt_seq_length), dtype=np.float32)

    def _create_empty_data(self):
        observations = {}
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            observations[obs_type] = np.zeros((self.num_agents, self.gpt_seq_length, *shape), dtype=np.float32)
        return observations

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
            new_td_indices = np.arange(self.gpt_seq_length)
        elif isinstance(td_indices, int):
            new_td_indices = np.array([td_indices])
        elif isinstance(td_indices, slice):
            new_td_indices = np.arange(self.gpt_seq_length)[td_indices]
        elif isinstance(td_indices, list) or isinstance(td_indices, np.ndarray):
            new_td_indices = np.array(td_indices)
        else:
            raise TypeError("Invalid type for td_indices. Must be int, slice, list, numpy array, or None.")

        new_num_agents = len(new_agent_indices)
        new_model_seq_length = len(new_td_indices)

        new_observation = EnvObservation(self.obs_shapes, self.obs_types, num_agents=new_num_agents, gpt_seq_length=new_model_seq_length)
        for obs_type in self.obs_types:
            new_observation.data[obs_type] = self.data[obs_type][new_agent_indices, new_td_indices]
        new_observation.mask = self.mask[new_agent_indices, new_td_indices]

        return new_observation
    
    def __setitem__(self, agent_indices, values):
        for obs_type in self.obs_types:
            self.data[obs_type][agent_indices] = values[obs_type]
    
    def reset(self):
        self.mask.fill(0.0)
        self.data = self._create_empty_data()
                            
    def shift(self, term_agents, dec_agents):
        assert isinstance(term_agents, np.ndarray), "'term_agents' must be a NumPy ndarray or None"
        assert isinstance(dec_agents, np.ndarray), "'dec_agents' must be a NumPy ndarray or None"

        # Roll data and mask for 'dec_agents'
        for key in self.data:
            self.data[key][dec_agents] = np.roll(self.data[key][dec_agents], shift=-1, axis=1)
        self.mask[dec_agents] = np.roll(self.mask[dec_agents], shift=-1, axis=1)

        # Mask out all previous data for 'term_agents' and indicate the start of a new episode
        self.mask[term_agents, :] = 0  

        # Set the last time dimension to 0 for data of 'dec_agents'
        # for key in self.data:
        #     self.data[key][dec_agents, -1] = 0
        self.mask[dec_agents, -1] = 1

    def to_vector(self):
        """
        Convert the observations to a vector format.
        Only takes into account obs_types that are vectors after the num_agents dimension.
        Resulting shape will be (num_agents, concatenated_data)
        This method selects the data from the first time step for each agent.
        """
        vectors = []
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            # Checking if the shape after num_agents is 1D
            if len(shape) == 1:
                data = self.data[obs_type]  # Select the first time step for all agents
                vectors.append(data)
        
        concatenated_data = np.concatenate(vectors, axis=-1)  # Concatenate along the data dimension
        return concatenated_data