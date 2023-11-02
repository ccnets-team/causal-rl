import numpy as np
import copy

class EnvObservation:
    def __init__(self, obs_shapes, obs_types, num_agents, num_td_steps):
        assert len(obs_shapes) == len(obs_types), "The length of obs_shapes and obs_types must be the same."
        self.obs_shapes = obs_shapes
        self.obs_types = obs_types
        self.num_agents = num_agents
        self.num_td_steps = num_td_steps
        self.data = self._create_empty_data()
        self.mask = np.zeros((self.num_agents, self.num_td_steps), dtype=np.float32)

    def _create_empty_data(self):
        observations = {}
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            observations[obs_type] = np.zeros((self.num_agents, self.num_td_steps, *shape), dtype=np.float32)
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
            new_td_indices = np.arange(self.num_td_steps)
        elif isinstance(td_indices, int):
            new_td_indices = np.array([td_indices])
        elif isinstance(td_indices, slice):
            new_td_indices = np.arange(self.num_td_steps)[td_indices]
        elif isinstance(td_indices, list) or isinstance(td_indices, np.ndarray):
            new_td_indices = np.array(td_indices)
        else:
            raise TypeError("Invalid type for td_indices. Must be int, slice, list, numpy array, or None.")

        new_num_agents = len(new_agent_indices)
        new_num_td_steps = len(new_td_indices)

        new_observation = EnvObservation(self.obs_shapes, self.obs_types, num_agents=new_num_agents, num_td_steps=new_num_td_steps)
        for obs_type in self.obs_types:
            new_observation.data[obs_type] = self.data[obs_type][new_agent_indices, new_td_indices]
        new_observation.mask = self.mask[new_agent_indices, new_td_indices]

        return new_observation
    
    def __setitem__(self, agent_indices, values):
        for obs_type in self.obs_types:
            self.data[obs_type][agent_indices] = values[obs_type]
    
    def reset(self):
        self.data = self._create_empty_data()
                        
    def shift(self, term_agents, dec_agents=None):
        """
        Shift the data to the left for 'dec_agents' and handle 'term_agents' by applying mask.
        :param term_agents: numpy.ndarray, indices of agents that terminated
        :param dec_agents: numpy.ndarray or None, indices of agents that made a decision; if None, all agents are considered
        """
        assert isinstance(term_agents, np.ndarray), "'term_agents' must be a NumPy ndarray"
        if dec_agents is not None:
            assert isinstance(dec_agents, np.ndarray), "'dec_agents' must be a NumPy ndarray or None"

        all_agents = np.arange(self.num_agents)
        dec_agents = all_agents if dec_agents is None else dec_agents

        # Roll data and mask for 'dec_agents'
        for key in self.data:
            self.data[key][dec_agents] = np.roll(self.data[key][dec_agents], shift=-1, axis=1)
        self.mask[dec_agents] = np.roll(self.mask[dec_agents], shift=-1, axis=1)

        # Mask out all previous data for 'term_agents' and indicate the start of a new episode
        self.mask[term_agents, :] = 0  

        # Set the last time dimension to 0 for data of 'dec_agents' (which might include 'term_agents')
        for key in self.data:
            self.data[key][dec_agents, -1] = 0
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