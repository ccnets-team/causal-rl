import numpy as np
import copy

class EnvObservations:
    def __init__(self, obs_shapes, obs_types, num_agents):
        assert len(obs_shapes) == len(obs_types), "The length of obs_shapes and obs_types must be the same."
        self.obs_shapes = obs_shapes
        self.obs_types = obs_types
        self.num_agents = num_agents
        self.data = self._create_empty_data()

    def copy(self):
        # Create a new instance of the same class
        new_instance = self.__class__(self.obs_shapes, self.obs_types, self.num_agents)
        
        # Copy the observation data over
        for obs_type in self.obs_types:
            new_instance.data[obs_type] = copy.copy(self.data[obs_type])

        return new_instance
    
    def _create_empty_data(self):
        observations = {}
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            observations[obs_type] = np.zeros((self.num_agents, *shape), dtype=np.float32)
        return observations
        
    def __getitem__(self, agent_indices):
        """Retrieve the observations for the given list of agent indices across all observation types."""
        if isinstance(agent_indices, int):
            # If a single integer is provided, wrap it in a list for consistent processing
            agent_indices = [agent_indices]
        
        if any(idx < 0 or idx >= self.num_agents for idx in agent_indices):
            raise IndexError(f"One or more agent indices out of bounds.")
        
        # Create a new AgentObservation instance with the reduced number of agents
        new_num_agentss = len(agent_indices)
        new_observation = EnvObservations(self.obs_shapes, self.obs_types, num_agents=new_num_agentss)
        
        for obs_type in self.obs_types:
            new_observation.data[obs_type] = self.data[obs_type][agent_indices]
        
        return new_observation

    def __setitem__(self, agent_indices, values):
        """Update the observations for the provided agent indices."""
        if isinstance(agent_indices, int):
            # If a single integer is provided, wrap it in a list for consistent processing
            agent_indices = [agent_indices]

        if any(idx < 0 or idx >= self.num_agents for idx in agent_indices):
            raise IndexError(f"One or more agent indices out of bounds.")

        # Ensure values is a dictionary and has all obs_types
        if not isinstance(values, dict):
            raise ValueError("Expected a dictionary of observations.")

        for obs_type in self.obs_types:
            if obs_type not in values:
                raise KeyError(f"Observation type {obs_type} missing in provided values.")
            
            # Get the shape for the current obs_type
            expected_shape = self.obs_shapes[self.obs_types.index(obs_type)]
            
            # Check the shape of the provided values for the obs_type
            if values[obs_type].shape[1:] != expected_shape:
                raise ValueError(f"Expected shape {expected_shape} for {obs_type}, but got {values[obs_type].shape[1:]}")

            self.data[obs_type][agent_indices] = values[obs_type]
    
    def reset(self):
        """Reset all observations to zero."""
        self.data = self._create_empty_data()
            
    def to_vector(self):
        """
        Convert the observations to a vector format.
        Only takes into account obs_types that are vectors after the num_agents dimension.
        Resulting shape will be (num_agents, concatenated_data)
        """
        vectors = []
        for obs_type, shape in zip(self.obs_types, self.obs_shapes):
            # Checking if the shape after num_agents is 1D
            if len(shape) == 1:
                data = self.data[obs_type]
                vectors.append(data)
        
        concatenated_data = np.concatenate(vectors, axis=1)  # Concatenate along the data dimension
        return concatenated_data