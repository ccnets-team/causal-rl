def _init_obs(obs_shapes):
    obs_types = [0]*len(obs_shapes)
    vector_sizes = []  
    image_shapes = []  
    for idx, shape in enumerate(obs_shapes):
        if len(shape) == 1:
            obs_types[idx] = f'vector_{idx}'
            vector_sizes.append(shape[0])
        elif len(shape) == 3:
            obs_types[idx] = f'image_{idx}'
            image_shapes.append(shape)
    return obs_types, vector_sizes, image_shapes

class ObservationSpecs:
    def __init__(self, obs_shapes):
        # Ensure obs_shapes is in list format
        if not isinstance(obs_shapes, list):
            obs_shapes = [obs_shapes]

        # Standardize each shape into tuple format
        self.obs_shapes = self._standardize_shapes(obs_shapes)
        self.obs_types, self.vector_sizes, self.image_shapes = _init_obs(self.obs_shapes)

    @staticmethod
    def _standardize_shapes(shapes):
        standardized_shapes = []
        for shape in shapes:
            if isinstance(shape, int):
                standardized_shapes.append((shape,))
            elif isinstance(shape, (list, tuple)):
                standardized_shapes.append(tuple(shape))
            else:
                raise ValueError(f"Unsupported shape type: {type(shape)}")
        return standardized_shapes

class StateSpecs:
    def __init__(self, vector_sizes, image_shapes, state_low = None, state_high = None):
        self.state_size = sum(vector_sizes)
        self.state_low = state_low
        self.state_high = state_high
    
class ActionSpecs:
    def __init__(self, continuous_action, discrete_action, action_low = None, action_high = None):
        self.use_discrete = discrete_action > 0
        self.action_size = discrete_action + continuous_action
        self.continuous_action = continuous_action
        self.discrete_action = discrete_action
        self.action_low = action_low
        self.action_high = action_high

class EnvConfig(ObservationSpecs, ActionSpecs, StateSpecs):
    def __init__(self, env_name, env_type, num_environments, num_agents, obs_shapes, continuous_action, discrete_action, state_low = None, state_high = None, action_low = None, action_high = None):
        # Initialize ObsSpecs
        ObservationSpecs.__init__(self, obs_shapes)

        # Initialize StateSpecs
        StateSpecs.__init__(self, self.vector_sizes, self.image_shapes, state_low, state_high)
        
        # Initialize ActionSpecs
        ActionSpecs.__init__(self, continuous_action, discrete_action, action_low, action_high)
        
        self.env_type = env_type
        self.env_name = env_name
        self.num_agents = num_agents
        self.num_environments = num_environments
        self.num_test_environments = 1
        self.samples_per_step = int(self.num_agents * self.num_environments)  
