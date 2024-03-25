import numpy as np
from utils.structure.env_observation import EnvObservation


class ReinforcementAgent: 
    def __init__(self, env_config, max_seq_len):
        self.max_seq_len = max_seq_len
        self.num_agents = env_config.num_agents
        self.use_discrete = env_config.use_discrete
        self.env_name = env_config.env_name
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.action_size = env_config.action_size
        
        self.observations = EnvObservation(self.obs_shapes, self.obs_types, self.num_agents, self.max_seq_len)
        self.agent_ids = np.array(range(self.num_agents), dtype=int)
        self.actions = np.zeros((self.num_agents, self.action_size), dtype = np.float32)
        self.content_lengths = np.zeros((self.num_agents), dtype = np.int32)
        self.agent_life = np.zeros((self.num_agents), dtype=bool)
        self.agent_dec = np.zeros((self.num_agents), dtype=bool)

        if not self.use_discrete:
            self.action_low = env_config.action_low
            self.action_high = env_config.action_high
                
        self.reset_agent()

    def reset_agent(self):
        self.actions.fill(0)
        self.content_lengths.fill(0)
        self.observations.reset()
        self.agent_life.fill(False) 
        self.agent_dec.fill(False) 