import numpy as np
from utils.structure.env_observation import EnvObservation


class ReinforcementAgent: 
    def __init__(self, env_config):
        self.gpt_seq_length = env_config.gpt_seq_length
        self.num_agents = env_config.num_agents
        self.use_discrete = env_config.use_discrete
        self.env_name = env_config.env_name
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.action_size = env_config.action_size
        self.exploration_rate = 1.0
        
        self.observations = EnvObservation(self.obs_shapes, self.obs_types, self.num_agents, self.gpt_seq_length)
        self.agent_ids = np.array(range(self.num_agents), dtype=int)
        self.actions = np.zeros((self.num_agents, self.action_size), dtype = np.float32)
        self.agent_life = np.zeros((self.num_agents), dtype=bool)
        self.agent_dec = np.zeros((self.num_agents), dtype=bool)
        self.padding_lengths = np.zeros(self.num_agents, dtype=np.int32)

        if not self.use_discrete:
            self.action_low = env_config.action_low
            self.action_high = env_config.action_high
                
        self.reset_agent()

    def _update_env_exploration_rate(self, exploration_rate):
        self.exploration_rate = exploration_rate

    def sample_padding_lengths(self, batch_size):
        """
        Samples sequence lengths within the specified range, adjusting probabilities 
        based on the exploration rate to promote varied sequence sampling. This method 
        encourages exploration by dynamically adjusting the likelihood of selecting different 
        sequence lengths, factoring in the current exploration rate to balance between 
        exploring new lengths and exploiting known advantageous lengths.
        """
        min_seq_length = 1
        max_seq_length = self.gpt_seq_length 
        
        sequence_lengths = np.arange(min_seq_length, max_seq_length + 1)
        
        # Compute relative lengths as ratios of the maximum sequence length.
        sequence_ratios = sequence_lengths / max_seq_length
        
        adjusted_ratios = np.power(sequence_ratios, 1 / max(self.exploration_rate, 1e-8) - 1) 
        
        # Normalize adjusted ratios to get probabilities for sampling.
        sequence_probs = adjusted_ratios / adjusted_ratios.sum()
        
        # Sample sequence lengths based on the computed probabilities.
        sampled_indices = np.random.choice(sequence_lengths, size=batch_size, replace=True, p=sequence_probs)
        
        reindexed_sampled_indices = sampled_indices - min_seq_length
        
        sampled_lengths = sequence_lengths[reindexed_sampled_indices]
        
        padding_seq_length = max_seq_length - sampled_lengths
        
        # Ensure sampled lengths are within the specified range.
        return np.clip(padding_seq_length, 0, max_seq_length - 1)
            
    def reset_agent(self):
        self.actions.fill(0)
        self.observations.reset()
        self.agent_life.fill(False) 
        self.agent_dec.fill(False) 
        self.padding_lengths = self.sample_padding_lengths(self.num_agents) 