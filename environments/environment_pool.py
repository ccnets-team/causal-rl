import numpy as np
from .mlagents_wrapper import MLAgentsEnvWrapper
from .gym_wrapper import GymEnvWrapper
from utils.structure.trajectories  import MultiEnvTrajectories
import torch

class EnvironmentPool: 
    def __init__(self, env_config, gpt_seq_length, device, test_env, use_graphics):
        super(EnvironmentPool, self).__init__()
        worker_num = 1 if test_env else env_config.num_environments
        
        w_id = 0 if test_env else 1
        w_id += 100
        self.device = device
        self.gpt_seq_length = gpt_seq_length
        # seq_exploit_factor adjusts the preference for longer sequence lengths during exploration,
        # with higher values promoting the selection of longer sequences. This factor fine-tunes
        # the balance between exploration and exploitation, ensuring that the learning process
        # is optimized for environments that require detailed sequence analysis for better
        # policy development and value estimation.
        self.seq_exploit_factor  = 2.0
        
        if env_config.env_type == "gym":
            self.env_list = [GymEnvWrapper(env_config, gpt_seq_length, test_env, use_graphics = use_graphics, seed= int(w_id + i)) \
                for i in range(worker_num)]
            
        elif env_config.env_type == "mlagents":
            self.env_list = [MLAgentsEnvWrapper(env_config, gpt_seq_length, test_env, use_graphics = use_graphics, \
                worker_id = int(w_id + i), seed= int(w_id + i)) \
                for i in range(worker_num)]
            
    def reset(self):
        for it in self.env_list:
            it.reset_environment()  

    def end(self):
        for it in self.env_list:
            it.env.close()

    def fetch_env(self):
        combined_transition = MultiEnvTrajectories()

        for env_idx, env in enumerate(self.env_list):
            agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated = env.output_transitions()
            combined_transition.add([env_idx] * len(agent_ids), agent_ids, obs, action, reward, next_obs, done_terminated, done_truncated)
        return combined_transition

    def step_env(self):
        for env in self.env_list:
            env.step_environment()

    def sample_sequence_length(self, batch_size, min_seq_length, max_seq_length, exploration_rate):
        # Create an array of possible sequence lengths
        possible_lengths = np.arange(min_seq_length, max_seq_length + 1)
        
        # Calculate a linearly changing ratio across the sequence lengths
        bias_ratio = possible_lengths/max_seq_length
        
        # Adjust the gradient weight based on the exploration rate
        gradient_biased_weights = np.power(bias_ratio, self.seq_exploit_factor*(1 - exploration_rate))
        
        # Normalize the weights to sum to 1
        weights = gradient_biased_weights / gradient_biased_weights.sum()

        # Sample sequence lengths based on these normalized weights
        sampled_lengths = np.random.choice(possible_lengths, size=batch_size, p=weights)
        return sampled_lengths
    
    def apply_effective_sequence_mask(self, padding_mask, exploration_rate):
        """
        Applies an effective sequence mask to the given padding mask based on 
        the exploration rate and random sequence lengths.
        """
        min_seq_length = 1
        max_seq_length = self.gpt_seq_length
        batch_size = padding_mask.size(0)
        
        random_seq_lengths = self.sample_sequence_length(batch_size, min_seq_length, max_seq_length, exploration_rate)

        effective_seq_length = torch.clamp(torch.tensor(random_seq_lengths, device=self.device), min_seq_length, max_seq_length)

        padding_seq_length = padding_mask.size(1) - effective_seq_length
        # Create a range tensor and apply the mask
        range_tensor = torch.arange(padding_mask.size(1), device=self.device).expand_as(padding_mask)
        mask_indices = range_tensor < padding_seq_length.unsqueeze(1)
        padding_mask[mask_indices] = 0.0
        
    def explore_env(self, trainer, training):
        trainer.set_train(training = training)
        np_state = np.concatenate([env.observations.to_vector() for env in self.env_list], axis=0)
        np_mask = np.concatenate([env.observations.mask for env in self.env_list], axis=0)

        state_tensor = torch.from_numpy(np_state).to(self.device)
        padding_mask = torch.from_numpy(np_mask).to(self.device)

        # In your training loop or function
        if training:
            exploration_rate = trainer.get_exploration_rate()
            self.apply_effective_sequence_mask(padding_mask, exploration_rate)
                        
        state_tensor = trainer.normalize_state(state_tensor)
        action_tensor = trainer.get_action(state_tensor, padding_mask, training=training)
            
        np_action = action_tensor.cpu().numpy()
        start_idx = 0
        for env in self.env_list:
            end_idx = start_idx + len(env.agent_dec)
            valid_action = np_action[start_idx:end_idx][env.agent_dec]

            select_valid_action = valid_action[:,-1,:]
            env.update(select_valid_action)
            start_idx = end_idx

    @staticmethod
    def create_train_environments(env_config, gpt_seq_length, device):
        return EnvironmentPool(env_config, gpt_seq_length, device, test_env=False, use_graphics = False)
    
    @staticmethod
    def create_test_environments(env_config, gpt_seq_length, device, use_graphics):
        return EnvironmentPool(env_config, gpt_seq_length, device, test_env=True, use_graphics = use_graphics)

