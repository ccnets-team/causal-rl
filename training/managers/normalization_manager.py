import torch
from preprocessing.normalizer.running_mean_std import RunningMeanStd
from preprocessing.normalizer.exponential_moving_mean_var import ExponentialMovingMeanVar

import numpy as np
from utils.structure.trajectories  import BatchTrajectory

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

class NormalizerBase:
    def __init__(self, vector_size, norm_type_key, normalization_params, device):
        self.normalizer = None
        self.vector_size = vector_size
        self.exponential_moving_alpha = normalization_params.exponential_moving_alpha 
        self.clip_norm_range = normalization_params.clip_norm_range 

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(vector_size, device)
        elif norm_type == "exponential_moving_mean_var":
            self.normalizer = ExponentialMovingMeanVar(vector_size, device, alpha = self.exponential_moving_alpha)
            
        self.device = device
                    
    def _update_normalizer(self, data):
        if self.normalizer is not None:
            # Convert to a PyTorch tensor if it's a NumPy array and move to the specified device
            if not isinstance(data, torch.Tensor):
                data = torch.FloatTensor(data).to(self.device)
            elif data.device != self.device:
                data = data.to(self.device)

            # Reshape the data: Merge all dimensions except the last into the first dimension
            # This matches the reshaping logic used in normalize_data
            reshaped_data = data.view(-1, data.shape[-1])

            # Update the normalizer with the reshaped data
            self.normalizer.update(reshaped_data)
                
    def normalize_data(self, data):
        if self.normalizer is not None:
            # Reshape data: Merge all dimensions except the last into the first dimension
            original_shape = data.shape
            data = data.view(-1, original_shape[-1])

            # Normalize and clamp data
            clip = self.clip_norm_range
            data = self.normalizer.normalize(data)
            data.clamp_(-clip, clip)

            # Reshape data back to its original shape
            data = data.view(*original_shape)

        return data
    
class NormalizationUtils:
    def __init__(self, env_config, normalization_params, gpt_seq_length, device):
        self.state_manager = NormalizerBase(env_config.state_size, 'state_normalizer', normalization_params, device=device)
        self.reward_manager = NormalizerBase(1, 'reward_normalizer', normalization_params, device=device)
        self.advantage_manager = NormalizerBase(gpt_seq_length, 'advantage_normalizer', normalization_params, device=device)
        self.advantage_normalizer = normalization_params.advantage_normalizer

        self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def normalize_state(self, state):
        return self.state_manager.normalize_data(state)

    def normalize_reward(self, reward):
        return self.reward_manager.normalize_data(reward)
    
    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def get_reward_normalizer(self):
        return self.reward_manager.normalizer

    def get_advantage_normalizer(self):
        return self.advantage_manager.normalizer

    def normalize_advantage(self, advantage):
        """Normalize the returns based on the specified normalizer type."""
        normalizer_type = self.advantage_normalizer
        if normalizer_type is None:
            normalized_advantage = advantage
        elif normalizer_type == 'L1_norm':
            normalized_advantage = advantage / (advantage.abs().mean(dim=0, keepdim=True) + 1e-8)
        elif normalizer_type == 'batch_norm':
            # Batch normalization - normalizing based on batch mean and std
            batch_mean_estimated = advantage.mean(dim=0, keepdim=True)
            batch_std_estimated = advantage.std(dim=0, keepdim=True) + 1e-8
            normalized_advantage = (advantage - batch_mean_estimated) / batch_std_estimated
        else:
            normalized_advantage = self.advantage_manager.normalize_data(advantage.squeeze(-1)).unsqueeze(-1)
            self.update_advantage(advantage.squeeze(-1))
        return normalized_advantage

    def transform_transition(self, trans: BatchTrajectory):
        trans.state = self.normalize_state(trans.state)
        trans.next_state = self.normalize_state(trans.next_state)
        trans.reward = self.normalize_reward(trans.reward)
        return trans
    
    def update_advantage(self, advantage):
        self.advantage_manager._update_normalizer(advantage)

    def update_normalizer(self, samples):
        for index in self.state_indices:
            data = np.stack([sample[index] for sample in samples], axis=0)
            self.state_manager._update_normalizer(data)

        for index in self.reward_indices:
            data = np.stack([sample[index] for sample in samples], axis=0)
            self.reward_manager._update_normalizer(data)