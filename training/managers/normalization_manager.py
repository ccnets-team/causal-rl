import torch
from preprocessing.normalizer.running_mean_std import RunningMeanStd
from preprocessing.normalizer.running_abs_mean import RunningAbsMean

import numpy as np
from utils.structure.trajectories  import BatchTrajectory

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

class NormalizerBase:
    def __init__(self, vector_size, norm_type_key, normalization_params, device, clip_norm_range=10.0):
        self.normalizer = None
        self.vector_size = vector_size
        self.clip_norm_range = clip_norm_range

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(vector_size, device)
        elif norm_type == "running_abs_mean":
            self.normalizer = RunningAbsMean(vector_size, device)
            
        self.device = device
                    
    def _update_normalizer(self, data):
        if self.normalizer is not None:
            # Convert to a PyTorch tensor if it's a NumPy array
            if not isinstance(data, torch.Tensor):
                data = torch.FloatTensor(data).to(self.device)
            elif data.device != self.device:
                data = data.to(self.device)

            # Reshape the data for the normalizer update
            first_seq_data = data.view(-1, *data.shape[2:])

            # Update the normalizer
            self.normalizer.update(first_seq_data)
            
    def normalize_data(self, data):
        if self.normalizer is not None:
            clip = self.clip_norm_range
            data = self.normalizer.normalize(data)
            data.clamp_(-clip, clip)
        return data
    
class NormalizationUtils:
    def __init__(self, env_config, normalization_params, model_seq_length, device):
        self.state_manager = NormalizerBase(env_config.state_size, 'state_normalizer', normalization_params, device=device)
        self.reward_manager = NormalizerBase(1, 'reward_normalizer', normalization_params, device=device)
        self.advantage_manager = NormalizerBase(model_seq_length, 'advantage_normalizer', normalization_params, device=device)
        self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def normalize_state(self, state):
        return self.state_manager.normalize_data(state)

    def normalize_reward(self, reward):
        return self.reward_manager.normalize_data(reward)

    def normalize_advantage(self, advantage):
        normalized_advantage = self.advantage_manager.normalize_data(advantage)
        self.advantage_manager._update_normalizer(advantage.squeeze(-1).unsqueeze(0))
        return normalized_advantage
    
    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def get_reward_normalizer(self):
        return self.reward_manager.normalizer

    def get_advantage_normalizer(self):
        return self.advantage_manager.normalizer

    def transform_transition(self, trans: BatchTrajectory):
        trans.state = self.normalize_state(trans.state)
        trans.next_state = self.normalize_state(trans.next_state)
        trans.reward = self.normalize_reward(trans.reward)
        return trans

    def update_normalizer(self, samples):
        for index in self.state_indices:
            data = np.stack([sample[index] for sample in samples], axis=0)
            self.state_manager._update_normalizer(data)

        for index in self.reward_indices:
            data = np.stack([sample[index] for sample in samples], axis=0)
            self.reward_manager._update_normalizer(data)

