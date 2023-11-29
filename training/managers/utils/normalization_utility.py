import torch
from preprocessing.normalizer.running_z_standardizer import RunningZStandardizer
import numpy as np
from utils.structure.trajectory_handler  import BatchTrajectory

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

class NormalizerBase:
    def __init__(self, vector_size, norm_type_key, normalization_params, device):
        self.normalizer = None
        self.vector_size = vector_size

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_z_standardizer":
            self.normalizer = RunningZStandardizer(vector_size, device)
        self.device = device
            
    def _update_normalizer(self, data):
        if self.normalizer is not None:
            data = torch.FloatTensor(data).to(self.device)
            first_seq_data = data.view(-1, *data.shape[2:])
            # first_seq_data = data[:, 0, ...]
            # Update the normalizer
            self.normalizer.update(first_seq_data)

    def normalize_data(self, data):
        if self.normalizer is not None:
            data = self.normalizer.normalize(data)
        return data
    
class NormalizationUtils:
    def __init__(self, env_config, normalization, device):
        self.state_manager = NormalizerBase(env_config.state_size, 'state_normalizer', normalization, device=device)
        self.reward_scale = normalization.reward_scale
        self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def normalize_state(self, state):
        return self.state_manager.normalize_data(state)

    def transform_reward(self, reward):
        return reward*self.reward_scale
    
    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def transform_transition(self, trans: BatchTrajectory):
        trans.state = self.normalize_state(trans.state)
        trans.next_state = self.normalize_state(trans.next_state)
        trans.reward = self.transform_reward(trans.reward)
        return trans
    
    def update_normalizer(self, samples):
        for index in self.state_indices:
            data = np.stack([sample[index] for sample in samples], axis=0)
            self.state_manager._update_normalizer(data)

