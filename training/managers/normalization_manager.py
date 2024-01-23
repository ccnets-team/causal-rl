import torch
from preprocessing.normalizer.running_mean_std import RunningMeanStd
from preprocessing.normalizer.running_abs_mean import RunningAbsMean
from preprocessing.normalizer.exponential_moving_mean_var import ExponentialMovingMeanVar
from preprocessing.normalizer.exponential_moving_abs_mean import ExponentialMovingAbsMean

import numpy as np
from utils.structure.trajectories  import BatchTrajectory

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

EXPONENTIAL_MOVING_ALPHA = 2e-3
CLIP_NORM_RANGE = 10.0

class NormalizerBase:
    def __init__(self, vector_size, norm_type_key, normalization_params, device):
        self.normalizer = None
        self.vector_size = vector_size
        self.exponential_moving_alpha = EXPONENTIAL_MOVING_ALPHA  # Alpha value for exponential moving average, used in 'exponential_moving_mean_var' normalizer to determine weighting of recent data.
        self.clip_norm_range = CLIP_NORM_RANGE  # The range within which normalized values are clipped, preventing excessively high normalization values.

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(vector_size, device)
        elif norm_type == "running_abs_mean":
            self.normalizer = RunningAbsMean(vector_size, device)
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
            # reshaped_data = data.view(-1, data.shape[-1])

            # Update the normalizer with the reshaped data
            self.normalizer.update(data)
                
    def _normalize_data(self, data):
        if self.normalizer is not None:
            # Reshape data: Merge all dimensions except the last into the first dimension
            original_shape = data.shape
            data = data.view(-1, original_shape[-1])
            # data = data.view(-1, original_shape[-1])

            # Normalize and clamp data
            clip = self.clip_norm_range
            data = self.normalizer.normalize(data)
            data.clamp_(-clip, clip)

            # Reshape data back to its original shape
            data = data.view(*original_shape)

        return data
    
class NormalizationUtils:
    def __init__(self, env_config, normalization_params, max_seq_length, device):
        self.state_manager = NormalizerBase(env_config.state_size, 'state_normalizer', normalization_params, device=device)
        self.reward_manager = NormalizerBase(1, 'reward_normalizer', normalization_params, device=device)
        self.advantage_manager = NormalizerBase(max_seq_length, 'advantage_normalizer', normalization_params, device=device)
        self.advantage_normalizer = normalization_params.advantage_normalizer
        self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def get_reward_normalizer(self):
        return self.reward_manager.normalizer

    def get_advantage_normalizer(self):
        return self.advantage_manager.normalizer
            
    def normalize_state(self, state):
        return self.state_manager._normalize_data(state)

    def normalize_reward(self, reward):
        return self.reward_manager._normalize_data(reward)

    def normalize_advantage(self, advantage):
        """Normalize the returns based on the specified normalizer type."""
        if self.advantage_normalizer is not None:
            _advantage = advantage.squeeze(-1).unsqueeze(1)
            self.advantage_manager._update_normalizer(_advantage)
            _normalized_advantage = self.advantage_manager._normalize_data(_advantage)
            normalized_advantage = _normalized_advantage.squeeze(1).unsqueeze(-1)
        return normalized_advantage

    def normalize_trajectories(self, trajectories: BatchTrajectory):
        trajectories.state = self.normalize_state(trajectories.state)
        trajectories.next_state = self.normalize_state(trajectories.next_state)
        trajectories.reward = self.normalize_reward(trajectories.reward)
        return trajectories

    def update_normalizer(self, trajectories: BatchTrajectory):
        states, actions, rewards, next_states, dones = trajectories

        # Update state normalizer
        for index in self.state_indices:
            if index == TRANSITION_STATE_IDX:
                state_data = states
            elif index == TRANSITION_NEXT_STATE_IDX:
                state_data = next_states
            else:
                raise ValueError("Invalid state index")
            self.state_manager._update_normalizer(state_data)

        # Update reward normalizer
        for index in self.reward_indices:
            if index == TRANSITION_REWARD_IDX:
                reward_data = rewards
            else:
                raise ValueError("Invalid reward index")
            self.reward_manager._update_normalizer(reward_data)