import torch
from normalizer.running_mean_std import RunningMeanStd
from normalizer.running_abs_mean import RunningAbsMean
from normalizer.exponential_moving_mean_var import ExponentialMovingMeanVar
from normalizer.hybrid_moving_mean_var import HybridMovingMeanVar

from utils.structure.trajectories  import BatchTrajectory
from training.trainer_utils import create_padding_mask_before_dones

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

REWARD_SIZE = 1

CLIP_NORM_RANGE = 10.0
EXPONENTIAL_MOVING_ALPHA = 1e-4

class NormalizerBase:
    def __init__(self, feature_size, norm_type_key, normalization_params, device):
        self.normalizer = None
        self.feature_size = feature_size
        self.clip_norm_range = CLIP_NORM_RANGE  # The range within which normalized values are clipped, preventing excessively high normalization values.
        self.exponential_moving_alpha = EXPONENTIAL_MOVING_ALPHA  # Alpha value for exponential moving average, used in 'exponential_moving_mean_var' normalizer to determine weighting of recent data.

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(feature_size, device)
        elif norm_type == "running_abs_mean":
            self.normalizer = RunningAbsMean(feature_size, device)
        elif norm_type == "exponential_moving_mean_var":
            self.normalizer = ExponentialMovingMeanVar(feature_size, device, alpha = self.exponential_moving_alpha)
        elif norm_type == "hybrid_moving_mean_var":
            self.normalizer = HybridMovingMeanVar(feature_size, device, alpha = self.exponential_moving_alpha)
            
        self.device = device
                    
    def _update_normalizer(self, data, padding_mask=None):
        if self.normalizer is not None:
            # Convert to a PyTorch tensor if it's a NumPy array and move to the specified device
            if not isinstance(data, torch.Tensor):
                data = torch.FloatTensor(data).to(self.device)
            elif data.device != self.device:
                data = data.to(self.device)
            # Update the normalizer with the reshaped data
            self.normalizer.update(data, padding_mask)
                
    def _normalize_feature(self, data):
        if self.normalizer is not None:
            # Normalize and clamp data
            clip = self.clip_norm_range
            data = self.normalizer.normalize(data)
            data.clamp_(-clip, clip)
        return data
    
class NormalizationUtils:
    def __init__(self, state_size, normalization_params, seq_length, device):
        self.state_manager = NormalizerBase(state_size, 'state_normalizer', normalization_params, device=device)
        self.reward_manager = NormalizerBase(REWARD_SIZE, 'reward_normalizer', normalization_params, device=device)
        self.advantage_manager = NormalizerBase(seq_length, 'advantage_normalizer', normalization_params, device=device)
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
        return self.state_manager._normalize_feature(state)

    def normalize_reward(self, reward):
        return self.reward_manager._normalize_feature(reward)

    def normalize_advantage(self, advantage, padding_mask=None):
        """
        Normalize advantage per sequence using an advantage normalizer.

        Treats each sequence length as a separate feature and normalizes the advantage
        for each sequence individually. If a normalizer is set, the advantage tensor 
        is reshaped to align with the normalizer's format, normalized, and then 
        reshaped back to its original structure. If no normalizer is set, the 
        original advantage values are returned.

        Args:
        - advantage (Tensor): Advantage values, with sequence length treated as a feature.
        - padding_mask (Tensor, optional): Mask to identify valid data points within each sequence.

        Returns:
        - Tensor: Advantage values normalized for each sequence if a normalizer is set; 
        otherwise, returns the unmodified advantage tensor.
        """
        if self.advantage_normalizer is None:
            return advantage
        elif self.advantage_normalizer == 'L1_norm':
            normalized_advantage = advantage / (advantage.abs().mean(dim=0, keepdim=True) + 1e-8)
        elif self.advantage_normalizer == 'batch_norm':
            # Batch normalization - normalizing based on batch mean and std
            batch_mean_estimated = advantage.mean(dim=0, keepdim=True)
            batch_std_estimated = advantage.std(dim=0, keepdim=True) + 1e-8
            normalized_advantage = (advantage - batch_mean_estimated) / batch_std_estimated
        else:
            reshaped_advantage = advantage.squeeze(-1).unsqueeze(1)
            if padding_mask is None:
                reshaped_padding_mask = None
            else:
                reshaped_padding_mask = padding_mask.squeeze(-1).unsqueeze(1)
            self.advantage_manager._update_normalizer(reshaped_advantage, reshaped_padding_mask)
            normalized_reshaped_advantage = self.advantage_manager._normalize_feature(reshaped_advantage)
            normalized_advantage = normalized_reshaped_advantage.squeeze(1).unsqueeze(-1)
        return normalized_advantage

    def normalize_trajectories(self, trajectories: BatchTrajectory):
        trajectories.state = self.normalize_state(trajectories.state)
        trajectories.next_state = self.normalize_state(trajectories.next_state)
        trajectories.reward = self.normalize_reward(trajectories.reward)
        return trajectories

    def update_normalizer(self, trajectories: BatchTrajectory):
        states, actions, rewards, next_states, dones = trajectories
        padding_mask = create_padding_mask_before_dones(dones)
        # Update state normalizer
        for index in self.state_indices:
            if index == TRANSITION_STATE_IDX:
                state_data = states
            elif index == TRANSITION_NEXT_STATE_IDX:
                state_data = next_states
            else:
                raise ValueError("Invalid state index")
            self.state_manager._update_normalizer(state_data, padding_mask)

        # Update reward normalizer
        for index in self.reward_indices:
            if index == TRANSITION_REWARD_IDX:
                reward_data = rewards
            else:
                raise ValueError("Invalid reward index")
            self.reward_manager._update_normalizer(reward_data, padding_mask)