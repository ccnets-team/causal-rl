import torch
from normalizer.running_mean_std import RunningMeanStd
from normalizer.running_abs_mean import RunningAbsMean

from utils.structure.data_structures  import BatchTrajectory

TRANSITION_STATE_IDX = 0
TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

REWARD_SIZE = 1

CLIP_NORM_RANGE = 10.0

def _normalize_l1_norm(advantage, padding_mask=None):
    if padding_mask is not None:
        valid_advantages = advantage * padding_mask
        sum_padding = padding_mask.sum(dim=0, keepdim=True).clamp(min=1)
        norm = valid_advantages.abs().sum(dim=0, keepdim=True) / sum_padding
    else:
        norm = advantage.abs().mean(dim=0, keepdim=True) + 1e-8
    normalized_advantage = advantage / norm
    return normalized_advantage

def _normalize_batch_norm(advantage, padding_mask=None):
    if padding_mask is not None:
        valid_advantages = advantage * padding_mask
        sum_padding = _calculate_sum_padding(padding_mask)
        batch_mean_estimated, batch_std_estimated = _estimate_batch_mean_and_std(valid_advantages, sum_padding)
        normalized_advantage = (valid_advantages - batch_mean_estimated) / batch_std_estimated
    else:
        batch_mean_estimated = advantage.mean(dim=0, keepdim=True)
        batch_std_estimated = advantage.std(dim=0, keepdim=True) + 1e-8
        normalized_advantage = (advantage - batch_mean_estimated) / batch_std_estimated
    return normalized_advantage

def _calculate_sum_padding(padding_mask):
    return padding_mask.sum(dim=0, keepdim=True).clamp(min=1)

def _estimate_batch_mean_and_std(advantages, sum_padding):
    batch_mean_estimated = advantages.sum(dim=0, keepdim=True) / sum_padding
    batch_var_estimated = ((advantages - batch_mean_estimated) ** 2).sum(dim=0, keepdim=True) / sum_padding
    batch_std_estimated = batch_var_estimated.sqrt() + 1e-8
    return batch_mean_estimated, batch_std_estimated

class NormalizerBase:
    def __init__(self, feature_size, norm_type_key, normalization_params, device):
        self.normalizer = None
        self.feature_size = feature_size
        self.clip_norm_range = CLIP_NORM_RANGE  # The range within which normalized values are clipped, preventing excessively high normalization values.

        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(feature_size, device)
        elif norm_type == "running_abs_mean":
            self.normalizer = RunningAbsMean(feature_size, device)
            
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
        self.value_manager = NormalizerBase(seq_length, 'value_normalizer', normalization_params, device=device)
        self.advantage_normalizer = normalization_params.advantage_normalizer
        self.value_normalizer = normalization_params.value_normalizer
        self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def get_reward_normalizer(self):
        return self.reward_manager.normalizer

    def get_advantage_normalizer(self):
        return self.advantage_manager.normalizer
            
    def normalize_states(self, state):
        return self.state_manager._normalize_feature(state)

    def normalize_rewards(self, reward):
        return self.reward_manager._normalize_feature(reward)
    
    def normalize_values(self, estimated_value, expected_value, padding_mask=None, normalized_value_scale = 1.0):
        """
        Applies normalization to estimated and expected values using the specified value normalizer,
        adjusting for sequence length variability. This function enhances model stability and performance by
        ensuring consistent value scaling across different sequences and environments.

        Args:
            estimated_value (torch.Tensor): The model's estimated values for each sequence.
            expected_value (torch.Tensor): The calculated expected values (returns) for each sequence.
            padding_mask (torch.Tensor, optional): A mask indicating valid entries for normalization.
            normalized_value_scale (float, optional): A scaling factor applied post-normalization to adjust the scale of normalized values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing normalized estimated and expected values.
        """
        if self.value_normalizer is None:
            return estimated_value, expected_value

        if self.value_normalizer == 'L1_norm':
            normalized_estimated_value = _normalize_l1_norm(estimated_value, padding_mask)
            normalized_expected_value = _normalize_l1_norm(expected_value, padding_mask)
        elif self.value_normalizer == 'batch_norm':
            normalized_estimated_value = _normalize_batch_norm(estimated_value, padding_mask)
            normalized_expected_value = _normalize_batch_norm(expected_value, padding_mask)
        else:
            reshaped_estimated_value = estimated_value.squeeze(-1).unsqueeze(1)
            reshaped_expected_value = expected_value.squeeze(-1).unsqueeze(1)
            if padding_mask is None:
                reshaped_padding_mask = None
            else:
                reshaped_padding_mask = padding_mask.squeeze(-1).unsqueeze(1)
            self.value_manager._update_normalizer(reshaped_expected_value, reshaped_padding_mask)

            normalized_reshaped_estimated_value = self.value_manager._normalize_feature(reshaped_estimated_value)
            normalized_estimated_value = normalized_reshaped_estimated_value.squeeze(1).unsqueeze(-1)

            normalized_reshaped_expected_value = self.value_manager._normalize_feature(reshaped_expected_value)
            normalized_expected_value = normalized_reshaped_expected_value.squeeze(1).unsqueeze(-1)
            
        normalized_estimated_value *= normalized_value_scale
        normalized_expected_value *= normalized_value_scale
        
        return normalized_estimated_value, normalized_expected_value

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
            return _normalize_l1_norm(advantage, padding_mask)
        elif self.advantage_normalizer == 'batch_norm':
            return _normalize_batch_norm(advantage, padding_mask)
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
        trajectories.state = self.normalize_states(trajectories.state)
        trajectories.next_state = self.normalize_states(trajectories.next_state)
        trajectories.reward = self.normalize_rewards(trajectories.reward)
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