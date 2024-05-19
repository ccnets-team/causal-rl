import torch
from normalizer.running_mean_std import RunningMeanStd

from utils.structure.data_structures  import BatchTrajectory

TRANSITION_STATE_IDX = 0
# TRANSITION_NEXT_STATE_IDX = 3
TRANSITION_REWARD_IDX = 2

REWARD_SIZE = 1

STATE_NORM_SCALE = 2
REWARD_NORM_SCALE = 1
SUM_REWARD_NORM_SCALE = 1
ADVANTAGE_NORM_SCALE = 1

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

def _normalize_l2_norm(advantage, padding_mask=None):
    if padding_mask is not None:
        valid_advantages = advantage * padding_mask
        sum_padding = padding_mask.sum(dim=0, keepdim=True).clamp(min=1)
        # Calculate L2 norm: sqrt of sum of squares of valid_advantages divided by the sum of padding_mask
        norm = (valid_advantages.pow(2).sum(dim=0, keepdim=True) / sum_padding).sqrt()
    else:
        # Calculate L2 norm without padding: sqrt of mean of squares of advantage
        norm = (advantage.pow(2).mean(dim=0, keepdim=True) + 1e-8).sqrt()
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
    def __init__(self, feature_size, norm_type_key, normalization_params, scale, device):
        self.normalizer = None
        self.feature_size = feature_size
        self.clip_norm_range = CLIP_NORM_RANGE  # The range within which normalized values are clipped, preventing excessively high normalization values.
        if norm_type_key == 'advantage_normalizer':
            update_decay_rate = 1e-3  # More emphasis on recent observations for advantage normalizer
        else:
            update_decay_rate = 1e-5  # Slower decay for other types of normalizers
                
        norm_type = getattr(normalization_params, norm_type_key)
        if norm_type == "running_mean_std":
            self.normalizer = RunningMeanStd(feature_size, scale, device, decay_rate=update_decay_rate)
        self.device = device
                    
    def _update_normalizer(self, data, padding_mask=None, feature_range = None):
        if self.normalizer is not None:
            # Convert to a PyTorch tensor if it's a NumPy array and move to the specified device
            if not isinstance(data, torch.Tensor):
                data = torch.FloatTensor(data).to(self.device)
            elif data.device != self.device:
                data = data.to(self.device)
            with torch.no_grad():
                # Update the normalizer with the reshaped data
                self.normalizer.update(data, padding_mask, feature_range = feature_range)
                
    def _normalize_last_dim(self, data, scale = 1, feature_range = None):
        if self.normalizer is not None:
            # Normalize and clamp data
            clip = self.clip_norm_range * scale
            data = self.normalizer.normalize(data, feature_range = feature_range)
            data.clamp_(-clip, clip)
        return data
    
class NormalizationManager:
    def __init__(self, state_size, value_size, normalization_params, seq_len, device):
        self.state_manager = NormalizerBase(state_size, 'state_normalizer', normalization_params, STATE_NORM_SCALE, device=device)
        self.reward_manager = NormalizerBase(REWARD_SIZE, 'reward_normalizer', normalization_params, REWARD_NORM_SCALE, device=device)
        self.sum_reward_manager = NormalizerBase(seq_len, 'sum_reward_normalizer', normalization_params, SUM_REWARD_NORM_SCALE, device=device)
        self.advantage_manager = NormalizerBase(int(seq_len * value_size), 'advantage_normalizer', normalization_params, ADVANTAGE_NORM_SCALE, device=device)
        self.advantage_normalizer = normalization_params.advantage_normalizer
        self.sum_reward_normalizer = normalization_params.sum_reward_normalizer
        self.state_indices = [TRANSITION_STATE_IDX]
        # self.state_indices = [TRANSITION_STATE_IDX, TRANSITION_NEXT_STATE_IDX]
        self.reward_indices = [TRANSITION_REWARD_IDX]

    def get_state_normalizer(self):
        return self.state_manager.normalizer

    def get_reward_normalizer(self):
        return self.reward_manager.normalizer

    def get_advantage_normalizer(self):
        return self.advantage_manager.normalizer
            
    def normalize_states(self, state, feature_range = None):
        return self.state_manager._normalize_last_dim(state, STATE_NORM_SCALE, feature_range)

    def normalize_rewards(self, reward):
        return self.reward_manager._normalize_last_dim(reward, REWARD_NORM_SCALE)
    
    def normalize_sum_rewards(self, sum_rewards, padding_mask=None, seq_range = None):
        """
        Applies normalization to estimated and expected values using the specified value normalizer,    
        adjusting for sequence length variability. 

        Args:
            sum_rewards (torch.Tensor): The calculated expected values (returns) for each sequence.
            padding_mask (torch.Tensor, optional): A mask indicating valid entries for normalization.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing normalized estimated and expected values.
        """
        if self.sum_reward_normalizer is None:
            return sum_rewards

        if self.sum_reward_normalizer == 'L1_norm':
            normalized_sum_rewards = _normalize_l1_norm(sum_rewards, padding_mask)
        elif self.sum_reward_normalizer == 'L2_norm':
            normalized_sum_rewards = _normalize_l2_norm(sum_rewards, padding_mask)
        elif self.sum_reward_normalizer == 'batch_norm':
            normalized_sum_rewards = _normalize_batch_norm(sum_rewards, padding_mask)
        else:
            reshaped_sum_rewards = sum_rewards.squeeze(-1).unsqueeze(1)
            if padding_mask is None:
                reshaped_padding_mask = None
            else:
                reshaped_padding_mask = padding_mask.squeeze(-1).unsqueeze(1)
            self.sum_reward_manager._update_normalizer(reshaped_sum_rewards, reshaped_padding_mask, feature_range = seq_range)

            normalized_reshaped_sum_rewards = self.sum_reward_manager._normalize_last_dim(reshaped_sum_rewards, SUM_REWARD_NORM_SCALE, feature_range = seq_range)
            normalized_sum_rewards = normalized_reshaped_sum_rewards.squeeze(1).unsqueeze(-1)
        
        return normalized_sum_rewards

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
        elif self.sum_reward_normalizer == 'L2_norm':
            return _normalize_l2_norm(advantage, padding_mask)
        elif self.advantage_normalizer == 'batch_norm':
            return _normalize_batch_norm(advantage, padding_mask)
        else:
            ori_advantage_shape = advantage.shape
            new_advantage_shape = (advantage.size(0), 1, -1)
            reshaped_advantage = advantage.reshape(new_advantage_shape)
            if padding_mask is None:
                reshaped_padding_mask = None
            else:
                reshaped_padding_mask = padding_mask.expand_as(advantage)
                reshaped_padding_mask = reshaped_padding_mask.reshape(new_advantage_shape)
            self.advantage_manager._update_normalizer(reshaped_advantage, reshaped_padding_mask)
            normalized_reshaped_advantage = self.advantage_manager._normalize_last_dim(reshaped_advantage, ADVANTAGE_NORM_SCALE)
            normalized_advantage = normalized_reshaped_advantage.reshape(ori_advantage_shape)
            return normalized_advantage

    def normalize_trajectories(self, trajectories: BatchTrajectory, feature_range = None):
        trajectories.trajectory_states = self.normalize_states(trajectories.trajectory_states, feature_range = feature_range)
        trajectories.reward = self.normalize_rewards(trajectories.reward)
        return trajectories

    def update_normalizer(self, trajectories: BatchTrajectory):
        states, _, rewards, _ = trajectories
        # Update state normalizer
        for index in self.state_indices:
            if index == TRANSITION_STATE_IDX:
                state_data = states
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