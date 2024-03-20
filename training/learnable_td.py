import torch
import torch.nn as nn
from .utils.value_util import compute_lambda_based_returns

UPDATE_LEARNABLE_TD_INTERVAL = 2

class LearnableTD(nn.Module):
    def __init__(self, max_seq_len, discount_factor, advantage_lambda, device):
        super(LearnableTD, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda
    
        discount_factor_init = discount_factor * torch.ones(1, device=self.device, dtype=torch.float)
        advantage_lambda_init = advantage_lambda * torch.ones(max_seq_len, device=self.device, dtype=torch.float)
        advantage_lambda_init[-1] = 1.0
        
        self.raw_gamma = nn.Parameter(self._init_value_for_tanh(discount_factor_init))
        self.raw_lambd = nn.Parameter(self._init_value_for_tanh(advantage_lambda_init))
        
        self.sum_reward_weights = None
        
    @property
    def gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(0.0)

    @property
    def lambd(self):
        return torch.tanh(self.raw_lambd).clamp_min(0.0)

    def _init_value_for_tanh(self, target):
        # Use logit function as the inverse of the sigmoid to initialize the value correctly
        return torch.atanh(target)

    def get_sum_reward_weights(self, seq_range, padding_mask=None):
        # Extract the start and end index from the sequence range
        start_idx, end_idx = seq_range
        
        # Select the relevant portion of sum reward weights based on the sequence range
        sum_reward_weights = self.sum_reward_weights[:, start_idx:end_idx]

        if padding_mask is None:
            return sum_reward_weights
        else:
            # Adjust the sum reward weights based on the padding mask
            masked_sum_reward_weights = sum_reward_weights * padding_mask
            # Normalize the masked sum reward weights relative to their original sum, adjusted for the valid (non-padded) parts
            normalization_factor = masked_sum_reward_weights.sum(dim=1, keepdim=True) / sum_reward_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            adjusted_sum_reward_weights = masked_sum_reward_weights / normalization_factor.clamp(min=1e-8)
        
        return adjusted_sum_reward_weights

    def calculate_sum_reward_weights(self):
        # Parameters are now accessed directly from the class attributes
        max_seq_len, gamma, td_lambdas, device = self.max_seq_len, self.gamma, self.lambd, self.device
        
        # (Rest of the method remains the same as your provided code)
        # Initialize tensors for value weights and sum reward weights with zeros.
        value_weights = torch.zeros(max_seq_len + 1, dtype=torch.float, device=device)
        raw_sum_reward_weights = torch.zeros(max_seq_len, dtype=torch.float, device=device)
        
        # Ensure the final value weight equals 1, setting up a base case for backward calculation.
        value_weights[-1] = 1

        # Backward pass to compute weights. This loop calculates the decayed weights for each timestep,
        for t in reversed(range(max_seq_len)):
            value_weights[t] = gamma * ((1 - td_lambdas[t]) + td_lambdas[t] * value_weights[t + 1].clone())
            raw_sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

        return raw_sum_reward_weights.unsqueeze(0).unsqueeze(-1)

    def calculate_sum_reward_scale(self, raw_sum_reward_weights):
        sum_reward_scale = 1 / raw_sum_reward_weights.mean().clamp_min(1e-8)
        return sum_reward_scale

    def update_sum_reward_weights(self):
        raw_sum_reward_weights = self.calculate_sum_reward_weights()
        normalized_reward_scale = self.calculate_sum_reward_scale(raw_sum_reward_weights)
        self.sum_reward_weights = normalized_reward_scale * raw_sum_reward_weights
    
    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        start_idx, end_idx = seq_range
        lambda_sequence = self.lambd[start_idx:end_idx]
        gamma_value = self.gamma
        return compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence)
        