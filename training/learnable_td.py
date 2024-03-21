import torch
import torch.nn as nn
from .utils.value_util import compute_lambda_based_returns

class LearnableTD(nn.Module):
    def __init__(self, max_seq_len, discount_factor, advantage_lambda, device):
        super(LearnableTD, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda
    
        discount_factor_init = discount_factor * torch.ones(1, device=self.device, dtype=torch.float)
        advantage_lambda_init = self._create_init_lambda_sequence(advantage_lambda, max_seq_len, device)
                
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

    def _create_init_lambda_sequence(self, target_mean, sequence_length, target_device):
        # Initialize lambda_sequence with zeros on target_device
        lambda_sequence = torch.zeros(sequence_length, device=target_device, dtype=torch.float)

        # Calculate initial and final values for the lambda sequence, considering target_mean
        initial_value = 2 * target_mean - 1
        final_value = 1  # Intended to ensure the last lambda value is 1

        # Adjust the sequence starting from 0 if initial_value is negative
        if initial_value < 0:
            initial_value = 0
            # Adjust final_value to maintain the mean, keeping in mind the explicit setting of the last value to 1
            final_value = 2 * target_mean

        # Generate a tensor that linearly progresses from initial_value to final_value
        # Adjust to fill all but the last value of lambda_sequence with the linear progression
        lambda_sequence[:-1] = torch.linspace(initial_value, final_value, steps=sequence_length - 1, device=target_device, dtype=torch.float)
        lambda_sequence[-1] = 1.0  # Explicitly set the last value to 1

        return lambda_sequence
    
    def get_sum_reward_weights(self, seq_range):
        # Extract the start and end index from the sequence range
        start_idx, end_idx = seq_range
        
        # Select the relevant portion of sum reward weights based on the sequence range
        sum_reward_weights = self.sum_reward_weights[:, start_idx:end_idx]
        
        return sum_reward_weights

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
        