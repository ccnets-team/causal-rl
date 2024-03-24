import torch
import torch.nn as nn
from .utils.value_util import compute_lambda_based_returns
from .utils.sequence_util import create_init_lambda_sequence
from .utils.tensor_util import LambdaGradScaler

LEARNABLE_TD_UPDATE_INTERVAL = 2

DISCOUNT_FACTOR = 0.99
AVERAGE_LAMBDA = 0.5

class LearnableTD(nn.Module):
    def __init__(self, max_seq_len, device):
        super(LearnableTD, self).__init__()
        self.device = device
        self.discount_factor = DISCOUNT_FACTOR
        self.average_lambda = AVERAGE_LAMBDA
    
        discount_factor_init = self.discount_factor * torch.ones(1, device=self.device, dtype=torch.float)
        advantage_lambda_init = create_init_lambda_sequence(self.average_lambda, max_seq_len, self.device)
                
        self.raw_gamma = nn.Parameter(self._init_value_for_tanh(discount_factor_init))
        self.raw_lambd = nn.Parameter(self._init_value_for_tanh(advantage_lambda_init))
        
        self.sum_reward_weights = None
        
    @property
    def gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(1e-8)

    @property
    def lambd(self):
        return LambdaGradScaler.apply(torch.tanh(self.raw_lambd).clamp_min(1e-8))

    def _init_value_for_tanh(self, target):
        # Use logit function as the inverse of the sigmoid to initialize the value correctly
        return torch.atanh(target)
    
    def get_sum_reward_weights(self, seq_range):
        # Extract the start and end index from the sequence range
        start_idx, end_idx = seq_range
        
        # Select the relevant portion of sum reward weights based on the sequence range
        sum_reward_weights = self.sum_reward_weights[:, start_idx:end_idx]
        
        return sum_reward_weights

    def update_sum_reward_weights(self, input_seq_len):
        # Parameters are now accessed directly from the class attributes
        gamma, td_lambdas, device = self.gamma, self.lambd[-input_seq_len:], self.device
        
        # (Rest of the method remains the same as your provided code)
        # Initialize tensors for value weights and sum reward weights with zeros.
        value_weights = torch.zeros(input_seq_len + 1, dtype=torch.float, device=device)
        raw_sum_reward_weights = torch.zeros(input_seq_len, dtype=torch.float, device=device)
        
        # Ensure the final value weight equals 1, setting up a base case for backward calculation.
        value_weights[-1] = 1
        
        with torch.no_grad():
            # Backward pass to compute weights. This loop calculates the decayed weights for each timestep,
            for t in reversed(range(input_seq_len)):
                value_weights[t] = gamma * ((1 - td_lambdas[t]) + td_lambdas[t] * value_weights[t + 1].clone())
                raw_sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

        raw_sum_reward_weights = raw_sum_reward_weights.unsqueeze(0).unsqueeze(-1)
        self.sum_reward_weights = raw_sum_reward_weights/raw_sum_reward_weights.mean().clamp_min(1e-8)

    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        start_idx, end_idx = seq_range
        lambda_sequence = self.lambd[start_idx:end_idx]
        gamma_value = self.gamma
        return compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence)
        