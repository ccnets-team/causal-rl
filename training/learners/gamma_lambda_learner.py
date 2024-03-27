import torch
import torch.nn as nn
from ..utils.value_util import compute_lambda_based_returns
from ..utils.sequence_util import create_init_lambda_sequence

LEARNABLE_TD_UPDATE_INTERVAL = 2
TARGET_TD_ERROR_SCALE = 1

DISCOUNT_FACTOR = 0.99
AVERAGE_LAMBDA = 0.9

class GammaLambdaLearner(nn.Module):
    def __init__(self, seq_len, device):
        super(GammaLambdaLearner, self).__init__()
        self.device = device
        self.discount_factor = DISCOUNT_FACTOR
        self.average_lambda = AVERAGE_LAMBDA
    
        discount_factor_init = self.discount_factor * torch.ones(1, device=self.device, dtype=torch.float)
        advantage_lambda_init = create_init_lambda_sequence(self.average_lambda, seq_len, self.device)
                
        self.raw_gamma = nn.Parameter(self._init_value_for_tanh(discount_factor_init))
        self.raw_lambd = nn.Parameter(self._init_value_for_tanh(advantage_lambda_init))

        self.input_sum_reward_weights = None
        self.td_sum_reward_weights = None

    def get_gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(1e-8)
    
    def get_lambda(self, seq_range):
        start_idx, end_idx = seq_range
        return (torch.tanh(self.raw_lambd).clamp_min(1e-8)[start_idx: end_idx]).clone()
    
    def save(self, path):
        torch.save({'raw_gamma': self.raw_gamma, 'raw_lambd': self.raw_lambd}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.raw_gamma = checkpoint['raw_gamma']
        self.raw_lambd = checkpoint['raw_lambd']
        
    def _init_value_for_tanh(self, target):
        # Use logit function as the inverse of the sigmoid to initialize the value correctly
        return torch.atanh(target)

    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        lambda_sequence = self.get_lambda(seq_range=seq_range)
        gamma_value = self.get_gamma()
        return compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence)
    
    def get_sum_reward_weights(self, use_td_extension_steps = False):
        if use_td_extension_steps:
            sum_reward_weights = self.td_sum_reward_weights
        else:
            sum_reward_weights = self.input_sum_reward_weights
        return sum_reward_weights

    def update_sum_reward_weights(self, input_seq_len, td_extension_steps):
        # Parameters are now accessed directly from the class attributes
        total_seq_len = input_seq_len + td_extension_steps
        gamma = self.get_gamma()

        td_lambd = self.get_lambda(seq_range=(-td_extension_steps, None))
        input_lambd = self.get_lambda(seq_range=(-input_seq_len, None))
        device = self.device
        
        # (Rest of the method remains the same as your provided code)
        # Initialize tensors for value weights and sum reward weights with zeros.
        value_weights = torch.zeros(total_seq_len + 1, dtype=torch.float, device=device)
        raw_sum_reward_weights = torch.zeros(total_seq_len, dtype=torch.float, device=device)
        
        # Ensure the final value weight equals 1, setting up a base case for backward calculation.
        value_weights[-1] = 1
        
        with torch.no_grad():
            lambd = torch.cat([input_lambd, td_lambd], dim=0)
            # Backward pass to compute weights. This loop calculates the decayed weights for each timestep,
            for t in reversed(range(total_seq_len)):
                value_weights[t] = gamma * ((1 - lambd[t]) + lambd[t] * value_weights[t + 1].clone())
                raw_sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)
            
            # normalized_sum_reward_weights = raw_sum_reward_weights/raw_sum_reward_weights.mean().clamp_min(1e-8)
            raw_sum_reward_weights = raw_sum_reward_weights.unsqueeze(0).unsqueeze(-1)
            normalized_sum_reward_weights = raw_sum_reward_weights/raw_sum_reward_weights.mean().clamp_min(1e-8)
            self.input_sum_reward_weights = normalized_sum_reward_weights[:,:input_seq_len]
            self.td_sum_reward_weights = normalized_sum_reward_weights[:,input_seq_len:]
            