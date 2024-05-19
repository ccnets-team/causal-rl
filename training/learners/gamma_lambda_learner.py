import torch
import torch.nn as nn
from ..utils.value_util import compute_lambda_based_returns
from ..utils.sequence_util import create_init_lambda_sequence
from ..utils.sequence_util import DISCOUNT_FACTOR, AVERAGE_LAMBDA

class GammaLambdaLearner(nn.Module):
    def __init__(self, gamma, lambd, lambda_seq_len, device):
        super(GammaLambdaLearner, self).__init__()
        self.device = device
        self.lambda_seq_len = lambda_seq_len
    
        if gamma is None:
            gamma_init = DISCOUNT_FACTOR
        else:
            gamma_init = gamma
        if lambd is None:
            lambda_sequence_init = create_init_lambda_sequence(AVERAGE_LAMBDA, lambda_seq_len, device)
        else:
            lambda_sequence_init = create_init_lambda_sequence(lambd, lambda_seq_len, device)
            
        if not isinstance(lambda_sequence_init, torch.Tensor):
            lambda_sequence_init = torch.tensor(lambda_sequence_init, device=device, dtype=torch.float)
                
        # Create raw_gamma as an nn.Parameter
        self.raw_gamma = nn.Parameter(self._init_value_for_tanh(gamma_init * torch.tensor(1.0, device=device, dtype=torch.float)))
        
        # Ensure lambda_sequence_init is processed correctly
        self.raw_lambd = nn.Parameter(self._init_value_for_tanh(lambda_sequence_init))

        self.input_sum_reward_weights = None
        self.td_sum_reward_weights = None

    def get_gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(1e-8)
    
    def get_lambdas(self, seq_range):
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
    
    def get_sum_reward_weights(self, use_td_extension_steps = False):
        if use_td_extension_steps:
            sum_reward_weights = self.td_sum_reward_weights
        else:
            sum_reward_weights = self.input_sum_reward_weights
        return sum_reward_weights

    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        lambda_sequence = self.get_lambdas(seq_range=seq_range)
        gamma_value = self.get_gamma()
        return compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence)

    def update_sum_reward_weights(self, input_seq_len, td_extension_steps):
        # Parameters are now accessed directly from the class attributes
        total_seq_len = input_seq_len + td_extension_steps
        gamma = self.get_gamma()

        td_lambd = self.get_lambdas(seq_range=(-td_extension_steps, None))
        input_lambd = self.get_lambdas(seq_range=(-input_seq_len, None))
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
            