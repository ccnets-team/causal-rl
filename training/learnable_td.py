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
        
    @property
    def gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(1e-8)

    @property
    def lambd(self):
        return LambdaGradScaler.apply(torch.tanh(self.raw_lambd).clamp_min(1e-8))

    def _init_value_for_tanh(self, target):
        # Use logit function as the inverse of the sigmoid to initialize the value correctly
        return torch.atanh(target)

    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        start_idx, end_idx = seq_range
        lambda_sequence = self.lambd[start_idx:end_idx]
        gamma_value = self.gamma
        return compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence)
        