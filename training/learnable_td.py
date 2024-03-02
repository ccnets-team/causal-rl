
import torch
import torch.nn as nn
from .trainer_utils import GradScaler

class LearnableTD(nn.Module):
    def __init__(self, max_seq_len, discount_factor, advantage_lambda, device):
        super(LearnableTD, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda

        # Initialize raw_gamma and raw_lambda as learnable parameters
        # Starting from 0 to center the sigmoid output around 0.5
        self.raw_gamma = nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float))
        self.raw_lambd = nn.Parameter(torch.zeros(max_seq_len, device=self.device, dtype=torch.float))
        self.sum_reward_weights = None

    @property
    def gamma(self):
        # Sigmoid transformation of raw_gamma ensures gamma stays within [0, 1].
        dynamic_gamma = self.discount_factor + (1 - self.discount_factor) * (2 * torch.sigmoid(self.raw_gamma) - 1)
        return GradScaler.apply(dynamic_gamma, 1.0)  # Applies a scaling factor of 1.0, keeping gamma unchanged.

    @property
    def lambd(self):
        # Sigmoid transformation of raw_lambd, ensuring lambda is within [0, 1].
        dynamic_lambda = self.advantage_lambda + (1 - self.advantage_lambda) * (2 * torch.sigmoid(self.raw_lambd) - 1)
        return GradScaler.apply(dynamic_lambda, 0.5)  # Scales lambda by 0.5 to accelerate its learning rate.

    def get_sum_reward_weights(self, seq_range):
        start_idx, end_idx = seq_range
        return self.sum_reward_weights[:, start_idx: end_idx]

    def update_sum_reward_weights(self):
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
            value_weights[t] = gamma * ((1 - td_lambdas[t]) + td_lambdas[t] * value_weights[t + 1])
            raw_sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

        sum_reward_weights = raw_sum_reward_weights.unsqueeze(0).unsqueeze(-1)
        self.sum_reward_weights = sum_reward_weights / sum_reward_weights.mean().clamp(min=1e-8)

    def calculate_lambda_returns(self, values, rewards, dones, seq_range):
        """
        Calculates lambda returns and sum of rewards for each timestep in a sequence with variable gamma and lambda.

        Args:
            values (torch.Tensor): The value estimates for each timestep.
            rewards (torch.Tensor): The rewards received at each timestep.
            dones (torch.Tensor): Indicates whether a timestep is terminal (1 if terminal, 0 otherwise).
            gammas (torch.Tensor): Discount factors for future rewards, varying per timestep.
            td_lambdas (torch.Tensor): Lambda parameters for TD(lambda) returns, varying per timestep.

        Returns:
            tuple: A tuple containing:
                - lambda_returns (torch.Tensor): The calculated lambda returns for each timestep.
                - sum_rewards (torch.Tensor): The cumulative sum of rewards for each timestep.
        """
        start_idx, end_idx = seq_range
        gamma, td_lambdas = self.gamma, self.lambd[start_idx:end_idx]
        segment_length = end_idx - start_idx
        
        # Initialize tensors for the segment's lambda returns and sum of rewards
        sum_rewards = torch.zeros_like(values)
        lambda_returns = torch.zeros_like(values)
        
        # Initialize the last timestep's lambda return to the last timestep's value within the segment
        lambda_returns[:, -1:] = values[:, -1:]

        # Iterate backwards through the segment
        for t in reversed(range(segment_length)):
            with torch.no_grad():
                # Calculate lambda return for each timestep:
                sum_rewards[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * (td_lambdas[t] * sum_rewards[:, t + 1, :])
            # Current reward + discounted future value, adjusted by td_lambda
            lambda_returns[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * ((1 -  td_lambdas[t]) * values[:, t + 1, :] + td_lambdas[t] * lambda_returns[:, t + 1, :].clone())

        # Remove the last timestep to align sum rewards with their corresponding states.lambda_returns includes the last timestep's value, so it is not shifted.  
        sum_rewards = sum_rewards[:, :-1, :]
        
        return lambda_returns, sum_rewards        