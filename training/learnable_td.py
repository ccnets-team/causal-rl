
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

    def update_sum_reward_weights(self):
        # Parameters are now accessed directly from the class attributes
        max_seq_len, gamma, td_lambda, device = self.max_seq_len, self.gamma, self.lambd, self.device
        
        # (Rest of the method remains the same as your provided code)
        # Initialize tensors for value weights and sum reward weights with zeros.
        value_weights = torch.zeros(max_seq_len + 1, dtype=torch.float, device=device)
        sum_reward_weights = torch.zeros(max_seq_len, dtype=torch.float, device=device)
        
        # Ensure the final value weight equals 1, setting up a base case for backward calculation.
        value_weights[-1] = 1

        # Backward pass to compute weights. This loop calculates the decayed weights for each timestep,
        for t in reversed(range(max_seq_len)):
            value_weights[t] = gamma * ((1 - td_lambda[t]) + td_lambda[t] * value_weights[t + 1])
            sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

        # Normalize sum reward weights to maintain a consistent scale across sequences.
        sum_reward_weights /= sum_reward_weights.mean().clamp(min=1e-8)

        # Reshape for compatibility with expected input formats.
        self.sum_reward_weights = sum_reward_weights.unsqueeze(0).unsqueeze(-1)

    def calculate_lambda_returns(self, values, rewards, dones):
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
        gamma, td_lambda = self.gamma, self.lambd
        
        # Determine the batch size and sequence length from the rewards shape
        batch_size, seq_len, _ = rewards.shape

        # Initialize lambda returns with the same shape as values
        sum_rewards = torch.zeros_like(values)
        lambda_returns = torch.zeros_like(values)

        # Set the last timestep's lambda return to the last timestep's value
        lambda_returns[:, -1:] = values[:, -1:]

        # Iterate backwards through each timestep in the sequence
        for t in reversed(range(seq_len)):
            with torch.no_grad():
                # Calculate lambda return for each timestep:
                sum_rewards[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * (td_lambda[t] * sum_rewards[:, t + 1, :])
            # Current reward + discounted future value, adjusted by td_lambda
            lambda_returns[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * ((1 -  td_lambda[t]) * values[:, t + 1, :] + td_lambda[t] * lambda_returns[:, t + 1, :].clone())

        # Remove the last timestep to align lambda returns with their corresponding states
        sum_rewards = sum_rewards[:, :-1, :]
        lambda_returns = lambda_returns[:, :-1, :]

        return lambda_returns, sum_rewards        