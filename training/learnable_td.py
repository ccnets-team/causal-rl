import torch
import torch.nn as nn

UPDATE_LEARNABLE_TD_INTERVAL = 4

class LearnableTD(nn.Module):
    def __init__(self, max_seq_len, discount_factor, advantage_lambda, device):
        super(LearnableTD, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda

        discount_factor_init = discount_factor * torch.ones(1, device=self.device, dtype=torch.float)
        advantage_lambda_init = advantage_lambda * torch.ones(max_seq_len, device=self.device, dtype=torch.float)
        
        self.raw_gamma = nn.Parameter(self._init_value_for_tanh(discount_factor_init))
        self.raw_lambd = nn.Parameter(self._init_value_for_tanh(advantage_lambda_init))
        
        self.sum_reward_weights = self.update_sum_reward_weights()

    def _init_value_for_tanh(self, target):
        # Use logit function as the inverse of the sigmoid to initialize the value correctly
        return torch.atanh(target)
    
    @property
    def gamma(self):
        return torch.tanh(self.raw_gamma).clamp_min(0.0)

    @property
    def lambd(self):
        return torch.tanh(self.raw_lambd).clamp_min(0.0)

    def clip_grad_norm_(self, max_grad_norm):
        if max_grad_norm is None:
            return

        total_norm = 0.0
        parameters = [self.raw_gamma, self.raw_lambd]  # Include both parameters
        scaling_factors = [1, 1 / self.max_seq_len]  # Example scaling for raw_lambd to balance its weight

        for param, scale in zip(parameters, scaling_factors):
            if param.grad is not None:
                # Scale the norm of each parameter's gradient by the scaling factor
                param_norm = (param.grad.data.norm(2) * scale) ** 2
                total_norm += param_norm

        total_norm = (total_norm ** 0.5)  # Take the square root to get the total norm

        clip_coef = max_grad_norm / (total_norm + 1e-8)  # Adjust for division by zero

        if clip_coef < 1:
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
                    
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
            value_weights[t] = gamma * ((1 - td_lambdas[t]) + td_lambdas[t] * value_weights[t + 1].clone())
            raw_sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

        sum_reward_weights = raw_sum_reward_weights.unsqueeze(0).unsqueeze(-1)
        self.sum_reward_weights = sum_reward_weights / sum_reward_weights.mean().clamp_min(1e-8)
        return self.sum_reward_weights

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