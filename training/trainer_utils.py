
import torch

def create_mask_from_dones(dones: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask where the initial columns are ones and subsequent columns are 
    the inverse of the `dones` tensor shifted by one.

    Args:
    - dones (torch.Tensor): The tensor based on which the mask is created.

    Returns:
    - mask (torch.Tensor): The resultant mask tensor.
    """
    mask = torch.ones_like(dones)

    if mask.size(1) > 1:
        # Reverse 'dones' along the specified axis (axis=1)
        reversed_dones = torch.flip(dones, dims=[1])

        # Perform cumulative sum on the reversed tensor
        cumulative_dones_reversed = torch.cumsum(reversed_dones[:,1:], dim=1)

        # Reverse the result back to get the cumulative sum in the original order
        cumulative_dones = torch.flip(cumulative_dones_reversed, dims=[1])
        
        mask[:, :-1, :] = 1 - cumulative_dones
    
    return mask

def shift_left_mask(mask: torch.Tensor) -> torch.Tensor:
    shifted_mask = torch.cat([torch.ones_like(mask[:, 1:, :]), mask[:, -1:, :]], dim=1)
    return shifted_mask

def masked_mean(tensor, mask):
    return tensor[mask>0].flatten().mean()

def masked_sum(tensor, mask):
    return tensor[mask>0].flatten().sum()

def calculate_accumulative_rewards(rewards, discount_factor, mask):
    batch_size, seq_len, _ = rewards.shape
    # Initialize a tensor for accumulative rewards with zeros
    accumulative_rewards = torch.zeros_like(rewards)

    # Loop backwards through the sequence
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # If it's the last step, the accumulative reward is just the immediate reward
            accumulative_rewards[:, t, :] = rewards[:, t, :]* mask[:, t, :]
        else:
            # Accumulate reward at step t with the discounted reward at t+1, but only where the mask is true
            accumulative_rewards[:, t, :] = (rewards[:, t, :] + discount_factor * accumulative_rewards[:, t+1, :])* mask[:, t, :]

    return accumulative_rewards

def compute_end_step(dones):
    # Add an extra column of ones to handle cases where an episode doesn't terminate
    padded_dones = torch.cat([dones, torch.ones_like(dones[:, :1])], dim=1)
    
    # Compute the first occurrence of `done` (1) or the padding (if no `done` occurred)
    end_step = (padded_dones.cumsum(dim=1) == 1).float().argmax(dim=1).squeeze(-1)
    
    return end_step

def get_end_future_value(future_values, end_step):
    # Create a tensor for batch indices [0, 1, 2, ..., batch_size-1]
    batch_indices = torch.arange(end_step.size(0), device=future_values.device)

    # Subtract 1 from end_step to convert to 0-based index, ensure it's a long tensor for indexing
    seq_indices = end_step.long() - 1

    # Use advanced indexing to select the corresponding future value for each sequence in the batch
    future_value_at_end_step = future_values[batch_indices, seq_indices]

    return future_value_at_end_step.unsqueeze(-1)

def compute_discounted_future_value(discount_factor, max_seq_len):
    # Create a range tensor [0, 1, 2, ..., max_seq_len-1]
    discount_exponents = max_seq_len - torch.arange(max_seq_len).unsqueeze(0)

    # Compute the discount factors by raising to the power of the exponents
    discount_factors = discount_factor ** discount_exponents

    # Return the discount factors with an additional dimension to match the expected shape
    return discount_factors.unsqueeze(-1)

def compute_gae(values, rewards, dones, gamma, tau=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
    - values (torch.Tensor): Estimated values with shape [batch_size, num_td_steps+1, 1].
    - rewards (torch.Tensor): Observed rewards with shape [batch_size, num_td_steps, 1].
    - dones (torch.Tensor): Done flags (1 if terminal state, else 0) with shape [batch_size, num_td_steps, 1].
    - gamma (float): Discount factor.
    - tau (float): GAE parameter for bias-variance trade-off.

    Returns:
    - advantages (torch.Tensor): Computed advantages with shape [batch_size, num_td_steps, 1].
    """
    # Copy the inputs to avoid modifying original tensors
    # Prepare tensor for advantages
    advantages = torch.zeros_like(rewards)
    gae = 0  # Initialize GAE

    # Iterate through timesteps in reverse to calculate GAE
    for t in reversed(range(rewards.size(1))):
        # Calculate temporal difference error
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        # Update GAE
        gae = delta + gamma * tau * gae * (1 - dones[:, t])
        # Store computed advantage
        advantages[:, t] = gae

    return advantages