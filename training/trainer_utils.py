
import torch

def compute_gae(values, rewards, dones, gamma=0.99, tau=0.95):
    # Assuming rewards and dones tensors have shape [batch_size, num_td_steps, 1] 
    # and values has shape [batch_size, num_td_steps+1, 1]
    # IMPORTANT: This function assumes the value of terminal states in `values` tensor is already 0.
    
    gae = 0
    advantages = torch.zeros_like(rewards)

    # Iterate through timesteps in reverse to calculate GAE
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * values[:, t+1] * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * tau * gae * (1 - dones[:, t])
        advantages[:, t] = gae

    return advantages

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
    mask[:, 1:, :] = 1 - dones[:, :-1, :]
    
    return mask

def masked_mean(tensor, mask):
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Multiply the tensor by the mask, zeroing out the elements of the tensor where mask is False
    masked_tensor = tensor * mask

    # Sum the masked tensor across the batch dimension (dim=0)
    sum_per_sequence = torch.sum(masked_tensor, dim=0)

    # Count the number of True entries in the mask per sequence for normalization
    count_per_sequence = torch.sum(mask, dim=0)

    # Handle potential division by zero in case some sequences are fully masked
    # If count_per_sequence is 0, replace it with 1 to prevent division by zero
    count_per_sequence = torch.clamp(count_per_sequence, min=1)

    # Calculate the mean by dividing the sum by the number of unmasked entries
    mean_per_sequence = sum_per_sequence / count_per_sequence

    return mean_per_sequence

def calculate_accumulative_rewards(rewards, end_step, discount_factor):
    batch_size, seq_len, _ = rewards.shape
    # Create the mask based on end_idx to identify valid reward positions
    mask = torch.arange(seq_len).to(end_step.device).unsqueeze(0).expand(batch_size, -1) < end_step.unsqueeze(-1)
    mask = mask.unsqueeze(-1) # Expand to match rewards shape 

    # Initialize a tensor for accumulative rewards with zeros
    accumulative_rewards = torch.zeros_like(rewards)

    # Loop backwards through the sequence
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # If it's the last step, the accumulative reward is just the immediate reward
            accumulative_rewards[:, t, :] = rewards[:, t, :] * mask[:, t, :]
        else:
            # Accumulate reward at step t with the discounted reward at t+1, but only where the mask is true
            accumulative_rewards[:, t, :] = (rewards[:, t, :] + discount_factor * accumulative_rewards[:, t+1, :]) * mask[:, t, :]

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

def compute_discounted_future_value(end_step, discount_factor, max_seq_len):
    # Create a range tensor [0, 1, 2, ..., max_seq_len-1]
    step_range = torch.arange(max_seq_len, device=end_step.device).unsqueeze(0)
    
    # Calculate the difference between end step and each step in the range
    # This will give us the exponent to raise the discount factor to.
    # end_step is broadcasted along the second dimension to match the shape of step_range
    discount_exponents = end_step.unsqueeze(-1) - step_range

    # Ensure that the exponent is not negative
    # Since a negative exponent would increase the value instead of discounting it,
    # we set it to zero where the step range is greater than or equal to end_step
    discount_exponents.clamp_(min=0)

    # Compute the discount factors by raising to the power of the exponents
    discount_factors = discount_factor ** discount_exponents

    # Return the discount factors with an additional dimension to match the expected shape
    return discount_factors.unsqueeze(-1)
