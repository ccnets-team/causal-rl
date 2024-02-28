
import torch
import torch.nn.functional as F

class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale_factor):
        ctx.scale_factor = scale_factor
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale_factor, None  # Scale the gradient by the scale factor

def create_padding_mask_before_dones(dones: torch.Tensor) -> torch.Tensor:
    """
    Creates a padding mask for a trajectory by sampling from the end of the sequence. The mask is set to 0 
    (masked) for elements occurring before the 'done' signal when viewed from the end of the trajectory. 
    This includes padding elements that are positioned on the left side of the first 'done' signal in the 
    reversed sequence. The elements from the 'done' signal to the end of the trajectory (rightmost end) 
    are unmasked (set to 1).

    This function is useful for trajectories where sampling starts from the end and padding occurs before 
    the 'done' signal in the reversed order.

    Args:
    - dones (torch.Tensor): The tensor representing the 'done' signals in the trajectory.

    Returns:
    - mask (torch.Tensor): The resultant padding mask tensor. In this mask, elements occurring before the 
      'done' signal in the reversed sequence are masked (set to 0), while the elements from the 'done' 
      signal to the end of the trajectory are unmasked (set to 1).
    """
    mask = torch.ones_like(dones)

    if mask.size(1) > 1:
        # Reverse 'dones' along the specified axis (axis=1)
        reversed_dones = torch.flip(dones, dims=[1])

        # Perform cumulative sum on the reversed tensor
        cumulative_dones_reversed = torch.cumsum(reversed_dones[:,1:], dim=1)
        
        cumulative_dones_reversed[cumulative_dones_reversed > 0] = 1

        # Reverse the result back to get the cumulative sum in the original order
        cumulative_dones = torch.flip(cumulative_dones_reversed, dims=[1])
        
        mask[:, :-1, :] = 1 - cumulative_dones
    
    return mask

def calculate_sum_reward_weights(max_seq_len, gammas, td_lambdas, device):
    # Initialize tensors for value weights and sum reward weights with zeros.
    # Value weights are for calculating discounted future values, and sum reward weights are for scaling rewards.
    value_weights = torch.zeros(max_seq_len + 1, dtype=torch.float, device=device)
    sum_reward_weights = torch.zeros(max_seq_len, dtype=torch.float, device=device)

    flattened_gammas = gammas.flatten()
    flattened_td_lambdas = td_lambdas.flatten()
    
    # Ensure the final value weight equals 1, setting up a base case for backward calculation.
    value_weights[-1] = 1

    # Backward pass to compute weights. This loop calculates the decayed weights for each timestep,
    # applying the gamma (discount factor) and td_lambda (trade-off between TD and MC) to adjust the contribution of future rewards.
    for t in reversed(range(max_seq_len)):
        # Update value_weights by blending the immediate reward (1) and discounted future value weights,
        # modulated by gamma and td_lambda for each timestep.
        value_weights[t] = flattened_gammas[t] * ((1 - flattened_td_lambdas[t]) * 1 + flattened_td_lambdas[t] * value_weights[t + 1])

        # Compute the sum reward weights as the complement of value weights, indicating the proportion of reward assigned to each timestep.
        sum_reward_weights[t] = torch.clamp_min(1 - value_weights[t], 1e-8)

    # Ensure all sum_reward_weights are within the [0, 1] range, validating the computation logic.
    assert torch.all(sum_reward_weights >= 0) and torch.all(sum_reward_weights <= 1)

    # Normalize sum reward weights to maintain a consistent scale across sequences, improving stability.
    # The normalization also ensures that the mean of these weights is adjusted to 1, preventing disproportionate scaling.
    sum_reward_weights /= sum_reward_weights.mean(dim=0, keepdim=True).clamp(min=1e-8)

    # Reshape sum reward weights for compatibility with expected input formats, preparing for further processing.
    sum_reward_weights = sum_reward_weights.unsqueeze(0).unsqueeze(-1)
    
    return sum_reward_weights

def calculate_lambda_returns(values, rewards, dones, gammas, td_lambdas):
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
    # Determine the batch size and sequence length from the rewards shape
    batch_size, seq_len, _ = rewards.shape

    # Initialize lambda returns with the same shape as values
    sum_rewards = torch.zeros_like(values)
    lambda_returns = torch.zeros_like(values)

    # Set the last timestep's lambda return to the last timestep's value
    lambda_returns[:, -1:] = values[:, -1:]

    # Iterate backwards through each timestep in the sequence
    for t in reversed(range(seq_len)):
        # Calculate lambda return for each timestep:
        sum_rewards[:, t, :] = rewards[:, t, :] + gammas[:, t, :] * (1 - dones[:, t, :]) * (td_lambdas[:, t, :] * sum_rewards[:, t + 1, :])
        
        # Current reward + discounted future value, adjusted by td_lambda
        lambda_returns[:, t, :] = rewards[:, t, :] + gammas[:, t, :] * (1 - dones[:, t, :]) * ((1 - td_lambdas[:, t, :]) * values[:, t + 1, :] + td_lambdas[:, t, :] * lambda_returns[:, t + 1, :])

    # Remove the last timestep to align lambda returns with their corresponding states
    sum_rewards = sum_rewards[:, :-1, :]
    lambda_returns = lambda_returns[:, :-1, :]

    return lambda_returns, sum_rewards

def masked_tensor_reduction(tensor, mask, reduction="batch"):
    """
    Performs a masked reduction on a tensor according to a specified method.
    
    Parameters:
    - tensor: Input tensor to be reduced.
    - mask: Boolean mask indicating valid elements for consideration in the reduction.
    - reduction: Type of reduction to apply. Options are "batch", "seq", "cross", or "all".
    
    Behavior:
    - "all": Returns the mean of all elements marked True by the mask, effectively ignoring masked elements.
    - "batch"/"seq": Computes the mean across the batch or sequence dimension for unmasked elements only,
      adjusted by the ratio of unmasked elements to the total batch size, ensuring fair representation of each dimension.
    - "cross": Integrates contributions from both batch-wise and sequence-wise reductions into a single tensor,
      with each dimension's contributions adjusted by scale factors to ensure their weights are balanced and equivalent
      to other reduction types, facilitating consistent learning signals across different dimensions.
            
    Returns:
    - A tensor representing the reduced values, scaled appropriately based on the reduction strategy and mask.
    
    Utilizes GradScaler to adjust gradients during backpropagation, ensuring proportional learning from unmasked elements.
    """
    if reduction not in {"batch", "seq", "cross", "all"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    
    mask_bool = mask.bool()
    masked_tensor = tensor * mask_bool  # Apply mask
    batch_size = mask.size(0)

    if reduction == "all":
        # Directly return mean of the masked elements
        return torch.mean(tensor[mask_bool])

    if reduction in {"batch", "seq"}:
        dim = 0 if reduction == "batch" else 1
        sum_per_dim = torch.sum(masked_tensor, dim=dim)
        count_per_dim = torch.sum(mask_bool, dim=dim).clamp(min=1)
        mean_across_dim = sum_per_dim / count_per_dim
        
        dim_scale_factor =  (count_per_dim/batch_size)
        mean_across_dim_adjusted = GradScaler.apply(mean_across_dim, dim_scale_factor)
        return mean_across_dim_adjusted

    elif reduction == "cross":
        sum_per_batch = torch.sum(masked_tensor, dim=0)
        count_per_batch = torch.sum(mask_bool, dim=0).clamp(min=1)
        mean_across_batch = sum_per_batch / count_per_batch 
        
        sum_per_sequence = torch.sum(masked_tensor, dim=1)
        count_per_sequence = torch.sum(mask_bool, dim=1).clamp(min=1)
        mean_across_sequence = sum_per_sequence / count_per_sequence 

        batch_scale_factor =  0.5 * (count_per_batch/batch_size)
        seq_scale_factor =  0.5 * (count_per_sequence/batch_size)
        
        # Adjust to ensure equal weight for batch and sequence reductions
        mean_across_batch_adjusted = GradScaler.apply(mean_across_batch, batch_scale_factor)
        mean_across_sequence_adjusted = GradScaler.apply(mean_across_sequence, seq_scale_factor)
        
        combined_reduction = torch.cat([mean_across_batch_adjusted, mean_across_sequence_adjusted], dim=0)
        return combined_reduction

def create_train_sequence_mask(padding_mask, train_seq_length):
    """
    Creates a selection mask for trajectories.

    :param padding_mask: Mask tensor from create_padding_mask_before_dones, shape [B, S, 1].
    :param train_seq_length: Length of the train sequence to select.
    :return: Selection mask tensor, shape [B, train_seq_length, 1].
    """
    batch_size, seq_len, _ = padding_mask.shape

    # Find the index of the first non-padding point after 'done'
    first_non_padding_idx = torch.argmax(padding_mask, dim=1, keepdim=True)
    end_non_padding_idx = first_non_padding_idx + train_seq_length
    end_select_idx = torch.clamp(end_non_padding_idx, max=seq_len)
    first_select_idx = end_select_idx - train_seq_length

    # Create a range tensor of shape [S]
    range_tensor = torch.arange(seq_len, device=padding_mask.device).unsqueeze(0).unsqueeze(-1)

    # Broadcast to shape [B, S, 1] and compare
    select_mask = (range_tensor >= first_select_idx) & (range_tensor < end_select_idx)
    
    return select_mask, end_select_idx

# Function to apply selection mask to a trajectory component
def apply_sequence_mask(component, model_seq_mask, model_seq_length):
    component_shape = component.shape
    return component[model_seq_mask.expand_as(component) > 0].reshape(component_shape[0], model_seq_length, component_shape[2])