
import torch
import torch.nn.functional as F

def get_discount_sequence(discount_factor, max_seq_len):
    return (discount_factor ** torch.arange(max_seq_len).unsqueeze(0)).unsqueeze(-1)

def calculate_normalized_reward_scale(gamma, td_lambda, gpt_seq_length, device):
    """
    Calculates a normalized reward scale that ensures consistent variance in value loss across sequences.
    This approach accounts for the cumulative impact of discount rates (gamma) and bootstrapping rates (lambda),
    providing a stable reward scaling mechanism adaptable to different sequence lengths and temporal dynamics.

    Args:
        gamma (float): The discount factor gamma, influencing the importance of future rewards.
        td_lambda (float): The lambda parameter for TD(lambda) returns, balancing immediate and future rewards.
        gpt_seq_length (int): The maximum sequence length for the GPT model.
        device (torch.device): The computational device (CPU or GPU).

    Returns:
        torch.Tensor: The normalized reward scale to be used for consistent reward adjustment across sequences.
    """
    # Obtain discount sequences for gamma and lambda, shaping them for sequence-wide application.
    gammas = get_discount_sequence(gamma, gpt_seq_length).to(device)
    lambdas = get_discount_sequence(td_lambda, gpt_seq_length).to(device)

    # Calculate temporal scaling factors by combining gammas and lambdas, reflecting both discounting and bootstrapping effects.
    temporal_scaling_factors = gammas * lambdas

    # Compute the cumulative sum of these factors to understand their aggregate effect over sequences.
    accumulative_scaling_factors = torch.cumsum(temporal_scaling_factors, dim=1)

    # The square of the normalized reward scale is determined by the mean of accumulative scaling factors,
    # ensuring adjustments are uniformly applied across varying sequence dynamics.
    normalized_reward_scale_square = accumulative_scaling_factors.mean()

    # Applying a square root to the mean provides a reward scale that compensates for the squared increase in TD error or advantage,
    # facilitating a consistent variance in value loss irrespective of sequence length, gamma, or lambda.
    normalized_reward_scale = torch.sqrt(normalized_reward_scale_square)

    return normalized_reward_scale


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

def calculate_lambda_returns(values, rewards, dones, gamma, td_lambda):
    # Determine the batch size and sequence length from the rewards shape
    batch_size, seq_len, _ = rewards.shape

    # Initialize lambda returns with the same shape as values
    lambda_returns = torch.zeros_like(values)

    # Set the last timestep's lambda return to the last timestep's value
    lambda_returns[:, -1:] = values[:, -1:]

    # Iterate backwards through each timestep in the sequence
    for t in reversed(range(seq_len)):
        # Calculate lambda return for each timestep:
        # Current reward + discounted future value, adjusted by td_lambda
        lambda_returns[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * ((1 - td_lambda) * values[:, t + 1, :] + td_lambda * lambda_returns[:, t + 1, :])

    # Remove the last timestep to align lambda returns with their corresponding states
    lambda_returns = lambda_returns[:, :-1, :]

    return lambda_returns

def masked_tensor_reduction(tensor, mask, reduction="batch"):
    # Dictionary mapping for reduction type to dimension
    reduction_to_dim = {"batch": 0, "seq": 1}

    # Handle the 'all' reduction case separately
    if reduction == "all":
        return tensor[mask > 0].flatten().mean()

    # Get the dimension for reduction from the dictionary
    dim = reduction_to_dim.get(reduction)
    if dim is None:
        raise ValueError("Invalid reduction type. Choose 'batch', 'seq', or 'all'.")

    # Ensure mask is a boolean tensor
    mask_bool = mask.bool()
    # Multiply the tensor by the mask, zeroing out the elements of the tensor where mask is False
    masked_tensor = tensor * mask_bool
    # Sum the masked tensor across the batch dimension (dim=0)
    sum_per_sequence = torch.sum(masked_tensor, dim=dim)
    # Count the number of True entries in the mask per sequence for normalization
    count_per_sequence = torch.sum(mask_bool, dim=dim)
    # Handle potential division by zero in case some sequences are fully masked
    # If count_per_sequence is 0, replace it with 1 to prevent division by zero
    count_per_sequence = torch.clamp(count_per_sequence, min=1)
    # Calculate the mean by dividing the sum by the number of unmasked entries
    mean_per_sequence = sum_per_sequence / count_per_sequence
    return mean_per_sequence
