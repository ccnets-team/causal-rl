
import torch
import torch.nn.functional as F

def get_discount_sequence(discount_factor, max_seq_len):
    return (discount_factor ** torch.arange(max_seq_len).unsqueeze(0)).unsqueeze(-1)

def create_sum_reward_scale(gamma, td_lambda, gpt_seq_length, device):
    """
    This function computes a scaling factor for normalizing the cumulative rewards received over a sequence of actions,
    taking into account the temporal discounting factor (gamma) and the lambda parameter for TD(lambda) returns. 
    The normalization process ensures that the variance of value loss remains consistent across sequences of different lengths,
    effectively balancing the significance of rewards received at various points in the sequence. 
    This is particularly important for maintaining the relevance of initial rewards, which might otherwise diminish due to the effects of gamma and lambda.
    
    The function operates within the context of sequences handled by a GPT model, accommodating for the maximum sequence length specified.
    It leverages the computational capabilities of the specified device (CPU or GPU) to perform the necessary calculations.

    Args:
        gamma (float): The discount factor applied to future rewards, diminishing their present value.
        td_lambda (float): The lambda parameter that influences the weighting of immediate versus future rewards in the TD(lambda) return calculation.
        gpt_seq_length (int): The maximum length of sequences that the GPT model can process, which dictates the scope of reward normalization.
        device (torch.device): The device (CPU or GPU) where the calculations will be executed, ensuring optimal performance.

    Returns:
        torch.Tensor: A tensor containing the scaling factors for each timestep within a sequence. These factors are designed to normalize the sum of rewards,
        preserving the impact of initial rewards by counteracting the natural decay introduced by the gamma and lambda parameters. This normalization
        facilitates a more equitable distribution of the significance of rewards across the sequence, enhancing the model's learning efficiency.
    """
    # Generate sequences for gamma and lambda, adjusted for sequence-wide application.
    gammas_sequence = get_discount_sequence(gamma, gpt_seq_length).to(device)
    lambdas_sequence = get_discount_sequence(td_lambda, gpt_seq_length).to(device)

    # Calculate the combined impact of discounting and bootstrapping for each timestep.
    temporal_scaling_factors = gammas_sequence * lambdas_sequence

    # Apply a weight to each timestep based on the combined temporal factors, giving more weight to earlier sequences.
    normalized_scale = torch.flip(temporal_scaling_factors, dims=[1])
    
    return normalized_scale


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
    """
    Calculates lambda returns and sum of rewards for each timestep in a sequence.

    Args:
        values (torch.Tensor): The value estimates for each timestep.
        rewards (torch.Tensor): The rewards received at each timestep.
        dones (torch.Tensor): Indicates whether a timestep is terminal (1 if terminal, 0 otherwise).
        gamma (float): Discount factor for future rewards.
        td_lambda (float): Lambda parameter for TD(lambda) returns.

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
        sum_rewards[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * (td_lambda * sum_rewards[:, t + 1, :])
        
        # Current reward + discounted future value, adjusted by td_lambda
        lambda_returns[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * ((1 - td_lambda) * values[:, t + 1, :] + td_lambda * lambda_returns[:, t + 1, :])

    # Remove the last timestep to align lambda returns with their corresponding states
    sum_rewards = sum_rewards[:, :-1, :]
    lambda_returns = lambda_returns[:, :-1, :]

    return lambda_returns, sum_rewards

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
