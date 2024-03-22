import torch
PADDING_LAMBDA_CAHIN_VAL_THRESHOLD = 1e-6

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

# Function to apply selection mask to a trajectory component
def apply_sequence_mask(component, model_seq_mask, model_seq_length):
    component_shape = component.shape
    return component[model_seq_mask.expand_as(component) > 0].reshape(component_shape[0], model_seq_length, component_shape[2])

def create_train_sequence_mask(padding_mask, train_seq_length):
    """
    Creates a selection mask for trajectories based on the specified training sequence length.

    :param padding_mask: Mask tensor indicating padding positions (0s for padding, 1s for data) 
                         from create_padding_mask_before_dones, shape [B, S, 1].
    :param train_seq_length: Length of the training sequence to select.
    :return: Selection mask tensor indicating the selected training sequence, shape [B, S, 1],
             and the end selection index tensor.
    """
    batch_size, seq_len, _ = padding_mask.shape

    # Find the index of the first valid data point (non-padding) following a 'done' signal
    first_valid_idx = torch.argmax(padding_mask, dim=1, keepdim=True)
    valid_sequence_part = (seq_len - first_valid_idx).float()
    
    # Calculate the ratio of the sequence length remaining after subtracting the training sequence length
    ratio = (seq_len - train_seq_length) / seq_len
    # Determine the end selection index based on the ratio and the left part of the sequence
    
    td_sequence_part = (ratio * valid_sequence_part).long()
    
    end_select_idx = seq_len - td_sequence_part
    
    # Ensure the end selection index is within valid range
    end_select_idx = torch.clamp(end_select_idx, min=train_seq_length, max=seq_len)
    
    # Calculate the start selection index for the training sequence
    first_select_idx = end_select_idx - train_seq_length

    # Create a range tensor of the sequence length [S]
    range_tensor = torch.arange(seq_len, device=padding_mask.device).unsqueeze(0).unsqueeze(-1)

    # Broadcast the range tensor to match the batch size and compare to generate the selection mask
    select_mask = (range_tensor >= first_select_idx) & (range_tensor < end_select_idx)
    
    return select_mask, end_select_idx

def adjust_padding_mask_based_on_lambda(padding_mask, lambda_sequence, padding_threshold=PADDING_LAMBDA_CAHIN_VAL_THRESHOLD):
    """
    Adjusts the padding mask based on the cumulative product of lambda values in the sequence. If the cumulative
    product of lambda values from a specific index onwards (when considering the sequence in reverse order) is smaller 
    than a padding_threshold, that part of the sequence is considered as padding and the mask is updated to reflect this, 
    aligning padding to the left.

    :param lambda_sequence: A tensor of lambda values for the sequence with shape [Seq Len].
    :param padding_mask: The original padding mask tensor with shape [Batch Size, Seq Len, 1], where 0 indicates padding.
    :param padding_threshold: The threshold below which the cumulative product indicates the need for padding.
    :return: An updated padding mask with potentially more padding based on lambda sequence analysis.
    """
    # Reverse the lambda sequence for cumulative product calculation from the end
    reversed_lambda = lambda_sequence.detach().clone().flip(dims=[0])
    # Calculate the cumulative product in the reversed order
    reversed_cumulative_product = torch.cumprod(reversed_lambda, dim=0)
    # Flip the cumulative product back to match the original sequence order
    chain_dependent_product = reversed_cumulative_product.flip(dims=[0])
    # Identify positions that do not meet the padding_threshold
    padding_criteria = chain_dependent_product > padding_threshold
    # Adjust dimensions to match the padding mask
    padding_criteria = padding_criteria.unsqueeze(0).unsqueeze(-1).float()

    padding_mask[:, :-1] *= padding_criteria[:, 1:]
    
    return padding_mask

def select_sequence_range(seq_range, *tensors):
    # Apply truncation based on the calculated idx
    if len(tensors) == 1:
        # If there's only one tensor, access it directly and apply the truncation
        truncated_tensors = tensors[0][:, seq_range]
    else:
        # If there are multiple tensors, truncate each tensor and pack them into a tuple
        truncated_tensors = tuple(tensor[:, seq_range] for tensor in tensors)

    return truncated_tensors

def create_prob_dist_from_lambdas(lambda_values):
    """Generates a probability distribution from a sequence of lambda values. This
    distribution reflects the likelihood of chain dependencies influenced by the lambda values,
    adjusted to ensure the entire distribution sums to 1."""
    lambda_sequence = lambda_values.detach().clone()
    lambda_sequence[-1] = 1  # Ensure stability by fixing the last lambda to 1
    reversed_lambda = torch.flip(lambda_sequence, dims=[0])
    reversed_cumulative_product = torch.cumprod(reversed_lambda, dim=0)
    chain_dependent_product = torch.flip(reversed_cumulative_product, dims=[0])
    
    chain_dependent_product[1:] *= (1 - lambda_sequence[:-1])
    mean_chain_dependent_product = chain_dependent_product / chain_dependent_product.sum()
    adjusted_probabilities = torch.flip(mean_chain_dependent_product, dims=[0])
    return adjusted_probabilities
