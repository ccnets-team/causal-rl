import torch
from .distribution_util import create_prob_dist_from_lambdas

# Constants to define thresholds for dynamically adjusting learnable sequence length based on the final value in a lambda sequence's probability distribution.
LEARNABLE_SEQ_LEN_MIN_THRESHOLD = 1/4  # Lower bound to potentially shorten the sequence.
LEARNABLE_SEQ_LEN_MAX_THRESHOLD = 3/4  # Upper bound to potentially extend the sequence.


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

def apply_sequence_mask(model_seq_mask, model_seq_length, *tensors):
    """
    Applies a selection mask to each tensor in `tensors`, reshaping them based on the `model_seq_length`.
    
    Args:
        model_seq_mask: A boolean mask indicating selected sequence positions.
        model_seq_length: The target sequence length after masking.
        tensors: A variable number of tensors to apply the mask on. Each tensor should have the shape [B, S, ...], 
                 where B is the batch size, and S is the sequence length.
    
    Returns:
        A tuple of tensors with the mask applied, each reshaped to [B, model_seq_length, ...]. If only one tensor is 
        provided, that tensor is returned directly instead of a tuple.
    """
    # Ensure the mask is expanded and reshaped appropriately for each tensor
    if len(tensors) == 1:
        # If there's only one tensor, apply the mask and reshape
        reshaped_tensor = tensors[0][model_seq_mask.expand_as(tensors[0])].reshape(tensors[0].shape[0], model_seq_length, -1)
        return reshaped_tensor
    else:    
        reshaped_tensors = tuple(tensor[model_seq_mask.expand_as(tensor)].reshape(tensor.shape[0], model_seq_length, -1) for tensor in tensors)
        return reshaped_tensors
def select_train_sequence(padding_mask, train_seq_length):
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
    # end_select_idx = torch.clamp(first_valid_idx.long() + train_seq_length, max=seq_len)

    # Calculate the start selection index for the training sequence
    first_select_idx = end_select_idx - train_seq_length

    # Create a range tensor of the sequence length [S]
    range_tensor = torch.arange(seq_len, device=padding_mask.device).unsqueeze(0).unsqueeze(-1)

    # Broadcast the range tensor to match the batch size and compare to generate the selection mask
    select_mask = (range_tensor >= first_select_idx) & (range_tensor < end_select_idx)
    
    return select_mask, end_select_idx

def calculate_learnable_sequence_length(lambda_sequence,
                                        lambda_chain_min_threshold=LEARNABLE_SEQ_LEN_MIN_THRESHOLD,
                                        lambda_chain_max_threshold=LEARNABLE_SEQ_LEN_MAX_THRESHOLD):
    """
    Calculates the optimal sequence length for training in a reinforcement learning environment, based on
    the relevance of state transitions determined by lambda values. This approach optimizes computational
    resources by focusing on significant associations between states.

    :param lambda_sequence: A tensor of lambda values representing the relevance of state transitions in a trajectory.
    :param lambda_chain_min_threshold: The minimum threshold of cumulative relevance score to consider a transition relevant.
    :param lambda_chain_max_threshold: The maximum threshold of cumulative relevance score to consider a transition relevant.
    :return: The optimal sequence length required for training, based on the relevance of state transitions.
    """
    input_seq_len = lambda_sequence.size(0)  # Use .size(0) to get the length of the tensor
    cumulative_relevance = create_prob_dist_from_lambdas(lambda_sequence)
    cumulative_relevance_score = cumulative_relevance[-1].item()
    # Determine the optimal length based on cumulative_relevance_score thresholds
    if cumulative_relevance_score > lambda_chain_max_threshold:
        optimal_length = input_seq_len + 1
    elif cumulative_relevance_score < lambda_chain_min_threshold:
        optimal_length = max(0, input_seq_len - 1)  # Ensure optimal_length is not negative
    else:
        optimal_length = input_seq_len  # Use the input length if within thresholds
    return optimal_length

def select_sequence_range(seq_range, *tensors):
    # Apply truncation based on the calculated idx
    if len(tensors) == 1:
        # If there's only one tensor, access it directly and apply the truncation
        truncated_tensors = tensors[0][:, seq_range]
    else:
        # If there are multiple tensors, truncate each tensor and pack them into a tuple
        truncated_tensors = tuple(tensor[:, seq_range] for tensor in tensors)

    return truncated_tensors

def create_init_lambda_sequence(target_mean, sequence_length, target_device):
    # Initialize lambda_sequence with zeros on target_device
    lambda_sequence = torch.zeros(sequence_length, device=target_device, dtype=torch.float)

    # Calculate initial and final values for the lambda sequence, considering target_mean
    initial_value = 2 * target_mean - 1
    final_value = 1  # Intended to ensure the last lambda value is 1

    # Adjust the sequence starting from 0 if initial_value is negative
    if initial_value <= 0:
        initial_value = target_mean/sequence_length
        # Adjust final_value to maintain the mean, keeping in mind the explicit setting of the last value to 1
        final_value = 2 * target_mean *(1 - 1/(2 * sequence_length))

    # Generate a tensor that linearly progresses from initial_value to final_value
    # Adjust to fill all but the last value of lambda_sequence with the linear progression
    lambda_sequence = torch.linspace(initial_value, final_value, steps=sequence_length, device=target_device, dtype=torch.float)

    return lambda_sequence