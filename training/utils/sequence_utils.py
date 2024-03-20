import torch

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