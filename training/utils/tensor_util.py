
import torch

class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale_factor):
        ctx.scale_factor = scale_factor
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale_factor, None  # Scale the gradient by the scale factor

class LambdaGradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Apply a different scaling based on the direction of the gradient
        scaled_grad_output = grad_output.clone()  # Clone to avoid in-place operations
        # Scale down negative gradients less (making the decrease slower)
        scaled_grad_output[grad_output < 0] *= (1 + 1/3)
        # Apply normal scaling to positive gradients
        scaled_grad_output[grad_output >= 0] *= (1 - 1/3)
        return scaled_grad_output, None
    
def shift_left_padding_mask(padding_mask):
    return torch.cat([padding_mask[:, 1:], torch.ones_like(padding_mask[:, -1:])], dim=1)

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
    seq_size = mask.size(1)
    total_count = torch.sum(mask_bool).clamp(min=1)
    total_scale_factor = (seq_size * batch_size / total_count)

    if reduction == "all":
        # Directly return mean of the masked elements
        return torch.mean(tensor[mask_bool])

    if reduction in {"batch", "seq"}:
        dim = 0 if reduction == "batch" else 1
        sum_per_dim = torch.sum(masked_tensor, dim=dim)
        count_per_dim = torch.sum(mask_bool, dim=dim).clamp(min=1)
        mean_across_dim = sum_per_dim / count_per_dim
        
        dim_scale_factor =  (count_per_dim/batch_size) * total_scale_factor
        mean_across_dim_adjusted = GradScaler.apply(mean_across_dim, dim_scale_factor)
        return mean_across_dim_adjusted

    elif reduction == "cross":
        sum_per_batch = torch.sum(masked_tensor, dim=0)
        count_per_batch = torch.sum(mask_bool, dim=0).clamp(min=1)
        mean_across_batch = sum_per_batch / count_per_batch 
        
        sum_per_sequence = torch.sum(masked_tensor, dim=1)
        count_per_sequence = torch.sum(mask_bool, dim=1).clamp(min=1)
        mean_across_sequence = sum_per_sequence / count_per_sequence 

        batch_scale_factor =  0.5 * (count_per_batch/batch_size) * total_scale_factor
        seq_scale_factor =  0.5 * (count_per_sequence/batch_size) * total_scale_factor
        
        # Adjust to ensure equal weight for batch and sequence reductions
        mean_across_batch_adjusted = GradScaler.apply(mean_across_batch, batch_scale_factor)
        mean_across_sequence_adjusted = GradScaler.apply(mean_across_sequence, seq_scale_factor)
        
        combined_reduction = torch.cat([mean_across_batch_adjusted, mean_across_sequence_adjusted], dim=0)
        return combined_reduction


def create_transformation_matrix(num_rows, num_cols):
    """
    Creates a transformation matrix for applications like image reconstruction, 
    where selective feature emphasis is desired. This matrix is designed to offer a nuanced approach 
    to feature transformation, balancing between de-emphasizing and emphasizing different features.

    The matrix is constructed row by row. For each row except the last, the feature emphasis 
    is distributed based on a dynamic order influenced by a rolling order distribution, 
    allowing for a diverse emphasis pattern across rows. This pattern aims to provide a 
    balanced transformation by varying which features are emphasized in each row, preventing 
    any single feature from being consistently deprioritized or overemphasized.

    In the last row, the remaining weights are assigned to ensure all features receive some level 
    of emphasis, maintaining the matrix's balance. The transformation matrix is then uniformly scaled 
    to ensure the intended distributional characteristics are preserved across both rows and columns.

    Parameters:
    - num_rows (int): The number of rows in the transformation matrix, corresponding to the transformed feature dimension.
    - num_cols (int): The number of columns in the transformation matrix, corresponding to the original feature dimension.

    Returns:
    - Tensor: A transformation matrix of shape (num_rows, num_cols), ready for application to data needing feature emphasis transformation.

    This transformation matrix is particularly useful in scenarios where not all input features are of equal relevance, 
    and a controlled, nuanced approach to feature emphasis is beneficial for the task at hand.
    """
    remaining_weights = torch.ones((num_cols), dtype=torch.float)  # Tracks remaining weights for distribution
    transformation_matrix = torch.zeros((num_rows, num_cols))  # Initialize transformation matrix
    
    order_distribution = torch.linspace(num_cols*1e-8, 1e-8, steps=num_cols)

    for row in range(num_rows):
        if row < num_rows - 1:
            # Sort `remaining_weights` to get indices for ascending order
            sorted_remaining, sorted_indices = torch.sort(remaining_weights + order_distribution)
            order_distribution = torch.roll(order_distribution, 1, 0)
            # Define lengths for zero and linearly increasing segments
            
            # Create the linearly increasing distribution segment
            distribution = torch.linspace(1, num_cols, steps=num_cols)
            
            # Scale the distribution to ensure sum matches expected proportion
            scaled_distribution = (num_cols / num_rows) * distribution / distribution.sum()
            
            # Assign scaled distribution values to their respective positions in the transformation matrix
            for col in range(num_cols):
                sorted_position = sorted_indices[col]
                transformation_matrix[row, sorted_position] = scaled_distribution[col]
                remaining_weights[sorted_position] -= scaled_distribution[col]
        else:
            # For the last row, directly assign remaining weights
            transformation_matrix[row] = remaining_weights
    
    # Adjust the transformation matrix to ensure overall balance
    transformation_matrix *= (num_rows / num_cols)
    
    return transformation_matrix.unsqueeze(0)

def shorten_tensor_sequences(padding_mask, *tensors, min_length=1):
    """
    Truncates sequences in the provided tensors based on a shared padding mask, removing padded elements
    from the start of each sequence. Optionally, also returns the truncated padding mask.
    All input tensors must have the same batch size and sequence length dimensions but can differ
    in other dimensions.

    :param padding_mask: A padding mask of shape [B, S, 1] indicating valid elements (1) and padding elements (0),
                         aligned to the right.
    :param tensors: A variable number of tensors with shapes [B, S, ?] to be truncated based on the padding mask.

    :return: The truncated padding mask, and a single tensor directly or a tuple of tensors, 
             with sequences truncated to remove leading padding, based on the padding mask.
    """
    original_length = padding_mask.size(1)
    
    # Calculate the index of the first non-zero sum across the sequences
    first_non_zero_index = torch.argmax((padding_mask.squeeze(-1).sum(dim=0) > 0).float()).item()

    shortened_length = original_length - first_non_zero_index

    # Determine the minimum sequence length after truncation, considering the provided min_length
    target_length = max(min(min_length, original_length), shortened_length)

    # Calculate the starting index for truncation to maintain the target length
    # Ensuring we do not exceed the original sequence length
    truncate_start_idx = max(original_length - target_length, 0)
    # Apply truncation based on the calculated idx
    if len(tensors) == 1:
        # If there's only one tensor, access it directly and apply the truncation
        truncated_tensors = tensors[0][:, truncate_start_idx:]
    else:
        # If there are multiple tensors, truncate each tensor and pack them into a tuple
        truncated_tensors = tuple(tensor[:, truncate_start_idx:] for tensor in tensors)

    # Truncate the padding mask using the same idx
    truncated_padding_mask = padding_mask[:, truncate_start_idx:]

    # Return the truncated tensors followed by the truncated padding mask
    # If there's only one tensor, it returns that tensor directly without wrapping it in a tuple
    return (*truncated_tensors, truncated_padding_mask) if len(tensors) > 1 else (truncated_tensors, truncated_padding_mask)

def keep_right_tensor_sequences(truncated_length, *tensors):
    """
    Truncates sequences in the provided tensors to keep only the specified amount of data from the right side,
    if the sequence length is greater than the truncated_length. If the sequence length is less than or equal
    to the truncated_length, it returns the original tensor(s). This function is useful for focusing on the most
    relevant portions of each sequence for learning.

    :param truncated_length: The length of the sequence to keep from the right side.
    :param tensors: A variable number of tensors with shapes [B, S, ?] to be truncated. Each tensor must have
                    the same batch size and sequence length dimensions but can differ in other dimensions.

    :return: A single tensor directly or a tuple of tensors with sequences truncated to keep only the
             specified truncated_length from the right side, or the original tensors if their sequence
             length is less than or equal to the truncated_length.
    """
    # Check each tensor's sequence length and truncate if necessary
    truncated_tensors = tuple(tensor[:, -truncated_length:] if tensor.size(1) > truncated_length else tensor
                              for tensor in tensors)

    # Return the truncated tensor(s)
    # If there's only one tensor, it returns that tensor directly without wrapping it in a tuple
    return truncated_tensors if len(tensors) > 1 else truncated_tensors[0]

def fill_up_to_end_idx(padding_mask, end_indices, fill_value):
    batch_size, seq_len, _ = padding_mask.shape
    
    seq_indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(padding_mask.device)

    seq_indices_expanded = seq_indices.expand_as(padding_mask)
    end_indices_expanded = end_indices.expand_as(padding_mask)

    mask = seq_indices_expanded < end_indices_expanded
    
    padding_mask[mask] = fill_value

    return padding_mask

def fill_from_start_idx(padding_mask, start_indices, fill_value):
    batch_size, seq_len, _ = padding_mask.shape
    
    seq_indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(padding_mask.device)

    seq_indices_expanded = seq_indices.expand_as(padding_mask)
    end_indices_expanded = start_indices.expand_as(padding_mask)

    mask = seq_indices_expanded >= end_indices_expanded
    
    padding_mask[mask] = fill_value

    return padding_mask

def pad_up_to_first_content(padding_mask, first_seq_idx, content_lengths):
    """
    Updates the padding_mask based on first_seq_idx and content_lengths.
    """
    # Adjust for zero-based indexing if first_seq_idx is one-based
    adjusted_first_seq_idx = first_seq_idx - 1
    
    # Ensure indices are within the valid range before gathering
    adjusted_first_seq_idx = torch.clamp(adjusted_first_seq_idx, 0, padding_mask.shape[1] - 1)
    
    # Use adjusted_first_seq_idx for gathering
    selected_content_length = torch.gather(content_lengths, 1, adjusted_first_seq_idx)

    # Calculate the actual indices to fill up to, considering content_lengths
    end_indices = adjusted_first_seq_idx - selected_content_length + 1 # +1 to include the first content index itself in the non-padded area
    
    padding_mask = fill_up_to_end_idx(padding_mask, end_indices, fill_value=0.0)
    
    return padding_mask

def pad_up_to_avg_content(padding_mask, content_lengths):
    """
    Updates the padding_mask based on first_seq_idx and content_lengths.
    """
    seq_len = padding_mask.size(1)
    range_tensor = torch.arange(1, seq_len + 1).unsqueeze(0).unsqueeze(-1).float().to(padding_mask.device)
    weight_tensor = range_tensor / range_tensor.mean().clamp(min=1e-8)
    weighted_content_lengths = weight_tensor * content_lengths
    mean_weighted_content_lengths = weighted_content_lengths.mean(dim = 1, keepdim = True)
    padding_lengths = seq_len - mean_weighted_content_lengths.long()
    
    mask = range_tensor <= padding_lengths
    padding_mask[mask] = 0
    
    return padding_mask

def prioritize_tensor_sequence(tensor, padding_mask):
    seq_len = tensor.size(1)
    device = tensor.device
    # Creating a range tensor [1, 2, ..., seq_len]
    range_tensor = torch.arange(1, seq_len + 1).unsqueeze(0).unsqueeze(-1).to(device).float()
    
    # Apply padding mask to range_tensor, setting padded positions to 0
    masked_range_tensor = padding_mask * range_tensor
    
    # Find the minimum non-zero (non-padded) value for each sequence
    # We replace 0s with inf to ensure they're not considered as minima
    min_values, _ = torch.where(masked_range_tensor > 0, masked_range_tensor, torch.full_like(masked_range_tensor, float('inf'))).min(dim=1, keepdim=True)
    
    # Calculate offsets from the min non-zero value, adding back the padding mask
    adusted_masked_tensor = (range_tensor - min_values + 1) * padding_mask
    
    grad_priority = adusted_masked_tensor/adusted_masked_tensor.mean(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Placeholder for applying grad_priority: Assuming you have a custom implementation for this
    prioritized_tensor = GradScaler.apply(tensor, grad_priority)
    
    return prioritized_tensor # Return the grad_priority for inspection or further use