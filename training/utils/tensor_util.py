
import torch

class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale_factor):
        ctx.scale_factor = scale_factor
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale_factor, None  # Scale the gradient by the scale factor
    
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

def shorten_tensor_sequences(padding_mask, *tensors):
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
    # Calculate idx based on your logic
    idx = torch.argmax((padding_mask.squeeze(-1).sum(dim=0) > 0).float()).item()
    
    # Apply truncation based on the calculated idx
    if len(tensors) == 1:
        # If there's only one tensor, access it directly and apply the truncation
        truncated_tensors = tensors[0][:, idx:]
    else:
        # If there are multiple tensors, truncate each tensor and pack them into a tuple
        truncated_tensors = tuple(tensor[:, idx:] for tensor in tensors)

    # Truncate the padding mask using the same idx
    truncated_padding_mask = padding_mask[:, idx:]

    # Return the truncated tensors followed by the truncated padding mask
    # If there's only one tensor, it returns that tensor directly without wrapping it in a tuple
    return (*truncated_tensors, truncated_padding_mask) if len(tensors) > 1 else (truncated_tensors, truncated_padding_mask)

def shift_left_padding_mask(padding_mask):
    return torch.cat([padding_mask[:, 1:], torch.ones_like(padding_mask[:, -1:])], dim=1)
