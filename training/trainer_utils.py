
import torch

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

# Function to apply selection mask to a trajectory component
def apply_sequence_mask(component, model_seq_mask, model_seq_length):
    component_shape = component.shape
    return component[model_seq_mask.expand_as(component) > 0].reshape(component_shape[0], model_seq_length, component_shape[2])

def calculate_value_size(state_size, min_threshold=1):
    """
    Calculates an optimal latent vector size based on the original state size. The calculated latent size 
    aims to be as close to the square of the state size as possible while considering computational 
    constraints represented by the threshold.

    The function ensures the latent size does not exceed a specified threshold, balancing between 
    achieving a square-like growth and maintaining computational efficiency.

    Args:
    - state_size (int): The size of the original state vector.
    - threshold (int): The maximum allowable size for the latent vector to ensure computational efficiency.

    Returns:
    - int: The calculated size of the latent vector, adhering to the computational constraints.
    """
    # Attempt to calculate a square-like size of the original state size
    proposed_latent_size = int(state_size ** 0.5)

    # Ensure the latent size does not exceed the threshold
    latent_size = max(proposed_latent_size, min_threshold)

    return latent_size

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