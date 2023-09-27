import torch

def scale_advantage(advantage, norm_type, epsilon = 1e-8):
    if norm_type == "none":
        return advantage
    elif norm_type == "batch_norm":
        mean = advantage.detach().mean()
        std = advantage.detach().std()

        normalized = (advantage - mean) / (std + epsilon)  # Added epsilon to avoid division by zero
        return normalized
    
    elif norm_type == "L1_norm":
        # Detach the tensor to prevent tracking of operations in the computational graph
        detached_data = advantage.detach()

        # Calculate the absolute value of the mean of detached_data
        mean_absolute_value = detached_data.abs().mean()

        scale_factor = 1 / mean_absolute_value if mean_absolute_value != 0 else 1

        # Return the original data scaled by the scale_factor
        scaled_data = advantage * scale_factor
        return scaled_data

    elif norm_type == "dynamic_scale":
        abs_advantage = torch.abs(advantage.detach())
        mean_abs_advantage = torch.mean(abs_advantage)
        scaled_advantage = advantage * (mean_abs_advantage / (abs_advantage + epsilon))
        return scaled_advantage
        
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}")

