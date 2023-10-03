        
def scale_advantage(advantage, advantage_scaler, epsilon = 1e-8):
    if advantage_scaler == "none":
        return advantage
    elif advantage_scaler == "batch_norm":
        mean = advantage.detach().mean()
        std = advantage.detach().std()

        normalized = (advantage - mean) / (std + epsilon)  # Added epsilon to avoid division by zero

        return normalized
    
    elif advantage_scaler == "L1_norm":
        """
        Scales the advantage by the mean of its absolute values (L1 Norm).
        Useful when you want to normalize the fluctuations in the magnitude
        of individual advantage values within the batch.
        """
        abs_mean_advantage = advantage.detach().abs().mean()
        scale_factor = 1 / (abs_mean_advantage + epsilon)
        scaled_advantage = advantage * scale_factor
        return scaled_advantage
    
    elif advantage_scaler == "abs_mean":
        """
        Scales the advantage by the absolute value of their mean.
        Useful when the overall direction (sign) of the batch of advantages is of importance.
        """
        mean_abs_advantage = advantage.detach().mean().abs()
        scale_factor = 1 / (mean_abs_advantage + epsilon)
        scaled_advantage = advantage * scale_factor
        return scaled_advantage
    
    elif advantage_scaler == "mixed_norm":
        """
        Combines both L1 Norm and the absolute value of mean of advantages for scaling.
        Useful when a balance between the magnitude and overall direction of the advantages is needed.
        """
        abs_mean_advantage = advantage.detach().abs().mean()
        mean_abs_advantage = advantage.detach().mean().abs()
        scale_factor = abs_mean_advantage / (mean_abs_advantage + epsilon) 
        scaled_advantage = advantage * scale_factor
        return scaled_advantage
    else:
        raise ValueError(f"Invalid norm_type: {advantage_scaler}")

