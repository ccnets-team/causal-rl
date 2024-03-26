import torch
import torch.nn.functional as F

def generate_gaussian_kernel(size, sigma, device):
    """Generates a 1D Gaussian kernel used for smoothing probability distributions.
    This kernel is centered and normalized, ensuring that its sum equals 1."""
    x = torch.arange(size).to(device) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def generate_asymmetric_gaussian_kernel(size, sigma, device):
    """Generates a 1D Gaussian kernel and manually adjusts it to have the left side weighted two times more than the right side."""
    # Basic Gaussian kernel generation
    x = torch.arange(size).to(device) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    
    # Manually adjust the kernel to weight the right side two times more than the left
    mid_point = size // 2
    kernel[-mid_point:] = 0
    kernel /= kernel.sum()  # Re-normalize
    
    return kernel.unsqueeze(0).unsqueeze(0)

def smooth_prob_dist(probs, kernel):
    """Applies a Gaussian kernel to smooth a probability distribution. This smoothing
    process helps in creating a more generalized distribution by averaging out sharp
    transitions or fluctuations."""
    probs = probs.unsqueeze(0)  # Ensure probs is a 2D tensor with shape (1, N)
    
    half_kernel_size = kernel.size(2) // 2
    
    smoothed_probs = F.conv1d(probs, kernel, padding=half_kernel_size)
    smoothed_probs = smoothed_probs.squeeze(0)  # Remove the extra dimension
    normalized_smoothed_probs = smoothed_probs / smoothed_probs.sum()  # Normalize
    
    return normalized_smoothed_probs

def create_prob_dist_from_lambdas(lambda_values):
    """
    Generates a normalized probability distribution from a sequence of lambda values.
    This distribution accounts for the influence of each lambda value on subsequent positions,
    adjusted to ensure the distribution sums to 1.
    """
    # Ensure the last lambda value is set to 1 for stability
    lambda_sequence = lambda_values.detach().clone()
    lambda_sequence[-1] = 1
    
    n = len(lambda_sequence)
    # Prepare a reversed lambda sequence expanded across rows for cumulative product calculation
    reversed_lambda_expanded = torch.flip(lambda_sequence, dims=[0]).unsqueeze(0).expand(n, -1)
    
    # Create masks for lower triangular matrix
    lower_tri_mask = torch.tril(torch.ones((n, n), dtype=bool), diagonal=-1)
    lower_tri_mask_incl_diag = torch.tril(torch.ones((n, n), dtype=bool), diagonal=0)
    
    # Set lower triangle including diagonal to 1 for cumulative product calculation
    reversed_lambda_prepared = reversed_lambda_expanded.clone()
    reversed_lambda_prepared[lower_tri_mask_incl_diag] = 1
    
    # Calculate the cumulative product, then zero out the lower triangle excluding the diagonal
    cumprod_reversed_lambda = torch.cumprod(reversed_lambda_prepared, dim=1)
    cumprod_reversed_lambda[lower_tri_mask] = 0

    # Prepare for adjustment by zeroing out lower triangle excluding the diagonal
    reversed_lambda_adjusted = reversed_lambda_prepared.clone()
    reversed_lambda_adjusted[lower_tri_mask] = 0
    
    # Adjust for non-terminal sequences by multiplying with (1 - lambda value) for each sequence
    cumprod_reversed_lambda[:, :-1] *= (1 - reversed_lambda_adjusted[:, 1:])
    
    # Calculate the contribution of each lambda across the sequence by averaging
    prob_dist = cumprod_reversed_lambda.mean(dim=0)
    
    return prob_dist
