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
    kernel[-mid_point:] /= 2
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
