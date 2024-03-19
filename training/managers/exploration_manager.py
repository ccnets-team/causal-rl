import torch
import torch.nn.functional as F

def generate_gaussian_kernel(size, sigma, device):
    """Generates a 1D Gaussian kernel used for smoothing probability distributions.
    This kernel is centered and normalized, ensuring that its sum equals 1."""
    x = torch.arange(size).to(device) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel
        
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
    reversed_lambda = torch.flip(lambda_values, dims=[0])
    reversed_cumulative_product = torch.cumprod(reversed_lambda, dim=0)
    chain_dependent_product = torch.flip(reversed_cumulative_product, dims=[0])
    
    chain_dependent_product[1:] *= (1 - lambda_values[:-1])
    mean_chain_dependent_product = chain_dependent_product / chain_dependent_product.sum()
    adjusted_probabilities = torch.flip(mean_chain_dependent_product, dims=[0])
    return adjusted_probabilities

class ExplorationUtils:
    def __init__(self, gpt_seq_length, learnable_td, device):
        self.device = device
        self.learnable_td = learnable_td
        self.gpt_seq_length = gpt_seq_length
        self.sequence_lengths = torch.arange(1, gpt_seq_length + 1, dtype=torch.long, device=self.device)
        self.kernel_size = int(gpt_seq_length * 2 - 1)
        self.sigma = 1.0
        # Generate the Gaussian kernel
        self.kernel = generate_gaussian_kernel(self.kernel_size, self.sigma, self.device).unsqueeze(0).unsqueeze(0)

    def sample_sequence_probabilities(self, use_smoothed_probs=False):
        """Generates or samples from a probability distribution for sequence lengths based on TD(λ) values.
        Optionally smooths the distribution using a Gaussian kernel for a more generalized probability curve."""
        learnable_td_lambda = self.learnable_td.lambd.detach().clone()
        learnable_td_lambda[-1] = 1  # Ensure stability by fixing the last lambda to 1

        lambda_sequence_probs = create_prob_dist_from_lambdas(learnable_td_lambda)
        if use_smoothed_probs:
            smoothed_sequence_probs = smooth_prob_dist(lambda_sequence_probs, self.kernel)
        else:
            smoothed_sequence_probs = lambda_sequence_probs
        return smoothed_sequence_probs

    def sample_padding_lengths(self, batch_size, max_seq_length):
        """Samples padding lengths for a batch of sequences based on a probability distribution,
        allowing for dynamic adjustments to sequence padding based on learned TD(λ) values."""
        sampled_sequence_probs = self.sample_sequence_probabilities(use_smoothed_probs=True)
        sampled_indices = torch.multinomial(sampled_sequence_probs, batch_size, replacement=True)
        sampled_lengths = self.sequence_lengths[sampled_indices]

        padding_seq_length = max_seq_length - sampled_lengths
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def get_optimal_padding_lengths(self):
        """Calculates optimal padding lengths for sequences, aiming to align with the most likely
        sequence length based on the smoothed probability distribution."""
        max_seq_length = self.gpt_seq_length
        sampled_sequence_probs = self.sample_sequence_probabilities(use_smoothed_probs=False)
        sampled_indices = torch.argmax(sampled_sequence_probs, dim=0)
        padding_seq_length = max_seq_length - self.sequence_lengths[sampled_indices]
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def get_current_padding_lengths(self, padding_mask):
        max_seq_length = padding_mask.size(1)
        cur_padding_lengths = padding_mask.size(1) - torch.sum(padding_mask, dim=1).long()
        return torch.clamp(cur_padding_lengths, 0, max_seq_length - 1)
    
    def apply_sequence_masking(self, padding_mask):
        """
        Identifies positions within the padding mask that should be updated to reflect 
        the dynamically sampled sequence lengths. This process involves determining which 
        parts of each sequence are considered 'padding' based on the sampled lengths.
        """
        batch_size = padding_mask.size(0)
        max_seq_length = padding_mask.size(1)
        cur_padding_lengths = self.get_current_padding_lengths(padding_mask)
        
        padding_lengths = self.sample_padding_lengths(batch_size, max_seq_length)
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0

        applied_padding_lengths = torch.max(cur_padding_lengths, padding_lengths)
        return padding_mask, applied_padding_lengths