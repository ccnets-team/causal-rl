import torch
import torch.nn.functional as F

def create_chain_probabilities_from_lambdas(lambda_values):
    reversed_lambda = torch.flip(lambda_values, dims=[0])
    reversed_cumulative_product = torch.cumprod(reversed_lambda, dim=0)
    chain_dependent_product = torch.flip(reversed_cumulative_product, dims=[0])
    
    chain_dependent_product[1:] *= (1 - lambda_values[:-1])
    mean_chain_dependent_product = chain_dependent_product / chain_dependent_product.sum()
    adjusted_probabilities = torch.flip(mean_chain_dependent_product, dims=[0])
    return adjusted_probabilities

def gaussian_kernel(size, sigma, device):
    """Generates a 1D Gaussian kernel."""
    x = torch.arange(size).to(device) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def smooth_probs(probs, kernel):
    """
    Smooths the probability distribution using a Gaussian kernel and normalizes the result.
    
    The kernel size is set to total_length*2 - 1 to ensure each element's contribution to its neighbors
    is preserved across the entire length of the input.
    """

    # Ensure probs is a 2D tensor with shape (1, N)
    probs = probs.unsqueeze(0)

    # Apply convolution to smooth the probabilities
    smoothed_probs = F.conv1d(probs, kernel, padding=kernel.size(2) // 2)
    
    smoothed_probs = smoothed_probs.squeeze(0)
    
    # Normalize the smoothed probabilities so their sum equals 1
    normalized_smoothed_probs = smoothed_probs / smoothed_probs.sum()
    
    return normalized_smoothed_probs

class ExplorationUtils:
    def __init__(self, gpt_seq_length, learnable_td, device):
        self.device = device
        self.learnable_td = learnable_td
        self.sequence_lengths = torch.arange(1, gpt_seq_length + 1, device=self.device)
        kernel_size = int(gpt_seq_length * 2 - 1)
        sigma = 0.5
        # Generate the Gaussian kernel
        self.kernel = gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
            
    def sample_padding_lengths(self, batch_size, gpt_seq_length):
        """
        Samples padding lengths for a batch of sequences based on previously learned Temporal Difference (TD) lambda values.
        By leveraging the TD(λ) values obtained during the training phase, this method fine-tunes the distribution of 
        sequence lengths to exploit the model's learned preferences. This exploitation of learned dynamics aims to enhance 
        the model's learning efficiency by adjusting sequence padding in a way that is informed by past learning experiences, 
        promoting a more targeted approach to sequence handling.

        Args:
        - batch_size (int): The number of sequences in the batch.
        - gpt_seq_length (int): The maximum sequence length supported by the GPT model.

        Returns:
        - torch.Tensor: A tensor of clamped padding lengths for each sequence in the batch.
        """
        max_seq_length = gpt_seq_length

        # Detach and clone learnable TD(λ) values to prevent original tensor modification, setting the last value to 1.
        learnable_td_lambd = self.learnable_td.lambd.detach().clone()
        learnable_td_lambd[-1] = 1

        # Generate a probability distribution for sequence lengths from the TD(λ) values, facilitating a balanced
        # approach to sampling lengths that optimizes the learning process.
        lambda_sequence_probs = create_chain_probabilities_from_lambdas(learnable_td_lambd)
        smoothed_sequence_probs = smooth_probs(lambda_sequence_probs, self.kernel)
        
        # Sample sequence lengths based on the TD(λ)-derived probabilities, aiming to dynamically adjust sequence padding.
        sampled_indices = torch.multinomial(smoothed_sequence_probs, batch_size, replacement=True)
        sampled_lengths = self.sequence_lengths[sampled_indices]

        # Compute padding lengths to adjust sampled sequence lengths to the maximum length, staying within valid bounds.
        padding_seq_length = max_seq_length - sampled_lengths
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def get_padding_lengths(self, padding_mask):
        cur_padding_lengths = padding_mask.size(1) - torch.sum(padding_mask, dim=1)
        return cur_padding_lengths
    
    def apply_sequence_masking(self, padding_mask):
        """
        Identifies positions within the padding mask that should be updated to reflect 
        the dynamically sampled sequence lengths. This process involves determining which 
        parts of each sequence are considered 'padding' based on the sampled lengths.
        """
        batch_size = padding_mask.size(0)
        max_seq_length = padding_mask.size(1)
        cur_padding_lengths = self.get_padding_lengths(padding_mask)
        
        padding_lengths = self.sample_padding_lengths(batch_size, max_seq_length)
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0

        applied_padding_lengths = torch.max(cur_padding_lengths, padding_lengths)

        return padding_mask, applied_padding_lengths