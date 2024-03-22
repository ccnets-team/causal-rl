import torch
import torch.nn.functional as F

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

class ExplorationUtils:
    def __init__(self, gpt_seq_length, learnable_td, device):
        self.device = device
        self.learnable_td = learnable_td
        self.gpt_seq_length = gpt_seq_length
        self.sequence_lengths = torch.arange(1, gpt_seq_length + 1, device=self.device)

    def sample_sequence_probabilities(self):
        """Generates or samples from a probability distribution for sequence lengths based on TD(λ) values.
        Optionally smooths the distribution using a Gaussian kernel for a more generalized probability curve."""
        learnable_td_lambda = self.learnable_td.lambd
        lambda_sequence_probs = create_prob_dist_from_lambdas(learnable_td_lambda)
        return lambda_sequence_probs

    def sample_padding_lengths(self, batch_size, max_seq_length):
        """Samples padding lengths for a batch of sequences based on a probability distribution,
        allowing for dynamic adjustments to sequence padding based on learned TD(λ) values."""
        sampled_sequence_probs = self.sample_sequence_probabilities()
        sampled_indices = torch.multinomial(sampled_sequence_probs, batch_size, replacement=True)
        sampled_lengths = self.sequence_lengths[sampled_indices]

        padding_seq_length = max_seq_length - sampled_lengths
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def get_optimal_padding_lengths(self):
        """Calculates optimal padding lengths for sequences, aiming to align with the most likely
        sequence length based on the smoothed probability distribution."""
        max_seq_length = self.gpt_seq_length
        sampled_sequence_probs = self.sample_sequence_probabilities()
        sampled_indices = torch.argmax(sampled_sequence_probs, dim=0)
        padding_seq_length = max_seq_length - self.sequence_lengths[sampled_indices]
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def get_current_padding_lengths(self, padding_mask):
        """Determines current padding lengths for each sequence based on the padding mask."""
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
        cur_padding_lengths = self.get_current_padding_lengths(padding_mask)
        
        padding_lengths = self.sample_padding_lengths(batch_size, max_seq_length)
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0

        applied_padding_lengths = torch.max(cur_padding_lengths, padding_lengths)

        return padding_mask, applied_padding_lengths