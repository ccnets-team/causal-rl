import torch
from ..utils.sequence_util import create_prob_dist_from_lambdas

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

    def sample_content_lengths(self, batch_size):
        """Samples padding lengths for a batch of sequences based on a probability distribution,
        allowing for dynamic adjustments to sequence padding based on learned TD(λ) values."""
        sampled_sequence_probs = self.sample_sequence_probabilities()
        sampled_indices = torch.multinomial(sampled_sequence_probs, batch_size, replacement=True)
        sampled_lengths = self.sequence_lengths[sampled_indices]
        return sampled_lengths

    def get_optimal_content_lengths(self):
        """Calculates optimal padding lengths for sequences, aiming to align with the most likely
        sequence length based on the smoothed probability distribution."""
        sampled_sequence_probs = self.sample_sequence_probabilities()
        sampled_indices = torch.argmax(sampled_sequence_probs, dim=0)
        content_lengths = self.sequence_lengths[sampled_indices]
        return content_lengths

    def get_content_lengths(self, padding_mask):
        """Determines current padding lengths for each sequence based on the padding mask."""
        return torch.sum(padding_mask, dim=1)
    
    def apply_sequence_masking(self, padding_mask):
        """
        Identifies positions within the padding mask that should be updated to reflect 
        the dynamically sampled sequence lengths. This process involves determining which 
        parts of each sequence are considered 'padding' based on the sampled lengths.
        """
        batch_size = padding_mask.size(0)
        seq_len = padding_mask.size(1)
        cur_content_lengths = self.get_content_lengths(padding_mask)
        
        sampled_content_lengths = self.sample_content_lengths(batch_size)
        sampled_padding_lengths = seq_len - sampled_content_lengths
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(seq_len, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < sampled_padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0

        selected_content_lengths = torch.min(cur_content_lengths, sampled_content_lengths)
        
        return padding_mask, selected_content_lengths