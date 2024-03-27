import torch
from ..utils.distribution_util import create_prob_dist_from_lambdas

class ExplorationManager:
    def __init__(self, gamma_lambda_learner, total_iterations, device):
        self.device = device
        self.total_iterations = total_iterations 
        self.gamma_lambda_learner_for_exploration = gamma_lambda_learner

    def sample_sequence_probabilities(self, input_seq_len):
        """Generates or samples from a probability distribution for sequence lengths based on TD(λ) values.
        Optionally smooths the distribution using a Gaussian kernel for a more generalized probability curve."""
        lambd = self.gamma_lambda_learner_for_exploration.get_lambdas(seq_range = (-input_seq_len, None))
        lambda_sequence_probs = create_prob_dist_from_lambdas(lambd)
        return lambda_sequence_probs

    def sample_content_lengths(self, batch_size, input_seq_len):
        """Samples padding lengths for a batch of sequences based on a probability distribution,
        allowing for dynamic adjustments to sequence padding based on learned TD(λ) values."""
        sampled_sequence_probs = self.sample_sequence_probabilities(input_seq_len)
        sampled_indices = torch.multinomial(sampled_sequence_probs, batch_size, replacement=True).long()
        sampled_lengths = sampled_indices + 1
        return sampled_lengths

    def get_optimal_content_lengths(self, input_seq_len):
        """Calculates optimal padding lengths for sequences, aiming to align with the most likely
        sequence length based on the smoothed probability distribution."""
        sampled_sequence_probs = self.sample_sequence_probabilities(input_seq_len)
        sampled_indices = torch.argmax(sampled_sequence_probs, dim=0).long()
        content_lengths = sampled_indices + 1
        return content_lengths

    def apply_sequence_masking(self, padding_mask):
        """
        Identifies positions within the padding mask that should be updated to reflect 
        the dynamically sampled sequence lengths. This process involves determining which 
        parts of each sequence are considered 'padding' based on the sampled lengths.
        """
        batch_size = padding_mask.size(0)
        seq_len = padding_mask.size(1)
        
        sampled_content_lengths = self.sample_content_lengths(batch_size, seq_len)
        sampled_padding_lengths = seq_len - sampled_content_lengths
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(seq_len, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < sampled_padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0
        
        return padding_mask