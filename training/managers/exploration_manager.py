import torch
from ..utils.distribution_util import create_prob_dist_from_lambdas

def compute_lin_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration - initial_exploration) / decay_steps

class ExplorationManager:
    def __init__(self, max_seq_len, get_input_seq_len_function, learnable_td, total_iterations, device):
        self.device = device
        self.initial_exploration = 1.0
        self.min_exploration = 0.01 
        self.decay_percentage = 0.8
        self.total_iterations = total_iterations 
        self.decay_factor = compute_lin_decay_factor(self.initial_exploration, self.min_exploration, self.total_iterations, self.decay_percentage)
        self.exploration_rate = self.initial_exploration
        
        self.get_input_seq_len_function = get_input_seq_len_function
        self.learnable_td_for_exploration = learnable_td
        self.sequence_lengths = torch.arange(1, max_seq_len + 1, device=self.device)

        self.smoothing_scale = 8
        self.initial_sigma = max_seq_len/self.smoothing_scale

    def get_exploration_rate(self):
        return self.exploration_rate
    
    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
            
    def sample_sequence_probabilities(self, input_seq_len, use_smoothed_probs=False):
        """Generates or samples from a probability distribution for sequence lengths based on TD(λ) values.
        Optionally smooths the distribution using a Gaussian kernel for a more generalized probability curve."""
        learnable_td_lambda = self.learnable_td_for_exploration.lambd[-input_seq_len:]
        lambda_sequence_probs = create_prob_dist_from_lambdas(learnable_td_lambda)
        
        if use_smoothed_probs:
            sequence_lengths = self.sequence_lengths[:input_seq_len]
            sequence_probs = sequence_lengths/sequence_lengths.sum().clamp_min(1e-8)
            smoothed_sequence_probs = self.exploration_rate * sequence_probs + (1 - self.exploration_rate) * lambda_sequence_probs  
        else:
            smoothed_sequence_probs = lambda_sequence_probs
        return smoothed_sequence_probs

    def sample_content_lengths(self, batch_size):
        """Samples padding lengths for a batch of sequences based on a probability distribution,
        allowing for dynamic adjustments to sequence padding based on learned TD(λ) values."""
        input_seq_len = self.get_input_seq_len_function()
        sampled_sequence_probs = self.sample_sequence_probabilities(input_seq_len, use_smoothed_probs=True)
        sampled_indices = torch.multinomial(sampled_sequence_probs, batch_size, replacement=True)
        sequence_lengths = self.sequence_lengths[:input_seq_len]
        sampled_lengths = sequence_lengths[sampled_indices]
        return sampled_lengths

    def get_optimal_content_lengths(self):
        """Calculates optimal padding lengths for sequences, aiming to align with the most likely
        sequence length based on the smoothed probability distribution."""
        input_seq_len = self.get_input_seq_len_function()
        sampled_sequence_probs = self.sample_sequence_probabilities(input_seq_len, use_smoothed_probs=False)
        sampled_indices = torch.argmax(sampled_sequence_probs, dim=0)
        sequence_lengths = self.sequence_lengths[:input_seq_len]
        content_lengths = sequence_lengths[sampled_indices]
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