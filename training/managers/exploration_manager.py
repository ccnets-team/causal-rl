import torch
import math

def compute_lin_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration - initial_exploration) / decay_steps

def compute_exp_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration / initial_exploration) ** (1/decay_steps)

class ExplorationUtils:
    def __init__(self, max_steps, device):
        self.device = device
        self.decay_percentage = None
        self.decay_factor = None
        # Default exploration rate at the start of training. High value (1.0) promotes initial random exploration.
        self.initial_exploration = 1.0
        # Minimum exploration rate, ensuring some level of exploration is maintained throughout training.
        self.min_exploration = 0.01
        # Defines the rate at which exploration decreases. A value of 0.8 means 80% of initial exploration will be reduced over max_steps.
        self.decay_percentage = 0.8
        # Default decay mode. 'linear' means exploration rate decreases linearly over time.
        self.decay_factor = compute_lin_decay_factor(self.initial_exploration, self.min_exploration, max_steps, self.decay_percentage)

        self.decay_mode = 'linear'
        self.exploration_rate = self.initial_exploration
        
    def update_exploration_rate(self):  
        if self.decay_mode == "linear":
            self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
        elif self.decay_mode == "exponential":
            self.exploration_rate = max(self.decay_factor * self.exploration_rate, self.min_exploration)

    def get_exploration_rate(self):
        return self.exploration_rate

    def sample_padding_lengths(self, batch_size, gpt_seq_length):
        """
        Samples sequence lengths within the specified range, adjusting probabilities 
        based on the exploration rate to promote varied sequence sampling. This method 
        encourages exploration by dynamically adjusting the likelihood of selecting different 
        sequence lengths, factoring in the current exploration rate to balance between 
        exploring new lengths and exploiting known advantageous lengths.
        """
        min_seq_length = 1
        max_seq_length = gpt_seq_length
        
        sequence_lengths = torch.arange(min_seq_length, max_seq_length + 1, device=self.device)
        
        # Compute relative lengths as ratios of the maximum sequence length.
        sequence_ratios = sequence_lengths.float() / max_seq_length
        
        adjusted_ratios = torch.pow(sequence_ratios, 1 / max(self.exploration_rate, 1e-8))
        
        # Normalize adjusted ratios to get probabilities for sampling.
        sequence_probs = adjusted_ratios / adjusted_ratios.sum()
        
        # Sample sequence lengths based on the computed probabilities.
        sampled_indices = torch.multinomial(sequence_probs, batch_size, replacement=True)
        
        sampled_lengths = sequence_lengths[sampled_indices]
        
        padding_seq_length = max_seq_length - sampled_lengths
        
        # Ensure sampled lengths are within the specified range.
        return torch.clamp(padding_seq_length, 0, max_seq_length - 1)

    def apply_sequence_masking(self, padding_mask):
        """
        Identifies positions within the padding mask that should be updated to reflect 
        the dynamically sampled sequence lengths. This process involves determining which 
        parts of each sequence are considered 'padding' based on the sampled lengths.
        """
        batch_size = padding_mask.size(0)
        max_seq_length = padding_mask.size(1)
        cur_padding_lengths = max_seq_length - torch.sum(padding_mask, dim=1)
        
        padding_lengths = self.sample_padding_lengths(batch_size, max_seq_length)
        
        # Calculate the number of padding slots needed for each sequence.
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        # Determine which slots will be marked as padding.
        sampled_padding_slots = range_tensor < padding_lengths.unsqueeze(1)

        padding_mask[sampled_padding_slots] = 0.0

        applied_padding_lengths = torch.max(cur_padding_lengths, padding_lengths)

        return padding_mask, applied_padding_lengths