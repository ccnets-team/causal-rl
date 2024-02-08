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
        self.initial_exploration = 0.0
        self.min_exploration = 0.0
        self.decay_percentage = None
        self.decay_factor = None
        # Default exploration rate at the start of training. High value (1.0) promotes initial random exploration.
        self.initial_exploration = 0.0
        # Minimum exploration rate, ensuring some level of exploration is maintained throughout training.
        self.min_exploration = 0.0
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
        
    def sample_dynamic_sequence_lengths(self, batch_size, min_seq_length, max_seq_length):
        # Dynamically samples sequence lengths based on the current exploration rate
        sequence_lengths = torch.arange(min_seq_length, max_seq_length + 1).to(self.device)
        
        sequence_ratios = sequence_lengths/max_seq_length
        
        sequence_probs = sequence_ratios/sequence_ratios.sum()
        
        sampled_indices = torch.multinomial(sequence_probs, batch_size, replacement=True)
        sampled_lengths = sequence_lengths[sampled_indices]
        return sampled_lengths
    
    def apply_exploration_masking(self, padding_mask):
        """
        Applies an effective sequence mask to the given padding mask based on 
        the exploration rate and random sequence lengths.
        """
        batch_size, max_seq_length = padding_mask.size()
        min_seq_length = 1
        
        random_seq_lengths = self.sample_dynamic_sequence_lengths(batch_size, min_seq_length, max_seq_length)

        effective_seq_length = torch.clamp(random_seq_lengths, min_seq_length, max_seq_length)

        padding_seq_length = max_seq_length - effective_seq_length
        # Create a range tensor and apply the mask
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        mask_indices = range_tensor < padding_seq_length.unsqueeze(1)
        padding_mask[mask_indices] = 0.0
        
        return padding_mask