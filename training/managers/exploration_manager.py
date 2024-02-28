import torch
import math
DECAY_UNIT_STEPS = 100000

def compute_lin_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration - initial_exploration) / decay_steps

def compute_exp_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration / initial_exploration) ** (1/decay_steps)

class ExplorationUtils:
    def __init__(self, gpt_seq_length, max_steps, device):
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

        desired_multiplier = max_steps/DECAY_UNIT_STEPS  # Indicates the desired multiplier for the longest vs. shortest sequence selection probability
        base_growth_rate = 2  # Base rate of growth for the sequence length preference
        adjustment_rate = math.log(desired_multiplier) / max(math.log(gpt_seq_length), 1e-8)  # Adjustment to align growth with the desired multiplier
        self.exponential_factor = base_growth_rate + adjustment_rate
        
        self.min_sample_ratio = 1
        self.max_sample_ratio = gpt_seq_length
        self.sequence_lengths = torch.arange(1, gpt_seq_length + 1, device=self.device)
        self.start_ratios = torch.full_like(self.sequence_lengths, fill_value=1)  # Starting from even chance for minimum length
        self.end_ratios = torch.pow(self.sequence_lengths.float(), self.exponential_factor)  # Ending with linear favoring towards longer lengths
            
    def update_exploration_rate(self):  
        if self.decay_mode == "linear":
            self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
        elif self.decay_mode == "exponential":
            self.exploration_rate = max(self.decay_factor * self.exploration_rate, self.min_exploration)

    def get_exploration_rate(self):
        return self.exploration_rate

    def sample_padding_lengths(self, batch_size, gpt_seq_length):
        """
        Dynamically samples sequence lengths, with a bias towards longer sequences as training progresses. 
        This approach is designed to gradually increase the probability of selecting longer sequence lengths, 
        thereby encouraging the model to adapt to and explore the complexities of more extended sequences over time. 
        The adjustment of sampling probabilities is guided by the exploration rate, 
        which is tuned to strike a balance between exploring the potential of longer sequences and exploiting the benefits of previously identified advantageous lengths. 
        This method aims to enhance the model's exposure to a wider range of sequence lengths, with a particular emphasis on extending its competency over longer sequences, 
        which are typically more challenging but potentially more informative.
        """        
        max_seq_length = gpt_seq_length
        
        # Calculate a weighted preference for each sequence length, influenced by the exploration rate.
        # This encourages the model to explore a variety of sequence lengths over time.
        adjusted_ratios = self.exploration_rate * self.start_ratios + (1 - self.exploration_rate) * self.end_ratios
        
        # Normalize adjusted ratios to get probabilities for sampling.
        sequence_probs = adjusted_ratios / adjusted_ratios.sum()
        
        # Sample sequence lengths based on the computed probabilities.
        sampled_indices = torch.multinomial(sequence_probs, batch_size, replacement=True)
        
        sampled_lengths = self.sequence_lengths[sampled_indices]
        
        padding_seq_length = max_seq_length - sampled_lengths
        
        # Ensure sampled lengths are within the specified range.
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