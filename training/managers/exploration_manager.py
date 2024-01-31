import torch
import math

def compute_lin_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration - initial_exploration) / decay_steps

def compute_exp_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration / initial_exploration) ** (1/decay_steps)

class BoltzmannExploration:
    def __init__(self):
        self.tau = 0.05
        self.min_temperature = 0.005
        # Adjusted decay rate as per the derived formula
        self.decay_rate = -math.log(self.min_temperature/self.tau)
            
    def apply(self, x, exploration_rate):
        # Computing temperature based on the adjusted decay rate
        temperature = max(self.tau * math.exp(-self.decay_rate * (1 - exploration_rate)), self.min_temperature)
        # Assuming x has some specific use and computation. Placeholder for actual computation with x.
        boltzmann_probs = torch.softmax(x / temperature, dim=-1)
        return boltzmann_probs
    
class ExplorationUtils:
    def __init__(self, max_steps, device):
        self.device = device
        self.initial_exploration = 0.0
        self.min_exploration = 0.0
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
        self.boltzmann_exploration = BoltzmannExploration()
        
    def update_exploration_rate(self):  
        if self.decay_mode == "linear":
            self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
        elif self.decay_mode == "exponential":
            self.exploration_rate = max(self.decay_factor * self.exploration_rate, self.min_exploration)

    def get_exploration_rate(self):
        return self.exploration_rate
        
    def sample_dynamic_sequence_lengths(self, batch_size, min_seq_length, max_seq_length):
        # Dynamically samples sequence lengths based on the current exploration rate
        possible_lengths = torch.arange(min_seq_length, max_seq_length + 1).to(self.device)
        
        sequence_probs = possible_lengths/possible_lengths.sum()
        
        # Apply Boltzmann exploration to adjust the temperature
        boltzmann_probs = self.boltzmann_exploration.apply(sequence_probs, self.exploration_rate)
        
        adjustment_factors = sequence_probs/torch.softmax(sequence_probs, dim=-1)
        
        adjustment_ratios = boltzmann_probs* adjustment_factors
        
        adjustment_probs = adjustment_ratios/adjustment_ratios.sum()
        
        # Sample sequence lengths based on Boltzmann probabilities
        sampled_indices = torch.multinomial(adjustment_probs, batch_size, replacement=True)
        sampled_lengths = possible_lengths[sampled_indices]
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