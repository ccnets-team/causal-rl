import numpy as np
import torch

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
        self.decay_mode = None
        self.decay_factor = None
        # Default exploration rate at the start of training. High value (1.0) promotes initial random exploration.
        self.initial_exploration = 1.0
        # Minimum exploration rate, ensuring some level of exploration is maintained throughout training.
        self.min_exploration = 0.01
        # Defines the rate at which exploration decreases. A value of 0.8 means 80% of initial exploration will be reduced over max_steps.
        self.decay_percentage = 0.8
        # Default decay mode. 'linear' means exploration rate decreases linearly over time.
        self.decay_mode = "linear"
        self.decay_factor = compute_lin_decay_factor(self.initial_exploration, self.min_exploration, max_steps, self.decay_percentage)
        self.exploration_rate = self.initial_exploration

        # The `max_exploit_factor` determines the extent to which longer sequences are favored during the exploration phase of training. 
        # A higher value of `max_exploit_factor` significantly increases the likelihood of selecting longer sequences as the exploration rate decreases, 
        # pushing the model towards exploitation by focusing on more extended sequences. This parameter is essential for tailoring the exploration-exploitation 
        # balance, especially in environments where longer sequences provide more information for policy development and value estimation. 
        # It enables the model to transition smoothly from initial exploration to focused exploitation, enhancing learning outcomes.
        self.max_exploit_factor = 10.0
        
    def update_exploration_rate(self):
        if self.decay_mode == "linear":
            self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
        elif self.decay_mode == "exponential":
            self.exploration_rate = max(self.decay_factor * self.exploration_rate, self.min_exploration)

    def get_exploration_rate(self):
        return self.exploration_rate
        
    def sample_dynamic_sequence_lengths(self, batch_size, min_seq_length, max_seq_length, exploration_rate):
        """
        Dynamically samples sequence lengths based on the current exploration rate, leveraging the `max_exploit_factor` to adjust the preference 
        towards longer sequences as exploration diminishes. Initially, when exploration is high, sequence lengths are sampled more uniformly, allowing the model 
        to explore a variety of sequence lengths broadly. As exploration decreases, the method increasingly biases the sampling towards longer sequences, 
        enhancing the model's ability to exploit its accumulated knowledge by focusing on more extended and potentially informative sequences. 
        This dynamic adjustment of sequence length sampling is instrumental in optimizing the model's performance over time, ensuring it can adapt its focus 
        based on the ongoing exploration-exploitation trade-off.
        """
        # Create an array of possible sequence lengths
        possible_lengths = np.arange(min_seq_length, max_seq_length + 1)
        
        # Calculate a linearly changing ratio across the sequence lengths
        bias_ratio = np.arange(0, max_seq_length) / (max_seq_length - 1)
        
        # Adjust the gradient weight based on the exploration rate
        gradient_biased_weights = possible_lengths * (1 + bias_ratio * (self.max_exploit_factor - 1) * (1 - exploration_rate))
        
        # Normalize the weights to sum to 1
        weights = gradient_biased_weights / gradient_biased_weights.sum()

        # Sample sequence lengths based on these normalized weights
        sampled_lengths = np.random.choice(possible_lengths, size=batch_size, p=weights)
        
        return sampled_lengths

    def apply_exploration_masking(self, padding_mask):
        """
        Applies an effective sequence mask to the given padding mask based on 
        the exploration rate and random sequence lengths.
        """
        batch_size, max_seq_length = padding_mask.size()
        min_seq_length = 1
        exploration_rate = self.exploration_rate
        
        random_seq_lengths = self.sample_dynamic_sequence_lengths(batch_size, min_seq_length, max_seq_length, exploration_rate)

        effective_seq_length = torch.clamp(torch.tensor(random_seq_lengths, device=self.device), min_seq_length, max_seq_length)

        padding_seq_length = max_seq_length - effective_seq_length
        # Create a range tensor and apply the mask
        range_tensor = torch.arange(max_seq_length, device=self.device).expand_as(padding_mask)
        mask_indices = range_tensor < padding_seq_length.unsqueeze(1)
        padding_mask[mask_indices] = 0.0
        
        return padding_mask