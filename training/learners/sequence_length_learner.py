import os
import torch
from ..utils.sequence_util import calculate_learnable_sequence_length

# Constants for TD extension and sequence length calculation
MIN_TD_EXTENSION_STEPS = 1
TD_EXTENSION_RATIO = 4  # Used to calculate the extension steps
INITIAL_SEQ_LEN_FRACTION = 1/2  # Fraction of max_seq_len for initial sequence length
SEQUENCE_LENGTH_UPDATE_INTERVAL = 1000

class SequenceLengthLearner:
    def __init__(self, max_seq_len, device):
        self.max_seq_len = max_seq_len
        self.min_seq_len = max_seq_len // 2
        self.input_seq_len = self.max_seq_len  # Initially set to max_seq_len; updated in first call
        self.device = device
        # Initialization flag to check if _initialize has been called
        self.is_init = False
        self._update_td_extension_steps()

    def _initialize(self):
        """Initialize or reset the input sequence length to a fraction of the maximum length."""
        self.input_seq_len = int(self.max_seq_len * INITIAL_SEQ_LEN_FRACTION)
        self._update_td_extension_steps()

    def _update_td_extension_steps(self):
        """Update the TD extension steps based on the current input sequence length."""
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.tot_seq_len = self.input_seq_len + self.td_extension_steps

    def update_learnable_length(self, lambd):
        """Updates learnable sequence length based on lambda value and adjusts TD extension steps."""
        if not self.is_init:
            self._initialize()
            self.is_init = True
        else:
            optimal_length = calculate_learnable_sequence_length(lambd)
            self.input_seq_len = min(max(optimal_length, self.min_seq_len), self.max_seq_len)
        
        self._update_td_extension_steps()

    # Getter methods for sequence length properties
    def get_input_seq_len(self):
        return self.input_seq_len

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_min_seq_len(self):
        return self.min_seq_len

    def get_total_seq_len(self):
        return self.tot_seq_len

    def get_td_extension_steps(self):
        return self.td_extension_steps
        
    def save(self, path):
        """Saves the learner's state using PyTorch's serialization."""
        save_dict = {
            'max_seq_len': self.max_seq_len,
            'input_seq_len': self.input_seq_len,
            'td_extension_steps': self.td_extension_steps
        }
        torch.save(save_dict, path)

    def load(self, path):
        """Loads the learner's state using PyTorch's serialization."""
        if os.path.isfile(path):
            load_dict = torch.load(path, map_location=self.device)
            self.max_seq_len = load_dict['max_seq_len']
            self.input_seq_len = load_dict['input_seq_len']
            self.td_extension_steps = load_dict['td_extension_steps']
            