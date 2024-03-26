import os
import torch
from ..utils.sequence_util import calculate_learnable_sequence_length 
from ..utils.tensor_util import keep_right_tensor_sequences

MIN_TD_EXTENSION_STEPS = 4
TD_EXTENSION_RATIO = 4  # Represents the divisor for calculating the extension steps
INITIAL_SEQ_LEN_FRACTION = 2/3  # Fraction of max_seq_len used to set the initial input sequence length
SEQUENCE_LENGTH_UPDATE_INTERVAL = 1000

class SequenceLengthLearner:
    def __init__(self, max_seq_len, device):
        self.max_seq_len = max_seq_len
        self.min_seq_len = max_seq_len//2
        self.input_seq_len = int(INITIAL_SEQ_LEN_FRACTION * max_seq_len)
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.tot_seq_len = self.input_seq_len + self.td_extension_steps
        self.device = device
        
    def get_input_seq_len(self):
        """Returns the current GPT input sequence length."""
        return self.input_seq_len

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_min_seq_len(self):
        return self.min_seq_len

    def get_total_seq_len(self):
        return self.tot_seq_len

    def get_td_extension_steps(self):
        return self.td_extension_steps

    def update_learnable_length(self, lambd):
        """Updates and returns the learnable length based on the current state of learnable_td."""
        optimal_length = calculate_learnable_sequence_length(lambd)
        
        input_seq_len = min(max(optimal_length, self.min_seq_len), self.max_seq_len)
        td_extension_steps = max(input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.input_seq_len = input_seq_len
        self.td_extension_steps = td_extension_steps
        self.tot_seq_len = input_seq_len + td_extension_steps 
        
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
            