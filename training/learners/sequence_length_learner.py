import os
import torch
from ..utils.sequence_util import calculate_learnable_sequence_length
from ..utils.sequence_util import MIN_TD_EXTENSION_STEPS, TD_EXTENSION_RATIO

class SequenceLengthLearner:
    def __init__(self, gamma_lambda_learner_for_seq, max_seq_len):
        self.max_seq_len = max_seq_len
        self.min_seq_len = max_seq_len // 4  # Minimum sequence length set to 1/4 of max_seq_len
        self.input_seq_len = max_seq_len//2  # Initially set to max_seq_len; updated based on learnings
        self.gamma_lambda_learner_for_seq = gamma_lambda_learner_for_seq
        self.init_half_seq_len = False
        
        # Initialize TD extension steps
        self._update_td_extension_steps()

    def _update_td_extension_steps(self):
        """Update TD extension steps based on the current input sequence length."""
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.total_seq_len = self.input_seq_len + self.td_extension_steps

    # Getter methods for sequence length properties
    def get_input_seq_len(self):
        return self.input_seq_len

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_min_seq_len(self):
        return self.min_seq_len

    def get_total_seq_len(self):
        return self.total_seq_len

    def get_max_td_extension_steps(self):
        return max(self.max_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)

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
            