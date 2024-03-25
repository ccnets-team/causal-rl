import os
import torch
from ..utils.sequence_util import calculate_learnable_sequence_length 
from ..utils.tensor_util import keep_right_tensor_sequences

MIN_TD_EXTENSION_STEPS = 4
MIN_GPT_SEQUENCE_LENGTH = 4
TD_EXTENSION_RATIO = 4  # Represents the divisor for calculating the extension steps
INITIAL_SEQ_LEN_FRACTION = 1  # Fraction of max_seq_len used to set the initial input sequence length
SEQUENCE_LENGTH_UPDATE_INTERVAL = 1000

class SequenceLengthLearner:
    def __init__(self, gamma_lambda_learner, max_seq_len, device):
        self.gamma_lambda_learner_for_seq = gamma_lambda_learner
        self.max_seq_len = max_seq_len
        self.input_seq_len = int(INITIAL_SEQ_LEN_FRACTION *max_seq_len)
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
        self.device = device
        
    def get_input_seq_len(self):
        """Returns the current GPT input sequence length."""
        return self.input_seq_len

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_total_seq_len(self):
        return self.input_seq_len + self.td_extension_steps

    def get_td_extension_steps(self):
        return self.td_extension_steps

    def update_learnable_length(self):
        """Updates and returns the learnable length based on the current state of learnable_td."""
        required_seq_len = calculate_learnable_sequence_length(self.gamma_lambda_learner_for_seq.lambd, self.get_input_seq_len())
        self.input_seq_len = min(max(required_seq_len, MIN_GPT_SEQUENCE_LENGTH), self.get_max_seq_len())
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
    
    def truncate_to_input_seq_len(self, *tensors, use_td_extension_steps=False):
        """
        Truncates the given tensors to only include the learnable_length amount from the right side,
        effectively focusing on the most relevant portions of each sequence for learning.
        
        :param tensors: Variable number of tensors to be truncated.
        :return: A single tensor directly or a tuple of tensors truncated to the learnable_length.
        """
        if use_td_extension_steps:
            total_td_len = self.get_total_seq_len()
            return keep_right_tensor_sequences(total_td_len, *tensors)
        else:
            # Unpack the tensors when passing them to the function to properly handle multiple tensors.
            input_seq_len = self.get_input_seq_len()
            return keep_right_tensor_sequences(input_seq_len, *tensors)
        
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
            