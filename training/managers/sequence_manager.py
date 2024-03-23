import torch
from ..utils.sequence_util import calculate_learnable_sequence_length 
from ..utils.tensor_util import keep_right_tensor_sequences
    
class SequenceManager:
    def __init__(self, learnable_td, gpt_seq_len, td_seq_len):
        self.learnable_td = learnable_td
        self.initial_gpt_seq_length = gpt_seq_len
        self.learnable_gpt_seq_len = gpt_seq_len
        self.learnable_td_seq_len = td_seq_len
        self.additional_td_length = max(td_seq_len - gpt_seq_len, 0)  # Renamed from diff_len

    def get_learnable_sequence_length(self):
        return self.learnable_gpt_seq_len, self.learnable_td_seq_len
    
    def update_learnable_length(self):
        """Updates and returns the learnable length based on the current state of learnable_td."""
        self.learnable_gpt_seq_len = calculate_learnable_sequence_length(self.learnable_td.lambd)
        # Adjust the learnable TD length based on the newly calculated GPT length and the additional TD length
        self.learnable_td_seq_len = self.learnable_gpt_seq_len + self.additional_td_length
    
    def truncate_to_learnable_length(self, *tensors):
        """
        Truncates the given tensors to only include the learnable_length amount from the right side,
        effectively focusing on the most relevant portions of each sequence for learning.
        
        :param tensors: Variable number of tensors to be truncated.
        :return: A single tensor directly or a tuple of tensors truncated to the learnable_length.
        """
        # Unpack the tensors when passing them to the function to properly handle multiple tensors.
        return keep_right_tensor_sequences(self.learnable_gpt_seq_len, *tensors)
