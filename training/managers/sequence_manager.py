import torch.nn.functional as F
from ..utils.sequence_util import calculate_learnable_sequence_length 
from ..utils.tensor_util import keep_right_tensor_sequences

MIN_TD_EXTENSION_STEPS = 4
MIN_GPT_SEQUENCE_LENGTH = 4
TD_EXTENSION_RATIO = 4  # Represents the divisor for calculating the extension steps

class SequenceManager:
    def __init__(self, learnable_td, max_seq_len):
        self.learnable_td = learnable_td
        self.max_seq_len = max_seq_len
        self.input_seq_len = max_seq_len//2
        self.td_extension_steps = max(max_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)

    def get_input_seq_len(self):
        """Returns the current GPT input sequence length."""
        return self.input_seq_len

    def get_td_extension_steps(self):
        return self.td_extension_steps
    
    def update_learnable_length(self):
        """Updates and returns the learnable length based on the current state of learnable_td."""
        required_seq_len = calculate_learnable_sequence_length(self.learnable_td.lambd, self.input_seq_len)
        self.input_seq_len = min(max(required_seq_len, MIN_GPT_SEQUENCE_LENGTH), self.max_seq_len)
        self.td_extension_steps = max(self.input_seq_len // TD_EXTENSION_RATIO, MIN_TD_EXTENSION_STEPS)
    
    def truncate_to_input_seq_len(self, *tensors, use_td_length=False):
        """
        Truncates the given tensors to only include the learnable_length amount from the right side,
        effectively focusing on the most relevant portions of each sequence for learning.
        
        :param tensors: Variable number of tensors to be truncated.
        :return: A single tensor directly or a tuple of tensors truncated to the learnable_length.
        """
        if use_td_length:
            total_td_len = self.td_extension_steps + self.input_seq_len
            return keep_right_tensor_sequences(total_td_len, *tensors)
        else:
            # Unpack the tensors when passing them to the function to properly handle multiple tensors.
            return keep_right_tensor_sequences(self.input_seq_len, *tensors)
        
    def extend_to_input_seq_len(self, tensor):
        """
        Extends the given tensor to match the original sequence length or a specified target length by padding it from the left.
        
        :param tensor: The tensor to be extended, typically representing an action or output from a model.
        :param target_length: Optional; the desired length to extend the tensor to. If not specified, uses the initial GPT sequence length.
        :return: The tensor extended to the specified or initial sequence length.
        """
        current_length = tensor.size(1)
        if current_length >= self.input_seq_len:
            return tensor  # No extension needed if the current length is already appropriate

        # Calculate the amount of padding needed on the left
        padding_length = self.input_seq_len - current_length
        # Perform the padding
        padded_tensor = F.pad(tensor, (padding_length, 0, 0, 0), 'constant', 0)
        
        return padded_tensor