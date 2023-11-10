import math
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class _GPTBase(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads, dropout):
        super(_GPTBase, self).__init__()   
        config = GPT2Config(
            vocab_size=1,  # This should be set to your actual vocab size
            n_embd=hidden_size,
            n_layer=num_layer,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.net = GPT2Model(config)

    def get_sinusoidal_positional_encodings(self, max_sequence_length, hidden_size):
        # Create a matrix of shape [max_sequence_length, hidden_size] with sinusoidal patterns
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        sinusoidal_encodings = torch.zeros(max_sequence_length, hidden_size)
        sinusoidal_encodings[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_encodings[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_encodings

class GPT(_GPTBase):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8, dropout = 0.0, max_sequence_length=100):
        super(GPT, self).__init__(num_layer, hidden_size, num_heads, dropout)
        # Pre-compute sinusoidal positional encodings
        self.register_buffer('positional_encodings', 
                             self.get_sinusoidal_positional_encodings(max_sequence_length, hidden_size))

    def forward(self, input_tensor, mask=None):
        attention_mask = mask.long() if mask is not None else None
        
        # Get positional encodings for the input sequence length
        position_embeddings = self.positional_encodings[:input_tensor.size(1), :]

        # Add the positional embeddings to the input embeddings
        input_tensor = input_tensor + position_embeddings.unsqueeze(0)

        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
        return output_tensor
