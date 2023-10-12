'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch
import torch.nn as nn
import numpy as np

def positional_encoding(max_seq_size: int, d_model: int) -> torch.Tensor:
    """Creates positional encodings to be added to input tensors."""
    pos_enc = torch.zeros(max_seq_size, d_model)
    pos = torch.arange(max_seq_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

class TransformerEncoder(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8):
        super(TransformerEncoder, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layer)

    def forward(self, x):
        # Check if the input tensor is 2D, and if so, unsqueeze it to make it 3D
        if len(x.shape) == 2:
            x.unsqueeze_(dim=1)
            was_unsqueezed = True
        else:
            was_unsqueezed = False

        seq_len = x.size(1)
        device = x.device  # Fetch the device from the input tensor
        
        pos_enc = positional_encoding(seq_len, self.hidden_size).to(device)[:seq_len, :]
        mask = create_causal_mask(seq_len, device)[:seq_len, :seq_len] 
        
        x_embed = x + pos_enc

        y = self.encoder(x_embed, mask=mask)
        
        # If the input tensor was unsqueezed, we squeeze the output tensor to return it to its original shape
        if was_unsqueezed:
            y.squeeze_(dim=1)

        return y