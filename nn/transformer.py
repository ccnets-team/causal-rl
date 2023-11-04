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

def create_forward_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device) * -float('inf'), diagonal=1)

def create_reverse_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device) * -float('inf'), diagonal=-1)

def create_memory_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Create a mask with -inf everywhere except the diagonal, where we have zeros
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask.fill_diagonal_(0.0)
    return mask

def expand_attention_mask(mask: torch.Tensor, src_key_padding_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    assert mask.dim() == 2, "Attention mask must be 2D with shape [seq_len, seq_len]"
    assert src_key_padding_mask.dim() == 2, "Source key padding mask must be 2D with shape [batch_size, seq_len]"
    
    # Ensure mask and src_key_padding_mask are compatible for broadcasting
    src_key_padding_mask = src_key_padding_mask.unsqueeze(1)
    
    # Extend mask by considering src_key_padding_mask
    combined_mask = mask.unsqueeze(0) + src_key_padding_mask
    
    # Find positions in the combined mask that have all -inf values
    all_inf_positions = (combined_mask == float('-inf')).all(dim=-1)
    
    # Create a diagonal mask using torch.eye
    diag_mask = torch.eye(combined_mask.size(-1), device=combined_mask.device).unsqueeze(0).bool()
    
    # Expand all_inf_positions to [B, S, S]
    expanded_all_inf_positions = all_inf_positions.unsqueeze(-1) & diag_mask
    
    # Set the diagonal elements to 0.0 where all values in the last dimension are -inf
    combined_mask[expanded_all_inf_positions] = 0.0
    
    return combined_mask.repeat(num_heads, 1, 1)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layer, hidden_size, reverse = False, num_heads: int = 8):
        super(TransformerDecoder, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.reverse = reverse
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layer)
    def forward(self, x, memory, mask=None):
        seq_len = x.size(1)
        device = x.device  # Fetch the device from the input tensor

        pos_enc = positional_encoding(seq_len, self.hidden_size).to(device)
        x = x + pos_enc  # Add positional encoding to the input
        
        if self.reverse:
            tgt_mask = create_reverse_mask(seq_len, device).type_as(x)
        else:
            tgt_mask = create_forward_mask(seq_len, device).type_as(x)
        memory_mask = create_memory_mask(seq_len, device).type_as(x)

        if mask is not None:
            src_key_padding_mask_bool = mask.squeeze(dim=-1).bool()
            src_key_padding_mask = torch.where(src_key_padding_mask_bool, torch.tensor(0.0, dtype=x.dtype, device = device), \
                torch.tensor(-float('inf'), dtype=x.dtype, device = device))
            tgt_mask = expand_attention_mask(tgt_mask, src_key_padding_mask, self.num_heads)
            memory_mask = expand_attention_mask(memory_mask, src_key_padding_mask, self.num_heads)

        y = self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return y
    
class ReverseTransformerDecoder(TransformerDecoder):
    def __init__(self, num_layer, hidden_size):
        super(ReverseTransformerDecoder, self).__init__(num_layer, hidden_size, reverse = True)   