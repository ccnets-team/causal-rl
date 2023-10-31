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

def correct_attention_mask(mask: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
    assert mask.dim() == 2, "Attention mask must be 2D with shape [seq_len, seq_len]"
    assert src_key_padding_mask.dim() == 2, "Source key padding mask must be 2D with shape [batch_size, seq_len]"
    
    corrected_mask = mask.clone()
    padding_mask = src_key_padding_mask.unsqueeze(1).to(mask.dtype)
    combined_mask = corrected_mask.unsqueeze(0) + padding_mask
    
    # Find positions in the combined mask that have all zero attention weights
    positions_to_correct = (combined_mask.sum(dim=-1) <= 0).all(dim=0)

    # For each position, if it is completely masked out across all batches, 
    # unmask the first non-padded position in any of the batches
    for pos in range(positions_to_correct.size(0)):
        if positions_to_correct[pos]:
            corrected_mask[pos,:] = 0.0
                    
    return corrected_mask

class TransformerEncoder(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8):
        super(TransformerEncoder, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layer)
            
    def forward(self, x, reverse=False, mask=None):
        # Check if the input tensor is 2D, and if so, unsqueeze it to make it 3D
        
        if len(x.shape) == 2:
            x.unsqueeze_(dim=1)
            was_unsqueezed = True
        else:
            was_unsqueezed = False
                    
        seq_len = x.size(1)
        device = x.device  # Fetch the device from the input tensor
        
        pos_enc = positional_encoding(seq_len, self.hidden_size).to(device)
        x = x + pos_enc  # Add positional encoding to the input
        
        if reverse:
            atten_mask = create_reverse_mask(seq_len, device).type_as(x)
        else:
            atten_mask = create_forward_mask(seq_len, device).type_as(x)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask_bool = mask.squeeze(dim=-1).bool()
            src_key_padding_mask = torch.where(src_key_padding_mask_bool, torch.tensor(0.0, dtype=x.dtype), torch.tensor(-float('inf'), dtype=x.dtype))
            atten_mask = correct_attention_mask(atten_mask, src_key_padding_mask)
            
        y = self.encoder(x, mask=atten_mask, src_key_padding_mask=src_key_padding_mask)
        
        # If the input was unsqueezed, squeeze it back to its original shape
        if was_unsqueezed:
            y.squeeze_(dim=1)
            
        return y
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8):
        super(TransformerDecoder, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layer)

    def forward(self, x, memory, reverse=False, mask=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            was_unsqueezed = True
        else:
            was_unsqueezed = False

        seq_len = x.size(1)
        device = x.device
        pos_enc = positional_encoding(seq_len, self.hidden_size).to(device)
        x = x + pos_enc  # Add positional encoding to the input

        if reverse:
            atten_mask = create_reverse_mask(seq_len, device).type_as(x)
        else:
            atten_mask = create_forward_mask(seq_len, device).type_as(x)
        
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask_bool = mask.squeeze(dim=-1).bool()
            src_key_padding_mask = torch.where(src_key_padding_mask_bool, torch.tensor(0.0, dtype=x.dtype), torch.tensor(-float('inf'), dtype=x.dtype))
            atten_mask = correct_attention_mask(atten_mask, src_key_padding_mask)
            
        y = self.decoder(x, memory, tgt_mask=atten_mask, tgt_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        if was_unsqueezed:
            y = y.squeeze(dim=1)

        return y