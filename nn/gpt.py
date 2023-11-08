'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config 


class _GPTBase(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads, dropout):
        super(_GPTBase, self).__init__()   
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=num_layer,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.net = GPT2Model(config)

class GPT(_GPTBase):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8, dropout = 0.0):
        super(GPT, self).__init__(num_layer, hidden_size, num_heads, dropout)   
        
    def forward(self, input_tensor, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = mask.long()

        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
        return output_tensor
    