'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config 


class _GPTBase(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads):
        super(_GPTBase, self).__init__()   
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=num_layer,
            n_head=num_heads,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            summary_first_dropout=0.0,            
            summary_use_proj=False,
            summary_proj_to_labels=False,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
        )
        self.net = GPT2Model(config)

class GPT(_GPTBase):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8):
        super(GPT, self).__init__(num_layer, hidden_size, num_heads)   
        
    def forward(self, input_tensor, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = mask.long()
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)

        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
        return output_tensor
    