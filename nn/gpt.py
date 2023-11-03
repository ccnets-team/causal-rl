'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config 

class GPT2(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8, reverse=False):
        super(GPT2, self).__init__()   
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            summary_first_dropout=0.0,
            n_layer=num_layer,
            n_head=num_heads,
            summary_use_proj=False,
            summary_proj_to_labels=False,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
        )
        self.net = GPT2Model(config)
        self.reverse = reverse

    def forward(self, input_tensor, mask=None):
        if self.reverse:
            input_tensor = torch.flip(input_tensor, dims=(1,))

        attention_mask = None
        if mask is not None:
            attention_mask = torch.flip(mask, dims=(1,)).long() if self.reverse else mask.long()
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(-1)

        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
            
        output_tensor = torch.flip(output_tensor, dims=(1,)) if self.reverse else output_tensor
            
        return output_tensor