'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch.nn as nn
from transformers import GPT2Model, GPT2Config 

class GPT2(nn.Module):
    def __init__(self, num_layer, hidden_size, num_heads: int = 8, reverse = False):
        super(GPT2, self).__init__()   
        self.num_heads = num_heads
        self.reverse = reverse
        # Define the configuration for a smaller GPT model
        config = GPT2Config(
            vocab_size=1,  # Not using token embeddings, so set to 1
            n_embd=hidden_size,  # Smaller hidden size
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            summary_first_dropout = 0.0,
            n_layer=num_layer,
            n_head=num_heads,    # Fewer attention heads
            n_positions=64,
            summary_use_proj=False,
            summary_activation=None,
            summary_proj_to_labels=False,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
        )
        # Initialize the model
        self.net = GPT2Model(config)
        
    def forward(self, input_tensor, mask=None):
        # Check if the input tensor is 2D, and if so, unsqueeze it to make it 3D
        if len(input_tensor.shape) == 2:
            input_tensor.unsqueeze_(dim=1)
            was_unsqueezed = True
        else:
            was_unsqueezed = False
                 
        if self.reverse:
            input_tensor = input_tensor.flip(dims=(1,))

        attention_mask = None
        if mask is not None:
            if self.reverse:
                attention_mask = mask.flip(dims=(1,))
            else:
                attention_mask = mask
                
        output = self.net(inputs_embeds = input_tensor, attention_mask = attention_mask)
        output_tensor = output.last_hidden_state
        # If the input was unsqueezed, squeeze it back to its original shape

        if self.reverse:
            output_tensor = output_tensor.flip(dims=(1,))

        if was_unsqueezed:
            output_tensor.squeeze_(dim=1)
            
        return output_tensor