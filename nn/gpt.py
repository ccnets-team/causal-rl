import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class GPTParams:
    def __init__(self, d_model, num_layers, dropout):
        """
        Initialize a GPT network.	
        Args:
        - d_model (int): Dimension of the model.
        - num_layers (int): Number of layers in the network.
        - dropout (float): Dropout rate.
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        # Initialize other GPT specific configurations here
        
class _GPTBase(nn.Module):
    def __init__(self, num_layer, d_model, num_heads, dropout):
        super(_GPTBase, self).__init__()   
        config = GPT2Config(
            vocab_size=d_model,  # This should be set to your actual vocab size
            n_embd=d_model,
            n_layer=num_layer,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.net = GPT2Model(config)

class GPT(_GPTBase):
    def __init__(self, num_layer, d_model, num_heads: int = 8, dropout = 0.0):
        super(GPT, self).__init__(num_layer, d_model, num_heads, dropout)

    def forward(self, input_tensor, mask=None):
        attention_mask = mask.long() if mask is not None else None
        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
        return output_tensor
