'''
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)
    
'''
import torch
import torch.nn as nn
from transformers import BartConfig, BartModel

class Bart(nn.Module):
    def __init__(self, num_layer, hidden_size, reverse=True, num_heads: int=8):
        super(Bart, self).__init__()
        config = BartConfig(
            vocab_size=hidden_size,
            decoder_layers=num_layer,
            decoder_ffn_dim=hidden_size * 4,
            decoder_attention_heads=num_heads,
            # decoder_layerdrop=0.0,
            # dropout=0.0,
            d_model=hidden_size,
            use_cache=False,
            is_encoder_decoder=True,
        )
        self.decoder = BartModel(config).decoder
        self.reverse = reverse

    def forward(self, input_tensor, encoder_hidden_states, mask=None):
        # If reversing the sequence, handle this for input, hidden states, and mask
        if self.reverse:
            input_tensor = input_tensor.flip(dims=[1])
            encoder_hidden_states = encoder_hidden_states.flip(dims=[1])

        if mask is not None:
            mask = mask.squeeze(-1)
            mask = mask.flip(dims=[1]).long() if self.reverse else mask.long()
                
        outputs = self.decoder(
            inputs_embeds=input_tensor,
            attention_mask=mask,
            encoder_attention_mask=mask,
            encoder_hidden_states = encoder_hidden_states,
        )
        output_tensor = outputs.last_hidden_state

        # If the sequence was reversed, flip it back
        if self.reverse:
            output_tensor = output_tensor.flip(dims=[1])

        return output_tensor