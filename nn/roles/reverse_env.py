'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.embedding_layer import ContinuousFeatureEmbeddingLayer

class RevEnv(nn.Module):
    # HIDDEN_SIZE_MULTIPLIER is used to enhance the network's capacity for effectively capturing the 
    # influence of the current state, based on the next state, action, and value. These elements 
    # represent the causative factors in the generative process of the reverse environment. A larger 
    # hidden size allows the network to more powerfully model these complex relationships and dynamics, 
    # which is essential for accurately reversing the environmental dynamics.

    def __init__(self, net, env_config, network_params):
        super(RevEnv, self).__init__()
        self.use_discrete = env_config.use_discrete
        self.state_size = env_config.state_size
        self.action_size = env_config.action_size
        self.hidden_size = int(network_params.rev_env_hidden_size_mul*network_params.hidden_size)
        self.num_layer = network_params.num_layer
        self.value_size = 1
            
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(self.state_size + self.action_size \
            + self.value_size, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, self.state_size, act_fn = 'none') 
        self.net = net(self.num_layer, self.hidden_size, dropout = network_params.dropout)
        self.apply(init_weights)

    def forward(self, next_state, action, value, mask=None):
        # The action is used in its raw form here as some training algorithms require the original
        # action values for computing log probabilities. The environment handles the application of
        # tanh for exploration separately, ensuring that there is no discrepancy in the usage of tanh
        # between the exploration phase and the training algorithm's requirements.
        action = torch.tanh(action) if not self.use_discrete else action
              
        # Reverse the padding mask if not None
        padding_mask = mask.flip(dims=[1]) if mask is not None else None

        # Embed and process the state, action, and value
        embedded_input = self._embed_and_process(next_state, action, value)
        processed_output = self.net(embedded_input, mask=padding_mask)

        # Reverse the processed output and apply the final layer
        reversed_output = processed_output.flip(dims=[1])
        return self.final_layer(reversed_output)

    def _embed_and_process(self, next_state, action, value):
        """
        Embeds the input features and reverses the embedded sequence.
        """
        z = self.embedding_layer(torch.cat([next_state, action, value], dim=-1))
        return z.flip(dims=[1])
