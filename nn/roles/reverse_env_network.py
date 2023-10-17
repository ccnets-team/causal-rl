'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from nn.transformer import TransformerEncoder

class RevEnv(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(RevEnv, self).__init__()
        self.value_size = 1
        use_discrete = env_config.use_discrete
        self.use_discrete = use_discrete 
        self.hidden_size = network_params.hidden_size
        input_size = env_config.state_size + env_config.action_size + self.value_size
        output_size = env_config.state_size
        num_layer = network_params.num_layer
        
        # Comment about joint representation for the actor and reverse-env network:
        # Concatenation (cat) is a more proper joint representation for actor and reverse-env joint type.
        # However, when the reward scale is too high, addition (add) seems more robust.
        # The decision of which method to use should be based on the specifics of the task and the nature of the data.
        self.net = net(num_layer, input_size, output_size, self.hidden_size)
            
        self.apply(init_weights)

    def forward(self, next_state, action, value, mask=None):
        if not self.use_discrete:
            action = torch.tanh(action)
        z = torch.cat([next_state, action, value], dim=-1)
        if len(z.shape) >= 3:
            if mask is None:
                state = self.net(z, reverse=True)
            else:
                state = self.net(z, reverse=True, src_key_padding_mask=mask)
        else:
            state = self.net(z)
        return state
