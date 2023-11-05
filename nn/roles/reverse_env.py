'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.embedding_layer import JointEmbeddingLayer, ContinuousFeatureEmbeddingLayer

class RevEnv(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(RevEnv, self).__init__()
        self.value_size = 1
        self.use_discrete = env_config.use_discrete
        self.state_size = env_config.state_size
        self.action_size = env_config.action_size
        self.hidden_size = network_params.hidden_size
        self.num_layer = network_params.num_layer
            
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(self.state_size + self.action_size \
            + self.value_size, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, self.state_size, act_fn = 'none') 
        self.net = net(self.num_layer, self.hidden_size)
        self.apply(init_weights)

    def forward(self, next_state, action, value, mask=None):
        if not self.use_discrete:
            action = torch.tanh(action)
        
        padding_mask = None
        if mask is None:
            padding_mask = mask
        else:
            padding_mask = mask.flip(dims=[1])

        z = self.embedding_layer(torch.cat([next_state, action, value], dim=-1))
        filp_z = z.flip(dims=[1])
        x = self.net(filp_z, mask=padding_mask)
        filp_x = x.flip(dims=[1])
        return self.final_layer(filp_x)   
