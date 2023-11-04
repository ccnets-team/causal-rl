'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.joint_embedding_layer import JointEmbeddingLayer

class RevEnv(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(RevEnv, self).__init__()
        self.value_size = 1
        self.use_discrete = env_config.use_discrete
        self.state_size = env_config.state_size
        self.action_size = env_config.action_size
        self.hidden_size = network_params.hidden_size
        self.num_layer = network_params.num_layer
            
        self.state_embedding_layer = create_layer(self.state_size, self.hidden_size, act_fn="tanh")
        self.ctx_embedding_layer = JointEmbeddingLayer(self.action_size, \
            self.value_size, output_size = self.hidden_size, joint_type = "cat")

        self.net = net(self.num_layer, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, env_config.state_size, act_fn = 'none') 
            
        self.apply(init_weights)

    def forward(self, next_state, action, value, mask=None):
        if not self.use_discrete:
            action = torch.tanh(action)
        _next_state = self.state_embedding_layer(next_state)
        _ctx = self.ctx_embedding_layer(action, value)
        x = self.net(_next_state, _ctx, mask=mask)
        return self.final_layer(x)   
