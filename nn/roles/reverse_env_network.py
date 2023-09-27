'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.triple_joint import TripleJointLayer

class RevEnv(nn.Module):
    def __init__(self, net, env_config, network_params, joint_type = 'add'):
        super(RevEnv, self).__init__()
        self.value_size = 1
        use_discrete = env_config.use_discrete
        self.use_discrete = use_discrete 
        self.hidden_size = network_params.hidden_size
        
        self.embedding_layer = TripleJointLayer.create(env_config.state_size, env_config.action_size, self.value_size, self.hidden_size, joint_type = joint_type)
        self.net = net(network_params.num_layer, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, env_config.state_size, act_fn = 'none') 
        
        self.apply(init_weights)
        
    def forward(self, next_state, action, value):
        if not self.use_discrete:
            action = torch.tanh(action)
        z = self.embedding_layer(next_state, action, value)
        x = self.net(z)
        return self.final_layer(x)


