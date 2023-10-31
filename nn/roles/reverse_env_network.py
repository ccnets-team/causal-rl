'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.joint_embedding_layer import JointEmbeddingLayer
from nn.transformer import TransformerEncoder, TransformerDecoder   

class RevEnv(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(RevEnv, self).__init__()
        self.value_size = 1
        use_discrete = env_config.use_discrete
        self.use_discrete = use_discrete 
        self.hidden_size = network_params.hidden_size
        num_layer = network_params.num_layer

        self.use_transformer_encoder: bool = False 

        if net is TransformerEncoder:
            self.use_transformer_encoder = True
        # Comment about joint representation for the actor and reverse-env network:
        # Concatenation (cat) is a more proper joint representation for actor and reverse-env joint type.
        # However, when the reward scale is too high, addition (add) seems more robust.
        # The decision of which method to use should be based on the specifics of the task and the nature of the data.
        self.embedding_layer = JointEmbeddingLayer(env_config.state_size, env_config.action_size, \
            self.value_size, output_size = self.hidden_size, joint_type = "cat")
        self.net = net(num_layer, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, env_config.state_size, act_fn = 'none') 
            
        self.apply(init_weights)

    def forward(self, next_state, action, value, mask=None):
        if not self.use_discrete:
            action = torch.tanh(action)
        if self.use_transformer_encoder:
            z = self.embedding_layer(next_state, action, value)
            state = self.net(z, reverse=True, mask=mask)
        else:
            state = self.net(z)
        state = self.final_layer(state)            
        return state
