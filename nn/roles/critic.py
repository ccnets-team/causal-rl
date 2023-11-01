'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
import torch

from ..utils.joint_embedding_layer import JointEmbeddingLayer
from nn.transformer import TransformerEncoder, TransformerDecoder   

class BaseCritic(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(BaseCritic, self).__init__()  
        hidden_size, num_layer = network_params.hidden_size, network_params.num_layer
        self.net = net(num_layer, hidden_size)
        value_size = 1
        self.final_layer = create_layer(hidden_size, value_size, act_fn = 'none') 

        self.use_discrete = env_config.use_discrete
        self.hidden_size = hidden_size

    def _forward(self, z, mask = None):
        value = self.net(z, mask = mask) 
        return self.final_layer(value)

class SingleInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(SingleInputCritic, self).__init__(net, env_config, network_params)
        self.embedding_layer = create_layer(env_config.state_size, self.hidden_size, act_fn = 'tanh')
        self.apply(init_weights)

    def forward(self, state, mask = None):
        z = self.embedding_layer(state)
        return self._forward(z, mask)

class DualInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(DualInputCritic, self).__init__(net, env_config, network_params)
        self.embedding_layer = JointEmbeddingLayer(env_config.state_size, env_config.action_size, \
            output_size = self.hidden_size, joint_type="cat")
        self.apply(init_weights)

    def forward(self, state, action, mask = None):
        if not self.use_discrete:
            action = torch.tanh(action)
        x = self.embedding_layer(state, action)
        return self._forward(x, mask)
