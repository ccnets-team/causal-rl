'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.embedding_layer import ContinuousFeatureEmbeddingLayer

class BaseCritic(nn.Module):
    def __init__(self, net, env_config, critic_params, input_size):
        super(BaseCritic, self).__init__()  
        self.d_model, self.num_layers = critic_params.d_model, critic_params.num_layers
        self.value_size = 1
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(input_size, self.d_model)
        self.final_layer = create_layer(self.d_model, self.value_size, act_fn = 'none') 
        self.use_discrete = env_config.use_discrete
        self.net = net(self.num_layers, self.d_model, dropout = critic_params.dropout)

    def _forward(self, _value, mask = None):
        value = self.net(_value, mask = mask) 
        value = self.final_layer(value)
        return value

class SingleInputCritic(BaseCritic):
    def __init__(self, net, env_config, critic_params):
        super(SingleInputCritic, self).__init__(net, env_config, critic_params, env_config.state_size)
        self.apply(init_weights)

    def forward(self, state, mask = None):
        _state = self.embedding_layer(state)
        return self._forward(_state, mask)

class DualInputCritic(BaseCritic):
    def __init__(self, net, env_config, critic_params):
        super(DualInputCritic, self).__init__(net, env_config, critic_params, env_config.state_size + env_config.action_size)
        self.apply(init_weights)

    def forward(self, state, action, mask = None):
        if not self.use_discrete:
            action = torch.tanh(action)
        _state = self.embedding_layer(torch.cat([state, action], dim = -1))
        value = self._forward(_state, mask)
        return value
    
