'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
import torch

from ..utils.embedding_layer import JointEmbeddingLayer, ContinuousFeatureEmbeddingLayer

class BaseCritic(nn.Module):
    def __init__(self, net, env_config, network_params, input_size):
        super(BaseCritic, self).__init__()  
        self.hidden_size, self.num_layer = network_params.hidden_size, network_params.num_layer
        self.value_size = network_params.value_size
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(input_size, self.hidden_size)
        self.final_layer = create_layer(self.hidden_size, self.value_size, act_fn = 'none') 
        self.use_discrete = env_config.use_discrete
        self.net = net(self.num_layer, self.hidden_size, dropout = network_params.dropout)
        self.layer_norm = nn.LayerNorm(self.value_size, elementwise_affine = False)

    def _forward(self, _value, mask = None):
        value = self.net(_value, mask = mask) 
        _value = self.final_layer(value)

        mean = _value.mean(dim = -1, keepdim = True)
        original_shape = _value.shape
        _value = _value.view(-1, self.value_size)  # Flatten the tensor to 2D if not already
        dist = self.layer_norm(_value)  # Apply layer normalization
        dist = dist.view(original_shape)  # Restore the original shape
        return mean, dist

class SingleInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(SingleInputCritic, self).__init__(net, env_config, network_params, env_config.state_size)
        self.apply(init_weights)

    def forward(self, state, mask = None):
        _state = self.embedding_layer(state)
        mean, dist = self._forward(_state, mask)
        return mean, dist

    def evaluate(self, state, mask = None):
        mean, _ = self.forward(state, mask)
        return mean

class DualInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(DualInputCritic, self).__init__(net, env_config, network_params, env_config.state_size + env_config.action_size)
        self.apply(init_weights)

    def forward(self, state, action, mask = None):
        if not self.use_discrete:
            action = torch.tanh(action)
        _state = self.embedding_layer(torch.cat([state, action], dim = -1))
        mean, dist = self._forward(_state, mask)
        return mean, dist

    def evaluate(self, state, mask = None):
        mean, _ = self.forward(state, mask)
        return mean
    
    # def forward(self, state, action, mask = None):
    #     if not self.use_discrete:
    #         action = torch.tanh(action)
    #     _state = self.embedding_layer(torch.cat([state, action], dim = -1))
    #     return self._forward(_state, mask)
