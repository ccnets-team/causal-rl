'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch.nn as nn
from ..utils.network_init import init_weights
import torch

class BaseCritic(nn.Module):
    def __init__(self, net, env_config, network_params, input_size):
        super(BaseCritic, self).__init__()  
        hidden_size, num_layer = network_params.hidden_size, network_params.num_layer
        value_size = 1
        self.net = net(num_layer, input_size = input_size, output_size = value_size, hidden_size = hidden_size)

        self.use_discrete = env_config.use_discrete
        self.hidden_size = hidden_size
        
    def _forward(self, state, mask = None):
        if mask is not None:
            return self.net(state, mask)
        return self.net(state)

class SingleInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(SingleInputCritic, self).__init__(net, env_config, network_params, env_config.state_size)
        self.apply(init_weights)

    def forward(self, state, mask=None):
        return self._forward(state, mask)

class DualInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        input_size = env_config.state_size + env_config.action_size
        super(DualInputCritic, self).__init__(net, env_config, network_params, input_size)
        self.apply(init_weights)

    def forward(self, state, action, mask=None):
        if not self.use_discrete:
            action = torch.tanh(action)
        x = torch.cat([state, action], dim = -1) 
        return self._forward(x, mask)
