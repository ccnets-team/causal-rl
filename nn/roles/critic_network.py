'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch.nn as nn
from ..utils.network_init import init_weights, create_layer
from ..utils.dual_joint import DualJointLayer
import torch

class BaseCritic(nn.Module):
    def __init__(self, net, env_config, network_params):
        super(BaseCritic, self).__init__()  
        hidden_size, num_layer = network_params.hidden_size, network_params.num_layer
        self.net = net(num_layer, hidden_size)
        self.value_size = 1
        self.final = create_layer(hidden_size, self.value_size)

        self.action_size = env_config.action_size
        self.state_size = env_config.state_size
        self.use_discrete = env_config.use_discrete
        self.hidden_size = hidden_size
        
    def _forward(self, state):
        x = self.net(state)
        return self.final(x)

class SingleInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params):
        super(SingleInputCritic, self).__init__(net, env_config, network_params)
        self.embedding_layer = create_layer(self.state_size, self.hidden_size, act_fn = "tanh") 
        self.apply(init_weights)

    def forward(self, state):
        emb_state = self.embedding_layer(state)
        return self._forward(emb_state)

class DualInputCritic(BaseCritic):
    def __init__(self, net, env_config, network_params, joint_type = 'cat'):
        super(DualInputCritic, self).__init__(net, env_config, network_params)
        self.embedding_layer = DualJointLayer.create(self.state_size, self.action_size, self.hidden_size, joint_type =joint_type)
        self.apply(init_weights)

    def forward(self, state, action):
        if not self.use_discrete:
            action = torch.tanh(action)
        emb_state = self.embedding_layer(state, action)
        return self._forward(emb_state)
