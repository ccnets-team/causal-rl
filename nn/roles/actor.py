'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn

from torch.distributions import Normal
from ..network_utils import init_weights, create_layer
from ..network_utils import ContinuousFeatureEmbeddingLayer

class _BaseActor(nn.Module):
    def __init__(self, net, env_config, use_deterministic, network_params, input_size):
        super(_BaseActor, self).__init__()
        
        # Environment and network configuration
        self.use_deterministic = use_deterministic
        self.use_discrete = env_config.use_discrete
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.num_layers, self.d_model = network_params.num_layers, network_params.d_model

        # Actor network layers
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(input_size, self.d_model)
        self.mean_layer = create_layer(self.d_model, self.action_size, act_fn='none')
        self.log_std_layer = create_layer(self.d_model, self.action_size, act_fn='none')
        self.net = net(self.num_layers, self.d_model, dropout = network_params.dropout) 
        self.value_size = 1
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def _compute_forward_pass(self, z, mask = None):
        y = self.net(z, mask = mask) 
        y = self.relu(y)
        mu, sigma = self.mean_layer(y), self.softplus(self.log_std_layer(y))
        return mu, sigma 
        
    def _sample_action(self, mean, std):
        if self.use_discrete:
            y = mean
            # Get the softmax probabilities from the mean (logits)
            y = torch.softmax(y, dim=-1)

            # # Sample an action from the softmax probabilities
            action_indices = torch.distributions.Categorical(probs=y).sample()
            action = torch.zeros_like(y).scatter_(-1, action_indices.unsqueeze(-1), 1.0)
        else:
            action = torch.normal(mean, std).to(mean.device)
        return action

    def _select_action(self, mean):
        if self.use_discrete:
            prob = torch.softmax(mean, dim=-1)  # Assuming 'mean' is already defined.
            action_indices = torch.argmax(prob, dim=-1)
            # Initializing a tensor of zeros for one-hot encoding.
            action = torch.zeros_like(mean)
            # Filling in '1' at the index of the selected action.
            action.scatter_(-1, action_indices.unsqueeze(-1), 1)
        else:
            action = mean
        return action
   
    def _predict_action(self, mean):
        if self.use_discrete:
            action = torch.softmax(mean, dim=-1)
        else:
            action = mean
        return action
    
class DualInputActor(_BaseActor):
    def __init__(self, net, env_config, use_deterministic, network_params):
        value_size = 1
        super().__init__(net, env_config, use_deterministic, network_params, env_config.state_size + value_size)
        self.apply(init_weights)
        
        # Comment about joint representation for the actor and reverse-env network:
        # Concatenation (cat) is a more proper joint representation for actor and reverse-env joint type.
        # However, when the reward scale is too high, addition (add) seems more robust.
        # The decision of which method to use should be based on the specifics of the task and the nature of the data.

    def forward(self, state, value, mask = None):
        if self.use_deterministic:
            return self.select_action(state, value, mask)
        z = self.embedding_layer(torch.cat([state, value], dim =-1))
        mean, std = self._compute_forward_pass(z, mask)
        return Normal(mean, std).rsample()

    def compute_mean_std(self, state, value, mask = None):
        z = self.embedding_layer(torch.cat([state, value], dim =-1))
        mean, std = self._compute_forward_pass(z, mask)
        return mean, std

    def predict_action(self, state, value, mask = None):
        mean, _ = self.compute_mean_std(state, value, mask)
        action = self._predict_action(mean)
        return action

    def sample_action(self, state, value, mask = None):
        mean, std = self.compute_mean_std(state, value, mask)
        action = self._sample_action(mean, std)
        return action
    
    def select_action(self, state, value, mask = None):
        mean, _ = self.compute_mean_std(state, value, mask)
        action = self._select_action(mean)
        return action