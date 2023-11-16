'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn

from torch.distributions import Normal
from ..utils.network_init import init_weights, create_layer
from nn.utils.noise_adder import EpsilonGreedy, OrnsteinUhlenbeck, BoltzmannExploration

from ..utils.embedding_layer import ContinuousFeatureEmbeddingLayer

log_std_min = -20
log_std_max = 2

class _BaseActor(nn.Module):
    def __init__(self, net, env_config, network_params, exploration_params, input_size):
        super(_BaseActor, self).__init__()
        
        # Environment and network configuration
        self.use_discrete = env_config.use_discrete
        self.state_size, self.action_size = env_config.state_size, env_config.action_size
        self.num_layer, self.hidden_size = network_params.num_layer, network_params.hidden_size

        # Actor network layers
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(input_size, self.hidden_size)
        self.mean_layer = create_layer(self.hidden_size, self.action_size, act_fn='none')
        self.log_std_layer = create_layer(self.hidden_size, self.action_size, act_fn='none')
        self.net = net(self.num_layer, self.hidden_size, dropout = network_params.dropout) 
        self.value_size = 1
        
        # Exploration strategy initialization
        self.noise_strategy, self.use_noise_before_activation, = self._initialize_noise_strategy(
            exploration_params
        )
        self.noise_type = exploration_params.noise_type 
        
    def _initialize_noise_strategy(self, exploration_params):
        noise_strategy = None
        use_noise_before_activation = False

        if exploration_params.noise_type == "epsilon_greedy":
            noise_strategy = EpsilonGreedy(self.use_discrete)
            use_noise_before_activation = not self.use_discrete
        elif exploration_params.noise_type == "ou" and not self.use_discrete:
            noise_strategy = OrnsteinUhlenbeck(self.use_discrete)
            use_noise_before_activation = True
        elif exploration_params.noise_type == "boltzmann" and self.use_discrete:
            noise_strategy = BoltzmannExploration(self.use_discrete)
            use_noise_before_activation = True

        return noise_strategy, use_noise_before_activation
        
    def apply_noise(self, y, exploration_rate = None):
        if self.noise_strategy is None:
            return y
        return self.noise_strategy.apply(y, exploration_rate)

    def reset_noise(self, reset = None):
        if self.noise_strategy is None:
            return
        if self.noise_type == "ou" and reset is not None:
            return self.noise_strategy.reset(reset)

    def _compute_forward_pass(self, z, mask = None):
        y = self.net(z, mask = mask) 
        mean, log_std = self.mean_layer(y), self.log_std_layer(y)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        return mean, std

    def _evaluate_action(self, mean, std):
        """
        Sample an action and compute its log probability given the distribution parameters.

        :param mean: Mean of the distribution (either Gaussian for continuous actions or logits for discrete actions)
        :param std: Standard deviation for continuous actions. Ignored for discrete actions.
        :return: sampled action and its log probability.
        """

        # Small constant for numerical stability
        epsilon = 1e-6  

        if self.use_discrete:
            # For discrete actions, sample an action using argmax (deterministic policy) 
            # and then compute its log probability
            probs = torch.softmax(mean, dim=-1)  # mean must be defined beforehand
            action_indices = torch.argmax(probs, dim=-1)
            # Create one-hot encoded tensor without using eye
            action = torch.zeros_like(mean)
            action.scatter_(-1, action_indices.unsqueeze(-1), 1)

            log_prob = torch.log(probs.gather(-1, action_indices.unsqueeze(-1)))
        else:
            # For continuous actions, sample an action from the Gaussian distribution,
            # squash it using the tanh function, and then compute its log probability
            dist = Normal(mean, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            unsquashed_log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            # Correct the log_prob due to the squashing operation
            log_prob = unsquashed_log_prob - torch.log(1 - action.pow(2) + epsilon).sum(dim=-1, keepdim=True)

        return action, log_prob
        
    def _log_prob(self, mean, std, raw_action):
        """
        Compute the log probability of taking an action given the distribution parameters.

        :param mean: Mean of the distribution (either Gaussian for continuous actions or logits for discrete actions)
        :param std: Standard deviation for continuous actions. Ignored for discrete actions.
        :param raw_action: The sampled action.
        :return: log probability of the action.
        """

        # Small constant for numerical stability
        epsilon = 1e-6  
        log_prob = None
        if self.use_discrete:
            # For discrete actions, compute log probabilities using softmax and gather
            probs = torch.softmax(mean, dim=-1)
            action_indices = torch.argmax(raw_action, dim=-1)
            log_prob = torch.log(probs.gather(-1, action_indices.unsqueeze(-1)))

        else:
            # For continuous actions, compute log probabilities using normal distribution
            # and the change of variables formula for tanh squashing
            dist = Normal(mean, std)
            action_0 = torch.tanh(raw_action)
            unsquashed_log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            squash_correction = torch.log(1 - action_0.pow(2) + epsilon).sum(dim=-1, keepdim=True)
            log_prob = unsquashed_log_prob - squash_correction
        return log_prob

    def _sample_action(self, mean, std, exploration_rate=None):
        if exploration_rate is None or exploration_rate == 0:
            return self._select_action(mean)

        if self.use_discrete:
            y = mean
            # Get the softmax probabilities from the mean (logits)network_params
            if self.use_noise_before_activation:
                y = self.apply_noise(y, exploration_rate)
            y = torch.softmax(y, dim=-1)
            if not self.use_noise_before_activation:
                y = self.apply_noise(y, exploration_rate)

            # # Sample an action from the softmax probabilities
            action_indices = torch.distributions.Categorical(probs=y).sample()
            action = torch.zeros_like(y).scatter_(-1, action_indices.unsqueeze(-1), 1.0)
        else:
            action = torch.normal(mean, std).to(mean.device)
            if self.use_noise_before_activation:
                action = self.apply_noise(action, exploration_rate)
        return action

    def _sample_action(self, mean, std, exploration_rate=None):
        if exploration_rate is None or exploration_rate == 0:
            return self._select_action(mean)

        if self.use_discrete:
            y = mean
            # Get the softmax probabilities from the mean (logits)
            if self.use_noise_before_activation:
                y = self.apply_noise(y, exploration_rate)
            y = torch.softmax(y, dim=-1)
            if not self.use_noise_before_activation:
                y = self.apply_noise(y, exploration_rate)

            # # Sample an action from the softmax probabilities
            action_indices = torch.distributions.Categorical(probs=y).sample()
            action = torch.zeros_like(y).scatter_(-1, action_indices.unsqueeze(-1), 1.0)
        else:
            action = torch.normal(mean, std).to(mean.device)
            if self.use_noise_before_activation:
                action = self.apply_noise(action, exploration_rate)
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

class SingleInputActor(_BaseActor):
    def __init__(self, net, env_config, network_params, exploration_params):
        super().__init__(net, env_config, network_params, exploration_params, env_config.state_size)
        self.apply(init_weights)

    def forward(self, state, mask = None):
        z = self.embedding_layer(state)
        mean, std = self._compute_forward_pass(z, mask)
        return mean, std

    def predict_action(self, state, mask = None):
        mean, _ = self.forward(state, mask)
        action = self._predict_action(mean)
        return action

    def sample_action(self, state, mask = None, exploration_rate = None):
        mean, std = self.forward(state, mask)
        action = self._sample_action(mean, std, exploration_rate)
        return action

    def select_action(self, state, mask = None):
        mean, _ = self.forward(state, mask)
        action = self._select_action(mean)
        return action

    def evaluate_action(self, state, mask = None):
        mean, std = self.forward(state, mask)
        action, log_prob = self._evaluate_action(mean, std)
        return action, log_prob

    def log_prob(self, state, action, mask = None):
        mean, std = self.forward(state, mask)
        log_prob = self._log_prob(mean, std, action)
        return log_prob

class DualInputActor(_BaseActor):
    def __init__(self, net, env_config, network_params, exploration_params):
        value_size = 1
        super().__init__(net, env_config, network_params, exploration_params, env_config.state_size + value_size)
        self.apply(init_weights)
        
        # Comment about joint representation for the actor and reverse-env network:
        # Concatenation (cat) is a more proper joint representation for actor and reverse-env joint type.
        # However, when the reward scale is too high, addition (add) seems more robust.
        # The decision of which method to use should be based on the specifics of the task and the nature of the data.

    def forward(self, state, value, mask = None):
        z = self.embedding_layer(torch.cat([state, value], dim =-1))
        mean, std = self._compute_forward_pass(z, mask)
        return mean, std

    def predict_action(self, state, value, mask = None):
        mean, _ = self.forward(state, value, mask)
        action = self._predict_action(mean)
        return action

    def sample_action(self, state, value, mask = None, exploration_rate = None):
        mean, std = self.forward(state, value, mask)
        action = self._sample_action(mean, std, exploration_rate)
        return action
    
    def select_action(self, state, value, mask = None):
        mean, _ = self.forward(state, value, mask)
        action = self._select_action(mean)
        return action