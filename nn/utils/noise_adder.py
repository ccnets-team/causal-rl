'''
    CCNet's configurators implementations.
'''
import torch
import math

class NoiseStrategy:
    def __init__(self, use_discrete):
        self.use_discrete = use_discrete

    def reset(self, reset_agents = None):
        raise NotImplementedError

    def apply(self, action, exploration_rate):
        raise NotImplementedError

class EpsilonGreedy(NoiseStrategy):
    def __init__(self, use_discrete):
        super(EpsilonGreedy, self).__init__(use_discrete)

    def reset(self, reset_agents = None):
        raise 

    def apply(self, y, exploration_rate):
        if exploration_rate is None or exploration_rate == 0:
            return y
        if self.use_discrete:
            batch_size = y.shape[0]
            num_actions = y.shape[1]

            # Sample actions based on softmax probabilities
            softmax_actions = torch.distributions.Categorical(probs=y).sample()

            # Generate random actions for the entire batch
            random_actions = torch.randint(0, num_actions, (batch_size,)).to(y.device)

            # Choose whether to use random actions based on epsilon
            choose_random = (torch.rand(batch_size) < exploration_rate).to(y.device)

            # Depending on the choice, select either the random action or softmax action
            chosen_actions = torch.where(choose_random, random_actions, softmax_actions)

            # Create a new tensor with all zeros except for the chosen action's index
            action = torch.zeros_like(y).scatter_(-1, chosen_actions.unsqueeze(-1), 1.0)
        else:            
            # Add Gaussian noise for exploration
            noise = torch.normal(mean=0, std=exploration_rate, size=y.shape).to(y.device)
            action = y + noise
            
        return action

class OrnsteinUhlenbeck(NoiseStrategy):
    def __init__(self, use_discrete, theta=0.15, sigma=0.2, dt=1e-2):
        super(OrnsteinUhlenbeck, self).__init__(use_discrete)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.ou_prev_state = None

    def reset(self, reset_agents = None):
        if reset_agents is not None and self.ou_prev_state is not None:
            self.ou_prev_state[reset_agents] = torch.zeros_like(self.ou_prev_state[reset_agents]).to(self.ou_prev_state.device)

    def apply(self, action, exploration_rate = None):
        if self.use_discrete:
            return action        
        """
        Apply Ornstein-Uhlenbeck noise to the given action. This type of noise is 
        correlated with the previous state of the noise, making it useful for 
        exploring environments where momentum or continuity of actions is important.
        """
 
        noise_shape  = torch.Size([action.size(0), 1, action.size(-1)])
        
        if self.ou_prev_state is None or self.ou_prev_state.shape != noise_shape:
            self.ou_prev_state = torch.zeros(noise_shape).to(action.device)

        noise = self.theta * (-self.ou_prev_state * self.dt) + self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.normal(mean=0, std=1, size=noise_shape ).to(action.device)
        
        self.ou_prev_state += noise  # Update the state with the current noise
        
        if len(action.shape) == 2:  # If action is 2D
            return action + noise.squeeze(1)
        elif len(action.shape) == 3:  # If action is 3D
            return action + noise.expand_as(action)
        else:
            raise ValueError("Unsupported action tensor shape.")
        
class BoltzmannExploration(NoiseStrategy):
    def __init__(self, use_discrete, tau=1.0):
        super(BoltzmannExploration, self).__init__(use_discrete)  # Assuming the superclass has these parameters.
        self.tau = tau
        self.min_temperature = 0.1
        # Adjusted decay rate as per the derived formula
        self.decay_rate = -math.log(0.1)

    def reset(self, reset_agents = None):
        raise 
            
    def apply(self, x, exploration_rate):
        # Computing temperature based on the adjusted decay rate
        temperature = max(self.tau * math.exp(-self.decay_rate * (1 - exploration_rate)), self.min_temperature)
        # Assuming x has some specific use and computation. Placeholder for actual computation with x.
        boltzmann_probs = x / temperature
        return boltzmann_probs

