
from utils.structure.env_config import EnvConfig
import gymnasium as gym
from typing import Type

def setup_gym_environment(env_name: str) -> Type[EnvConfig]:
    env = gym.make(env_name)
    # Similar to the original function's implementation to set the parameters for gym environment
    env.close()

    obs_shapes = env.observation_space.shape
    continuous_action_size = 0 if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    discrete_action_size = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else 0
    
    state_low, state_high = env.observation_space.low, env.observation_space.high
    action_low, action_high = None, None
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_size = env.action_space.n
    else:
        continuous_action_size = env.action_space.shape[0]
        action_low, action_high = env.action_space.low, env.action_space.high
    num_agents = None
    return env_name, num_agents, [obs_shapes], continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high 

