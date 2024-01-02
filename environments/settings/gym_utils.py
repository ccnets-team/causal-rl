
from utils.structure.env_config import EnvConfig
import gymnasium as gym
from typing import Type

def is_key_valid(key, survive_keywords=['survive']):
    """
    Determines if a given key is a valid reward key and if it is related to survival.

    This function is particularly relevant for the 'ant-v4' environment in Gymnasium when using vectorized environments (Vec).
    In contrast to environments like 'hopper', 'halfcheetah', and 'walker2d', 'ant-v4' uses a unique system where it provides 
    a 'reward_survive' at both terminal and non-terminal signals. This is especially important when using Vec environments, as 
    the rewards are processed differently compared to single-instance environments. The function assists in adjusting or cancelling 
    the effect of survival rewards at the terminal signal in these vectorized scenarios.

    Usage example:
        self.env = gym.make(env_name, render_mode='human') if use_graphics else gym.make_vec(env_name, num_envs=self.num_agents)

    Parameters:
    key (str): The key to check.
    survive_keywords (list): List of keywords related to survival.

    Returns:
    tuple: (bool, bool) where the first bool indicates if the key is a valid reward key,
           and the second bool indicates if it is related to survival.
    """
    key_lower = key.lower()
    if key_lower.startswith('reward'):
        return True, any(survive_word in key_lower for survive_word in survive_keywords)
    return False, False

def get_final_observations_from_info(info, observations):
    done_observations = observations.copy()
    if 'final_observation' in info:
        for idx, agent_final_obs in enumerate(info['final_observation']):
            if agent_final_obs is None:
                continue
            done_observations[idx] = agent_final_obs
           
    return done_observations

def get_final_rewards_from_info(info, rewards, terminated):
    done_rewards = rewards.copy()
    if 'final_info' in info:
        for idx, agent_final_info in enumerate(info['final_info']):
            if agent_final_info is not None:
                for key in agent_final_info:
                    valid_reward, is_survive_related = is_key_valid(key)
                    if valid_reward:
                        if is_survive_related and terminated[idx]:
                            # Subtract 'reward_survive' or similar if terminated is True for this index
                            done_rewards[idx] -= agent_final_info[key]

    return done_rewards

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

