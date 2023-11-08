from utils.printer import print_env_specs
import numpy as np
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters

from typing import Tuple, Optional, Union, Type, Dict, List
from environments.settings.gym_config import setup_gym_environment
from environments.settings.mlagents_config import setup_mlagents_environment

GYM_NUM_ENVIRONMENTS = 1
MLAGENTS_ENV_SPECIFIC_ARGS = {
    "3DBall": {'max_steps': 50000},
    "3DBallHard": {'max_steps': 50000},
    "Worm": {'max_steps': 500000},
    "Crawler": {'hidden_size': 192, 'max_steps': 500000},
    "Walker": {'hidden_size': 256, 'max_steps': 500000},
    "Hallway": {'state_normalizer': 'none', 'max_steps': 500000},
    "PushBlock": {'state_normalizer': 'none', 'batch_size': 1024, 'max_steps': 200000},
    "Pyramids": {'state_normalizer': 'none', 'batch_size': 1024, 'curiosity_factor': 0.1}
}

GYM_ENV_SPECIFIC_ARGS = {
    "Ant-": {'reward_scale': 0.01, 'max_steps': 500000},
    "Reacher-": {'reward_scale': 0.01, 'max_steps': 500000},
    "Pusher-": {'reward_scale': 0.01, 'max_steps': 500000},
    "Humanoid-": {'reward_scale': 0.1, 'hidden_size': 256, 'max_steps': 500000},
    "HumanoidStandup-": {'reward_scale': 0.001, 'hidden_size': 256, 'max_steps': 500000}
}

def analyze_env(env_name):
    env_config, rl_params = None, None
    use_unity = True
    if "-v" in env_name:
        use_unity = False
    env_config, rl_params = initialize_and_configure_parameters(env_name, is_unity=use_unity)
    print_env_specs(env_config)
    return env_config, rl_params

def initialize_and_configure_parameters(env_name: str, is_unity: bool = False) -> Tuple[Optional[Type[EnvConfig]], Type[RLParameters]]:
    env_specific_args = MLAGENTS_ENV_SPECIFIC_ARGS if is_unity else GYM_ENV_SPECIFIC_ARGS

    env_config = setup_environment(env_name, is_unity)
    if env_config is None:
        return None

    env_name, num_agents, obs_shapes, continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high = env_config

    rl_params = RLParameters()
    apply_configuration_to_parameters(env_specific_args, env_name, rl_params)

    min_samples_per_step = rl_params.training.minimum_samples_per_step()
    if is_unity:
        num_environments = max(1, int(np.ceil(min_samples_per_step / num_agents)))
    else:
        num_environments = GYM_NUM_ENVIRONMENTS
        num_agents = max(1, int(np.ceil(min_samples_per_step/num_environments)))
        
    env_config = create_environment_config(
        env_name, 'mlagents' if is_unity else 'gym', num_environments, num_agents,
        obs_shapes, continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high)

    noise_type = "boltzmann" if env_config.use_discrete else "ou"
    state_normalizer = determine_state_normalizer(env_config=env_config)

    rl_params.exploration.noise_type = noise_type
    rl_params.normalization.state_normalizer= state_normalizer
    apply_configuration_to_parameters(env_specific_args, env_name, rl_params)

    return env_config, rl_params

def setup_environment(env_name: str, use_mlagents: bool = False) -> Optional[Type[EnvConfig]]:
    try:
        if use_mlagents:
            return setup_mlagents_environment(env_name)
        return setup_gym_environment(env_name)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return None

def apply_configuration_to_parameters(env_configs: Dict, env_name: str, rl_params: RLParameters) -> None:
    for env_prefix, settings in env_configs.items():
        if env_name.startswith(env_prefix):
            for param_object in rl_params:
                for key, value in settings.items():
                    if hasattr(param_object, key):
                        setattr(param_object, key, value)

def create_environment_config(
    env_name: str,
    env_type: str,
    num_environments: int,
    num_agents: int,
    obs_shapes: List[Union[int, Tuple[int, int, int]]],  # Adjust as per actual types
    continuous_action_size: int,
    discrete_action_size: int,
    state_low: Optional[Union[float, np.ndarray]],
    state_high: Optional[Union[float, np.ndarray]],
    action_low: Optional[Union[float, np.ndarray]],
    action_high: Optional[Union[float, np.ndarray]]
) -> EnvConfig:  # The return type hint
    env_config = EnvConfig(
        env_name, env_type, num_environments, num_agents, obs_shapes,
        continuous_action_size, discrete_action_size, state_low, state_high,
        action_low, action_high
    )
    # Set any other default values for EnvironmentConfig here if needed
    return env_config

def determine_state_normalizer(env_config):
    if env_config.state_low is None or env_config.state_high is None:
        return "running_z_standardizer"
    elif (env_config.state_low >= 0).all() and (env_config.state_high <= 1).all():
        return "none"
    else:
        return "running_z_standardizer"
