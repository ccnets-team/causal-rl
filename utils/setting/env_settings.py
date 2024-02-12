from utils.printer import print_env_specs
import numpy as np
from utils.structure.env_config import EnvConfig
from utils.setting.rl_params import RLParameters
from typing import Tuple, Optional, Union, Type, Dict, List
from environments.settings.gym_utils import setup_gym_environment
from environments.settings.mlagents_utils import setup_mlagents_environment

MLAGENTS_ENV_SPECIFIC_ARGS = {
    "3DBallHard": {},
    "Worm": {},
    "Crawler": {},
    "Walker": {}, 
    "Hallway": {},
    "PushBlock": {},
    "Pyramids": {}
}

GYM_ENV_SPECIFIC_ARGS = {
    "InvertedDoublePendulum-": {},   
    "Pusher-": {},   
    "Reacher-": {},
    "Swimmer-": {},
    "Hopper-": {},      
    "Walker2d-": {},   
    "Ant-": {},
    "HalfCheetah-": {},   
    "Humanoid-": {},    
    "HumanoidStandup-": {}
}
GYM_NUM_ENVIRONMENTS = 1

def analyze_env(env_name):
    rl_params = None
    use_gym = False
    if "-v" in env_name:
        use_gym = True
    rl_params = configure_parameters(env_name, use_gym=use_gym)
    return rl_params

def calculate_min_samples_per_step(training_params):
    # Calculate minimum samples per step
    samples_per_step = int(max(1, np.ceil(training_params.batch_size/(training_params.replay_ratio))))
    return samples_per_step        

def configure_parameters(env_name: str, use_gym: bool = False) -> Tuple[Optional[Type[EnvConfig]], Type[RLParameters]]:
    env_specific_args = GYM_ENV_SPECIFIC_ARGS if use_gym else MLAGENTS_ENV_SPECIFIC_ARGS

    env_config = setup_environment(env_name, use_gym)
    if env_config is None:
        return None, None

    env_name, num_agents, obs_shapes, continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high = env_config

    rl_params = RLParameters()

    min_samples_per_step = calculate_min_samples_per_step(rl_params.training)
    if use_gym:
        num_environments = GYM_NUM_ENVIRONMENTS
        num_agents = max(1, int(np.ceil(min_samples_per_step/num_environments)))
    else:
        num_environments = max(1, int(np.ceil(min_samples_per_step / num_agents)))
        
    env_config = create_environment_config(
        env_name, 'gym' if use_gym else 'mlagents', num_environments, num_agents,
        obs_shapes, continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high)
    
    apply_configuration_to_parameters(env_specific_args, env_name, rl_params)
    
    print_env_specs(env_config)
    rl_params.init_env_config(env_config)

    return rl_params

def setup_environment(env_name: str, use_gym: bool = False) -> Optional[Type[EnvConfig]]:
    try:
        if use_gym:
            return setup_gym_environment(env_name)
        return setup_mlagents_environment(env_name)
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
                        
def update_env_config(rl_params: RLParameters, env_config: EnvConfig) -> None:
    # List of sub-parameter objects within rl_params
    sub_params_objects = [
        rl_params.training, 
        rl_params.algorithm, 
        rl_params.network, 
        rl_params.optimization, 
        rl_params.normalization
    ]
    # Iterate over each sub-parameter object
    for sub_param in sub_params_objects:
        # Iterate over the attributes of the sub-parameter object
        for attr_name in dir(sub_param):
            # Skip any special methods or attributes
            if attr_name.startswith('__'):
                continue
            
            # Ensure the attribute exists in both rl_params (accessible via sub_param) and env_config, and is not a method
            if hasattr(env_config, attr_name) and hasattr(sub_param, attr_name):
                rl_param_value = getattr(sub_param, attr_name)
                # Ensure the value is not a method before setting it
                if not callable(rl_param_value):
                    setattr(env_config, attr_name, rl_param_value)
                    print(f"Updated env_config.{attr_name} with {rl_param_value}")

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