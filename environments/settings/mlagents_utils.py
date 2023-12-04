import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment

def setup_mlagents_environment(env_name):
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(width=1280, height=720)
    file_name = "../unity_environments/" + env_name + "/"
    base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT
    env = UnityEnvironment(file_name=file_name, base_port=base_port, no_graphics=True, side_channels=[channel])
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    num_agents = len(decision_steps)
    continuous_action_size = spec.action_spec.continuous_size
    discrete_action_size = np.prod(spec.action_spec.discrete_branches) - 1 if spec.action_spec.discrete_branches else 0
    obs_shapes = [obs.shape for obs in spec.observation_specs]
    env.close()
    state_low, state_high, action_low, action_high = None, None, None, None

    return env_name, num_agents, obs_shapes, continuous_action_size, discrete_action_size, state_low, state_high, action_low, action_high

