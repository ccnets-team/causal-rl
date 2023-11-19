# import gym
import numpy as np
def is_key_valid(key):
    key_lower = key.lower()
    return key_lower.startswith('reward')

def get_final_rewards_from_info(info, num_agentss):
    done_rewards = np.zeros(num_agentss, dtype=np.float32)

    # Process rewards from final_info for terminated agents
    if 'final_info' in info and info['final_info'] is not None:
        final_infos = info['final_info']
        for idx, agent_final_info in enumerate(final_infos):
            if agent_final_info is None:
                continue
            for key in agent_final_info.keys():
                if is_key_valid(key):
                    done_rewards[idx] += agent_final_info[key]

    return done_rewards

def get_final_observations_from_info(info, done_observations):
    # Process observations from final_info for terminated agents
    if 'final_observation' in info and info['final_observation'] is not None:
        final_observation = info['final_observation']
        for idx, agent_final_obs in enumerate(final_observation):
            if agent_final_obs is None:
                continue
            done_observations[idx] = agent_final_obs

    return done_observations
