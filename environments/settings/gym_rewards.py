# import gym
import numpy as np
def is_key_valid(key, substrings):
    key_lower = key.lower()
    return key_lower.startswith('reward') and any(sub in key_lower for sub in substrings)

def get_final_rewards_from_info(ongoing_terminated, info, num_agentss):
    done_sum_immediate_rewards = np.zeros(num_agentss, dtype=np.float32)
    done_sum_values = np.zeros(num_agentss, dtype=np.float32)

    # Process rewards from final_info for terminated agents
    if 'final_info' in info and info['final_info'] is not None:
        final_infos = info['final_info']
        for idx, agent_final_info in enumerate(final_infos):
            if agent_final_info is None:
                continue
            for key in agent_final_info.keys():
                if is_key_valid(key, ['linup', 'impact', 'ctrl', 'forward']):
                    done_sum_immediate_rewards[idx] += agent_final_info[key]

                if ongoing_terminated[idx]:
                    if is_key_valid(key, ['dist', 'near']):
                        done_sum_values[idx] += agent_final_info[key]
                else:
                    if is_key_valid(key, ['survive', 'healthy', 'alive', 'dist', 'near']):
                        done_sum_values[idx] += agent_final_info[key]

    return done_sum_immediate_rewards, done_sum_values

def get_ongoing_rewards_from_info(info, num_agentss):
    ongoing_sum_immediate_rewards = np.zeros(num_agentss, dtype=np.float32)
    ongoing_sum_values = np.zeros(num_agentss, dtype=np.float32)

    # Get the keys for immediate rewards
    immediate_reward_keys = [key for key in info.keys() 
                            if is_key_valid(key, ['linup', 'impact', 'ctrl', 'forward'])]

    for key in immediate_reward_keys:
        agent_rewards = np.array(info[key])
        for idx, reward in enumerate(agent_rewards):
            ongoing_sum_immediate_rewards[idx] += reward

    # Get the keys for value rewards
    value_keys = [key for key in info.keys() 
                if is_key_valid(key, ['survive', 'healthy', 'alive', 'dist', 'near'])]

    for key in value_keys:
        agent_values = np.array(info[key])
        for idx, value in enumerate(agent_values):
            ongoing_sum_values[idx] += value

    return ongoing_sum_immediate_rewards, ongoing_sum_values

def get_final_observations_from_info(info, done_observations):
    # Process observations from final_info for terminated agents
    if 'final_observation' in info and info['final_observation'] is not None:
        final_observation = info['final_observation']
        for idx, agent_final_obs in enumerate(final_observation):
            if agent_final_obs is None:
                continue
            done_observations[idx] = agent_final_obs

    return done_observations
