# import gym
import numpy as np

def get_final_rewards_from_info(info, num_agentss):
    done_sum_immediate_rewards = np.zeros(num_agentss, dtype=np.float32)
    done_sum_values = np.zeros(num_agentss, dtype=np.float32)

    # Process rewards from final_info for terminated agents
    if 'final_info' in info and info['final_info'] is not None:
        final_infos = info['final_info']
        for idx, agent_final_info in enumerate(final_infos):
            if agent_final_info is None:
                continue
            for key in agent_final_info:
                if ('linup' in key.lower() or 'impact' in key.lower() or 'ctrl' in key.lower() or 'forward' in key.lower()) and 'reward' in key.lower() and not key.startswith('_'):
                    done_sum_immediate_rewards[idx] += agent_final_info[key]
                if ('survive' in key.lower() or 'healthy' in key.lower() or 'dist' in key.lower()) and 'reward' in key.lower() and not key.startswith('_'):
                    done_sum_values[idx] += agent_final_info[key]
    return done_sum_immediate_rewards, done_sum_values

def get_ongoing_rewards_from_info(info, num_agentss):
    ongoing_sum_immediate_rewards = np.zeros(num_agentss, dtype=np.float32)
    ongoing_sum_values = np.zeros(num_agentss, dtype=np.float32)
    
    # Get the keys for rewards
    immediate_reward_keys = [key for key in info.keys() 
                            if ('linup' in key.lower() or 'impact' in key.lower() or 'ctrl' in key.lower() or 'forward' in key.lower()) 
                            and 'reward' in key.lower() 
                            and not key.startswith('_')]

    for key in immediate_reward_keys:
        agent_rewards = np.array(info[key])
        for idx, reward in enumerate(agent_rewards):
            ongoing_sum_immediate_rewards[idx] += reward

    value_keys = [key for key in info.keys() 
                if ('survive' in key.lower() or 'healthy' in key.lower() or 'dist' in key.lower()) 
                and 'reward' in key.lower() 
                and not key.startswith('_')]

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
