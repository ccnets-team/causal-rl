def is_key_valid(key):
    key_lower = key.lower()
    return key_lower.startswith('reward')

def get_final_observations_from_info(info, observations):
    done_observations = observations.copy()
    # Process observations from final_info for terminated agents
    if 'final_observation' in info and info['final_observation'] is not None:
        final_observation = info['final_observation']
        for idx, agent_final_obs in enumerate(final_observation):
            if agent_final_obs is None:
                continue
            done_observations[idx] = agent_final_obs

    return done_observations
