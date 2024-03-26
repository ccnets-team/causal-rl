from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from .settings.reinforcement_agent import ReinforcementAgent

class MLAgentsEnvWrapper(ReinforcementAgent, AgentExperienceCollector): 
    def __init__(self, env_config, seq_len, test_env, use_graphics: bool, device, worker_id, seed = 0, time_scale = 256):
        ReinforcementAgent.__init__(self, env_config, seq_len, device)
        AgentExperienceCollector.__init__(self, env_config, 'cpu')        
        self.env_time_scale = 1.5 if use_graphics else time_scale
        
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(width = 1280, height = 720, time_scale = self.env_time_scale)
        self.worker_id = worker_id
        self.file_name = "../unity_environments/" + env_config.env_name +"/"

        self.base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT + worker_id
        self.env = UnityEnvironment(file_name=self.file_name, base_port = self.base_port, \
                                    no_graphics= not use_graphics, seed = seed, worker_id = self.worker_id, side_channels=[self.channel])

        self.use_discrete = env_config.use_discrete

        self.reset_environment()
        self.reset_agent()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

        return
    def reset_environment(self):
        self.env.reset()
            
    def init_variables(self):
        agent_ids = self.agent_ids[self.agent_life]
        states = self.observations[self.agent_life, -1].to_vector()
        
        actions = self.actions[self.agent_life]
        content_lengthss = self.content_lengths[self.agent_life]
        self.agent_dec.fill(False)
        return agent_ids, states, actions, content_lengthss

    def get_steps(self):
        self.dec, self.term = self.env.get_steps(self.behavior_name)

        # First, get the processed observations in dictionary format.
        dec_obs = self.init_observation(self.dec.obs)
        term_obs = self.init_observation(self.term.obs)
        # Now, return the relevant information.
        return self.term.agent_id, self.dec.agent_id, self.term.reward, self.dec.reward, term_obs, dec_obs

    def update_agent_life(self, term_agents, dec_agents):
        self.agent_life[term_agents] = False
        self.agent_life[dec_agents] = True
    
    def step(self):
        # Retrieve and process environment steps
        agent_ids, state, action, content_lengths = self.init_variables()
        term_agents, dec_agents, term_reward, dec_reward, term_next_obs, dec_next_obs = self.get_steps()
        self.update_agent_life(term_agents, dec_agents)

        # Shift the trajectory data to make room for the new observations
        self.observations.shift(term_agents, dec_agents)
                
        # Update observations with new data
        self.observations[dec_agents, -1] = dec_next_obs  # Update only the most recent time step
        self.agent_dec[dec_agents] = True
        
        # Process and select relevant observations
        term_next_obs = self.select_observations(term_next_obs)
        dec_next_obs = self.select_observations(dec_next_obs)
        # dec_next_obs = self.convert_obs(dec_next_obs, encode_manager, use_add = True)

        # Store transitions in the experience buffer
        if len(agent_ids) > 0:
            self.push_transitions(agent_ids, state, action, term_agents, term_reward, term_next_obs, done_terminated=True, done_truncated=False, content_lengths = content_lengths)
            dec_agents, dec_reward, dec_next_obs = self.filter_data(dec_agents, term_agents, dec_reward, dec_next_obs)
            self.push_transitions(agent_ids, state, action, dec_agents, dec_reward, dec_next_obs, done_terminated=False, done_truncated=False, content_lengths = content_lengths)
        
    def update(self, action, content_lengths):
        action_tuple = ActionTuple()
        if self.use_discrete:
            discrete_action = np.argmax(action, axis=1) + 1
            discrete_action = discrete_action.reshape(-1, 1)
            action_tuple.add_discrete(discrete_action)
        else:
            continuous_action = np.tanh(action)
            action_tuple.add_continuous(continuous_action)
            
        self.actions[self.agent_dec] = action
        self.content_lengthss[self.agent_dec] = content_lengths
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        return