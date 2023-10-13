from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from utils.structure.env_observations import EnvObservations

class MLAgentsEnvWrapper(AgentExperienceCollector):
    def __init__(self, env_config, test_env, use_graphics: bool, worker_id, seed = 0, time_scale = 256):
        num_agents = env_config.num_agents
        super(MLAgentsEnvWrapper, self).__init__(num_agents, env_config)
        if use_graphics:
            self.time_scale = 1.5
        else:
            self.time_scale = time_scale
        
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(width = 1280, height = 720, time_scale = self.time_scale)
        self.worker_id = worker_id
        self.file_name = "../unity_environments/" + env_config.env_name +"/"

        self.base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT + worker_id
        self.env = UnityEnvironment(file_name=self.file_name, base_port = self.base_port, \
                                    no_graphics= not use_graphics, seed = seed, worker_id = self.worker_id, side_channels=[self.channel])

        self.use_discrete = env_config.use_discrete

        self.actions = np.zeros((env_config.num_agents, env_config.action_size), dtype = np.float32)
        # self.observations = np.zeros((env_config.num_agents, env_config.state_size), dtype = np.float32)
        self.observations = EnvObservations(self.obs_shapes, self.obs_types, self.num_agents)

        self.agent_ids = np.array([i for i in range(env_config.num_agents)], dtype=int)
        self.agent_life = np.zeros((env_config.num_agents), dtype=bool)
        self.agent_dec = np.zeros((env_config.num_agents), dtype=bool)
        
        self.agent_reset = np.zeros((env_config.num_agents), dtype=bool)

        self.reset_env()
        self.reset_agents()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

        return
    def reset_env(self):
        self.env.reset()
        
        self.actions.fill(0)
        self.observations.reset()
        # self.states.fill(0)
        self.agent_life.fill(False) 
        self.agent_dec.fill(False) 
        self.done = False
            
    def init_variables(self):
        agent_ids = self.agent_ids[self.agent_life]
        states = self.observations[self.agent_life].to_vector()
        actions = self.actions[self.agent_life]
        self.agent_dec.fill(False)
        return agent_ids, states, actions

    def get_steps(self):
        self.dec, self.term = self.env.get_steps(self.behavior_name)

        # First, get the processed observations in dictionary format.
        dec_obs = self.init_observation(self.dec.obs)
        term_obs = self.init_observation(self.term.obs)
        # Now, return the relevant information.
        return self.term.agent_id, self.dec.agent_id, self.term.reward, self.dec.reward, term_obs, dec_obs

    def update_agent_life(self, term_agents, dec_agents):
        self.agent_life[term_agents] = False
        self.agent_reset[term_agents] = True 
        self.agent_life[dec_agents] = True
        
    def step_environment(self):
        agent_ids, state, action = self.init_variables()
        term_agents, dec_agents, term_reward, dec_reward, term_next_obs, dec_next_obs = self.get_steps()
        self.update_agent_life(term_agents, dec_agents)
        
        self.observations[dec_agents] = dec_next_obs
        self.agent_dec[dec_agents] = True
        
        term_next_obs = self.select_observations(term_next_obs)
        dec_next_obs = self.select_observations(dec_next_obs)
        # dec_next_obs = self.convert_obs(dec_next_obs, encode_manager, use_add = True)

        if len(agent_ids) > 0:
            self.push_transitions(agent_ids, state, action, term_agents, term_reward, term_next_obs, term=True)
            dec_agents, dec_reward, dec_next_obs = self.filter_data(dec_agents, term_agents, dec_reward, dec_next_obs)
            self.push_transitions(agent_ids, state, action, dec_agents, dec_reward, dec_next_obs)
        return self.done
    
    def update(self, action):
        action_tuple = ActionTuple()
        if self.use_discrete:
            descrete_action = np.argmax(action, axis=1) + 1
            descrete_action = descrete_action.reshape(-1, 1)
            action_tuple.add_discrete(descrete_action)
        else:
            continuous_action = np.tanh(action)
            action_tuple.add_continuous(continuous_action)
        self.actions[self.agent_dec] = action
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        return
