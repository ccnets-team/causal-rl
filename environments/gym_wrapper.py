import gymnasium as gym
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from environments.settings.gym_utils import get_final_observations_from_info
from utils.structure.env_observation import EnvObservation

class GymEnvWrapper(AgentExperienceCollector):
    """
    A wrapper class for gym environments to collect agent experiences and interact with the environment.
    
    Attributes:
        MAX_RANDOM_SEED (int): The maximum value for environment random seeding.
        ...
    """
    MAX_RANDOM_SEED = 1000  # Maximum value for environment random seed

    def __init__(self, env_config, num_td_steps, test_env: bool, use_graphics: bool = False, seed: int = 0):
        """
        Initializes the gym environment with the given configuration.
        
        Parameters:
            env_config: An object containing environment configuration. The number of agents is 
                        determined based on the `test_env` and `use_graphics` flags.
            test_env (bool): A flag indicating if this is a test environment.
            use_graphics (bool): A flag indicating if graphics should be used (visual rendering).
            seed (int): A seed for environment randomization.
        """
        num_agents = 1 if test_env or use_graphics else env_config.num_agents
        super(GymEnvWrapper, self).__init__(num_agents, env_config)
        self.num_agents = num_agents
        self.use_discrete = env_config.use_discrete

        if not self.use_discrete:
            self.action_low = env_config.action_low
            self.action_high = env_config.action_high
        
        self.prev_value = np.zeros(self.num_agents)
        
        if use_graphics:
            self.time_scale = 1.5

        env_name = env_config.env_name
        self.env_name = env_name
        self.test_env = test_env
        self.seed = seed
        self.obs_shapes = env_config.obs_shapes
        self.obs_types = env_config.obs_types
        self.agents = np.arange(self.num_agents)
        self.observations = EnvObservation(self.obs_shapes, self.obs_types, self.num_agents, num_td_steps)

        self.agent_dec = np.ones(self.num_agents, dtype=bool)
        self.agent_life = np.zeros(self.num_agents, dtype=bool)

        self.agent_reset = np.zeros(self.num_agents, dtype=bool)

        self.env = gym.make(env_name, render_mode='human') if use_graphics else gym.make_vec(env_name, num_envs=self.num_agents) 
        self.use_graphics = use_graphics
        all_dec_agents = list(range(self.num_agents))
        self.all_dec_agents = np.array(all_dec_agents)
        self.reset_environment()

    def format_and_assign_observations(self, raw_observations: np.ndarray, observations):
        """
        Processes raw observations from the environment and assigns them to the agents.
        
        Parameters:
            raw_observations (np.ndarray): The raw observations obtained from the environment.
            observations: The EnvObservation object to be populated with processed observations.
        """
        offset = 0
        # We will create a dictionary to store slices of data for each observation type
        sliced_data = {}
        for key in observations.obs_types:
            shape = observations.data[key].shape[2:]  # Exclude num_agents dimension
            end_offset = offset + np.prod(shape)

            # Reshape slice to match observation shape
            if self.use_graphics:
                slice_data = raw_observations[offset:end_offset].reshape((-1, *shape))
                sliced_data[key] = np.array(slice_data, dtype=np.float32)
            else:
                slice_data = raw_observations[:, offset:end_offset].reshape((-1, *shape))
                sliced_data[key] = np.array(slice_data, dtype=np.float32)

            offset = end_offset

        # Assign sliced data to all agents in the observations
        observations[:, -1] = sliced_data
        
    def reset_environment(self):
        """
        Resets the environment and prepares for a new episode.
        """
        self.running_cnt = 0
        random_num = np.random.randint(0, self.MAX_RANDOM_SEED)
        obs, _ = self.env.reset(seed=random_num)
        self.observations.reset()
        self.format_and_assign_observations(obs, self.observations)
        
    def step_environment(self) -> bool:
        """
        Steps the environment with the given action input.
        
        Returns:
            A boolean indicating whether the step was successful.
        """
        return False
    
    def update(self, action, value) -> bool:
        """
        Updates the environment state with the given action.
        
        Parameters:
            action: The action to be taken in the environment.
            
        Returns:
            Currently always returns False. Intended to return a boolean indicating whether the 
            update was successful.
        """
        self.running_cnt += 1
        action_input = self._get_action_input(action)

        _next_obs, _reward, _terminated, _truncated, _info = self.env.step(action_input)
        np_terminated = np.array(_terminated, np.bool8)
        np_truncated = np.array(_truncated, np.bool8)
        np_next_obs = np.array(_next_obs, np.float32)
        np_reward = np.array(_reward, np.float32)

        np_done = np.logical_or(np_terminated, np_truncated)
        
        next_observation = get_final_observations_from_info(_info, np_next_obs)
        if not self.test_env:
            self.update_agent_data(self.agents, self.observations[:, -1].to_vector(), action, np_reward, next_observation, np_terminated, np_truncated, value)
        else:
            if self.use_graphics:
                self.append_agent_transition(0, self.observations[:, -1].to_vector(), action, np_reward, next_observation, np_terminated, np_truncated, value)
            else:
                self.append_agent_transition(0, self.observations[:, -1].to_vector()[0], action[0], np_reward[0], next_observation[0], np_terminated[0], np_truncated[0], value[0])

        self.process_terminated_and_decision_agents(np_done, np_next_obs)            

        if self.use_graphics and np_done.any():
            self.reset_environment()

        self.agent_life[~np_done] = True 
        self.agent_life[np_done] = False
        self.agent_reset[np_done] = True 
        return False

    def process_terminated_and_decision_agents(self, done: np.ndarray, next_obs: np.ndarray):
        """
        Handles the observations and state updates for terminated and decision-required agents.
        
        Parameters:
            done (np.ndarray): An array indicating which agents are done.
            next_obs (np.ndarray): The next observations for the agents.
        """
        term_agents = np.where(done)[0]
        self.observations.shift(term_agents, self.all_dec_agents)
        self.format_and_assign_observations(next_obs, self.observations)
        
    def _get_action_input(self, action) -> np.ndarray:
        """
        Processes the action input based on the current settings (discrete/continuous).
        In graphics mode, the action is taken directly from the first element of the action array.
        
        Parameters:
            action: The raw action input from the agent(s).
        
        Returns:
            The processed action input suitable for the environment.
        """
        if self.use_graphics:
            action_input = action[0]
        else:
            action_input = action[:]        
        if self.use_discrete:
            return np.argmax(action_input, axis=-1)
        action0 = np.tanh(action_input)
        return self.action_low + (action0 + 1.0) * 0.5 * (self.action_high - self.action_low)
