import gymnasium as gym
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from .settings.reinforcement_agent import ReinforcementAgent

class GymEnvWrapper(ReinforcementAgent, AgentExperienceCollector):
    """
    A wrapper class for gym environments to collect agent experiences and interact with the environment.
    
    Attributes:
        MAX_RANDOM_SEED (int): The maximum value for environment random seeding.
        ...
    """
    MAX_RANDOM_SEED = 1000  # Maximum value for environment random seed

    def __init__(self, env_config, test_env: bool, use_graphics: bool = False, seed: int = 0):
        """
        Initializes the gym environment with the given configuration.
        
        Parameters:
            env_config: An object containing environment configuration. The number of agents is 
                        determined based on the `test_env` and `use_graphics` flags.
            test_env (bool): A flag indicating if this is a test environment.
            use_graphics (bool): A flag indicating if graphics should be used (visual rendering).
            seed (int): A seed for environment randomization.
        """
        ReinforcementAgent.__init__(self, env_config)
        AgentExperienceCollector.__init__(self, env_config)
        self.num_agents = 1 if test_env or use_graphics else env_config.num_agents
        self.use_discrete = env_config.use_discrete
        self.env_name = env_config.env_name
        self.test_env = test_env
        self.seed = seed
        self.use_graphics = use_graphics
        self.env = gym.make(self.env_name, render_mode='human') if self.use_graphics else gym.make_vec(self.env_name, num_envs=self.num_agents) 
        self.reset_environment()
        self.agent_dec.fill(True)

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
        
    def step(self) -> bool:
        """
        Steps the environment with the given action input.
        """
        
    def update(self, action, padding_lengths) -> bool:
        """
        Updates the environment state with the given action.

        Parameters:
            action: The action to be taken in the environment.

        Returns:
            Currently always returns False. Intended to return a boolean indicating whether the 
            update was successful.

        The function performs several key operations:
        - Processes the action and advances the environment state.
        - Converts environment outputs to NumPy arrays for consistency.
        - Adjusts rewards and obtains final observations from environment info.
        - Updates agent data with new observations, actions, and rewards.
        - Handles agent transitions based on termination and truncation flags.
        - Resets the environment if necessary, particularly when graphics are enabled.
        - Tracks the active status of agents, marking those that need resetting.
        """
        self.running_cnt += 1
        action_input = self._get_action_input(action)
        self.padding_lengths = padding_lengths
        
        _next_obs, _reward, _terminated, _truncated, _info = self.env.step(action_input)
        np_terminated = np.array(_terminated, np.bool8)
        np_truncated = np.array(_truncated, np.bool8)
        np_next_obs = np.array(_next_obs, np.float32)
        np_reward = np.array(_reward, np.float32)

        np_done = np.logical_or(np_terminated, np_truncated)

        next_observation = np_next_obs.copy()
        
        if np_truncated.ndim == 0:
            # Handle the scalar case
            if np_truncated:
                # Perform the action for the scalar True value
                # You need to define what to do in this case
                next_observation = _info["final_observation"]
        else:
            # Handle the iterable case
            for idx, trunc in enumerate(np_truncated):
                if trunc:
                    next_observation[idx] = _info["final_observation"][idx]
        
        if not self.test_env:
            self.append_transitions(self.agent_ids, self.observations[:, -1].to_vector(), action, np_reward, next_observation, np_terminated, np_truncated, padding_lengths)
        else:
            if self.use_graphics:
                self.add_transition(0, self.observations[:, -1].to_vector(), action, np_reward, next_observation, np_terminated, np_truncated, padding_lengths)
            else:
                self.add_transition(0, self.observations[:, -1].to_vector()[0], action[0], np_reward[0], next_observation[0], np_terminated[0], np_truncated[0], padding_lengths[0])

        self.process_terminated_and_decision_agents(np_done, np_next_obs)            

        if self.use_graphics and np_done.any():
            self.reset_environment()

        return False

    def process_terminated_and_decision_agents(self, done: np.ndarray, next_obs: np.ndarray):
        """
        Handles the observations and state updates for terminated and decision-required agents.
        
        Parameters:
            done (np.ndarray): An array indicating which agents are done.
            next_obs (np.ndarray): The next observations for the agents.
        """
        term_agents = np.where(done)[0]
        self.observations.shift(term_agents, self.agent_ids)
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
