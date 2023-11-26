import gymnasium as gym
import numpy as np
from .settings.agent_experience_collector import AgentExperienceCollector
from environments.settings.gym_rewards import get_final_rewards_from_info, get_final_observations_from_info
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
            env_config: An object containing environment configuration such as number of agents.
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
        self.reset_env()

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
        
    def reset_env(self):
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
    
    def update(self, action) -> bool:
        """
        Updates the environment state with the given action.
        
        Parameters:
            action: The action to be taken in the environment.
            
        Returns:
            A boolean indicating whether the update was successful.
        """
        self.running_cnt += 1
        action_input = self._get_action_input(action)

        next_obs, reward, terminated, truncated, info = self.env.step(action_input)
        ongoing_terminated = np.array(terminated, np.bool8)
        ongoing_truncated = np.array(truncated, np.bool8)
        ongoing_next_obs = np.array(next_obs, np.float32)
        ongoing_reward = np.array(reward, np.float32)

        done = np.logical_or(ongoing_terminated, ongoing_truncated)

        if not self.test_env:
            self.update_for_training(ongoing_terminated, ongoing_truncated, info, action, ongoing_next_obs, ongoing_reward)
        else:
            self.update_for_test(ongoing_terminated, ongoing_truncated, action, ongoing_next_obs, ongoing_reward)

        self.agent_life[~done] = True 
        self.agent_life[done] = False
        self.agent_reset[done] = True 
        
        return False

    def update_for_training(self, ongoing_terminated, ongoing_truncated, info, action, ongoing_next_obs: np.ndarray, ongoing_reward: np.ndarray):
        """
        Processes the information for training update.
        
        Parameters:
            done (np.ndarray): An array indicating which agents are done.
            info: The info provided by the environment after stepping through it.
            action: The action taken by the agents.
            ongoing_next_obs (np.ndarray): The next observations for the agents.
        """
        final_reward = get_final_rewards_from_info(info, self.num_agents)

        final_next_observation = np.zeros_like(ongoing_next_obs)
        final_next_observation = get_final_observations_from_info(info, final_next_observation)
        
        done = np.logical_or(ongoing_terminated, ongoing_truncated)
        
        next_obs = np.where(done[:, np.newaxis], final_next_observation, ongoing_next_obs)

        reward = np.where(done, final_reward, ongoing_reward)

        self.update_agent_data(self.agents, self.observations[:, -1].to_vector(), action, reward, next_obs, ongoing_terminated, ongoing_truncated)
        
        self.process_terminated_and_decision_agents(done, ongoing_next_obs)


    def update_for_test(self, ongoing_terminated, ongoing_truncated, action, ongoing_next_obs: np.ndarray, ongoing_reward: np.ndarray):
        """
        Processes the information for test update.
        
        Parameters:
            done (np.ndarray): An array indicating which agents are done.
            action: The action taken by the agents.
            ongoing_next_obs (np.ndarray): The next observations for the agents.
            ongoing_reward (np.ndarray): The immediate rewards received by the agents.
        """
        next_obs = ongoing_next_obs
        reward = ongoing_reward
        
        done = np.logical_or(ongoing_terminated, ongoing_truncated)
        
        if self.use_graphics:
            self.append_agent_transition(0, self.observations[:, -1].to_vector(), action, reward, next_obs, ongoing_terminated, ongoing_truncated)
        else:
            self.append_agent_transition(0, self.observations[:, -1].to_vector()[0], action[0], reward[0], next_obs[0], ongoing_terminated[0], ongoing_truncated[0])

        self.process_terminated_and_decision_agents(done, next_obs)

        if done.any():
            self.reset_env()

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
