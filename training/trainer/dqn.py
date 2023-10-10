# DQN (Deep Q-Network) Source:
# Title: Playing Atari with Deep Reinforcement Learning
# Authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra & Martin Riedmiller
# Publication: NIPS 2013 Deep Learning Workshop
# Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn.functional as F
from training.base_trainer import BaseTrainer
from utils.structure.trajectory_handler  import BatchTrajectory
from nn.roles.actor_network import SingleInputActor as QNetwork
import copy
import torch as Tensor
from utils.structure.metrics_recorder import create_training_metrics
        
class DQN(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        """
        Initializes the DQN Trainer.
        
        Parameters:
        - env_config: Environment configuration object.
        - rl_params: Reinforcement Learning parameters object.
        - device: Device to which model will be allocated.
        """
        trainer_name = "dqn"
        self.network_names = ["q_network"]
        network_params, exploration_params = rl_params.network, rl_params.exploration
        policy_network = network_params.policy_network

        self.q_network = QNetwork(policy_network, env_config, network_params, exploration_params).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        
        super(DQN, self).__init__(trainer_name, env_config, rl_params, \
                                   networks = [self.q_network], \
                                   target_networks = [self.target_q_network],
                                   device = device
                                   )

    def get_action(self, state, training: bool):
        """
        Determines the action to be taken based on the current state.
        
        Parameters:
        - state (Tensor): The current state of the environment.
        - training (bool): Flag to determine whether the model is in training mode.
        
        Returns:
        - action (int): The action chosen by the policy.
        
        Notes:
        During training, actions are sampled based on the exploration rate. During evaluation, the action with the highest Q-value is selected.
        """
        with torch.no_grad():
            if training:
                epsilon = self.get_exploration_rate() 
                action = self.q_network.sample_action(state, epsilon)
            else:
                action = self.q_network.select_action(state)
        return action
    

    def train_model(self, trajectory: BatchTrajectory):
        """
        Train the model on a batch of trajectories.
        
        Parameters:
        - trajectory (BatchTrajectory): The sampled trajectories consisting of states, actions, rewards, next states, and dones.
        
        Returns:
        - metrics (dict): A dictionary containing various training metrics like estimated value, expected value, and value loss.
        
        Notes:
        This method calculates the loss using the Mean Squared Error between the Q-values of the chosen action and the expected Q-values, and then updates the model parameters.
        """
        self.set_train(training=True)
        optimizer = self.get_optimizers()[0]
        
        states, actions, rewards, next_states, dones = self.select_trajectory_segment(trajectory)

        expected_value = self.calculate_expected_value(rewards, next_states, dones).detach()

        predicted_q_value, _ = self.q_network(states)

        # Choose the action with highest probability
        discrete_actions = actions.argmax(dim=-1, keepdim=True)
        q_value_for_action = predicted_q_value.gather(-1, discrete_actions)      

        loss = self.calculate_value_loss(q_value_for_action, expected_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_model()

        metrics = create_training_metrics(
            estimated_value=q_value_for_action,
            expected_value=expected_value,
            value_loss=loss
        )
        return metrics


    def trainer_calculate_future_value(self, next_state: Tensor):
        """
        Calculates the discounted future value of the next state.
        
        Parameters:
        - next_state (Tensor): The next state of the environment.
        
        Returns:
        - future_value (float): The discounted future value of the next state.
        
        Notes:
        This method calculates the future value of the next state by taking the maximum Q-value of the next state, discounted by the discount factor raised to the power of the end step.
        """
        with torch.no_grad():
            next_q_values, _ = self.target_q_network(next_state)
            next_q_value, _ = next_q_values.max(dim=1, keepdim = True)
            future_value = next_q_value
        return future_value


    def update_model(self):
        """
        Updates the target networks and learning rate schedulers.
        
        Notes:
        This method should be called after every update to the model parameters to ensure that the target networks and learning rate schedulers are synchronized with the latest model parameters.
        """
        self.update_target_networks()
        self.update_schedulers()
