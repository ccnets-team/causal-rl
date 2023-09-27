# A2C (Advantage Actor-Critic) Source:
# Title: Asynchronous Methods for Deep Reinforcement Learning
# Authors: Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver & Koray Kavukcuoglu
# Publication: ICML 2016
# Link: https://arxiv.org/abs/1602.01783

import torch
import copy
import torch.nn.functional as F
from training.base_trainer import BaseTrainer
from utils.structure.trajectory_handler  import BatchTrajectory
from nn.roles.critic_network import SingleInputCritic
from nn.roles.actor_network import SingleInputActor
from utils.structure.metrics_recorder import create_training_metrics

class A2C(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        """
        Initializes the A2C Trainer.
        
        Parameters:
        - env_config (object): Environment configuration object, containing the specifications of the environment.
        - rl_params (object): Reinforcement Learning parameters object, containing various training-related parameters.
        - device (str): Device to which model will be allocated, typically 'cuda' or 'cpu'.
        """
        trainer_name = "a2c"
        self.network_names = ["critic", "actor"]
        network_params, exploration_params = rl_params.network, rl_params.exploration
        network = network_params.network
        
        self.critic = SingleInputCritic(network, env_config, network_params).to(device)
        self.actor = SingleInputActor(network, env_config, network_params, exploration_params).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        super(A2C, self).__init__(trainer_name, env_config, rl_params, 
                                  networks = [self.critic, self.actor], 
                                  target_networks = [self.target_critic, None], 
                                  device = device
                                  )

    def get_action(self, state, training):
        """
        Obtains an action for a given state and whether it is in training or evaluation mode.
        
        Parameters:
        - state (Tensor): The current state of the environment.
        - training (bool): If True, the model is in training mode, else it is in evaluation mode.
        
        Returns:
        - action (int): The chosen action.
        """
        with torch.no_grad():
            if training:
                epsilon = self.get_exploration_rate()
                action = self.actor.sample_action(state, epsilon)
            else:
                action = self.actor.select_action(state)
        return action

    def train_model(self, trajectory: BatchTrajectory):
        """
        Trains the model with the given trajectory batch.
        
        Parameters:
        - trajectory (BatchTrajectory): The batched trajectory data, typically containing states, actions, rewards, next_states, and dones.
        
        Returns:
        - metrics (dict): A dictionary containing training metrics like estimated_value, expected_value, advantage, value_loss, and actor_loss.
        """
        self.set_train(training=True)
        critic_optimizer, actor_optimizer = self.get_optimizers()

        states, actions, rewards, next_states, dones = trajectory

        state, action = self.select_transitions(states, actions)

        estimated_value = self.critic(state)
        
        # Compute the advantage and expected value
        expected_value, advantage = self.compute_values(states, rewards, next_states, dones, estimated_value)

        # Compute critic loss
        value_loss = F.mse_loss(estimated_value, expected_value)
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        log_prob = self.actor.log_prob(state, action)
        
        actor_loss = -(log_prob * advantage.detach()).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        self.update_step()

        metrics = create_training_metrics(
            estimated_value=estimated_value,
            expected_value=expected_value,
            advantage=advantage,
            value_loss=value_loss,
            actor_loss=actor_loss
        )
        return metrics

    def update_step(self):
        """
        Performs updating step, updating target networks and schedulers if any.
        """
        self.update_target_networks()
        self.update_schedulers()

    def trainer_calculate_future_value(self, gamma, end_step, next_state):
        """
        Calculates the future value for a given next_state with discount factor gamma and end_step.
        
        Parameters:
        - gamma (float): The discount factor.
        - end_step (int): The ending step index.
        - next_state (Tensor): The next state of the environment.
        
        Returns:
        - future_value (Tensor): The calculated future value.
        """
        with torch.no_grad():
            future_value = (gamma**end_step) * self.target_critic(next_state)
        return future_value