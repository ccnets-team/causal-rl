# DDPG (Deep Deterministic Policy Gradients) Source:
# Title: Continuous control with deep reinforcement learning
# Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra
# Publication: ICLR 2016
# Link: https://arxiv.org/abs/1509.02971

import torch
import torch.nn.functional as F

from utils.structure.trajectories  import BatchTrajectory
from training.base_trainer import BaseTrainer

from nn.roles.critic import DualInputCritic
from nn.roles.actor import SingleInputActor
import copy
from utils.structure.metrics_recorder import create_training_metrics
from training.trainer_utils import create_padding_mask_before_dones, adaptive_sequence_reduction, calculate_value_loss

class DDPG(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        """
        Initialize the DDPG trainer.
        
        Parameters:
        - env_config: Environment configuration object.
        - rl_params: Reinforcement Learning parameters object.
        - device: Device to which model will be allocated.
        """
        trainer_name = "ddpg"
        self.network_names = ["critic", "actor"]
        network_params, exploration_params = rl_params.network, rl_params.exploration
        critic_network = network_params.critic_network
        actor_network = network_params.actor_network

        self.critic = DualInputCritic(critic_network, env_config, network_params).to(device)
        self.actor = SingleInputActor(actor_network, env_config, network_params, exploration_params).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)

        super(DDPG, self).__init__(trainer_name, env_config, rl_params, \
                                   networks = [self.critic, self.actor], \
                                   target_networks= [self.target_critic, self.target_actor],
                                   device = device
                                   )

    def get_action(self, state, mask = None, training: bool = False):
        """
        Get action based on the state and whether the model is in training mode.
        
        Parameters:
        - state (Tensor): The current state of the environment.
        - training (bool): Flag to indicate whether the model is in training mode.
        
        Returns:
        - action (Tensor): The action chosen by the actor network.
        
        Notes:
        During training, actions are sampled with noise for exploration. During evaluation, the action with the highest value is selected.
        """
        with torch.no_grad():
            if training:
                exploration_rate = self.get_exploration_rate()
                action = self.actor.sample_action(state, mask=mask, exploration_rate=exploration_rate)
            else:
                action = self.actor.select_action(state, mask=mask)
        return action
    
    def train_model(self, trajectory: BatchTrajectory):
        """
        Train the DDPG model based on the provided trajectory.
        
        Parameters:
        - trajectory (BatchTrajectory): Sampled trajectory consisting of states, actions, rewards, next_states, and dones.
        
        Returns:
        - metrics (dict): Dictionary containing various training metrics like estimated value, expected value, and losses.
        
        Notes:
        This method optimizes both the actor and the critic networks.
        """
        self.set_train(training=True)
        critic_optimizer, actor_optimizer = self.get_optimizers()

        states, actions, rewards, next_states, dones = trajectory
        mask = create_padding_mask_before_dones(dones)

        # Critic Update
        target_Q = self.calculate_expected_value(rewards, next_states, dones).detach()

       # Compute the target Q value

        # Get current Q estimate
        current_Q = self.critic(states, actions, mask = mask)

        # Compute critic loss
        critic_loss = calculate_value_loss(current_Q, target_Q, mask = mask)

        # Optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        actor_loss = -adaptive_sequence_reduction(self.critic(states, self.actor(states), mask = mask), mask)

        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        self.update_model()

        metrics = create_training_metrics(
            estimated_value=current_Q,
            expected_value=target_Q,
            critic_loss=critic_loss,
            actor_loss=actor_loss
        )        
        return metrics

    def trainer_calculate_future_value(self, next_state, mask = None, use_target = False):
        """
        Calculate the future value of the next state using the target networks.
        
        Parameters:
        - next_state (Tensor): The next state of the environment.
        
        Returns:
        - future_value (Tensor): The calculated future value of the next state.
        
        Notes:
        This method calculates the future value by using the target actor to predict the next action and the target critic to estimate the Q-value of the next state-action pair.
        """
        with torch.no_grad():
            if use_target:
                next_action = self.target_actor(next_state, mask)
                # Add discounted future value element-wise for each item in the batch
                future_value = self.target_critic(next_state, next_action, mask)
            else:
                next_action = self.actor(next_state, mask)
                # Add discounted future value element-wise for each item in the batch
                future_value = self.critic(next_state, next_action, mask)
        return future_value    
    
    
    def update_model(self):
        """
        Updates the target networks and the learning rate schedulers.
        
        Notes:
        This method should be called after every update to the model parameters to synchronize the target networks and learning rate schedulers with the latest model parameters.
        """
        self.update_target_networks()
        self.update_schedulers()
