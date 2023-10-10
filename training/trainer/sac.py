# SAC (Soft Actor-Critic) Source:
# Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
# Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
# Publication: ICML 2018
# Link: https://arxiv.org/abs/1801.01290

import torch
import torch.nn.functional as F
import copy
from utils.structure.trajectory_handler  import BatchTrajectory
from training.base_trainer import BaseTrainer
from nn.roles.critic_network import DualInputCritic as Critic
from nn.roles.actor_network import SingleInputActor as PolicyNetwork
from nn.roles.critic_network import SingleInputCritic as ValueNetwork
from utils.structure.metrics_recorder import create_training_metrics

class SAC(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        """
        Initializes the SAC Trainer.
        
        Parameters:
        - env_config: Environment configuration object.
        - rl_params: Reinforcement Learning parameters object.
        - device: Device to which model will be allocated.
        """
        network_params, exploration_params, optimization_params = rl_params.network, rl_params.exploration, rl_params.optimization
        value_network = network_params.value_network
        policy_network = network_params.policy_network
        self.critic1 = Critic(value_network, env_config, network_params).to(device)
        self.critic2 = Critic(value_network, env_config, network_params).to(device)
        self.value = ValueNetwork(value_network, env_config, network_params).to(device)
        self.policy = PolicyNetwork(policy_network, env_config, network_params, exploration_params).to(device)
        
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        super(SAC, self).__init__('sac', env_config, rl_params, \
                                   networks = [self.value, self.policy, self.critic1, self.critic2], \
                                   target_networks= [None, None, self.target_critic1, self.target_critic2], \
                                   device = device)
        
        self.target_entropy = -torch.prod(torch.Tensor(env_config.action_size).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=optimization_params.lr)        

    def get_action(self, state, training):
        """
        Given a state, selects an action using the policy network.
        
        Parameters:
        state (torch.Tensor): The current state tensor.
        training (bool): A flag indicating whether the model is in training mode.
        
        Returns:
        torch.Tensor: The selected action tensor.
        """
        with torch.no_grad():
            actor = self.policy  # Simplifying by using the policy directly
            action = actor.select_action(state)
        return action
    
    def update_value_network(self, state, next_state):
        """
        Updates the value network based on the given state and the next_state.
        
        Parameters:
        state (torch.Tensor): The current state tensor.
        next_state (torch.Tensor): The next state tensor.
        
        Returns:
        torch.Tensor: The computed value loss tensor.
        """
        with torch.no_grad():
            next_action, next_log_pi = self.policy.evaluate_action(next_state)
            qf1_next, qf2_next = self.critic1(next_state, next_action), self.critic2(next_state, next_action)
            min_qf_next = torch.min(qf1_next, qf2_next)
            v_target = min_qf_next - self.alpha * next_log_pi
        
        v_pred = self.value(state)
        value_loss = self.calculate_value_loss(v_pred, v_target)
        return value_loss

    
    def update_q_functions(self, state, action, reward, next_state, done):
        """
        Computes the loss for the Q-functions based on the given transitions.
        
        Parameters:
        state (torch.Tensor): The current state tensor.
        action (torch.Tensor): The action tensor.
        reward (torch.Tensor): The reward tensor.
        next_state (torch.Tensor): The next state tensor.
        done (torch.Tensor): The done tensor indicating whether the episode has ended.
        
        Returns:
        torch.Tensor: The computed Q-functions loss tensor.
        """
        with torch.no_grad():
            next_v = self.value(next_state)
            next_q_value = reward + self.discount_factor * (1 - done) * next_v

        qf1 = self.critic1(state, action)
        qf2 = self.critic2(state, action)

        qf1_loss = self.calculate_value_loss(qf1, next_q_value)
        qf2_loss = self.calculate_value_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        return qf_loss

    def update_policy_network(self, state):
        """
        Updates the policy network based on the given state.
        
        Parameters:
        state (torch.Tensor): The current state tensor.
        
        Returns:
        tuple: A tuple containing the computed policy loss tensor and the log_pi tensor.
        """
        new_action, log_pi = self.policy.evaluate_action(state)
        qf1_pi = self.critic1(state, new_action)
        qf2_pi = self.critic2(state, new_action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
        
        return policy_loss, log_pi

    def optimize_alpha(self, log_pi):
        """
        Optimizes the alpha parameter based on the log_pi values.
        
        Parameters:
        log_pi (torch.Tensor): The log_pi tensor obtained from the policy network.
        """

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

    def train_model(self, trajectory: BatchTrajectory):
        """
        Conducts the training step based on the provided trajectory batch.
        
        Parameters:
        trajectory (BatchTrajectory): The batch of trajectories containing state, action, reward, next_state, and done tensors.
        
        Returns:
        dict: A dictionary containing the computed value loss and actor loss metrics.
        """
        self.set_train(training=True)
        value_optimizer, policy_optimizer, critic1_optimizer, critic2_optimizer = self.get_optimizers()
        
        states, actions, rewards, next_states, dones = self.select_trajectory_segment(trajectory)

        state, action, reward, next_state, done = self.select_first_transitions(states, actions, rewards, next_states, dones)

        # ------------ Value Network Update ------------
        value_loss = self.update_value_network(state, next_state)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # ------------ Q-functions Update ------------
        # Use the updated value network for the target Q-values
                
        qf_loss = self.update_q_functions(state, action, reward, next_state, done)
        critic1_optimizer.zero_grad()
        critic2_optimizer.zero_grad()
        qf_loss.backward()
        critic1_optimizer.step()
        critic2_optimizer.step()

        # ------------ Policy Network Update ------------
        policy_loss, log_pi = self.update_policy_network(state)
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        self.optimize_alpha(log_pi)  # Assuming log_pi is obtained from the policy network during update.
        
        self.update_step()

        metrics = create_training_metrics(
            value_loss=value_loss,
            actor_loss=policy_loss
        )
        return metrics
    
    def update_step(self):
        """
        Performs the necessary updates at each step, such as updating the target networks.
        """
        self.update_target_networks()

