# TD3 (Twin Delayed Deep Deterministic Policy Gradients) Source:
# Title: Addressing Function Approximation Error in Actor-Critic Methods
# Authors: Scott Fujimoto, Herke van Hoof, David Meger
# Publication: ICML 2018
# Link: https://arxiv.org/abs/1802.09477

import torch
import torch.nn.functional as F
import copy
from utils.structure.trajectory_handler  import BatchTrajectory
from training.base_trainer import BaseTrainer
from nn.roles.actor import SingleInputActor
from nn.roles.critic import DualInputCritic
from utils.structure.metrics_recorder import create_training_metrics
from training.trainer_utils import create_padding_mask_before_dones, masked_tensor_mean, calculate_value_loss

class TD3(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        """
        Initializes the TD3 Trainer.
        
        Parameters:
        - env_config: Environment configuration object.
        - rl_params: Reinforcement Learning parameters object.
        - device: Device to which model will be allocated.
        """
        trainer_name = "td3"
        self.network_names = ["critic1", "critic2", "actor"]
        
        self.total_steps = 0
        self.policy_update = 2
        network_params, exploration_params = rl_params.network, rl_params.exploration
        critic_network = network_params.critic_network
        actor_network = network_params.actor_network
        
        self.critic1 = DualInputCritic(critic_network, env_config, network_params).to(device)
        self.critic2 = DualInputCritic(critic_network, env_config, network_params).to(device)
        self.actor = SingleInputActor(actor_network, env_config, network_params, exploration_params).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_loss = None

        super(TD3, self).__init__(trainer_name, env_config, rl_params, 
                                  networks = [self.critic1, self.critic2, self.actor], 
                                  target_networks = [self.target_critic1, self.target_critic2, self.target_actor], 
                                  device = device
                                  )
        self.policy_noise = self.get_exploration_rate()

    def get_action(self, state, mask = None, training: bool = False):
        """
        Selects an action given the current state, using the actor network.
        
        Parameters:
        state (torch.Tensor): The current state tensor.
        training (bool): A flag indicating whether the model is in training mode.
        
        Returns:
        torch.Tensor: The selected action tensor.
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
        Trains the model using the provided trajectory batch, optimizing both actor and critics.
        
        Parameters:
        trajectory (BatchTrajectory): A batch of trajectories containing state, action, reward, next_state, and done tensors.
        
        Returns:
        dict: A dictionary containing various training metrics such as estimated value, expected value, and losses.
        """
        self.set_train(training=True)
        critic1_optimizer, critic2_optimizer, actor_optimizer = self.get_optimizers()

        state, action, rewards, next_state, done = trajectory
        mask = create_padding_mask_before_dones(done)


        # Critic Training
        with torch.no_grad():            
            target_value = self.calculate_expected_value(rewards, next_state, done)

        current_Q1 = self.critic1(state, action, mask)
        current_Q2 = self.critic2(state, action, mask)
        critic1_loss = calculate_value_loss(current_Q1, target_value, mask)
        critic2_loss = calculate_value_loss(current_Q2, target_value, mask)

        critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_optimizer.step()

        critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_optimizer.step()

        # Actor Training
        if self.total_steps % self.policy_update == 0:
            self.actor_loss = -masked_tensor_mean(self.critic1(state, self.actor(state)), mask)
            actor_optimizer.zero_grad()
            self.actor_loss.backward()
            actor_optimizer.step()
        
        self.update_step()

        metrics = create_training_metrics(
            estimated_value=current_Q1,
            expected_value=target_value,
            critic_loss1=critic1_loss,
            critic_loss2=critic2_loss,
            actor_loss=self.actor_loss
        )
        return metrics

    def update_step(self):
        """
        Performs necessary updates at each step, such as updating target networks, schedulers and increasing total steps.
        """
        self.update_target_networks()
        self.update_schedulers()
        self.total_steps += 1
        self.policy_noise = self.get_exploration_rate()

    def trainer_calculate_future_value(self, next_state, mask = None, use_target = False):
        """
        Calculates the future value of the next state using target actor and critics.
        
        Parameters:
        gamma (float): The discount factor.
        end_step (int): The end step value.
        next_state (torch.Tensor): The next state tensor.
        
        Returns:
        torch.Tensor: The calculated future value tensor.
        """
        with torch.no_grad():
            if use_target:
                next_action = self.target_actor(next_state, mask)
                noise = torch.normal(torch.zeros_like(next_action), self.policy_noise)
                new_next_action = noise + noise
                target_Q1 = self.target_critic1(next_state, new_next_action, mask)
                target_Q2 = self.target_critic2(next_state, new_next_action, mask)
            else:
                next_action = self.actor(next_state, mask)
                noise = torch.normal(torch.zeros_like(next_action), self.policy_noise)
                new_next_action = noise + noise
                target_Q1 = self.critic1(next_state, new_next_action, mask)
                target_Q2 = self.critic2(next_state, new_next_action, mask)
            target_Q = torch.min(target_Q1, target_Q2)
            future_value = target_Q
        return future_value
