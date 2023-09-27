import torch
import copy
from training.base_trainer import BaseTrainer
from nn.roles.critic_network import SingleInputCritic
from nn.roles.actor_network import DualInputActor
from nn.roles.reverse_env_network import RevEnv
from utils.structure.trajectory_handler  import BatchTrajectory
from utils.structure.metrics_recorder import create_training_metrics

class CausalRL(BaseTrainer):
    def __init__(self, env_config, rl_params, device):
        trainer_name = "crl"
        self.network_names = ["critic", "actor", "rev_env"]
        network_params, exploration_params = rl_params.network, rl_params.exploration
        net = network_params.network
        
        self.use_curiosity = rl_params.algorithm.use_curiosity
        self.curiosity_factor = rl_params.algorithm.curiosity_factor

        self.critic = SingleInputCritic(net, env_config, network_params).to(device)
        self.actor = DualInputActor(net, env_config, network_params, exploration_params).to(device)
        self.revEnv = RevEnv(net, env_config, network_params).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        super(CausalRL, self).__init__(trainer_name, env_config, rl_params, 
                                        networks = [self.critic, self.actor, self.revEnv], \
                                        target_networks =[self.target_critic, None, None], \
                                        device = device
                                        )

    def get_action(self, state, training):
        exploration_rate = self.get_exploration_rate()
        with torch.no_grad():
            estimated_value = self.critic(state)
            if training:
                action = self.actor.sample_action(state, estimated_value, exploration_rate)
            else:
                action = self.actor.select_action(state, estimated_value)
        return action
    
    def train_model(self, trajectory: BatchTrajectory):
        """Training method for the model."""
        self.set_train(training=True)
        states, actions, rewards, next_states, dones = trajectory

        # Extract the appropriate inputs
        state, action, next_state = self.select_transitions(states, actions, next_states)
        
        estimated_value = self.critic(state)
            
        inferred_action = self.actor.predict_action(state, estimated_value)
        reversed_state = self.revEnv(next_state, action, estimated_value)
        recurred_state = self.revEnv(next_state, inferred_action, estimated_value.detach())
        
        forward_cost = self.cost_fn(recurred_state, reversed_state)
        reverse_cost = self.cost_fn(reversed_state, state)
        recurrent_cost = self.cost_fn(recurred_state, state)
        
        coop_critic_error = self.error_fn(forward_cost + reverse_cost, recurrent_cost)
        coop_actor_error = self.error_fn(recurrent_cost + forward_cost, reverse_cost)
        coop_revEnv_error = self.error_fn(reverse_cost + recurrent_cost, forward_cost)      

        if self.use_curiosity:
            rewards = self.trainer_calculate_curiosity_rewards(forward_cost, rewards)
            
        expected_value, advantage = self.compute_values(states, rewards, next_states, dones, estimated_value)
        
        value_loss = self.calculate_value_loss(estimated_value, expected_value)

        critic_loss = (coop_critic_error).mean()

        actor_loss =  (advantage * coop_actor_error).mean()       

        revEnv_loss = (coop_revEnv_error).mean()

        self.backwards(
            [self.critic, self.actor, self.revEnv],
            [[value_loss, critic_loss], [actor_loss], [revEnv_loss]])

        self.update_step()

        metrics = create_training_metrics(
            estimated_value=estimated_value,
            expected_value=expected_value,
            advantage=advantage,
            value_loss=value_loss,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            revEnv_loss=revEnv_loss,
            forward_cost=forward_cost,
            reverse_cost=reverse_cost,
            recurrent_cost=recurrent_cost,
            coop_critic_error=coop_critic_error,
            coop_actor_error=coop_actor_error,
            coop_revEnv_error=coop_revEnv_error
        )
        return metrics

    def update_step(self):
        self.update_optimizers()
        self.update_target_networks()
        self.update_schedulers()
            
    def trainer_calculate_curiosity_rewards(self, cost, rewards):
        with torch.no_grad():
            curiosity_reward = self.curiosity_factor * cost
            if self.use_gae_advantage:
                rewards += curiosity_reward
            else:
                rewards[:, 0, :] += curiosity_reward
        return rewards

    def trainer_calculate_future_value(self, gamma, end_step, next_state):
        target_network, _, _ = self.get_target_networks()
        with torch.no_grad():
            future_value = (gamma**end_step) * target_network(next_state)
        return future_value

    def cost_fn(self, predict, target):
        cost = (predict - target.detach()).abs()
        cost = cost.mean(dim = -1, keepdim = True)
        return cost
    
    def error_fn(self, predict, target):
        error = (predict - target.detach()).abs()
        return error 

    def backwards(self, networks, network_errors):
        num_network = len(networks)
        for network in networks:
            network.requires_grad_(False)        
        for net_idx, (network, errors) in enumerate(zip(networks, network_errors)):
            # Zero gradient for all networks from current index onward
            network.requires_grad_(True)
            for net in networks[net_idx:]:
                net.zero_grad()
            num_error = len(errors)
            for err_idx, error in enumerate(errors):
                retain_graph = (net_idx < num_network - 1) or (err_idx < num_error - 1) 
                error.backward(retain_graph=retain_graph)
            # Prevent gradient updates for the current network
            network.requires_grad_(False)
        # Restore gradient update capability for all networks
        for network in networks:
            network.requires_grad_(True)
        return
