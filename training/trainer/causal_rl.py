'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Title: Invertible-reasoning policy and reverse dynamics for causal reinforcement learning
    Author:
        PARK, JunHo
    Reference:    
        Algorithm Details: https://www.linkedin.com/feed/update/urn:li:activity:7109901892675125248
        Publication: https://patents.google.com/patent/WO2023167576A2/
'''
import torch
import copy
from training.base_trainer import BaseTrainer
from nn.roles.critic import SingleInputCritic
from nn.roles.actor import DualInputActor
from nn.roles.reverse_env import RevEnv
from utils.structure.trajectory_handler  import BatchTrajectory
from utils.structure.metrics_recorder import create_training_metrics
from training.trainer_utils import create_mask_from_dones, masked_mean
class CausalRL(BaseTrainer):

    # This is the initialization of our Causal Reinforcement Learning (CRL) framework, setting up the networks and parameters.
    def __init__(self, env_config, rl_params, device):
        trainer_name = "causal_rl"
        self.network_names = ["critic", "actor", "rev_env"]
        network_params, exploration_params = rl_params.network, rl_params.exploration
        critic_network = network_params.critic_network
        actor_network = network_params.actor_network
        reverse_env_network = network_params.reverse_env_network
        
        self.critic = SingleInputCritic(critic_network, env_config, network_params).to(device)
        self.actor = DualInputActor(actor_network, env_config, network_params, exploration_params).to(device)
        self.revEnv = RevEnv(reverse_env_network, env_config, network_params).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        super(CausalRL, self).__init__(trainer_name, env_config, rl_params, 
                                        networks = [self.critic, self.actor, self.revEnv], \
                                        target_networks =[self.target_critic, None, None], \
                                        device = device
                                        )

    # This function uses the critic to estimate the value of a state, and then uses the actor to determine the action to take.
    # If in training mode, it samples an action based on exploration rate. Otherwise, it simply selects the most likely action.        
    def get_action(self, state, mask = None, training: bool = False):
        exploration_rate = self.get_exploration_rate()
        with torch.no_grad():
            estimated_value = self.critic(state)
            if training:
                action = self.actor.sample_action(state, estimated_value, mask=mask, exploration_rate=exploration_rate)
            else:
                action = self.actor.select_action(state, estimated_value, mask)
        return action

    # This is the core training method of our Causal RL approach.
    # The model employs a cooperative setup among a Critic, an Actor, and a Reverse-environment to learn from the environment's transitions.
    # The various errors (forward, reverse, recurrent) between states are computed based on discrepancies among the current state, reversed state, and recurred state.
    # If curiosity-driven learning is activated, rewards are also influenced by the cooperative actor error.
    # The expected sum of rewards and the estimated value of the critic are used to calculate value loss, and subsequently, critic, actor, and reverse-environment losses.
    # Finally, based on these losses, backpropagation is used to adjust the parameters of the respective networks.
    def train_model(self, trajectory: BatchTrajectory):
        """Training method for the model."""

        self.set_train(training=True)
    
        # Extract the appropriate trajectory segment based on the use_sequence_batch and done flag.
        states, actions, rewards, next_states, dones = trajectory
        mask = create_mask_from_dones(dones)

        # Get the estimated value of the current state from the critic network.
        estimated_value = self.critic(states, mask)
            
        # Predict the action that the actor would take for the current state and its estimated value.
        inferred_action = self.actor.predict_action(states, estimated_value, mask)
        
        # Calculate the reversed state using the original action.
        reversed_state = self.revEnv(next_states, actions, estimated_value, mask)
        # Calculate the recurred state using the inferred action.
        recurred_state = self.revEnv(next_states, inferred_action, estimated_value.detach(), mask)
        
        # Compute the forward cost by checking the discrepancy between the recurred and reversed states.
        forward_cost = self.cost_fn(recurred_state, reversed_state)
        # Compute the reverse cost by checking the discrepancy between the reversed state and the original state.
        reverse_cost = self.cost_fn(reversed_state, states)
        # Compute the recurrent cost by checking the discrepancy between the recurred state and the original state.
        recurrent_cost = self.cost_fn(recurred_state, states)
        
        # Calculate the cooperative critic error using forward and reverse costs in relation to the recurrent cost.
        coop_critic_error = self.error_fn(forward_cost + reverse_cost, recurrent_cost)
        # Calculate the cooperative actor error using recurrent and forward costs in relation to the reverse cost.
        coop_actor_error = self.error_fn(recurrent_cost + forward_cost, reverse_cost)
        # Calculate the cooperative reverse-environment error using reverse and recurrent costs in relation to the forward cost.
        coop_revEnv_error = self.error_fn(reverse_cost + recurrent_cost, forward_cost)      

        # Compute the expected value of the next state and the advantage of taking an action in the current state.
        expected_value, advantage = self.compute_values(trajectory, estimated_value, intrinsic_value=coop_revEnv_error)
            
        # Calculate the value loss based on the difference between estimated and expected values.
        value_loss = self.calculate_value_loss(estimated_value, expected_value, mask)   

        # Derive the critic loss from the cooperative critic error.
        critic_loss = masked_mean(coop_critic_error, mask)

        # Calculate the actor loss by multiplying the advantage with the cooperative actor error.
        actor_loss =  masked_mean(advantage * coop_actor_error, mask)       

        # Derive the reverse-environment loss from the cooperative reverse-environment error.
        revEnv_loss = masked_mean(coop_revEnv_error, mask)
        # Perform backpropagation to adjust the network parameters based on calculated losses.
        self.backwards(
            [self.critic, self.actor, self.revEnv],
            [[value_loss, critic_loss], [actor_loss], [revEnv_loss]])

        # Update the network parameters.
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

    def trainer_calculate_future_value(self, next_state, mask = None):
        with torch.no_grad():
            future_value = self.target_critic(next_state, mask)
        return future_value
    
    def cost_fn(self, predict, target):
        cost = (predict - target.detach()).abs()
        cost = cost.mean(dim=-1, keepdim=True)  # Compute the mean across the state_size dimension
        return cost
    
    def error_fn(self, predict, target):
        error = (predict - target.detach()).abs()
        return error 

    def backwards(self, networks, network_errors):
        """
        Compute the backward pass for each network based on the associated errors.

        In the Causal Reinforcement Learning framework, each network targets a distinct aspect of the causal graph 
        connecting the `next_state`, `action`, and `value` to the current `state`. The purpose of these networks is 
        to capture independent variables in this causal relationship. By backpropagating distinct errors 
        through each network, we ensure that each network hones in on its specific part of the causal graph.

        Step-by-step Explanation:
        1. Initially, all networks are set to not require gradients. This ensures that during the error 
        backpropagation, gradients won't be accidentally updated for the wrong network.
        2. For each network:
        - The network is set to require gradients, making it the current target for the backpropagation.
        - The gradients of this and all subsequent networks in the list are zeroed out. This ensures 
            that gradient accumulation from any previous passes doesn't interfere with the current backpropagation.
        - For each error associated with the network, the error is backpropagated. The computation graph is 
            retained if there are more networks or errors to process; otherwise, it's discarded.
        - Once the errors for the current network are processed, it's set to not require gradients, ensuring 
            that the next error backpropagation won't affect it.
        3. Finally, all networks are set to require gradients again, preparing them for potential future forward passes.

        By this systematic, ordered backpropagation, we ensure each network learns its specific piece of the causal 
        puzzle, enhancing the independence and robustness of the overall learning process.
        
        :param networks: List of neural networks for which the gradients will be computed.
        :param network_errors: List of lists containing errors associated with each network.        
        """
        num_network = len(networks)
        # Temporarily disable gradient computation for all networks
        for network in networks:
            network.requires_grad_(False)        
        for net_idx, (network, errors) in enumerate(zip(networks, network_errors)):
            # Enable gradient computation for the current network
            network.requires_grad_(True)
            # Zero out the gradient for all networks starting from the current network
            for net in networks[net_idx:]:
                net.zero_grad()
            num_error = len(errors)
            for err_idx, error in enumerate(errors):
                # Decide whether to retain the computation graph based on the network and error index
                retain_graph = (net_idx < num_network - 1) or (err_idx < num_error - 1) 

                # Apply the discounted gradients in the backward pass
                error.backward(torch.ones_like(error), retain_graph=retain_graph)
            # Prevent gradient updates for the current network after its errors have been processed
            network.requires_grad_(False)
        # Restore gradient computation capability for all networks for potential future forward passes
        for network in networks:
            network.requires_grad_(True)
        return