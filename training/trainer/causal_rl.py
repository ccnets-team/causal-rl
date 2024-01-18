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
from utils.structure.trajectories  import BatchTrajectory
from utils.structure.metrics_recorder import create_training_metrics
from training.trainer_utils import create_padding_mask_before_dones
from training.trainer_hook import HybridHook
        
class CausalRL(BaseTrainer):

    # This is the initialization of our Causal Reinforcement Learning (CRL) framework, setting up the networks and parameters.
    def __init__(self, env_config, rl_params, device):
        self.trainer_name = rl_params.trainer_name
        self.trainer_variant = rl_params.trainer_variant
        self.network_names = ["critic", "actor", "rev_env"]
        network_params, exploration_params, optimization_params = rl_params.network, rl_params.exploration, rl_params.optimization
        critic_network = network_params.critic_network
        actor_network = network_params.actor_network
        rev_env_network = network_params.rev_env_network
        
        self.action_ratio = env_config.action_size / (env_config.action_size + env_config.state_size)
        self.state_ratio = 1.0 - self.action_ratio
        self.critic_hook = HybridHook(self.state_ratio, self.action_ratio)
        self.actor_hook = HybridHook(self.state_ratio, self.action_ratio)
        self.revEnv_hook = HybridHook(self.state_ratio, self.action_ratio)
        
        self.critic = SingleInputCritic(critic_network, env_config, network_params.critic_params).to(device)
        self.actor = DualInputActor(actor_network, env_config, network_params.actor_params, exploration_params).to(device)
        self.revEnv = RevEnv(rev_env_network, env_config, network_params.rev_env_params).to(device)
        self.target_critic = copy.deepcopy(self.critic) if optimization_params.use_target_network else None

        super(CausalRL, self).__init__(self.trainer_name, env_config, rl_params, 
                                        networks = [self.critic, self.actor, self.revEnv], \
                                        target_networks =[self.target_critic, None, None], \
                                        device = device
                                        )

    # This function uses the critic to estimate the value of a state, and then uses the actor to determine the action to take.
    # If in training mode, it samples an action based on exploration rate. Otherwise, it simply selects the most likely action.        
    def get_action(self, state, mask = None, training: bool = False):
        exploration_rate = self.get_exploration_rate()
        with torch.no_grad():
            estimated_value = self.critic(state, mask)
            if training:
                action = self.actor.sample_action(state, estimated_value, mask, exploration_rate=exploration_rate)
            else:
                action = self.actor.select_action(state, estimated_value, mask)
                
        return action

    def train_model(self, trajectory: BatchTrajectory):
        if self.trainer_variant == 'classic':
            return self.causal_rl_classic(trajectory)
        elif self.trainer_variant == 'inverse':
            return self.causal_rl_inverse(trajectory)
        elif self.trainer_variant == 'hybrid':
            return self.causal_rl_hybrid(trajectory)
        else:
            assert False, "trainer_mode is not defined"

    # This is the core training method of our Causal RL approach.
    # The model employs a cooperative setup among a Critic, an Actor, and a Reverse-environment to learn from the environment's transitions.
    # The various errors (forward, reverse, recurrent) between states are computed based on discrepancies among the current state, reversed state, and recurred state.
    # If curiosity-driven learning is activated, rewards are also influenced by the cooperative actor error.
    # The expected sum of rewards and the estimated value of the critic are used to calculate value loss, and subsequently, critic, actor, and reverse-environment losses.
    # Finally, based on these losses, backpropagation is used to adjust the parameters of the respective networks.
    def causal_rl_classic(self, trajectory: BatchTrajectory):
        """Training method for the model."""

        self.set_train(training=True)
    
        # Extract the appropriate trajectory segment based on the use_sequence_batch and done flag.
        states, actions, rewards, next_states, dones = trajectory
        padding_mask = create_padding_mask_before_dones(dones)
        # Get the estimated value of the current state from the critic network.
        estimated_value = self.critic(states, padding_mask)
            
        # Predict the action that the actor would take for the current state and its estimated value.
        inferred_action = self.actor(states, estimated_value, padding_mask)

        # Invoke the process_parallel_rev_env method to compute the reversed and recurred states in parallel.
        # This step enhances computational efficiency by processing these states simultaneously.
        reversed_state, recurred_state = self.process_parallel_rev_env(next_states, actions, inferred_action, estimated_value, padding_mask)
                
        forward_cost, reverse_cost, recurrent_cost = self.compute_transition_costs_from_states(states, reversed_state, recurred_state, reduce_feture_dim = False)
        
        coop_critic_error, coop_actor_error, coop_revEnv_error = self.compute_cooperative_errors_from_costs(forward_cost, reverse_cost, recurrent_cost, reduce_feture_dim = True)

        # Compute the expected value of the next state and the advantage of taking an action in the current state.
        expected_value, advantage = self.compute_values(trajectory, estimated_value)
            
        # Calculate the value loss based on the difference between estimated and expected values.
        value_loss = self.calculate_value_loss(estimated_value, expected_value, padding_mask)   

        # Derive the critic loss from the cooperative critic error.
        critic_loss = self.select_tensor_reduction(coop_critic_error, padding_mask)
        
        # Calculate the actor loss by multiplying the advantage with the cooperative actor error.
        actor_loss = self.select_tensor_reduction(advantage * coop_actor_error, padding_mask)       

        # Derive the reverse-environment loss from the cooperative reverse-environment error.
        revEnv_loss = self.select_tensor_reduction(coop_revEnv_error, padding_mask)
        
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

    def causal_rl_inverse(self, trajectory: BatchTrajectory):
        """Training method for the model."""

        self.set_train(training=True)
    
        # Extract the appropriate trajectory segment based on the use_sequence_batch and done flag.
        states, actions, rewards, next_states, dones = trajectory
        padding_mask = create_padding_mask_before_dones(dones)
        # Get the estimated value of the current state from the critic network.
        estimated_value = self.critic(states, padding_mask)
            
        # Predict the action that the actor would take for the current state and its estimated value.
        reversed_states = self.revEnv(next_states, actions, estimated_value, padding_mask)

        inferred_action, recurred_action = self.process_parallel_actor(states, reversed_states, estimated_value, padding_mask)

        forward_cost, reverse_cost, recurrent_cost = self.compute_transition_costs_from_actions(actions, inferred_action, recurred_action, reduce_feture_dim = True)

        coop_critic_error, coop_actor_error, coop_revEnv_error = self.compute_cooperative_errors_from_costs(forward_cost, reverse_cost, recurrent_cost, reduce_feture_dim = False)

        # Compute the expected value of the next state and the advantage of taking an action in the current state.
        expected_value, advantage = self.compute_values(trajectory, estimated_value)
            
        # Calculate the value loss based on the difference between estimated and expected values.
        value_loss = self.calculate_value_loss(estimated_value, expected_value, padding_mask)   

        # Derive the critic loss from the cooperative critic error.
        critic_loss = self.select_tensor_reduction(coop_critic_error, padding_mask)
        
        # Calculate the actor loss by multiplying the advantage with the cooperative actor error.
        actor_loss = self.select_tensor_reduction(advantage * coop_actor_error, padding_mask)       

        # Derive the reverse-environment loss from the cooperative reverse-environment error.
        revEnv_loss = self.select_tensor_reduction(coop_revEnv_error, padding_mask)
        
        # Perform backpropagation to adjust the network parameters based on calculated losses.
        self.backwards(
            [self.critic, self.revEnv, self.actor],
            [[value_loss, critic_loss], [revEnv_loss], [actor_loss]])

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

    def causal_rl_hybrid(self, trajectory: BatchTrajectory):
        """Training method for the model."""

        self.set_train(training=True)
    
        # Extract the appropriate trajectory segment based on the use_sequence_batch and done flag.
        states, actions, rewards, next_states, dones = trajectory
        padding_mask = create_padding_mask_before_dones(dones)
        # Get the estimated value of the current state from the critic network.
        estimated_value = self.critic(states, padding_mask)
            
        # Predict the action that the actor would take for the current state and its estimated value.
        inferred_action = self.actor(states, estimated_value, padding_mask)

        # Invoke the process_parallel_rev_env method to compute the reversed and recurred states in parallel.
        # This step enhances computational efficiency by processing these states simultaneously.
        reversed_state, recurred_state = self.process_parallel_rev_env(next_states, actions, inferred_action, estimated_value, padding_mask)
        recurred_action = self.actor(reversed_state, estimated_value.detach(), padding_mask)
        
        forward_cost1, reverse_cost1, recurrent_cost1 = self.compute_transition_costs_from_states(states, reversed_state, recurred_state, reduce_feture_dim = False)
        forward_cost2, reverse_cost2, recurrent_cost2 = self.compute_transition_costs_from_actions(actions, inferred_action, recurred_action, reduce_feture_dim = True)

        coop_critic_error1, coop_actor_error1, coop_revEnv_error1 = self.compute_cooperative_errors_from_costs(forward_cost1, reverse_cost1, recurrent_cost1, reduce_feture_dim = True)
        coop_critic_error2, coop_actor_error2, coop_revEnv_error2 = self.compute_cooperative_errors_from_costs(forward_cost2, reverse_cost2, recurrent_cost2, reduce_feture_dim = False)

        # Compute the expected value of the next state and the advantage of taking an action in the current state.
        expected_value, advantage = self.compute_values(trajectory, estimated_value)
            
        # Calculate the value loss based on the difference between estimated and expected values.
        value_loss = self.calculate_value_loss(estimated_value, expected_value, padding_mask)   

        # Derive the critic loss from the cooperative critic error.
        coop_critic_error = self.critic_hook.hybrid(coop_critic_error1, coop_critic_error2)
        critic_loss = self.select_tensor_reduction(coop_critic_error, padding_mask)
        
        coop_actor_error = self.actor_hook.hybrid(coop_actor_error1, coop_actor_error2)
        actor_loss = self.select_tensor_reduction(advantage * coop_actor_error, padding_mask)

        coop_revEnv_error = self.revEnv_hook.hybrid(coop_revEnv_error1, coop_revEnv_error2)
        revEnv_loss = self.select_tensor_reduction(coop_revEnv_error, padding_mask)
        
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
            forward_cost=forward_cost1 + forward_cost2,
            reverse_cost=reverse_cost1 + reverse_cost2,
            recurrent_cost=recurrent_cost1 + recurrent_cost2,
            coop_critic_error=coop_critic_error1 + coop_critic_error2,
            coop_actor_error=coop_actor_error1 + coop_actor_error2,
            coop_revEnv_error=coop_revEnv_error1 + coop_revEnv_error2
        )
        return metrics
    
    def compute_transition_costs_from_states(self, states, reversed_state, recurred_state, reduce_feture_dim = False):
        # Compute the forward cost by checking the discrepancy between the recurred and reversed states.
        forward_cost = self.cost_fn(recurred_state, reversed_state, reduce_feture_dim)
        # Compute the reverse cost by checking the discrepancy between the reversed state and the original state.
        reverse_cost = self.cost_fn(reversed_state, states, reduce_feture_dim)
        # Compute the recurrent cost by checking the discrepancy between the recurred state and the original state.
        recurrent_cost = self.cost_fn(recurred_state, states, reduce_feture_dim)
        return forward_cost, reverse_cost, recurrent_cost
    
    def compute_transition_costs_from_actions(self, actions, inferred_action, recurred_action, reduce_feture_dim = True):
        if self.use_discrete:
            # Compute the forward cost by checking the discrepancy between the recurred and reversed states.
            forward_cost = self.cost_fn(inferred_action, actions, reduce_feture_dim)
            # Compute the reverse cost by checking the discrepancy between the reversed state and the original state.
            reverse_cost = self.cost_fn(recurred_action, inferred_action, reduce_feture_dim)
            # Compute the recurrent cost by checking the discrepancy between the recurred state and the original state.
            recurrent_cost = self.cost_fn(recurred_action, actions, reduce_feture_dim)
        else:
            # Compute the forward cost by checking the discrepancy between the recurred and reversed states.
            forward_cost = self.cost_fn(inferred_action.tanh(), actions.tanh(), reduce_feture_dim)
            # Compute the reverse cost by checking the discrepancy between the reversed state and the original state.
            reverse_cost = self.cost_fn(recurred_action.tanh(), inferred_action.tanh(), reduce_feture_dim)
            # Compute the recurrent cost by checking the discrepancy between the recurred state and the original state.
            recurrent_cost = self.cost_fn(recurred_action.tanh(), actions.tanh(), reduce_feture_dim)
        return forward_cost, reverse_cost, recurrent_cost

    def compute_cooperative_errors_from_costs(self, forward_cost, reverse_cost, recurrent_cost, reduce_feture_dim = True):
        # Calculate the cooperative critic error using forward and reverse costs in relation to the recurrent cost.
        coop_critic_error = self.error_fn(forward_cost + reverse_cost, recurrent_cost, reduce_feture_dim)
        # Calculate the cooperative actor error using recurrent and forward costs in relation to the reverse cost.
        coop_actor_error = self.error_fn(recurrent_cost + forward_cost, reverse_cost, reduce_feture_dim)
        # Calculate the cooperative reverse-environment error using reverse and recurrent costs in relation to the forward cost.
        coop_revEnv_error = self.error_fn(reverse_cost + recurrent_cost, forward_cost, reduce_feture_dim)      
        return coop_critic_error, coop_actor_error, coop_revEnv_error

    def process_parallel_rev_env(self, next_states, actions, inferred_action, estimated_value, padding_mask):
        """
        Process the reverse-environment states in parallel.

        :param next_states: Tensor of next states from the environment.
        :param actions: Tensor of actions taken.
        :param inferred_action: Tensor of inferred actions from the actor network.
        :param estimated_value: Tensor of estimated values from the critic network.
        :param padding_mask: Tensor for padding mask.
        :return: A tuple of (reversed_state, recurred_state).
        """
        batch_size = len(next_states)
        
        # Concatenate inputs for reversed and recurred states
        combined_next_states = torch.cat((next_states, next_states), dim=0)
        combined_actions = torch.cat((actions, inferred_action), dim=0)
        combined_estimated_values = torch.cat((estimated_value, estimated_value.detach()), dim=0)
        combined_padding_masks = torch.cat((padding_mask, padding_mask), dim=0)

        # Process the concatenated inputs through self.revEnv
        combined_states = self.revEnv(combined_next_states, combined_actions, combined_estimated_values, combined_padding_masks)

        # Split the results back into reversed_state and recurred_state
        reversed_state, recurred_state = combined_states.split(batch_size, dim=0)

        return reversed_state, recurred_state

    def process_parallel_actor(self, states, reversed_states, estimated_value, padding_mask):
        """
        Process the reverse-environment states in parallel.

        :param next_states: Tensor of next states from the environment.
        :param actions: Tensor of actions taken.
        :param inferred_action: Tensor of inferred actions from the actor network.
        :param estimated_value: Tensor of estimated values from the critic network.
        :param padding_mask: Tensor for padding mask.
        :return: A tuple of (reversed_state, recurred_state).
        """
        batch_size = len(states)
        
        # Concatenate inputs for reversed and recurred states
        combined_states = torch.cat((states, reversed_states), dim=0)
        combined_estimated_values = torch.cat((estimated_value, estimated_value.detach()), dim=0)
        combined_padding_masks = torch.cat((padding_mask, padding_mask), dim=0)

        # Process the concatenated inputs through self.revEnv
        combined_actions = self.actor(combined_states, combined_estimated_values, combined_padding_masks)

        # Split the results back into reversed_state and recurred_state
        inferred_action, recurred_action = combined_actions.split(batch_size, dim=0)
        
        return inferred_action, recurred_action
                
    def update_step(self):
        self.clip_gradients()
        self.update_optimizers()
        self.update_target_networks()
        self.update_schedulers()

    def trainer_calculate_future_value(self, next_state, mask):
        with torch.no_grad():
            if self.use_target_network:  
                future_value = self.target_critic(next_state, mask)
            else:
                future_value = self.critic(next_state, mask)
        return future_value    

    def cost_fn(self, predict, target, reduce_feture_dim = False):
        cost = (predict - target.detach()).abs()
        if reduce_feture_dim:
            cost = cost.mean(dim=-1, keepdim=True)  # Compute the mean across the state_size dimension
        return cost
    
    def error_fn(self, predict, target, reduce_feture_dim = False):
        """
        Compute a balanced error between the combined predicted costs and a single target cost.

        The function addresses a specific scenario in Causal Reinforcement Learning where the predicted value 
        ('predict') is the sum of two cost values, and the target value ('target') is a single cost. The error 
        is the absolute difference between these values, halved to maintain proportional gradient scales. This 
        approach prevents the potential doubling of gradient magnitude due to the comparison of summed costs 
        against a single cost, thereby ensuring stable and effective learning.

        :param predict: Tensor representing the sum of two predicted cost values.
        :param target: Tensor representing a single target cost value.
        :return: Balanced error tensor.
        """
        error = (predict - target.detach()).abs()
        if reduce_feture_dim:
            error = error.mean(dim=-1, keepdim=True)  # Compute the mean across the state_size dimension
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
    