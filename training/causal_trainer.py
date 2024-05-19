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
from utils.structure.data_structures  import BatchTrajectory
from utils.structure.metrics_recorder import create_training_metrics
from utils.init import set_seed
from training.managers.normalization_manager import STATE_NORM_SCALE

class CausalTrainer(BaseTrainer):
    # This is the initialization of our Causal Reinforcement Learning (CRL) framework, setting up the networks and parameters.
    def __init__(self, rl_params, device):
        self.trainer_name = 'causal_rl'
        self.network_names = ["critic", "actor", "rev_env"]
        critic_network = rl_params.network.critic_network
        actor_network = rl_params.network.actor_network
        rev_env_network = rl_params.network.rev_env_network
        self.critic_params = rl_params.critic_params
        self.actor_params = rl_params.actor_params
        self.rev_env_params = rl_params.rev_env_params

        env_config = rl_params.env_config
                
        self.critic = SingleInputCritic(critic_network, env_config, rl_params.critic_params).to(device)
        self.actor = DualInputActor(actor_network, env_config, rl_params.use_deterministic, rl_params.actor_params).to(device)
        self.revEnv = RevEnv(rev_env_network, env_config, rl_params.rev_env_params).to(device)
        self.target_critic = copy.deepcopy(self.critic) 

        super(CausalTrainer, self).__init__(env_config, rl_params, 
                                        networks = [self.critic, self.actor, self.revEnv], \
                                        target_networks =[self.target_critic, None, None], \
                                        device = device
                                        )

    # This function uses the critic to estimate the value of a state, and then uses the actor to determine the action to take.
    # If in training mode, it samples an action based on exploration rate. Otherwise, it simply selects the most likely action.        
    def get_action(self, states, padding_mask=None, training: bool = False):
        with torch.no_grad():
            estimated_value = self.critic(states, padding_mask)
            if training and not self.use_deterministic:
                action = self.actor.sample_action(states, estimated_value, padding_mask)
            else:
                action = self.actor.select_action(states, estimated_value, padding_mask)
        return action

    def train_model(self, trajectory: BatchTrajectory):
        """Training method for the model."""

        self.init_train()
    
        # Selects a trajectory segment optimized for sequence-based model input, focusing on recent experiences.
        states, actions, rewards, next_states, dones, padding_mask, end_value = self.select_sequence(trajectory)
        
        # Get the estimated value of the current state from the critic network.
        estimated_value = self.critic(states, padding_mask)    

        # Predict the action that the actor would take for the current state and its estimated value.
        inferred_action = self.actor(states, estimated_value, padding_mask)

        reversed_state, recurred_state = self.process_parallel_rev_env(next_states, actions, inferred_action, estimated_value, padding_mask)
        
        forward_cost, reverse_cost, recurrent_cost = self.compute_transition_costs_from_states(states, reversed_state, recurred_state)
        
        coop_critic_error, coop_actor_error, coop_revEnv_error = self.compute_cooperative_errors_from_costs(forward_cost, reverse_cost, recurrent_cost, reduce_feture_dim = True)

        expected_value = self.compute_expected_value(states, rewards, dones, padding_mask, end_value)
        
        advantage = self.compute_advantage(estimated_value, expected_value, padding_mask)
            
        bipolar_advantage_loss = self.calculate_bipolar_advantage_loss(estimated_value, expected_value, padding_mask)
        
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
            [self.gamma_lambda_learner, self.critic, self.actor, self.revEnv],
            [[bipolar_advantage_loss], [value_loss + critic_loss], [actor_loss], [revEnv_loss]])

        # Update the network parameters.
        self.update_step()
        
        metrics = create_training_metrics(
            estimated_value=estimated_value,
            expected_value=expected_value,
            advantage=advantage,
            value_loss=value_loss,
            critic_loss=critic_loss * self.error_to_state_size_ratio,
            actor_loss=actor_loss * self.error_to_state_size_ratio,
            revEnv_loss=revEnv_loss * self.error_to_state_size_ratio,
            coop_critic_error=coop_critic_error * self.error_to_state_size_ratio,
            coop_actor_error=coop_actor_error * self.error_to_state_size_ratio,
            coop_revEnv_error=coop_revEnv_error * self.error_to_state_size_ratio,
            forward_cost=forward_cost,
            reverse_cost=reverse_cost,
            recurrent_cost=recurrent_cost,
            padding_mask = padding_mask
        )
        return metrics

    def trainer_calculate_future_value(self, next_state, mask):
        with torch.no_grad():
            future_value = self.target_critic(next_state, mask)
        return future_value    
        
    def compute_transition_costs_from_states(self, states, reversed_state, recurred_state):
        # Compute the forward cost by checking the discrepancy between the recurred and reversed states.
        forward_cost = self.cost_fn(recurred_state, reversed_state)
        # Compute the reverse cost by checking the discrepancy between the reversed state and the original state.
        reverse_cost = self.cost_fn(reversed_state, states/STATE_NORM_SCALE)
        # Compute the recurrent cost by checking the discrepancy between the recurred state and the original state.
        recurrent_cost = self.cost_fn(recurred_state, states/STATE_NORM_SCALE)
        return forward_cost, reverse_cost, recurrent_cost
    
    def compute_cooperative_errors_from_costs(self, forward_cost, reverse_cost, recurrent_cost, reduce_feture_dim = False):
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
        The presence of dropout in the reverse-environment model introduces stochasticity into the process, 
        affecting the reproducibility and consistency of the generated states (reversed_state and recurred_state). 
        To address this, the code employs a conditional mechanism:

        1. When dropout is enabled (`self.rev_env_params.dropout > 0.0`), 
        the method ensures identical stochasticity for both reversed_state and recurred_state by resetting the random seed (`set_seed(self.train_iter)`) before each call to `self.revEnv`. 
        This approach guarantees that both state generations encounter the same dropout pattern, maintaining consistency in their stochastic components. 

        2. In the absence of dropout (`self.rev_env_params.dropout == 0.0`), 
        the method concatenates inputs for both reversed and recurred states and processes them in a single pass through `self.revEnv`. 
        This approach is computationally efficient and leverages batch processing capabilities. 
        Since there's no dropout to introduce variability, there's no need for the seed-resetting step to ensure identical stochastic behavior between the state generations.

        :param next_states: Tensor of next states from the environment.
        :param actions: Tensor of actions taken.
        :param inferred_action: Tensor of inferred actions from the actor network.
        :param estimated_value: Tensor of estimated values from the critic network.
        :param padding_mask: Tensor for padding mask.
        :return: A tuple of (reversed_state, recurred_state).
        """
        batch_size = len(next_states)

        if self.rev_env_params.dropout > 0.0:
            set_seed(self.train_iter)
            reversed_state = self.revEnv(next_states, actions, estimated_value, padding_mask)

            set_seed(self.train_iter)
            recurred_state = self.revEnv(next_states, inferred_action, estimated_value.detach(), padding_mask)
        elif self.rev_env_params.dropout == 0.0:
            # Concatenate inputs for reversed and recurred states
            combined_next_states = torch.cat((next_states, next_states), dim=0)
            combined_actions = torch.cat((actions, inferred_action), dim=0)
            combined_estimated_values = torch.cat((estimated_value, estimated_value.detach()), dim=0)
            combined_padding_masks = torch.cat((padding_mask, padding_mask), dim=0)

            # Process the concatenated inputs through self.revEnv
            combined_states = self.revEnv(combined_next_states, combined_actions, combined_estimated_values, combined_padding_masks)

            # Split the results back into reversed_state and recurred_state
            reversed_state, recurred_state = combined_states.split(batch_size, dim=0)
        else:
            assert False, "Invalid dropout value for reverse environment model."
        
        return reversed_state, recurred_state

    def cost_fn(self, predict, target):
        cost = (predict - target.detach()).abs()
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
            reduced_error = torch.matmul(error, self.error_transformation_matrix)
        else:
            reduced_error = error 
        return reduced_error

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
                if error is None:
                    continue
                # Apply the discounted gradients in the backward pass
                error.backward(torch.ones_like(error), retain_graph=retain_graph)
            # Prevent gradient updates for the current network after its errors have been processed
            network.requires_grad_(False)
        # Restore gradient computation capability for all networks for potential future forward passes
        for network in networks:
            network.requires_grad_(True)
        return
    