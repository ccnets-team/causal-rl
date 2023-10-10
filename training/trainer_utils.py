
import torch

def compute_gae(values, rewards, dones, gamma=0.99, tau=0.95):
    # Assuming rewards and dones tensors have shape [batch_size, num_td_steps, 1] 
    # and values has shape [batch_size, num_td_steps+1, 1]
    # IMPORTANT: This function assumes the value of terminal states in `values` tensor is already 0.
    
    gae = 0
    advantages = torch.zeros_like(rewards)

    # Iterate through timesteps in reverse to calculate GAE
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * values[:, t+1] * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * tau * gae * (1 - dones[:, t])
        advantages[:, t] = gae

    return advantages

def get_end_next_state(next_states, end_step):
    # Create batch indices tensor with the same device as end_step
    batch_indices = torch.arange(end_step.size(0)).to(end_step.device)

    # Compute indices for next_state based on end_step
    state_indices = (end_step - 1).squeeze().long()

    # Extract the corresponding next_states using the computed indices
    next_state = next_states[batch_indices, state_indices, :]
    return next_state

def get_discounted_rewards(rewards, discount_factors):
    batch_size, seq_len, _ = rewards.shape
    accumulative_rewards = torch.zeros_like(rewards)
    discount_factors = discount_factors.unsqueeze(-1)

    for t in range(seq_len):
        end_idx = min(t + discount_factors.shape[1], seq_len)
        offset = end_idx - t
        accumulative_rewards[:, t, :] = (rewards[:, t:end_idx, :] * discount_factors[:, :offset]).sum(dim=1)

    return accumulative_rewards