
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

    for t in range(seq_len):
        accumulative_rewards[:, t, :] = (rewards[:, t:, :] * discount_factors[:, :seq_len - t, :]).sum(dim=1)

    return accumulative_rewards

def get_trajectory_mask(dones):
    # Create a zeros tensor of the same shape as 'dones'
    mask = torch.zeros_like(dones)
    
    _dones = dones.squeeze(-1)

    # For sequences where there's a 'done' signal at the end, set all values in that sequence to 1
    mask[_dones[:, -1] == 1, :] = 1
        
    # For sequences without a 'done' signal at the end, set only the first transition to 1
    mask[_dones[:, -1] == 0, 0] = 1
    
    scaling_factor = torch.ones_like(mask).sum() / mask.sum()
    return mask, scaling_factor
    
def reshape_tensors_based_on_mask(tensor, mask):
    """
    Reshape the tensor by selecting only the non-zero mask values.
    """
    # Use the mask to filter out unwanted values in the tensor.
    filtered_tensor = tensor[mask > 0]
    
    # Reshape the tensor to [batch_size*seq_size - ?, 1]
    reshaped_tensor = filtered_tensor.view(-1, 1)
    
    return reshaped_tensor