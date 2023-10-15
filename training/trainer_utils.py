
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
        accumulative_rewards[:, t, :] = (rewards[:, t:, :] *  discount_factors[:, :seq_len - t, :]).sum(dim=1)

    return accumulative_rewards

def get_termination_step(dones):
    num_td_steps = dones.shape[1]
    end_step = torch.argmax(dones, dim=1) + 1
    done = (dones.sum(dim=1) > 0).float()
    end_step[done == 0] = num_td_steps  # If not terminated, set to num_td_steps
    return done, end_step

def masked_mean(tensor, mask):
    return tensor[mask > 0].flatten().mean()

def create_mask_from_dones(dones: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask where the initial columns are ones and subsequent columns are 
    the inverse of the `dones` tensor shifted by one.

    Args:
    - dones (torch.Tensor): The tensor based on which the mask is created.

    Returns:
    - mask (torch.Tensor): The resultant mask tensor.
    """
    mask = torch.ones_like(dones)
    mask[:, 1:, :] = 1 - dones[:, :-1, :]
    
    return mask