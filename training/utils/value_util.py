import torch

def compute_lambda_based_returns(values, rewards, dones, gamma_value, lambda_sequence):
    """
    Calculates lambda returns and sum of rewards for each timestep in a sequence with variable gamma and lambda.

    Args:
        values (torch.Tensor): The value estimates for each timestep.
        rewards (torch.Tensor): The rewards received at each timestep.
        dones (torch.Tensor): Indicates whether a timestep is terminal (1 if terminal, 0 otherwise).
        gammas (torch.Tensor): Discount factors for future rewards, varying per timestep.
        lambdas (torch.Tensor): Lambda parameters for TD(lambda) returns, varying per timestep.

    Returns:
        tuple: A tuple containing:
            - lambda_returns (torch.Tensor): The calculated lambda returns for each timestep.
            - sum_rewards (torch.Tensor): The cumulative sum of rewards for each timestep.
    """
    segment_length = len(lambda_sequence)
    lambd = lambda_sequence
    gamma = gamma_value
    
    # Initialize tensors for the segment's lambda returns and sum of rewards
    sum_rewards = torch.zeros_like(values[:,:,:1])
    lambda_returns = torch.zeros_like(values)
    
    # Initialize the last timestep's lambda return to the last timestep's value within the segment
    lambda_returns[:, -1:] = values[:, -1:]

    # Iterate backwards through the segment
    for t in reversed(range(segment_length)):
        with torch.no_grad():
            # Calculate lambda return for each timestep:
            sum_rewards[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * (lambd[t] * sum_rewards[:, t + 1, :])
        # Current reward + discounted future value, adjusted by td_lambda
        lambda_returns[:, t, :] = rewards[:, t, :] + gamma * (1 - dones[:, t, :]) * ((1 -  lambd[t]) * values[:, t + 1, :] + lambd[t] * lambda_returns[:, t + 1, :].clone())

    # Remove the last timestep to align sum rewards with their corresponding states.lambda_returns includes the last timestep's value, so it is not shifted.  
    sum_rewards = sum_rewards[:, :-1, :]
    
    return lambda_returns, sum_rewards        

def calculate_value_size(state_size, min_threshold=1):
    """
    Calculates an optimal latent vector size based on the original state size. The calculated latent size 
    aims to be as close to the square of the state size as possible while considering computational 
    constraints represented by the threshold.

    The function ensures the latent size does not exceed a specified threshold, balancing between 
    achieving a square-like growth and maintaining computational efficiency.

    Args:
    - state_size (int): The size of the original state vector.
    - threshold (int): The maximum allowable size for the latent vector to ensure computational efficiency.

    Returns:
    - int: The calculated size of the latent vector, adhering to the computational constraints.
    """
    # Attempt to calculate a square-like size of the original state size
    proposed_latent_size = int(state_size ** 0.5)

    # Ensure the latent size does not exceed the threshold
    latent_size = max(proposed_latent_size, min_threshold)

    return latent_size

    
