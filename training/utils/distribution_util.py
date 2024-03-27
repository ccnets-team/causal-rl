import torch

def create_prob_dist_from_lambdas(lambda_values):
    """
    Generates a normalized probability distribution from a sequence of lambda values.
    This distribution accounts for the influence of each lambda value on subsequent positions,
    adjusted to ensure the distribution sums to 1.
    """
    # Ensure the last lambda value is set to 1 for stability
    lambda_sequence = lambda_values.detach().clone()
    
    n = len(lambda_sequence)
    # Prepare a reversed lambda sequence expanded across rows for cumulative product calculation
    reversed_lambda_expanded = torch.flip(lambda_sequence, dims=[0]).unsqueeze(0).expand(n, -1)
    
    # Create masks for lower triangular matrix
    lower_tri_mask = torch.tril(torch.ones((n, n), dtype=bool), diagonal=-1)
    lower_tri_mask_incl_diag = torch.tril(torch.ones((n, n), dtype=bool), diagonal=0)
    
    # Set lower triangle including diagonal to 1 for cumulative product calculation
    reversed_lambda_prepared = reversed_lambda_expanded.clone()
    reversed_lambda_prepared[lower_tri_mask_incl_diag] = 1
    
    # Calculate the cumulative product, then zero out the lower triangle excluding the diagonal
    cumprod_reversed_lambda = torch.cumprod(reversed_lambda_prepared, dim=1)
    cumprod_reversed_lambda[lower_tri_mask] = 0

    # Prepare for adjustment by zeroing out lower triangle excluding the diagonal
    reversed_lambda_adjusted = reversed_lambda_prepared.clone()
    
    # Adjust for non-terminal sequences by multiplying with (1 - lambda value) for each sequence
    cumprod_reversed_lambda[:, :-1] *= (1 - reversed_lambda_adjusted[:, 1:])

    # Calculate sequence length ratios, accounting for their position in the sequence
    sequence_lengths = torch.arange(1, n + 1, dtype=torch.float32, device=lambda_values.device)
    sequence_ratios = sequence_lengths / sequence_lengths.sum()
    reversed_sequence_ratios = torch.flip(sequence_ratios, dims=[0]).unsqueeze(1)
    
    # Calculate the contribution of each lambda across the sequence
    sequence_contribution = reversed_sequence_ratios * cumprod_reversed_lambda
    
    # Calculate the contribution of each lambda across the sequence by summing the contributions
    lambda_prob_dist = sequence_contribution.sum(dim=0)
    
    return lambda_prob_dist
