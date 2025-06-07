import torch

def crps_loss_general(yps, y):
    """ Calculates the CRPS loss. If  y has multiple dimensions, it calculates the CRPS loss for each dimension.
    If y has a single dimension, it calculates the CRPS loss for that dimension.

    Args:
        yps: Tensor of predicted samples [batch x num_samples x dims]
        y: Target values [batch x dims]

    Returns:
        CRPS loss value per batch element [batch x dims]
    """
    num_samples = yps.shape[1]
    if y.dim() == 2:
        y = y.unsqueeze(1)
    mrank = torch.argsort(torch.argsort(yps, dim=1), dim=1)
    return ((2 / (num_samples * (num_samples - 1))) * (yps - y) * (((num_samples - 1) * (y < yps)) - mrank)).sum(
        axis=1
    )


def energy_score_loss(y_pred_samples, y_true):
    """
    Compute energy score loss for multivariate samples.
    
    Energy Score = 2 * E[||Y - X||] - E[||X - X'||]
    where Y is true value, X and X' are independent samples from prediction
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
    
    Returns:
        Energy score loss (lower is better)
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    
    # Expand y_true to match samples shape: [batch_size, n_samples, output_dim]
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples, -1)
    
    # First term: 2 * E[||Y - X||]
    # Compute L2 norm between true values and samples
    diff_true_pred = y_true_expanded - y_pred_samples  # [batch_size, n_samples, output_dim]
    norm_true_pred = torch.norm(diff_true_pred, dim=2)  # [batch_size, n_samples]
    first_term = 2 * torch.mean(norm_true_pred, dim=1)  # [batch_size]
    
    # Second term: E[||X - X'||]
    # Compute pairwise distances between samples
    # Reshape for broadcasting: [batch_size, n_samples, 1, output_dim] and [batch_size, 1, n_samples, output_dim]
    samples_i = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    samples_j = y_pred_samples.unsqueeze(1)  # [batch_size, 1, n_samples, output_dim]
    
    # Compute pairwise differences and norms
    diff_samples = samples_i - samples_j  # [batch_size, n_samples, n_samples, output_dim]
    norm_samples = torch.norm(diff_samples, dim=3)  # [batch_size, n_samples, n_samples]
    
    # Average over all pairs (excluding diagonal)
    mask = ~torch.eye(n_samples, dtype=torch.bool, device=y_pred_samples.device)
    norm_samples_masked = norm_samples[:, mask]  # [batch_size, n_samples*(n_samples-1)]
    second_term = torch.mean(norm_samples_masked, dim=1)  # [batch_size]
    
    # Energy score
    energy_scores = first_term - second_term  # [batch_size]
    
    return energy_scores



