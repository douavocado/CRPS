import torch
import numpy as np
from typing import Union, Optional


def crps_distance(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    biased: bool = False
) -> Union[torch.Tensor, float]:
    """
    Compute the CRPS distance between two distributions using Monte Carlo estimation.
    
    The CRPS distance is defined as the sum of marginal energy distances (1D CRPS) across all dimensions:
    CRPS_distance(P, Q) = Σ_i ED_1D(P_i, Q_i)
    
    where P_i and Q_i are the marginal distributions for dimension i, and ED_1D is the 1D energy distance
    with L1 norm (absolute value), which is equivalent to CRPS for univariate distributions.
    
    For each dimension i, the 1D energy distance is:
    ED_1D(P_i, Q_i) = 2 * E[|X_i - Y_i|] - E[|X_i - X'_i|] - E[|Y_i - Y'_i|]
    
    where:
    - X_i, X'_i are independent samples from marginal distribution P_i
    - Y_i, Y'_i are independent samples from marginal distribution Q_i
    
    Args:
        samples_p: Samples from distribution P, shape (n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (n_samples_q, dimension)
        biased: If True, uses biased Monte Carlo estimation (includes diagonal terms).
                If False, uses unbiased estimation (excludes diagonal terms).
                Default: False (unbiased)
    
    Returns:
        CRPS distance estimate (scalar)
    
    Examples:
        >>> # Generate samples from two 2D Gaussian distributions
        >>> import torch
        >>> samples_p = torch.randn(1000, 2)  # Standard normal
        >>> samples_q = torch.randn(1000, 2) + 1.0  # Shifted normal
        >>> 
        >>> # Compute CRPS distance
        >>> crps_dist = crps_distance(samples_p, samples_q)
        >>> print(f"CRPS distance: {crps_dist:.4f}")
        >>> 
        >>> # Compare biased vs unbiased estimation
        >>> crps_biased = crps_distance(samples_p, samples_q, biased=True)
        >>> crps_unbiased = crps_distance(samples_p, samples_q, biased=False)
        >>> print(f"Biased: {crps_biased:.4f}, Unbiased: {crps_unbiased:.4f}")
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(samples_p, np.ndarray):
        samples_p = torch.from_numpy(samples_p).float()
    if isinstance(samples_q, np.ndarray):
        samples_q = torch.from_numpy(samples_q).float()
    
    # Validate inputs
    if samples_p.dim() != 2 or samples_q.dim() != 2:
        raise ValueError("Input samples must be 2D tensors with shape (n_samples, dimension)")
    
    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError("Samples from both distributions must have the same dimensionality")
    
    n_p, dim = samples_p.shape
    n_q, _ = samples_q.shape
    
    # Ensure we have sufficient samples for unbiased estimation
    if not biased and (n_p < 2 or n_q < 2):
        raise ValueError("Need at least 2 samples per distribution for unbiased estimation")
    
    # Calculate 1D energy distance for each dimension and sum them
    total_crps_distance = 0.0
    
    for d in range(dim):
        # Extract samples for dimension d
        samples_p_d = samples_p[:, d]  # (n_p,)
        samples_q_d = samples_q[:, d]  # (n_q,)
        
        # Calculate 1D energy distance for this dimension
        ed_1d = _energy_distance_1d(samples_p_d, samples_q_d, biased=biased)
        
        # Add to total CRPS distance
        total_crps_distance += ed_1d
    
    # Return as scalar (convert to float if single value)
    if isinstance(total_crps_distance, torch.Tensor) and total_crps_distance.numel() == 1:
        return total_crps_distance.item()
    else:
        return total_crps_distance


def _energy_distance_1d(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    biased: bool = False
) -> torch.Tensor:
    """
    Compute the 1D energy distance between two univariate distributions.
    
    The 1D energy distance with L1 norm (absolute value) is equivalent to CRPS:
    ED_1D(P, Q) = 2 * E[|X - Y|] - E[|X - X'|] - E[|Y - Y'|]
    
    Args:
        samples_p: 1D samples from distribution P, shape (n_samples_p,)
        samples_q: 1D samples from distribution Q, shape (n_samples_q,)
        biased: If True, uses biased Monte Carlo estimation
        
    Returns:
        1D energy distance (scalar tensor)
    """
    n_p = samples_p.shape[0]
    n_q = samples_q.shape[0]
    device = samples_p.device
    
    # First term: 2 * E[|X - Y|]
    # Compute all pairwise absolute differences between samples from P and Q
    samples_p_expanded = samples_p.unsqueeze(1)  # (n_p, 1)
    samples_q_expanded = samples_q.unsqueeze(0)  # (1, n_q)
    
    # Compute pairwise absolute differences
    diff_pq = torch.abs(samples_p_expanded - samples_q_expanded)  # (n_p, n_q)
    
    # Monte Carlo estimate of E[|X - Y|]
    term1 = 2 * torch.mean(diff_pq)
    
    # Second term: -E[|X - X'|]
    # Compute pairwise absolute differences within samples from P
    samples_p_i = samples_p.unsqueeze(1)  # (n_p, 1)
    samples_p_j = samples_p.unsqueeze(0)  # (1, n_p)
    
    diff_pp = torch.abs(samples_p_i - samples_p_j)  # (n_p, n_p)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term2 = torch.mean(diff_pp)
    else:
        # Exclude diagonal terms (unbiased estimation)
        mask_p = ~torch.eye(n_p, dtype=torch.bool, device=device)
        term2 = torch.mean(diff_pp[mask_p])
    
    # Third term: -E[|Y - Y'|]
    # Compute pairwise absolute differences within samples from Q
    samples_q_i = samples_q.unsqueeze(1)  # (n_q, 1)
    samples_q_j = samples_q.unsqueeze(0)  # (1, n_q)
    
    diff_qq = torch.abs(samples_q_i - samples_q_j)  # (n_q, n_q)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term3 = torch.mean(diff_qq)
    else:
        # Exclude diagonal terms (unbiased estimation)
        mask_q = ~torch.eye(n_q, dtype=torch.bool, device=device)
        term3 = torch.mean(diff_qq[mask_q])
    
    # Compute 1D energy distance
    energy_dist_1d = term1 - term2 - term3
    
    return energy_dist_1d


def crps_distance_batched(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    biased: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute CRPS distance for batched samples from two distributions.
    
    This function handles batched computation where you have multiple sets of samples
    from each distribution (e.g., different batch elements).
    
    Args:
        samples_p: Samples from distribution P, shape (batch_size, n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (batch_size, n_samples_q, dimension)
        biased: If True, uses biased Monte Carlo estimation. Default: False (unbiased)
    
    Returns:
        CRPS distance for each batch element, shape (batch_size,)
    
    Examples:
        >>> # Generate batched samples
        >>> import torch
        >>> batch_size = 10
        >>> samples_p = torch.randn(batch_size, 100, 2)  # 10 batches of 100 samples each
        >>> samples_q = torch.randn(batch_size, 100, 2) + 1.0  # Shifted
        >>> 
        >>> # Compute CRPS distance for each batch
        >>> crps_batch = crps_distance_batched(samples_p, samples_q)
        >>> print(f"CRPS distances shape: {crps_batch.shape}")
        >>> print(f"Mean CRPS distance: {crps_batch.mean():.4f}")
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(samples_p, np.ndarray):
        samples_p = torch.from_numpy(samples_p).float()
    if isinstance(samples_q, np.ndarray):
        samples_q = torch.from_numpy(samples_q).float()
    
    # Validate inputs
    if samples_p.dim() != 3 or samples_q.dim() != 3:
        raise ValueError("Input samples must be 3D tensors with shape (batch_size, n_samples, dimension)")
    
    if samples_p.shape[0] != samples_q.shape[0]:
        raise ValueError("Batch sizes must match")
    
    if samples_p.shape[2] != samples_q.shape[2]:
        raise ValueError("Samples from both distributions must have the same dimensionality")
    
    batch_size = samples_p.shape[0]
    n_p = samples_p.shape[1]
    n_q = samples_q.shape[1]
    
    # Ensure we have sufficient samples for unbiased estimation
    if not biased and (n_p < 2 or n_q < 2):
        raise ValueError("Need at least 2 samples per distribution for unbiased estimation")
    
    # Compute CRPS distance for each batch element
    crps_distances = []
    
    for i in range(batch_size):
        crps_dist = crps_distance(samples_p[i], samples_q[i], biased=biased)
        crps_distances.append(crps_dist)
    
    # Convert to tensor
    crps_distances = torch.tensor(crps_distances, dtype=samples_p.dtype, device=samples_p.device)
    
    return crps_distances

