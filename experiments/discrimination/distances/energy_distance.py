import torch
import numpy as np
from typing import Union, Optional


def energy_distance(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    beta: float = 1.0,
    biased: bool = False
) -> Union[torch.Tensor, float]:
    """
    Compute the energy distance between two distributions using Monte Carlo estimation.
    
    The energy distance between distributions P and Q is defined as:
    ED(P, Q) = 2 * E[||X - Y||^β] - E[||X - X'||^β] - E[||Y - Y'||^β]
    
    where:
    - X, X' are independent samples from distribution P
    - Y, Y' are independent samples from distribution Q
    - β ∈ (0, 2] is the power parameter (typically 1.0 for L1 or 2.0 for L2). It is proper scoring distance/metric for β ∈ (0, 2)
    
    Args:
        samples_p: Samples from distribution P, shape (n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (n_samples_q, dimension)  
        beta: Power parameter for the distance metric (default: 1.0)
        biased: If True, uses biased Monte Carlo estimation (includes diagonal terms).
                If False, uses unbiased estimation (excludes diagonal terms).
                Default: False (unbiased)
    
    Returns:
        Energy distance estimate (scalar)
    
    Examples:
        >>> # Generate samples from two 2D Gaussian distributions
        >>> import torch
        >>> samples_p = torch.randn(1000, 2)  # Standard normal
        >>> samples_q = torch.randn(1000, 2) + 1.0  # Shifted normal
        >>> 
        >>> # Compute energy distance
        >>> ed = energy_distance(samples_p, samples_q)
        >>> print(f"Energy distance: {ed:.4f}")
        >>> 
        >>> # Compare biased vs unbiased estimation
        >>> ed_biased = energy_distance(samples_p, samples_q, biased=True)
        >>> ed_unbiased = energy_distance(samples_p, samples_q, biased=False)
        >>> print(f"Biased: {ed_biased:.4f}, Unbiased: {ed_unbiased:.4f}")
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
    
    if beta <= 0 or beta > 2:
        raise ValueError("Beta parameter must be in (0, 2]")
    
    n_p, dim = samples_p.shape
    n_q, _ = samples_q.shape
    
    # Ensure we have sufficient samples for unbiased estimation
    if not biased and (n_p < 2 or n_q < 2):
        raise ValueError("Need at least 2 samples per distribution for unbiased estimation")
    
    # First term: 2 * E[||X - Y||^β]
    # Compute all pairwise distances between samples from P and Q
    samples_p_expanded = samples_p.unsqueeze(1)  # (n_p, 1, dim)
    samples_q_expanded = samples_q.unsqueeze(0)  # (1, n_q, dim)
    
    # Compute pairwise differences and distances
    diff_pq = samples_p_expanded - samples_q_expanded  # (n_p, n_q, dim)
    dist_pq = torch.norm(diff_pq, dim=2) ** beta  # (n_p, n_q)
    
    # Monte Carlo estimate of E[||X - Y||^β]
    term1 = 2 * torch.mean(dist_pq)
    
    # Second term: -E[||X - X'||^β]
    # Compute pairwise distances within samples from P
    samples_p_i = samples_p.unsqueeze(1)  # (n_p, 1, dim)
    samples_p_j = samples_p.unsqueeze(0)  # (1, n_p, dim)
    
    diff_pp = samples_p_i - samples_p_j  # (n_p, n_p, dim)
    dist_pp = torch.norm(diff_pp, dim=2) ** beta  # (n_p, n_p)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term2 = torch.mean(dist_pp)
    else:
        # Exclude diagonal terms (unbiased estimation)
        # Create mask to exclude diagonal
        mask_p = ~torch.eye(n_p, dtype=torch.bool, device=samples_p.device)
        term2 = torch.mean(dist_pp[mask_p])
    
    # Third term: -E[||Y - Y'||^β]
    # Compute pairwise distances within samples from Q
    samples_q_i = samples_q.unsqueeze(1)  # (n_q, 1, dim)
    samples_q_j = samples_q.unsqueeze(0)  # (1, n_q, dim)
    
    diff_qq = samples_q_i - samples_q_j  # (n_q, n_q, dim)
    dist_qq = torch.norm(diff_qq, dim=2) ** beta  # (n_q, n_q)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term3 = torch.mean(dist_qq)
    else:
        # Exclude diagonal terms (unbiased estimation)
        # Create mask to exclude diagonal
        mask_q = ~torch.eye(n_q, dtype=torch.bool, device=samples_q.device)
        term3 = torch.mean(dist_qq[mask_q])
    
    # Compute energy distance
    energy_dist = term1 - term2 - term3
    
    # Return as scalar (convert to float if single value)
    if energy_dist.numel() == 1:
        return energy_dist.item()
    else:
        return energy_dist


def energy_distance_batched(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    beta: float = 1.0,
    biased: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute energy distance for batched samples from two distributions.
    
    This function handles batched computation where you have multiple sets of samples
    from each distribution (e.g., different batch elements).
    
    Args:
        samples_p: Samples from distribution P, shape (batch_size, n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (batch_size, n_samples_q, dimension)
        beta: Power parameter for the distance metric (default: 1.0)
        biased: If True, uses biased Monte Carlo estimation. Default: False (unbiased)
    
    Returns:
        Energy distance for each batch element, shape (batch_size,)
    
    Examples:
        >>> # Generate batched samples
        >>> import torch
        >>> batch_size = 10
        >>> samples_p = torch.randn(batch_size, 100, 2)  # 10 batches of 100 samples each
        >>> samples_q = torch.randn(batch_size, 100, 2) + 1.0  # Shifted
        >>> 
        >>> # Compute energy distance for each batch
        >>> ed_batch = energy_distance_batched(samples_p, samples_q)
        >>> print(f"Energy distances shape: {ed_batch.shape}")
        >>> print(f"Mean energy distance: {ed_batch.mean():.4f}")
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
    
    # Compute energy distance for each batch element
    energy_distances = []
    
    for i in range(batch_size):
        ed = energy_distance(samples_p[i], samples_q[i], beta=beta, biased=biased)
        energy_distances.append(ed)
    
    # Convert to tensor
    energy_distances = torch.tensor(energy_distances, dtype=samples_p.dtype, device=samples_p.device)
    
    return energy_distances


