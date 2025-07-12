import torch
import numpy as np
from typing import Union, Optional, Callable
from math import gamma
import math


def mmd_distance(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    kernel_type: str = 'gaussian',
    biased: bool = False,
    **kernel_kwargs
) -> Union[torch.Tensor, float]:
    """
    Compute the Maximum Mean Discrepancy (MMD) distance between two distributions using Monte Carlo estimation.
    
    The MMD distance between distributions P and Q is defined as:
    MMD²(P, Q) = E_{X,X'~P}[k(X,X')] + E_{Y,Y'~Q}[k(Y,Y')] - 2*E_{X~P,Y~Q}[k(X,Y)]
    
    where:
    - X, X' are independent samples from distribution P
    - Y, Y' are independent samples from distribution Q
    - k is a positive definite kernel function
    
    Args:
        samples_p: Samples from distribution P, shape (n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (n_samples_q, dimension)
        kernel_type: Type of kernel to use ('gaussian', 'laplacian', 'polynomial', 'rq', 'matern')
        biased: If True, uses biased Monte Carlo estimation (includes diagonal terms).
                If False, uses unbiased estimation (excludes diagonal terms).
                Default: False (unbiased)
        **kernel_kwargs: Additional keyword arguments for the kernel function:
            - sigma: Kernel bandwidth parameter (default: 1.0)
            - gamma: Inverse bandwidth for Laplacian kernel (default: 1.0)
            - degree: Degree for polynomial kernel (default: 2)
            - alpha: Scale parameter for rational quadratic kernel (default: 1.0)
            - nu: Smoothness parameter for Matérn kernel (default: 1.5)
    
    Returns:
        MMD distance estimate (scalar)
    
    Examples:
        >>> # Generate samples from two 2D Gaussian distributions
        >>> import torch
        >>> samples_p = torch.randn(1000, 2)  # Standard normal
        >>> samples_q = torch.randn(1000, 2) + 1.0  # Shifted normal
        >>> 
        >>> # Compute MMD distance with Gaussian kernel
        >>> mmd = mmd_distance(samples_p, samples_q, kernel_type='gaussian', sigma=1.0)
        >>> print(f"MMD distance: {mmd:.4f}")
        >>> 
        >>> # Compare different kernels
        >>> mmd_gaussian = mmd_distance(samples_p, samples_q, kernel_type='gaussian', sigma=1.0)
        >>> mmd_laplacian = mmd_distance(samples_p, samples_q, kernel_type='laplacian', gamma=1.0)
        >>> print(f"Gaussian: {mmd_gaussian:.4f}, Laplacian: {mmd_laplacian:.4f}")
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
    device = samples_p.device
    
    # Ensure we have sufficient samples for unbiased estimation
    if not biased and (n_p < 2 or n_q < 2):
        raise ValueError("Need at least 2 samples per distribution for unbiased estimation")
    
    # Get kernel function
    kernel_fn = _get_kernel_function(kernel_type, device, **kernel_kwargs)
    
    # First term: E_{X,X'~P}[k(X,X')]
    # Compute pairwise kernel values within samples from P
    samples_p_i = samples_p.unsqueeze(1)  # (n_p, 1, dim)
    samples_p_j = samples_p.unsqueeze(0)  # (1, n_p, dim)
    
    # Reshape for kernel computation
    samples_p_i_flat = samples_p_i.expand(-1, n_p, -1).reshape(-1, dim)
    samples_p_j_flat = samples_p_j.expand(n_p, -1, -1).reshape(-1, dim)
    
    # Compute kernel values
    kernel_pp_flat = kernel_fn(samples_p_i_flat, samples_p_j_flat)
    kernel_pp = kernel_pp_flat.reshape(n_p, n_p)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term1 = torch.mean(kernel_pp)
    else:
        # Exclude diagonal terms (unbiased estimation)
        mask_p = ~torch.eye(n_p, dtype=torch.bool, device=device)
        term1 = torch.mean(kernel_pp[mask_p])
    
    # Second term: E_{Y,Y'~Q}[k(Y,Y')]
    # Compute pairwise kernel values within samples from Q
    samples_q_i = samples_q.unsqueeze(1)  # (n_q, 1, dim)
    samples_q_j = samples_q.unsqueeze(0)  # (1, n_q, dim)
    
    # Reshape for kernel computation
    samples_q_i_flat = samples_q_i.expand(-1, n_q, -1).reshape(-1, dim)
    samples_q_j_flat = samples_q_j.expand(n_q, -1, -1).reshape(-1, dim)
    
    # Compute kernel values
    kernel_qq_flat = kernel_fn(samples_q_i_flat, samples_q_j_flat)
    kernel_qq = kernel_qq_flat.reshape(n_q, n_q)
    
    if biased:
        # Include diagonal terms (biased estimation)
        term2 = torch.mean(kernel_qq)
    else:
        # Exclude diagonal terms (unbiased estimation)
        mask_q = ~torch.eye(n_q, dtype=torch.bool, device=device)
        term2 = torch.mean(kernel_qq[mask_q])
    
    # Third term: 2 * E_{X~P,Y~Q}[k(X,Y)]
    # Compute all pairwise kernel values between samples from P and Q
    samples_p_expanded = samples_p.unsqueeze(1)  # (n_p, 1, dim)
    samples_q_expanded = samples_q.unsqueeze(0)  # (1, n_q, dim)
    
    # Reshape for kernel computation
    samples_p_flat = samples_p_expanded.expand(-1, n_q, -1).reshape(-1, dim)
    samples_q_flat = samples_q_expanded.expand(n_p, -1, -1).reshape(-1, dim)
    
    # Compute kernel values
    kernel_pq_flat = kernel_fn(samples_p_flat, samples_q_flat)
    kernel_pq = kernel_pq_flat.reshape(n_p, n_q)
    
    # Monte Carlo estimate
    term3 = 2 * torch.mean(kernel_pq)
    
    # Compute MMD squared
    mmd_squared = term1 + term2 - term3
    
    # Ensure non-negative due to numerical errors
    mmd_squared = torch.clamp(mmd_squared, min=0.0)
    
    # Return MMD (take square root)
    mmd_val = torch.sqrt(mmd_squared)
    
    # Return as scalar (convert to float if single value)
    if mmd_val.numel() == 1:
        return mmd_val.item()
    else:
        return mmd_val


def _get_kernel_function(kernel_type: str, device: torch.device, **kwargs) -> Callable:
    """
    Get the appropriate kernel function based on the kernel type.
    
    Args:
        kernel_type: Type of kernel ('gaussian', 'laplacian', 'polynomial', 'rq', 'matern')
        device: Device for computations
        **kwargs: Additional kernel parameters
        
    Returns:
        Kernel function
    """
    # Extract kernel parameters with defaults
    sigma = kwargs.get('sigma', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    degree = kwargs.get('degree', 2)
    alpha = kwargs.get('alpha', 1.0)
    nu = kwargs.get('nu', 1.5)
    
    def gaussian_kernel(x1, x2):
        """Gaussian (RBF) kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.exp(-squared_dist / (2 * sigma ** 2))
    
    def laplacian_kernel(x1, x2):
        """Laplacian kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||)"""
        dist = torch.norm(x1 - x2, dim=-1)
        return torch.exp(-gamma * dist)
    
    def polynomial_kernel(x1, x2):
        """Polynomial kernel: K(x1, x2) = (1 + <x1, x2>)^degree"""
        dot_product = torch.sum(x1 * x2, dim=-1)
        return torch.pow(1 + dot_product, degree)
    
    def rational_quadratic_kernel(x1, x2):
        """Rational quadratic kernel: K(x1, x2) = (1 + ||x1 - x2||^2 / (2*alpha*sigma^2))^(-alpha)"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.pow(1 + squared_dist / (2 * alpha * sigma ** 2), -alpha)
    
    def matern_kernel(x1, x2):
        """Matérn kernel with smoothness parameter nu"""
        dist = torch.norm(x1 - x2, dim=-1)
        scaled_dist = torch.sqrt(2 * nu) * dist / sigma
        
        if nu == 0.5:
            # Exponential kernel
            return torch.exp(-scaled_dist)
        elif nu == 1.5:
            # Matérn 3/2
            return (1 + scaled_dist) * torch.exp(-scaled_dist)
        elif nu == 2.5:
            # Matérn 5/2
            return (1 + scaled_dist + (scaled_dist ** 2) / 3) * torch.exp(-scaled_dist)
        else:
            # General Matérn kernel (more computationally expensive)
            # Handle the case where distance is 0
            zero_mask = (scaled_dist == 0)
            scaled_dist = torch.where(zero_mask, torch.tensor(1e-8, device=device), scaled_dist)
            
            coeff = (2 ** (1 - nu)) / gamma(nu)
            bessel_term = torch.special.modified_bessel_k(nu, scaled_dist)
            kernel_val = coeff * (scaled_dist ** nu) * bessel_term
            
            # Set kernel value to 1 where distance was originally 0
            kernel_val = torch.where(zero_mask, torch.tensor(1.0, device=device), kernel_val)
            return kernel_val
    
    # Select kernel function
    if kernel_type == 'gaussian':
        return gaussian_kernel
    elif kernel_type == 'laplacian':
        return laplacian_kernel
    elif kernel_type == 'polynomial':
        return polynomial_kernel
    elif kernel_type == 'rq' or kernel_type == 'rational_quadratic':
        return rational_quadratic_kernel
    elif kernel_type == 'matern':
        return matern_kernel
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available options: 'gaussian', 'laplacian', 'polynomial', 'rq', 'matern'")


def mmd_distance_batched(
    samples_p: Union[torch.Tensor, np.ndarray],
    samples_q: Union[torch.Tensor, np.ndarray],
    kernel_type: str = 'gaussian',
    biased: bool = False,
    **kernel_kwargs
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute MMD distance for batched samples from two distributions.
    
    This function handles batched computation where you have multiple sets of samples
    from each distribution (e.g., different batch elements).
    
    Args:
        samples_p: Samples from distribution P, shape (batch_size, n_samples_p, dimension)
        samples_q: Samples from distribution Q, shape (batch_size, n_samples_q, dimension)
        kernel_type: Type of kernel to use
        biased: If True, uses biased Monte Carlo estimation. Default: False (unbiased)
        **kernel_kwargs: Additional kernel parameters
    
    Returns:
        MMD distance for each batch element, shape (batch_size,)
    
    Examples:
        >>> # Generate batched samples
        >>> import torch
        >>> batch_size = 10
        >>> samples_p = torch.randn(batch_size, 100, 2)  # 10 batches of 100 samples each
        >>> samples_q = torch.randn(batch_size, 100, 2) + 1.0  # Shifted
        >>> 
        >>> # Compute MMD distance for each batch
        >>> mmd_batch = mmd_distance_batched(samples_p, samples_q, kernel_type='gaussian', sigma=1.0)
        >>> print(f"MMD distances shape: {mmd_batch.shape}")
        >>> print(f"Mean MMD distance: {mmd_batch.mean():.4f}")
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
    
    # Compute MMD distance for each batch element
    mmd_distances = []
    
    for i in range(batch_size):
        mmd = mmd_distance(samples_p[i], samples_q[i], kernel_type=kernel_type, biased=biased, **kernel_kwargs)
        mmd_distances.append(mmd)
    
    # Convert to tensor
    mmd_distances = torch.tensor(mmd_distances, dtype=samples_p.dtype, device=samples_p.device)
    
    return mmd_distances

