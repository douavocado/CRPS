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


def energy_score_loss(y_pred_samples, y_true, norm_dim=False):
    """
    Compute energy score loss for multivariate samples using biased Monte Carlo estimation.
    
    Energy Score = 2 * E[||Y - X||] - E[||X - X'||]
    where Y is true value, X and X' are independent samples from prediction
    
    Uses unbiased Monte Carlo estimate for E[||X - X'||] (includes diagonal terms).
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        norm_dim: Whether to normalise the energy score by the output dimension
    
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
    
    # Biased Monte Carlo estimate: include all pairs (including diagonal)
    second_term = torch.mean(norm_samples, dim=(1, 2))  # [batch_size]
    # Now multiply by n_samples/(n_samples-1) to get unbiased estimate
    second_term = second_term * (n_samples / (n_samples - 1))
    
    # Energy score
    energy_scores = first_term - second_term  # [batch_size]
    
    if norm_dim:
        energy_scores = energy_scores / output_dim
    return energy_scores


def variogram_score_loss(y_pred_samples, y_true, weights=None, p=1.0, coords=None):
    """
    Compute variogram score loss for multivariate samples.
    
    The variogram score is defined as:
    S^{(p)}(P,y) = sum_{i,j}^d w_{ij} * (|y_i - y_j|^p - E_{X~P}|X_i - X_j|^p)^2
    
    where:
    - P is the predicted distribution (represented by samples)
    - y is the vector of true targets
    - d is the dimensionality of y and X
    - w_{ij} are the weights
    - X is a sample from the predicted distribution P
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        weights: Weight matrix [output_dim, output_dim] or None for uniform weights
        p: Power parameter (default: 1.0)
        coords: Coordinates for each dimension [output_dim, coord_dim] for spatial weighting
    
    Returns:
        Variogram score loss (lower is better) [batch_size]
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    device = y_pred_samples.device
    
    # Create weights if not provided
    if weights is None:
        if coords is not None:
            # Create weights inversely proportional to spatial distance
            # coords shape: [output_dim, coord_dim]
            coord_diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # [output_dim, output_dim, coord_dim]
            spatial_distances = torch.norm(coord_diffs, dim=2)  # [output_dim, output_dim]
            # Add small epsilon to avoid division by zero and handle diagonal
            epsilon = 1e-6
            weights = 1.0 / (spatial_distances + epsilon)
            # Set diagonal to 0 (no self-comparison)
            weights.fill_diagonal_(0.0)
        else:
            # Uniform weights (excluding diagonal)
            weights = torch.ones(output_dim, output_dim, device=device)
            weights.fill_diagonal_(0.0)
    
    # Ensure weights are on the correct device
    weights = weights.to(device)
    
    # Compute pairwise differences in true values
    # y_true: [batch_size, output_dim]
    y_true_expanded_i = y_true.unsqueeze(2)  # [batch_size, output_dim, 1]
    y_true_expanded_j = y_true.unsqueeze(1)  # [batch_size, 1, output_dim]
    true_pairwise_diffs = torch.abs(y_true_expanded_i - y_true_expanded_j)  # [batch_size, output_dim, output_dim]
    true_pairwise_diffs_p = torch.pow(true_pairwise_diffs, p)  # [batch_size, output_dim, output_dim]
    
    # Compute expected pairwise differences in predicted samples
    # y_pred_samples: [batch_size, n_samples, output_dim]
    pred_expanded_i = y_pred_samples.unsqueeze(3)  # [batch_size, n_samples, output_dim, 1]
    pred_expanded_j = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    pred_pairwise_diffs = torch.abs(pred_expanded_i - pred_expanded_j)  # [batch_size, n_samples, output_dim, output_dim]
    pred_pairwise_diffs_p = torch.pow(pred_pairwise_diffs, p)  # [batch_size, n_samples, output_dim, output_dim]
    
    # Take expectation over samples
    expected_pred_diffs = torch.mean(pred_pairwise_diffs_p, dim=1)  # [batch_size, output_dim, output_dim]
    
    # Compute the squared differences
    squared_diffs = torch.pow(true_pairwise_diffs_p - expected_pred_diffs, 2)  # [batch_size, output_dim, output_dim]
    
    # Apply weights and sum
    weighted_squared_diffs = weights.unsqueeze(0) * squared_diffs  # [batch_size, output_dim, output_dim]
    variogram_scores = torch.sum(weighted_squared_diffs, dim=(1, 2))  # [batch_size]
    
    return variogram_scores

def kernel_score_loss(y_pred_samples, y_true, kernel_type='gaussian', **kwargs):
    """
    Compute kernel score loss for multivariate samples.
    
    The kernel score is defined as:
    S(P,y) = E_{X,X'~P} K(X,X') - 2 * E_{X~P} K(X,y)
    where K is a kernel function.

    Uses biased Monte Carlo estimate for E[K(X,X')].

    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        kernel_type: Type of kernel to use ('gaussian' or 'matern')
        **kwargs: Additional keyword arguments for the kernel function:
            - sigma: Kernel bandwidth parameter (default: 1.0)
            - nu: Smoothness parameter for Matérn kernel (default: 1.5)

    Returns:
        Kernel score loss (lower is better) [batch_size]
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    device = y_pred_samples.device
    
    # Extract kernel parameters
    sigma = kwargs.get('sigma', 1.0)
    nu = kwargs.get('nu', 1.5)  # Only used for Matérn kernel
    
    def gaussian_kernel(x1, x2, sigma=1.0):
        """Gaussian (RBF) kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.exp(-squared_dist / (2 * sigma ** 2))
    
    def matern_kernel(x1, x2, sigma=1.0, nu=1.5):
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
            from math import gamma
            import torch.special
            
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
        kernel_fn = lambda x1, x2: gaussian_kernel(x1, x2, sigma=sigma)
    elif kernel_type == 'matern':
        kernel_fn = lambda x1, x2: matern_kernel(x1, x2, sigma=sigma, nu=nu)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available options: 'gaussian', 'matern'")
    
    # First term: E_{X,X'~P} K(X,X') using biased Monte Carlo estimate
    # Compute all pairwise kernel values between samples (including diagonal)
    samples_expanded_i = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    samples_expanded_j = y_pred_samples.unsqueeze(1)  # [batch_size, 1, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_i_flat = samples_expanded_i.expand(-1, -1, n_samples, -1).reshape(-1, output_dim)
    samples_j_flat = samples_expanded_j.expand(-1, n_samples, -1, -1).reshape(-1, output_dim)
    
    # Compute kernel values
    kernel_values_flat = kernel_fn(samples_i_flat, samples_j_flat)  # [batch_size * n_samples * n_samples]
    kernel_values = kernel_values_flat.reshape(batch_size, n_samples, n_samples)
    
    # Biased Monte Carlo estimate: include all pairs (including diagonal)
    first_term = torch.mean(kernel_values, dim=(1, 2))  # [batch_size]
    
    # Second term: 2 * E_{X~P} K(X,y)
    # Expand y_true to match samples shape
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_flat = y_pred_samples.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    y_true_flat = y_true_expanded.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    
    # Compute kernel values between samples and true values
    kernel_true_flat = kernel_fn(samples_flat, y_true_flat)  # [batch_size * n_samples]
    kernel_true = kernel_true_flat.reshape(batch_size, n_samples)  # [batch_size, n_samples]
    
    # Monte Carlo estimate
    second_term = 2 * torch.mean(kernel_true, dim=1)  # [batch_size]
    
    # Kernel score: E[K(X,X')] - 2*E[K(X,y)]
    kernel_scores = first_term - second_term  # [batch_size]
    
    return kernel_scores

def create_loss_function(config):
    """
    Create a combined loss function from a configuration dictionary.
    
    Args:
        config: Dictionary with the following keys:
            - losses: List of loss function strings (must be non-empty)
                     Valid options: 'crps_loss_general', 'energy_score_loss', 'variogram_score_loss', 'kernel_score_loss'
            - loss_function_args: Optional dictionary of dictionaries containing arguments for each loss function
                                 Format: {loss_function_name: {arg1: value1, arg2: value2, ...}}
            - coefficients: Optional dictionary of multipliers for each loss function
                           Format: {loss_function_name: coefficient_value}
                           If not provided, defaults to 1.0 for all loss functions
    
    Returns:
        A PyTorch-compatible loss function that can be used with autograd
    
    Example:
        config = {
            'losses': ['energy_score_loss', 'variogram_score_loss'],
            'loss_function_args': {
                'energy_score_loss': {'norm_dim': True},
                'variogram_score_loss': {'p': 2.0}
            },
            'coefficients': {
                'energy_score_loss': 1.0,
                'variogram_score_loss': 0.5
            }
        }
        loss_fn = create_loss_function(config)
        loss = loss_fn(y_pred_samples, y_true)
    """
    # Available loss functions
    available_losses = {
        'crps_loss_general': crps_loss_general,
        'energy_score_loss': energy_score_loss,
        'variogram_score_loss': variogram_score_loss,
        'kernel_score_loss': kernel_score_loss
    }
    
    # Validate configuration
    if 'losses' not in config or not config['losses']:
        raise ValueError("Config must contain a non-empty 'losses' list")
    
    losses = config['losses']
    loss_function_args = config.get('loss_function_args', {})
    coefficients = config.get('coefficients', {})
    
    # Validate loss function names
    for loss_name in losses:
        if loss_name not in available_losses:
            raise ValueError(f"Unknown loss function: {loss_name}. Available options: {list(available_losses.keys())}")
    
    # Set default coefficients to 1.0 if not provided
    for loss_name in losses:
        if loss_name not in coefficients:
            coefficients[loss_name] = 1.0
    
    def combined_loss_function(y_pred_samples, y_true, return_components=False, **kwargs):
        """
        Combined loss function that computes weighted sum of specified losses.
        
        Args:
            y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
            y_true: True values [batch_size, output_dim]
            return_components: If True, return individual loss components along with total
            **kwargs: Additional keyword arguments that can be passed to individual loss functions
        
        Returns:
            If return_components=False: Combined loss value [batch_size]
            If return_components=True: (total_loss, individual_losses_dict)
        """
        total_loss = None
        individual_losses = {}
        
        for loss_name in losses:
            loss_fn = available_losses[loss_name]
            coefficient = coefficients[loss_name]
            
            # Get arguments for this specific loss function
            loss_args = loss_function_args.get(loss_name, {})
            
            # Merge with any additional kwargs passed to the combined function
            merged_args = {**loss_args, **kwargs}
            
            # Compute loss
            try:
                loss_value = loss_fn(y_pred_samples, y_true, **merged_args)
            except TypeError as e:
                # Handle case where loss function doesn't accept some arguments
                # Try with only the arguments it expects
                import inspect
                sig = inspect.signature(loss_fn)
                valid_args = {k: v for k, v in merged_args.items() if k in sig.parameters}
                loss_value = loss_fn(y_pred_samples, y_true, **valid_args)
            
            # Store individual loss (before weighting)
            individual_losses[loss_name] = loss_value.mean().item() if hasattr(loss_value, 'mean') else loss_value
            
            # Apply coefficient
            weighted_loss = coefficient * loss_value
            
            # Add to total loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        
        if return_components:
            return total_loss, individual_losses
        else:
            return total_loss
    
    return combined_loss_function



