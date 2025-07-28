import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import warnings
from typing import Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path

from experiments.ar.dataset.dataset import MultivariatARDataset
from experiments.ar.training.train_functions import generate_autoregressive_samples


def compute_optimal_kde_bandwidth(samples: np.ndarray, method: str = "silverman") -> float:
    """
    Compute optimal KDE bandwidth using various automatic selection methods.
    
    Args:
        samples: Array of samples, shape (n_samples, n_dimensions)
        method: Method for bandwidth selection
               - "silverman": Silverman's rule of thumb
               - "scott": Scott's rule of thumb  
               - "iqr": Interquartile range method
               - "cv": Cross-validation (experimental)
               
    Returns:
        Optimal bandwidth value
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    # Ensure samples is a numpy array
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    # Check for invalid values and clip extreme values
    if not np.isfinite(samples).all():
        print(f"    Warning: Found non-finite values in samples, removing them...")
        samples = samples[np.isfinite(samples).all(axis=1)]
        if len(samples) == 0:
            print(f"    Warning: No finite samples found, using default bandwidth")
            return 0.1
    
    # Clip extreme values to prevent overflow (keep within reasonable range)
    max_val = np.percentile(np.abs(samples), 99.9)
    if max_val > 1e6:  # If values are extremely large
        print(f"    Warning: Very large sample values detected (max: {max_val:.2e}), clipping...")
        samples = np.clip(samples, -1e6, 1e6)
    
    n_samples, n_dims = samples.shape
    
    if method.lower() == "silverman":
        # Silverman's rule: h = (4σ^5 / (3n))^(1/5) ≈ 1.06 * σ * n^(-1/5)
        # For multivariate: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * σ
        if n_dims == 1:
            std = np.std(samples, ddof=1)
            # Safeguard against infinite/very large std
            if not np.isfinite(std) or std > 1e3:
                print(f"    Warning: Invalid std ({std}), using robust estimator")
                mad = np.median(np.abs(samples - np.median(samples))) / 0.6745  # MAD estimator
                std = min(mad, 0.5)  # Cap at smaller reasonable value
                if not np.isfinite(std) or std < 1e-6:  # If MAD also fails
                    std = 0.1  # Use fixed reasonable value
            bandwidth = 1.06 * std * (n_samples ** (-1/5))
        else:
            # For multivariate case, use geometric mean of marginal bandwidths
            marginal_bandwidths = []
            for dim in range(n_dims):
                std = np.std(samples[:, dim], ddof=1)
                # Safeguard against infinite/very large std
                if not np.isfinite(std) or std > 1e3:
                    print(f"    Warning: Invalid std for dim {dim} ({std}), using robust estimator")
                    mad = np.median(np.abs(samples[:, dim] - np.median(samples[:, dim]))) / 0.6745
                    std = min(mad, 0.5)  # Cap at smaller reasonable value
                    if not np.isfinite(std) or std < 1e-6:  # If MAD also fails
                        std = 0.1  # Use fixed reasonable value
                h_dim = 1.06 * std * (n_samples ** (-1/5))
                marginal_bandwidths.append(h_dim)
            # Safeguard against log of zero or negative values
            marginal_bandwidths = [max(h, 1e-6) for h in marginal_bandwidths]
            bandwidth = np.exp(np.mean(np.log(marginal_bandwidths)))
    
    elif method.lower() == "scott":
        # Scott's rule: h = n^(-1/(d+4)) * σ
        if n_dims == 1:
            std = np.std(samples, ddof=1)
            # Safeguard against infinite/very large std
            if not np.isfinite(std) or std > 1e3:
                print(f"    Warning: Invalid std ({std}), using robust estimator")
                mad = np.median(np.abs(samples - np.median(samples))) / 0.6745  # MAD estimator
                std = min(mad, 0.5)  # Cap at smaller reasonable value
                if not np.isfinite(std) or std < 1e-6:  # If MAD also fails
                    std = 0.1  # Use fixed reasonable value
            bandwidth = std * (n_samples ** (-1/5))
        else:
            # For multivariate case
            marginal_bandwidths = []
            for dim in range(n_dims):
                std = np.std(samples[:, dim], ddof=1)
                # Safeguard against infinite/very large std
                if not np.isfinite(std) or std > 1e3:
                    print(f"    Warning: Invalid std for dim {dim} ({std}), using robust estimator")
                    mad = np.median(np.abs(samples[:, dim] - np.median(samples[:, dim]))) / 0.6745
                    std = min(mad, 0.5)  # Cap at smaller reasonable value
                    if not np.isfinite(std) or std < 1e-6:  # If MAD also fails
                        std = 0.1  # Use fixed reasonable value
                h_dim = std * (n_samples ** (-1/(n_dims + 4)))
                marginal_bandwidths.append(h_dim)
            # Safeguard against log of zero or negative values
            marginal_bandwidths = [max(h, 1e-6) for h in marginal_bandwidths]
            bandwidth = np.exp(np.mean(np.log(marginal_bandwidths)))
    
    elif method.lower() == "iqr":
        # IQR-based method: more robust to outliers
        if n_dims == 1:
            iqr = np.percentile(samples, 75) - np.percentile(samples, 25)
            # Use min of std and IQR/1.34 (robust estimator)
            std_robust = min(np.std(samples, ddof=1), iqr / 1.34)
            bandwidth = 1.06 * std_robust * (n_samples ** (-1/5))
        else:
            marginal_bandwidths = []
            for dim in range(n_dims):
                iqr = np.percentile(samples[:, dim], 75) - np.percentile(samples[:, dim], 25)
                std_robust = min(np.std(samples[:, dim], ddof=1), iqr / 1.34)
                h_dim = 1.06 * std_robust * (n_samples ** (-1/5))
                marginal_bandwidths.append(h_dim)
            bandwidth = np.exp(np.mean(np.log(marginal_bandwidths)))
    
    elif method.lower() == "cv":
        # Cross-validation approach (more computationally expensive)
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.neighbors import KernelDensity
            
            # Try different bandwidths
            bandwidths = np.logspace(-2, 1, 20)
            kde = KernelDensity(kernel='gaussian')
            
            # Use cross-validation to find best bandwidth
            grid = GridSearchCV(kde, {'bandwidth': bandwidths}, cv=5, verbose=0)
            grid.fit(samples)
            bandwidth = grid.best_params_['bandwidth']
        except:
            # Fall back to Silverman if CV fails
            return compute_optimal_kde_bandwidth(samples, method="silverman")
    
    else:
        raise ValueError(f"Unknown bandwidth selection method: {method}")
    
    # Ensure bandwidth is positive, finite, and reasonable
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        print(f"    Warning: Invalid bandwidth ({bandwidth}), using default value")
        bandwidth = 0.1
    else:
        # Clamp bandwidth to reasonable range
        bandwidth = np.clip(bandwidth, 1e-6, 10.0)
    
    return float(bandwidth)


def _check_sample_health(samples: np.ndarray) -> None:
    """Check for potential issues in the samples that could cause plotting problems."""
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    # Basic statistics
    sample_min = np.min(samples)
    sample_max = np.max(samples)
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    
    # Check for problematic values
    n_nan = np.sum(np.isnan(samples))
    n_inf = np.sum(np.isinf(samples))
    n_very_large = np.sum(np.abs(samples) > 1e6)
    
    print(f"    Sample diagnostics: min={sample_min:.2e}, max={sample_max:.2e}, mean={sample_mean:.2e}, std={sample_std:.2e}")
    
    if n_nan > 0:
        print(f"    WARNING: Found {n_nan} NaN values in samples")
    if n_inf > 0:
        print(f"    WARNING: Found {n_inf} infinite values in samples")
    if n_very_large > 0:
        print(f"    WARNING: Found {n_very_large} very large values (>1e6) in samples")
    if sample_std > 1e3:
        print(f"    WARNING: Very large standard deviation ({sample_std:.2e}) - possible model instability")


def resolve_kde_bandwidth(kde_bandwidth: Union[str, float], samples: np.ndarray) -> float:
    """
    Resolve KDE bandwidth parameter, computing automatically if needed.
    
    Args:
        kde_bandwidth: Either a numeric value or "auto" for automatic detection
        samples: Sample data for automatic bandwidth computation
        
    Returns:
        Resolved bandwidth value
    """
    if isinstance(kde_bandwidth, str) and kde_bandwidth.lower() == "auto":
        # Check sample health before computing bandwidth
        _check_sample_health(samples)
        
        # Automatically compute optimal bandwidth
        optimal_bw = compute_optimal_kde_bandwidth(samples, method="silverman")
        print(f"    Auto-detected KDE bandwidth: {optimal_bw:.4f}")
        return optimal_bw
    elif isinstance(kde_bandwidth, (int, float)):
        return float(kde_bandwidth)
    else:
        # Try to parse as float, fall back to auto if it fails
        try:
            return float(kde_bandwidth)
        except (ValueError, TypeError):
            print(f"    Warning: Could not parse kde_bandwidth '{kde_bandwidth}', using auto-detection")
            optimal_bw = compute_optimal_kde_bandwidth(samples, method="silverman")
            print(f"    Auto-detected KDE bandwidth: {optimal_bw:.4f}")
            return optimal_bw


def select_plot_dimensions(
    dimension: int, 
    dataset: Optional[MultivariatARDataset] = None,
    random_state: Optional[int] = None
) -> Tuple[int, int]:
    """
    Select appropriate dimensions for 2D plotting based on data characteristics.
    
    For spatial data: Choose spatially adjacent dimensions
    For non-spatial high-dimensional data: Choose random dimensions
    For low-dimensional data: Use first two dimensions
    
    Args:
        dimension: Total number of dimensions
        dataset: Optional dataset to check for spatial arguments
        random_state: Random seed for dimension selection
        
    Returns:
        Tuple of (dim1, dim2) indices to plot
    """
    if dimension <= 2:
        # For 1D or 2D, use what we have
        return (0, min(1, dimension - 1))
    
    # Check if we have spatial data
    has_spatial = (dataset is not None and 
                   hasattr(dataset, 'spatial_args') and 
                   dataset.spatial_args is not None)
    
    if has_spatial:
        # For spatial data, choose adjacent dimensions
        grid_size = dataset.spatial_args['grid_size']
        height, width = grid_size
        
        # Verify grid consistency
        if height * width != dimension:
            print(f"Warning: Grid size {height}x{width}={height*width} doesn't match dimension {dimension}")
            has_spatial = False
        else:
            # Choose adjacent dimensions (prefer horizontal adjacency for visualization)
            if random_state is not None:
                np.random.seed(random_state)
            
            # Try to find horizontally adjacent dimensions (same row, consecutive columns)
            max_attempts = 100
            for _ in range(max_attempts):
                row = np.random.randint(0, height)
                col = np.random.randint(0, width - 1)  # -1 to ensure we can pick col+1
                
                dim1 = row * width + col
                dim2 = row * width + (col + 1)
                
                if dim1 < dimension and dim2 < dimension:
                    print(f"Selected spatially adjacent dimensions: {dim1} and {dim2} "
                          f"(grid positions ({row},{col}) and ({row},{col+1}))")
                    return (dim1, dim2)
            
            # If horizontal adjacency fails, try vertical
            for _ in range(max_attempts):
                row = np.random.randint(0, height - 1)  # -1 to ensure we can pick row+1
                col = np.random.randint(0, width)
                
                dim1 = row * width + col
                dim2 = (row + 1) * width + col
                
                if dim1 < dimension and dim2 < dimension:
                    print(f"Selected spatially adjacent dimensions: {dim1} and {dim2} "
                          f"(grid positions ({row},{col}) and ({row+1},{col}))")
                    return (dim1, dim2)
            
            print("Warning: Could not find valid adjacent spatial dimensions, falling back to random selection")
            has_spatial = False
    
    if not has_spatial:
        # For non-spatial high-dimensional data, choose random dimensions
        if random_state is not None:
            np.random.seed(random_state)
        
        dim1, dim2 = np.random.choice(dimension, size=2, replace=False)
        print(f"Selected random dimensions for plotting: {dim1} and {dim2}")
        return (int(dim1), int(dim2))
    
    # Fallback to first two dimensions
    return (0, 1)


def plot_ar_inference(
    model: torch.nn.Module,
    dataset: MultivariatARDataset,
    sample_indices: List[int],
    n_samples: int = 100,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    device: str = 'cpu',
    show_intermediate_steps: bool = True,
    contour_levels: List[float] = [0.65, 0.95, 0.99],
    kde_bandwidth: Union[str, float] = "auto",
    random_state: Optional[int] = None
) -> Dict:
    """
    Plot AR model inference process with KDE contours and ground truth distributions.
    
    This function visualizes:
    1. Model predictions vs ground truth for each timestep
    2. KDE contours at specified quantile levels
    3. Intermediate steps for autoregressive generation
    4. True underlying distributions from ground truth AR parameters
    
    For high-dimensional data (>2D):
    - Spatial data: Selects spatially adjacent dimensions for meaningful correlation plots
    - Non-spatial data: Randomly selects 2 dimensions for visualization
    
    Args:
        model: Trained AR model (FGN Encoder, MLP Sampler, or Affine Normal)
        dataset: MultivariatARDataset containing ground truth parameters
        sample_indices: List of sample indices to visualize from the dataset
        n_samples: Number of samples to generate for predictions
        figsize: Figure size for the plot
        save_path: Optional path to save the figure
        device: Device to run inference on
        show_intermediate_steps: Whether to show intermediate autoregressive steps
        contour_levels: Confidence levels for KDE contours (e.g., [0.65, 0.95, 0.99])
        kde_bandwidth: Bandwidth for KDE estimation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing prediction results and plotting data
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    model.eval()
    model = model.to(device)
    
    # Get dataset information
    data_info = dataset.get_data_info()
    ground_truth_params = dataset.get_ground_truth_params()
    input_timesteps = data_info['input_timesteps']
    output_timesteps = data_info['output_timesteps']
    dimension = data_info['dimension']
    
    print(f"Plotting AR inference for {len(sample_indices)} samples:")
    print(f"  Input timesteps: {input_timesteps}, Output timesteps: {output_timesteps}")
    print(f"  Dimension: {dimension}, Samples: {n_samples}")
    print(f"  Model type: {model.__class__.__name__}")
    
    # Prepare results storage
    results = {
        'predictions': [],
        'targets': [],
        'inputs': [],
        'sample_indices': sample_indices,
        'ground_truth_distributions': []
    }
    
    # Create figure
    n_samples_to_plot = len(sample_indices)
    if show_intermediate_steps and output_timesteps > 1:
        # Show each timestep separately
        fig, axes = plt.subplots(
            n_samples_to_plot, output_timesteps + 1,  # +1 for input
            figsize=(figsize[0] * (output_timesteps + 1) / 3, figsize[1] * n_samples_to_plot / 2)
        )
    else:
        # Show only final predictions
        fig, axes = plt.subplots(
            n_samples_to_plot, 2,  # Input and final output
            figsize=(figsize[0], figsize[1] * n_samples_to_plot / 2)
        )
    
    # Ensure axes is 2D
    if n_samples_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    # Process each sample
    for sample_idx, dataset_idx in enumerate(sample_indices):
        print(f"\nProcessing sample {sample_idx + 1}/{n_samples_to_plot} (dataset index {dataset_idx})")
        
        # Get sample from dataset
        sample_data = dataset[dataset_idx]
        inputs = sample_data['input'].unsqueeze(0).to(device)  # Add batch dimension
        targets = sample_data['target'].unsqueeze(0)  # Keep on CPU for now
        
        # Denormalize inputs and targets for visualization (if dataset was normalized)
        inputs_denorm = dataset.denormalise_tensor(inputs.cpu()).to(device)
        targets_denorm = dataset.denormalise_tensor(targets.cpu())
        
        # Store denormalized inputs and targets for plotting
        results['inputs'].append(inputs_denorm.cpu())
        results['targets'].append(targets_denorm)
        
        # Flatten inputs for single timestep case (use original normalized inputs for model inference)
        batch_size, input_timesteps_actual, input_dimension = inputs.shape
        inputs_flat = inputs.reshape(batch_size, input_dimension)
        
        # Generate predictions (model expects normalized inputs and produces normalized outputs)
        with torch.no_grad():
            if output_timesteps == 1:
                # Single timestep prediction
                if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                    predictions = model.forward(n_samples=n_samples, batch_size=batch_size)
                else:
                    predictions = model(inputs_flat, n_samples=n_samples)
                
                # Reshape for consistency: [batch_size, n_samples, 1, dimension]
                predictions = predictions.unsqueeze(2)
            else:
                # Multi-timestep autoregressive prediction
                predictions = generate_autoregressive_samples(
                    model, inputs_flat, output_timesteps, dimension, n_samples, batch_size
                )
        
        # Denormalize predictions for visualization
        predictions = predictions.cpu()  # Shape: [1, n_samples, output_timesteps, dimension]
        
        # Denormalize each timestep of predictions
        predictions_denorm = torch.zeros_like(predictions)
        for t in range(output_timesteps):
            # Extract predictions for timestep t: [1, n_samples, dimension]
            pred_t = predictions[:, :, t, :]
            # Denormalize: [1, n_samples, dimension]
            pred_t_denorm = dataset.denormalise_tensor(pred_t)
            predictions_denorm[:, :, t, :] = pred_t_denorm
        
        results['predictions'].append(predictions_denorm)
        
        # Get ground truth distribution parameters for this sample
        gt_distributions = _get_ground_truth_distributions(
            dataset, dataset_idx, output_timesteps, ground_truth_params
        )
        results['ground_truth_distributions'].append(gt_distributions)
        
        # Plot for this sample (use denormalized data)
        # Use sample-specific random state to ensure different samples get different dimensions
        # but the same sample gets consistent dimensions across epochs
        sample_random_state = (random_state + dataset_idx) if random_state is not None else dataset_idx
        _plot_sample_inference(
            sample_idx, axes, inputs_denorm.cpu(), targets_denorm, predictions_denorm, gt_distributions,
            contour_levels, kde_bandwidth, show_intermediate_steps, dimension, dataset, sample_random_state
        )
    
    # Add overall title and adjust layout
    model_name = model.__class__.__name__
    fig.suptitle(f'{model_name} AR Inference Visualization\n'
                 f'Contours: {[int(p*100) for p in contour_levels]}% | '
                 f'Samples: {n_samples} | Dimension: {dimension}D',
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # plt.show()
    
    return results


def _get_ground_truth_distributions(
    dataset: MultivariatARDataset,
    dataset_idx: int,
    output_timesteps: int,
    ground_truth_params: Dict
) -> List[Dict]:
    """
    Get ground truth distribution parameters for each output timestep.
    
    Computes conditional distributions based only on data available during 
    autoregressive inference (input sequence + expected values of previous predictions).
    
    Returns:
        List of dicts containing 'mean', 'cov' for each timestep
    """
    # Get the sample data to determine the time context
    sample_data = dataset[dataset_idx]
    series_idx = sample_data['series_idx'].item()
    time_start = sample_data['time_start'].item()
    
    # Get the input sequence (only data available to the model)
    inputs = sample_data['input']  # Shape: [input_timesteps, dimension]
    
    # Denormalize inputs for ground truth computation in original space
    inputs = dataset.denormalise_tensor(inputs)
    
    # Extract AR parameters from ground truth
    generation_metadata = ground_truth_params['generation_metadata']
    A_matrices = generation_metadata['A_matrices']  # List of AR coefficient matrices
    noise_cov = generation_metadata['noise_cov']
    dimension = dataset.config['dimension']
    ar_order = dataset.config['ar_order']
    input_timesteps = dataset.config['input_timesteps']
    
    # Convert to numpy if they're tensors
    if torch.is_tensor(A_matrices):
        A_matrices = [A_matrices.numpy()]
    elif isinstance(A_matrices, list) and torch.is_tensor(A_matrices[0]):
        A_matrices = [A.numpy() for A in A_matrices]
    
    if torch.is_tensor(noise_cov):
        noise_cov = noise_cov.numpy()
        
    if torch.is_tensor(inputs):
        inputs = inputs.numpy()
    
    distributions = []
    
    # Build autoregressive context starting with input sequence
    # This simulates what's available during actual AR inference
    ar_context = inputs.copy()  # Start with input sequence
    
    # Track accumulated prediction covariances for proper uncertainty propagation
    accumulated_covariances = []  # Store covariance for each prediction step
    
    # For each output timestep, compute the conditional distribution
    for t in range(output_timesteps):
        # Current prediction time
        pred_time = time_start + input_timesteps + t
        
        # Get the most recent ar_order timesteps for conditioning
        if len(ar_context) >= ar_order:
            # Use the last ar_order timesteps
            context_for_prediction = ar_context[-ar_order:]
        else:
            # If we don't have enough history, pad with zeros or use available data
            context_for_prediction = ar_context
        
        # Compute conditional mean: μ = Σ A_i * y_{t-i}
        mean = np.zeros(dimension)
        for i, A_i in enumerate(A_matrices):
            if i < len(context_for_prediction):
                # Use y_{t-i-1} for AR(p) where i=0 corresponds to lag-1
                lag_idx = -(i + 1)
                if abs(lag_idx) <= len(context_for_prediction):
                    mean += A_i @ context_for_prediction[lag_idx]
        
        # Compute proper prediction covariance accounting for uncertainty propagation
        if t == 0:
            # First prediction: only noise uncertainty
            cov = noise_cov.copy()
        else:
            # Subsequent predictions: noise + accumulated uncertainty from previous predictions
            cov = noise_cov.copy()
            
            # Add uncertainty from previous predictions weighted by AR coefficients
            for i, A_i in enumerate(A_matrices):
                # Check if we have a prediction at the required lag
                pred_lag_idx = t - (i + 1)  # Index in our prediction sequence
                if pred_lag_idx >= 0 and pred_lag_idx < len(accumulated_covariances):
                    # Add A_i * Σ_{t-i-1} * A_i^T to account for uncertainty propagation
                    prev_cov = accumulated_covariances[pred_lag_idx]
                    cov += A_i @ prev_cov @ A_i.T # + noise_cov.copy()
        
        # Store this step's covariance for future propagation
        accumulated_covariances.append(cov.copy())
        
        distributions.append({
            'mean': mean,
            'cov': cov.copy(),
            'time_index': pred_time,
            'forecast_step': t + 1  # 1-indexed forecast horizon
        })
        
        # For next iteration: append the expected value (mean) of current prediction
        # This simulates using the expected value rather than unknown ground truth
        ar_context = np.vstack([ar_context, mean.reshape(1, -1)])
    
    return distributions


def _plot_sample_inference(
    sample_idx: int,
    axes: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    gt_distributions: List[Dict],
    contour_levels: List[float],
    kde_bandwidth: Union[str, float],
    show_intermediate_steps: bool,
    dimension: int,
    dataset: Optional[MultivariatARDataset] = None,
    random_state: Optional[int] = None
):
    """
    Plot inference for a single sample.
    """
    # predictions shape: [1, n_samples, output_timesteps, dimension]
    # targets shape: [1, output_timesteps, dimension]
    
    n_samples = predictions.shape[1]
    output_timesteps = predictions.shape[2]
    
    # Determine which dimensions to plot using intelligent selection
    # This ensures each sample gets different dimensions but remains consistent across epochs
    dim_pair = select_plot_dimensions(dimension, dataset, random_state)
    plot_dims = 2 if dim_pair[0] != dim_pair[1] else 1
    
    col_idx = 0
    
    # Plot input
    ax_input = axes[sample_idx, col_idx]
    _plot_input_context(ax_input, inputs, targets, dim_pair)
    ax_input.set_title(f'Sample {sample_idx + 1}: Input Context', fontsize=10)
    col_idx += 1
    
    # Plot outputs
    if show_intermediate_steps and output_timesteps > 1:
        # Plot each timestep separately
        for t in range(output_timesteps):
            ax = axes[sample_idx, col_idx]
            
            # Extract predictions and targets for this timestep
            pred_t = predictions[0, :, t, :]  # [n_samples, dimension]
            target_t = targets[0, t, :]  # [dimension]
            gt_dist = gt_distributions[t]
            
            _plot_timestep_predictions(
                ax, pred_t, target_t, gt_dist, contour_levels, kde_bandwidth, 
                dim_pair, f't+{t+1}'
            )
            col_idx += 1
    else:
        # Plot only final timestep
        ax = axes[sample_idx, col_idx]
        
        # Use final timestep
        final_t = output_timesteps - 1
        pred_final = predictions[0, :, final_t, :]  # [n_samples, dimension]
        target_final = targets[0, final_t, :]  # [dimension]
        gt_dist = gt_distributions[final_t]
        
        timestep_label = f't+{final_t+1}' if output_timesteps > 1 else 'Output'
        _plot_timestep_predictions(
            ax, pred_final, target_final, gt_dist, contour_levels, kde_bandwidth,
            dim_pair, timestep_label
        )


def _plot_input_context(ax, inputs, targets, dim_pair):
    """Plot input context and immediate target."""
    # inputs: [1, input_timesteps, dimension]
    # targets: [1, output_timesteps, dimension]
    
    dim1, dim2 = dim_pair
    
    # Plot input trajectory
    input_traj = inputs[0, :, dim1] if dim1 == dim2 else inputs[0, :, [dim1, dim2]]
    
    if dim1 == dim2:
        # 1D case
        times = np.arange(len(input_traj))
        ax.plot(times, input_traj, 'b-o', label='Input', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Dimension {dim1}')
    else:
        # 2D case
        ax.plot(input_traj[:, 0], input_traj[:, 1], 'b-o', label='Input', markersize=4)
        
        # Plot first target for context
        target_first = targets[0, 0, [dim1, dim2]]
        ax.plot(target_first[0], target_first[1], 'ro', label='Next Target', markersize=6)
        
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_timestep_predictions(
    ax, predictions, target, gt_distribution, contour_levels, kde_bandwidth, dim_pair, timestep_label
):
    """Plot predictions vs ground truth for a single timestep."""
    # predictions: [n_samples, dimension]
    # target: [dimension]
    
    dim1, dim2 = dim_pair
    
    if dim1 == dim2:
        # 1D plotting
        _plot_1d_predictions(ax, predictions[:, dim1], target[dim1], gt_distribution, 
                           contour_levels, kde_bandwidth, timestep_label, dim1)
    else:
        # 2D plotting
        _plot_2d_predictions(ax, predictions[:, [dim1, dim2]], target[[dim1, dim2]], 
                           gt_distribution, contour_levels, kde_bandwidth, timestep_label, 
                           dim1, dim2)


def _plot_1d_predictions(ax, predictions, target, gt_distribution, contour_levels, kde_bandwidth, timestep_label, dim_idx):
    """Plot 1D predictions with KDE and ground truth."""
    # Resolve bandwidth (auto-detect if needed)
    resolved_bandwidth = resolve_kde_bandwidth(kde_bandwidth, predictions)
    
    # Create KDE using ALL samples
    kde = KernelDensity(bandwidth=resolved_bandwidth, kernel='gaussian')
    kde.fit(predictions.reshape(-1, 1))
    
    # Create evaluation points
    x_min, x_max = predictions.min() - 2*predictions.std(), predictions.max() + 2*predictions.std()
    x_eval = np.linspace(x_min, x_max, 200)
    
    # Evaluate KDE
    log_density = kde.score_samples(x_eval.reshape(-1, 1))
    density = np.exp(log_density)
    
    # Plot predicted density
    ax.plot(x_eval, density, 'b-', linewidth=2, label='Predicted KDE')
    ax.fill_between(x_eval, density, alpha=0.3, color='blue')
    
    # Plot ground truth density if available
    if gt_distribution is not None and 'mean' in gt_distribution:
        gt_mean = gt_distribution['mean'][dim_idx]  # Use the actual dimension being plotted
        gt_var = gt_distribution['cov'][dim_idx, dim_idx]
        gt_std = np.sqrt(gt_var)
        
        gt_density = stats.norm.pdf(x_eval, gt_mean, gt_std)
        ax.plot(x_eval, gt_density, 'r--', linewidth=2, label='True Distribution')
    
    # Plot target
    ax.axvline(target.item(), color='red', linestyle='-', linewidth=2, label='True Target')
    
    # Sample points for scatter plot if too many
    if len(predictions) > 100:
        np.random.seed(42)  # For reproducibility
        scatter_indices = np.random.choice(len(predictions), size=100, replace=False)
        scatter_predictions = predictions[scatter_indices]
        scatter_label = f'Samples (showing 100/{len(predictions)})'
    else:
        scatter_predictions = predictions
        scatter_label = f'Samples (n={len(predictions)})'
    
    # Plot samples
    ax.scatter(scatter_predictions, np.zeros_like(scatter_predictions), alpha=0.5, color='blue', s=10, label=scatter_label)
    
    ax.set_title(f'Predictions at {timestep_label}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_2d_predictions(ax, predictions, target, gt_distribution, contour_levels, kde_bandwidth, 
                        timestep_label, dim1, dim2):
    """Plot 2D predictions with KDE contours and ground truth."""
    # Sample points for scatter plot if too many
    if len(predictions) > 100:
        np.random.seed(42)  # For reproducibility
        scatter_indices = np.random.choice(len(predictions), size=100, replace=False)
        scatter_predictions = predictions[scatter_indices]
        scatter_label = f'Predicted Samples (showing 100/{len(predictions)})'
    else:
        scatter_predictions = predictions
        scatter_label = f'Predicted Samples (n={len(predictions)})'
    
    # Plot predicted samples (subset for visual clarity)
    ax.scatter(scatter_predictions[:, 0], scatter_predictions[:, 1], alpha=0.6, s=20, color='blue', 
              label=scatter_label)
    
    # Plot target
    ax.plot(target[0], target[1], 'ro', markersize=8, label='True Target')
    
    # Create KDE contours for predictions using ALL samples
    try:
        _plot_kde_contours(ax, predictions, contour_levels, kde_bandwidth, 'blue', 'Predicted')
    except Exception as e:
        warnings.warn(f"Could not plot KDE contours for predictions: {e}")
    
    # Plot ground truth distribution contours
    if gt_distribution is not None and 'mean' in gt_distribution:
        try:
            _plot_ground_truth_contours(ax, gt_distribution, contour_levels, dim1, dim2, 'red', 'True')
        except Exception as e:
            warnings.warn(f"Could not plot ground truth contours: {e}")
    
    ax.set_title(f'Predictions at {timestep_label}', fontsize=10)
    ax.set_xlabel(f'Dimension {dim1}')
    ax.set_ylabel(f'Dimension {dim2}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_kde_contours(ax, samples, contour_levels, bandwidth, color, label_prefix):
    """Plot KDE contours from samples."""
    if len(samples) < 3:
        return
    
    # Resolve bandwidth (auto-detect if needed)
    resolved_bandwidth = resolve_kde_bandwidth(bandwidth, samples)
    
    # Fit KDE
    kde = KernelDensity(bandwidth=resolved_bandwidth, kernel='gaussian')
    kde.fit(samples)
    
    # Create grid for evaluation
    x_min, x_max = samples[:, 0].min() - 2*samples[:, 0].std(), samples[:, 0].max() + 2*samples[:, 0].std()
    y_min, y_max = samples[:, 1].min() - 2*samples[:, 1].std(), samples[:, 1].max() + 2*samples[:, 1].std()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Evaluate KDE
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx.shape)
    
    # Calculate contour levels
    flat_density = density.flatten()
    sorted_density = np.sort(flat_density)[::-1]
    cumsum_density = np.cumsum(sorted_density)
    cumsum_density = cumsum_density / cumsum_density[-1]
    
    contour_data = []
    for level in contour_levels:
        idx = np.argmax(cumsum_density >= level)
        contour_data.append((level, sorted_density[idx]))
    
    # Sort by contour values (density) in increasing order for matplotlib
    contour_data.sort(key=lambda x: x[1])
    contour_values = [val for level, val in contour_data]
    
    # Plot contours
    contours = ax.contour(xx, yy, density, levels=contour_values, colors=color, linestyles='--', alpha=0.8)


def _plot_ground_truth_contours(ax, gt_distribution, contour_levels, dim1, dim2, color, label_prefix):
    """Plot contours for ground truth multivariate normal distribution."""
    mean = gt_distribution['mean'][[dim1, dim2]]
    cov = gt_distribution['cov'][np.ix_([dim1, dim2], [dim1, dim2])]
    
    # Create grid
    x_std = np.sqrt(cov[0, 0])
    y_std = np.sqrt(cov[1, 1])
    
    x_min, x_max = mean[0] - 4*x_std, mean[0] + 4*x_std
    y_min, y_max = mean[1] - 4*y_std, mean[1] + 4*y_std
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.dstack((xx, yy))
    
    # Evaluate PDF
    rv = multivariate_normal(mean, cov)
    density = rv.pdf(grid_points)
    
    # Calculate quantile levels
    flat_density = density.flatten()
    sorted_density = np.sort(flat_density)[::-1]
    cumsum_density = np.cumsum(sorted_density)
    cumsum_density = cumsum_density / cumsum_density[-1]
    
    contour_data = []
    for level in contour_levels:
        idx = np.argmax(cumsum_density >= level)
        contour_data.append((level, sorted_density[idx]))
    
    # Sort by contour values (density) in increasing order for matplotlib
    contour_data.sort(key=lambda x: x[1])
    contour_values = [val for level, val in contour_data]
    
    # Plot contours
    contours = ax.contour(xx, yy, density, levels=contour_values, colors=color, alpha=0.8)


def compare_model_predictions(
    models: List[torch.nn.Module],
    model_names: List[str],
    dataset: MultivariatARDataset,
    sample_idx: int,
    n_samples: int = 100,
    figsize: Tuple[int, int] = (20, 6),
    save_path: Optional[str] = None,
    device: str = 'cpu',
    random_state: Optional[int] = None
) -> Dict:
    """
    Compare predictions from multiple models on the same sample.
    
    Args:
        models: List of trained models to compare
        model_names: Names for each model
        dataset: Dataset to get sample from
        sample_idx: Index of sample to analyze
        n_samples: Number of samples to generate
        figsize: Figure size
        save_path: Optional save path
        device: Device for inference
        random_state: Random seed for dimension selection
        
    Returns:
        Dictionary with prediction results for each model
    """
    results = {}
    
    # Get sample data
    sample_data = dataset[sample_idx]
    inputs = sample_data['input'].unsqueeze(0).to(device)
    targets = sample_data['target'].unsqueeze(0)
    
    # Denormalize for visualization
    inputs_denorm = dataset.denormalise_tensor(inputs.cpu()).to(device)
    targets_denorm = dataset.denormalise_tensor(targets.cpu())
    
    # Create figure
    n_models = len(models)
    output_timesteps = targets.shape[2]
    
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    # Generate predictions for each model
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        model.eval()
        model = model.to(device)
        
        # Get predictions
        with torch.no_grad():
            batch_size, input_timesteps_actual, input_dimension = inputs.shape
            inputs_flat = inputs.reshape(batch_size, input_dimension)
            
            if output_timesteps == 1:
                if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                    predictions = model.forward(n_samples=n_samples, batch_size=batch_size)
                else:
                    predictions = model(inputs_flat, n_samples=n_samples)
                predictions = predictions.unsqueeze(2)
            else:
                predictions = generate_autoregressive_samples(
                    model, inputs_flat, output_timesteps, inputs.shape[-1], n_samples, batch_size
                )
        
        # Denormalize predictions for visualization
        predictions_cpu = predictions.cpu()
        predictions_denorm = torch.zeros_like(predictions_cpu)
        for t in range(output_timesteps):
            pred_t = predictions_cpu[:, :, t, :]
            pred_t_denorm = dataset.denormalise_tensor(pred_t)
            predictions_denorm[:, :, t, :] = pred_t_denorm
        
        results[model_name] = {
            'predictions': predictions_denorm,
            'targets': targets_denorm,
            'inputs': inputs_denorm.cpu()
        }
        
        # Plot final predictions using intelligent dimension selection
        ax = axes[i]
        dimension = predictions_denorm.shape[-1]
        # Use sample-specific random state for consistent but varied dimension selection
        sample_random_state = (random_state + sample_idx) if random_state is not None else sample_idx
        dim_pair = select_plot_dimensions(dimension, dataset, random_state=sample_random_state)
        
        final_pred = predictions_denorm[0, :, -1, dim_pair]  # [n_samples, 2] - denormalized
        final_target = targets_denorm[0, -1, dim_pair]  # [2] - denormalized
        
        # Sample points for scatter plot if too many
        if len(final_pred) > 100:
            np.random.seed(42)  # For reproducibility
            scatter_indices = np.random.choice(len(final_pred), size=100, replace=False)
            scatter_pred = final_pred[scatter_indices]
            scatter_label = f'{model_name} Samples (showing 100/{len(final_pred)})'
        else:
            scatter_pred = final_pred
            scatter_label = f'{model_name} Samples'
        
        ax.scatter(scatter_pred[:, 0], scatter_pred[:, 1], alpha=0.6, s=20, label=scatter_label)
        ax.plot(final_target[0], final_target[1], 'ro', markersize=8, label='True Target')
        
        # Add KDE contours using ALL samples
        try:
            _plot_kde_contours(ax, final_pred.numpy(), [0.65, 0.95], "auto", 'blue', model_name)
        except:
            pass
        
        ax.set_title(f'{model_name}')
        ax.set_xlabel(f'Dimension {dim_pair[0]}')
        ax.set_ylabel(f'Dimension {dim_pair[1]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Comparison: Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results


def plot_spatial_grid_heatmaps(
    data: np.ndarray,
    grid_size: Tuple[int, int],
    timesteps_to_plot: Optional[List[int]] = None,
    title: str = "Spatial Grid Evolution",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    time_labels: Optional[List[str]] = None
) -> Dict:
    """
    Plot spatial grid data as heatmaps over time.
    
    Creates a comprehensive visualization showing spatial patterns evolving over time.
    The plot is saved to disk and not displayed interactively.
    
    Args:
        data: Array of shape (n_timesteps, dimension) where dimension = height * width
        grid_size: Tuple (height, width) defining the spatial grid dimensions
        timesteps_to_plot: List of timestep indices to plot. If None, plots all timesteps for complete rollout
        title: Title for the overall plot
        figsize: Figure size. If None, automatically determined based on grid layout
        save_path: Optional path to save the figure
        vmin, vmax: Color scale limits. If None, uses data min/max
        cmap: Colormap to use for heatmaps
        show_colorbar: Whether to show colorbar
        time_labels: Custom labels for each timestep. If None, uses "t={i}"
        
    Returns:
        Dictionary containing plot metadata and reshaped data
    """
    height, width = grid_size
    n_timesteps, dimension = data.shape
    
    # Validate grid size
    if height * width != dimension:
        raise ValueError(f"Grid size {height}x{width}={height*width} doesn't match dimension {dimension}")
    
    # Determine which timesteps to plot
    if timesteps_to_plot is None:
        timesteps_to_plot = list(range(n_timesteps))
    
    n_plots = len(timesteps_to_plot)
    
    # Determine subplot layout - optimized for many timesteps
    if n_plots <= 6:
        n_cols = n_plots
        n_rows = 1
    elif n_plots <= 12:
        n_cols = 6
        n_rows = 2
    elif n_plots <= 18:
        n_cols = 6
        n_rows = 3
    elif n_plots <= 24:
        n_cols = 6
        n_rows = 4
    else:
        # For very many timesteps, use compact layout
        n_cols = 8
        n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Auto-determine figure size if not provided
    if figsize is None:
        base_size = 2.5  # Smaller base size for many timesteps
        figsize = (n_cols * base_size, n_rows * base_size)
        # Ensure minimum readable size
        figsize = (max(figsize[0], 8), max(figsize[1], 4))
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes) if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Reshape data to spatial grids
    spatial_data = data.reshape(n_timesteps, height, width)
    
    # Determine color scale
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    
    # Plot each timestep
    heatmaps = []
    for i, t_idx in enumerate(timesteps_to_plot):
        ax = axes[i]
        
        # Create heatmap
        im = ax.imshow(
            spatial_data[t_idx], 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            origin='upper',
            aspect='equal'
        )
        heatmaps.append(im)
        
        # Set title
        if time_labels is not None:
            ax.set_title(time_labels[i], fontsize=10)
        else:
            ax.set_title(f't={t_idx}', fontsize=10)
        
        # Remove tick labels for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines for better spatial understanding
        if max(height, width) <= 10:  # Only for small grids
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    if show_colorbar:
        # Use the last heatmap for colorbar reference
        cbar = plt.colorbar(heatmaps[-1], ax=axes[:n_plots], shrink=0.8, aspect=20)
        cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Set overall title
    plt.suptitle(title, fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial heatmap saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return {
        'spatial_data': spatial_data,
        'grid_size': grid_size,
        'timesteps_plotted': timesteps_to_plot,
        'color_range': (vmin, vmax),
        'n_plots': n_plots,
        'figure_size': figsize
    }


def plot_spatial_ar_comparison(
    model: torch.nn.Module,
    dataset: MultivariatARDataset,
    sample_indices: List[int],
    n_samples: int = 100,
    n_prediction_steps: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
    cmap: str = 'viridis',
    random_state: Optional[int] = None
) -> Dict:
    """
    Compare spatial AR model predictions with ground truth using heatmaps.
    
    This function creates a comprehensive visualization showing:
    1. Input spatial patterns
    2. True spatial evolution over prediction horizon
    3. Model prediction samples (mean and individual samples)
    4. Prediction uncertainty visualization
    
    Args:
        model: Trained AR model
        dataset: MultivariatARDataset with spatial_args
        sample_indices: List of sample indices to visualize
        n_samples: Number of model samples to generate
        n_prediction_steps: Number of prediction steps to visualize
        figsize: Figure size. If None, automatically determined
        save_path: Optional path to save the figure
        device: Device to run inference on
        cmap: Colormap for heatmaps
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing prediction results and plot metadata
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    # Check if dataset has spatial arguments
    if not hasattr(dataset, 'spatial_args') or dataset.spatial_args is None:
        raise ValueError("Dataset must have spatial_args for spatial visualization")
    
    grid_size = dataset.spatial_args['grid_size']
    height, width = grid_size
    
    model.eval()
    model.to(device)
    
    n_samples_to_plot = min(len(sample_indices), 3)  # Limit to 3 samples for readability
    
    # Auto-determine figure size
    if figsize is None:
        # Each sample gets a row, with columns for: input, true evolution, pred mean, pred std
        figsize = (4 * (5 + 2), 4 * n_samples_to_plot)  # Use fixed size for layout
    
    results = {}
    
    for sample_idx, data_idx in enumerate(sample_indices[:n_samples_to_plot]):
        # Get sample data
        sample_data = dataset[data_idx]
        x_input = sample_data['input']  # Shape: (input_timesteps, dimension)
        y_true = sample_data['target']  # Shape: (output_timesteps, dimension)
        
        # Denormalize inputs and targets for visualization
        x_input_denorm = dataset.denormalise_tensor(x_input)
        y_true_denorm = dataset.denormalise_tensor(y_true)
        
        # Prepare input for model (use original normalized data)
        x_batch = x_input.unsqueeze(0).float().to(device)  # Add batch dimension
        
        # Determine actual prediction steps based on available data
        actual_prediction_steps = min(n_prediction_steps, y_true.shape[0])
        
        # Generate model predictions
        with torch.no_grad():
            prediction_samples = generate_autoregressive_samples(
                model=model,
                initial_input=x_batch.squeeze(1),  # Remove timestep dimension for single input
                output_timesteps=actual_prediction_steps,
                output_dimension=x_batch.shape[-1],
                n_samples=n_samples,
                batch_size=x_batch.shape[0],
                use_consistent_noise=False
            )  # Shape: (batch_size, n_samples, n_steps, dimension)
        
        # Remove batch dimension and move to CPU, then denormalize predictions
        prediction_samples = prediction_samples.squeeze(0).cpu()  # (n_samples, n_steps, dimension)
        
        # Denormalize predictions for visualization
        prediction_samples_denorm = torch.zeros_like(prediction_samples)
        for t in range(actual_prediction_steps):
            # Extract predictions for timestep t: [n_samples, dimension]
            pred_t = prediction_samples[:, t, :]
            # Denormalize: [n_samples, dimension]
            pred_t_denorm = dataset.denormalise_tensor(pred_t)
            prediction_samples_denorm[:, t, :] = pred_t_denorm
        
        prediction_samples = prediction_samples_denorm.numpy()  # (n_samples, n_steps, dimension)
        x_input_np = x_input_denorm.cpu().numpy()  # Use denormalized input
        y_true_np = y_true_denorm.cpu().numpy()    # Use denormalized target
        
        # Compute prediction statistics
        pred_mean = np.mean(prediction_samples, axis=0)  # (n_steps, dimension)
        pred_std = np.std(prediction_samples, axis=0)    # (n_steps, dimension)
        
        # Create figure for this sample
        fig = plt.figure(figsize=figsize)
        
        # Plot input
        if x_input_np.shape[0] == 1:  # Single input timestep
            input_grid = x_input_np[0].reshape(height, width)
            plt.subplot(1, n_prediction_steps + 3, 1)
            plt.imshow(input_grid, cmap=cmap, origin='upper')
            plt.title('Input')
            plt.xticks([])
            plt.yticks([])
        
        # Plot true evolution
        true_grids = y_true_np[:actual_prediction_steps].reshape(actual_prediction_steps, height, width)
        vmin_true = np.min(true_grids)
        vmax_true = np.max(true_grids)
        
        for t in range(actual_prediction_steps):
            plt.subplot(1, actual_prediction_steps + 3, t + 2)
            plt.imshow(true_grids[t], cmap=cmap, vmin=vmin_true, vmax=vmax_true, origin='upper')
            plt.title(f'True t+{t+1}')
            plt.xticks([])
            plt.yticks([])
        
        # Plot prediction mean
        pred_grids = pred_mean.reshape(actual_prediction_steps, height, width)
        vmin_pred = np.min(pred_grids)
        vmax_pred = np.max(pred_grids)
        
        plt.subplot(1, actual_prediction_steps + 3, actual_prediction_steps + 2)
        # Show all prediction timesteps in a single subplot (mean across time)
        mean_pred_grid = np.mean(pred_grids, axis=0)
        plt.imshow(mean_pred_grid, cmap=cmap, origin='upper')
        plt.title('Pred Mean')
        plt.xticks([])
        plt.yticks([])
        
        # Plot prediction uncertainty
        plt.subplot(1, actual_prediction_steps + 3, actual_prediction_steps + 3)
        # Show uncertainty (std across time and samples)
        std_grids = pred_std.reshape(actual_prediction_steps, height, width)
        mean_std_grid = np.mean(std_grids, axis=0)
        plt.imshow(mean_std_grid, cmap='Reds', origin='upper')
        plt.title('Pred Std')
        plt.xticks([])
        plt.yticks([])
        
        plt.suptitle(f'Spatial AR Prediction: Sample {data_idx}', fontsize=14)
        plt.tight_layout()
        
        # Save individual sample plot
        if save_path:
            sample_save_path = save_path.replace('.png', f'_sample_{data_idx}.png')
            plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
            print(f"Spatial comparison saved to: {sample_save_path}")
        
        # Close the figure to free memory
        plt.close()
        
        # Store results
        results[f'sample_{data_idx}'] = {
            'input': x_input_np,
            'true_targets': y_true_np,
            'predictions': prediction_samples,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'grid_size': grid_size
        }
    
    return results


def create_spatial_ar_visualizations(
    dataset: MultivariatARDataset,
    save_dir: str,
    n_series_to_plot: int = 3,
    n_timesteps_to_plot: Optional[int] = None,  # None means use all timesteps
    model: Optional[torch.nn.Module] = None,
    n_prediction_samples: int = 50
) -> Dict:
    """
    Create comprehensive spatial AR visualizations for training results.
    
    This function is designed to be called during/after training to generate
    spatial visualizations when spatial_args are used.
    
    Args:
        dataset: MultivariatARDataset with spatial configuration
        save_dir: Directory to save visualizations
        n_series_to_plot: Number of different time series to visualize
        n_timesteps_to_plot: Number of timesteps to show per series. If None, shows all timesteps for complete rollouts
        model: Optional trained model for prediction comparisons
        n_prediction_samples: Number of samples for prediction visualization
        
    Returns:
        Dictionary containing paths to saved visualizations and metadata
    """
    save_dir = Path(save_dir)
    spatial_plots_dir = save_dir / "spatial_visualizations"
    spatial_plots_dir.mkdir(exist_ok=True)
    
    results = {
        'plot_paths': [],
        'grid_size': None,
        'n_series_plotted': 0,
        'n_timesteps_plotted': 0
    }
    
    # Check if spatial data is available
    if not hasattr(dataset, 'spatial_args') or dataset.spatial_args is None:
        print("No spatial arguments found in dataset - skipping spatial visualizations")
        return results
    
    grid_size = dataset.spatial_args['grid_size']
    results['grid_size'] = grid_size
    
    print(f"Creating spatial AR visualizations with grid size {grid_size}")
    
    # Plot ground truth spatial evolution for several series
    for series_idx in range(min(n_series_to_plot, len(dataset))):
        # Get a sample from the dataset
        sample = dataset[series_idx * 50]  # Space out the samples
        
        # Combine input and target for full temporal sequence
        input_data = sample['input']    # (input_timesteps, dimension)
        target_data = sample['target']  # (output_timesteps, dimension)
        
        # Denormalize data for visualization
        input_data_denorm = dataset.denormalise_tensor(input_data).cpu().numpy()
        target_data_denorm = dataset.denormalise_tensor(target_data).cpu().numpy()
        
        # Create full sequence using denormalized data
        full_sequence = np.concatenate([input_data_denorm, target_data_denorm], axis=0)
        
        # Determine timesteps to plot
        if n_timesteps_to_plot is None:
            # Show all timesteps for complete rollout
            timesteps_to_show = full_sequence.shape[0]
            data_to_plot = full_sequence
        else:
            # Limit timesteps to specified number
            timesteps_to_show = min(n_timesteps_to_plot, full_sequence.shape[0])
            data_to_plot = full_sequence[:timesteps_to_show]
        
        # Create time labels
        n_input = input_data.shape[0]
        time_labels = []
        for t in range(timesteps_to_show):
            if t < n_input:
                if n_input == 1:
                    time_labels.append('Input')
                else:
                    time_labels.append(f'Input t-{n_input-1-t}')
            else:
                time_labels.append(f'Target t+{t-n_input+1}')
        
        # Create heatmap visualization
        save_path = spatial_plots_dir / f"spatial_evolution_series_{series_idx+1}.png"
        
        plot_result = plot_spatial_grid_heatmaps(
            data=data_to_plot,
            grid_size=grid_size,
            title=f"Spatial AR Evolution - Series {series_idx+1}",
            save_path=str(save_path),
            time_labels=time_labels,
            cmap='viridis'
        )
        
        results['plot_paths'].append(str(save_path))
        results['n_series_plotted'] += 1
        results['n_timesteps_plotted'] = timesteps_to_show
    
    # If model is provided, create prediction comparison plots
    if model is not None:
        print("Creating spatial prediction comparison plots...")
        
        comparison_save_path = spatial_plots_dir / "spatial_prediction_comparison.png"
        
        # Select some sample indices for comparison
        sample_indices = [10, 25, 40]  # Different samples from dataset
        
        comparison_results = plot_spatial_ar_comparison(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            n_samples=n_prediction_samples,
            save_path=str(comparison_save_path),
            device='cpu'
        )
        
        results['plot_paths'].extend([
            str(comparison_save_path.with_suffix('').with_name(f"{comparison_save_path.stem}_sample_{idx}.png"))
            for idx in sample_indices
        ])
        results['prediction_comparison'] = comparison_results
    
    print(f"Created {len(results['plot_paths'])} spatial visualization plots in {spatial_plots_dir}")
    
    return results
