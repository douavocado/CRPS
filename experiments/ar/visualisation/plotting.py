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

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.ar.dataset.dataset import MultivariatARDataset
from experiments.ar.training.train_functions import generate_autoregressive_samples


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
    kde_bandwidth: float = 0.1,
    random_state: Optional[int] = None
) -> Dict:
    """
    Plot AR model inference process with KDE contours and ground truth distributions.
    
    This function visualizes:
    1. Model predictions vs ground truth for each timestep
    2. KDE contours at specified quantile levels
    3. Intermediate steps for autoregressive generation
    4. True underlying distributions from ground truth AR parameters
    
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
        
        # Store inputs and targets
        results['inputs'].append(inputs.cpu())
        results['targets'].append(targets)
        
        # Flatten inputs for single timestep case
        batch_size, input_timesteps_actual, input_dimension = inputs.shape
        inputs_flat = inputs.reshape(batch_size, input_dimension)
        
        # Generate predictions
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
        
        # Move predictions to CPU for plotting
        predictions = predictions.cpu()  # Shape: [1, n_samples, output_timesteps, dimension]
        results['predictions'].append(predictions)
        
        # Get ground truth distribution parameters for this sample
        gt_distributions = _get_ground_truth_distributions(
            dataset, dataset_idx, output_timesteps, ground_truth_params
        )
        results['ground_truth_distributions'].append(gt_distributions)
        
        # Plot for this sample
        _plot_sample_inference(
            sample_idx, axes, inputs.cpu(), targets, predictions, gt_distributions,
            contour_levels, kde_bandwidth, show_intermediate_steps, dimension
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
    
    plt.show()
    
    return results


def _get_ground_truth_distributions(
    dataset: MultivariatARDataset,
    dataset_idx: int,
    output_timesteps: int,
    ground_truth_params: Dict
) -> List[Dict]:
    """
    Get ground truth distribution parameters for each output timestep.
    
    Returns:
        List of dicts containing 'mean', 'cov' for each timestep
    """
    # Get the sample data to determine the time context
    sample_data = dataset[dataset_idx]
    series_idx = sample_data['series_idx'].item()
    time_start = sample_data['time_start'].item()
    
    # Get the full series for context
    full_series = dataset.get_full_series(series_idx, normalised=True)
    
    # Extract AR parameters from ground truth
    generation_metadata = ground_truth_params['generation_metadata']
    A_matrices = generation_metadata['A_matrices']  # List of AR coefficient matrices
    noise_cov = generation_metadata['noise_cov']
    dimension = dataset.config['dimension']
    ar_order = dataset.config['ar_order']
    
    # Convert to numpy if they're tensors
    if torch.is_tensor(A_matrices):
        A_matrices = [A_matrices.numpy()]
    elif isinstance(A_matrices, list) and torch.is_tensor(A_matrices[0]):
        A_matrices = [A.numpy() for A in A_matrices]
    
    if torch.is_tensor(noise_cov):
        noise_cov = noise_cov.numpy()
    
    distributions = []
    
    # For each output timestep, compute the conditional distribution
    for t in range(output_timesteps):
        # Time index for this prediction
        pred_time = time_start + dataset.config['input_timesteps'] + t
        
        # Get the AR context (previous ar_order timesteps)
        if pred_time >= ar_order:
            # We have enough history
            context_start = pred_time - ar_order
            context = full_series[context_start:pred_time]  # Shape: [ar_order, dimension]
            
            # Compute conditional mean: μ = Σ A_i * y_{t-i}
            mean = np.zeros(dimension)
            for i, A_i in enumerate(A_matrices):
                if i < len(context):
                    mean += A_i @ context[-(i+1)]  # A_i @ y_{t-i-1}
            
            # Covariance is the noise covariance (for AR processes)
            cov = noise_cov.copy()
        else:
            # Not enough history - use unconditional distribution (approximation)
            mean = np.zeros(dimension)
            cov = noise_cov.copy()
        
        distributions.append({
            'mean': mean,
            'cov': cov,
            'time_index': pred_time
        })
    
    return distributions


def _plot_sample_inference(
    sample_idx: int,
    axes: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    gt_distributions: List[Dict],
    contour_levels: List[float],
    kde_bandwidth: float,
    show_intermediate_steps: bool,
    dimension: int
):
    """
    Plot inference for a single sample.
    """
    # predictions shape: [1, n_samples, output_timesteps, dimension]
    # targets shape: [1, output_timesteps, dimension]
    
    n_samples = predictions.shape[1]
    output_timesteps = predictions.shape[2]
    
    # Determine which dimensions to plot (use first 2 dimensions for 2D plots)
    plot_dims = min(2, dimension)
    dim_pairs = [(0, 1)] if plot_dims == 2 else [(0, 0)]  # For 1D, plot against itself
    
    col_idx = 0
    
    # Plot input
    ax_input = axes[sample_idx, col_idx]
    _plot_input_context(ax_input, inputs, targets, dim_pairs[0])
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
                dim_pairs[0], f't+{t+1}'
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
            dim_pairs[0], timestep_label
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
                           contour_levels, kde_bandwidth, timestep_label)
    else:
        # 2D plotting
        _plot_2d_predictions(ax, predictions[:, [dim1, dim2]], target[[dim1, dim2]], 
                           gt_distribution, contour_levels, kde_bandwidth, timestep_label, 
                           dim1, dim2)


def _plot_1d_predictions(ax, predictions, target, gt_distribution, contour_levels, kde_bandwidth, timestep_label):
    """Plot 1D predictions with KDE and ground truth."""
    # Create KDE
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
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
        gt_mean = gt_distribution['mean'][0]  # Assuming we're plotting first dimension
        gt_var = gt_distribution['cov'][0, 0]
        gt_std = np.sqrt(gt_var)
        
        gt_density = stats.norm.pdf(x_eval, gt_mean, gt_std)
        ax.plot(x_eval, gt_density, 'r--', linewidth=2, label='True Distribution')
    
    # Plot target
    ax.axvline(target.item(), color='red', linestyle='-', linewidth=2, label='True Target')
    
    # Plot samples
    ax.scatter(predictions, np.zeros_like(predictions), alpha=0.5, color='blue', s=10, label=f'Samples (n={len(predictions)})')
    
    ax.set_title(f'Predictions at {timestep_label}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_2d_predictions(ax, predictions, target, gt_distribution, contour_levels, kde_bandwidth, 
                        timestep_label, dim1, dim2):
    """Plot 2D predictions with KDE contours and ground truth."""
    # Plot predicted samples
    ax.scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, s=20, color='blue', 
              label=f'Predicted Samples (n={len(predictions)})')
    
    # Plot target
    ax.plot(target[0], target[1], 'ro', markersize=8, label='True Target')
    
    # Create KDE contours for predictions
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
    
    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
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
    
    # Add labels using the correctly matched level and contour_val
    if len(contour_data) > 0:
        # Only label the highest confidence level (lowest density value after sorting)
        level, contour_val = contour_data[0]  # First item after sorting by density
        ax.clabel(contours, levels=[contour_val], inline=True, fontsize=8, 
                 fmt=f'{label_prefix} {int(level*100)}%')


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
    
    # Add labels using the correctly matched level and contour_val
    if len(contour_data) > 0:
        # Only label the highest confidence level (lowest density value after sorting)
        level, contour_val = contour_data[0]  # First item after sorting by density
        ax.clabel(contours, levels=[contour_val], inline=True, fontsize=8, 
                 fmt=f'{label_prefix} {int(level*100)}%')


def compare_model_predictions(
    models: List[torch.nn.Module],
    model_names: List[str],
    dataset: MultivariatARDataset,
    sample_idx: int,
    n_samples: int = 100,
    figsize: Tuple[int, int] = (20, 6),
    save_path: Optional[str] = None,
    device: str = 'cpu'
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
        
    Returns:
        Dictionary with prediction results for each model
    """
    results = {}
    
    # Get sample data
    sample_data = dataset[sample_idx]
    inputs = sample_data['input'].unsqueeze(0).to(device)
    targets = sample_data['target'].unsqueeze(0)
    
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
        
        results[model_name] = {
            'predictions': predictions.cpu(),
            'targets': targets,
            'inputs': inputs.cpu()
        }
        
        # Plot final predictions (2D visualization)
        ax = axes[i]
        final_pred = predictions[0, :, -1, :2].cpu()  # [n_samples, 2]
        final_target = targets[0, -1, :2]  # [2]
        
        ax.scatter(final_pred[:, 0], final_pred[:, 1], alpha=0.6, s=20, label=f'{model_name} Samples')
        ax.plot(final_target[0], final_target[1], 'ro', markersize=8, label='True Target')
        
        # Add KDE contours
        try:
            _plot_kde_contours(ax, final_pred.numpy(), [0.65, 0.95], 0.1, 'blue', model_name)
        except:
            pass
        
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Comparison: Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results
