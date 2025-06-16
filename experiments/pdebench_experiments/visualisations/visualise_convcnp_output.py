"""
Visualisation script for the ConvCNP Sampler model.

This script loads a trained ConvCNPSampler model and a test sample from the
CFD2DDataset, runs the model to get probabilistic predictions, and visualises
the ground truth, mean prediction, and prediction uncertainty (standard deviation)
for each predicted timestep.

Usage:
    python visualise_convcnp_output.py --config path/to/config.yml --sample_idx <int> --output_dir <path>

Example:
    python experiments/pdebench_experiments/visualisations/visualise_convcnp_output.py --config experiments/pdebench_experiments/training/configs/convcnp_config.yml --sample_idx 0 --output_dir experiments/pdebench_experiments/visualisations/figures
"""

import argparse
import yaml
import torch
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Adjusting python path to allow imports from parent directories
# This assumes the script is in experiments/pdebench_experiments/visualisations
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from models.convcnp_sampler import ConvCNPSampler
from experiments.pdebench_experiments.dataset.dataset import CFD2DDataset


def infer_grid_size_from_coords(coords: np.ndarray, min_grid_size: int = 16, max_grid_size: int = 256):
    """
    Infer appropriate grid size from coordinate spacing.
    
    Args:
        coords: Coordinate array, shape (N, 2) with x, y coordinates
        min_grid_size: Minimum allowed grid size
        max_grid_size: Maximum allowed grid size
        
    Returns:
        Inferred grid size based on coordinate spacing
    """
    if len(coords) < 2:
        return min_grid_size
    
    # Calculate extent of coordinates
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range == 0 or y_range == 0:
        return min_grid_size
    
    # Find minimum spacing in x and y directions
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    
    # Calculate minimum differences (grid spacing)
    if len(unique_x) > 1:
        min_dx = np.min(np.diff(unique_x))
    else:
        min_dx = x_range
        
    if len(unique_y) > 1:
        min_dy = np.min(np.diff(unique_y))
    else:
        min_dy = y_range
    
    # Estimate grid size based on range/spacing
    # Use the smaller of the two dimensions to ensure square pixels
    min_spacing = min(min_dx, min_dy)
    if min_spacing > 0:
        # Calculate how many grid points we need for each dimension
        grid_size_x = int(np.ceil(x_range / min_spacing)) + 1
        grid_size_y = int(np.ceil(y_range / min_spacing)) + 1
        
        # Use the maximum to ensure we capture the full resolution
        inferred_size = max(grid_size_x, grid_size_y)
        
        # Apply constraints
        inferred_size = max(min_grid_size, min(inferred_size, max_grid_size))
    else:
        inferred_size = min_grid_size
    
    return inferred_size


def create_grid_from_coords(coords: np.ndarray, values: np.ndarray, grid_size: int = None):
    """
    Convert coordinate-based data to a regular grid for visualisation.
    
    Args:
        coords: Coordinate array, shape (N, 2) with x, y coordinates
        values: Values at coordinates, shape (N,)
        grid_size: Size of the output grid. If None, will be inferred from coordinate spacing
        
    Returns:
        2D grid of interpolated values, shape (grid_size, grid_size)
    """
    # Infer grid size if not provided
    if grid_size is None:
        grid_size = infer_grid_size_from_coords(coords)
    
    # Define regular grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Add small padding to avoid edge issues
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.05
    
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate values to grid
    grid_values = griddata(
        coords, values, (xi_grid, yi_grid), 
        method='cubic', fill_value=np.nan
    )
    
    return grid_values, (x_min, x_max, y_min, y_max)


def visualise_predictions(
    ground_truth: torch.Tensor,
    prediction_mean: torch.Tensor,
    prediction_std: torch.Tensor,
    target_coords: torch.Tensor,
    input_data: torch.Tensor,
    input_coords: torch.Tensor,
    input_variables: list,
    output_path: Path,
    target_variable: str
):
    """
    Visualises model predictions against ground truth, including input variables.

    Args:
        ground_truth: Ground truth data, shape (T_out, N_out).
        prediction_mean: Mean of model predictions, shape (N_out, T_out).
        prediction_std: Standard deviation of predictions, shape (N_out, T_out).
        target_coords: Coordinates of target points, shape (N_out, 2).
        input_data: Input data, shape (T_lag, N_in, n_input_vars).
        input_coords: Coordinates of input points, shape (N_in, 2).
        input_variables: List of input variable names.
        output_path: Path to save the visualisation figure.
        target_variable: Name of the target variable for plot titles.
    """
    time_predict = ground_truth.shape[0]
    time_lag = input_data.shape[0]
    n_input_vars = len(input_variables)
    
    # Convert to numpy for plotting
    ground_truth = ground_truth.cpu().numpy()
    prediction_mean = prediction_mean.cpu().numpy()
    prediction_std = prediction_std.cpu().numpy()
    target_coords = target_coords.cpu().numpy()
    input_data = input_data.cpu().numpy()
    input_coords = input_coords.cpu().numpy()
    
    # Calculate total number of columns: input variables + 3 prediction plots
    n_cols = n_input_vars + 3
    
    fig, axes = plt.subplots(time_predict, n_cols, figsize=(6 * n_cols, 6 * time_predict), squeeze=False)
    fig.suptitle(f"ConvCNP: Inputs and Predictions for '{target_variable}'", fontsize=16)

    for t in range(time_predict):
        col_idx = 0
        
        # Plot input variables (using the most recent timestep as reference)
        for var_idx, var_name in enumerate(input_variables):
            # Use the most recent input timestep (time_lag - 1)
            input_values = input_data[time_lag - 1, :, var_idx]
            
            # Convert to grid
            grid_data, extent = create_grid_from_coords(input_coords, input_values)
            
            ax = axes[t, col_idx]
            im = ax.imshow(grid_data, extent=extent, origin='lower', cmap='viridis', aspect='equal')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Input: {var_name}")
            ax.set_xlabel("x-coordinate")
            ax.set_ylabel("y-coordinate")
            col_idx += 1
        
        # Plot target predictions
        gt = ground_truth[t, :]
        mean_pred = prediction_mean[:, t]
        std_pred = prediction_std[:, t]
        
        # Create grids for target data
        gt_grid, target_extent = create_grid_from_coords(target_coords, gt)
        mean_grid, _ = create_grid_from_coords(target_coords, mean_pred)
        std_grid, _ = create_grid_from_coords(target_coords, std_pred)
        
        # Calculate vmin/vmax for consistent scaling
        vmin = min(np.nanmin(gt_grid), np.nanmin(mean_grid))
        vmax = max(np.nanmax(gt_grid), np.nanmax(mean_grid))
        
        # Plot Ground Truth
        ax = axes[t, col_idx]
        im = ax.imshow(gt_grid, extent=target_extent, origin='lower', cmap='viridis', 
                      vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Ground Truth (t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        col_idx += 1
        
        # Plot Mean Prediction
        ax = axes[t, col_idx]
        im = ax.imshow(mean_grid, extent=target_extent, origin='lower', cmap='viridis', 
                      vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Mean Prediction (t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        col_idx += 1
        
        # Plot Uncertainty (Std Dev)
        ax = axes[t, col_idx]
        im = ax.imshow(std_grid, extent=target_extent, origin='lower', cmap='plasma', aspect='equal')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Prediction Uncertainty (Std Dev, t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualisation saved to {output_path}")
    plt.close()


def main(args):
    """
    Main visualisation function.
    """
    config_path = Path(args.config)
    if not config_path.is_file():
        # Try resolving from project root
        config_path = project_root / args.config
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found at {args.config}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    data_config = config['data']
    data_path = project_root / data_config['path']
    
    dataset = CFD2DDataset(
        data_path=data_path,
        config=data_config['config'],
        split='test',
        normalise=True  # Ensure normalisation is on to use denormalise method
    )
    
    if args.sample_idx >= len(dataset):
        raise ValueError(f"sample_idx {args.sample_idx} is out of bounds for test set with size {len(dataset)}")

    # Get a specific sample
    sample = dataset[args.sample_idx]
    
    # Create a batch of size 1
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()}

    # Initialize model
    model_config = config['model']
    model_config['n_input_vars'] = len(data_config['config']['input_variables'])
    model_config['time_lag'] = data_config['config']['time_lag']
    model_config['time_predict'] = data_config['config']['time_predict']
    
    model = ConvCNPSampler(**model_config).to(device)
    
    # Load trained model
    save_dir = project_root / config['logging']['save_dir']
    checkpoint_path = save_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # Prepare data for model
    input_data = batch['input_data'].to(device)
    input_coords = batch['input_coords'].to(device)
    target_data = batch['target_data'].to(device)
    target_coords = batch['target_coords'].to(device)

    # Get model predictions
    with torch.no_grad():
        predictions = model(
            input_data,
            input_coords,
            target_coords,
            n_samples=config['training'].get('n_crps_samples', 50)
        ) # (B, n_samples, N_out, T_out)

    # Process predictions
    # Squeeze batch dimension as B=1
    predictions = predictions.squeeze(0) # (n_samples, N_out, T_out)
    target_data = target_data.squeeze(0) # (T_out, N_out)
    target_coords = target_coords.squeeze(0) # (N_out, 2)
    input_data_squeezed = input_data.squeeze(0) # (T_lag, N_in, n_input_vars)
    input_coords_squeezed = input_coords.squeeze(0) # (N_in, 2)

    # Denormalise data
    target_variable = data_config['config']['target_variable']
    input_variables = data_config['config']['input_variables']
    
    # Denormalise requires tensors
    denorm_predictions = dataset.denormalise_prediction(predictions, target_variable)
    denorm_target_data = dataset.denormalise_prediction(target_data, target_variable)
    
    # Denormalise input data for each variable
    denorm_input_data = torch.zeros_like(input_data_squeezed)
    for i, var_name in enumerate(input_variables):
        for t in range(input_data_squeezed.shape[0]):
            denorm_input_data[t, :, i] = dataset.denormalise_prediction(
                input_data_squeezed[t, :, i], var_name
            )
    
    # Compute mean and std
    pred_mean = denorm_predictions.mean(dim=0) # (N_out, T_out)
    pred_std = denorm_predictions.std(dim=0)   # (N_out, T_out)

    # Visualise
    output_dir = project_root / args.output_dir
    output_path = output_dir / f"convcnp_prediction_sample_{args.sample_idx}.png"
    
    visualise_predictions(
        ground_truth=denorm_target_data,
        prediction_mean=pred_mean,
        prediction_std=pred_std,
        target_coords=target_coords,
        input_data=denorm_input_data,
        input_coords=input_coords_squeezed,
        input_variables=input_variables,
        output_path=output_path,
        target_variable=target_variable
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise ConvCNP Sampler model predictions.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file used for training.')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample from the test set to visualise.')
    parser.add_argument('--output_dir', type=str, default='experiments/pdebench_experiments/visualisations/figures', help='Directory to save the output figure.')
    
    args = parser.parse_args()
    main(args) 