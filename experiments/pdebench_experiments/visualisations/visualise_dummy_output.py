"""
Visualisation script for the Dummy model.

This script loads a dummy baseline model and a test sample from the
CFD2DDataset, runs the model to get predictions, and visualises
the ground truth, mean prediction, and prediction uncertainty (standard deviation)
for each predicted timestep.

Usage:
    python visualise_dummy_output.py --config path/to/config.yml --sample_idx <int> --output_dir <path>

Example:
    python experiments/pdebench_experiments/visualisations/visualise_dummy_output.py --config experiments/pdebench_experiments/training/configs/dummy_config.yml --sample_idx 0 --output_dir experiments/pdebench_experiments/visualisations/figures
"""

import argparse
import yaml
import torch
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjusting python path to allow imports from parent directories
# This assumes the script is in experiments/pdebench_experiments/visualisations
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from models.dummy_model import DummyModel
from experiments.pdebench_experiments.dataset.dataset import CFD2DDataset


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
    fig.suptitle(f"Dummy Model: Inputs and Predictions for '{target_variable}'", fontsize=16)

    for t in range(time_predict):
        col_idx = 0
        
        # Plot input variables (using the most recent timestep as reference)
        input_x = input_coords[:, 0]
        input_y = input_coords[:, 1]
        
        for var_idx, var_name in enumerate(input_variables):
            # Use the most recent input timestep (time_lag - 1)
            input_values = input_data[time_lag - 1, :, var_idx]
            
            ax = axes[t, col_idx]
            sc = ax.scatter(input_x, input_y, c=input_values, cmap='viridis')
            plt.colorbar(sc, ax=ax)
            ax.set_title(f"Input: {var_name}")
            ax.set_xlabel("x-coordinate")
            ax.set_ylabel("y-coordinate")
            ax.set_aspect('equal', adjustable='box')
            col_idx += 1
        
        # Plot target predictions
        x = target_coords[:, 0]
        y = target_coords[:, 1]
        
        gt = ground_truth[t, :]
        mean_pred = prediction_mean[:, t]
        std_pred = prediction_std[:, t]
        
        vmin = min(gt.min(), mean_pred.min())
        vmax = max(gt.max(), mean_pred.max())
        
        # Plot Ground Truth
        ax = axes[t, col_idx]
        sc = ax.scatter(x, y, c=gt, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Ground Truth (t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_aspect('equal', adjustable='box')
        col_idx += 1
        
        # Plot Mean Prediction
        ax = axes[t, col_idx]
        sc = ax.scatter(x, y, c=mean_pred, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Mean Prediction (t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_aspect('equal', adjustable='box')
        col_idx += 1
        
        # Plot Uncertainty (Std Dev)
        # Note: For dummy model, all samples are identical, so std will be 0
        ax = axes[t, col_idx]
        sc = ax.scatter(x, y, c=std_pred, cmap='plasma')
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Prediction Uncertainty (Std Dev, t={t+1})")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_aspect('equal', adjustable='box')
        
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

    # Initialize dummy model
    model_config = config.get('model', {})
    input_variables = data_config['config']['input_variables']
    target_variable = data_config['config']['target_variable']
    time_predict = data_config['config']['time_predict']
    
    model = DummyModel(
        input_variables=input_variables,
        target_variable=target_variable,
        time_predict=time_predict,
        **model_config
    ).to(device)
    
    print(f"Dummy model initialised")
    print(f"  Input variables: {input_variables}")
    print(f"  Target variable: {target_variable}")
    print(f"  Time predict: {time_predict}")

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
            n_samples=config.get('training', {}).get('n_crps_samples', 50)
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

    # Note: For dummy model, std will be 0 since all samples are identical
    print(f"Prediction std range: [{pred_std.min().item():.6f}, {pred_std.max().item():.6f}]")
    if pred_std.max().item() < 1e-6:
        print("Note: Dummy model produces deterministic predictions (std ≈ 0)")

    # Visualise
    output_dir = project_root / args.output_dir
    output_path = output_dir / f"dummy_prediction_sample_{args.sample_idx}.png"
    
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
    parser = argparse.ArgumentParser(description="Visualise Dummy model predictions.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample from the test set to visualise.')
    parser.add_argument('--output_dir', type=str, default='experiments/pdebench_experiments/visualisations/figures', help='Directory to save the output figure.')
    
    args = parser.parse_args()
    main(args) 