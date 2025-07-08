import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.model_selection import train_test_split
import yaml
from datetime import datetime
import shutil

# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import our models
from models.mlp_crps_sampler import MLPSampler
from models.affine_normal import SimpleAffineNormal
from models.fgn_encoder_sampler import FGNEncoderSampler

# Import our data generation function
from experiments.data.generate import generate_toy_data_multidim_extended

# Import our training functions
from experiments.training.train_functions import (
    train_model, evaluate_model, prepare_data_loaders, create_training_config
)

# Import our evaluation functions
from experiments.training.evaluation import evaluate_model as eval_model, create_evaluation_config, evaluate_affine_metrics, evaluate_metrics

# Import our plotting functions
from experiments.visualisation.plotting import plot_training_history, plot_prediction_samples


def clear_directory_contents(directory_path):
    """Clear all contents of a directory if it exists"""
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def run_experiment(seed, run_id, base_config):
    """Run a single experiment with given seed and configuration"""
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create run directory and clear if exists
    run_dir = os.path.join(os.path.dirname(__file__), 'figures', f'run_{run_id:03d}_seed_{seed}')
    clear_directory_contents(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create saved models directory for this run and clear if exists
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models', f'run_{run_id:03d}_seed_{seed}')
    clear_directory_contents(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    # Extract model type
    model_type = base_config.get('model_type', 'MLPSampler')
    
    print(f"\n=== Running experiment {run_id} with seed {seed} ({model_type} - Energy Score Loss) ===")
    
    # Extract configuration
    x_dim = base_config.get('x_dim', 1)
    y_dim = base_config['y_dim']
    data_size = base_config['data_size']
    n_epochs = base_config['n_epochs']
    patience = base_config['patience']
    train_n_samples = base_config['train_n_samples']
    n_samples = base_config['n_samples']
    batch_size = base_config['batch_size']
    
    # Extract noise parameters from base_config
    noise_type = base_config.get('noise_type', 'gaussian')
    noise_scale = base_config.get('noise_scale', 0.1)
    target_correlation = base_config.get('target_correlation', 0.8)
    mean_function = base_config.get('mean_function', 'zero')
    
    # Prepare noise-specific parameters
    noise_kwargs = {}
    if noise_type == 'student_t':
        noise_kwargs['df'] = base_config.get('student_t_df', 3)
    elif noise_type == 'laplace_asymmetric':
        location = base_config.get('laplace_location', None)
        scale = base_config.get('laplace_scale', None)
        if location is not None or scale is not None:
            noise_kwargs['asymmetry_params'] = {}
            if location is not None:
                noise_kwargs['asymmetry_params']['location'] = np.array(location) if isinstance(location, list) else location
            if scale is not None:
                noise_kwargs['asymmetry_params']['scale'] = np.array(scale) if isinstance(scale, list) else scale
    elif noise_type == 'gamma':
        shape_params = base_config.get('gamma_shape_params', None)
        if shape_params is not None:
            noise_kwargs['shape_params'] = np.array(shape_params) if isinstance(shape_params, list) else shape_params
    elif noise_type == 'lognormal':
        noise_kwargs['sigma'] = base_config.get('lognormal_sigma', 1.0)
    
    print(f"Using noise type: {noise_type} with parameters: {noise_kwargs}")
    
    # Generate toy data using the extended function
    data_dict = generate_toy_data_multidim_extended(
        n_samples=data_size, 
        x_dim=x_dim if model_type in ['MLPSampler', 'FGNEncoderSampler'] else 1,  # x_dim doesn't matter for SimpleAffineNormal
        y_dim=y_dim, 
        noise_type=noise_type,
        noise_scale=noise_scale,
        mean_function=mean_function,
        target_correlation=target_correlation,
        **noise_kwargs
    )

    x_train_tensor = data_dict['x_train_tensor']
    y_train_tensor = data_dict['y_train_tensor']
    x_test_tensor = data_dict['x_test_tensor']
    y_test_tensor = data_dict['y_test_tensor']
    noise_args = data_dict['noise_args']

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_tensor, y_train_tensor, test_size=0.2, random_state=42
    )

    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

    # Create model based on type
    if model_type == 'SimpleAffineNormal':
        model = SimpleAffineNormal(output_dim=y_dim)
        print(f"Model parameters: A matrix shape {model.A.shape}, b vector shape {model.b.shape}")
        save_path = os.path.join(models_dir, 'affine_normal.pt')
        learning_rate = base_config.get('learning_rate', 0.01)
    elif model_type == 'FGNEncoderSampler':
        hidden_size = base_config.get('hidden_size', 3)
        latent_dim = base_config.get('latent_dim', 3)
        n_layers = base_config.get('n_layers', 3)
        dropout_rate = base_config.get('dropout_rate', 0.0)
        zero_inputs = base_config.get('zero_inputs', True)
        non_linear = base_config.get('non_linear', False)
        model = FGNEncoderSampler(
            input_size=x_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            output_size=y_dim,
            zero_inputs=zero_inputs,
            non_linear=non_linear
        )
        print(model)
        save_path = os.path.join(models_dir, 'fgn_encoder_sampler.pt')
        learning_rate = base_config.get('learning_rate', 0.001)
    else:  # MLPSampler
        hidden_size = base_config.get('hidden_size', 3)
        latent_dim = base_config.get('latent_dim', 3)
        n_layers = base_config.get('n_layers', 3)
        dropout_rate = base_config.get('dropout_rate', 0.0)
        sample_layer_index = base_config.get('sample_layer_index', 1)
        zero_inputs = base_config.get('zero_inputs', True)
        non_linear = base_config.get('non_linear', False)
        activation_function = base_config.get('activation_function', 'relu')
        
        model = MLPSampler(
            input_size=x_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            output_size=y_dim,
            sample_layer_index=sample_layer_index,
            zero_inputs=zero_inputs,
            non_linear=non_linear,
            activation_function=activation_function
        )
        print(model)
        save_path = os.path.join(models_dir, 'mlp_sampler.pt')
        learning_rate = base_config.get('learning_rate', 0.001)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare training configuration using helper function
    training_config = create_training_config(
        model_type=model_type,
        loss_type='energy_score',
        n_epochs=n_epochs,
        patience=patience,
        learning_rate=learning_rate,
        n_samples=n_samples,
        train_n_samples=train_n_samples,
        device=device,
        save_path=save_path,
        track_weights=base_config.get('track_weights', False),
        track_samples_every=base_config.get('track_samples_every', 5),
        track_moments=base_config.get('track_moments', False),
        moments_to_track=base_config.get('moments_to_track', [1, 2, 3, 4]),
        noise_samples=data_dict.get('noise_samples', None),
        x_test=x_test_tensor,
        y_test=y_test_tensor,
        noise_args=noise_args
    )

    # Train the model with unified training function
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config
    )

    # Plot training history
    history_plot = plot_training_history(training_results['history'])
    history_plot_path = os.path.join(run_dir, 'training_history.png')
    history_plot.savefig(history_plot_path)
    plt.close(history_plot)

    # Evaluate model on test set using energy score loss
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=batch_size
    )
    
    # Create evaluation config for loss evaluation with higher sample count
    loss_eval_config = create_evaluation_config(
        model_type=model_type,
        evaluation_type='loss',
        loss_type='energy_score',
        n_samples=n_samples,
        device=device
    )
    
    test_loss = eval_model(model, test_loader, loss_eval_config)
    print(f"Test Energy Score Loss: {test_loss:.4f}")
    
    if model_type == 'SimpleAffineNormal':
        # Calculate additional metrics using the new unified approach (optional)
        # metrics_eval_config = create_evaluation_config(
        #     model_type='SimpleAffineNormal',
        #     evaluation_type='metrics',
        #     n_samples=n_samples,
        #     device=device
        # )
        # metrics = eval_model(model, test_loader, metrics_eval_config)
        
        # Or use legacy function for backward compatibility
        metrics = evaluate_affine_metrics(model, y_test_tensor, n_samples=n_samples, device=device)
        
        # Plot prediction samples
        samples_plot = plot_prediction_samples(
            None, y_test_tensor, model, 
            n_samples=n_samples, n_points=1, device=device,
            noise_args=noise_args
        )
        samples_plot_path = os.path.join(run_dir, 'prediction_samples.png')
        samples_plot.savefig(samples_plot_path)
        plt.close(samples_plot)
        
        # Get learned parameters
        mean, cov = model.get_mean_and_covariance()
        learned_params = {
            'mean': mean.detach().cpu().tolist(),
            'covariance': cov.detach().cpu().tolist(),
            'A_matrix': model.A.detach().cpu().tolist(),
            'b_vector': model.b.detach().cpu().tolist()
        }
        print(f"Learned mean: {mean.detach().cpu().numpy()}")
        print(f"Learned covariance:\n{cov.detach().cpu().numpy()}")
        
    else:  # MLPSampler or FGNEncoderSampler
        # Calculate additional metrics
        metrics = evaluate_metrics(model, x_test_tensor, y_test_tensor, n_samples=n_samples, device=device)
        
        # Plot prediction samples with true confidence intervals
        samples_plot = plot_prediction_samples(
            x_test_tensor, y_test_tensor, model, 
            n_samples=n_samples, n_points=1, device=device,
            noise_args=noise_args
        )
        samples_plot_path = os.path.join(run_dir, 'prediction_samples.png')
        samples_plot.savefig(samples_plot_path)
        plt.close(samples_plot)

        learned_params = {}

    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test CRPS: {metrics['crps']:.4f}")
    print(f"Test Log Likelihood: {metrics['log_likelihood']:.4f}")
    print("Calibration:")
    for alpha, value in metrics['calibration'].items():
        print(f"  {alpha} interval: {value:.4f}")
    
    # Save results to config
    run_config = base_config.copy()
    run_config.update({
        'seed': seed,
        'run_id': run_id,
        'run_timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_type': model_type,
        'loss_function': 'energy_score',
        'learned_parameters': learned_params,
        'results': {
            'test_energy_score_loss': float(test_loss),
            'test_mse': float(metrics['mse']),
            'test_crps': float(metrics['crps']),
            'test_log_likelihood': float(metrics['log_likelihood']),
            'calibration': {k: float(v) for k, v in metrics['calibration'].items()},
            'training_epochs': len(training_results['history']['train_loss'])
        }
    })
    
    # Save config file
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {run_dir}")
    return run_config

def main():
    """Main function to run multiple experiments"""
    
    # Define base configuration
    base_config = {
        # Data parameters
        'x_dim': 1,
        'y_dim': 2,
        'data_size': 2000,
        
        # Noise parameters - NEW SECTION
        'noise_type': 'gaussian',  # Options: 'gaussian', 'student_t', 'laplace_symmetric', 'laplace_asymmetric', 'gamma', 'lognormal'
        'noise_scale': 0.3,
        'target_correlation': 0.6,
        'mean_function': 'zero',
        
        # Noise-specific parameters (only used if corresponding noise_type is selected)
        'student_t_df': 3,  # Degrees of freedom for Student-t
        'laplace_location': [0.0, 0.0],  # Location parameters for asymmetric Laplace (list for multivariate)
        'laplace_scale': [1.0, 1.0],     # Scale parameters for asymmetric Laplace (list for multivariate)
        'gamma_shape_params': [2.0, 1.5],  # Shape parameters for gamma distribution (list for multivariate)
        'lognormal_sigma': 0.5,  # Standard deviation in log space for lognormal
        
        # Model selection - Change this to switch between models
        'model_type': 'FGNEncoderSampler',  # Options: 'MLPSampler', 'FGNEncoderSampler', or 'SimpleAffineNormal'
        
        # Model parameters for MLPSampler (ignored if using SimpleAffineNormal)
        'hidden_size': [16,32,64],
        'latent_dim': 2,
        'n_layers': 3,
        'sample_layer_index': 1, # Not used for FGNEncoderSampler
        'dropout_rate': 0.0,
        'zero_inputs': True,
        'non_linear': True,
        'activation_function': 'relu',  # Options: 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'gelu', 'elu', 'softplus'
        
        # Training parameters
        'n_epochs': 100,
        'patience': 10,
        'train_n_samples': 10,
        'n_samples': 2000,
        'batch_size': 64,
        'learning_rate': 0.0005,  # Only used for SimpleAffineNormal
        
        # Weight tracking parameters
        'track_weights': True,  # Enable weight evolution tracking
        'track_samples_every': 5,  # Create prediction plots every N epochs
        'track_moments': True,  # NEW: Enable moments tracking during training
        'moments_to_track': [1, 2, 3, 4],  # NEW: Which moments to track (mean, variance, skewness, kurtosis)
        
        # Loss function
        'loss_function': 'energy_score'
    }
    
    # Define seeds for multiple runs
    seeds = [1, 42, 123, 456, 789]
    
    # Store results from all runs
    all_results = []
    
    # Run experiments
    for i, seed in enumerate(seeds):
        result = run_experiment(seed, i+1, base_config)
        all_results.append(result)
    
    # Save summary of all runs
    summary_dir = os.path.join(os.path.dirname(__file__), 'figures')
    summary_config = {
        'experiment_summary': {
            'total_runs': len(seeds),
            'seeds_used': seeds,
            'base_config': base_config,
            'summary_timestamp': datetime.now().isoformat(),
            'model_type': base_config['model_type'],
            'loss_function': 'energy_score'
        },
        'all_results': all_results
    }
    
    summary_path = os.path.join(summary_dir, 'experiment_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary_config, f, default_flow_style=False, indent=2)
    
    print(f"\n=== Experiment Summary ({base_config['model_type']} - Energy Score Loss) ===")
    print(f"Completed {len(seeds)} runs with seeds: {seeds}")
    print(f"Summary saved to: {summary_path}")
    
    # Print quick statistics
    test_energy_score_values = [result['results']['test_energy_score_loss'] for result in all_results]
    test_mse_values = [result['results']['test_mse'] for result in all_results]
    test_crps_values = [result['results']['test_crps'] for result in all_results]
    test_log_likelihood_values = [result['results']['test_log_likelihood'] for result in all_results]
    
    print(f"\nTest Energy Score Loss - Mean: {np.mean(test_energy_score_values):.4f}, Std: {np.std(test_energy_score_values):.4f}")
    print(f"Test MSE - Mean: {np.mean(test_mse_values):.4f}, Std: {np.std(test_mse_values):.4f}")
    print(f"Test CRPS - Mean: {np.mean(test_crps_values):.4f}, Std: {np.std(test_crps_values):.4f}")
    print(f"Test Log Likelihood - Mean: {np.mean(test_log_likelihood_values):.4f}, Std: {np.std(test_log_likelihood_values):.4f}")

if __name__ == "__main__":
    main() 