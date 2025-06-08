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

# Import our data generation function
from experiments.data.generate import generate_toy_data_multidim

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
    
    print(f"\n=== Running experiment {run_id} with seed {seed} ({model_type}) ===")
    
    # Extract configuration
    x_dim = base_config.get('x_dim', 1)
    y_dim = base_config['y_dim']
    data_size = base_config['data_size']
    n_epochs = base_config['n_epochs']
    patience = base_config['patience']
    n_samples = base_config['n_samples']
    batch_size = base_config['batch_size']
    
    # Generate toy data
    data_dict = generate_toy_data_multidim(
        n_samples=data_size, 
        x_dim=x_dim if model_type == 'MLPSampler' else 1,  # x_dim doesn't matter for SimpleAffineNormal
        y_dim=y_dim, 
        dependent_noise=True, 
        mean_function='zero', 
        target_correlation=0.9
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
        
    else:  # MLPSampler
        hidden_size = base_config.get('hidden_size', 64)
        latent_dim = base_config.get('latent_dim', 10)
        n_layers = base_config.get('n_layers', 3)
        dropout_rate = base_config.get('dropout_rate', 0.1)
        sample_layer_index = base_config.get('sample_layer_index', 1)
        zero_inputs = base_config.get('zero_inputs', True)
        non_linear = base_config.get('non_linear', False)
        
        model = MLPSampler(
            input_size=x_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            output_size=y_dim,
            sample_layer_index=sample_layer_index,
            zero_inputs=zero_inputs,
            non_linear=non_linear
        )
        save_path = os.path.join(models_dir, 'mlp_sampler.pt')
    loss_type = base_config.get('loss_type', 'crps')
    learning_rate = base_config.get('learning_rate', 0.001)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare training configuration using helper function
    train_n_samples = base_config.get('train_n_samples', 10)
    training_config = create_training_config(
        model_type=model_type,
        loss_type=loss_type,
        n_epochs=n_epochs,
        patience=patience,
        learning_rate=learning_rate,
        n_samples=train_n_samples,
        device=device,
        save_path=save_path,
        track_weights=base_config.get('track_weights', False),
        track_samples_every=base_config.get('track_samples_every', 5),
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

    # Evaluate model on test set
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=batch_size
    )
    
    # Create evaluation config for loss evaluation with higher sample count
    loss_eval_config = create_evaluation_config(
        model_type=model_type,
        evaluation_type='loss',
        loss_type=loss_type,
        n_samples=n_samples,
        device=device
    )
    
    test_loss = eval_model(model, test_loader, loss_eval_config)
    
    if model_type == 'SimpleAffineNormal':
        print(f"Test Negative Log Likelihood: {test_loss:.4f}")
        
        # Calculate additional metrics
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
        
    else:  # MLPSampler
        print(f"Test CRPS Loss: {test_loss:.4f}")
        
        # Calculate additional metrics using the new unified approach (optional)
        # metrics_eval_config = create_evaluation_config(
        #     model_type='MLPSampler',
        #     evaluation_type='metrics',
        #     n_samples=n_samples,
        #     device=device
        # )
        # metrics = eval_model(model, test_loader, metrics_eval_config)
        
        # Or use legacy function for backward compatibility
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
    result_dict = {
        'test_mse': float(metrics['mse']),
        'test_crps': float(metrics['crps']),
        'test_log_likelihood': float(metrics['log_likelihood']),
        'calibration': {k: float(v) for k, v in metrics['calibration'].items()},
        'training_epochs': len(training_results['history']['train_loss'])
    }
    
    if model_type == 'SimpleAffineNormal':
        result_dict['test_neg_log_likelihood'] = float(test_loss)
    else:
        result_dict['test_crps_loss'] = float(test_loss)
    
    run_config.update({
        'seed': seed,
        'run_id': run_id,
        'run_timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_type': model_type,
        'learned_parameters': learned_params,
        'results': result_dict
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
        'y_dim': 3,
        'data_size': 2000,
        
        # Model selection - Change this to switch between models
        'model_type': 'SimpleAffineNormal',  # Options: 'MLPSampler' or 'SimpleAffineNormal'
        
        # Model parameters for MLPSampler (ignored if using SimpleAffineNormal)
        'hidden_size': 64,
        'latent_dim': 10,
        'n_layers': 3,
        'dropout_rate': 0.1,
        'sample_layer_index': 1,
        'zero_inputs': True,
        'non_linear': False,
        
        # Training parameters
        'n_epochs': 100,
        'patience': 10,
        'train_n_samples': 10,
        'n_samples': 1000,
        'batch_size': 64,
        'learning_rate': 0.01,  # Only used for SimpleAffineNormal
        
        # Weight tracking parameters
        'track_weights': True,  # Enable weight evolution tracking
        'track_samples_every': 5,  # Create prediction plots every N epochs

        # Loss function
        'loss_function': 'crps',
        
        # Data generation parameters
        'dependent_noise': True,
        'mean_function': 'zero',
        'target_correlation': 0.9,
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
            'model_type': base_config['model_type']
        },
        'all_results': all_results
    }
    
    summary_path = os.path.join(summary_dir, 'experiment_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary_config, f, default_flow_style=False, indent=2)
    
    print(f"\n=== Experiment Summary ({base_config['model_type']}) ===")
    print(f"Completed {len(seeds)} runs with seeds: {seeds}")
    print(f"Summary saved to: {summary_path}")
    
    # Print quick statistics
    test_mse_values = [result['results']['test_mse'] for result in all_results]
    test_crps_values = [result['results']['test_crps'] for result in all_results]
    test_log_likelihood_values = [result['results']['test_log_likelihood'] for result in all_results]
    
    print(f"Test MSE - Mean: {np.mean(test_mse_values):.4f}, Std: {np.std(test_mse_values):.4f}")
    print(f"Test CRPS - Mean: {np.mean(test_crps_values):.4f}, Std: {np.std(test_crps_values):.4f}")
    print(f"Test Log Likelihood - Mean: {np.mean(test_log_likelihood_values):.4f}, Std: {np.std(test_log_likelihood_values):.4f}")

if __name__ == "__main__":
    main()








