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

# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our model
from models.mlp_crps_sampler import MLPSampler

# Import our data generation function
from experiments.data.generate import generate_toy_data_multidim

# Import our training functions
from experiments.training.train_functions import (
    train_model, evaluate_model, prepare_data_loaders, evaluate_metrics
)

from experiments.visualisation.plotting import plot_training_history, plot_prediction_samples

def run_experiment(seed, run_id, base_config):
    """Run a single experiment with given seed and configuration"""
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create run directory
    run_dir = os.path.join(os.path.dirname(__file__), 'figures', f'run_{run_id:03d}_seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create saved models directory for this run
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models', f'run_{run_id:03d}_seed_{seed}')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n=== Running experiment {run_id} with seed {seed} ===")
    
    # Extract configuration
    x_dim = base_config['x_dim']
    y_dim = base_config['y_dim']
    data_size = base_config['data_size']
    hidden_size = base_config['hidden_size']
    latent_dim = base_config['latent_dim']
    n_layers = base_config['n_layers']
    dropout_rate = base_config['dropout_rate']
    sample_layer_index = base_config['sample_layer_index']
    zero_inputs = base_config['zero_inputs']
    n_epochs = base_config['n_epochs']
    patience = base_config['patience']
    train_n_samples = base_config['train_n_samples']
    n_samples = base_config['n_samples']
    batch_size = base_config['batch_size']
    
    # Generate toy data
    data_dict = generate_toy_data_multidim(
        n_samples=data_size, 
        x_dim=x_dim, 
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

    # Create model
    model = MLPSampler(
        input_size=x_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        output_size=y_dim,
        sample_layer_index=sample_layer_index,
        zero_inputs=zero_inputs
    )

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join(models_dir, 'mlp_sampler.pt')

    # Train the model
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        patience=patience,
        n_samples=train_n_samples,
        device=device,
        save_path=save_path
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
    test_loss = evaluate_model(model, test_loader, device=device, n_samples=n_samples)
    print(f"Test CRPS Loss: {test_loss:.4f}")

    # Calculate additional metrics
    metrics = evaluate_metrics(model, x_test_tensor, y_test_tensor, n_samples=n_samples, device=device)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test CRPS: {metrics['crps']:.4f}")
    print(f"Test Log Likelihood: {metrics['log_likelihood']:.4f}")
    print("Calibration:")
    for alpha, value in metrics['calibration'].items():
        print(f"  {alpha} interval: {value:.4f}")

    # Plot prediction samples with true confidence intervals
    samples_plot = plot_prediction_samples(
        x_test_tensor, y_test_tensor, model, 
        n_samples=n_samples, n_points=5, device=device,
        noise_args=noise_args
    )
    samples_plot_path = os.path.join(run_dir, 'prediction_samples.png')
    samples_plot.savefig(samples_plot_path)
    plt.close(samples_plot)

    # Also create a plot with more test points to better visualize the distribution
    samples_plot_more = plot_prediction_samples(
        x_test_tensor, y_test_tensor, model, 
        n_samples=n_samples, n_points=10, device=device,
        noise_args=noise_args
    )
    samples_plot_more_path = os.path.join(run_dir, 'prediction_samples_more.png')
    samples_plot_more.savefig(samples_plot_more_path)
    plt.close(samples_plot_more)
    
    # Save results to config
    run_config = base_config.copy()
    run_config.update({
        'seed': seed,
        'run_id': run_id,
        'run_timestamp': datetime.now().isoformat(),
        'device': str(device),
        'results': {
            'test_crps_loss': float(test_loss),
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
        'y_dim': 3,
        'data_size': 5000,
        
        # Model parameters
        'hidden_size': 64,
        'latent_dim': 3,
        'n_layers': 3,
        'dropout_rate': 0.0,
        'sample_layer_index': 1,
        'zero_inputs': True, # if true, x_dim is ignored and we use a trainable vector instead
        
        # Training parameters
        'n_epochs': 100,
        'patience': 5,
        'train_n_samples': 10,
        'n_samples': 1000,
        'batch_size': 64,
        
        # Data generation parameters
        'dependent_noise': True,
        'mean_function': 'zero',
        'target_correlation': 0.9
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
            'summary_timestamp': datetime.now().isoformat()
        },
        'all_results': all_results
    }
    
    summary_path = os.path.join(summary_dir, 'experiment_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary_config, f, default_flow_style=False, indent=2)
    
    print(f"\n=== Experiment Summary ===")
    print(f"Completed {len(seeds)} runs with seeds: {seeds}")
    print(f"Summary saved to: {summary_path}")
    
    # Print quick statistics
    test_mse_values = [result['results']['test_mse'] for result in all_results]
    test_crps_values = [result['results']['test_crps'] for result in all_results]
    test_log_likelihood_values = [result['results']['test_log_likelihood'] for result in all_results]
    
    print(f"\nTest MSE - Mean: {np.mean(test_mse_values):.4f}, Std: {np.std(test_mse_values):.4f}")
    print(f"Test CRPS - Mean: {np.mean(test_crps_values):.4f}, Std: {np.std(test_crps_values):.4f}")
    print(f"Test Log Likelihood - Mean: {np.mean(test_log_likelihood_values):.4f}, Std: {np.std(test_log_likelihood_values):.4f}")

if __name__ == "__main__":
    main()








