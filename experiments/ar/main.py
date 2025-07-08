#!/usr/bin/env python3
"""
Main training script for AR time series models.

This script provides a comprehensive framework for training probabilistic models
on multivariate autoregressive time series data with configurable:
- Data generation parameters
- Model architectures
- Loss functions 
- Training hyperparameters

Usage:
    python main.py --config config.yaml
    python main.py --config-dict '{"data": {...}, "model": {...}, ...}'
"""

import argparse
import os
import sys
import yaml
import json
import torch
import shutil
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.ar.dataset.dataset import create_ar_dataloader, get_example_config
from experiments.ar.training.train_functions import train_ar_model, evaluate_ar_model
from experiments.data.generate_time_series import generate_multivariate_ar
from models.fgn_encoder_sampler import FGNEncoderSampler
from models.mlp_crps_sampler import MLPSampler
from models.affine_normal import SimpleAffineNormal
from common.losses import create_loss_function


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the training script.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            # Data generation parameters
            'n_timesteps': 200,
            'dimension': 3,
            'ar_order': 2,
            'noise_scale': 0.1,
            'n_series': 500,
            
            # Dataset parameters (NOTE: Input must be single timestep, output can be multi-timestep via autoregression)
            'input_timesteps': 1,   # l - MUST be 1 for current models
            'output_timesteps': 5,  # k - Can be > 1 (autoregressive generation)
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            
            # Optional custom parameters (set to None to use auto-generated)
            'A_matrices': None,
            'noise_cov': None,
            'random_state': 42
        },
        
        'model': {
            'type': 'fgn_encoder',  # Options: 'fgn_encoder', 'mlp_sampler', 'affine_normal'
            
            # FGN Encoder parameters
            'fgn_encoder': {
                'input_size': None,  # Will be set automatically from data
                'hidden_size': 64,
                'latent_dim': 16,
                'n_layers': 3,
                'dropout_rate': 0.1,
                'output_size': None,  # Will be set automatically from data
                'zero_inputs': False,
                'non_linear': True,
                'activation_function': 'relu'
            },
            
            # MLP Sampler parameters
            'mlp_sampler': {
                'input_size': None,  # Will be set automatically from data
                'hidden_size': [64, 32],  # Two hidden layers for n_layers=3 (input->hidden->hidden->output)
                'latent_dim': 16,
                'n_layers': 3,
                'dropout_rate': 0.1,
                'output_size': None,  # Will be set automatically from data
                'sample_layer_index': 2,
                'zero_inputs': False,
                'non_linear': True,
                'activation_function': 'relu'
            },
            
            # Affine Normal parameters
            'affine_normal': {
                'output_dim': None  # Will be set automatically from data
            }
        },
        
        'loss': {
            'losses': ['crps_loss_general'],  # Can include multiple losses
            'loss_function_args': {
                'crps_loss_general': {},
                'energy_score_loss': {'norm_dim': False},
                'variogram_score_loss': {'p': 1.0},
                'kernel_score_loss': {'kernel_type': 'gaussian', 'sigma': 1.0},
                'mmd_loss': {'kernel_type': 'gaussian', 'sigma': 1.0}
            },
            'coefficients': {
                'crps_loss_general': 1.0,
                'energy_score_loss': 1.0,
                'variogram_score_loss': 1.0,
                'kernel_score_loss': 1.0,
                'mmd_loss': 1.0
            }
        },
        
        'training': {
            'n_epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_samples': 50,  # Number of samples for probabilistic models
            'patience': 15,  # Early stopping patience
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'num_workers': 4,
            'shuffle_train': True,
            'normalise_data': True,
            'verbose': True,
            
            # Optimizer parameters
            'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop'
            'optimizer_args': {
                'weight_decay': 0.0,
                'betas': [0.9, 0.999],
                'eps': 1e-8
            }
        },
        
        'experiment': {
            'name': 'ar_experiment',
            'save_dir': 'results',
            'save_model': True,
            'save_results': True,
            'log_interval': 10,  # Log every N epochs
            'evaluate_test': True
        },
        
        'tracking': {
            'enabled': False,  # Default disabled
            'track_every': 10,
            'sample_indices': [0, 1, 2],
            'n_samples': 100,
            'kde_bandwidth': 0.1,
            'contour_levels': [0.65, 0.95, 0.99],
            'max_checkpoints': 10,
            'generate_plots_after_training': True,
            'create_animation': False
        }
    }


def create_model(model_config: Dict, input_size: int, output_size: int) -> torch.nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        input_size: Input size for the model (flattened input timesteps * dimension)
        output_size: Output size for the model (flattened output timesteps * dimension)
        
    Returns:
        PyTorch model
    """
    model_type = model_config['type']
    
    if model_type == 'fgn_encoder':
        config = model_config.get('fgn_encoder', {}).copy() if model_config.get('fgn_encoder') is not None else {}
        config['input_size'] = input_size
        config['output_size'] = output_size
        return FGNEncoderSampler(**config)
    
    elif model_type == 'mlp_sampler':
        config = model_config.get('mlp_sampler', {}).copy() if model_config.get('mlp_sampler') is not None else {}
        config['input_size'] = input_size
        config['output_size'] = output_size
        return MLPSampler(**config)
    
    elif model_type == 'affine_normal':
        config = model_config.get('affine_normal', {}).copy() if model_config.get('affine_normal') is not None else {}
        config['output_dim'] = output_size
        return SimpleAffineNormal(**config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def setup_optimizer(model: torch.nn.Module, training_config: Dict) -> torch.optim.Optimizer:
    """
    Setup optimizer based on configuration.
    
    Args:
        model: PyTorch model
        training_config: Training configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    optimizer_type = training_config['optimizer'].lower()
    learning_rate = training_config['learning_rate']
    optimizer_args = training_config.get('optimizer_args', {})
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **optimizer_args)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, **optimizer_args)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, **optimizer_args)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def setup_device(device_config: str) -> str:
    """
    Setup compute device based on configuration.
    
    Args:
        device_config: Device configuration ('auto', 'cuda', 'cpu')
        
    Returns:
        Device string
    """
    if device_config == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return device_config


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        Dictionary containing training results and metadata
    """
    # Setup experiment directory
    exp_config = config['experiment']
    save_dir = Path(exp_config['save_dir']) / exp_config['name']
    
    # Remove existing experiment directory if it exists
    if save_dir.exists():
        print(f"Removing existing experiment directory: {save_dir}")
        shutil.rmtree(save_dir)
    
    # Create fresh experiment directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"=== AR Time Series Training Experiment ===")
    print(f"Experiment: {exp_config['name']}")
    print(f"Save directory: {save_dir}")
    
    # Setup device
    device = setup_device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\n=== Creating Data Loaders ===")
    data_config = config['data']
    training_config = config['training']
    
    train_loader = create_ar_dataloader(
        config=data_config,
        split='train',
        batch_size=training_config['batch_size'],
        shuffle=training_config['shuffle_train'],
        num_workers=training_config['num_workers'],
        normalise=training_config['normalise_data'],
        random_state=data_config.get('random_state')
    )
    
    val_loader = create_ar_dataloader(
        config=data_config,
        split='val',
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        normalise=training_config['normalise_data'],
        random_state=data_config.get('random_state')
    )
    
    # Calculate input and output sizes
    input_size = data_config['dimension']  # Single timestep input, so just the dimension
    output_size = data_config['dimension']  # Single timestep per step (autoregressive), so just the dimension
    
    print(f"Input size: {input_size} (timesteps: {data_config['input_timesteps']}, dim: {data_config['dimension']})")
    print(f"Output size: {output_size} (timesteps: {data_config['output_timesteps']}, dim: {data_config['dimension']})")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(config['model'], input_size, output_size)
    print(f"Model type: {config['model']['type']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    print("\n=== Creating Loss Function ===")
    loss_fn = create_loss_function(config['loss'])
    print(f"Loss functions: {config['loss']['losses']}")
    print(f"Loss coefficients: {config['loss']['coefficients']}")
    
    # Train model
    print("\n=== Training Model ===")
    training_results = train_ar_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        n_epochs=training_config['n_epochs'],
        learning_rate=training_config['learning_rate'],
        n_samples=training_config['n_samples'],
        patience=training_config['patience'],
        device=device,
        verbose=training_config['verbose'],
        tracking_config=config.get('tracking', {}),
        save_dir=str(save_dir),
        model_config=config['model']
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    
    # Evaluate on test set if requested
    test_results = None
    if exp_config['evaluate_test']:
        print("\n=== Evaluating on Test Set ===")
        test_loader = create_ar_dataloader(
            config=data_config,
            split='test',
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers'],
            normalise=training_config['normalise_data'],
            random_state=data_config.get('random_state')
        )
        
        test_loss = evaluate_ar_model(
            model=training_results['model'],
            data_loader=test_loader,
            loss_fn=loss_fn,
            n_samples=training_config['n_samples'],
            device=device
        )
        
        test_results = {'test_loss': test_loss}
        print(f"Test loss: {test_loss:.6f}")
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Save results
    if exp_config['save_model']:
        model_path = save_dir / 'best_model.pth'
        torch.save(training_results['model'].state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    if exp_config['save_results']:
        results = {
            'training_history': training_results['history'],
            'best_val_loss': training_results['best_val_loss'],
            'test_results': test_results,
            'config': config,
            'model_info': {
                'type': config['model']['type'],
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            },
            'tracking': training_results.get('tracking', {})
        }
        
        results_path = save_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")
        
        # Print tracking summary if available
        tracking_info = training_results.get('tracking', {})
        if tracking_info.get('enabled', False):
            print(f"\n=== Tracking Summary ===")
            print(f"Tracked epochs: {tracking_info.get('tracked_epochs', [])}")
            print(f"Total checkpoints: {tracking_info.get('total_checkpoints', 0)}")
            print(f"Checkpoints directory: {tracking_info.get('checkpoints_dir', 'N/A')}")
            if tracking_info.get('plot_paths'):
                print(f"Generated plots: {len(tracking_info['plot_paths'])}")
                print(f"Plots directory: {tracking_info.get('plots_dir', 'N/A')}")
    
    return {
        'training_results': training_results,
        'test_results': test_results,
        'save_dir': save_dir,
        'config': config
    }


def load_config(config_path: Optional[str] = None, config_dict: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or command line.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: JSON string containing configuration
        
    Returns:
        Configuration dictionary
    """
    if config_path and config_dict:
        raise ValueError("Cannot specify both config_path and config_dict")
    
    # Start with default configuration
    config = get_default_config()
    
    if config_path:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge user config with defaults
        config = deep_merge_dicts(config, user_config)
    
    elif config_dict:
        print("Loading configuration from command line dictionary")
        user_config = json.loads(config_dict)
        config = deep_merge_dicts(config, user_config)
    
    else:
        print("Using default configuration")
    
    return config


def deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Dictionary to merge into base
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train AR time series models with configurable architectures and loss functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default configuration
    python main.py
    
    # Load configuration from file
    python main.py --config config.yaml
    
    # Override configuration via command line
    python main.py --config-dict '{"model": {"type": "mlp_sampler"}, "training": {"n_epochs": 200}}'
    
    # Create example configuration file
    python main.py --save-example-config example_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--config-dict', 
        type=str,
        help='JSON string containing configuration overrides'
    )
    
    parser.add_argument(
        '--save-example-config',
        type=str,
        help='Save example configuration to specified file and exit'
    )
    
    args = parser.parse_args()
    
    # Handle example config creation
    if args.save_example_config:
        config = get_default_config()
        with open(args.save_example_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Example configuration saved to: {args.save_example_config}")
        return
    
    try:
        # Load configuration
        config = load_config(args.config, args.config_dict)
        
        # Run training
        results = train_model(config)
        
        print(f"\n=== Experiment Complete ===")
        print(f"Results saved to: {results['save_dir']}")
        
        return results
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
