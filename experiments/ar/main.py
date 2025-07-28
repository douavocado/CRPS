#!/usr/bin/env python3
"""
Main training script for AR time series models.

This script provides a comprehensive framework for training probabilistic models
on multivariate autoregressive time series data with configurable:
- Data generation parameters
- Model architectures
- Loss functions 
- Training hyperparameters
- Multi-seed experiments

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
from typing import Dict, Any, Optional, List
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
from common.losses import create_loss_function, create_multi_instance_loss_function


def create_separate_loss_functions(loss_config: Dict[str, Any]) -> tuple:
    """
    Create separate training, validation, and testing loss functions based on configuration.
    
    Args:
        loss_config: Loss configuration dictionary
        
    Returns:
        Tuple of (training_loss_fn, validation_loss_fn, testing_loss_fn)
    """
    training_loss_config = loss_config.get('training_loss')
    validation_loss_config = loss_config.get('validation_loss')
    testing_loss_config = loss_config.get('testing_loss')
    
    # Check for required loss configurations
    if training_loss_config is None:
        raise ValueError("training_loss must be defined in the loss configuration")
    
    if testing_loss_config is None:
        raise ValueError("testing_loss must be defined in the loss configuration")
    
    # Create training and testing loss functions
    training_loss_fn = create_loss_function_from_config(training_loss_config)
    testing_loss_fn = create_loss_function_from_config(testing_loss_config)
    
    # Create validation loss function (defaults to training loss if not specified)
    if validation_loss_config is not None:
        validation_loss_fn = create_loss_function_from_config(validation_loss_config)
    else:
        validation_loss_fn = training_loss_fn
        print("validation_loss not specified, using training_loss for validation")
    
    return training_loss_fn, validation_loss_fn, testing_loss_fn


def create_loss_function_from_config(config: Dict[str, Any]):
    """
    Create a loss function from a configuration, supporting both old and new formats.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss function
    """
    # Support new multi-instance format
    if 'loss_instances' in config:
        return create_multi_instance_loss_function(config)
    # Support old format
    else:
        return create_loss_function(config)


def get_loss_info(loss_config: Dict[str, Any]) -> str:
    """
    Get a human-readable description of a loss configuration.
    
    Args:
        loss_config: Loss configuration dictionary
        
    Returns:
        String description of the loss configuration
    """
    if 'loss_instances' in loss_config:
        instances = loss_config['loss_instances']
        descriptions = []
        for instance in instances:
            name = instance.get('name', 'unnamed')
            loss_fn = instance.get('loss_function', 'unknown')
            coeff = instance.get('coefficient', 1.0)
            descriptions.append(f"{name}({loss_fn}, coeff={coeff})")
        return f"Multi-instance: {', '.join(descriptions)}"
    else:
        losses = loss_config.get('losses', [])
        coefficients = loss_config.get('coefficients', {})
        descriptions = []
        for loss in losses:
            coeff = coefficients.get(loss, 1.0)
            descriptions.append(f"{loss}(coeff={coeff})")
        return f"Combined: {', '.join(descriptions)}"


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
            # New format: Separate training, validation, and testing losses (required)
            'training_loss': {
                'loss_instances': [
                    {
                        'name': 'crps_training',
                        'loss_function': 'crps_loss_general',
                        'coefficient': 1.0,
                        'spatial_pooling': {'enabled': False},
                        'loss_args': {}
                    }
                ]
            },
            'validation_loss': None,  # If None, defaults to training_loss
            'testing_loss': {
                'loss_instances': [
                    {
                        'name': 'crps_testing',
                        'loss_function': 'crps_loss_general',
                        'coefficient': 1.0,
                        'spatial_pooling': {'enabled': False},
                        'loss_args': {}
                    }
                ]
            },
            
            # Legacy format (deprecated - use for backward compatibility only)
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
            'evaluate_test': True,
            # Multi-seed experiment configuration
            'seeds': [42],  # List of seeds for multi-seed experiments
            'parallel_seeds': False  # If True, run seeds in parallel (experimental)
        },
        
        'tracking': {
            'enabled': False,  # Default disabled
            'track_every': 10,
            'sample_indices': [0, 1, 2],
            'n_samples': 100,
            'kde_bandwidth': "auto",  # Default to automatic bandwidth detection
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


def run_multi_seed_experiments(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run experiments with multiple seeds, saving results in seed-specific subfolders.
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        Dictionary containing results from all seed experiments
    """
    exp_config = config['experiment']
    seeds = exp_config.get('seeds', [42])
    
    # If only one seed, run single experiment
    if len(seeds) == 1:
        config_copy = config.copy()
        config_copy['data']['random_state'] = seeds[0]
        return train_single_seed_model(config_copy, seed=seeds[0])
    
    # Multi-seed experiment setup
    base_save_dir = Path(exp_config['save_dir']) / exp_config['name']
    
    # Remove existing experiment directory if it exists
    if base_save_dir.exists():
        print(f"Removing existing experiment directory: {base_save_dir}")
        shutil.rmtree(base_save_dir)
    
    # Create base experiment directory
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base configuration
    with open(base_save_dir / 'base_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"=== Multi-Seed AR Time Series Training Experiment ===")
    print(f"Experiment: {exp_config['name']}")
    print(f"Seeds: {seeds}")
    print(f"Base directory: {base_save_dir}")
    
    # Run experiments for each seed
    all_results = {}
    aggregated_results = {
        'best_val_losses': [],
        'test_losses': [],
        'training_histories': {},
        'seed_results': {}
    }
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(seeds)} with seed {seed}")
        print(f"{'='*60}")
        
        # Create seed-specific configuration
        config_copy = config.copy()
        config_copy['data']['random_state'] = seed
        
        # Update experiment config for this seed
        config_copy['experiment']['name'] = f"{exp_config['name']}/seed_{seed}"
        
        # Run training for this seed
        try:
            seed_results = train_single_seed_model(config_copy, seed=seed, parent_dir=base_save_dir)
            all_results[f'seed_{seed}'] = seed_results
            
            # Aggregate results
            aggregated_results['best_val_losses'].append(seed_results['training_results']['best_val_loss'])
            aggregated_results['seed_results'][f'seed_{seed}'] = seed_results
            
            if seed_results['test_results'] and 'test_loss' in seed_results['test_results']:
                aggregated_results['test_losses'].append(seed_results['test_results']['test_loss'])
            
            # Store training history
            aggregated_results['training_histories'][f'seed_{seed}'] = seed_results['training_results']['history']
            
        except Exception as e:
            print(f"Error in seed {seed} experiment: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'seed_{seed}'] = {'error': str(e)}
    
    # Compute aggregated statistics
    if aggregated_results['best_val_losses']:
        val_losses = np.array(aggregated_results['best_val_losses'])
        aggregated_results['val_loss_stats'] = {
            'mean': float(np.mean(val_losses)),
            'std': float(np.std(val_losses)),
            'min': float(np.min(val_losses)),
            'max': float(np.max(val_losses)),
            'median': float(np.median(val_losses))
        }
        
        print(f"\n=== Validation Loss Statistics Across Seeds ===")
        print(f"Mean: {aggregated_results['val_loss_stats']['mean']:.6f} ± {aggregated_results['val_loss_stats']['std']:.6f}")
        print(f"Min: {aggregated_results['val_loss_stats']['min']:.6f}")
        print(f"Max: {aggregated_results['val_loss_stats']['max']:.6f}")
        print(f"Median: {aggregated_results['val_loss_stats']['median']:.6f}")
    
    if aggregated_results['test_losses']:
        test_losses = np.array(aggregated_results['test_losses'])
        aggregated_results['test_loss_stats'] = {
            'mean': float(np.mean(test_losses)),
            'std': float(np.std(test_losses)),
            'min': float(np.min(test_losses)),
            'max': float(np.max(test_losses)),
            'median': float(np.median(test_losses))
        }
        
        print(f"\n=== Test Loss Statistics Across Seeds ===")
        print(f"Mean: {aggregated_results['test_loss_stats']['mean']:.6f} ± {aggregated_results['test_loss_stats']['std']:.6f}")
        print(f"Min: {aggregated_results['test_loss_stats']['min']:.6f}")
        print(f"Max: {aggregated_results['test_loss_stats']['max']:.6f}")
        print(f"Median: {aggregated_results['test_loss_stats']['median']:.6f}")
    
    # Save aggregated results
    aggregated_results['config'] = config
    aggregated_results['seeds'] = seeds
    aggregated_results['n_seeds'] = len(seeds)
    
    results_path = base_save_dir / 'aggregated_results.json'
    with open(results_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2, default=str)
    print(f"\nAggregated results saved to: {results_path}")
    
    return {
        'all_results': all_results,
        'aggregated_results': aggregated_results,
        'base_save_dir': base_save_dir,
        'config': config
    }


def train_single_seed_model(config: Dict[str, Any], seed: int, parent_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train a model for a single seed.
    
    Args:
        config: Complete configuration dictionary
        seed: Random seed for this experiment
        parent_dir: Parent directory for saving results (optional)
        
    Returns:
        Dictionary containing training results and metadata
    """
    # Setup experiment directory
    exp_config = config['experiment']
    
    if parent_dir is not None:
        # Use seed-specific subdirectory under parent
        save_dir = parent_dir / f"seed_{seed}"
    else:
        # Use standard directory structure
        save_dir = Path(exp_config['save_dir']) / exp_config['name']
        # Remove existing experiment directory if it exists
        if save_dir.exists():
            print(f"Removing existing experiment directory: {save_dir}")
            shutil.rmtree(save_dir)
    
    # Create fresh experiment directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration with seed information
    config_to_save = config.copy()
    config_to_save['experiment']['actual_seed'] = seed
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False)
    
    print(f"=== AR Time Series Training (Seed {seed}) ===")
    print(f"Experiment: {exp_config['name']}")
    print(f"Save directory: {save_dir}")
    print(f"Random seed: {seed}")
    
    # Set PyTorch random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
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
    
    # Create loss functions
    print("\n=== Creating Loss Functions ===")
    training_loss_fn, validation_loss_fn, testing_loss_fn = create_separate_loss_functions(config['loss'])
    
    # Print loss information
    print("Using separate training, validation, and testing losses:")
    print(f"Training loss: {get_loss_info(config['loss']['training_loss'])}")
    
    # Handle validation loss (may be None, defaulting to training loss)
    if config['loss']['validation_loss'] is not None:
        print(f"Validation loss: {get_loss_info(config['loss']['validation_loss'])}")
    else:
        print("Validation loss: Same as training loss (validation_loss not specified)")
    
    print(f"Testing loss: {get_loss_info(config['loss']['testing_loss'])}")
    
    # Train model
    print("\n=== Training Model ===")
    training_results = train_ar_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        testing_loss_fn=testing_loss_fn,
        n_epochs=training_config['n_epochs'],
        learning_rate=training_config['learning_rate'],
        n_samples=training_config['n_samples'],
        patience=training_config['patience'],
        device=device,
        verbose=training_config['verbose'],
        tracking_config=config.get('tracking', {}),
        save_dir=str(save_dir),
        model_config=config['model'],
        training_config=training_config
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
        
        # Check if we should show test breakdown
        show_test_breakdown = hasattr(testing_loss_fn, '_is_multi_instance') and testing_loss_fn._is_multi_instance
        
        if show_test_breakdown:
            test_loss, test_breakdown = evaluate_ar_model(
                model=training_results['model'],
                data_loader=test_loader,
                loss_fn=testing_loss_fn,
                n_samples=training_config['n_samples'],
                device=device,
                return_breakdown=True
            )
            
            test_results = {
                'test_loss': test_loss,
                'test_breakdown': test_breakdown
            }
            print(f"Test loss: {test_loss:.6f}")
            print("Test breakdown:")
            for component, value in test_breakdown.items():
                print(f"  {component}: {value:.6f}")
        else:
            test_loss = evaluate_ar_model(
                model=training_results['model'],
                data_loader=test_loader,
                loss_fn=testing_loss_fn,
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
            'tracking': training_results.get('tracking', {}),
            'loss_breakdown_info': {
                'has_training_breakdown': hasattr(training_loss_fn, '_is_multi_instance') and training_loss_fn._is_multi_instance,
                'has_validation_breakdown': hasattr(validation_loss_fn, '_is_multi_instance') and validation_loss_fn._is_multi_instance,
                'has_testing_breakdown': hasattr(testing_loss_fn, '_is_multi_instance') and testing_loss_fn._is_multi_instance,
                'training_loss_instances': getattr(training_loss_fn, '_loss_instances', None),
                'validation_loss_instances': getattr(validation_loss_fn, '_loss_instances', None),
                'testing_loss_instances': getattr(testing_loss_fn, '_loss_instances', None)
            }
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
        
        # Run training (multi-seed or single-seed)
        results = run_multi_seed_experiments(config)
        
        print(f"\n=== Experiment Complete ===")
        if 'base_save_dir' in results:
            print(f"Results saved to: {results['base_save_dir']}")
        else:
            print(f"Results saved to: {results['save_dir']}")
        
        return results
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
