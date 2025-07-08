import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def create_training_config(model_type='MLPSampler', loss_type='crps', **kwargs):
    """
    Create a training configuration dictionary with default values.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('MLPSampler' or 'SimpleAffineNormal')
    loss_type : str
        Type of loss function ('crps', 'energy_score', or 'log_likelihood')
    **kwargs : dict
        Additional configuration parameters to override defaults
        
    Returns:
    --------
    dict
        Complete training configuration dictionary
    """
    # Set defaults based on model type
    if model_type == 'SimpleAffineNormal':
        default_lr = 0.001
        if loss_type not in ['log_likelihood', 'energy_score', 'crps']:
            raise ValueError(f"Loss type '{loss_type}' not supported for SimpleAffineNormal")
    elif model_type in ['MLPSampler', 'FGNEncoderSampler']:
        default_lr = 0.001
        if loss_type not in ['crps', 'energy_score']:
            raise ValueError(f"Loss type '{loss_type}' not supported for {model_type}")
    else:
        raise ValueError(f"Model type '{model_type}' not recognized")
    
    # Default configuration
    config = {
        # Model and loss configuration
        'model_type': model_type,
        'loss_type': loss_type,
        
        # Training parameters
        'n_epochs': 100,
        'patience': 10,
        'learning_rate': default_lr,
        'n_samples': 100,  # For CRPS and energy score loss
        'device': 'cpu',
        'verbose': True,
        
        # Saving and tracking
        'save_path': None,
        'track_weights': False,
        'track_samples_every': 5,
        'track_moments': False,  # NEW: Enable moments tracking
        'moments_to_track': [1, 2, 3, 4],  # NEW: Which moments to track
        'noise_samples': None,  # NEW: Noise samples for true moments calculation
        
        # Test data for intermediate plots
        'x_test': None,
        'y_test': None,
        'noise_args': None,
        
        # Optional parameters
        'optimizer': None,
        'criterion': None,
    }
    
    # Update with provided kwargs
    config.update(kwargs)
    
    return config

def train_model(model, train_loader, val_loader, training_config):
    """
    Unified training function for all model types and loss functions.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train (MLPSampler, SimpleAffineNormal, etc.)
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    training_config : dict
        Configuration dictionary containing all training parameters.
        Use create_training_config() for easy configuration creation.
        Required keys:
        - 'model_type': str ('MLPSampler', 'FGNEncoderSampler', or 'SimpleAffineNormal')
        - 'loss_type': str ('crps', 'energy_score', or 'log_likelihood')
        
        Optional keys with defaults:
        - 'n_epochs': int (default: 100)
        - 'patience': int (default: 10)
        - 'learning_rate': float (default: 0.001)
        - 'n_samples': int (default: 100) - For CRPS and energy score loss
        - 'device': str (default: 'cpu')
        - 'verbose': bool (default: True)
        - 'save_path': str or None (default: None)
        - 'track_weights': bool (default: False)
        - 'track_samples_every': int (default: 5)
        - 'track_moments': bool (default: False) - Enable moments tracking
        - 'moments_to_track': list (default: [1, 2, 3, 4]) - Which moments to track
        - 'noise_samples': numpy.ndarray or None (default: None) - Noise samples for true moments calculation
        - 'x_test': torch.Tensor or None (default: None)
        - 'y_test': torch.Tensor or None (default: None)
        - 'noise_args': dict or None (default: None)
        - 'optimizer': torch.optim or None (default: None)
        - 'criterion': function or None (default: None)
        
    Returns:
    --------
    dict
        Dictionary containing training history and best model information
    """
    # Extract configuration
    model_type = training_config.get('model_type', 'MLPSampler')
    loss_type = training_config.get('loss_type', 'crps')
    n_epochs = training_config.get('n_epochs', 100)
    patience = training_config.get('patience', 10)
    learning_rate = training_config.get('learning_rate', 0.001)
    n_samples = training_config.get('n_samples', 1000)
    train_n_samples = training_config.get('train_n_samples', 10)
    device = training_config.get('device', 'cpu')
    verbose = training_config.get('verbose', True)
    save_path = training_config.get('save_path', None)
    track_weights = training_config.get('track_weights', False)
    track_samples_every = training_config.get('track_samples_every', 5)
    track_moments = training_config.get('track_moments', False)  # NEW
    moments_to_track = training_config.get('moments_to_track', [1, 2, 3, 4])  # NEW
    noise_samples = training_config.get('noise_samples', None)  # NEW
    x_test = training_config.get('x_test', None)
    y_test = training_config.get('y_test', None)
    noise_args = training_config.get('noise_args', None)
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = training_config.get('optimizer', None)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up loss function based on model type and loss type
    criterion = training_config.get('criterion', None)
    if criterion is None:
        criterion = _get_loss_function(model, model_type, loss_type, train_n_samples)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    # Initialize weight tracker if requested
    weight_tracker = None
    intermediate_plots_dir = None
    weight_plots_dir = None
    model_checkpoints_dir = None
    if track_weights or (y_test is not None):
        from experiments.visualisation.weight_tracking import WeightTracker
        from experiments.visualisation.plotting import create_intermediate_sample_plots, create_training_progression_plot
        weight_tracker = WeightTracker(model, track_weights=track_weights, 
                                     track_samples_every=track_samples_every)
        
        # Create directory for intermediate plots if saving model
        if save_path is not None:
            base_dir = os.path.dirname(save_path)
            intermediate_plots_dir = os.path.join(base_dir, 'intermediate_plots')
            weight_plots_dir = os.path.join(base_dir, 'weight_evolution')
            model_checkpoints_dir = os.path.join(base_dir, 'model_checkpoints')
            os.makedirs(intermediate_plots_dir, exist_ok=True)
            os.makedirs(weight_plots_dir, exist_ok=True)
            os.makedirs(model_checkpoints_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Use tqdm for progress bar if verbose
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]") if verbose else train_loader
        
        for x_batch, y_batch in train_iterator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss based on model type and loss type
            loss = _compute_loss(model, x_batch, y_batch, criterion, model_type, loss_type, train_n_samples)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
        
        # Validation phase
        val_loss = _evaluate_loss_internal(model, val_loader, criterion, model_type, loss_type, train_n_samples, device)
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or model_type in ['MLPSampler', 'FGNEncoderSampler']):
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Track weights and create intermediate plots
        if weight_tracker is not None:
            # Record weights
            weight_tracker.record_weights(epoch + 1)
            
            # Create intermediate sample plots if requested
            if (y_test is not None and 
                weight_tracker.should_record_samples(epoch + 1) and 
                intermediate_plots_dir is not None):
                try:
                    # For SimpleAffineNormal, x_test should be None
                    x_for_plot = x_test if model_type in ['MLPSampler', 'FGNEncoderSampler'] else None
                    print(f"n_samples for intermediate plots: {n_samples}")
                    plot_path = create_intermediate_sample_plots(
                        model, x_for_plot, y_test, noise_args, epoch + 1,
                        intermediate_plots_dir, n_samples=n_samples, device=device
                    )
                    weight_tracker.record_sample_epoch(epoch + 1)
                    
                    # Save model checkpoint at this epoch
                    if model_checkpoints_dir is not None:
                        checkpoint_path = os.path.join(model_checkpoints_dir, f'model_epoch_{epoch+1:03d}.pt')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'model_type': model_type,
                            'val_loss': val_loss
                        }, checkpoint_path)
                    
                    if verbose and ((epoch + 1) % 10 == 0 or model_type in ['MLPSampler', 'FGNEncoderSampler']):
                        print(f"  Saved intermediate samples: {os.path.basename(plot_path)}")
                except Exception as e:
                    if verbose and ((epoch + 1) % 10 == 0 or model_type in ['MLPSampler', 'FGNEncoderSampler']):
                        print(f"  Warning: Failed to create intermediate plot: {e}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            patience_counter = 0
            
            # Save best model if path provided
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model weights
    if best_model is not None:
        model.load_state_dict(best_model['model_state_dict'])
    
    # Create weight evolution plots if tracking was enabled
    weight_evolution_info = {}
    if weight_tracker is not None and weight_tracker.track_weights:
        try:
            figures = weight_tracker.plot_weight_evolution(save_dir=weight_plots_dir, noise_args=noise_args)
            weight_evolution_info = {
                'n_plots_created': len(figures),
                'sample_epochs': weight_tracker.sample_epochs.copy(),
                'weight_plots_dir': weight_plots_dir if save_path else None
            }
            # Close figures to save memory
            for fig in figures:
                plt.close(fig)
            if verbose:
                print(f"Created {len(figures)} weight evolution plots")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to create weight evolution plots: {e}")
    
    # Create training progression landscape plot if we have sample epochs
    progression_plot_path = None
    if (weight_tracker is not None and 
        y_test is not None and 
        len(weight_tracker.sample_epochs) > 0 and 
        model_checkpoints_dir is not None):
        try:
            progression_plot_path = create_training_progression_plot(
                model=model,
                x_test=x_test,
                y_test=y_test,
                noise_args=noise_args,
                base_save_dir=os.path.dirname(save_path) if save_path else None,
                sample_epochs=weight_tracker.sample_epochs,
                model_checkpoints_dir=model_checkpoints_dir,
                model_type=model_type,
                n_samples=n_samples,
                test_point_idx=0,
                device=device
            )
            if progression_plot_path and verbose:
                print(f"Created training progression landscape plot: {os.path.basename(progression_plot_path)}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to create training progression plot: {e}")
    
    # Create training progression moments plot if requested
    moments_plot_path = None
    if (track_moments and 
        weight_tracker is not None and 
        y_test is not None and 
        len(weight_tracker.sample_epochs) > 0 and 
        model_checkpoints_dir is not None):
        from experiments.visualisation.plotting import create_training_progression_moments_plot
        moments_plot_path = create_training_progression_moments_plot(
            model=model,
            x_test=x_test,
            y_test=y_test,
            noise_args=noise_args,
            base_save_dir=os.path.dirname(save_path) if save_path else None,
            sample_epochs=weight_tracker.sample_epochs,
            model_checkpoints_dir=model_checkpoints_dir,
            model_type=model_type,
            moments_to_track=moments_to_track,
            n_samples=n_samples,
            test_point_idx=0,
            device=device,
            noise_samples=noise_samples
        )
        if moments_plot_path and verbose:
            print(f"Created training progression moments plot: {os.path.basename(moments_plot_path)}")

        # create a plot of difference in moment tensors
        from experiments.visualisation.plotting import create_training_progression_moment_tensors_plot
        moment_tensors_plot_path = create_training_progression_moment_tensors_plot(
            model=model,
            x_test=x_test,
            y_test=y_test,
            noise_args=noise_args,
            base_save_dir=os.path.dirname(save_path) if save_path else None,
            sample_epochs=weight_tracker.sample_epochs,
            model_checkpoints_dir=model_checkpoints_dir,
            model_type=model_type,
            moments_to_track=moments_to_track,
            n_samples=n_samples,
            test_point_idx=0,
            device=device,
        )
        if moment_tensors_plot_path and verbose:
            print(f"Created training progression moment tensors plot: {os.path.basename(moment_tensors_plot_path)}")

    # Update weight evolution info with progression plot
    if weight_evolution_info:
        weight_evolution_info['progression_plot_path'] = progression_plot_path
        weight_evolution_info['moments_plot_path'] = moments_plot_path
    else:
        weight_evolution_info = {
            'progression_plot_path': progression_plot_path,
            'moments_plot_path': moments_plot_path
        }
    
    return {
        'model': model,
        'history': history,
        'best_epoch': best_model['epoch'] if best_model else n_epochs,
        'best_val_loss': best_val_loss,
        'weight_tracking': weight_evolution_info
    }

def _get_loss_function(model, model_type, loss_type, n_samples):
    """Get the appropriate loss function based on model and loss type."""
    if model_type == 'SimpleAffineNormal':
        if loss_type == 'log_likelihood':
            return lambda x, y: -model.log_likelihood(y).mean()
        elif loss_type == 'energy_score':
            from common.losses import energy_score_loss
            def energy_loss_fn(x, y):
                batch_size = y.shape[0]
                samples = model.forward(n_samples=n_samples, batch_size=batch_size)
                energy_scores = energy_score_loss(samples, y)
                return energy_scores.mean()
            return energy_loss_fn
        elif loss_type == 'crps':
            return lambda x, y: model.crps_loss(y, n_samples=n_samples)
        else:
            raise ValueError(f"Loss type '{loss_type}' not supported for SimpleAffineNormal")
    
    elif model_type in ['MLPSampler', 'FGNEncoderSampler']:
        if loss_type == 'crps':
            return lambda x, y: model.crps_loss(x, y, n_samples=n_samples)
        elif loss_type == 'energy_score':
            return lambda x, y: model.energy_score_loss(x, y, n_samples=n_samples)
        else:
            raise ValueError(f"Loss type '{loss_type}' not supported for {model_type}")
    
    else:
        raise ValueError(f"Model type '{model_type}' not recognized")

def _compute_loss(model, x_batch, y_batch, criterion, model_type, loss_type, n_samples):
    """Compute loss based on model type."""
    if model_type == 'SimpleAffineNormal':
        # SimpleAffineNormal doesn't use x_batch for most loss functions
        return criterion(x_batch, y_batch)
    else:
        # MLPSampler uses both x and y
        return criterion(x_batch, y_batch)

def _evaluate_loss_internal(model, data_loader, criterion, model_type, loss_type, n_samples, device):
    """Evaluate model loss during training (internal function)."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = _compute_loss(model, x_batch, y_batch, criterion, model_type, loss_type, n_samples)
            losses.append(loss.item())
    
    return np.mean(losses)

def evaluate_model(model, data_loader, training_config):
    """
    Evaluate a model on a dataset using the same configuration as training.
    
    Parameters:
    -----------
    model : nn.Module
        The model to evaluate
    data_loader : DataLoader
        DataLoader for evaluation data
    training_config : dict
        Same configuration dict used for training
        
    Returns:
    --------
    float
        Average loss on the dataset
    """
    # Import here to avoid circular imports
    from .evaluation import evaluate_model as eval_model, create_evaluation_config
    
    model_type = training_config.get('model_type', 'MLPSampler')
    loss_type = training_config.get('loss_type', 'crps')
    n_samples = training_config.get('n_samples', 100)
    device = training_config.get('device', 'cpu')
    
    # Create evaluation config for loss evaluation
    eval_config = create_evaluation_config(
        model_type=model_type,
        evaluation_type='loss',
        loss_type=loss_type,
        n_samples=n_samples,
        device=device
    )
    
    return eval_model(model, data_loader, eval_config)

def prepare_data_loaders(x_train, y_train, x_val, y_val, batch_size=32):
    """
    Prepare DataLoader objects for training and validation.
    
    Parameters:
    -----------
    x_train : torch.Tensor
        Training inputs
    y_train : torch.Tensor
        Training targets
    x_val : torch.Tensor
        Validation inputs
    y_val : torch.Tensor
        Validation targets
    batch_size : int
        Batch size
        
    Returns:
    --------
    tuple
        (train_loader, val_loader)
    """
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

# Legacy function names for backward compatibility
def train_affine_model_energy(model, train_loader, val_loader, **kwargs):
    """Legacy wrapper for SimpleAffineNormal with energy score loss."""
    training_config = {
        'model_type': 'SimpleAffineNormal',
        'loss_type': 'energy_score',
        **kwargs
    }
    return train_model(model, train_loader, val_loader, training_config)

def train_affine_model(model, train_loader, val_loader, **kwargs):
    """Legacy wrapper for SimpleAffineNormal with log likelihood loss."""
    training_config = {
        'model_type': 'SimpleAffineNormal',
        'loss_type': 'log_likelihood',
        **kwargs
    }
    return train_model(model, train_loader, val_loader, training_config)
