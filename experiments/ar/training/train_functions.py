"""
Simple training and evaluation functions for AR time series models.

Core functions that assume the loss function is already provided.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional


def generate_autoregressive_samples(model, initial_input, output_timesteps, output_dimension, n_samples, batch_size):
    """
    Generate samples autoregressively for multi-timestep outputs.
    
    For the first timestep, generate n_samples predictions. For each of those samples,
    proceed to autoregressively rollout with single sample predictions for subsequent timesteps.
    
    Args:
        model: PyTorch model that takes [batch_size, input_dim] and returns [batch_size, n_samples, output_dim]
        initial_input: Initial input tensor of shape [batch_size, input_dim]
        output_timesteps: Number of timesteps to generate
        output_dimension: Dimension of each timestep output
        n_samples: Number of samples to generate
        batch_size: Batch size
        
    Returns:
        samples: Generated samples of shape [batch_size, n_samples, output_timesteps, output_dimension]
    """
    device = initial_input.device
    
    # Initialize storage for all timestep outputs
    all_samples = torch.zeros(batch_size, n_samples, output_timesteps, output_dimension, device=device)
    
    # First timestep: generate n_samples predictions
    if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
        # SimpleAffineNormal doesn't use input
        first_step_samples = model.forward(n_samples=n_samples, batch_size=batch_size)  # [batch_size, n_samples, output_dim]
    else:
        # Standard models use initial input
        first_step_samples = model(initial_input, n_samples=n_samples)  # [batch_size, n_samples, output_dim]
    
    # Store first timestep samples
    all_samples[:, :, 0, :] = first_step_samples
    
    # For each sample trajectory, continue autoregressively with single samples
    if output_timesteps > 1:
        for sample_idx in range(n_samples):
            # Current input for this trajectory is the sample from previous timestep
            current_input = first_step_samples[:, sample_idx, :]  # Shape: [batch_size, output_dim]
            
            # Generate remaining timesteps for this trajectory
            for t in range(1, output_timesteps):
                if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                    # SimpleAffineNormal doesn't use input - generate single sample
                    step_sample = model.forward(n_samples=1, batch_size=batch_size)  # [batch_size, 1, output_dim]
                    step_sample = step_sample.squeeze(1)  # [batch_size, output_dim]
                else:
                    # Standard models use current input - generate single sample
                    step_sample = model(current_input, n_samples=1)  # [batch_size, 1, output_dim]
                    step_sample = step_sample.squeeze(1)  # [batch_size, output_dim]
                
                # Store sample for this timestep and trajectory
                all_samples[:, sample_idx, t, :] = step_sample
                
                # Update input for next timestep in this trajectory
                current_input = step_sample
    
    return all_samples


def train_ar_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    n_samples: int = 50,
    patience: int = 15,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    tracking_config: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    model_config: Optional[Dict] = None
) -> Dict:
    """
    Train a model on AR time series data.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function that takes (y_pred_samples, y_true) and returns loss
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        n_samples: Number of samples for probabilistic models
        patience: Early stopping patience
        device: Device to train on
        verbose: Whether to print progress
        tracking_config: Configuration for AR inference tracking (optional)
        save_dir: Directory to save tracking data (optional)
        model_config: Model configuration for tracker (optional)
        
    Returns:
        Dictionary containing training history, best model, and tracking info
    """
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize AR inference tracker if requested
    ar_tracker = None
    if tracking_config and tracking_config.get('enabled', False):
        from experiments.ar.tracking.tracking import ARInferenceTracker
        
        if save_dir is None:
            raise ValueError("save_dir must be provided when tracking is enabled")
        
        ar_tracker = ARInferenceTracker(
            save_dir=save_dir,
            track_every=tracking_config.get('track_every', 10),
            sample_indices=tracking_config.get('sample_indices', [0, 1, 2]),
            n_samples=tracking_config.get('n_samples', n_samples),
            kde_bandwidth=tracking_config.get('kde_bandwidth', 0.1),
            contour_levels=tracking_config.get('contour_levels', [0.65, 0.95, 0.99]),
            max_checkpoints=tracking_config.get('max_checkpoints', None),
            enabled=True
        )
        if verbose:
            print(f"AR inference tracking enabled (every {ar_tracker.track_every} epochs)")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    if verbose:
        print(f"Starting training for {n_epochs} epochs on {device}...")
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Progress bar for training
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]") if verbose else train_loader
        
        for batch in train_iterator:
            # Extract batch data
            inputs = batch['input'].to(device)  # Shape: (batch_size, input_timesteps, dimension)
            targets = batch['target'].to(device)  # Shape: (batch_size, output_timesteps, dimension)
            
            # Validate timestep requirements for models
            batch_size, input_timesteps, input_dimension = inputs.shape
            _, output_timesteps, output_dimension = targets.shape
            
            # Current models only support single timestep inputs (but can generate multi-timestep outputs autoregressively)
            if input_timesteps != 1:
                raise ValueError(f"Current models only support single timestep inputs. "
                               f"Got input_timesteps={input_timesteps}. "
                               f"Please set input_timesteps=1 in your data configuration.")
            
            # Flatten inputs for single timestep case
            inputs_flat = inputs.reshape(batch_size, input_dimension)  # Shape: (batch_size, input_dimension)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Generate samples and compute loss
            if output_timesteps == 1:
                # Single timestep output - direct forward pass
                if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                    samples = model.forward(n_samples=n_samples, batch_size=batch_size)
                else:
                    samples = model(inputs_flat, n_samples=n_samples) # Shape: [B, S, D]
                
                targets_reshaped = targets.reshape(batch_size, output_dimension) # Shape: [B, D]
                
                loss_per_batch_item = loss_fn(samples, targets_reshaped) # Shape: [B]
            else:
                # Multi-timestep output - autoregressive generation
                samples_sequence = generate_autoregressive_samples(model, inputs_flat, output_timesteps, output_dimension, n_samples, batch_size) # Shape: [B, S, T, D]
                
                # Compute loss explicitly per timestep and sum
                total_loss_for_batch = 0
                for t in range(output_timesteps):
                    samples_t = samples_sequence[:, :, t, :]  # Shape: [B, S, D]
                    targets_t = targets[:, t, :]              # Shape: [B, D]
                    
                    # loss_fn returns a loss per batch item, shape [B]
                    loss_t_batch = loss_fn(samples_t, targets_t) 
                    total_loss_for_batch += loss_t_batch # element-wise sum, shape [B]
                
                # Average loss over timesteps for each batch item
                loss_per_batch_item = total_loss_for_batch / output_timesteps # Shape: [B]

            # Average over the batch to get a scalar loss
            if hasattr(loss_per_batch_item, 'mean'):
                loss = loss_per_batch_item.mean()
            else:
                loss = loss_per_batch_item
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Update progress bar
            if verbose:
                train_iterator.set_postfix({'loss': loss.item()})
        
        # Validation phase
        val_loss = evaluate_ar_model(model, val_loader, loss_fn, n_samples, device)
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Track model state if AR tracking is enabled
        if ar_tracker and ar_tracker.should_track(epoch + 1):
            additional_info = {
                'epoch_in_training': epoch + 1,
                'total_epochs': n_epochs,
                'n_samples_used': n_samples
            }
            ar_tracker.track_epoch(
                epoch=epoch + 1,
                model=model,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                optimizer=optimizer,
                additional_info=additional_info
            )
        
        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (patience: {patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    # Generate inference plots if requested and tracking was enabled
    tracking_summary = {}
    plot_paths = {}
    if ar_tracker and tracking_config.get('generate_plots_after_training', False):
        if verbose:
            print("\n=== Generating AR Inference Plots ===")
        
        try:
            plot_paths = ar_tracker.generate_inference_plots(
                dataset=train_loader.dataset,
                model_config=model_config,
                device=device,
                show_intermediate_steps=True
            )
            if verbose and plot_paths:
                print(f"Generated {len(plot_paths)} inference plots")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate inference plots: {e}")
    
    # Get tracking summary for results
    if ar_tracker:
        tracking_summary = ar_tracker.get_tracking_summary()
        tracking_summary['plot_paths'] = plot_paths
    
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'tracking': tracking_summary
    }


def evaluate_ar_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    n_samples: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    return_predictions: bool = False
) -> Tuple:
    """
    Evaluate a model on AR time series data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        loss_fn: Loss function that takes (y_pred_samples, y_true) and returns loss
        n_samples: Number of samples for probabilistic models
        device: Device to evaluate on
        return_predictions: If True, return predictions along with loss
        
    Returns:
        If return_predictions=False: loss (float)
        If return_predictions=True: (loss, predictions, targets)
    """
    model.eval()
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract batch data
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Validate timestep requirements for models
            batch_size, input_timesteps, input_dimension = inputs.shape
            _, output_timesteps, output_dimension = targets.shape
            
            # Current models only support single timestep inputs (but can generate multi-timestep outputs autoregressively)
            if input_timesteps != 1:
                raise ValueError(f"Current models only support single timestep inputs. "
                               f"Got input_timesteps={input_timesteps}. "
                               f"Please set input_timesteps=1 in your data configuration.")
            
            # Flatten inputs for single timestep case
            inputs_flat = inputs.reshape(batch_size, input_dimension)
            
            # Generate samples and compute loss
            if output_timesteps == 1:
                # Single timestep output - direct forward pass
                if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                    # SimpleAffineNormal has different forward signature
                    samples = model.forward(n_samples=n_samples, batch_size=batch_size)
                else:
                    # Standard interface for other models
                    samples = model(inputs_flat, n_samples=n_samples)
                
                # Reshape for loss computation
                targets_reshaped = targets.reshape(batch_size, output_dimension)
                loss_per_batch_item = loss_fn(samples, targets_reshaped) # Shape: [B]
                samples_for_return = samples.unsqueeze(2) # Reshape to [B, S, 1, D] for consistency
            else:
                # Multi-timestep output - autoregressive generation
                samples_sequence = generate_autoregressive_samples(model, inputs_flat, output_timesteps, output_dimension, n_samples, batch_size)
                
                # Compute loss explicitly per timestep and sum
                total_loss_for_batch = 0
                for t in range(output_timesteps):
                    samples_t = samples_sequence[:, :, t, :]  # Shape: [B, S, D]
                    targets_t = targets[:, t, :]              # Shape: [B, D]
                    
                    loss_t_batch = loss_fn(samples_t, targets_t) 
                    total_loss_for_batch += loss_t_batch
                
                loss_per_batch_item = total_loss_for_batch / output_timesteps
                samples_for_return = samples_sequence
            
            # Average over the batch to get a scalar loss
            if hasattr(loss_per_batch_item, 'mean'):
                loss = loss_per_batch_item.mean()
            else:
                loss = loss_per_batch_item
            
            all_losses.append(loss.item())
            
            if return_predictions:
                all_predictions.append(samples_for_return.cpu())
                all_targets.append(targets.cpu())
    
    # Average loss
    avg_loss = np.mean(all_losses)
    
    if return_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return avg_loss, predictions, targets
    else:
        return avg_loss
