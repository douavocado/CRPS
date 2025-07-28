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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from common.losses import ProgressiveTrainingState


def generate_autoregressive_samples(model, initial_input, output_timesteps, output_dimension, n_samples, batch_size, use_consistent_noise=False):
    """
    Generate samples autoregressively for multi-timestep outputs.
    
    Args:
        model: PyTorch model that takes [batch_size, input_dim] and returns [batch_size, n_samples, output_dim]
        initial_input: Initial input tensor of shape [batch_size, input_dim]
        output_timesteps: Number of timesteps to generate
        output_dimension: Dimension of each timestep output
        n_samples: Number of samples to generate
        batch_size: Batch size
        use_consistent_noise: If True, each sample trajectory uses the same noise pattern throughout all timesteps.
                            If False, uses original behavior with random noise at each timestep.
        
    Returns:
        samples: Generated samples of shape [batch_size, n_samples, output_timesteps, output_dimension]
    """
    device = initial_input.device
    
    # Initialize storage for all timestep outputs
    all_samples = torch.zeros(batch_size, n_samples, output_timesteps, output_dimension, device=device)
    
    if use_consistent_noise:
        # CONSISTENT NOISE MODE: Each trajectory uses the same noise pattern throughout all timesteps
        
        # Generate deterministic seeds for each sample trajectory to ensure consistent noise
        # These seeds will be used for ALL timesteps (including the first) for each trajectory
        trajectory_seeds = torch.randint(0, 2**31 - 1, (n_samples,)).tolist()
        
        # First timestep: generate n_samples predictions using individual seeds for each sample
        # This ensures each trajectory starts with a specific noise pattern that can be reproduced
        first_step_samples = torch.zeros(batch_size, n_samples, output_dimension, device=device)
        
        for sample_idx in range(n_samples):
            trajectory_seed = trajectory_seeds[sample_idx]
            
            if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                # SimpleAffineNormal doesn't use input
                try:
                    sample = model.forward(n_samples=1, batch_size=batch_size, random_seed=trajectory_seed)  # [batch_size, 1, output_dim]
                except TypeError:
                    # Fallback if SimpleAffineNormal doesn't support random_seed
                    sample = model.forward(n_samples=1, batch_size=batch_size)  # [batch_size, 1, output_dim]
            else:
                # Standard models use initial input
                try:
                    sample = model(initial_input, n_samples=1, random_seed=trajectory_seed)  # [batch_size, 1, output_dim]
                except TypeError:
                    # Fallback for models that don't support random_seed
                    sample = model(initial_input, n_samples=1)  # [batch_size, 1, output_dim]
            
            first_step_samples[:, sample_idx, :] = sample.squeeze(1)  # [batch_size, output_dim]
        
        # Store first timestep samples
        all_samples[:, :, 0, :] = first_step_samples
        
    else:
        # ORIGINAL MODE: Standard random noise at each timestep
        
        # First timestep: generate n_samples predictions with random noise (original behavior)
        if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
            # SimpleAffineNormal doesn't use input
            first_step_samples = model.forward(n_samples=n_samples, batch_size=batch_size)  # [batch_size, n_samples, output_dim]
        else:
            # Standard models use initial input
            first_step_samples = model(initial_input, n_samples=n_samples)  # [batch_size, n_samples, output_dim]
        
        # Store first timestep samples
        all_samples[:, :, 0, :] = first_step_samples
    
    # For each sample trajectory, continue autoregressively
    if output_timesteps > 1:
        
        if use_consistent_noise:
            # CONSISTENT NOISE MODE: Use the same deterministic seeds for each trajectory
            
            for sample_idx in range(n_samples):
                # Current input for this trajectory is the sample from previous timestep
                current_input = first_step_samples[:, sample_idx, :]  # Shape: [batch_size, output_dim]
                
                # Use the SAME deterministic seed for this sample trajectory across ALL timesteps
                # This ensures the same noise pattern is used throughout the entire trajectory
                trajectory_seed = trajectory_seeds[sample_idx]
                
                # Generate remaining timesteps for this trajectory using the SAME seed
                for t in range(1, output_timesteps):
                    # Use the same seed for all timesteps in this trajectory to maintain
                    # the same noise pattern as was used in the first timestep
                    timestep_seed = trajectory_seed  # Same seed for all timesteps in this trajectory
                    
                    if hasattr(model, '__class__') and 'SimpleAffineNormal' in model.__class__.__name__:
                        # SimpleAffineNormal doesn't use input - generate deterministic single sample
                        try:
                            step_sample = model.forward(n_samples=1, batch_size=batch_size, random_seed=timestep_seed)  # [batch_size, 1, output_dim]
                        except TypeError:
                            # Fallback if SimpleAffineNormal doesn't support random_seed
                            step_sample = model.forward(n_samples=1, batch_size=batch_size)  # [batch_size, 1, output_dim]
                        step_sample = step_sample.squeeze(1)  # [batch_size, output_dim]
                    else:
                        # Standard models (MLPSampler, FGNEncoderSampler) use current input - generate deterministic single sample
                        try:
                            step_sample = model(current_input, n_samples=1, random_seed=timestep_seed)  # [batch_size, 1, output_dim]
                        except TypeError:
                            # Fallback for models that don't support random_seed
                            step_sample = model(current_input, n_samples=1)  # [batch_size, 1, output_dim]
                        step_sample = step_sample.squeeze(1)  # [batch_size, output_dim]
                    
                    # Store sample for this timestep and trajectory
                    all_samples[:, sample_idx, t, :] = step_sample
                    
                    # Update input for next timestep in this trajectory
                    current_input = step_sample
                    
        else:
            # ORIGINAL MODE: Standard random noise at each timestep
            
            for sample_idx in range(n_samples):
                # Current input for this trajectory is the sample from previous timestep
                current_input = first_step_samples[:, sample_idx, :]  # Shape: [batch_size, output_dim]
                
                # Generate remaining timesteps for this trajectory (original behavior)
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
    training_loss_fn: callable = None,
    validation_loss_fn: callable = None,
    testing_loss_fn: callable = None,
    loss_fn: callable = None,  # Backward compatibility
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    n_samples: int = 50,
    patience: int = 15,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    tracking_config: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    model_config: Optional[Dict] = None,
    use_consistent_ar_noise: bool = False,
    training_config: Optional[Dict] = None
) -> Dict:
    """
    Train a model on AR time series data.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        training_loss_fn: Loss function for training that takes (y_pred_samples, y_true) and returns loss
        validation_loss_fn: Loss function for validation that takes (y_pred_samples, y_true) and returns loss
        testing_loss_fn: Loss function for testing that takes (y_pred_samples, y_true) and returns loss
        loss_fn: (Deprecated) Legacy single loss function for backward compatibility
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        n_samples: Number of samples for probabilistic models
        patience: Early stopping patience
        device: Device to train on
        verbose: Whether to print progress
        tracking_config: Configuration for AR inference tracking (optional)
        save_dir: Directory to save tracking data (optional)
        model_config: Model configuration for tracker (optional)
        use_consistent_ar_noise: If True, each sample trajectory uses the same noise pattern throughout all timesteps
        
    Returns:
        Dictionary containing training history, best model, and tracking info
    """
    # Handle backward compatibility for loss functions
    if loss_fn is not None and (training_loss_fn is None or validation_loss_fn is None):
        # Legacy mode: use the same loss function for both training and validation
        training_loss_fn = loss_fn
        validation_loss_fn = loss_fn
        if verbose:
            print("Using legacy single loss function for both training and validation")
    elif training_loss_fn is None or validation_loss_fn is None:
        raise ValueError("Must provide either 'loss_fn' (legacy) or both 'training_loss_fn' and 'validation_loss_fn'")
    elif verbose:
        print("Using separate training, validation, and testing loss functions")
    
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
        'epochs': [],
        'train_breakdown': [],  # Store training loss breakdowns
        'val_breakdown': [],    # Store validation loss breakdowns
        'progressive_training': []  # Store progressive training state per epoch
    }
    
    # Check if we should show loss breakdown
    show_training_breakdown = hasattr(training_loss_fn, '_is_multi_instance') and training_loss_fn._is_multi_instance
    show_validation_breakdown = hasattr(validation_loss_fn, '_is_multi_instance') and validation_loss_fn._is_multi_instance
    
    # Initialize progressive training state for multi-instance training losses
    progressive_state = None
    if show_training_breakdown:
        progressive_state = ProgressiveTrainingState()
        if verbose:
            print("Progressive training state initialized for multi-instance training loss")
    
    if verbose:
        print(f"Starting training for {n_epochs} epochs on {device}...")
        if show_training_breakdown:
            print("Training loss breakdown will be shown (multiple loss instances detected)")
        if show_validation_breakdown:
            print("Validation loss breakdown will be shown (multiple loss instances detected)")
    
    # Training loop
    for epoch in range(n_epochs):
        # Update progressive training state
        if progressive_state is not None:
            progressive_state.update_epoch(epoch)
        
        # Training phase
        model.train()
        train_losses = []
        train_breakdown_total = {}
        
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
                
                # Compute loss with potential breakdown
                if show_training_breakdown:
                    loss_per_batch_item, breakdown = training_loss_fn(
                        samples, targets_reshaped, return_components=True, 
                        progressive_state=progressive_state, verbose=verbose
                    )
                    # Accumulate breakdown for epoch average
                    for component, value in breakdown['weighted_losses'].items():
                        if component not in train_breakdown_total:
                            train_breakdown_total[component] = []
                        train_breakdown_total[component].append(value)
                else:
                    loss_per_batch_item = training_loss_fn(
                        samples, targets_reshaped, progressive_state=progressive_state
                    ) # Shape: [B]
            else:
                # Multi-timestep output - autoregressive generation
                samples_sequence = generate_autoregressive_samples(model, inputs_flat, output_timesteps, output_dimension, n_samples, batch_size, use_consistent_ar_noise) # Shape: [B, S, T, D]
                
                # Compute loss explicitly per timestep and sum
                total_loss_for_batch = 0
                total_breakdown_for_batch = {}
                
                for t in range(output_timesteps):
                    samples_t = samples_sequence[:, :, t, :]  # Shape: [B, S, D]
                    targets_t = targets[:, t, :]              # Shape: [B, D]
                    
                    # training_loss_fn returns a loss per batch item, shape [B]
                    if show_training_breakdown:
                        loss_t_batch, breakdown_t = training_loss_fn(
                            samples_t, targets_t, return_components=True, 
                            progressive_state=progressive_state, verbose=verbose
                        )
                        # Accumulate breakdown across timesteps
                        for component, value in breakdown_t['weighted_losses'].items():
                            if component not in total_breakdown_for_batch:
                                total_breakdown_for_batch[component] = 0
                            total_breakdown_for_batch[component] += value
                    else:
                        loss_t_batch = training_loss_fn(
                            samples_t, targets_t, progressive_state=progressive_state
                        ) 
                    
                    total_loss_for_batch += loss_t_batch # element-wise sum, shape [B]
                
                # Average loss over timesteps for each batch item
                loss_per_batch_item = total_loss_for_batch / output_timesteps # Shape: [B]
                
                # Average breakdown over timesteps and accumulate
                if show_training_breakdown:
                    for component, value in total_breakdown_for_batch.items():
                        avg_value = value / output_timesteps
                        if component not in train_breakdown_total:
                            train_breakdown_total[component] = []
                        train_breakdown_total[component].append(avg_value)

            # Average over the batch to get a scalar loss
            if hasattr(loss_per_batch_item, 'mean'):
                loss = loss_per_batch_item.mean()
            else:
                loss = loss_per_batch_item
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping if specified
            grad_clip_norm = training_config.get('grad_clip_norm', None)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Update progress bar
            if verbose:
                train_iterator.set_postfix({'loss': loss.item()})
        
        # Validation phase
        val_loss, val_breakdown = evaluate_ar_model(
            model, val_loader, validation_loss_fn, n_samples, device, 
            use_consistent_ar_noise=use_consistent_ar_noise, return_breakdown=show_validation_breakdown
        )
        
        # Update progressive training state with validation results
        if progressive_state is not None and val_breakdown is not None:
            # Update with overall validation loss
            progressive_state.update_validation_loss('default', val_loss)
            
            # Update individual validation loss instances if available
            if 'individual_losses' in val_breakdown:
                # New structure with individual_losses key
                for instance_name, loss_value in val_breakdown['individual_losses'].items():
                    progressive_state.update_validation_loss(instance_name, loss_value)
                
                # Update combined validation losses
                progressive_state.update_combined_validation_losses(val_breakdown['individual_losses'])
            else:
                # Flat dictionary structure (current case)
                for instance_name, loss_value in val_breakdown.items():
                    progressive_state.update_validation_loss(instance_name, loss_value)
                
                # Update combined validation losses
                progressive_state.update_combined_validation_losses(val_breakdown)
                
                # Debug: Show combined validation loss tracking info
                if verbose and epoch == 0:  # Only show once at the beginning
                    print(f"Progressive Training: Tracking individual validation losses: {list(val_breakdown.keys())}")
                    if hasattr(progressive_state, 'combined_validation_definitions'):
                        print(f"Progressive Training: Defined combined losses: {list(progressive_state.combined_validation_definitions.keys())}")
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Record breakdowns
        epoch_train_breakdown = {}
        epoch_val_breakdown = {}
        
        if show_training_breakdown and train_breakdown_total:
            for component, values in train_breakdown_total.items():
                epoch_train_breakdown[component] = np.mean(values)
        
        if show_validation_breakdown and val_breakdown:
            epoch_val_breakdown = val_breakdown.copy()
        
        history['train_breakdown'].append(epoch_train_breakdown)
        history['val_breakdown'].append(epoch_val_breakdown)
        
        # Record progressive training state
        progressive_info = {}
        if progressive_state is not None:
            progressive_info = {
                'active_losses': list(progressive_state.active_losses),
                'ever_activated_losses': list(progressive_state.ever_activated_losses),
                'plateau_counts': progressive_state.validation_plateau_counts.copy(),
                'improvement_counts': progressive_state.validation_improvement_counts.copy(),
                'activation_epochs': progressive_state.loss_activation_epochs.copy(),
                'deactivation_epochs': progressive_state.loss_deactivation_epochs.copy()
            }
        history['progressive_training'].append(progressive_info)
        
        # Print progress with breakdown if available
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Show progressive training status
            if progressive_state is not None and hasattr(training_loss_fn, '_loss_instances'):
                active_count = len(progressive_state.active_losses)
                total_count = len(training_loss_fn._loss_instances)
                print(f"  Progressive Training: {active_count}/{total_count} loss instances active")
                
                # Show plateau information for losses with validation-based conditions
                for instance in training_loss_fn._loss_instances:
                    instance_name = instance['name']
                    progressive_config = instance.get('progressive_training')
                    if progressive_config:
                        is_active = progressive_state.is_loss_active(instance_name)
                        status = "ACTIVE" if is_active else "inactive"
                        
                        # Check for validation plateau conditions
                        activation_conditions = progressive_config.get('activation_conditions', [])
                        deactivation_conditions = progressive_config.get('deactivation_conditions', [])
                        plateau_info = ""
                        
                        # Show activation conditions info
                        if activation_conditions:
                            for condition in activation_conditions:
                                if condition.get('type') == 'combined_validation_plateau':
                                    combined_name = condition.get('combined_name', 'unknown')
                                    plateau_count = progressive_state.get_plateau_count(combined_name)
                                    required_plateau = condition.get('plateau_epochs', 5)
                                    plateau_info += f" (activation: {combined_name} plateau {plateau_count}/{required_plateau})"
                                elif condition.get('type') == 'validation_plateau':
                                    val_instance = condition.get('validation_loss_instance', 'default')
                                    plateau_count = progressive_state.get_plateau_count(val_instance)
                                    required_plateau = condition.get('plateau_epochs', 5)
                                    plateau_info += f" (activation: {val_instance} plateau {plateau_count}/{required_plateau})"
                        
                        # Show deactivation conditions info
                        if deactivation_conditions:
                            for condition in deactivation_conditions:
                                if condition.get('type') == 'combined_validation_plateau':
                                    combined_name = condition.get('combined_name', 'unknown')
                                    plateau_count = progressive_state.get_plateau_count(combined_name)
                                    required_plateau = condition.get('plateau_epochs', 5)
                                    plateau_info += f" (deactivation: {combined_name} plateau {plateau_count}/{required_plateau})"
                        
                        print(f"    {instance_name}: {status}{plateau_info}")
            
            # Show training breakdown
            if show_training_breakdown and train_breakdown_total:
                print("  Training breakdown:")
                for component, values in train_breakdown_total.items():
                    avg_component_loss = np.mean(values)
                    print(f"    {component}: {avg_component_loss:.6f}")
            
            # Show validation breakdown
            if show_validation_breakdown and val_breakdown:
                print("  Validation breakdown:")
                for component, value in val_breakdown.items():
                    print(f"    {component}: {value:.6f}")
        
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
    
    # Generate spatial visualizations if spatial data is detected
    spatial_visualization_results = {}
    if save_dir and hasattr(train_loader.dataset, 'spatial_args') and train_loader.dataset.spatial_args is not None:
        if verbose:
            print("\n=== Generating Spatial AR Visualizations ===")
        
        try:
            from experiments.ar.visualisation.plotting import create_spatial_ar_visualizations
            
            spatial_visualization_results = create_spatial_ar_visualizations(
                dataset=train_loader.dataset,
                save_dir=save_dir,
                n_series_to_plot=3,
                n_timesteps_to_plot=None,  # Show all timesteps for complete rollout
                model=model,
                n_prediction_samples=min(n_samples, 50)  # Limit for performance
            )
            
            if verbose and spatial_visualization_results.get('plot_paths'):
                print(f"Generated {len(spatial_visualization_results['plot_paths'])} spatial visualization plots")
                print(f"Grid size: {spatial_visualization_results.get('grid_size')}")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate spatial visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Get tracking summary for results
    if ar_tracker:
        tracking_summary = ar_tracker.get_tracking_summary()
        tracking_summary['plot_paths'] = plot_paths
        tracking_summary['spatial_visualizations'] = spatial_visualization_results
    else:
        tracking_summary = {'spatial_visualizations': spatial_visualization_results}
    
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
    return_predictions: bool = False,
    use_consistent_ar_noise: bool = False,
    return_breakdown: bool = False
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
        use_consistent_ar_noise: If True, each sample trajectory uses the same noise pattern throughout all timesteps
        return_breakdown: If True, return a dictionary of component losses
        
    Returns:
        If return_predictions=False and return_breakdown=False: loss (float)
        If return_predictions=True and return_breakdown=False: (loss, predictions, targets)
        If return_predictions=False and return_breakdown=True: (loss, breakdown_dict)
        If return_predictions=True and return_breakdown=True: (loss, predictions, targets, breakdown_dict)
    """
    model.eval()
    all_losses = []
    all_predictions = []
    all_targets = []
    breakdown_accumulator = {}
    has_breakdown = hasattr(loss_fn, '_is_multi_instance') and loss_fn._is_multi_instance
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract batch data
            inputs = batch['input'].to(device)  # Shape: (batch_size, input_timesteps, dimension)
            targets = batch['target'].to(device)  # Shape: (batch_size, output_timesteps, dimension)
            
            # Validate timestep requirements
            batch_size, input_timesteps, input_dimension = inputs.shape
            _, output_timesteps, output_dimension = targets.shape
            
            if input_timesteps != 1:
                raise ValueError(f"Current models only support single timestep inputs. "
                               f"Got input_timesteps={input_timesteps}.")
            
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
                
                # Compute loss with potential breakdown
                if return_breakdown and has_breakdown:
                    loss_per_batch_item, batch_breakdown = loss_fn(samples, targets_reshaped, return_components=True)
                    # Accumulate breakdown
                    for component, value in batch_breakdown['weighted_losses'].items():
                        if component not in breakdown_accumulator:
                            breakdown_accumulator[component] = []
                        breakdown_accumulator[component].append(value)
                else:
                    loss_per_batch_item = loss_fn(samples, targets_reshaped) # Shape: [B]
                
                samples_for_return = samples.unsqueeze(2) # Reshape to [B, S, 1, D] for consistency
            else:
                # Multi-timestep output - autoregressive generation
                samples_sequence = generate_autoregressive_samples(model, inputs_flat, output_timesteps, output_dimension, n_samples, batch_size, use_consistent_ar_noise)
                
                # Compute loss explicitly per timestep and sum
                total_loss_for_batch = 0
                total_breakdown_for_batch = {}
                
                for t in range(output_timesteps):
                    samples_t = samples_sequence[:, :, t, :]  # Shape: [B, S, D]
                    targets_t = targets[:, t, :]              # Shape: [B, D]
                    
                    # Compute timestep loss with potential breakdown
                    if return_breakdown and has_breakdown:
                        loss_t_batch, breakdown_t = loss_fn(samples_t, targets_t, return_components=True)
                        # Accumulate breakdown across timesteps
                        for component, value in breakdown_t['weighted_losses'].items():
                            if component not in total_breakdown_for_batch:
                                total_breakdown_for_batch[component] = 0
                            total_breakdown_for_batch[component] += value
                    else:
                        loss_t_batch = loss_fn(samples_t, targets_t) 
                    
                    total_loss_for_batch += loss_t_batch
                
                loss_per_batch_item = total_loss_for_batch / output_timesteps
                samples_for_return = samples_sequence
                
                # Average breakdown over timesteps and accumulate
                if return_breakdown and has_breakdown:
                    for component, value in total_breakdown_for_batch.items():
                        avg_value = value / output_timesteps
                        if component not in breakdown_accumulator:
                            breakdown_accumulator[component] = []
                        breakdown_accumulator[component].append(avg_value)
            
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
    
    # Average breakdown if requested
    final_breakdown = {}
    if return_breakdown and breakdown_accumulator:
        for component, values in breakdown_accumulator.items():
            final_breakdown[component] = np.mean(values)
    
    # Return based on what was requested
    if return_predictions and return_breakdown:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return avg_loss, predictions, targets, final_breakdown
    elif return_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return avg_loss, predictions, targets
    elif return_breakdown:
        return avg_loss, final_breakdown
    else:
        return avg_loss
