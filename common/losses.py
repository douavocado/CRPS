import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
import numpy as np


class ProgressiveTrainingState:
    """
    Tracks training state for progressive loss activation and deactivation.
    
    This class monitors epoch progression, validation loss plateaus, and other
    training metrics to determine when progressive training loss instances should
    be activated or deactivated.
    """
    
    def __init__(self):
        self.current_epoch = 0
        self.validation_history = {}  # {instance_name: [loss_values]}
        self.validation_best = {}     # {instance_name: best_loss_value}
        self.validation_plateau_counts = {}  # {instance_name: epochs_since_improvement}
        self.validation_improvement_counts = {}  # {instance_name: epochs_since_last_plateau_reset}
        self.loss_activation_epochs = {}     # {instance_name: epoch_when_activated}
        self.loss_deactivation_epochs = {}   # {instance_name: epoch_when_deactivated}
        self.active_losses = set()           # Currently active loss instance names
        self.ever_activated_losses = set()   # Losses that have been activated at least once
        
        # New: Support for combined validation losses
        self.combined_validation_history = {}    # {combined_name: [loss_values]}
        self.combined_validation_best = {}       # {combined_name: best_loss_value}
        self.combined_validation_plateau_counts = {}  # {combined_name: epochs_since_improvement}
        self.combined_validation_definitions = {}     # {combined_name: {'type': 'sum', 'instances': [...]}}
        
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
        
    def update_validation_loss(self, instance_name: str, loss_value: float):
        """Update validation loss for a specific instance."""
        if instance_name not in self.validation_history:
            self.validation_history[instance_name] = []
            self.validation_best[instance_name] = float('inf')
            self.validation_plateau_counts[instance_name] = 0
            self.validation_improvement_counts[instance_name] = 0
            
        self.validation_history[instance_name].append(loss_value)
        
        # Check if this is a new best
        if loss_value < self.validation_best[instance_name]:
            self.validation_best[instance_name] = loss_value
            # Reset plateau count on new best
            self.validation_plateau_counts[instance_name] = 0
            # Track that we had an improvement
            self.validation_improvement_counts[instance_name] = 1
        else:
            # Increment plateau count - counts epochs without beating the best
            self.validation_plateau_counts[instance_name] += 1
            # Increment improvement count if we had a previous improvement
            if self.validation_improvement_counts[instance_name] > 0:
                self.validation_improvement_counts[instance_name] += 1
    
    def define_combined_validation_loss(self, combined_name: str, combination_type: str, instances: list, weights: list = None):
        """
        Define a combined validation loss from multiple instances.
        
        Args:
            combined_name: Unique name for the combined loss
            combination_type: 'sum' or 'weighted_sum'
            instances: List of instance names to combine
            weights: List of weights for weighted_sum (should match length of instances)
        """
        if combination_type not in ['sum', 'weighted_sum']:
            raise ValueError(f"Unknown combination_type: {combination_type}. Must be 'sum' or 'weighted_sum'")
        
        if combination_type == 'weighted_sum' and (weights is None or len(weights) != len(instances)):
            raise ValueError("For 'weighted_sum', weights must be provided and match the length of instances")
        
        if weights is None:
            weights = [1.0] * len(instances)
            
        self.combined_validation_definitions[combined_name] = {
            'type': combination_type,
            'instances': instances,
            'weights': weights
        }
        
        # Initialize tracking for this combined loss
        if combined_name not in self.combined_validation_history:
            self.combined_validation_history[combined_name] = []
            self.combined_validation_best[combined_name] = float('inf')
            self.combined_validation_plateau_counts[combined_name] = 0
    
    def update_combined_validation_losses(self, individual_losses: dict):
        """
        Update combined validation losses based on current individual losses.
        
        Args:
            individual_losses: Dictionary mapping instance names to loss values
        """
        for combined_name, definition in self.combined_validation_definitions.items():
            instances = definition['instances']
            weights = definition['weights']
            combination_type = definition['type']
            
            # Check if all required instances are available
            missing_instances = [inst for inst in instances if inst not in individual_losses]
            if missing_instances:
                continue  # Skip this combined loss if not all components are available
            
            # Compute combined loss
            if combination_type == 'sum':
                combined_value = sum(individual_losses[inst] for inst in instances)
            elif combination_type == 'weighted_sum':
                combined_value = sum(w * individual_losses[inst] for w, inst in zip(weights, instances))
            
            # Update tracking for this combined loss
            self.combined_validation_history[combined_name].append(combined_value)
            
            # Check if this is a new best
            if combined_value < self.combined_validation_best[combined_name]:
                self.combined_validation_best[combined_name] = combined_value
                # Reset plateau count on new best
                self.combined_validation_plateau_counts[combined_name] = 0
            else:
                # Increment plateau count
                self.combined_validation_plateau_counts[combined_name] += 1
                
    def get_plateau_count(self, instance_name: str) -> int:
        """Get number of epochs since validation improvement for an instance or combined loss."""
        # Check individual losses first
        if instance_name in self.validation_plateau_counts:
            return self.validation_plateau_counts[instance_name]
        # Check combined losses
        elif instance_name in self.combined_validation_plateau_counts:
            return self.combined_validation_plateau_counts[instance_name]
        else:
            return 0
        
    def mark_loss_activated(self, instance_name: str):
        """Mark a loss instance as activated and reset plateau counters for related combined losses."""
        if instance_name not in self.active_losses:
            self.active_losses.add(instance_name)
            self.ever_activated_losses.add(instance_name)
            self.loss_activation_epochs[instance_name] = self.current_epoch
            
            # Reset plateau counters for combined losses that include this instance
            for combined_name, definition in self.combined_validation_definitions.items():
                if instance_name in definition['instances']:
                    if combined_name in self.combined_validation_plateau_counts:
                        self.combined_validation_plateau_counts[combined_name] = 0
                        # Also reset the best value to current if history exists
                        if combined_name in self.combined_validation_history and self.combined_validation_history[combined_name]:
                            self.combined_validation_best[combined_name] = self.combined_validation_history[combined_name][-1]
            
    def mark_loss_deactivated(self, instance_name: str):
        """Mark a loss instance as deactivated. Once deactivated, it cannot be reactivated."""
        if instance_name in self.active_losses:
            self.active_losses.remove(instance_name)
            self.loss_deactivation_epochs[instance_name] = self.current_epoch
            
    def get_epochs_since_activation(self, instance_name: str) -> Optional[int]:
        """Get number of epochs since a loss was activated."""
        if instance_name not in self.loss_activation_epochs:
            return None
        return self.current_epoch - self.loss_activation_epochs[instance_name]
        
    def get_epochs_since_deactivation(self, instance_name: str) -> Optional[int]:
        """Get number of epochs since a loss was deactivated."""
        if instance_name not in self.loss_deactivation_epochs:
            return None
        return self.current_epoch - self.loss_deactivation_epochs[instance_name]
        
    def is_loss_active(self, instance_name: str) -> bool:
        """Check if a loss instance is currently active."""
        return instance_name in self.active_losses
        
    def was_loss_ever_active(self, instance_name: str) -> bool:
        """Check if a loss instance was ever activated."""
        return instance_name in self.ever_activated_losses
        
    def get_improvement_count(self, instance_name: str) -> int:
        """Get number of epochs since validation started improving after a plateau."""
        return self.validation_improvement_counts.get(instance_name, 0)
    



def evaluate_progressive_condition(condition: Dict[str, Any], state: ProgressiveTrainingState) -> bool:
    """
    Evaluate a single progressive training condition.
    
    Args:
        condition: Dictionary specifying the condition to evaluate
        state: Current training state
        
    Returns:
        True if condition is met, False otherwise
        
    Supported condition types for activation:
    - epoch_ge: Activate when current epoch >= specified value
    - epoch_lt: Activate when current epoch < specified value  
    - epoch_between: Activate when epoch is in range [start, end)
    - validation_plateau: Activate when validation hasn't improved for N epochs
    - validation_improved: Activate when validation has improved recently
    - loss_activated_for: Activate when another loss has been active for N epochs
    - loss_deactivated_for: Activate when another loss has been deactivated for N epochs
    - combined_validation_plateau: Activate when a sum/weighted sum of validation losses hasn't improved for N epochs
    - total_validation_plateau: Activate when total validation loss hasn't improved for N epochs
    """
    condition_type = condition.get('type')
    
    if condition_type == 'epoch_ge':
        threshold = condition.get('epoch', 0)
        return state.current_epoch >= threshold
        
    elif condition_type == 'epoch_lt':
        threshold = condition.get('epoch', float('inf'))
        return state.current_epoch < threshold
        
    elif condition_type == 'epoch_between':
        start = condition.get('start_epoch', 0)
        end = condition.get('end_epoch', float('inf'))
        return start <= state.current_epoch < end
        
    elif condition_type == 'validation_plateau':
        instance_name = condition.get('validation_loss_instance', 'default')
        plateau_epochs = condition.get('plateau_epochs', 5)
        return state.get_plateau_count(instance_name) >= plateau_epochs
        
    elif condition_type == 'validation_improved':
        instance_name = condition.get('validation_loss_instance', 'default')
        improvement_window = condition.get('improvement_window', 1)
        improvement_count = state.get_improvement_count(instance_name)
        return 0 < improvement_count <= improvement_window
        
    elif condition_type == 'loss_activated_for':
        target_loss = condition.get('target_loss_instance')
        epochs_since = condition.get('epochs_since_activation', 1)
        
        if target_loss is None:
            return False
            
        epochs_active = state.get_epochs_since_activation(target_loss)
        if epochs_active is None:
            return False
            
        return epochs_active >= epochs_since
        
    elif condition_type == 'loss_deactivated_for':
        target_loss = condition.get('target_loss_instance')
        epochs_since = condition.get('epochs_since_deactivation', 1)
        
        if target_loss is None:
            return False
            
        epochs_deactivated = state.get_epochs_since_deactivation(target_loss)
        if epochs_deactivated is None:
            return False
            
        return epochs_deactivated >= epochs_since
        
    elif condition_type == 'combined_validation_plateau':
        # Define and track a combined validation loss on-the-fly
        combined_name = condition.get('combined_name')
        combination_type = condition.get('combination_type', 'sum')
        instances = condition.get('validation_loss_instances', [])
        weights = condition.get('weights', None)
        plateau_epochs = condition.get('plateau_epochs', 5)
        
        if not combined_name or not instances:
            warnings.warn(f"combined_validation_plateau condition missing required fields: combined_name={combined_name}, instances={instances}")
            return False
            
        # Define the combined loss if not already defined
        if combined_name not in state.combined_validation_definitions:
            try:
                state.define_combined_validation_loss(combined_name, combination_type, instances, weights)
            except ValueError as e:
                warnings.warn(f"Failed to define combined validation loss '{combined_name}': {e}")
                return False
        
        # Check plateau count for the combined loss
        return state.get_plateau_count(combined_name) >= plateau_epochs
        
    elif condition_type == 'total_validation_plateau':
        # Plateau on total validation loss (sum of all active loss instances)
        plateau_epochs = condition.get('plateau_epochs', 5)
        return state.get_plateau_count('default') >= plateau_epochs
        
    else:
        warnings.warn(f"Unknown progressive condition type: {condition_type}")
        return True  # Default to active if unknown condition


def evaluate_progressive_conditions(conditions: List[Dict[str, Any]], state: ProgressiveTrainingState) -> bool:
    """
    Evaluate multiple progressive training conditions with AND logic.
    
    Args:
        conditions: List of condition dictionaries
        state: Current training state
        
    Returns:
        True if ALL conditions are met, False otherwise
    """
    if not conditions:
        return True  # No conditions means always active
        
    return all(evaluate_progressive_condition(cond, state) for cond in conditions)


def _format_condition_summary(conditions: List[Dict[str, Any]]) -> str:
    """Format a list of conditions into a human-readable summary."""
    condition_summary = []
    for condition in conditions:
        condition_type = condition.get('type')
        
        if condition_type == 'epoch_ge':
            condition_summary.append(f"epoch >= {condition.get('epoch')}")
        elif condition_type == 'epoch_lt':
            condition_summary.append(f"epoch < {condition.get('epoch')}")
        elif condition_type == 'epoch_between':
            start = condition.get('start_epoch', 0)
            end = condition.get('end_epoch', 'inf')
            condition_summary.append(f"{start} <= epoch < {end}")
        elif condition_type == 'validation_plateau':
            val_instance = condition.get('validation_loss_instance', 'default')
            plateau_epochs = condition.get('plateau_epochs', 5)
            condition_summary.append(f"{val_instance} plateau >= {plateau_epochs} epochs")
        elif condition_type == 'validation_improved':
            val_instance = condition.get('validation_loss_instance', 'default')
            window = condition.get('improvement_window', 1)
            condition_summary.append(f"{val_instance} improved within {window} epochs")
        elif condition_type == 'loss_activated_for':
            target_loss = condition.get('target_loss_instance')
            epochs_since = condition.get('epochs_since_activation', 1)
            condition_summary.append(f"{target_loss} active for >= {epochs_since} epochs")
        elif condition_type == 'loss_deactivated_for':
            target_loss = condition.get('target_loss_instance')
            epochs_since = condition.get('epochs_since_deactivation', 1)
            condition_summary.append(f"{target_loss} deactivated for >= {epochs_since} epochs")
        elif condition_type == 'combined_validation_plateau':
            combined_name = condition.get('combined_name', 'unknown')
            plateau_epochs = condition.get('plateau_epochs', 5)
            instances = condition.get('validation_loss_instances', [])
            combination_type = condition.get('combination_type', 'sum')
            condition_summary.append(f"combined '{combined_name}' ({combination_type} of {instances}) plateau >= {plateau_epochs} epochs")
        elif condition_type == 'total_validation_plateau':
            plateau_epochs = condition.get('plateau_epochs', 5)
            condition_summary.append(f"total validation plateau >= {plateau_epochs} epochs")
        else:
            condition_summary.append(f"unknown condition: {condition_type}")
    
    return " AND ".join(condition_summary) if condition_summary else "no conditions"


def crps_loss_general(yps, y):
    """ Calculates the CRPS loss. If  y has multiple dimensions, it calculates the CRPS loss for each dimension.
    If y has a single dimension, it calculates the CRPS loss for that dimension.

    Args:
        yps: Tensor of predicted samples [batch x num_samples x dims]
        y: Target values [batch x dims]

    Returns:
        CRPS loss value per batch element [batch x dims]
    """
    num_samples = yps.shape[1]
    if y.dim() == 2:
        y = y.unsqueeze(1)
    mrank = torch.argsort(torch.argsort(yps, dim=1), dim=1)
    return ((2 / (num_samples * (num_samples - 1))) * (yps - y) * (((num_samples - 1) * (y < yps)) - mrank)).sum(
        axis=1
    )


def energy_score_loss(y_pred_samples, y_true, norm_dim=False):
    """
    Compute energy score loss for multivariate samples using biased Monte Carlo estimation.
    
    Energy Score = 2 * E[||Y - X||] - E[||X - X'||]
    where Y is true value, X and X' are independent samples from prediction
    
    Uses unbiased Monte Carlo estimate for E[||X - X'||] (includes diagonal terms).
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        norm_dim: Whether to normalise the energy score by the output dimension
    
    Returns:
        Energy score loss (lower is better)
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    
    # Expand y_true to match samples shape: [batch_size, n_samples, output_dim]
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples, -1)
    
    # First term: 2 * E[||Y - X||]
    # Compute L2 norm between true values and samples
    diff_true_pred = y_true_expanded - y_pred_samples  # [batch_size, n_samples, output_dim]
    norm_true_pred = torch.norm(diff_true_pred, dim=2)  # [batch_size, n_samples]
    first_term = 2 * torch.mean(norm_true_pred, dim=1)  # [batch_size]
    
    # Second term: E[||X - X'||]
    # Compute pairwise distances between samples
    # Reshape for broadcasting: [batch_size, n_samples, 1, output_dim] and [batch_size, 1, n_samples, output_dim]
    samples_i = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    samples_j = y_pred_samples.unsqueeze(1)  # [batch_size, 1, n_samples, output_dim]
    
    # Compute pairwise differences and norms
    diff_samples = samples_i - samples_j  # [batch_size, n_samples, n_samples, output_dim]
    norm_samples = torch.norm(diff_samples, dim=3)  # [batch_size, n_samples, n_samples]
    
    # Biased Monte Carlo estimate: include all pairs (including diagonal)
    second_term = torch.mean(norm_samples, dim=(1, 2))  # [batch_size]
    # Now multiply by n_samples/(n_samples-1) to get unbiased estimate
    second_term = second_term * (n_samples / (n_samples - 1))
    
    # Energy score
    energy_scores = first_term - second_term  # [batch_size]
    
    if norm_dim:
        energy_scores = energy_scores / output_dim
    return energy_scores


def variogram_score_loss(y_pred_samples, y_true, weights=None, p=1.0, coords=None):
    """
    Compute variogram score loss for multivariate samples.
    
    The variogram score is defined as:
    S^{(p)}(P,y) = sum_{i,j}^d w_{ij} * (|y_i - y_j|^p - E_{X~P}|X_i - X_j|^p)^2
    
    where:
    - P is the predicted distribution (represented by samples)
    - y is the vector of true targets
    - d is the dimensionality of y and X
    - w_{ij} are the weights
    - X is a sample from the predicted distribution P
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        weights: Weight matrix [output_dim, output_dim] or None for uniform weights
        p: Power parameter (default: 1.0)
        coords: Coordinates for each dimension [output_dim, coord_dim] for spatial weighting
    
    Returns:
        Variogram score loss (lower is better) [batch_size]
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    device = y_pred_samples.device
    
    # Create weights if not provided
    if weights is None:
        if coords is not None:
            # Create weights inversely proportional to spatial distance
            # coords shape: [output_dim, coord_dim]
            coord_diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # [output_dim, output_dim, coord_dim]
            spatial_distances = torch.norm(coord_diffs, dim=2)  # [output_dim, output_dim]
            # Add small epsilon to avoid division by zero and handle diagonal
            epsilon = 1e-6
            weights = 1.0 / (spatial_distances + epsilon)
            # Set diagonal to 0 (no self-comparison)
            weights.fill_diagonal_(0.0)
        else:
            # Uniform weights (excluding diagonal)
            weights = torch.ones(output_dim, output_dim, device=device)
            weights.fill_diagonal_(0.0)
    
    # Ensure weights are on the correct device
    weights = weights.to(device)
    
    # Compute pairwise differences in true values
    # y_true: [batch_size, output_dim]
    y_true_expanded_i = y_true.unsqueeze(2)  # [batch_size, output_dim, 1]
    y_true_expanded_j = y_true.unsqueeze(1)  # [batch_size, 1, output_dim]
    true_pairwise_diffs = torch.abs(y_true_expanded_i - y_true_expanded_j)  # [batch_size, output_dim, output_dim]
    true_pairwise_diffs_p = torch.pow(true_pairwise_diffs, p)  # [batch_size, output_dim, output_dim]
    
    # Compute expected pairwise differences in predicted samples
    # y_pred_samples: [batch_size, n_samples, output_dim]
    pred_expanded_i = y_pred_samples.unsqueeze(3)  # [batch_size, n_samples, output_dim, 1]
    pred_expanded_j = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    pred_pairwise_diffs = torch.abs(pred_expanded_i - pred_expanded_j)  # [batch_size, n_samples, output_dim, output_dim]
    pred_pairwise_diffs_p = torch.pow(pred_pairwise_diffs, p)  # [batch_size, n_samples, output_dim, output_dim]
    
    # Take expectation over samples
    expected_pred_diffs = torch.mean(pred_pairwise_diffs_p, dim=1)  # [batch_size, output_dim, output_dim]
    
    # Compute the squared differences
    squared_diffs = torch.pow(true_pairwise_diffs_p - expected_pred_diffs, 2)  # [batch_size, output_dim, output_dim]
    
    # Apply weights and sum
    weighted_squared_diffs = weights.unsqueeze(0) * squared_diffs  # [batch_size, output_dim, output_dim]
    variogram_scores = torch.sum(weighted_squared_diffs, dim=(1, 2))  # [batch_size]
    
    return variogram_scores

def kernel_score_loss(y_pred_samples, y_true, kernel_type='gaussian', **kwargs):
    """
    Compute kernel score loss for multivariate samples.
    
    The kernel score is defined as:
    S(P,y) = E_{X,X'~P} K(X,X') - 2 * E_{X~P} K(X,y)
    where K is a kernel function.

    Uses biased Monte Carlo estimate for E[K(X,X')].

    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        kernel_type: Type of kernel to use ('gaussian' or 'matern')
        **kwargs: Additional keyword arguments for the kernel function:
            - sigma: Kernel bandwidth parameter (default: 1.0)
            - nu: Smoothness parameter for Matérn kernel (default: 1.5)

    Returns:
        Kernel score loss (lower is better) [batch_size]
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    device = y_pred_samples.device
    
    # Extract kernel parameters
    sigma = kwargs.get('sigma', 1.0)
    nu = kwargs.get('nu', 1.5)  # Only used for Matérn kernel
    
    def gaussian_kernel(x1, x2, sigma=1.0):
        """Gaussian (RBF) kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.exp(-squared_dist / (2 * sigma ** 2))
    
    def matern_kernel(x1, x2, sigma=1.0, nu=1.5):
        """Matérn kernel with smoothness parameter nu"""
        dist = torch.norm(x1 - x2, dim=-1)
        scaled_dist = torch.sqrt(2 * nu) * dist / sigma
        
        if nu == 0.5:
            # Exponential kernel
            return torch.exp(-scaled_dist)
        elif nu == 1.5:
            # Matérn 3/2
            return (1 + scaled_dist) * torch.exp(-scaled_dist)
        elif nu == 2.5:
            # Matérn 5/2
            return (1 + scaled_dist + (scaled_dist ** 2) / 3) * torch.exp(-scaled_dist)
        else:
            # General Matérn kernel (more computationally expensive)
            from math import gamma
            import torch.special
            
            # Handle the case where distance is 0
            zero_mask = (scaled_dist == 0)
            scaled_dist = torch.where(zero_mask, torch.tensor(1e-8, device=device), scaled_dist)
            
            coeff = (2 ** (1 - nu)) / gamma(nu)
            bessel_term = torch.special.modified_bessel_k(nu, scaled_dist)
            kernel_val = coeff * (scaled_dist ** nu) * bessel_term
            
            # Set kernel value to 1 where distance was originally 0
            kernel_val = torch.where(zero_mask, torch.tensor(1.0, device=device), kernel_val)
            return kernel_val
    
    # Select kernel function
    if kernel_type == 'gaussian':
        kernel_fn = lambda x1, x2: gaussian_kernel(x1, x2, sigma=sigma)
    elif kernel_type == 'matern':
        kernel_fn = lambda x1, x2: matern_kernel(x1, x2, sigma=sigma, nu=nu)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available options: 'gaussian', 'matern'")
    
    # First term: E_{X,X'~P} K(X,X') using biased Monte Carlo estimate
    # Compute all pairwise kernel values between samples (including diagonal)
    samples_expanded_i = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    samples_expanded_j = y_pred_samples.unsqueeze(1)  # [batch_size, 1, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_i_flat = samples_expanded_i.expand(-1, -1, n_samples, -1).reshape(-1, output_dim)
    samples_j_flat = samples_expanded_j.expand(-1, n_samples, -1, -1).reshape(-1, output_dim)
    
    # Compute kernel values
    kernel_values_flat = kernel_fn(samples_i_flat, samples_j_flat)  # [batch_size * n_samples * n_samples]
    kernel_values = kernel_values_flat.reshape(batch_size, n_samples, n_samples)
    
    # Biased Monte Carlo estimate: include all pairs (including diagonal)
    first_term = torch.mean(kernel_values, dim=(1, 2))  # [batch_size]
    
    # Second term: 2 * E_{X~P} K(X,y)
    # Expand y_true to match samples shape
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_flat = y_pred_samples.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    y_true_flat = y_true_expanded.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    
    # Compute kernel values between samples and true values
    kernel_true_flat = kernel_fn(samples_flat, y_true_flat)  # [batch_size * n_samples]
    kernel_true = kernel_true_flat.reshape(batch_size, n_samples)  # [batch_size, n_samples]
    
    # Monte Carlo estimate
    second_term = 2 * torch.mean(kernel_true, dim=1)  # [batch_size]
    
    # Kernel score: E[K(X,X')] - 2*E[K(X,y)]
    kernel_scores = first_term - second_term  # [batch_size]
    
    return kernel_scores

def mmd_loss(y_pred_samples, y_true, kernel_type='gaussian', **kwargs):
    """
    Compute Maximum Mean Discrepancy (MMD) loss for multivariate samples.
    
    The MMD between predicted distribution P and true delta distribution δ_y is:
    MMD²(P, δ_y) = E_{X,X'~P}[k(X,X')] - 2*E_{X~P}[k(X,y)] + k(y,y)
    
    Since k(y,y) = 1 for normalised kernels, this simplifies to:
    MMD²(P, δ_y) = E_{X,X'~P}[k(X,X')] - 2*E_{X~P}[k(X,y)] + 1
    
    Uses biased Monte Carlo estimate for E[k(X,X')].

    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        kernel_type: Type of kernel to use ('gaussian', 'laplacian', 'polynomial', or 'rq')
        **kwargs: Additional keyword arguments for the kernel function:
            - sigma: Kernel bandwidth parameter (default: 1.0)
            - gamma: Inverse bandwidth for Laplacian kernel (default: 1.0)
            - degree: Degree for polynomial kernel (default: 2)
            - alpha: Scale parameter for rational quadratic kernel (default: 1.0)

    Returns:
        MMD loss (lower is better) [batch_size]
    """
    batch_size, n_samples, output_dim = y_pred_samples.shape
    device = y_pred_samples.device
    
    # Extract kernel parameters
    sigma = kwargs.get('sigma', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    degree = kwargs.get('degree', 2)
    alpha = kwargs.get('alpha', 1.0)
    
    def gaussian_kernel(x1, x2, sigma=1.0):
        """Gaussian (RBF) kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.exp(-squared_dist / (2 * sigma ** 2))
    
    def laplacian_kernel(x1, x2, gamma=1.0):
        """Laplacian kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||)"""
        dist = torch.norm(x1 - x2, dim=-1)
        return torch.exp(-gamma * dist)
    
    def polynomial_kernel(x1, x2, degree=2):
        """Polynomial kernel: K(x1, x2) = (1 + <x1, x2>)^degree"""
        dot_product = torch.sum(x1 * x2, dim=-1)
        return torch.pow(1 + dot_product, degree)
    
    def rational_quadratic_kernel(x1, x2, alpha=1.0, sigma=1.0):
        """Rational quadratic kernel: K(x1, x2) = (1 + ||x1 - x2||^2 / (2*alpha*sigma^2))^(-alpha)"""
        squared_dist = torch.sum((x1 - x2) ** 2, dim=-1)
        return torch.pow(1 + squared_dist / (2 * alpha * sigma ** 2), -alpha)
    
    # Select kernel function
    if kernel_type == 'gaussian':
        kernel_fn = lambda x1, x2: gaussian_kernel(x1, x2, sigma=sigma)
    elif kernel_type == 'laplacian':
        kernel_fn = lambda x1, x2: laplacian_kernel(x1, x2, gamma=gamma)
    elif kernel_type == 'polynomial':
        kernel_fn = lambda x1, x2: polynomial_kernel(x1, x2, degree=degree)
    elif kernel_type == 'rq' or kernel_type == 'rational_quadratic':
        kernel_fn = lambda x1, x2: rational_quadratic_kernel(x1, x2, alpha=alpha, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available options: 'gaussian', 'laplacian', 'polynomial', 'rq'")
    
    # First term: E_{X,X'~P}[k(X,X')] using biased Monte Carlo estimate
    # Compute all pairwise kernel values between samples (including diagonal)
    samples_expanded_i = y_pred_samples.unsqueeze(2)  # [batch_size, n_samples, 1, output_dim]
    samples_expanded_j = y_pred_samples.unsqueeze(1)  # [batch_size, 1, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_i_flat = samples_expanded_i.expand(-1, -1, n_samples, -1).reshape(-1, output_dim)
    samples_j_flat = samples_expanded_j.expand(-1, n_samples, -1, -1).reshape(-1, output_dim)
    
    # Compute kernel values
    kernel_values_flat = kernel_fn(samples_i_flat, samples_j_flat)  # [batch_size * n_samples * n_samples]
    kernel_values = kernel_values_flat.reshape(batch_size, n_samples, n_samples)
    
    # Biased Monte Carlo estimate: include all pairs (including diagonal)
    first_term = torch.mean(kernel_values, dim=(1, 2))  # [batch_size]
    
    # Second term: 2 * E_{X~P}[k(X,y)]
    # Expand y_true to match samples shape
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, output_dim]
    
    # Reshape for kernel computation
    samples_flat = y_pred_samples.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    y_true_flat = y_true_expanded.reshape(-1, output_dim)  # [batch_size * n_samples, output_dim]
    
    # Compute kernel values between samples and true values
    kernel_true_flat = kernel_fn(samples_flat, y_true_flat)  # [batch_size * n_samples]
    kernel_true = kernel_true_flat.reshape(batch_size, n_samples)  # [batch_size, n_samples]
    
    # Monte Carlo estimate
    second_term = 2 * torch.mean(kernel_true, dim=1)  # [batch_size]
    
    # Third term: k(y,y) = 1 for normalised kernels
    third_term = 1.0
    
    # MMD²: E[k(X,X')] - 2*E[k(X,y)] + k(y,y)
    mmd_squared = first_term - second_term + third_term  # [batch_size]
    
    # Return MMD (take square root, but ensure non-negative first)
    mmd_squared = torch.clamp(mmd_squared, min=0.0)  # Ensure non-negative due to numerical errors
    mmd_loss_values = torch.sqrt(mmd_squared)  # [batch_size]
    
    return mmd_loss_values

def spatial_pool_2d(data, pool_type='mean', kernel_size=2, stride=None, spatial_shape=None):
    """
    Apply spatial pooling to data representing a 2D spatial grid.
    
    Args:
        data: Input tensor [batch_size, n_samples, height*width] or [batch_size, height*width]
        pool_type: Type of pooling ('mean', 'max', 'min', 'median', 'percentile')
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride for pooling (defaults to kernel_size if None)
        spatial_shape: Tuple (height, width) for reshaping flattened spatial data
        
    Returns:
        Pooled data with reduced spatial dimensions
    """
    # Handle kernel_size and stride
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    original_shape = data.shape
    batch_size = original_shape[0]
    
    # Determine if we have samples dimension
    has_samples = len(original_shape) == 3
    if has_samples:
        n_samples = original_shape[1]
        spatial_dim = original_shape[2]
    else:
        spatial_dim = original_shape[1]
    
    # Infer spatial shape if not provided
    if spatial_shape is None:
        # Assume square spatial grid
        spatial_size = int(spatial_dim ** 0.5)
        if spatial_size * spatial_size != spatial_dim:
            raise ValueError(f"Cannot infer square spatial shape from dimension {spatial_dim}. Please provide spatial_shape.")
        spatial_shape = (spatial_size, spatial_size)
    
    height, width = spatial_shape
    if height * width != spatial_dim:
        raise ValueError(f"Spatial shape {spatial_shape} doesn't match spatial dimension {spatial_dim}")
    
    # Reshape to spatial format
    if has_samples:
        # [batch_size, n_samples, height*width] -> [batch_size*n_samples, 1, height, width]
        data_spatial = data.view(batch_size * n_samples, 1, height, width)
    else:
        # [batch_size, height*width] -> [batch_size, 1, height, width]
        data_spatial = data.view(batch_size, 1, height, width)
    
    # Apply pooling
    if pool_type == 'mean':
        pooled = F.avg_pool2d(data_spatial, kernel_size=kernel_size, stride=stride)
    elif pool_type == 'max':
        pooled = F.max_pool2d(data_spatial, kernel_size=kernel_size, stride=stride)
    elif pool_type == 'min':
        # PyTorch doesn't have min_pool2d, so we implement it using -max_pool2d(-x)
        pooled = -F.max_pool2d(-data_spatial, kernel_size=kernel_size, stride=stride)
    elif pool_type == 'median':
        # For median, we need to use unfold and compute median manually
        unfolded = F.unfold(data_spatial, kernel_size=kernel_size, stride=stride)
        # unfolded: [batch*samples, kernel_h*kernel_w, num_windows]
        pooled_values, _ = torch.median(unfolded, dim=1)
        # Reshape back to spatial format
        out_height = (height - kernel_size[0]) // stride[0] + 1
        out_width = (width - kernel_size[1]) // stride[1] + 1
        pooled = pooled_values.view(-1, 1, out_height, out_width)
    elif pool_type.startswith('percentile'):
        # Extract percentile value (e.g., 'percentile_75' -> 75)
        try:
            percentile = float(pool_type.split('_')[1])
            if not 0 <= percentile <= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        except (IndexError, ValueError):
            raise ValueError(f"Invalid percentile specification: {pool_type}. Use format 'percentile_XX' where XX is 0-100")
        
        # Use unfold and compute percentile
        unfolded = F.unfold(data_spatial, kernel_size=kernel_size, stride=stride)
        pooled_values = torch.quantile(unfolded, q=percentile/100.0, dim=1)
        # Reshape back to spatial format
        out_height = (height - kernel_size[0]) // stride[0] + 1
        out_width = (width - kernel_size[1]) // stride[1] + 1
        pooled = pooled_values.view(-1, 1, out_height, out_width)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}. Available options: 'mean', 'max', 'min', 'median', 'percentile_XX'")
    
    # Flatten spatial dimensions back
    pooled_flat = pooled.view(pooled.shape[0], -1)  # [batch*samples, pooled_spatial_dim]
    
    # Reshape back to original format
    if has_samples:
        # [batch*samples, pooled_spatial_dim] -> [batch_size, n_samples, pooled_spatial_dim]
        pooled_output = pooled_flat.view(batch_size, n_samples, -1)
    else:
        # [batch, pooled_spatial_dim] -> [batch_size, pooled_spatial_dim]
        pooled_output = pooled_flat
    
    return pooled_output

def apply_spatial_pooling(y_pred_samples, y_true, pooling_config):
    """
    Apply spatial pooling to both predictions and true values.
    
    Args:
        y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
        y_true: True values [batch_size, output_dim]
        pooling_config: Dictionary with pooling configuration:
            - enabled: bool, whether to apply pooling
            - pool_type: str, type of pooling operation
            - kernel_size: int or tuple, pooling window size
            - stride: int or tuple, pooling stride (optional)
            - spatial_shape: tuple, (height, width) for spatial reshaping
    
    Returns:
        Tuple of (pooled_y_pred_samples, pooled_y_true)
    """
    if not pooling_config.get('enabled', False):
        return y_pred_samples, y_true
    
    pool_type = pooling_config.get('pool_type', 'mean')
    kernel_size = pooling_config.get('kernel_size', 2)
    stride = pooling_config.get('stride', None)
    spatial_shape = pooling_config.get('spatial_shape', None)
    
    # Apply pooling to predictions
    pooled_y_pred = spatial_pool_2d(
        y_pred_samples, 
        pool_type=pool_type, 
        kernel_size=kernel_size, 
        stride=stride,
        spatial_shape=spatial_shape
    )
    
    # Apply pooling to true values
    pooled_y_true = spatial_pool_2d(
        y_true, 
        pool_type=pool_type, 
        kernel_size=kernel_size, 
        stride=stride,
        spatial_shape=spatial_shape
    )
    
    return pooled_y_pred, pooled_y_true

def create_loss_function(config):
    """
    Create a combined loss function from a configuration dictionary.
    
    Args:
        config: Dictionary with the following keys:
            - losses: List of loss function strings (must be non-empty)
                     Valid options: 'crps_loss_general', 'energy_score_loss', 'variogram_score_loss', 'kernel_score_loss', 'mmd_loss'
            - loss_function_args: Optional dictionary of dictionaries containing arguments for each loss function
                                 Format: {loss_function_name: {arg1: value1, arg2: value2, ...}}
            - coefficients: Optional dictionary of multipliers for each loss function
                           Format: {loss_function_name: coefficient_value}
                           If not provided, defaults to 1.0 for all loss functions
            - spatial_pooling: Optional dictionary for spatial pooling configuration:
                              - enabled: bool, whether to apply spatial pooling (default: False)
                              - pool_type: str, pooling operation ('mean', 'max', 'min', 'median', 'percentile_XX')
                              - kernel_size: int or tuple, pooling window size (default: 2)
                              - stride: int or tuple, pooling stride (default: same as kernel_size)
                              - spatial_shape: tuple (height, width), spatial dimensions for reshaping
    
    Returns:
        A PyTorch-compatible loss function that can be used with autograd
    
    Example:
        config = {
            'losses': ['energy_score_loss', 'variogram_score_loss'],
            'loss_function_args': {
                'energy_score_loss': {'norm_dim': True},
                'variogram_score_loss': {'p': 2.0}
            },
            'coefficients': {
                'energy_score_loss': 1.0,
                'variogram_score_loss': 0.5
            },
            'spatial_pooling': {
                'enabled': True,
                'pool_type': 'mean',
                'kernel_size': 2,
                'spatial_shape': (64, 64)
            }
        }
        loss_fn = create_loss_function(config)
        loss = loss_fn(y_pred_samples, y_true)
    """
    # Available loss functions
    available_losses = {
        'crps_loss_general': crps_loss_general,
        'energy_score_loss': energy_score_loss,
        'variogram_score_loss': variogram_score_loss,
        'kernel_score_loss': kernel_score_loss,
        'mmd_loss': mmd_loss
    }
    
    # Validate configuration
    if 'losses' not in config or not config['losses']:
        raise ValueError("Config must contain a non-empty 'losses' list")
    
    losses = config['losses']
    loss_function_args = config.get('loss_function_args', {})
    coefficients = config.get('coefficients', {})
    spatial_pooling_config = config.get('spatial_pooling', {'enabled': False})
    
    # Validate loss function names
    for loss_name in losses:
        if loss_name not in available_losses:
            raise ValueError(f"Unknown loss function: {loss_name}. Available options: {list(available_losses.keys())}")
    
    # Set default coefficients to 1.0 if not provided
    for loss_name in losses:
        if loss_name not in coefficients:
            coefficients[loss_name] = 1.0
    
    def combined_loss_function(y_pred_samples, y_true, return_components=False, **kwargs):
        """
        Combined loss function that computes weighted sum of specified losses.
        
        Args:
            y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
            y_true: True values [batch_size, output_dim]
            return_components: If True, return individual loss components along with total
            **kwargs: Additional keyword arguments that can be passed to individual loss functions
        
        Returns:
            If return_components=False: Combined loss value [batch_size]
            If return_components=True: (total_loss, breakdown_dict)
        """
        # Apply spatial pooling if enabled
        if spatial_pooling_config.get('enabled', False):
            y_pred_samples, y_true = apply_spatial_pooling(y_pred_samples, y_true, spatial_pooling_config)
        
        total_loss = None
        individual_losses = {}
        weighted_losses = {}
        
        for loss_name in losses:
            loss_fn = available_losses[loss_name]
            coefficient = coefficients[loss_name]
            
            # Get arguments for this specific loss function
            loss_args = loss_function_args.get(loss_name, {})
            
            # Merge with any additional kwargs passed to the combined function
            merged_args = {**loss_args, **kwargs}
            
            # Compute loss
            try:
                loss_value = loss_fn(y_pred_samples, y_true, **merged_args)
            except TypeError as e:
                # Handle case where loss function doesn't accept some arguments
                # Try with only the arguments it expects
                import inspect
                sig = inspect.signature(loss_fn)
                valid_args = {k: v for k, v in merged_args.items() if k in sig.parameters}
                loss_value = loss_fn(y_pred_samples, y_true, **valid_args)
            
            # Ensure loss_value is per-batch (reduce any extra dimensions)
            if loss_value.dim() > 1:
                # Sum over all dimensions except batch dimension
                loss_value = loss_value.sum(dim=tuple(range(1, loss_value.dim())))
            
            # Store individual loss (before weighting)
            individual_losses[loss_name] = loss_value.mean().item() if hasattr(loss_value, 'mean') else loss_value
            
            # Apply coefficient
            weighted_loss = coefficient * loss_value
            weighted_losses[loss_name] = weighted_loss.mean().item() if hasattr(weighted_loss, 'mean') else weighted_loss
            
            # Add to total loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        
        if return_components:
            # Return both individual (unweighted) and weighted losses for detailed breakdown
            breakdown = {
                'individual_losses': individual_losses,
                'weighted_losses': weighted_losses,
                'active_losses': active_losses,
                'total_loss': total_loss.mean().item() if hasattr(total_loss, 'mean') else total_loss
            }
            return total_loss, breakdown
        else:
            return total_loss
    
    # Store metadata for breakdown reporting
    combined_loss_function._is_multi_instance = len(losses) > 1
    combined_loss_function._losses = losses
    combined_loss_function._coefficients = coefficients
    
    return combined_loss_function


def create_multi_instance_loss_function(config):
    """
    Create a combined loss function that supports multiple instances of the same loss function
    with different spatial pooling configurations and progressive training activation.
    
    Args:
        config: Dictionary with the following structure:
            - loss_instances: List of loss instance dictionaries, each containing:
                - name: str, unique identifier for this loss instance
                - loss_function: str, name of the loss function to use
                - coefficient: float, multiplier for this loss (default: 1.0)
                - spatial_pooling: dict, spatial pooling config for this instance (optional)
                - loss_args: dict, arguments specific to this loss function (optional)
                - progressive_training: dict, progressive training config (optional):
                    - activation_conditions: list of condition dicts, ALL must be true for activation
                    - deactivation_conditions: list of condition dicts, ALL must be true for deactivation
                    - conditions: (legacy) same as activation_conditions
                        Each condition can be:
                        - {'type': 'epoch_ge', 'epoch': N} - when epoch >= N
                        - {'type': 'epoch_lt', 'epoch': N} - when epoch < N
                        - {'type': 'epoch_between', 'start_epoch': N, 'end_epoch': M} - when N <= epoch < M
                        - {'type': 'validation_plateau', 'validation_loss_instance': 'name', 'plateau_epochs': N}
                        - {'type': 'validation_improved', 'validation_loss_instance': 'name', 'improvement_window': N}
                        - {'type': 'loss_activated_for', 'target_loss_instance': 'name', 'epochs_since_activation': N}
                        - {'type': 'loss_deactivated_for', 'target_loss_instance': 'name', 'epochs_since_deactivation': N}
    
    Returns:
        A PyTorch-compatible loss function that can be used with autograd
    
    Example:
        config = {
            'loss_instances': [
                {
                    'name': 'energy_coarse',
                    'loss_function': 'energy_score_loss',
                    'coefficient': 1.0,
                    'spatial_pooling': {
                        'enabled': True,
                        'pool_type': 'mean',
                        'kernel_size': 4,
                        'spatial_shape': (64, 64)
                    },
                    'loss_args': {'norm_dim': True}
                },
                {
                    'name': 'energy_fine',
                    'loss_function': 'energy_score_loss',
                    'coefficient': 0.5,
                    'progressive_training': {
                        'conditions': [
                            {'type': 'epoch_ge', 'epoch': 10},
                            {'type': 'validation_plateau', 'validation_loss_instance': 'energy_coarse', 'plateau_epochs': 3}
                        ]
                    },
                    'spatial_pooling': {
                        'enabled': True,
                        'pool_type': 'max',
                        'kernel_size': 2,
                        'spatial_shape': (64, 64)
                    },
                    'loss_args': {'norm_dim': False}
                },
                {
                    'name': 'crps_full',
                    'loss_function': 'crps_loss_general',
                    'coefficient': 0.3,
                    'progressive_training': {
                        'conditions': [
                            {'type': 'loss_activated_for', 'target_loss_instance': 'energy_fine', 'epochs_since_activation': 5}
                        ]
                    },
                    'spatial_pooling': {'enabled': False}
                }
            ]
        }
        loss_fn = create_multi_instance_loss_function(config)
        loss = loss_fn(y_pred_samples, y_true, progressive_state=state)
    """
    # Available loss functions
    available_losses = {
        'crps_loss_general': crps_loss_general,
        'energy_score_loss': energy_score_loss,
        'variogram_score_loss': variogram_score_loss,
        'kernel_score_loss': kernel_score_loss,
        'mmd_loss': mmd_loss
    }
    
    # Validate configuration
    if 'loss_instances' not in config or not config['loss_instances']:
        raise ValueError("Config must contain a non-empty 'loss_instances' list")
    
    loss_instances = config['loss_instances']
    
    # Validate loss instances
    instance_names = set()
    for i, instance in enumerate(loss_instances):
        if 'name' not in instance:
            raise ValueError(f"Loss instance {i} must have a 'name' field")
        if 'loss_function' not in instance:
            raise ValueError(f"Loss instance {i} must have a 'loss_function' field")
        
        name = instance['name']
        loss_function = instance['loss_function']
        
        if name in instance_names:
            raise ValueError(f"Duplicate loss instance name: '{name}'")
        instance_names.add(name)
        
        if loss_function not in available_losses:
            raise ValueError(f"Unknown loss function in instance '{name}': {loss_function}. "
                           f"Available options: {list(available_losses.keys())}")
    
    def multi_instance_loss_function(y_pred_samples, y_true, return_components=False, progressive_state=None, **kwargs):
        """
        Multi-instance loss function that computes weighted sum of specified loss instances
        with progressive training support.
        
        Args:
            y_pred_samples: Predicted samples [batch_size, n_samples, output_dim]
            y_true: True values [batch_size, output_dim]
            return_components: If True, return individual loss components along with total
            progressive_state: ProgressiveTrainingState instance for tracking activation conditions
            **kwargs: Additional keyword arguments that can be passed to individual loss functions
        
        Returns:
            If return_components=False: Combined loss value [batch_size]
            If return_components=True: (total_loss, individual_losses_dict)
        """
        total_loss = None
        individual_losses = {}
        weighted_losses = {}
        active_losses = {}  # Track which losses are active due to progressive training
        
        for instance in loss_instances:
            name = instance['name']
            loss_function_name = instance['loss_function']
            coefficient = instance.get('coefficient', 1.0)
            spatial_pooling_config = instance.get('spatial_pooling', {'enabled': False})
            loss_args = instance.get('loss_args', {})
            progressive_config = instance.get('progressive_training', None)
            
            # Check if this loss instance should be active
            is_active = True
            if progressive_config is not None and progressive_state is not None:
                # Check activation conditions (support both 'conditions' and 'activation_conditions')
                activation_conditions = progressive_config.get('activation_conditions', 
                                                             progressive_config.get('conditions', []))
                deactivation_conditions = progressive_config.get('deactivation_conditions', [])
                
                # Determine if loss should be active
                if not progressive_state.was_loss_ever_active(name):
                    # Loss has never been activated - check activation conditions
                    is_active = evaluate_progressive_conditions(activation_conditions, progressive_state)
                    
                    if is_active:
                        progressive_state.mark_loss_activated(name)
                        if kwargs.get('verbose', False):
                            condition_summary = _format_condition_summary(activation_conditions)
                            print(f"Progressive Training: Activated '{name}' at epoch {progressive_state.current_epoch}")
                            print(f"Activation conditions met: {condition_summary}")
                            # Report any plateau counter resets
                            for combined_name, definition in progressive_state.combined_validation_definitions.items():
                                if name in definition['instances']:
                                    print(f"  Reset plateau counter for combined loss '{combined_name}'")
                else:
                    # Loss has been activated before - check if it should remain active
                    is_active = progressive_state.is_loss_active(name)
                    
                    if is_active and deactivation_conditions:
                        # Check if deactivation conditions are met
                        should_deactivate = evaluate_progressive_conditions(deactivation_conditions, progressive_state)
                        
                        if should_deactivate:
                            progressive_state.mark_loss_deactivated(name)
                            is_active = False
                            if kwargs.get('verbose', False):
                                condition_summary = _format_condition_summary(deactivation_conditions)
                                print(f"Progressive Training: Deactivated '{name}' at epoch {progressive_state.current_epoch}")
                                print(f"Deactivation conditions met: {condition_summary}")
                    # Once deactivated, losses cannot be reactivated
                    # This ensures truly progressive training without oscillations
            
            # Track active status for reporting
            active_losses[name] = is_active
            
            # Skip inactive losses
            if not is_active:
                individual_losses[name] = 0.0
                weighted_losses[name] = 0.0
                continue
            
            # Get the loss function
            loss_fn = available_losses[loss_function_name]
            
            # Apply spatial pooling for this specific instance
            if spatial_pooling_config.get('enabled', False):
                instance_y_pred, instance_y_true = apply_spatial_pooling(
                    y_pred_samples, y_true, spatial_pooling_config
                )
            else:
                instance_y_pred, instance_y_true = y_pred_samples, y_true
            
            # Merge arguments (exclude progressive_state from loss function args)
            merged_args = {k: v for k, v in {**loss_args, **kwargs}.items() 
                          if k != 'progressive_state'}
            
            # Compute loss
            try:
                loss_value = loss_fn(instance_y_pred, instance_y_true, **merged_args)
            except TypeError as e:
                # Handle case where loss function doesn't accept some arguments
                import inspect
                sig = inspect.signature(loss_fn)
                valid_args = {k: v for k, v in merged_args.items() if k in sig.parameters}
                loss_value = loss_fn(instance_y_pred, instance_y_true, **valid_args)
            
            # Ensure loss_value is per-batch (reduce any extra dimensions)
            if loss_value.dim() > 1:
                loss_value = loss_value.sum(dim=tuple(range(1, loss_value.dim())))
            
            # Store individual loss (before weighting)
            individual_losses[name] = loss_value.mean().item() if hasattr(loss_value, 'mean') else loss_value
            
            # Apply coefficient
            weighted_loss = coefficient * loss_value
            weighted_losses[name] = weighted_loss.mean().item() if hasattr(weighted_loss, 'mean') else weighted_loss
            
            # Add to total loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        
        # Handle case where no losses are active (total_loss is None)
        if total_loss is None:
            # Create a zero tensor with the same shape as the input batch
            batch_size = y_true.shape[0]
            total_loss = torch.zeros(batch_size, device=y_true.device, dtype=y_true.dtype)
        
        if return_components:
            # Return both individual (unweighted) and weighted losses for detailed breakdown
            breakdown = {
                'individual_losses': individual_losses,
                'weighted_losses': weighted_losses,
                'total_loss': total_loss.mean().item() if hasattr(total_loss, 'mean') else total_loss
            }
            return total_loss, breakdown
        else:
            return total_loss
    
    # Store metadata for breakdown reporting
    multi_instance_loss_function._is_multi_instance = True
    multi_instance_loss_function._loss_instances = loss_instances
    
    return multi_instance_loss_function


def create_loss_function_unified(config):
    """
    Unified loss function creator that supports both original and multi-instance formats.
    Provides full backward compatibility with existing configurations.
    
    Args:
        config: Dictionary with either:
            
            ORIGINAL FORMAT (backward compatible):
            - losses: List of loss function strings
            - loss_function_args: Dict of arguments for each loss function  
            - coefficients: Dict of multipliers for each loss function
            - spatial_pooling: Optional global spatial pooling config
            
            MULTI-INSTANCE FORMAT (new):
            - loss_instances: List of loss instance dictionaries
            
    Returns:
        A PyTorch-compatible loss function
        
    Examples:
        # Original format (existing configs work unchanged)
        config = {
            'losses': ['energy_score_loss'],
            'coefficients': {'energy_score_loss': 1.0},
            'spatial_pooling': {'enabled': True, 'pool_type': 'mean', 'kernel_size': 2}
        }
        
        # Multi-instance format (new functionality)
        config = {
            'loss_instances': [
                {
                    'name': 'energy_coarse',
                    'loss_function': 'energy_score_loss',
                    'coefficient': 1.0,
                    'spatial_pooling': {'enabled': True, 'pool_type': 'mean', 'kernel_size': 4}
                }
            ]
        }
    """
    # Check which format is being used
    if 'loss_instances' in config:
        # New multi-instance format
        return create_multi_instance_loss_function(config)
    else:
        # Original format - use existing function for full compatibility
        return create_loss_function(config)

# Backward compatibility alias
create_ar_loss_function = create_loss_function_unified


def test_spatial_pooling_example():
    """
    Example demonstrating spatial pooling functionality with loss functions.
    This function serves as both documentation and a test.
    """
    import torch
    
    # Create sample spatial data (e.g., 64x64 image flattened to 4096 dimensions)
    batch_size = 4
    n_samples = 10
    height, width = 64, 64
    spatial_dim = height * width  # 4096
    
    # Generate sample data
    torch.manual_seed(42)
    y_pred_samples = torch.randn(batch_size, n_samples, spatial_dim)
    y_true = torch.randn(batch_size, spatial_dim)
    
    print(f"Original data shapes:")
    print(f"  y_pred_samples: {y_pred_samples.shape}")
    print(f"  y_true: {y_true.shape}")
    
    # Test spatial pooling directly
    pooling_config = {
        'enabled': True,
        'pool_type': 'mean',
        'kernel_size': 4,  # 4x4 pooling
        'stride': 4,       # Non-overlapping
        'spatial_shape': (height, width)
    }
    
    pooled_pred, pooled_true = apply_spatial_pooling(y_pred_samples, y_true, pooling_config)
    
    print(f"\nAfter 4x4 mean pooling (stride=4):")
    print(f"  pooled_pred: {pooled_pred.shape}")
    print(f"  pooled_true: {pooled_true.shape}")
    print(f"  Spatial dimensions reduced from {height}x{width} to {height//4}x{width//4}")
    
    # Test different pooling types
    pooling_types = ['mean', 'max', 'min', 'median', 'percentile_75']
    for pool_type in pooling_types:
        config = {
            'enabled': True,
            'pool_type': pool_type,
            'kernel_size': 2,
            'spatial_shape': (height, width)
        }
        try:
            pooled_pred, pooled_true = apply_spatial_pooling(y_pred_samples, y_true, config)
            print(f"  {pool_type} pooling: {pooled_pred.shape} ✓")
        except Exception as e:
            print(f"  {pool_type} pooling: Error - {e}")
    
    # Test with loss functions
    print(f"\nTesting with loss functions:")
    
    # Configuration for combined loss with spatial pooling
    loss_config = {
        'losses': ['energy_score_loss', 'crps_loss_general'],
        'coefficients': {
            'energy_score_loss': 1.0,
            'crps_loss_general': 0.5
        },
        'spatial_pooling': {
            'enabled': True,
            'pool_type': 'mean',
            'kernel_size': 8,  # 8x8 pooling for more reduction
            'spatial_shape': (height, width)
        }
    }
    
    # Create loss function
    loss_fn = create_loss_function(loss_config)
    
    # Compute loss without pooling
    loss_config_no_pool = loss_config.copy()
    loss_config_no_pool['spatial_pooling'] = {'enabled': False}
    loss_fn_no_pool = create_loss_function(loss_config_no_pool)
    
    with torch.no_grad():
        loss_with_pooling = loss_fn(y_pred_samples, y_true)
        loss_without_pooling = loss_fn_no_pool(y_pred_samples, y_true)
    
    print(f"  Loss without pooling: {loss_without_pooling.mean().item():.4f}")
    print(f"  Loss with 8x8 pooling: {loss_with_pooling.mean().item():.4f}")
    print(f"  Computation performed on {(height//8) * (width//8)} pooled dimensions instead of {spatial_dim}")
    
    return True

if __name__ == "__main__":
    # Run the example/test
    test_spatial_pooling_example()



