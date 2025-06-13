"""
Dummy model for baseline comparison in CFD prediction tasks.

This model implements a simple baseline strategy:
- If the target variable is also one of the input variables, it predicts the target
  by simply copying the most recent input timestep for that variable, using spatial
  interpolation when the input and target spatial locations differ.
- If the target variable is not in the input variables, it raises an error as this
  baseline cannot handle that case.

This model is compatible with the `CFD2DDataset` class and follows the same interface
as the ConvCNP sampler model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class DummyModel(nn.Module):
    """
    Dummy baseline model for CFD prediction.
    
    This model implements a simple persistence forecast: it predicts the target variable
    by copying the most recent input timestep for that same variable, using spatial
    interpolation when necessary.
    """
    
    def __init__(self, 
                 input_variables: List[str],
                 target_variable: str,
                 time_predict: int,
                 **kwargs):
        """
        Initialise the dummy model.
        
        Args:
            input_variables: List of input variable names
            target_variable: Name of target variable to predict
            time_predict: Number of timesteps to predict
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__()
        
        self.input_variables = input_variables
        self.target_variable = target_variable
        self.time_predict = time_predict
        
        # Check if target variable is in input variables
        if self.target_variable not in self.input_variables:
            raise ValueError(
                f"Dummy model cannot predict target variable '{self.target_variable}' "
                f"because it is not in the input variables {self.input_variables}. "
                f"The dummy model can only predict variables that are also provided as inputs."
            )
        
        # Find the index of the target variable in the input variables
        self.target_var_idx = self.input_variables.index(self.target_variable)
        
        print(f"DummyModel initialised:")
        print(f"  Input variables: {self.input_variables}")
        print(f"  Target variable: {self.target_variable} (index {self.target_var_idx})")
        print(f"  Time predict: {self.time_predict}")
    
    def _rbf_interpolation(self, input_coords: torch.Tensor, target_coords: torch.Tensor, 
                          input_values: torch.Tensor, length_scale: float = 0.1) -> torch.Tensor:
        """
        Perform RBF-based spatial interpolation from input locations to target locations.
        
        Args:
            input_coords: Input coordinates, shape (B, N_in, 2)
            target_coords: Target coordinates, shape (B, N_out, 2)
            input_values: Values at input locations, shape (B, N_in)
            length_scale: RBF kernel length scale
            
        Returns:
            Interpolated values at target locations, shape (B, N_out)
        """
        # Compute pairwise distances: (B, N_out, N_in)
        dists = torch.cdist(target_coords, input_coords)
        
        # RBF weights
        weights = torch.exp(-dists.pow(2) / (2 * length_scale ** 2))
        
        # Normalise weights
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted interpolation: (B, N_out, N_in) @ (B, N_in, 1) -> (B, N_out, 1)
        interpolated = torch.bmm(weights_norm, input_values.unsqueeze(-1)).squeeze(-1)
        
        return interpolated
    
    def forward(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                target_coords: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Forward pass through the dummy model.
        
        Args:
            input_data: Context values, shape (B, T_in, N_in, V_in)
            input_coords: Context coordinates, shape (B, N_in, 2)
            target_coords: Target coordinates, shape (B, N_out, 2)
            n_samples: Number of samples to generate (for compatibility)
            
        Returns:
            Predicted samples, shape (B, n_samples, N_out, T_out)
        """
        batch_size = input_data.shape[0]
        n_target_spatial = target_coords.shape[1]
        
        # Extract the most recent timestep for the target variable
        # input_data shape: (B, T_in, N_in, V_in)
        most_recent_values = input_data[:, -1, :, self.target_var_idx]  # (B, N_in)
        
        # Interpolate from input locations to target locations
        interpolated_values = self._rbf_interpolation(
            input_coords, target_coords, most_recent_values
        )  # (B, N_out)
        
        # Create predictions for all future timesteps (persistence forecast)
        # Repeat the interpolated values for all prediction timesteps
        predictions = interpolated_values.unsqueeze(1).repeat(1, self.time_predict, 1)  # (B, T_out, N_out)
        
        # Expand to multiple samples (all samples are identical for this baseline)
        # Shape: (B, n_samples, N_out, T_out)
        samples = predictions.unsqueeze(1).repeat(1, n_samples, 1, 1).permute(0, 1, 3, 2)
        
        return samples
    
    def energy_score_loss(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                         target_coords: torch.Tensor, target_data: torch.Tensor,
                         n_samples: int = 50) -> torch.Tensor:
        """
        Computes the energy score loss for the given data.
        
        Note: For the dummy model, all samples are identical, so the energy score
        will not be a proper scoring rule. This is implemented for interface compatibility.
        
        Args:
            input_data: Context values, (B, T_in, N_in, V_in)
            input_coords: Context coordinates, (B, N_in, 2)
            target_coords: Target coordinates, (B, N_out, 2)
            target_data: Ground truth target values, (B, T_out, N_out)
            n_samples: Number of samples to use for energy score calculation
            
        Returns:
            Scalar energy score loss
        """
        # Import here to avoid circular imports
        from torchnaut.crps import crps_loss_mv
        
        samples = self.forward(input_data, input_coords, target_coords, n_samples)
        # samples shape: (B, n_samples, N_out, T_out)
        # need to transpose to: (B, N_out, n_samples, T_out)
        samples = samples.permute(0, 2, 1, 3)  # (B, N_out, n_samples, T_out)
        y_true = target_data.permute(0, 2, 1)  # (B, N_out, T_out)
        loss = crps_loss_mv(samples, y_true)
        return loss.mean()
    
    def crps_loss(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                  target_coords: torch.Tensor, target_data: torch.Tensor,
                  n_samples: int = 50) -> torch.Tensor:
        """
        Computes the CRPS loss for the given data.
        
        Note: For the dummy model, all samples are identical, so this reduces to
        mean absolute error. This is implemented for interface compatibility.
        
        Args:
            input_data: Context values, (B, T_in, N_in, V_in)
            input_coords: Context coordinates, (B, N_in, 2)
            target_coords: Target coordinates, (B, N_out, 2)
            target_data: Ground truth target values, (B, T_out, N_out)
            n_samples: Number of samples to use for CRPS calculation
            
        Returns:
            Scalar CRPS loss (equivalent to MAE for deterministic predictions)
        """
        # Import here to avoid circular imports
        from common.losses import crps_loss_general
        
        samples = self.forward(input_data, input_coords, target_coords, n_samples)
        # samples shape: (B, n_samples, N_out, T_out)
        # target_data shape: (B, T_out, N_out)
        
        # Reshape for CRPS calculation
        # For CRPS, we need: yps [batch, num_samples, dims], y [batch, dims]
        # We'll treat each spatial location as a separate batch element
        B, n_samples, N_out, T_out = samples.shape
        
        # Reshape samples to (B * N_out, n_samples, T_out)
        samples_reshaped = samples.permute(0, 2, 1, 3).reshape(B * N_out, n_samples, T_out)
        
        # Reshape target_data to (B * N_out, T_out)
        target_reshaped = target_data.permute(0, 2, 1).reshape(B * N_out, T_out)
        
        # Compute CRPS loss
        loss = crps_loss_general(samples_reshaped, target_reshaped)
        return loss.mean()
    
    def mse_loss(self, input_data: torch.Tensor, input_coords: torch.Tensor,
                 target_coords: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss for the given data.
        
        This provides a simpler loss function that's more meaningful for deterministic predictions.
        
        Args:
            input_data: Context values, (B, T_in, N_in, V_in)
            input_coords: Context coordinates, (B, N_in, 2)
            target_coords: Target coordinates, (B, N_out, 2)
            target_data: Ground truth target values, (B, T_out, N_out)
            
        Returns:
            Scalar MSE loss
        """
        # Get prediction (just use first sample since they're all identical)
        samples = self.forward(input_data, input_coords, target_coords, n_samples=1)
        prediction = samples[:, 0, :, :].permute(0, 2, 1)  # (B, T_out, N_out)
        
        # Compute MSE
        mse = F.mse_loss(prediction, target_data)
        return mse


if __name__ == '__main__':
    # Add the parent directory to Python path for imports
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Example usage and testing
    print("Testing DummyModel...")
    
    # Test case 1: Target variable is in input variables (should work)
    print("\nTest 1: Target variable in input variables")
    try:
        model = DummyModel(
            input_variables=['Vx', 'Vy', 'pressure'],
            target_variable='pressure',
            time_predict=3
        )
        print("✓ Model creation successful")
        
        # Test forward pass
        B, T_in, V_in, N_in = 2, 4, 3, 50
        T_out, N_out = 3, 25
        
        input_data = torch.randn(B, T_in, N_in, V_in)
        input_coords = torch.rand(B, N_in, 2)
        target_coords = torch.rand(B, N_out, 2)
        target_data = torch.randn(B, T_out, N_out)
        
        samples = model(input_data, input_coords, target_coords, n_samples=10)
        print(f"✓ Forward pass successful, output shape: {samples.shape}")
        
        # Test loss computation
        crps_loss = model.crps_loss(input_data, input_coords, target_coords, target_data)
        mse_loss = model.mse_loss(input_data, input_coords, target_coords, target_data)
        print(f"✓ CRPS loss: {crps_loss.item():.4f}")
        print(f"✓ MSE loss: {mse_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test case 2: Target variable not in input variables (should fail)
    print("\nTest 2: Target variable not in input variables")
    try:
        model_fail = DummyModel(
            input_variables=['Vx', 'Vy'],
            target_variable='pressure',  # Not in input variables
            time_predict=3
        )
        print("✗ Model creation should have failed!")
        
    except ValueError as e:
        print(f"✓ Expected error caught: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\nDummyModel testing completed!") 