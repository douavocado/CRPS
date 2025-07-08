import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnaut.crps import EpsilonSampler, crps_loss, crps_loss_mv
from torchnaut import crps

from common.losses import crps_loss_general

class MLPSampler(nn.Module):
    """
    Multi-layer Perceptron that generates random samples for probabilistic predictions.
    Uses the EpsilonSampler from torchnaut to enable sample-based predictions.
    """
    def __init__(self, input_size=1, hidden_size=64, latent_dim=16, n_layers=2, dropout_rate=0.0, output_size=1, sample_layer_index=1, zero_inputs=False, non_linear=True, activation_function='relu'):
        super(MLPSampler, self).__init__()
        self.zero_inputs = zero_inputs # If true, we ignore inputs to the model. We replace the initial layer with a trainable vector.
        
        # Define activation function mapping
        self.activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus()
        }
        
        # Validate and set activation function
        if activation_function not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_function}. "
                           f"Supported functions: {list(self.activation_functions.keys())}")
        
        self.activation_fn = self.activation_functions[activation_function]
        
        # Calculate total number of hidden layers needed
        # For zero_inputs=False: n_layers includes input->hidden, hidden layers, and hidden->output
        # For zero_inputs=True: n_layers includes only hidden layers and hidden->output
        if not zero_inputs:
            # n_layers = 1 (input->hidden) + num_hidden_layers + 1 (hidden->output)
            # So num_hidden_layers = n_layers - 2
            # But we need at least 1 hidden layer, so max(1, n_layers - 1) for practical purposes
            total_hidden_layers = max(1, n_layers - 1)
        else:
            # For zero_inputs, n_layers represents the total depth from first hidden to output
            # So num_hidden_layers = n_layers - 1 (excluding the output layer)
            total_hidden_layers = max(1, n_layers - 1)
        
        # Handle hidden_size parameter - can be int or list
        if isinstance(hidden_size, (list, tuple)):
            self.hidden_sizes = list(hidden_size)
            # Validate that the length matches expected number of hidden layers
            if len(self.hidden_sizes) != total_hidden_layers:
                raise ValueError(f"Length of hidden_sizes list ({len(self.hidden_sizes)}) must match "
                               f"number of hidden layers ({total_hidden_layers}). "
                               f"With n_layers={n_layers} and zero_inputs={zero_inputs}, "
                               f"expected {total_hidden_layers} hidden layer sizes.")
        else:
            # Convert single hidden_size to list for consistent handling
            self.hidden_sizes = [hidden_size] * total_hidden_layers
        
        if self.zero_inputs:
            self.input_vector = nn.Parameter(torch.randn(self.hidden_sizes[0]), requires_grad=True)
        else:
            self.input_vector = None
        
        # Network architecture
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.non_linear = non_linear
        
        # Create layers
        layers = []
        hidden_layer_idx = 0
        
        if not self.zero_inputs: # need to map input dimension to hidden size
            # First layer from input_size to first hidden size
            layers.append(nn.Linear(input_size, self.hidden_sizes[hidden_layer_idx]))
            if self.non_linear:
                layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            hidden_layer_idx += 1
        
        # Adding layers before randomisation layer (sample_layer_index - 1 additional hidden layers)
        for _ in range(sample_layer_index - 1):
            if hidden_layer_idx >= len(self.hidden_sizes):
                break
            prev_size = self.hidden_sizes[hidden_layer_idx - 1]
            curr_size = self.hidden_sizes[hidden_layer_idx]
            layers.append(nn.Linear(prev_size, curr_size))
            if self.non_linear:
                layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            hidden_layer_idx += 1
        
        # add randomisation early on for non-linear modelling of noise
        layers.append(EpsilonSampler(latent_dim))
        
        # Determine input size for layer after EpsilonSampler
        if hidden_layer_idx > 0:
            prev_size = self.hidden_sizes[hidden_layer_idx - 1] + latent_dim
        else:
            # This case happens when zero_inputs=True and sample_layer_index=1
            prev_size = self.hidden_sizes[0] + latent_dim
        
        # Remaining hidden layers after EpsilonSampler
        remaining_layers = len(self.hidden_sizes) - hidden_layer_idx
        
        if remaining_layers == 0:
            # Go straight to output layer
            layers.append(nn.Linear(prev_size, output_size))
            if self.non_linear:
                layers.append(self.activation_fn)
        else:
            # Add remaining hidden layers
            for i in range(remaining_layers):
                curr_size = self.hidden_sizes[hidden_layer_idx + i]
                layers.append(nn.Linear(prev_size, curr_size))
                if self.non_linear and i < remaining_layers - 1: # don't add non-linearity to the last layer
                    layers.append(self.activation_fn)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = curr_size
            
            # Add output layer
            layers.append(nn.Linear(prev_size, output_size))
        
        # Final layer that outputs features to be combined with random samples
        self.feature_extractor = nn.Sequential(*layers)
        
    def forward(self, x, n_samples=None):
        """
        Forward pass through the network. If zero_inputs is True we directly feed self.input_vector to the feature extractor.
        
        Args:
            x: Input tensor of shape [batch_size, input_size] (single timestep only)
            n_samples: Number of samples to generate (optional)
            
        Returns:
            samples: Predicted samples of shape [batch_size, n_samples, output_size]
        """
        # Validate input shape - only single timestep supported
        if not self.zero_inputs and len(x.shape) != 2:
            raise ValueError(f"MLPSampler only supports single timestep inputs. "
                           f"Expected shape [batch_size, input_size], got {x.shape}. "
                           f"If you have time series data [batch_size, timesteps, dims], "
                           f"this model is not suitable for multi-timestep inputs.")
        
        # Extract features
        with crps.EpsilonSampler.n_samples(n_samples): # Override the default number of n_samples.
            if self.zero_inputs:
                # Project input_vector to match batch size
                batch_size = x.shape[0]
                batch_input_vector = self.input_vector.expand(batch_size, -1)
                predictions = self.feature_extractor(batch_input_vector)
            else:
                predictions = self.feature_extractor(x)
        
        return predictions # shape [batch_size, n_samples, output_size]
    
    def crps_loss(self, x, y, n_samples=None):
        """
        Compute CRPS loss using the torchnaut implementation. If the output_size is greater than 1, the loss is computed for each output dimension and then averaged.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, output_size]
            n_samples: Number of samples to generate (optional)
            
        Returns:
            loss: Scalar CRPS loss (mean across batch, and across output_size)
        """
        # Generate samples
        samples = self.forward(x, n_samples=n_samples) # shape [batch_size, n_samples, output_size]
        
        # Compute CRPS loss using torchnaut        
        # Get per-sample losses, if output_size > 1, the loss is computed for each output dimension and then averaged
        if self.output_size > 1:
            per_sample_loss = crps_loss_general(samples, y)
        else:
            per_sample_loss = crps_loss(samples.squeeze(-1), y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean()
    
    def energy_score_loss(self, x, y, n_samples=None):
        """
        Compute energy score loss using the torchnaut implementation.

        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, output_size]
            n_samples: Number of samples to generate (optional)
            
        Returns:
            loss: Scalar energy score loss (mean across batch, and across output_size)
        """
        samples = self.forward(x, n_samples=n_samples) # shape [batch_size, n_samples, output_size]
        
        # Compute energy score loss using torchnaut
        if self.output_size > 1:
            per_sample_loss = crps_loss_mv(samples, y)
        else:
            # then the energy score loss is the same as crps loss
            per_sample_loss = crps_loss(samples.squeeze(-1), y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean()