import torch
import torch.nn as nn
from torchnaut.crps import crps_loss, crps_loss_mv

from common.losses import crps_loss_general


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalisation layer that uses a conditioning embedding.
    """
    def __init__(self, normalized_shape, conditioning_dim):
        super(ConditionalLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.conditioning_dim = conditioning_dim
        
        # Standard layer norm parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
        # Conditioning projection layers
        self.gamma_proj = nn.Linear(conditioning_dim, normalized_shape)
        self.beta_proj = nn.Linear(conditioning_dim, normalized_shape)
        
        self.eps = 1e-5
    
    def forward(self, x, conditioning):
        """
        Args:
            x: Input tensor to normalize [batch_size, ..., normalized_shape]
            conditioning: Conditioning embedding [batch_size, conditioning_dim]
        
        Returns:
            Normalized and conditioned tensor
        """
        # Standard layer normalisation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply conditioning
        gamma_cond = self.gamma + self.gamma_proj(conditioning)
        beta_cond = self.beta + self.beta_proj(conditioning)
        
        # Ensure broadcasting works correctly
        while len(gamma_cond.shape) < len(x_norm.shape):
            gamma_cond = gamma_cond.unsqueeze(-2)
            beta_cond = beta_cond.unsqueeze(-2)
        
        return gamma_cond * x_norm + beta_cond


class FGNEncoderSampler(nn.Module):
    """
    FGN (Feature-based Gaussian Noise) Encoder Sampler model.
    Uses conditional layer normalisation with encoded noise as conditioning.
    """
    def __init__(self, input_size=1, hidden_size=64, latent_dim=16, n_layers=2, 
                 dropout_rate=0.0, output_size=1, zero_inputs=False, non_linear=True, 
                 activation_function='relu'):
        super(FGNEncoderSampler, self).__init__()
        
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
        
        self.zero_inputs = zero_inputs
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.non_linear = non_linear
        self.dropout_rate = dropout_rate
        
        # Handle hidden_size parameter - can be int or list
        if isinstance(hidden_size, (list, tuple)):
            self.hidden_sizes = list(hidden_size)
            if len(self.hidden_sizes) != n_layers:
                raise ValueError(f"Length of hidden_sizes list ({len(self.hidden_sizes)}) must match "
                               f"n_layers ({n_layers}).")
        else:
            self.hidden_sizes = [hidden_size] * n_layers
        
        # Noise encoder - transforms latent noise to conditioning embedding
        self.noise_encoder = nn.Linear(latent_dim, latent_dim)
        
        # Input handling for zero_inputs case
        if self.zero_inputs:
            self.input_vector = nn.Parameter(torch.randn(self.hidden_sizes[0]), requires_grad=True)
        else:
            self.input_vector = None
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        # Determine layer sizes
        layer_sizes = []
        if not self.zero_inputs:
            layer_sizes.append(input_size)
        layer_sizes.extend(self.hidden_sizes)
        layer_sizes.append(output_size)
        
        # Create layers with conditional layer norms
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            
            # Conditional layer norm (pre-norm style)
            if i == 0 and not self.zero_inputs:
                # First layer norm for input
                self.layer_norms.append(ConditionalLayerNorm(input_dim, latent_dim))
            elif i > 0 or self.zero_inputs:
                # Layer norm for hidden layers
                prev_output_dim = layer_sizes[i]
                self.layer_norms.append(ConditionalLayerNorm(prev_output_dim, latent_dim))
            else:
                # No layer norm for this case
                self.layer_norms.append(None)
            
            # Linear layer
            self.layers.append(nn.Linear(input_dim, output_dim))
            
            # Dropout
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
    
    def _generate_encoded_noise(self, batch_size, n_samples, device, random_seed=None):
        """
        Generate encoded noise for conditioning.
        
        Args:
            batch_size: Batch size
            n_samples: Number of samples to generate
            device: Device to generate noise on
            random_seed: Random seed for deterministic sampling. Only used when n_samples=1.
            
        Returns:
            encoded_noise: [batch_size, n_samples, latent_dim]
        """
        # Generate noise with optional deterministic seeding
        if random_seed is not None:
            # Save current random state
            current_state = torch.get_rng_state()
            # Set deterministic seed
            torch.manual_seed(random_seed)
        
        # Sample noise from standard multivariate normal
        noise = torch.randn(batch_size, n_samples, self.latent_dim, device=device)
        
        # Restore random state if we used a seed
        if random_seed is not None:
            torch.set_rng_state(current_state)
        
        # Encode noise through transformation matrix
        encoded_noise = self.noise_encoder(noise)
        
        return encoded_noise
    
    def forward(self, x, n_samples=None, random_seed=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size] (single timestep only)
            n_samples: Number of samples to generate (optional, defaults to 10)
            random_seed: Random seed for deterministic sampling. Only used when n_samples=1.
            
        Returns:
            samples: Predicted samples of shape [batch_size, n_samples, output_size]
        """
        if n_samples is None:
            n_samples = 10
        
        # Validate random_seed usage
        if random_seed is not None and n_samples != 1:
            raise ValueError(f"random_seed can only be used when n_samples=1, but got n_samples={n_samples}")
        
        # Validate input shape - only single timestep supported
        if len(x.shape) != 2:
            raise ValueError(f"FGNEncoderSampler only supports single timestep inputs. "
                           f"Expected shape [batch_size, input_size], got {x.shape}. "
                           f"If you have time series data [batch_size, timesteps, dims], "
                           f"this model is not suitable for multi-timestep inputs.")
        
        batch_size = x.shape[0]
        device = x.device
        
        # Generate encoded noise for conditioning
        encoded_noise = self._generate_encoded_noise(batch_size, n_samples, device, random_seed)
        
        # Prepare input
        if self.zero_inputs:
            # Use learnable input vector
            current_input = self.input_vector.unsqueeze(0).expand(batch_size, -1)
        else:
            current_input = x
        
        # Expand input to match sample dimension
        current_input = current_input.unsqueeze(1).expand(-1, n_samples, -1)
        
        # Forward through layers
        for i, (layer_norm, linear_layer) in enumerate(zip(self.layer_norms, self.layers)):
            # Apply conditional layer norm (pre-norm style)
            if layer_norm is not None:
                # Reshape for layer norm application
                original_shape = current_input.shape
                current_input_flat = current_input.reshape(-1, current_input.shape[-1])
                encoded_noise_flat = encoded_noise.reshape(-1, encoded_noise.shape[-1])
                
                normed_input = layer_norm(current_input_flat, encoded_noise_flat)
                current_input = normed_input.reshape(original_shape)
            
            # Apply linear transformation
            current_input = linear_layer(current_input)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1 and self.non_linear:
                current_input = self.activation_fn(current_input)
            
            # Apply dropout (except for last layer)
            if self.dropouts is not None and i < len(self.layers) - 1:
                current_input = self.dropouts[i](current_input)
        
        return current_input  # shape [batch_size, n_samples, output_size]
    
    def crps_loss(self, x, y, n_samples=None, random_seed=None):
        """
        Compute CRPS loss using the torchnaut implementation.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, output_size]
            n_samples: Number of samples to generate (optional)
            random_seed: Random seed for deterministic sampling. Only used when n_samples=1.
            
        Returns:
            loss: Scalar CRPS loss (mean across batch and output dimensions)
        """
        # Generate samples
        samples = self.forward(x, n_samples=n_samples, random_seed=random_seed)  # shape [batch_size, n_samples, output_size]
        
        # Compute CRPS loss using torchnaut
        if self.output_size > 1:
            per_sample_loss = crps_loss_general(samples, y)
        else:
            per_sample_loss = crps_loss(samples.squeeze(-1), y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean()
    
    def energy_score_loss(self, x, y, n_samples=None, random_seed=None):
        """
        Compute energy score loss using the torchnaut implementation.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, output_size]
            n_samples: Number of samples to generate (optional)
            random_seed: Random seed for deterministic sampling. Only used when n_samples=1.
            
        Returns:
            loss: Scalar energy score loss (mean across batch and output dimensions)
        """
        samples = self.forward(x, n_samples=n_samples, random_seed=random_seed)  # shape [batch_size, n_samples, output_size]
        
        # Compute energy score loss using torchnaut
        if self.output_size > 1:
            per_sample_loss = crps_loss_mv(samples, y)
        else:
            # For 1D output, energy score is the same as CRPS
            per_sample_loss = crps_loss(samples.squeeze(-1), y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean() 