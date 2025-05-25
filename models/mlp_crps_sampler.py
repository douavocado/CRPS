import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnaut.crps import EpsilonSampler, crps_loss, crps_loss_mv
from torchnaut import crps

class MLPSampler(nn.Module):
    """
    Multi-layer Perceptron that generates random samples for probabilistic predictions.
    Uses the EpsilonSampler from torchnaut to enable sample-based predictions.
    """
    def __init__(self, input_size=1, hidden_size=64, latent_dim=16, n_layers=2, dropout_rate=0.0, output_size=1, sample_layer_index=1):
        super(MLPSampler, self).__init__()
        
        # Network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Create layers
        layers = []
        # First layer from input_size to hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        # Adding first layers before randomisation layer
        for _ in range(sample_layer_index - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # add randomisation early on for non-linear modelling of noise
        layers.append(EpsilonSampler(latent_dim))
        layers.append(nn.Linear(hidden_size + latent_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for _ in range(min(n_layers - sample_layer_index - 1, 0)):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Final layer that outputs features to be combined with random samples
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, n_samples=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            n_samples: Number of samples to generate (optional)
            
        Returns:
            samples: Predicted samples of shape [batch_size, n_samples]
        """
        # Extract features
        with crps.EpsilonSampler.n_samples(n_samples): # Override the default number of n_samples.
            features = self.feature_extractor(x)
            predictions = self.output_layer(features)
        
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
            per_sample_loss = crps_loss_mv(samples, y)
        else:
            per_sample_loss = crps_loss(samples.squeeze(-1), y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean() 