import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnaut.crps import EpsilonSampler, crps_loss

class MLPSampler(nn.Module):
    """
    Multi-layer Perceptron that generates random samples for probabilistic predictions.
    Uses the EpsilonSampler from torchnaut to enable sample-based predictions.
    """
    def __init__(self, input_size=1, hidden_size=64, latent_dim=16, n_layers=2, dropout_rate=0.1):
        super(MLPSampler, self).__init__()
        
        # Network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        # Create layers
        layers = []
        # First layer from input_size to hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Final layer that outputs features to be combined with random samples
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
        # Epsilon sampler for adding random dimensions
        self.sampler = EpsilonSampler(latent_dim)
        
        # Final prediction layer that combines extracted features with random samples
        self.prediction_layer = nn.Linear(hidden_size + latent_dim, 1)
        
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
        features = self.feature_extractor(x)
        features = self.output_layer(features)
        
        # Add random samples using epsilon sampler
        augmented = self.sampler(features, n_samples=n_samples)
        
        # Generate predictions from the combined features and random samples
        # Shape: [batch_size, n_samples, 1]
        predictions = self.prediction_layer(augmented)
        
        # Remove the last dimension to get [batch_size, n_samples]
        return predictions.squeeze(-1)
    
    def crps_loss(self, x, y, n_samples=None):
        """
        Compute CRPS loss using the torchnaut implementation
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, 1]
            n_samples: Number of samples to generate (optional)
            
        Returns:
            loss: Scalar CRPS loss (mean across batch)
        """
        # Generate samples
        samples = self.forward(x, n_samples=n_samples)
        
        # Compute CRPS loss using torchnaut
        # Ensure y is properly shaped [batch_size, 1]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Get per-sample losses
        per_sample_loss = crps_loss(samples, y)
        
        # Return mean loss (scalar)
        return per_sample_loss.mean() 