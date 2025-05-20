import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MLPGaussian(nn.Module):
    """
    Multi-layer Perceptron with two heads for predicting mean and variance
    of a Gaussian distribution. Has a single hidden layer.
    """
    def __init__(self, input_size=1, hidden_size=32):
        super(MLPGaussian, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Mean prediction head
        self.mean_head = nn.Linear(hidden_size, 1)
        
        # Variance prediction head (will output log(variance) for numerical stability)
        self.logvar_head = nn.Linear(hidden_size, 1)
        
        # Small constant for numerical stability
        self.eps = 1e-6
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            mean: Predicted mean of shape [batch_size, 1]
            var: Predicted variance of shape [batch_size, 1]
        """
        # Shared hidden layer
        x = F.relu(self.fc1(x))
        
        # Mean prediction
        mean = self.mean_head(x)
        
        # Variance prediction (ensure positivity by exponentiating)
        logvar = self.logvar_head(x)
        # Clamp logvar to avoid extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar) + self.eps
        
        return mean, var
    
    def log_likelihood_loss(self, x, y):
        """
        Negative log-likelihood loss for a univariate Gaussian
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, 1]
            
        Returns:
            loss: Negative log-likelihood loss
        """
        mean, var = self.forward(x)
        
        # Log-likelihood of normal distribution: -0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        neg_log_likelihood = 0.5 * (math.log(2 * math.pi) + 
                                   torch.log(var) + 
                                   torch.pow(y - mean, 2) / var)
        
        return neg_log_likelihood.mean()
    
    def crps_loss(self, x, y):
        """
        Continuous Ranked Probability Score (CRPS) for a Gaussian distribution
        
        For normal distributions, the CRPS has the closed form:
        CRPS(N(μ,σ), y) = σ * [-1/sqrt(π) + 2*φ(z) + z*(2*Φ(z)-1)]
        
        where:
        - z = (y-μ)/σ is the standardized prediction error
        - φ is the PDF of the standard normal
        - Φ is the CDF of the standard normal
        
        Lower CRPS is better, so we return the positive value to minimize.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            y: Target tensor of shape [batch_size, 1]
            
        Returns:
            loss: CRPS loss (positive value to minimize)
        """
        mean, var = self.forward(x)
        std = torch.sqrt(var)
        
        # Standardized prediction error
        z = (y - mean) / std
        
        # Standard normal PDF
        norm_pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        
        # Standard normal CDF
        norm_cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))
        
        # CRPS formula - note we take positive value to minimize
        crps = std * (-1 / math.sqrt(math.pi) + 2 * norm_pdf + z * (2 * norm_cdf - 1))
        
        # We want positive values to minimize
        return torch.abs(crps.mean()) 