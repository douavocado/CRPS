import torch
import torch.nn as nn

from common.losses import crps_loss_general


class SimpleAffineNormal(nn.Module):
    """
    Neural network that represents a pure affine transformation of independent normal random variables.
    
    This represents: y = Ax + b where x ~ N(0,I)
    - x are l independent standard normal random variables  
    - A is a learnable l×l transformation matrix
    - b is a learnable l-dimensional bias vector
    
    The resulting distribution is y ~ N(b, AA^T).
    """
    
    def __init__(self, output_dim):
        """
        Args:
            output_dim: Dimension of output (l in the formulation)
        """
        super(SimpleAffineNormal, self).__init__()
        
        self.output_dim = output_dim
        
        # Learnable transformation matrix A (l×l)
        self.A = nn.Parameter(torch.randn(output_dim, output_dim))
        
        # Learnable bias vector b (l×1)  
        self.b = nn.Parameter(torch.randn(output_dim))
        
    def forward(self, n_samples=1, batch_size=1):
        """
        Generate samples from y = Ax + b where x ~ N(0,I).
        
        Args:
            n_samples: Number of samples to generate per batch item
            batch_size: Number of batch items
            
        Returns:
            Samples tensor [batch_size, n_samples, output_dim] if batch_size > 1
            Samples tensor [n_samples, output_dim] if batch_size = 1 (for backwards compatibility)
        """

        # Generate samples for batch: [batch_size, n_samples, output_dim]
        x = torch.randn(batch_size, n_samples, self.output_dim, device=self.A.device)
        
        # Apply affine transformation: y = Ax + b
        # x: [batch_size, n_samples, output_dim]
        # A.T: [output_dim, output_dim] 
        # We want to apply the same A to all batch items and samples
        y = torch.matmul(x, self.A.T) + self.b  # Broadcasting handles the bias addition
        
        return y
    
    def get_mean_and_covariance(self):
        """
        Get the mean and covariance matrix of the resulting distribution.
        
        Returns:
            mean: Mean vector [output_dim] (which is b)
            cov: Covariance matrix [output_dim, output_dim] (which is AA^T)
        """
        mean = self.b
        cov = torch.matmul(self.A, self.A.T)
        return mean, cov

    def crps_loss(self, y, n_samples=None):
        """
        Compute CRPS loss using the torchnaut implementation.
        
        Args:
            y: Target values [batch_size, output_dim]
            n_samples: Number of samples to generate for CRPS computation
            
        Returns:
            Scalar CRPS loss (mean across batch and dimensions)
        """
        batch_size = y.shape[0]
        samples = self.forward(n_samples=n_samples, batch_size=batch_size)
        per_sample_loss = crps_loss_general(samples, y)
        return per_sample_loss.mean()
    
    def log_likelihood(self, y):
        """
        Compute log likelihood of observations under N(b, AA^T).
        
        Args:
            y: Observations [batch_size, output_dim]
            
        Returns:
            Log likelihood [batch_size]
        """
        mean, cov = self.get_mean_and_covariance()
        
        # Add small regularisation for numerical stability
        cov_reg = cov + 1e-6 * torch.eye(self.output_dim, device=cov.device)
        
        # Use PyTorch's multivariate normal
        dist = torch.distributions.MultivariateNormal(mean, cov_reg)
        return dist.log_prob(y)


# Example usage
if __name__ == "__main__":
    print("=== Simple Affine Normal Implementation ===")
    
    # Create model with 3-dimensional output
    model = SimpleAffineNormal(output_dim=3)
    
    # Generate samples
    samples = model.forward(n_samples=100)
    print(f"Sample shape: {samples.shape}")  # [100, 3]
    
    # Get mean and covariance
    mean, cov = model.get_mean_and_covariance()
    print(f"Mean: {mean}")
    print(f"Covariance matrix shape: {cov.shape}")  # [3, 3]
    print(f"Covariance matrix:\n{cov}")
    
    # Compute log likelihood
    y_obs = torch.randn(50, 3)
    log_liks = model.log_likelihood(y_obs)
    print(f"Log likelihood shape: {log_liks.shape}")  # [50]
    print(f"Mean log likelihood: {log_liks.mean().item():.4f}") 