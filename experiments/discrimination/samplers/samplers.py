import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import warnings


class BaseSampler(ABC):
    """Base class for all multivariate distribution samplers."""
    
    def __init__(self, dimension: int, **kwargs):
        """
        Initialise the sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            **kwargs: Additional parameters specific to each distribution
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        self.dimension = dimension
        self.rng = np.random.RandomState(kwargs.get('random_state', None))
    
    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from the distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, dimension) containing the samples
        """
        pass
    
    def __call__(self, n_samples: int) -> np.ndarray:
        """Allow the sampler to be called directly."""
        return self.sample(n_samples)


class MultivariateGaussianSampler(BaseSampler):
    """Sampler for multivariate Gaussian distributions."""
    
    def __init__(self, dimension: int, mean: Optional[np.ndarray] = None, 
                 cov: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the multivariate Gaussian sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            mean: Mean vector (defaults to zero vector)
            cov: Covariance matrix (defaults to identity matrix)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        self.mean = mean if mean is not None else np.zeros(dimension)
        self.cov = cov if cov is not None else np.eye(dimension)
        
        # Validate inputs
        if self.mean.shape != (dimension,):
            raise ValueError(f"Mean must have shape ({dimension},)")
        if self.cov.shape != (dimension, dimension):
            raise ValueError(f"Covariance must have shape ({dimension}, {dimension})")
        
        # Check if covariance is positive semi-definite
        try:
            self.chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive semi-definite")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from multivariate Gaussian distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate standard normal samples
        z = self.rng.randn(n_samples, self.dimension)
        
        # Transform to desired distribution
        samples = z @ self.chol.T + self.mean
        
        return samples


class MultivariateStudentTSampler(BaseSampler):
    """Sampler for multivariate Student-t distributions."""
    
    def __init__(self, dimension: int, df: float, loc: Optional[np.ndarray] = None,
                 scale: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the multivariate Student-t sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            df: Degrees of freedom
            loc: Location parameter (defaults to zero vector)
            scale: Scale matrix (defaults to identity matrix)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        
        self.df = df
        self.loc = loc if loc is not None else np.zeros(dimension)
        self.scale = scale if scale is not None else np.eye(dimension)
        
        # Validate inputs
        if self.loc.shape != (dimension,):
            raise ValueError(f"Location must have shape ({dimension},)")
        if self.scale.shape != (dimension, dimension):
            raise ValueError(f"Scale must have shape ({dimension}, {dimension})")
        
        # Check if scale is positive definite
        try:
            self.chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError:
            raise ValueError("Scale matrix must be positive definite")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from multivariate Student-t distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate standard normal samples
        z = self.rng.randn(n_samples, self.dimension)
        
        # Generate chi-squared samples for normalisation
        chi2_samples = self.rng.chisquare(self.df, n_samples)
        
        # Transform to Student-t
        samples = z @ self.chol.T
        samples = samples / np.sqrt(chi2_samples / self.df)[:, np.newaxis]
        samples = samples + self.loc
        
        return samples


class MultivariateLaplacianSampler(BaseSampler):
    """Sampler for multivariate Laplacian distributions."""
    
    def __init__(self, dimension: int, loc: Optional[np.ndarray] = None,
                 scale: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the multivariate Laplacian sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            loc: Location parameter (defaults to zero vector)
            scale: Scale matrix (defaults to identity matrix)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        self.loc = loc if loc is not None else np.zeros(dimension)
        self.scale = scale if scale is not None else np.eye(dimension)
        
        # Validate inputs
        if self.loc.shape != (dimension,):
            raise ValueError(f"Location must have shape ({dimension},)")
        if self.scale.shape != (dimension, dimension):
            raise ValueError(f"Scale must have shape ({dimension}, {dimension})")
        
        # Check if scale is positive definite
        try:
            self.chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError:
            raise ValueError("Scale matrix must be positive definite")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from multivariate Laplacian distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate standard Laplacian samples using difference of exponentials
        exp1 = self.rng.exponential(1, (n_samples, self.dimension))
        exp2 = self.rng.exponential(1, (n_samples, self.dimension))
        laplacian_samples = exp1 - exp2
        
        # Transform to desired distribution
        samples = laplacian_samples @ self.chol.T + self.loc
        
        return samples


class MultivariateUniformSampler(BaseSampler):
    """Sampler for multivariate uniform distributions on hypercubes."""
    
    def __init__(self, dimension: int, low: Optional[np.ndarray] = None,
                 high: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the multivariate uniform sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            low: Lower bounds (defaults to zero vector)
            high: Upper bounds (defaults to ones vector)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        self.low = low if low is not None else np.zeros(dimension)
        self.high = high if high is not None else np.ones(dimension)
        
        # Validate inputs
        if self.low.shape != (dimension,):
            raise ValueError(f"Low bounds must have shape ({dimension},)")
        if self.high.shape != (dimension,):
            raise ValueError(f"High bounds must have shape ({dimension},)")
        if np.any(self.low >= self.high):
            raise ValueError("Low bounds must be less than high bounds")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from multivariate uniform distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate uniform samples in [0, 1]
        uniform_samples = self.rng.uniform(0, 1, (n_samples, self.dimension))
        
        # Transform to desired bounds
        samples = uniform_samples * (self.high - self.low) + self.low
        
        return samples


class DirichletSampler(BaseSampler):
    """Sampler for Dirichlet distributions (samples on simplex)."""
    
    def __init__(self, dimension: int, alpha: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the Dirichlet sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            alpha: Concentration parameters (defaults to ones vector)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        self.alpha = alpha if alpha is not None else np.ones(dimension)
        
        # Validate inputs
        if self.alpha.shape != (dimension,):
            raise ValueError(f"Alpha must have shape ({dimension},)")
        if np.any(self.alpha <= 0):
            raise ValueError("Alpha parameters must be positive")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from Dirichlet distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate gamma samples
        gamma_samples = np.zeros((n_samples, self.dimension))
        for i in range(self.dimension):
            gamma_samples[:, i] = self.rng.gamma(self.alpha[i], 1, n_samples)
        
        # Normalise to get Dirichlet samples
        samples = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)
        
        return samples


class MixtureGaussianSampler(BaseSampler):
    """Sampler for mixture of multivariate Gaussian distributions."""
    
    def __init__(self, dimension: int, n_components: int, 
                 means: Optional[np.ndarray] = None,
                 covariances: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the mixture of Gaussians sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            n_components: Number of mixture components
            means: Mean vectors for each component (n_components, dimension)
            covariances: Covariance matrices for each component (n_components, dimension, dimension)
            weights: Mixture weights (defaults to uniform)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        self.n_components = n_components
        
        # Default parameters
        if means is None:
            means = self.rng.randn(n_components, dimension)
        if covariances is None:
            covariances = np.array([np.eye(dimension) for _ in range(n_components)])
        if weights is None:
            weights = np.ones(n_components) / n_components
        
        self.means = means
        self.covariances = covariances
        self.weights = weights
        
        # Validate inputs
        if self.means.shape != (n_components, dimension):
            raise ValueError(f"Means must have shape ({n_components}, {dimension})")
        if self.covariances.shape != (n_components, dimension, dimension):
            raise ValueError(f"Covariances must have shape ({n_components}, {dimension}, {dimension})")
        if self.weights.shape != (n_components,):
            raise ValueError(f"Weights must have shape ({n_components},)")
        if not np.allclose(self.weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1")
        
        # Precompute Cholesky decompositions
        self.chols = []
        for i in range(n_components):
            try:
                chol = np.linalg.cholesky(self.covariances[i])
                self.chols.append(chol)
            except np.linalg.LinAlgError:
                raise ValueError(f"Covariance matrix {i} must be positive semi-definite")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from mixture of Gaussians distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Sample component assignments
        components = self.rng.choice(self.n_components, n_samples, p=self.weights)
        
        # Generate samples
        samples = np.zeros((n_samples, self.dimension))
        for i in range(self.n_components):
            mask = components == i
            n_comp_samples = mask.sum()
            
            if n_comp_samples > 0:
                # Generate standard normal samples
                z = self.rng.randn(n_comp_samples, self.dimension)
                
                # Transform to desired distribution
                comp_samples = z @ self.chols[i].T + self.means[i]
                samples[mask] = comp_samples
        
        return samples


class WishartSampler(BaseSampler):
    """Sampler for Wishart distributions (positive definite matrices)."""
    
    def __init__(self, dimension: int, df: float, scale: Optional[np.ndarray] = None, **kwargs):
        """
        Initialise the Wishart sampler.
        
        Args:
            dimension: The dimensionality of the distribution
            df: Degrees of freedom
            scale: Scale matrix (defaults to identity matrix)
            **kwargs: Additional parameters
        """
        super().__init__(dimension, **kwargs)
        
        if df < dimension:
            raise ValueError("Degrees of freedom must be at least the dimension")
        
        self.df = df
        self.scale = scale if scale is not None else np.eye(dimension)
        
        # Validate inputs
        if self.scale.shape != (dimension, dimension):
            raise ValueError(f"Scale must have shape ({dimension}, {dimension})")
        
        # Check if scale is positive definite
        try:
            self.chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError:
            raise ValueError("Scale matrix must be positive definite")
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from Wishart distribution."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        samples = np.zeros((n_samples, self.dimension, self.dimension))
        
        for i in range(n_samples):
            # Generate random matrix
            A = self.rng.randn(int(self.df), self.dimension)
            
            # Create Wishart sample
            W = A.T @ A
            
            # Transform by scale matrix
            samples[i] = self.chol @ W @ self.chol.T
        
        return samples


# Convenience function to create samplers
def create_sampler(distribution_type: str, dimension: int, **kwargs) -> BaseSampler:
    """
    Create a sampler for a specific distribution type.
    
    Args:
        distribution_type: Type of distribution ('gaussian', 'student_t', 'laplacian', 
                          'uniform', 'dirichlet', 'mixture_gaussian', 'wishart')
        dimension: Dimensionality of the distribution
        **kwargs: Additional parameters for the specific sampler
        
    Returns:
        Appropriate sampler instance
    """
    samplers = {
        'gaussian': MultivariateGaussianSampler,
        'student_t': MultivariateStudentTSampler,
        'laplacian': MultivariateLaplacianSampler,
        'uniform': MultivariateUniformSampler,
        'dirichlet': DirichletSampler,
        'mixture_gaussian': MixtureGaussianSampler,
        'wishart': WishartSampler
    }
    
    if distribution_type not in samplers:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return samplers[distribution_type](dimension, **kwargs)
