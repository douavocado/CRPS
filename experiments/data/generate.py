import torch
import numpy as np
from sklearn.model_selection import train_test_split

def generate_toy_data_1d(n_samples=500, noise_type='gaussian', mean_function='gaussian_rbf', **kwargs):
    """Generate a toy 1D regression dataset with heteroscedastic noise"""
    # Generate inputs between -3 and 3
    x = np.random.uniform(-3, 3, n_samples)
    # sort x
    x = np.sort(x)
    
    base_noise_scale = np.random.uniform(0.05, 0.5)
    # generate noise from non-gaussian distributions
    if noise_type == 'gaussian':
        noise = np.random.normal(size=n_samples) * base_noise_scale
        noise_args = {'base_noise_scale': base_noise_scale}
    elif noise_type == 'student_t':
        df = np.random.uniform(1, 10)
        noise = np.random.standard_t(df=df, size=n_samples) * base_noise_scale
        noise_args = {'df': df,
                      'base_noise_scale': base_noise_scale}
    elif noise_type == 'cauchy':
        noise = np.random.standard_cauchy(size=n_samples) * base_noise_scale
        noise_args = {'base_noise_scale': base_noise_scale}
    elif noise_type == 'uniform':
        noise = np.random.uniform(0, 1, size=n_samples) * base_noise_scale
        noise_args = {'base_noise_scale': base_noise_scale}
    elif noise_type == 'exponential':
        if 'scale' in kwargs:
            scale = kwargs['scale']
        else:
            scale = 1
        noise = np.random.exponential(scale=scale, size=n_samples) * base_noise_scale
        noise_args = {'scale': scale,
                      'base_noise_scale': base_noise_scale}
    elif noise_type == 'poisson':
        if 'scale' in kwargs:
            scale = kwargs['scale']
        else:
            scale = 1
        noise = np.random.poisson(lam=scale, size=n_samples) * base_noise_scale
        noise_args = {'scale': scale,
                      'base_noise_scale': base_noise_scale}
    elif noise_type == 'gamma':
        if 'scale' in kwargs:
            scale = kwargs['scale']
        else:
            scale = 1
        if 'shape' in kwargs:
            shape = kwargs['shape']
        else:
            shape = 1
        noise = np.random.gamma(shape=shape, scale=scale, size=n_samples) * base_noise_scale
        noise_args = {'scale': scale,
                      'shape': shape,
                      'base_noise_scale': base_noise_scale}
    elif noise_type == 'lognormal':
        if 'mean' in kwargs:
            mean = kwargs['mean']
        else:
            mean = 0
        if 'scale' in kwargs:
            scale = kwargs['scale']
        else:
            scale = 1
        noise = np.random.lognormal(mean=mean, sigma=scale, size=n_samples) * base_noise_scale
        noise_args = {'mean': mean,
                      'scale': scale,
                      'base_noise_scale': base_noise_scale}
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")
    
    if mean_function == 'sin':
        scale = np.random.uniform(0.5, 1.5)
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        mean = scale * np.sin(frequency * x + phase)
    elif mean_function == 'linear':
        slope = np.random.uniform(-0.5, 0.5)
        intercept = np.random.uniform(-1, 1)
        mean = slope * x + intercept
    elif mean_function == 'gaussian_rbf':
        # draw from gaussian process with squared exponential kernel
        amplitude = np.random.uniform(0.5, 1.5)
        lengthscale = np.random.uniform(0.5, 2.0)
        # Set mean function to 0
        mean_f = np.zeros_like(x)
        # Compute covariance matrix with squared exponential kernel
        cov = amplitude * np.exp(-0.5 * (x[:, None] - x[None, :])**2 / lengthscale**2)
        # Sample from the Gaussian process with mean 0
        mean = np.random.multivariate_normal(mean_f, cov)
    else:
        # Default to sin if not specified
        scale = 1.0
        frequency = 1.0
        phase = 0
        mean = scale * np.sin(frequency * x + phase)
    
    y = mean + noise
    
    # Create indices for train-test split while preserving order
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Use the indices to split the data, preserving the original order within each split
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    mean_train, mean_test = mean[train_indices], mean[test_indices]
    
    # Sort the train and test sets by x values to maintain order
    train_sort_idx = np.argsort(x_train)
    test_sort_idx = np.argsort(x_test)
    
    x_train = x_train[train_sort_idx]
    y_train = y_train[train_sort_idx]
    mean_train = mean_train[train_sort_idx]
    
    x_test = x_test[test_sort_idx]
    y_test = y_test[test_sort_idx]
    mean_test = mean_test[test_sort_idx]
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train).view(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test).view(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    return (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, 
            x_train, y_train, x_test, y_test, noise_args, mean_train, mean_test)

def generate_toy_data_multidim(n_samples=500, x_dim=1, y_dim=1, dependent_noise=False, A=None, noise_scale=0.1, **kwargs):
    """
    Generate a multivariate regression dataset with Gaussian noise
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    x_dim : int
        Dimension of input features
    y_dim : int
        Dimension of output features
    dependent_noise : bool
        If True, use a full covariance matrix for noise; if False, use diagonal covariance
    A : np.ndarray or None
        Linear transformation matrix of shape (x_dim, y_dim). If None, randomly generated.
    noise_scale : float
        Base scale for the noise
    
    Returns:
    --------
    Tuple containing:
        - PyTorch tensors for training and testing (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
        - NumPy arrays (x_train, y_train, x_test, y_test)
        - Noise parameters and mean values
    """
    # Generate random inputs
    x = np.random.uniform(-3, 3, (n_samples, x_dim))
    
    # Generate or use provided transformation matrix A
    if A is None:
        A = np.random.uniform(-1, 1, (x_dim, y_dim))
    
    # Generate mean values using linear transformation
    mean = np.dot(x, A)
    
    # Generate noise based on covariance structure
    if dependent_noise:
        # Generate a random positive definite covariance matrix
        random_matrix = np.random.randn(y_dim, y_dim)
        cov_matrix = np.dot(random_matrix, random_matrix.T)
        # Ensure reasonable scale
        cov_matrix = noise_scale * cov_matrix / np.max(np.abs(cov_matrix))
        # Generate multivariate normal noise
        noise = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
    else:
        # Independent noise (diagonal covariance)
        variances = np.random.uniform(0.01, noise_scale, y_dim)
        cov_matrix = np.diag(variances)
        noise = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
    
    # Combine mean and noise
    y = mean + noise
    
    # Store noise parameters
    noise_args = {
        'cov_matrix': cov_matrix,
        'dependent_noise': dependent_noise,
        'A': A
    }
    
    # Create indices for train-test split
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split the data
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    mean_train, mean_test = mean[train_indices], mean[test_indices]
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    return {
        'x_train_tensor': x_train_tensor,
        'y_train_tensor': y_train_tensor,
        'x_test_tensor': x_test_tensor,
        'y_test_tensor': y_test_tensor,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'noise_args': noise_args,
        'mean_train': mean_train,
        'mean_test': mean_test
    }