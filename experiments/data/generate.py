import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .kernels import default_kernel_params, get_kernel_function, evaluate_gp_mean

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

def generate_toy_data_multidim(n_samples=500, x_dim=1, y_dim=1, dependent_noise=False, noise_scale=0.1, mean_function='gaussian',
                               kernel_type='rbf', kernel_params=None, target_correlation=None, **kwargs):
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
    noise_scale : float
        Base scale for the noise
    mean_function : str
        The type of function used for transformation of input features. Defaults to 'gaussian', which is a gaussian process with zero mean.
    kernel_type : str
        Type of kernel for GP mean functions ('rbf', 'matern', 'linear')
    kernel_params : dict or None
        Parameters for the kernel. If None, random parameters are generated.
    target_correlation : float or None
        If not None and dependent_noise is True, attempts to generate a covariance matrix
        with an average correlation coefficient magnitude close to this value. 
        The correlations will have random signs but the average magnitude will be close to this value.
        Must be between 0 and 1.
    
    Returns:
    --------
    Dict containing training/testing data and metadata
    """
    # Generate random inputs
    x = np.random.uniform(-3, 3, (n_samples, x_dim))

    # Store noise parameters
    noise_args = {
        'dependent_noise': dependent_noise,
        'mean_function': mean_function,
        'target_correlation': target_correlation,
    }
    
    if mean_function == 'gaussian':
        # Draw from gaussian process with zero mean
        # Generate or use provided kernel parameters
        if kernel_params is None:
            kernel_params = default_kernel_params(kernel_type)
        
        # Get the appropriate kernel function
        kernel_function = get_kernel_function(kernel_type)
        
        # Sample GP functions for each output dimension
        gp_functions = []
        mean = np.zeros((n_samples, y_dim))
        
        for j in range(y_dim):
            # Compute covariance matrix at training points
            if kernel_type == 'rbf':
                K = kernel_function(x, x, 
                                kernel_params['lengthscale'], 
                                kernel_params['amplitude'])
            elif kernel_type == 'matern':
                K = kernel_function(x, x, 
                                kernel_params['lengthscale'], 
                                kernel_params['amplitude'], 
                                kernel_params['nu'])
            elif kernel_type == 'linear':
                K = kernel_function(x, x, 
                                kernel_params['variance'], 
                                kernel_params['offset'])
            
            # Add small diagonal for numerical stability
            K += 1e-6 * np.eye(n_samples)
            
            # Sample function values from GP prior
            f_values = np.random.multivariate_normal(np.zeros(n_samples), K)
            mean[:, j] = f_values
            
            # Store information for later evaluation at new points
            gp_functions.append({
                'x_train': x.copy(),
                'f_train': f_values.copy(),
                'kernel_type': kernel_type,
                'kernel_params': kernel_params.copy()
            })
        
        # update noise_args with gp_functions
        noise_args['gp_functions'] = gp_functions
    elif mean_function == 'zero':
        # Set mean function to zero, independent of input features
        mean = np.zeros((n_samples, y_dim))
    else:
        raise ValueError(f"Invalid mean function: {mean_function}")
    
    # Generate noise based on covariance structure
    if dependent_noise:
        if target_correlation is not None:
            # Validate target correlation
            if not 0.0 <= target_correlation <= 1.0:
                raise ValueError("target_correlation must be between 0 and 1")
            
            # Create a correlation matrix with the desired average correlation magnitude
            if y_dim > 1:
                # Start with a diagonal correlation matrix (identity)
                corr_matrix = np.eye(y_dim)
                
                # Fill the off-diagonal elements with random correlations of the target magnitude
                for i in range(y_dim):
                    for j in range(i+1, y_dim):  # Only upper triangle to avoid duplicates
                        # Random sign (+1 or -1) * target correlation magnitude
                        sign = np.random.choice([-1, 1])
                        corr_value = sign * target_correlation
                        corr_matrix[i, j] = corr_value
                        corr_matrix[j, i] = corr_value  # Symmetric matrix
                
                # Generate random standard deviations
                std_devs = np.sqrt(np.random.uniform(0.01, noise_scale, y_dim))
                
                # Convert correlation matrix to covariance matrix
                cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
            else:
                # For 1D case, just use a variance
                cov_matrix = np.array([[noise_scale]])
        else:
            # Original approach: Generate a random positive definite covariance matrix
            random_matrix = np.random.randn(y_dim, y_dim)
            cov_matrix = np.dot(random_matrix, random_matrix.T)
            # Ensure reasonable scale
            cov_matrix = noise_scale * cov_matrix / np.max(np.abs(cov_matrix))
        
        # Generate multivariate normal noise
        noise = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
        
        # Calculate and store the actual average correlation if target was specified
        if target_correlation is not None and y_dim > 1:
            # Convert covariance to correlation
            diag_sqrt = np.sqrt(np.diag(cov_matrix))
            actual_corr_matrix = cov_matrix / np.outer(diag_sqrt, diag_sqrt)
            
            # Calculate average of off-diagonal elements (absolute values for magnitude)
            mask = ~np.eye(y_dim, dtype=bool)  # mask to exclude diagonal
            actual_avg_corr = np.mean(np.abs(actual_corr_matrix[mask]))
            noise_args['actual_avg_correlation_magnitude'] = float(actual_avg_corr)
            
            # Also store the actual correlations (not just magnitude)
            noise_args['actual_avg_correlation'] = float(np.mean(actual_corr_matrix[mask]))
    else:
        # Independent noise (diagonal covariance)
        variances = np.random.uniform(0.01, noise_scale, y_dim)
        cov_matrix = np.diag(variances)
        noise = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
    
    # update noise_args with cov_matrix
    noise_args['cov_matrix'] = cov_matrix
    
    # Combine mean and noise
    y = mean + noise
    
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