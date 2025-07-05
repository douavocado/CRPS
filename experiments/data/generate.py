import torch
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_t, chi2, gamma
from scipy.special import kv, gamma as gamma_func
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
                # Ensure the covariance matrix is symmetric and positive-semi-definite
                cov_matrix = (cov_matrix + cov_matrix.T) / 2
                # now ensure that min eigenvalue is positive
                eigvals = np.real(np.linalg.eigvalsh(cov_matrix))
                if np.min(eigvals) < 0:
                    # Add a small positive value to diagonal to make it positive-semi-definite
                    cov_matrix += (abs(np.min(eigvals)) + 1e-6) * np.eye(y_dim)
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

def generate_toy_data_multidim_extended(n_samples=500, x_dim=1, y_dim=1, noise_type='gaussian', 
                                       noise_scale=0.1, mean_function='gaussian',
                                       kernel_type='rbf', kernel_params=None, target_correlation=None, 
                                       **noise_kwargs):
    """
    Generate a multivariate regression dataset with various multivariate noise distributions
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    x_dim : int
        Dimension of input features
    y_dim : int
        Dimension of output features
    noise_type : str
        Type of noise distribution ('gaussian', 'student_t', 'laplace_symmetric', 
        'laplace_asymmetric', 'gamma', 'lognormal')
    noise_scale : float
        Base scale for the noise
    mean_function : str
        The type of function used for transformation of input features. Defaults to 'gaussian', which is a gaussian process with zero mean.
    kernel_type : str
        Type of kernel for GP mean functions ('rbf', 'matern', 'linear')
    kernel_params : dict or None
        Parameters for the kernel. If None, random parameters are generated.
    target_correlation : float or None
        If not None, attempts to generate a covariance matrix with an average correlation coefficient magnitude close to this value.
    **noise_kwargs : dict
        Additional parameters for specific noise distributions:
        - For student_t: df (degrees of freedom, default: 3)
        - For laplace_asymmetric: asymmetry_params (dict with 'location' and 'scale' vectors)
        - For gamma: shape_params (array of shape parameters for each dimension)
        - For lognormal: sigma (standard deviation in log space)
    
    Returns:
    --------
    Dict containing training/testing data and metadata
    """
    # Generate random inputs
    x = np.random.uniform(-3, 3, (n_samples, x_dim))

    # Store noise parameters
    noise_args = {
        'noise_type': noise_type,
        'mean_function': mean_function,
        'target_correlation': target_correlation,
        'noise_scale': noise_scale,
    }
    
    # Generate mean function (same as before)
    if mean_function == 'gaussian':
        # Draw from gaussian process with zero mean
        if kernel_params is None:
            kernel_params = default_kernel_params(kernel_type)
        
        kernel_function = get_kernel_function(kernel_type)
        
        gp_functions = []
        mean = np.zeros((n_samples, y_dim))
        
        for j in range(y_dim):
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
            
            K += 1e-6 * np.eye(n_samples)
            f_values = np.random.multivariate_normal(np.zeros(n_samples), K)
            mean[:, j] = f_values
            
            gp_functions.append({
                'x_train': x.copy(),
                'f_train': f_values.copy(),
                'kernel_type': kernel_type,
                'kernel_params': kernel_params.copy()
            })
        
        noise_args['gp_functions'] = gp_functions
    elif mean_function == 'zero':
        mean = np.zeros((n_samples, y_dim))
    else:
        raise ValueError(f"Invalid mean function: {mean_function}")
    
    # Generate covariance matrix
    if target_correlation is not None and y_dim > 1:
        # Validate target correlation
        if not 0.0 <= target_correlation <= 1.0:
            raise ValueError("target_correlation must be between 0 and 1")
        
        # Create a correlation matrix with the desired average correlation magnitude
        corr_matrix = np.eye(y_dim)
        
        # Fill the off-diagonal elements with random correlations of the target magnitude
        for i in range(y_dim):
            for j in range(i+1, y_dim):
                sign = np.random.choice([-1, 1])
                corr_value = sign * target_correlation
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value
        
        # Generate random standard deviations
        std_devs = np.sqrt(np.random.uniform(0.01, noise_scale, y_dim))
        
        # Convert correlation matrix to covariance matrix
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Ensure positive-semi-definite
        eigvals = np.real(np.linalg.eigvalsh(cov_matrix))
        if np.min(eigvals) < 0:
            cov_matrix += (abs(np.min(eigvals)) + 1e-6) * np.eye(y_dim)
    else:
        # Default covariance structure
        if y_dim > 1:
            random_matrix = np.random.randn(y_dim, y_dim)
            cov_matrix = np.dot(random_matrix, random_matrix.T)
            cov_matrix = noise_scale * cov_matrix / np.max(np.abs(cov_matrix))
        else:
            cov_matrix = np.array([[noise_scale**2]])
    
    # Generate noise based on distribution type
    if noise_type == 'gaussian':
        noise = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
        
    elif noise_type == 'student_t':
        # Multivariate Student-t distribution
        df = noise_kwargs.get('df', 3)
        
        # Generate multivariate t-distribution
        # Method: Generate chi-squared random variable, then scale multivariate normal
        chi2_samples = np.random.chisquare(df, n_samples)
        normal_samples = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
        
        # Scale by chi-squared samples
        scaling_factors = np.sqrt(df / chi2_samples)
        noise = normal_samples * scaling_factors[:, np.newaxis]
        
        noise_args['df'] = df
        
    elif noise_type == 'laplace_symmetric':
        # Symmetric multivariate Laplace distribution
        # Using the construction from the Wikipedia page
        # Generate exponential random variable for mixing
        exponential_samples = np.random.exponential(1, n_samples)
        
        # Generate multivariate normal samples
        normal_samples = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
        
        # Mix according to symmetric Laplace construction
        noise = normal_samples * np.sqrt(exponential_samples)[:, np.newaxis]
        
    elif noise_type == 'laplace_asymmetric':
        # Asymmetric multivariate Laplace distribution
        # Parameters for asymmetry
        asymmetry_params = noise_kwargs.get('asymmetry_params', {})
        location = asymmetry_params.get('location', np.zeros(y_dim))
        scale = asymmetry_params.get('scale', np.ones(y_dim))
        
        # Generate exponential random variable for mixing
        exponential_samples = np.random.exponential(1, n_samples)
        
        # Generate multivariate normal samples
        normal_samples = np.random.multivariate_normal(np.zeros(y_dim), cov_matrix, size=n_samples)
        
        # Apply asymmetry transformation
        noise = (normal_samples * np.sqrt(exponential_samples)[:, np.newaxis] * scale[np.newaxis, :] + 
                location[np.newaxis, :])
        
        noise_args['asymmetry_params'] = {'location': location, 'scale': scale}
        
    elif noise_type == 'gamma':
        # Multivariate Gamma using latent thinned construction
        shape_params = noise_kwargs.get('shape_params', np.ones(y_dim))
        
        if len(shape_params) != y_dim:
            raise ValueError(f"shape_params must have length {y_dim}")
        
        # Generate latent Gamma variables
        # Using the construction from the Dan MacKinlay reference
        # Generate independent gamma variables
        gamma_samples = np.zeros((n_samples, y_dim))
        for i in range(y_dim):
            gamma_samples[:, i] = np.random.gamma(shape_params[i], scale=noise_scale, size=n_samples)
        
        # Apply correlation structure using Gaussian copula transformation
        if target_correlation is not None and y_dim > 1:
            # Transform to uniform using gamma CDF
            uniform_samples = np.zeros_like(gamma_samples)
            for i in range(y_dim):
                uniform_samples[:, i] = gamma.cdf(gamma_samples[:, i], shape_params[i], scale=noise_scale)
            
            # Transform to correlated normal
            normal_samples = np.random.multivariate_normal(np.zeros(y_dim), corr_matrix, size=n_samples)
            
            # Use rank-based correlation
            for i in range(y_dim):
                ranks = np.argsort(np.argsort(normal_samples[:, i]))
                sorted_gamma = np.sort(gamma_samples[:, i])
                gamma_samples[:, i] = sorted_gamma[ranks]
        
        noise = gamma_samples
        noise_args['shape_params'] = shape_params
        
    elif noise_type == 'lognormal':
        # Multivariate lognormal distribution
        sigma = noise_kwargs.get('sigma', 1.0)
        
        # Generate multivariate normal in log space
        log_normal_samples = np.random.multivariate_normal(np.zeros(y_dim), 
                                                          sigma**2 * cov_matrix, 
                                                          size=n_samples)
        
        # Transform to lognormal
        noise = np.exp(log_normal_samples) - 1  # Subtract 1 to center around 0
        
        noise_args['sigma'] = sigma
        
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")
    
    # Store distribution parameters for quantile reconstruction
    noise_args['cov_matrix'] = cov_matrix
    
    # Calculate actual correlation if target was specified
    if target_correlation is not None and y_dim > 1:
        # Calculate empirical correlation of the generated noise
        empirical_corr = np.corrcoef(noise.T)
        mask = ~np.eye(y_dim, dtype=bool)
        noise_args['actual_avg_correlation_magnitude'] = float(np.mean(np.abs(empirical_corr[mask])))
        noise_args['actual_avg_correlation'] = float(np.mean(empirical_corr[mask]))
        noise_args['empirical_correlation_matrix'] = empirical_corr
    
    # Store sample statistics for quantile reconstruction
    noise_args['sample_mean'] = np.mean(noise, axis=0)
    noise_args['sample_cov'] = np.cov(noise.T)
    noise_args['sample_quantiles'] = {
        'q25': np.percentile(noise, 25, axis=0),
        'q50': np.percentile(noise, 50, axis=0),
        'q75': np.percentile(noise, 75, axis=0),
        'q95': np.percentile(noise, 95, axis=0),
        'q99': np.percentile(noise, 99, axis=0)
    }
    
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
        'mean_test': mean_test,
        'noise_samples': noise  # Full noise samples for analysis
    }

def reconstruct_noise_quantiles(noise_args, confidence_levels=[0.95, 0.99]):
    """
    Reconstruct quantile contours/confidence intervals from noise metadata
    
    Parameters:
    -----------
    noise_args : dict
        Noise metadata from generate_toy_data_multidim_extended
    confidence_levels : list
        List of confidence levels to compute (e.g., [0.95, 0.99])
    
    Returns:
    --------
    dict : Dictionary containing quantile information for each confidence level
    """
    noise_type = noise_args['noise_type']
    y_dim = noise_args['sample_mean'].shape[0]
    
    quantile_info = {}
    
    for alpha in confidence_levels:
        if noise_type == 'gaussian':
            # For multivariate Gaussian, use chi-squared quantiles
            chi2_quantile = chi2.ppf(alpha, df=y_dim)
            quantile_info[f'radius_{alpha}'] = np.sqrt(chi2_quantile)
            
        elif noise_type == 'student_t':
            # For multivariate Student-t
            df = noise_args['df']
            # Hotelling's T-squared distribution
            f_quantile = chi2.ppf(alpha, df=y_dim)
            quantile_info[f'radius_{alpha}'] = np.sqrt(f_quantile * (df + y_dim - 1) / (df - 2))
            
        else:
            # For other distributions, use empirical quantiles
            # Use the stored sample quantiles
            q_key = f'q{int(alpha*100)}'
            if q_key in noise_args['sample_quantiles']:
                quantile_info[f'empirical_{alpha}'] = noise_args['sample_quantiles'][q_key]
            else:
                # Interpolate if exact quantile not available
                # This would require the full samples, so we'll use approximation
                quantile_info[f'empirical_{alpha}'] = noise_args['sample_quantiles']['q95']
    
    # Add covariance information
    quantile_info['covariance_matrix'] = noise_args['cov_matrix']
    quantile_info['sample_covariance'] = noise_args['sample_cov']
    
    return quantile_info