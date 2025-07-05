import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

def rbf_kernel(x1, x2, lengthscale, amplitude):
    """
    Radial Basis Function (RBF) kernel, also known as squared exponential kernel.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Input points (n_samples, n_features)
    lengthscale : float
        Length scale parameter
    amplitude : float
        Amplitude parameter
    
    Returns:
    --------
    K : array
        Kernel matrix
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    
    # Compute squared distances
    sq_distances = cdist(x1, x2, metric='sqeuclidean')
    
    # Apply RBF kernel
    K = amplitude * np.exp(-0.5 * sq_distances / lengthscale**2)
    
    return K

def matern_kernel(x1, x2, lengthscale, amplitude, nu):
    """
    Matérn kernel.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Input points (n_samples, n_features)
    lengthscale : float
        Length scale parameter
    amplitude : float
        Amplitude parameter
    nu : float
        Smoothness parameter
    
    Returns:
    --------
    K : array
        Kernel matrix
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    
    # Compute distances
    distances = cdist(x1, x2, metric='euclidean')
    
    # Special cases for common nu values
    if nu == 0.5:
        # Exponential kernel
        K = amplitude * np.exp(-distances / lengthscale)
    elif nu == 1.5:
        # Matern 3/2
        sqrt3_r = np.sqrt(3) * distances / lengthscale
        K = amplitude * (1 + sqrt3_r) * np.exp(-sqrt3_r)
    elif nu == 2.5:
        # Matern 5/2
        sqrt5_r = np.sqrt(5) * distances / lengthscale
        K = amplitude * (1 + sqrt5_r + 5 * distances**2 / (3 * lengthscale**2)) * np.exp(-sqrt5_r)
    else:
        # General Matern kernel
        sqrt2nu_r = np.sqrt(2 * nu) * distances / lengthscale
        
        # Handle the case where distance is 0
        K = np.zeros_like(distances)
        nonzero_mask = distances > 0
        
        if np.any(nonzero_mask):
            K[nonzero_mask] = (amplitude * (2**(1-nu)) / gamma(nu) * 
                              (sqrt2nu_r[nonzero_mask]**nu) * 
                              kv(nu, sqrt2nu_r[nonzero_mask]))
        
        # Set diagonal elements (where distance is 0)
        diag_mask = distances == 0
        K[diag_mask] = amplitude
    
    return K

def linear_kernel(x1, x2, variance, offset):
    """
    Linear kernel.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Input points (n_samples, n_features)
    variance : float
        Variance parameter
    offset : float
        Offset parameter
    
    Returns:
    --------
    K : array
        Kernel matrix
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    
    # Linear kernel: variance * (x1 @ x2.T) + offset
    K = variance * np.dot(x1, x2.T) + offset
    
    return K

def default_kernel_params(kernel_type):
    """
    Generate default kernel parameters for different kernel types.
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel ('rbf', 'matern', 'linear')
    
    Returns:
    --------
    dict : Dictionary of kernel parameters
    """
    if kernel_type == 'rbf':
        return {
            'lengthscale': np.random.uniform(0.5, 2.0),
            'amplitude': np.random.uniform(0.5, 1.5)
        }
    elif kernel_type == 'matern':
        return {
            'lengthscale': np.random.uniform(0.5, 2.0),
            'amplitude': np.random.uniform(0.5, 1.5),
            'nu': np.random.choice([0.5, 1.5, 2.5])  # Common values
        }
    elif kernel_type == 'linear':
        return {
            'variance': np.random.uniform(0.1, 1.0),
            'offset': np.random.uniform(0.0, 0.5)
        }
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def get_kernel_function(kernel_type):
    """
    Get the kernel function for a given kernel type.
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel ('rbf', 'matern', 'linear')
    
    Returns:
    --------
    function : Kernel function
    """
    if kernel_type == 'rbf':
        return rbf_kernel
    elif kernel_type == 'matern':
        return matern_kernel
    elif kernel_type == 'linear':
        return linear_kernel
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def evaluate_gp_mean(x_new, gp_function):
    """
    Evaluate GP mean function at new points using kernel regression.
    
    Parameters:
    -----------
    x_new : array-like
        New input points
    gp_function : dict
        GP function information with keys: 'x_train', 'f_train', 'kernel_type', 'kernel_params'
    
    Returns:
    --------
    mean_new : array
        Mean function values at new points
    """
    x_train = gp_function['x_train']
    f_train = gp_function['f_train']
    kernel_type = gp_function['kernel_type']
    kernel_params = gp_function['kernel_params']
    
    # Get kernel function
    kernel_function = get_kernel_function(kernel_type)
    
    # Compute kernel matrices
    if kernel_type == 'rbf':
        K_train = kernel_function(x_train, x_train, 
                                 kernel_params['lengthscale'], 
                                 kernel_params['amplitude'])
        K_cross = kernel_function(x_new, x_train, 
                                 kernel_params['lengthscale'], 
                                 kernel_params['amplitude'])
    elif kernel_type == 'matern':
        K_train = kernel_function(x_train, x_train, 
                                 kernel_params['lengthscale'], 
                                 kernel_params['amplitude'], 
                                 kernel_params['nu'])
        K_cross = kernel_function(x_new, x_train, 
                                 kernel_params['lengthscale'], 
                                 kernel_params['amplitude'], 
                                 kernel_params['nu'])
    elif kernel_type == 'linear':
        K_train = kernel_function(x_train, x_train, 
                                 kernel_params['variance'], 
                                 kernel_params['offset'])
        K_cross = kernel_function(x_new, x_train, 
                                 kernel_params['variance'], 
                                 kernel_params['offset'])
    
    # Add noise for numerical stability
    K_train += 1e-6 * np.eye(K_train.shape[0])
    
    # Compute mean predictions
    try:
        # Use Cholesky decomposition for numerical stability
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L, f_train)
        alpha = np.linalg.solve(L.T, alpha)
        mean_new = np.dot(K_cross, alpha)
    except np.linalg.LinAlgError:
        # Fall back to standard solve if Cholesky fails
        alpha = np.linalg.solve(K_train, f_train)
        mean_new = np.dot(K_cross, alpha)
    
    return mean_new 