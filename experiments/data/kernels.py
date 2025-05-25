import numpy as np
from scipy.special import gamma, kv

def rbf_kernel(X1, X2, lengthscale, amplitude):
    """
    RBF/Squared Exponential kernel
    
    Parameters:
    -----------
    X1 : array-like, shape (n1, d)
        First set of input points
    X2 : array-like, shape (n2, d)
        Second set of input points
    lengthscale : float
        Length scale parameter that controls smoothness
    amplitude : float
        Amplitude parameter that controls output variance
        
    Returns:
    --------
    array-like, shape (n1, n2)
        Kernel matrix between X1 and X2
    """
    # Compute pairwise squared distances
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return amplitude**2 * np.exp(-0.5 * dists / lengthscale**2)

def matern_kernel(X1, X2, lengthscale, amplitude, nu):
    """
    Matérn kernel with various smoothness parameters
    
    Parameters:
    -----------
    X1 : array-like, shape (n1, d)
        First set of input points
    X2 : array-like, shape (n2, d)
        Second set of input points
    lengthscale : float
        Length scale parameter that controls smoothness
    amplitude : float
        Amplitude parameter that controls output variance
    nu : float
        Smoothness parameter (common values: 0.5, 1.5, 2.5)
        
    Returns:
    --------
    array-like, shape (n1, n2)
        Kernel matrix between X1 and X2
    """
    dists = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
    
    if nu == 0.5:
        # Exponential kernel (special case of Matérn)
        return amplitude**2 * np.exp(-dists / lengthscale)
    elif nu == 1.5:
        scaled_dists = np.sqrt(3) * dists / lengthscale
        return amplitude**2 * (1 + scaled_dists) * np.exp(-scaled_dists)
    elif nu == 2.5:
        scaled_dists = np.sqrt(5) * dists / lengthscale
        return amplitude**2 * (1 + scaled_dists + scaled_dists**2/3) * np.exp(-scaled_dists)
    else:
        # General Matérn kernel
        scaled_dists = np.sqrt(2 * nu) * dists / lengthscale
        scaled_dists = np.where(scaled_dists == 0, 1e-10, scaled_dists)  # Avoid division by zero
        K = amplitude**2 * (2**(1-nu) / gamma(nu)) * (scaled_dists**nu) * kv(nu, scaled_dists)
        K = np.where(dists == 0, amplitude**2, K)  # Handle diagonal elements
        return K

def linear_kernel(X1, X2, variance, offset):
    """
    Linear kernel (dot product with optional bias)
    
    Parameters:
    -----------
    X1 : array-like, shape (n1, d)
        First set of input points
    X2 : array-like, shape (n2, d)
        Second set of input points
    variance : float
        Variance parameter scaling the dot product
    offset : float
        Bias term added to the dot product
        
    Returns:
    --------
    array-like, shape (n1, n2)
        Kernel matrix between X1 and X2
    """
    return variance * (np.dot(X1, X2.T) + offset)

def get_kernel_function(kernel_type):
    """
    Return the appropriate kernel function based on kernel_type
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel ('rbf', 'matern', 'linear')
        
    Returns:
    --------
    function
        The corresponding kernel function
    """
    if kernel_type == 'rbf':
        return rbf_kernel
    elif kernel_type == 'matern':
        return matern_kernel
    elif kernel_type == 'linear':
        return linear_kernel
    else:
        raise ValueError(f"Kernel type {kernel_type} not implemented")

def default_kernel_params(kernel_type, random=True):
    """
    Return default parameters for the specified kernel type
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel ('rbf', 'matern', 'linear')
    random : bool
        If True, return random parameters; if False, return fixed defaults
        
    Returns:
    --------
    dict
        Dictionary of kernel parameters
    """
    if kernel_type == 'rbf':
        if random:
            return {
                'lengthscale': np.random.uniform(0.5, 2.0),
                'amplitude': np.random.uniform(0.5, 1.5)
            }
        else:
            return {'lengthscale': 1.0, 'amplitude': 1.0}
            
    elif kernel_type == 'matern':
        if random:
            return {
                'lengthscale': np.random.uniform(0.5, 2.0),
                'amplitude': np.random.uniform(0.5, 1.5),
                'nu': np.random.choice([0.5, 1.5, 2.5])  # Common values for Matérn
            }
        else:
            return {'lengthscale': 1.0, 'amplitude': 1.0, 'nu': 1.5}
            
    elif kernel_type == 'linear':
        if random:
            return {
                'variance': np.random.uniform(0.5, 1.5),
                'offset': np.random.uniform(-0.5, 0.5)
            }
        else:
            return {'variance': 1.0, 'offset': 0.0}
            
    else:
        raise ValueError(f"Default parameters not implemented for kernel type: {kernel_type}")

def evaluate_gp_mean(x_new, gp_func):
    """
    Evaluate GP mean function at new points using stored GP function information
    
    Parameters:
    -----------
    x_new : array-like, shape (n_new, d)
        New input points where to evaluate the GP
    gp_func : dict
        Dictionary containing GP function information:
        - x_train: Training inputs
        - f_train: GP function values at training inputs
        - kernel_type: Type of kernel
        - kernel_params: Kernel parameters
        
    Returns:
    --------
    array-like, shape (n_new,)
        Mean function values at new points
    """
    # Extract GP function parameters
    x_train = gp_func['x_train']
    f_train = gp_func['f_train']
    kernel_type = gp_func['kernel_type']
    kernel_params = gp_func['kernel_params']
    
    # Get kernel function
    kernel_function = get_kernel_function(kernel_type)
    
    # Compute cross-covariance between new points and training points
    if kernel_type == 'rbf':
        K_cross = kernel_function(x_new, x_train, 
                              kernel_params['lengthscale'], 
                              kernel_params['amplitude'])
        # Compute training covariance matrix
        K_train = kernel_function(x_train, x_train,
                              kernel_params['lengthscale'], 
                              kernel_params['amplitude'])
    elif kernel_type == 'matern':
        K_cross = kernel_function(x_new, x_train,
                              kernel_params['lengthscale'],
                              kernel_params['amplitude'],
                              kernel_params['nu'])
        K_train = kernel_function(x_train, x_train,
                              kernel_params['lengthscale'],
                              kernel_params['amplitude'],
                              kernel_params['nu'])
    elif kernel_type == 'linear':
        K_cross = kernel_function(x_new, x_train,
                              kernel_params['variance'],
                              kernel_params['offset'])
        K_train = kernel_function(x_train, x_train,
                              kernel_params['variance'],
                              kernel_params['offset'])
    
    # Add jitter to training covariance for numerical stability
    K_train += 1e-6 * np.eye(len(x_train))
    
    # GP predictive mean: K_cross @ K_train^{-1} @ f_train
    return K_cross @ np.linalg.solve(K_train, f_train) 