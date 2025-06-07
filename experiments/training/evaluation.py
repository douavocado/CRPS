import torch
import numpy as np
from scipy.stats import chi2, gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from common.losses import crps_loss_general, energy_score_loss
from ..visualisation.plotting import predict_samples


def create_evaluation_config(model_type='MLPSampler', evaluation_type='loss', loss_type='crps', **kwargs):
    """
    Create an evaluation configuration dictionary with default values.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('MLPSampler' or 'SimpleAffineNormal')
    evaluation_type : str
        Type of evaluation ('loss' or 'metrics')
    loss_type : str
        Type of loss function for loss evaluation ('crps', 'energy_score', or 'log_likelihood')
    **kwargs : dict
        Additional configuration parameters to override defaults
        
    Returns:
    --------
    dict
        Complete evaluation configuration dictionary
    """
    # Validate combinations
    if model_type == 'SimpleAffineNormal':
        if loss_type not in ['log_likelihood', 'energy_score', 'crps']:
            raise ValueError(f"Loss type '{loss_type}' not supported for SimpleAffineNormal")
    elif model_type == 'MLPSampler':
        if loss_type not in ['crps', 'energy_score']:
            raise ValueError(f"Loss type '{loss_type}' not supported for MLPSampler")
    else:
        raise ValueError(f"Model type '{model_type}' not recognized")
    
    if evaluation_type not in ['loss', 'metrics']:
        raise ValueError(f"Evaluation type '{evaluation_type}' not recognized. Use 'loss' or 'metrics'")
    
    # Default configuration
    config = {
        # Model and evaluation configuration
        'model_type': model_type,
        'evaluation_type': evaluation_type,
        'loss_type': loss_type,
        
        # Evaluation parameters
        'n_samples': 1000,  # Higher default for evaluation
        'device': 'cpu',
        
        # Metrics evaluation specific
        'calibration_levels': [0.5, 0.9, 0.95, 0.99],  # For MLPSampler
        'confidence_levels': [0.1, 0.2, 0.3],  # For SimpleAffineNormal (alpha values)
    }
    
    # Update with provided kwargs
    config.update(kwargs)
    
    return config


def evaluate_model(model, data_loader, evaluation_config):
    """
    Unified evaluation function for all model types and evaluation types.
    
    Parameters:
    -----------
    model : nn.Module
        The model to evaluate (MLPSampler, SimpleAffineNormal, etc.)
    data_loader : DataLoader
        DataLoader for evaluation data
    evaluation_config : dict
        Configuration dictionary containing all evaluation parameters.
        Use create_evaluation_config() for easy configuration creation.
        Required keys:
        - 'model_type': str ('MLPSampler' or 'SimpleAffineNormal')
        - 'evaluation_type': str ('loss' or 'metrics')
        - 'loss_type': str ('crps', 'energy_score', or 'log_likelihood') - for loss evaluation
        
        Optional keys with defaults:
        - 'n_samples': int (default: 1000)
        - 'device': str (default: 'cpu')
        - 'calibration_levels': list (default: [0.5, 0.9, 0.95, 0.99]) - for MLPSampler metrics
        - 'confidence_levels': list (default: [0.1, 0.2, 0.3]) - for SimpleAffineNormal metrics
        
    Returns:
    --------
    float or dict
        If evaluation_type is 'loss': returns float (average loss)
        If evaluation_type is 'metrics': returns dict with metrics
    """
    evaluation_type = evaluation_config.get('evaluation_type', 'loss')
    
    if evaluation_type == 'loss':
        return _evaluate_loss(model, data_loader, evaluation_config)
    elif evaluation_type == 'metrics':
        return _evaluate_metrics(model, data_loader, evaluation_config)
    else:
        raise ValueError(f"Evaluation type '{evaluation_type}' not recognized")


def _evaluate_loss(model, data_loader, evaluation_config):
    """Evaluate model loss on test data."""
    model_type = evaluation_config.get('model_type', 'MLPSampler')
    loss_type = evaluation_config.get('loss_type', 'crps')
    n_samples = evaluation_config.get('n_samples', 1000)
    device = evaluation_config.get('device', 'cpu')
    
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if model_type == 'SimpleAffineNormal':
                if loss_type == 'log_likelihood':
                    log_likelihood = model.log_likelihood(y_batch)
                    loss = -log_likelihood.mean()
                elif loss_type == 'energy_score':
                    batch_size = y_batch.shape[0]
                    # Generate samples
                    samples = model.forward(n_samples=n_samples, batch_size=batch_size)
                    # Compute energy score loss
                    energy_scores = energy_score_loss(samples, y_batch)
                    loss = energy_scores.mean()
                elif loss_type == 'crps':
                    loss = model.crps_loss(y_batch, n_samples=n_samples)
                else:
                    raise ValueError(f"Loss type '{loss_type}' not supported for SimpleAffineNormal")
                    
            elif model_type == 'MLPSampler':
                if loss_type == 'crps':
                    loss = model.crps_loss(x_batch, y_batch, n_samples=n_samples)
                elif loss_type == 'energy_score':
                    loss = model.energy_score_loss(x_batch, y_batch, n_samples=n_samples)
                else:
                    raise ValueError(f"Loss type '{loss_type}' not supported for MLPSampler")
            else:
                raise ValueError(f"Model type '{model_type}' not recognized")
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches


def _evaluate_metrics(model, data_loader, evaluation_config):
    """Evaluate comprehensive metrics on test data."""
    model_type = evaluation_config.get('model_type', 'MLPSampler')
    n_samples = evaluation_config.get('n_samples', 1000)
    device = evaluation_config.get('device', 'cpu')
    
    model.eval()
    model.to(device)
    
    # Collect all data from data_loader
    all_x, all_y = [], []
    for x_batch, y_batch in data_loader:
        all_x.append(x_batch)
        all_y.append(y_batch)
    
    x_test = torch.cat(all_x, dim=0).to(device)
    y_test = torch.cat(all_y, dim=0).to(device)
    
    if model_type == 'SimpleAffineNormal':
        return _evaluate_affine_metrics(model, y_test, evaluation_config)
    elif model_type == 'MLPSampler':
        return _evaluate_mlp_metrics(model, x_test, y_test, evaluation_config)
    else:
        raise ValueError(f"Model type '{model_type}' not recognized")


def _evaluate_affine_metrics(model, y_test, evaluation_config):
    """Evaluate metrics for SimpleAffineNormal model."""
    n_samples = evaluation_config.get('n_samples', 1000)
    device = evaluation_config.get('device', 'cpu')
    confidence_levels = evaluation_config.get('confidence_levels', [0.1, 0.2, 0.3])
    
    with torch.no_grad():
        # Generate samples from the model
        samples = model.forward(n_samples=n_samples, batch_size=y_test.shape[0])
        # Shape: [test_batch_size, n_samples, output_dim]
        
        # Get mean and covariance
        mean, cov = model.get_mean_and_covariance()
        mean_pred = mean.unsqueeze(0).repeat(y_test.shape[0], 1)  # [test_batch_size, output_dim]
        
        # Calculate MSE
        mse = torch.mean((mean_pred - y_test) ** 2).item()
        
        # Calculate log likelihood
        log_likelihood = model.log_likelihood(y_test).mean().item()
        
        # Calculate CRPS using samples
        crps = crps_loss_general(samples, y_test).mean().item()
        
        # Calculate calibration (coverage probabilities)
        calibration = {}
        for alpha in confidence_levels:
            # For multivariate case, we need to compute Mahalanobis distance
            diff = y_test - mean_pred  # [batch_size, output_dim]
            
            # Add regularization for numerical stability
            cov_reg = cov + 1e-6 * torch.eye(cov.shape[0], device=device)
            cov_inv = torch.inverse(cov_reg)
            
            # Mahalanobis distance squared
            mahal_dist_sq = torch.sum(diff * torch.matmul(diff, cov_inv), dim=1)
            
            # Chi-squared quantile for the given confidence level
            chi2_quantile = chi2.ppf(1 - alpha, df=y_test.shape[1])
            
            # Coverage probability
            coverage = torch.mean((mahal_dist_sq <= chi2_quantile).float()).item()
            calibration[f'{int((1-alpha)*100)}%'] = coverage
    
    return {
        'mse': mse,
        'crps': crps,
        'log_likelihood': log_likelihood,
        'calibration': calibration
    }


def _evaluate_mlp_metrics(model, x_test, y_test, evaluation_config):
    """Evaluate metrics for MLPSampler model."""
    n_samples = evaluation_config.get('n_samples', 1000)
    device = evaluation_config.get('device', 'cpu')
    calibration_levels = evaluation_config.get('calibration_levels', [0.5, 0.9, 0.95, 0.99])
    
    # Generate samples
    samples = predict_samples(model, x_test, n_samples=n_samples, device=device)
    
    # Convert to numpy
    samples_np = samples.cpu().numpy()  # shape: [batch_size, n_samples, output_size]
    y_test_np = y_test.cpu().numpy()    # shape: [batch_size, output_size]
    
    # Calculate mean predictions
    mean_preds = samples_np.mean(axis=1)  # shape: [batch_size, output_size]
    
    # Calculate MSE
    mse = mean_squared_error(y_test_np, mean_preds)
    
    # Calculate CRPS
    samples_tensor = torch.from_numpy(samples_np).to(device)
    crps = crps_loss_general(samples_tensor, y_test).mean().item()
    
    # Calculate calibration (percentage of true values within prediction intervals)
    calibration = {}
    for alpha in calibration_levels:
        lower = np.percentile(samples_np, (1-alpha)/2 * 100, axis=1)
        upper = np.percentile(samples_np, (1+alpha)/2 * 100, axis=1)
        in_interval = np.logical_and(y_test_np >= lower, y_test_np <= upper)
        calibration[f'{alpha:.2f}'] = np.mean(in_interval)
    
    # Calculate log likelihood using density estimation
    log_likelihoods = []
    batch_size, n_samples_actual, output_size = samples_np.shape
    
    for i in range(batch_size):
        sample_set = samples_np[i]  # shape: [n_samples, output_size]
        true_value = y_test_np[i]   # shape: [output_size]
        
        try:
            if output_size == 1:
                # For 1D output, use Gaussian KDE
                sample_set_1d = sample_set.flatten()  # shape: [n_samples]
                if len(np.unique(sample_set_1d)) > 1:  # Check for variance
                    kde = gaussian_kde(sample_set_1d)
                    density_value = kde(true_value[0])[0]
                    # Avoid log(0) by setting a minimum density value
                    density_value = max(density_value, 1e-10)
                    log_likelihood = np.log(density_value)
                else:
                    # If all samples are identical, use a small probability
                    log_likelihood = -10.0
            else:
                # For multidimensional output, use Gaussian Mixture Model
                # Try different numbers of components and select based on BIC
                best_gmm = None
                best_bic = np.inf
                
                for n_components in range(1, min(6, n_samples_actual // 10 + 1)):
                    try:
                        gmm = GaussianMixture(
                            n_components=n_components, 
                            covariance_type='full',
                            random_state=42,
                            max_iter=100
                        )
                        gmm.fit(sample_set)
                        bic = gmm.bic(sample_set)
                        
                        if bic < best_bic:
                            best_bic = bic
                            best_gmm = gmm
                    except:
                        continue
                
                if best_gmm is not None:
                    # Evaluate log likelihood at the true value
                    log_likelihood = best_gmm.score_samples([true_value])[0]
                else:
                    # Fallback: assume a simple multivariate Gaussian
                    try:
                        mean = np.mean(sample_set, axis=0)
                        cov = np.cov(sample_set.T)
                        # Add small regularisation to avoid singular covariance
                        cov += np.eye(output_size) * 1e-6
                        
                        # Calculate log likelihood using multivariate normal
                        diff = true_value - mean
                        log_likelihood = -0.5 * (
                            output_size * np.log(2 * np.pi) +
                            np.log(np.linalg.det(cov)) +
                            diff.T @ np.linalg.inv(cov) @ diff
                        )
                    except:
                        log_likelihood = -10.0
                        
        except Exception as e:
            # If density estimation fails, assign a low log likelihood
            log_likelihood = -10.0
            
        log_likelihoods.append(log_likelihood)
    
    # Calculate mean log likelihood
    mean_log_likelihood = np.mean(log_likelihoods)
    
    return {
        'mse': mse,
        'crps': crps,
        'calibration': calibration,
        'log_likelihood': mean_log_likelihood
    }


# Legacy functions for backward compatibility
def evaluate_affine_model_energy(model, test_loader, device='cpu', n_samples=100):
    """Legacy wrapper for SimpleAffineNormal with energy score loss evaluation."""
    evaluation_config = create_evaluation_config(
        model_type='SimpleAffineNormal',
        evaluation_type='loss',
        loss_type='energy_score',
        device=device,
        n_samples=n_samples
    )
    return evaluate_model(model, test_loader, evaluation_config)


def evaluate_affine_model(model, test_loader, device='cpu'):
    """Legacy wrapper for SimpleAffineNormal with log likelihood loss evaluation."""
    evaluation_config = create_evaluation_config(
        model_type='SimpleAffineNormal',
        evaluation_type='loss',
        loss_type='log_likelihood',
        device=device
    )
    return evaluate_model(model, test_loader, evaluation_config)


def evaluate_affine_metrics(model, y_test, n_samples=1000, device='cpu'):
    """Legacy wrapper for SimpleAffineNormal metrics evaluation."""
    # Create a dummy DataLoader for the unified interface
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(y_test.shape[0], 1), y_test),
        batch_size=len(y_test)
    )
    
    evaluation_config = create_evaluation_config(
        model_type='SimpleAffineNormal',
        evaluation_type='metrics',
        device=device,
        n_samples=n_samples
    )
    return evaluate_model(model, test_loader, evaluation_config)


def evaluate_metrics(model, x_test, y_test, n_samples=100, device='cpu'):
    """Legacy wrapper for MLPSampler metrics evaluation."""
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=len(x_test)
    )
    
    evaluation_config = create_evaluation_config(
        model_type='MLPSampler',
        evaluation_type='metrics',
        device=device,
        n_samples=n_samples
    )
    return evaluate_model(model, test_loader, evaluation_config)