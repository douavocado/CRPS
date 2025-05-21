import torch
import numpy as np
from scipy import special
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os

# Add the project root directory to the system path to import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.mlp_gaussian import MLPGaussian

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_toy_data(n_samples=500, noise_type='student_t', mean_function='gaussian_rbf', **kwargs):
    """Generate a toy 1D regression dataset with heteroscedastic noise"""
    # Generate inputs between -3 and 3
    x = np.random.uniform(-3, 3, n_samples)
    # sort x
    x = np.sort(x)
    
    base_noise_scale = np.random.uniform(0.05, 0.5)
    # generate noise from non-gaussian distributions
    if noise_type == 'student_t':
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
    elif mean_function == 'sawtooth':
        shift = np.random.uniform(-1, 1)
        slope = np.random.uniform(-2.5, 2.5)
        period = np.random.uniform(0.1, 1.0)
        amplitude = np.random.uniform(0.5, 2.5)
        # Implement sawtooth function: rises linearly and then drops sharply
        mean = amplitude * (2 * (np.mod(slope * x + shift, period) / period) - 1)
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
    elif mean_function == 'gaussian_matern':
        # draw from gaussian process with matern kernel
        amplitude = np.random.uniform(0.5, 1.5)
        lengthscale = np.random.uniform(0.5, 2.0)
        # Set mean function to 0
        mean_f = np.zeros_like(x)
        # Compute covariance matrix with matern kernel
        # Compute distance matrix
        dist = np.abs(x[:, None] - x[None, :])
        
        # Matern kernel with nu=3/2
        nu = 1.5
        sqrt_3 = np.sqrt(3)
        cov = amplitude * (1 + sqrt_3 * dist / lengthscale) * np.exp(-sqrt_3 * dist / lengthscale)
        
        # Sample from the Gaussian process with mean 0
        mean = np.random.multivariate_normal(mean_f, cov)
    elif mean_function == 'step':
        # Create a step function with random step positions
        num_steps = np.random.randint(2, 5)
        step_positions = np.sort(np.random.uniform(-2.5, 2.5, num_steps))
        step_heights = np.random.uniform(-1.5, 1.5, num_steps + 1)
        
        mean = np.zeros_like(x)
        for i in range(len(step_positions)):
            mean[x > step_positions[i]] = step_heights[i + 1]
        
        # Initial value for x <= first step position
        mean[x <= step_positions[0]] = step_heights[0]
    else:
        raise ValueError(f"Invalid mean function: {mean_function}")
        
    # Plot and save the mean function
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, 'r-', label='True Mean Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Mean Function: {mean_function}')
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs('experiments/1d_regression/figures', exist_ok=True)
    plt.savefig(f'experiments/1d_regression/figures/mean_function_{mean_function}.png')
    plt.close()
    
    
    y = mean + noise
    # Get the true mean values for train and test sets
    # Create indices for train-test split while preserving order
    n_samples = len(x)
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
    
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_train, y_train, x_test, y_test, noise_args, mean_train, mean_test

def train_model(model, x, y, loss_fn, n_epochs=2000, init_lr=3e-4):
    """Train the model using the specified loss function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 100  # For early stopping
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute loss
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()
        
        # Record loss and update scheduler
        curr_loss = loss.item()
        losses.append(curr_loss)
        scheduler.step(curr_loss)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {curr_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping logic
        if curr_loss < best_loss:
            best_loss = curr_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience or optimizer.param_groups[0]['lr'] < 1e-5:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return losses

def evaluate_predictions(model, x, y_true):
    """Evaluate model predictions and return metrics"""
    with torch.no_grad():
        mean, var = model(x)
    
    mean = mean.numpy().flatten()
    std = np.sqrt(var.numpy().flatten())
    y_true = y_true.numpy().flatten()
    
    # Calculate MSE
    mse = np.mean((mean - y_true) ** 2)
    
    # Make sure we don't get divide by zero or log(0) errors
    safe_var = np.maximum(var.numpy().flatten(), 1e-6)
    
    # Calculate negative log-likelihood
    nll = 0.5 * np.mean(np.log(2 * np.pi * safe_var) + 
                       ((y_true - mean) ** 2) / safe_var)
    
    # Calculate CRPS for Gaussian predictions using the correct formula
    z = (y_true - mean) / std
    norm_pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    norm_cdf = 0.5 * (1 + special.erf(z / np.sqrt(2.0)))
    
    # CRPS formula - absolute value to ensure positive
    crps = np.mean(std * (-1 / np.sqrt(np.pi) + 2 * norm_pdf + z * (2 * norm_cdf - 1)))
    
    return {
        'mse': mse,
        'nll': nll,
        'crps': crps
    }

def plot_results(x, y, true_mean, models, titles, noise_args, noise_type):
    """Plot the results of different models"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Sort the x values for smooth plotting
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Plot the ground truth with confidence bands
    for ax in axes:
        ax.scatter(x, y, color='black', s=10, alpha=0.6, label='Data')
        ax.plot(x_sorted, true_mean, 'r-', label='True Function')
        
        # Plot confidence bands for the true data-generating process
        if noise_type == 'student_t':
            # For Student's t-distribution, 95% CI uses t-distribution quantiles
            # When df > 2, we can calculate the 95% CI using the t-distribution
            if noise_args['df'] > 2:
                # Calculate the t-value for 97.5th percentile (for 95% CI)
                t_value = special.stdtrit(noise_args['df'], 0.975)
                # Scale by base_noise_scale
                upper = true_mean + noise_args['base_noise_scale'] * t_value
                lower = true_mean - noise_args['base_noise_scale'] * t_value
            else:
                # For df ≤ 2, variance is undefined or infinite, use a reasonable approximation
                t_value = 3.0  # A conservative estimate
                upper = true_mean + noise_args['base_noise_scale'] * t_value
                lower = true_mean - noise_args['base_noise_scale'] * t_value
        elif noise_type == 'cauchy':
            # variance is undefined for cauchy distribution, so we use a large value
            upper = true_mean + 3 * noise_args['base_noise_scale']
            lower = true_mean - 3 * noise_args['base_noise_scale']
        elif noise_type == 'uniform':
            # For uniform distribution U(0,1), 95% CI is approximately ±0.95*range/2
            # For a uniform distribution scaled by base_noise_scale
            upper = true_mean + 0.95 * noise_args['base_noise_scale']
            lower = true_mean - 0.95 * noise_args['base_noise_scale']
        elif noise_type == 'exponential':
            # For exponential distribution, 95% CI is asymmetric
            # Using quantiles of exponential distribution with scale parameter
            lower_quantile = -np.log(1-0.025) / (noise_args['scale'])  # 2.5th percentile
            upper_quantile = -np.log(1-0.975) / (noise_args['scale'])  # 97.5th percentile
            print("lower_quantile: ", lower_quantile)
            print("upper_quantile: ", upper_quantile)
            upper = true_mean + upper_quantile * noise_args['base_noise_scale']
            lower = true_mean + lower_quantile * noise_args['base_noise_scale']
        elif noise_type == 'poisson':
            # For Poisson distribution, 95% CI can be approximated using normal approximation
            # when lambda is large enough (typically > 10)
            # For Poisson(λ), the variance is also λ
            lambda_val = noise_args['scale']
            if lambda_val > 10:
                # Normal approximation: mean ± 1.96 * sqrt(lambda)
                upper = true_mean + 1.96 * np.sqrt(lambda_val) * noise_args['base_noise_scale']
                lower = true_mean - 1.96 * np.sqrt(lambda_val) * noise_args['base_noise_scale']
            else:
                # For smaller lambda, use quantiles from Poisson distribution
                # We can approximate using the fact that Poisson is discrete
                lower_quantile = special.poisson.ppf(0.025, lambda_val)
                upper_quantile = special.poisson.ppf(0.975, lambda_val)
                upper = true_mean + upper_quantile * noise_args['base_noise_scale']
                lower = true_mean - lower_quantile * noise_args['base_noise_scale']
        elif noise_type == 'gamma':
            # For gamma distribution, 95% CI can be calculated using gamma distribution quantiles
            # Gamma(shape, scale) has mean = shape * scale and variance = shape * scale^2
            shape = noise_args['shape']
            scale = noise_args['scale']
            
            # Calculate the quantiles for 2.5th and 97.5th percentiles
            lower_quantile = special.gammaincinv(shape, 0.025) * scale
            upper_quantile = special.gammaincinv(shape, 0.975) * scale
            
            # Scale by base_noise_scale
            upper = true_mean + upper_quantile * noise_args['base_noise_scale']
            lower = true_mean + lower_quantile * noise_args['base_noise_scale']
        elif noise_type == 'lognormal':
            # For lognormal distribution, 95% CI can be calculated using lognormal distribution quantiles
            mean = noise_args['mean']
            sigma = noise_args['scale']
            
            # Calculate the quantiles for 2.5th and 97.5th percentiles
            lower_quantile = np.exp(mean + sigma * special.ndtri(0.025))
            upper_quantile = np.exp(mean + sigma * special.ndtri(0.975))
            
            # Scale by base_noise_scale
            upper = true_mean + upper_quantile * noise_args['base_noise_scale']
            lower = true_mean + lower_quantile * noise_args['base_noise_scale']

        ax.fill_between(x_sorted, lower, upper, color='red', alpha=0.2, label='True 95% CI')
    
    # Plot the model predictions
    for i, (model, title) in enumerate(zip(models, titles)):
        with torch.no_grad():
            mean, var = model(torch.FloatTensor(x_sorted).view(-1, 1))
            
        mean = mean.numpy().flatten()
        std = np.sqrt(var.numpy().flatten())
        
        # Plot mean prediction
        axes[i].plot(x_sorted, mean, 'b-', label='Predicted Mean')
        
        # Plot confidence bands
        upper = mean + 2 * std
        lower = mean - 2 * std
        axes[i].fill_between(x_sorted, lower, upper, color='blue', alpha=0.2, label='Predicted 95% CI')
        
        axes[i].set_title(title + f"data-generating process: {noise_type}", fontsize=16)
        axes[i].set_xlabel('x', fontsize=14)
        axes[i].set_ylabel('y', fontsize=14)
        axes[i].legend(fontsize=12)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/1d_regression/figures/gaussian_not_well_specified_model_comparison.png')
    plt.show()

def main():
    noise_type = 'lognormal'
    # Generate toy data
    x_train, y_train, x_test, y_test, x_np, y_np, x_tnp, y_tnp, noise_args, true_mean_train, true_mean_test = generate_toy_data(n_samples=5000, mean_function='gaussian_rbf', noise_type=noise_type)
    
    # Set different random seeds for initializing models
    torch.manual_seed(42)
    model_nll = MLPGaussian(input_size=1, hidden_size=64)
    
    torch.manual_seed(43)  # Different seed for CRPS model
    model_crps = MLPGaussian(input_size=1, hidden_size=64)
    
    # Train with negative log-likelihood loss
    print("Training with Negative Log-Likelihood Loss...")
    nll_losses = train_model(model_nll, x_train, y_train, model_nll.log_likelihood_loss)
    
    # Train with CRPS loss
    print("\nTraining with CRPS Loss...")
    crps_losses = train_model(model_crps, x_train, y_train, model_crps.crps_loss)
    
    # Evaluate models
    print("\nEvaluating models...")
    nll_metrics = evaluate_predictions(model_nll, x_test, y_test)
    crps_metrics = evaluate_predictions(model_crps, x_test, y_test)
    
    print("\nNLL Model Metrics:")
    print(f"MSE: {nll_metrics['mse']:.6f}")
    print(f"NLL: {nll_metrics['nll']:.6f}")
    print(f"CRPS: {nll_metrics['crps']:.6f}")
    
    print("\nCRPS Model Metrics:")
    print(f"MSE: {crps_metrics['mse']:.6f}")
    print(f"NLL: {crps_metrics['nll']:.6f}")
    print(f"CRPS: {crps_metrics['crps']:.6f}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(nll_losses, label='NLL Loss')
    plt.plot(crps_losses, label='CRPS Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/1d_regression/figures/gaussian_not_well_specified_training_curves.png')
    
    # Plot model predictions
    plot_results(x_tnp, y_tnp, true_mean_test, [model_nll, model_crps], 
                ['Negative Log-Likelihood Model', 'CRPS Model'], 
                noise_args, noise_type)

if __name__ == '__main__':
    main()
