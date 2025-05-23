import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchnaut.crps import EpsilonSampler
import scipy.special as special



# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our model
from models.mlp_crps_sampler import MLPSampler
from experiments.data.generate import generate_toy_data_1d



def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=500, lr=0.001, batch_size=64, n_samples=100):
    """Train the model using CRPS loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Batch training
        permutation = torch.randperm(x_train.size(0))
        total_loss = 0.0
        
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.crps_loss(batch_x, batch_y, n_samples=n_samples)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(indices)
        
        avg_train_loss = total_loss / x_train.size(0)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_loss = model.crps_loss(x_test, y_test, n_samples=n_samples).item()
            test_losses.append(test_loss)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def plot_results(model, x_train, y_train, x_test, y_test, mean_train, mean_test, noise_type, noise_args, n_samples=500):
    """Plot the results of the model"""
    # Create directory if it doesn't exist
    os.makedirs('experiments/1d_regression/figures', exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate samples for visualization
    with torch.no_grad():
        # Use more samples for visualization
        with EpsilonSampler.n_samples(n_samples):
            train_samples = model(x_train)
            test_samples = model(x_test)
    
    # Convert to numpy for plotting
    x_train_np = x_train.numpy().flatten()
    y_train_np = y_train.numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    y_test_np = y_test.numpy().flatten()
    
    train_samples_np = train_samples.numpy()
    test_samples_np = test_samples.numpy()
    
    # Calculate percentiles for prediction intervals
    train_mean = np.mean(train_samples_np, axis=1)
    train_lower = np.percentile(train_samples_np, 5, axis=1)
    train_upper = np.percentile(train_samples_np, 95, axis=1)

    test_mean = np.mean(test_samples_np, axis=1)
    test_lower = np.percentile(test_samples_np, 5, axis=1)
    test_upper = np.percentile(test_samples_np, 95, axis=1)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    # if training data is too much, sample random 500 points to plot
    if x_train_np.shape[0] > 500:
        random_indices = np.random.choice(x_train_np.shape[0], size=500, replace=False)
        plot_x_train_np = x_train_np[random_indices]
        plot_y_train_np = y_train_np[random_indices]
    else:
        plot_x_train_np = x_train_np
        plot_y_train_np = y_train_np
    
    plt.scatter(plot_x_train_np, plot_y_train_np, color='blue', alpha=0.5, label='Training Data')
    
    # Calculate true 90% prediction intervals based on noise type
    base_noise_scale = noise_args['base_noise_scale']

    true_lower = None
    true_upper = None
    if noise_type == 'gaussian':
        mean_noise = 0 # mean of gaussian is 0
        true_upper = mean_train + 1.645 * base_noise_scale # 95 and 5th percentiles of normal distribution
        true_lower = mean_train - 1.645 * base_noise_scale
    elif noise_type == 'student_t':
        df = noise_args['df']
        mean_noise = 0 # mean of student t for df > 1 is 0
        # For Student's t-distribution, 90% CI uses t-distribution quantiles
        if df > 2:
            # Calculate the t-value for 95th and 5th percentiles
            t_value_upper = special.stdtrit(df, 0.95)
            t_value_lower = special.stdtrit(df, 0.05)
            # Scale by base_noise_scale
            true_upper = mean_train + base_noise_scale * t_value_upper
            true_lower = mean_train + base_noise_scale * t_value_lower
        else:
            # For df ≤ 2, variance is undefined or infinite, use a reasonable approximation
            t_value = 2.0  # A conservative estimate for 90% interval
            true_upper = mean_train + base_noise_scale * t_value
            true_lower = mean_train - base_noise_scale * t_value
            mean_noise = 0 # mean of student t for df > 1 is 0
    elif noise_type == 'cauchy':
        # For Cauchy distribution, use quantile function
        upper_quantile = special.stdtrit(1, 0.95)  # Cauchy is t with df=1
        lower_quantile = special.stdtrit(1, 0.05)
        true_upper = mean_train + base_noise_scale * upper_quantile
        true_lower = mean_train + base_noise_scale * lower_quantile
        mean_noise = 0 # mean of cauchy is undefined, but for visualization purposes we can set it to 0
    elif noise_type == 'uniform':
        # For uniform distribution U(0,1), 90% CI is between [0.05, 0.95] * base_noise_scale
        true_upper = mean_train + 0.95 * base_noise_scale
        true_lower = mean_train + 0.05 * base_noise_scale
        mean_noise = base_noise_scale * 0.5
    
    elif noise_type == 'exponential':
        scale = noise_args.get('scale', 1)
        # For exponential distribution, 90% CI is asymmetric
        # Using quantiles of exponential distribution with scale parameter
        lower_quantile = -np.log(1-0.05) / scale  # 5th percentile
        upper_quantile = -np.log(1-0.95) / scale  # 95th percentile
        true_upper = mean_train + upper_quantile * base_noise_scale
        true_lower = mean_train + lower_quantile * base_noise_scale
        mean_noise = base_noise_scale * (1/scale)
    
    elif noise_type == 'poisson':
        scale = noise_args.get('scale', 1)  # lambda parameter
        # For Poisson distribution, use quantiles
        if scale > 10:
            # Normal approximation for large lambda
            true_upper = mean_train + 1.645 * np.sqrt(scale) * base_noise_scale  # 95th percentile
            true_lower = mean_train - 1.645 * np.sqrt(scale) * base_noise_scale  # 5th percentile
        else:
            # For smaller lambda, use quantiles from Poisson distribution
            lower_quantile = special.poisson.ppf(0.05, scale)
            upper_quantile = special.poisson.ppf(0.95, scale)
            true_upper = mean_train + upper_quantile * base_noise_scale
            true_lower = mean_train + lower_quantile * base_noise_scale
        
        mean_noise = base_noise_scale * scale
    
    elif noise_type == 'gamma':
        shape = noise_args.get('shape', 1)
        scale = noise_args.get('scale', 1)
        # For gamma distribution, use quantiles
        lower_quantile = special.gammaincinv(shape, 0.05) * scale
        upper_quantile = special.gammaincinv(shape, 0.95) * scale
        true_upper = mean_train + upper_quantile * base_noise_scale
        true_lower = mean_train + lower_quantile * base_noise_scale
        mean_noise = base_noise_scale * scale * shape
    elif noise_type == 'lognormal':
        mean_param = noise_args.get('mean', 0)
        scale_param = noise_args.get('scale', 1)
        # Calculate the quantiles for 5th and 95th percentiles
        lower_quantile = np.exp(mean_param + scale_param * special.ndtri(0.05))
        upper_quantile = np.exp(mean_param + scale_param * special.ndtri(0.95))
        # Scale by base_noise_scale and add to mean
        true_upper = mean_train + upper_quantile * base_noise_scale
        true_lower = mean_train + lower_quantile * base_noise_scale
        mean_noise = base_noise_scale * np.exp(mean_param) * (np.exp(scale_param**2/2))
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

    # Plot true 90% prediction interval if calculated
    if true_lower is not None and true_upper is not None:
        plt.fill_between(x_train_np, true_lower, true_upper, color='red', alpha=0.15, label=f'True 90% PI ({noise_type})')
        plt.plot(x_train_np, true_upper, 'r--', linewidth=1, alpha=0.7)
        plt.plot(x_train_np, true_lower, 'r--', linewidth=1, alpha=0.7)
    
    # Plot prediction intervals
    plt.plot(x_train_np, train_mean, 'b-', linewidth=2, label='Predicted Mean (Train)')
    plt.fill_between(x_train_np, train_lower, train_upper, color='blue', alpha=0.2, label='Predicted 90% PI')
    
    # plt.plot(x_test_np, test_median, 'r-', linewidth=2, label='Predicted Median (Test)')
    # plt.fill_between(x_test_np, test_lower, test_upper, color='red', alpha=0.2, label='90% PI (Test)')

    # the true mean of the data is the sume of the noiseless y and the mean of the noise distribution
    plt.plot(x_train_np, mean_train + mean_noise*np.ones_like(x_train_np), 'g--', linewidth=2, label='True Mean (Train)')
    
    plt.title(f'CRPS-based MLP Sampler: Predictions with 90% Prediction Intervals\nNoise Type: {noise_type}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'experiments/1d_regression/figures/crps_samples_predictions_{noise_type}.png')
    plt.close()
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('CRPS Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig('experiments/1d_regression/figures/crps_samples_loss.png')
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    noise_type = 'exponential'

    # Generate toy data
    data = generate_toy_data_1d(n_samples=2000, noise_type=noise_type, mean_function='gaussian_rbf')
    x_train, y_train, x_test, y_test, x_train_np, y_train_np, x_test_np, y_test_np, noise_args, mean_train, mean_test = data
    
    print(f"Generated dataset with {x_train.shape[0]} training samples and {x_test.shape[0]} test samples")
    print(f"Noise type: {noise_type} with noise_args={noise_args}")
    
    # Create and train the model
    model = MLPSampler(input_size=1, hidden_size=64, latent_dim=16, n_layers=5, dropout_rate=0.1)
    train_losses, test_losses = train_model(model, x_train, y_train, x_test, y_test, 
                                           epochs=50, lr=0.001, batch_size=64, n_samples=100)
    
    # Plot and save results
    plot_results(model, x_train, y_train, x_test, y_test, mean_train, mean_test, noise_type, noise_args, n_samples=500)
