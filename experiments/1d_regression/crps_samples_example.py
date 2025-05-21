import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchnaut.crps import EpsilonSampler

# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our model
from models.mlp_crps_sampler import MLPSampler

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
    else:
        # Default to student_t if not specified
        df = 3
        noise = np.random.standard_t(df=df, size=n_samples) * base_noise_scale
        noise_args = {'df': df,
                      'base_noise_scale': base_noise_scale}
    
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
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def plot_results(model, x_train, y_train, x_test, y_test, mean_train, mean_test, n_samples=500):
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
    train_median = np.median(train_samples_np, axis=1)
    train_lower = np.percentile(train_samples_np, 5, axis=1)
    train_upper = np.percentile(train_samples_np, 95, axis=1)
    
    test_median = np.median(test_samples_np, axis=1)
    test_lower = np.percentile(test_samples_np, 5, axis=1)
    test_upper = np.percentile(test_samples_np, 95, axis=1)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(x_train_np, y_train_np, color='blue', alpha=0.5, label='Training Data')
    plt.plot(x_train_np, mean_train, 'g--', linewidth=2, label='True Mean (Train)')
    
    # # Plot test data
    # plt.scatter(x_test_np, y_test_np, color='red', alpha=0.5, label='Test Data')
    # plt.plot(x_test_np, mean_test, 'y--', linewidth=2, label='True Mean (Test)')
    
    # Plot prediction intervals
    plt.plot(x_train_np, train_median, 'b-', linewidth=2, label='Predicted Median (Train)')
    plt.fill_between(x_train_np, train_lower, train_upper, color='blue', alpha=0.2, label='90% PI (Train)')
    
    # plt.plot(x_test_np, test_median, 'r-', linewidth=2, label='Predicted Median (Test)')
    # plt.fill_between(x_test_np, test_lower, test_upper, color='red', alpha=0.2, label='90% PI (Test)')
    
    plt.title('CRPS-based MLP Sampler: Predictions with 90% Prediction Intervals')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Save the figure
    plt.savefig('experiments/1d_regression/figures/crps_samples_predictions.png')
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
    
    # Generate toy data
    data = generate_toy_data(n_samples=500, noise_type='student_t', mean_function='gaussian_rbf')
    x_train, y_train, x_test, y_test, x_train_np, y_train_np, x_test_np, y_test_np, noise_args, mean_train, mean_test = data
    
    print(f"Generated dataset with {x_train.shape[0]} training samples and {x_test.shape[0]} test samples")
    print(f"Noise type: student_t with df={noise_args['df']:.2f}, scale={noise_args['base_noise_scale']:.4f}")
    
    # Create and train the model
    model = MLPSampler(input_size=1, hidden_size=64, latent_dim=16, n_layers=3, dropout_rate=0.1)
    train_losses, test_losses = train_model(model, x_train, y_train, x_test, y_test, 
                                           epochs=500, lr=0.001, batch_size=64, n_samples=100)
    
    # Plot and save results
    plot_results(model, x_train, y_train, x_test, y_test, mean_train, mean_test, n_samples=500)
