import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from sklearn.model_selection import train_test_split

# Add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our model
from models.mlp_crps_sampler import MLPSampler

# Import our data generation function
from experiments.data.generate import generate_toy_data_multidim

# Import our training functions
from experiments.training.train_functions import (
    train_model, evaluate_model, prepare_data_loaders, evaluate_metrics
)

from experiments.visualisation.plotting import plot_training_history, plot_prediction_samples

# Set random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Specify input and output dimensions
x_dim = 2
y_dim = 2
data_size = 5000

# Generate toy data
data_dict = generate_toy_data_multidim(n_samples=data_size, x_dim=x_dim, y_dim=y_dim, dependent_noise=True, mean_function='zero', target_correlation=1.0)

x_train_tensor = data_dict['x_train_tensor']
y_train_tensor = data_dict['y_train_tensor']
x_test_tensor = data_dict['x_test_tensor']
y_test_tensor = data_dict['y_test_tensor']
noise_args = data_dict['noise_args']  # Extract noise_args for visualization

# Split training data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

# Prepare data loaders
batch_size = 64
train_loader, val_loader = prepare_data_loaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

# Define model parameters
hidden_size = 8
latent_dim = 2
n_layers = 3
dropout_rate = 0.0
sample_layer_index = 1

# Create model
model = MLPSampler(
    input_size=x_dim,
    hidden_size=hidden_size,
    latent_dim=latent_dim,
    n_layers=n_layers,
    dropout_rate=dropout_rate,
    output_size=y_dim,
    sample_layer_index=sample_layer_index
)

# Training parameters
n_epochs = 100
patience = 5
train_n_samples = 10
n_samples = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'mlp_sampler.pt')

# Train the model
training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    patience=patience,
    n_samples=train_n_samples,
    device=device,
    save_path=save_path
)

# Plot training history
history_plot = plot_training_history(training_results['history'])
history_plot_path = os.path.join(os.path.dirname(__file__), 'figures', 'training_history.png')
os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
history_plot.savefig(history_plot_path)
plt.close(history_plot)

# Evaluate model on test set
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor),
    batch_size=batch_size
)
test_loss = evaluate_model(model, test_loader, device=device, n_samples=n_samples)
print(f"Test CRPS Loss: {test_loss:.4f}")

# Calculate additional metrics
metrics = evaluate_metrics(model, x_test_tensor, y_test_tensor, n_samples=n_samples, device=device)
print(f"Test MSE: {metrics['mse']:.4f}")
print(f"Test CRPS: {metrics['crps']:.4f}")
print("Calibration:")
for alpha, value in metrics['calibration'].items():
    print(f"  {alpha} interval: {value:.4f}")

# Plot prediction samples with true confidence intervals
samples_plot = plot_prediction_samples(
    x_test_tensor, y_test_tensor, model, 
    n_samples=n_samples, n_points=5, device=device,
    noise_args=noise_args  # Pass noise_args to visualize true confidence intervals
)
samples_plot_path = os.path.join(os.path.dirname(__file__), 'figures', 'prediction_samples.png')
os.makedirs(os.path.dirname(samples_plot_path), exist_ok=True)
samples_plot.savefig(samples_plot_path)
plt.close(samples_plot)

# Also create a plot with more test points to better visualize the distribution
samples_plot_more = plot_prediction_samples(
    x_test_tensor, y_test_tensor, model, 
    n_samples=n_samples, n_points=10, device=device,
    noise_args=noise_args
)
samples_plot_more_path = os.path.join(os.path.dirname(__file__), 'figures', 'prediction_samples_more.png')
samples_plot_more.savefig(samples_plot_more_path)
plt.close(samples_plot_more)








