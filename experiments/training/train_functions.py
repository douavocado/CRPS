import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from ..visualisation.plotting import predict_samples

def train_model(model, train_loader, val_loader, criterion=None, optimizer=None, 
                n_epochs=100, patience=10, n_samples=100, device='cpu', 
                verbose=True, save_path=None):
    """
    Train a MLPSampler model with early stopping based on validation loss.
    
    Parameters:
    -----------
    model : MLPSampler
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : function, optional
        Loss function. If None, model.crps_loss will be used
    optimizer : torch.optim, optional
        Optimizer. If None, Adam with lr=0.001 will be used
    n_epochs : int
        Maximum number of epochs to train
    patience : int
        Number of epochs to wait for improvement before early stopping
    n_samples : int
        Number of samples to generate for CRPS calculation
    device : str
        Device to use for training ('cpu' or 'cuda')
    verbose : bool
        Whether to print progress
    save_path : str, optional
        Path to save the best model
        
    Returns:
    --------
    dict
        Dictionary containing training history and best model
    """
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set up criterion if not provided
    if criterion is None:
        criterion = lambda x, y: model.crps_loss(x, y, n_samples=n_samples)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Use tqdm for progress bar if verbose
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]") if verbose else train_loader
        
        for x_batch, y_batch in train_iterator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = criterion(x_batch, y_batch)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
        
        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion, device, n_samples)
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            patience_counter = 0
            
            # Save best model if path provided
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model weights
    model.load_state_dict(best_model['model_state_dict'])
    
    return {
        'model': model,
        'history': history,
        'best_epoch': best_model['epoch'],
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, data_loader, criterion=None, device='cpu', n_samples=100):
    """
    Evaluate a model on a dataset.
    
    Parameters:
    -----------
    model : MLPSampler
        The model to evaluate
    data_loader : DataLoader
        DataLoader for evaluation data
    criterion : function, optional
        Loss function. If None, model.crps_loss will be used
    device : str
        Device to use for evaluation ('cpu' or 'cuda')
    n_samples : int
        Number of samples to generate for CRPS calculation
        
    Returns:
    --------
    float
        Average loss on the dataset
    """
    model.eval()
    losses = []
    
    # Set up criterion if not provided
    if criterion is None:
        criterion = lambda x, y: model.crps_loss(x, y, n_samples=n_samples)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(x_batch, y_batch)
            losses.append(loss.item())
    
    return np.mean(losses)

def prepare_data_loaders(x_train, y_train, x_val, y_val, batch_size=32):
    """
    Prepare DataLoader objects for training and validation.
    
    Parameters:
    -----------
    x_train : torch.Tensor
        Training inputs
    y_train : torch.Tensor
        Training targets
    x_val : torch.Tensor
        Validation inputs
    y_val : torch.Tensor
        Validation targets
    batch_size : int
        Batch size
        
    Returns:
    --------
    tuple
        (train_loader, val_loader)
    """
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def evaluate_metrics(model, x_test, y_test, n_samples=100, device='cpu'):
    """
    Evaluate various metrics on test data.
    
    Parameters:
    -----------
    model : MLPSampler
        Trained model
    x_test : torch.Tensor
        Test inputs
    y_test : torch.Tensor
        Test targets
    n_samples : int
        Number of samples to generate
    device : str
        Device to use for prediction ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary of metrics including:
        - mse: Mean squared error
        - crps: Continuous Ranked Probability Score
        - calibration: Coverage probabilities for prediction intervals
        - log_likelihood: Log likelihood estimated via density estimation
                         (using KDE for 1D, GMM for multidimensional outputs)
    """
    # Generate samples
    samples = predict_samples(model, x_test, n_samples=n_samples, device=device)
    
    # Convert to numpy
    samples_np = samples.cpu().numpy()  # shape: [batch_size, n_samples, output_size]
    y_test_np = y_test.cpu().numpy()    # shape: [batch_size, output_size]
    
    # Calculate mean predictions
    mean_preds = samples_np.mean(axis=1)  # shape: [batch_size, output_size]
    
    # Calculate MSE
    mse = mean_squared_error(y_test_np, mean_preds)
    
    # Calculate CRPS (already done in model evaluation)
    crps = evaluate_model(model, torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=len(x_test)), device=device, n_samples=n_samples)
    
    # Calculate calibration (percentage of true values within prediction intervals)
    calibration = {}
    for alpha in [0.5, 0.9, 0.95, 0.99]:
        lower = np.percentile(samples_np, (1-alpha)/2 * 100, axis=1)
        upper = np.percentile(samples_np, (1+alpha)/2 * 100, axis=1)
        in_interval = np.logical_and(y_test_np >= lower, y_test_np <= upper)
        calibration[f'{alpha:.2f}'] = np.mean(in_interval)
    
    # Calculate log likelihood using density estimation
    # For each test point, we estimate the density of the predicted samples
    # and evaluate the log likelihood of the true value under this density
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
