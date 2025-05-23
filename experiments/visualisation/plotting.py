import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal, norm
import matplotlib.gridspec as gridspec

def predict_samples(model, x, n_samples=100, device='cpu'):
    """
    Generate prediction samples from the model.
    
    Parameters:
    -----------
    model : MLPSampler
        The trained model
    x : torch.Tensor
        Input tensor of shape [batch_size, input_size]
    n_samples : int
        Number of samples to generate
    device : str
        Device to use for prediction ('cpu' or 'cuda')
        
    Returns:
    --------
    torch.Tensor
        Predicted samples of shape [batch_size, n_samples, output_size]
    """
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        samples = model(x, n_samples=n_samples)
    
    return samples

def plot_prediction_samples(x_test, y_test, model, n_samples=100, n_points=5, device='cpu', noise_args=None):
    """
    Plot prediction samples for randomly selected test points.
    Handles multi-dimensional outputs by visualizing empirical densities.
    
    Parameters:
    -----------
    x_test : torch.Tensor
        Test inputs
    y_test : torch.Tensor
        Test targets
    model : MLPSampler
        Trained model
    n_samples : int
        Number of samples to generate
    n_points : int
        Number of test points to plot
    device : str
        Device to use for prediction ('cpu' or 'cuda')
    noise_args : dict, optional
        Dictionary containing noise parameters from generate_toy_data_multidim
        - cov_matrix: covariance matrix of the noise
        - A: transformation matrix
        - dependent_noise: whether noise is dependent across dimensions
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Select random test points
    indices = np.random.choice(len(x_test), min(n_points, len(x_test)), replace=False)
    x_subset = x_test[indices]
    y_subset = y_test[indices]
    
    # Generate samples
    samples = predict_samples(model, x_subset, n_samples=n_samples, device=device)
    
    # Convert to numpy for plotting
    samples_np = samples.cpu().numpy()
    x_subset_np = x_subset.cpu().numpy()
    y_subset_np = y_subset.cpu().numpy()
    
    # Get output dimension
    output_dim = y_subset_np.shape[1]
    
    # Calculate true means if noise_args is provided
    true_means = None
    if noise_args is not None and 'A' in noise_args:
        A = noise_args['A']
        true_means = np.dot(x_subset_np, A)
    
    # For outputs with more than 2 dimensions, randomly select 2 dimensions to visualize
    dims_to_plot = list(range(output_dim))
    if output_dim > 2:
        dims_to_plot = np.random.choice(output_dim, 2, replace=False)
    
    # Create figure based on output dimensionality
    if output_dim == 1:
        # Single dimension case - histogram for each test point
        fig, axes = plt.subplots(1, n_points, figsize=(4*n_points, 4))
        if n_points == 1:
            axes = [axes]
        
        for i in range(n_points):
            ax = axes[i]
            # Plot histogram of samples
            ax.hist(samples_np[i, :, 0], bins=20, alpha=0.7, density=True)
            
            # # Plot true value
            # ax.axvline(y_subset_np[i, 0], color='r', linestyle='--', label='True Value')
            
            # Plot mean prediction
            mean_pred = samples_np[i, :, 0].mean()
            ax.axvline(mean_pred, color='g', linestyle='-', label='Mean Prediction')
            
            # Plot true confidence intervals if noise_args is provided
            if noise_args is not None and 'cov_matrix' in noise_args:
                cov_matrix = noise_args['cov_matrix']
                if true_means is not None:
                    true_mean = true_means[i, 0]
                    
                    # For 1D, we can directly use the standard deviation
                    std_dev = np.sqrt(cov_matrix[0, 0])
                    
                    # Plot true mean
                    ax.axvline(true_mean, color='blue', linestyle='-', label='True Mean')
                    
                    # Plot confidence intervals (68%, 95%, 99.7%)
                    for n_std, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                        lower = true_mean - n_std * std_dev
                        upper = true_mean + n_std * std_dev
                        ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                    
                    # Plot true distribution curve
                    x_range = np.linspace(true_mean - 4*std_dev, true_mean + 4*std_dev, 1000)
                    pdf = norm.pdf(x_range, true_mean, std_dev)
                    ax.plot(x_range, pdf, 'b-', alpha=0.7, label='True Distribution')
            
            ax.set_title(f"Test Point {i+1}: X = {x_subset_np[i, 0]:.2f}")
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Density')
            
        # Add a common legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
        
    else:
        # Multi-dimensional case - scatter plots with marginal histograms for each test point
        fig = plt.figure(figsize=(5*n_points, 6))
        
        # Create a grid layout with space for marginal histograms
        outer_grid = gridspec.GridSpec(1, n_points, figure=fig)
        
        for i in range(n_points):
            # Create a nested grid for the main plot and marginal histograms
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, 
                                                        subplot_spec=outer_grid[i], 
                                                        width_ratios=[4, 1], 
                                                        height_ratios=[4, 1],
                                                        wspace=0.05, 
                                                        hspace=0.1)
            
            # Create the main scatter plot
            ax_scatter = plt.Subplot(fig, inner_grid[0, 0])
            fig.add_subplot(ax_scatter)
            
            # Create marginal histogram plots
            ax_histx = plt.Subplot(fig, inner_grid[1, 0])  # bottom histogram
            ax_histy = plt.Subplot(fig, inner_grid[0, 1])  # right histogram
            fig.add_subplot(ax_histx)
            fig.add_subplot(ax_histy)
            
            # For 2D output or when visualizing 2 dimensions from higher-dim output
            dim1, dim2 = dims_to_plot[0], dims_to_plot[1] if len(dims_to_plot) > 1 else dims_to_plot[0]
            
            # Plot sample points on the main scatter plot
            if samples_np[i, :, dim1].shape[0] > 100:
                sample_indices = np.random.choice(samples_np[i, :, dim1].shape[0], 100, replace=False)
                ax_scatter.scatter(samples_np[i, sample_indices, dim1], samples_np[i, sample_indices, dim2], 
                          alpha=0.3, s=10, label='Subselected 100/{} Samples'.format(samples_np[i, :, dim1].shape[0]))
            else:
                ax_scatter.scatter(samples_np[i, :, dim1], samples_np[i, :, dim2], 
                          alpha=0.3, s=10, label='Samples')
            
            # Plot mean prediction
            mean_pred_dim1 = samples_np[i, :, dim1].mean()
            mean_pred_dim2 = samples_np[i, :, dim2].mean()
            ax_scatter.scatter(mean_pred_dim1, mean_pred_dim2, 
                      color='g', marker='o', s=100, label='Mean Prediction')
            
            # Plot true confidence intervals if noise_args is provided
            if noise_args is not None and 'cov_matrix' in noise_args and true_means is not None:
                cov_matrix = noise_args['cov_matrix']
                true_mean = true_means[i]
                
                # Extract the relevant dimensions from the covariance matrix
                if output_dim > 2:
                    # For higher dimensions, extract the 2D submatrix for the selected dimensions
                    sub_cov = np.array([[cov_matrix[dim1, dim1], cov_matrix[dim1, dim2]],
                                        [cov_matrix[dim2, dim1], cov_matrix[dim2, dim2]]])
                    true_mean_2d = np.array([true_mean[dim1], true_mean[dim2]])
                else:
                    # For 2D, use the full covariance matrix
                    sub_cov = cov_matrix
                    true_mean_2d = true_mean
                
                # Plot true mean
                ax_scatter.scatter(true_mean_2d[0], true_mean_2d[1], 
                          color='blue', marker='+', s=100, label='True Mean')
                
                # Plot confidence ellipses
                from matplotlib.patches import Ellipse
                
                def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
                    """
                    Plot an ellipse representing the covariance matrix cov centered at pos.
                    
                    Parameters:
                    -----------
                    cov : array_like, shape (2, 2)
                        Covariance matrix
                    pos : array_like, shape (2,)
                        Center position
                    nstd : float
                        Number of standard deviations for the ellipse
                    **kwargs : dict
                        Additional arguments for the Ellipse patch
                    """
                    # Find and sort eigenvalues to get axes lengths
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    
                    # Sort eigenvalues in decreasing order
                    idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]
                    
                    # Width and height are "full" widths, not radius
                    width, height = 2 * nstd * np.sqrt(eigvals)
                    
                    # Compute rotation angle
                    theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    
                    # Create the ellipse patch
                    ellip = Ellipse(xy=pos, width=width, height=height,
                                    angle=theta, **kwargs)
                    return ellip
                
                # Plot confidence ellipses for 1, 2, and 3 standard deviations
                for nstd, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                    ellipse = plot_cov_ellipse(sub_cov, true_mean_2d, nstd=nstd,
                                              alpha=alpha, edgecolor='blue', fc='blue',
                                              label=label if i == 0 else None)
                    ax_scatter.add_patch(ellipse)
                
                # Try to add density contours for the true distribution
                try:
                    # Create a grid for the contour plot
                    x_min, x_max = ax_scatter.get_xlim()
                    y_min, y_max = ax_scatter.get_ylim()
                    X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    
                    # Create the multivariate normal distribution
                    rv = multivariate_normal(true_mean_2d, sub_cov)
                    
                    # Evaluate the PDF on the grid
                    Z = rv.pdf(positions.T)
                    Z = Z.reshape(X.shape)
                    
                    # Plot contours
                    ax_scatter.contour(X, Y, Z, 5, colors='blue', alpha=0.5, linestyles='--')
                except:
                    pass  # Skip if contour plotting fails
            
            # Add density contours for the predicted samples
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([samples_np[i, :, dim1], samples_np[i, :, dim2]])
                z = gaussian_kde(xy)(xy)
                
                # Sort points by density for better visualization
                idx = z.argsort()
                x, y, z = samples_np[i, idx, dim1], samples_np[i, idx, dim2], z[idx]
                
                # Try to add contour lines if possible
                try:
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(gaussian_kde(xy)(positions).T, X.shape)
                    ax_scatter.contour(X, Y, Z, 5, colors='k', alpha=0.3)
                except:
                    pass  # Skip contours if they can't be calculated
            except:
                pass  # Skip density estimation if scipy is not available
            
            # Add marginal histograms
            # Get the x and y limits from the scatter plot
            x_min, x_max = ax_scatter.get_xlim()
            y_min, y_max = ax_scatter.get_ylim()
            
            # Plot marginal histograms
            ax_histx.hist(samples_np[i, :, dim1], bins=20, alpha=0.7, density=True, color='gray')
            ax_histy.hist(samples_np[i, :, dim2], bins=20, alpha=0.7, density=True, 
                         orientation='horizontal', color='gray')
            
            # Plot true marginal PDFs if noise_args is provided
            if noise_args is not None and 'cov_matrix' in noise_args and true_means is not None:
                cov_matrix = noise_args['cov_matrix']
                true_mean = true_means[i]
                
                # For dimension 1 (x-axis)
                std_dev_dim1 = np.sqrt(cov_matrix[dim1, dim1])
                true_mean_dim1 = true_mean[dim1]
                x_range = np.linspace(x_min, x_max, 1000)
                pdf_dim1 = norm.pdf(x_range, true_mean_dim1, std_dev_dim1)
                ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, label='True PDF')
                ax_histx.axvline(true_mean_dim1, color='blue', linestyle='-')
                ax_histx.axvline(mean_pred_dim1, color='g', linestyle='-')
                
                # For dimension 2 (y-axis)
                std_dev_dim2 = np.sqrt(cov_matrix[dim2, dim2])
                true_mean_dim2 = true_mean[dim2]
                y_range = np.linspace(y_min, y_max, 1000)
                pdf_dim2 = norm.pdf(y_range, true_mean_dim2, std_dev_dim2)
                ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7)
                ax_histy.axhline(true_mean_dim2, color='blue', linestyle='-')
                ax_histy.axhline(mean_pred_dim2, color='g', linestyle='-')
            
            # Remove tick labels from marginal plots that overlap with main plot
            ax_histx.set_xticklabels([])
            ax_histy.set_yticklabels([])
            
            # Remove some spines
            ax_histx.spines['right'].set_visible(False)
            ax_histx.spines['top'].set_visible(False)
            ax_histy.spines['right'].set_visible(False)
            ax_histy.spines['top'].set_visible(False)
            
            # Set labels for the main plot
            ax_scatter.set_title(f"Test Point {i+1}: X = {', '.join([f'{x_subset_np[i, j]:.2f}' for j in range(x_subset_np.shape[1])])}")
            ax_scatter.set_xlabel(f'Dimension {dim1+1}')
            ax_scatter.set_ylabel(f'Dimension {dim2+1}')
            
            # Match the limits
            ax_histx.set_xlim(ax_scatter.get_xlim())
            ax_histy.set_ylim(ax_scatter.get_ylim())
        
        # Add a common legend to the figure
        handles, labels = ax_scatter.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    
    return fig

def plot_training_history(history):
    """
    Plot training and validation loss history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_loss'], label='Training Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig 