import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal, norm, multivariate_t, gamma as gamma_dist
import matplotlib.gridspec as gridspec
from experiments.data.kernels import evaluate_gp_mean
from experiments.data.generate import reconstruct_noise_quantiles
from itertools import combinations
import os

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

def plot_prediction_samples(x_test, y_test, model, n_samples=100, n_points=5, device='cpu', noise_args=None, epoch=None):
    """
    Plot prediction samples for randomly selected test points.
    Handles multi-dimensional outputs by visualizing empirical densities.
    Works with both MLPSampler and SimpleAffineNormal models.
    Now supports various noise types: gaussian, student_t, laplace_symmetric, laplace_asymmetric, gamma, lognormal
    
    Parameters:
    -----------
    x_test : torch.Tensor or None
        Test inputs (required for MLPSampler, can be None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    model : MLPSampler or SimpleAffineNormal
        Trained model
    n_samples : int
        Number of samples to generate
    n_points : int
        Number of test points to plot
    device : str
        Device to use for prediction ('cpu' or 'cuda')
    noise_args : dict, optional
        Dictionary containing noise parameters from generate_toy_data_multidim_extended:
        - noise_type: Type of noise distribution
        - cov_matrix: Covariance matrix of the noise
        - gp_functions: List of GP function information dictionaries
        - mean_function: Mean function type
        - sample_quantiles: Empirical quantiles from the noise distribution
        - Additional parameters specific to each noise type
    epoch : int, optional
        Training epoch number to include in plot titles
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Detect model type
    model_type = type(model).__name__
    is_affine_model = model_type == 'SimpleAffineNormal'
    
    # Select random test points
    indices = np.random.choice(len(y_test), min(n_points, len(y_test)), replace=False)
    y_subset = y_test[indices]
    
    # Generate samples based on model type
    if is_affine_model:
        # For SimpleAffineNormal, generate samples from the fixed distribution
        model.eval()
        model.to(device)
        with torch.no_grad():
            samples = model.forward(n_samples=n_samples, batch_size=n_points)  # [n_points, n_samples, output_dim]
    else:
        # For MLPSampler, generate samples per input point
        if x_test is None:
            raise ValueError("x_test is required for MLPSampler models")
        x_subset = x_test[indices]
        samples = predict_samples(model, x_subset, n_samples=n_samples, device=device)
    
    # Convert to numpy for plotting
    samples_np = samples.cpu().numpy()
    y_subset_np = y_subset.cpu().numpy()
    
    # Get input subset for labeling (if available)
    x_subset_np = None
    if not is_affine_model and x_test is not None:
        x_subset_np = x_test[indices].cpu().numpy()
    
    # Get output dimension
    output_dim = y_subset_np.shape[1]
    
    # Calculate true means if noise_args is provided
    true_means = None
    if noise_args is not None:
        if 'mean_function' in noise_args and noise_args['mean_function'] == 'zero':
            if is_affine_model:
                # For affine models, true mean is the same for all points
                true_means = np.zeros((n_points, output_dim))
            else:
                true_means = np.zeros((len(x_subset_np), output_dim))
        elif 'gp_functions' in noise_args and not is_affine_model:
            gp_functions = noise_args['gp_functions']
            y_dim = len(gp_functions)
            true_means = np.zeros((len(x_subset_np), y_dim))
            
            # Evaluate each GP function at the new points
            for j, gp_func in enumerate(gp_functions):
                true_means[:, j] = evaluate_gp_mean(x_subset_np, gp_func)
    
    # Get noise type and prepare distribution plotting
    noise_type = noise_args.get('noise_type', 'gaussian') if noise_args else 'gaussian'
    
    # Create figure based on output dimensionality
    if output_dim == 1:
        # Single dimension case - histogram for each test point
        fig, axes = plt.subplots(1, n_points, figsize=(4*n_points, 4))
        if n_points == 1:
            axes = [axes]
        
        for i in range(n_points):
            ax = axes[i]
            # Plot histogram of samples
            ax.hist(samples_np[i, :, 0], bins=20, alpha=0.7, density=True, label='Predicted Samples')
            
            # Plot mean prediction
            mean_pred = samples_np[i, :, 0].mean()
            ax.axvline(mean_pred, color='g', linestyle='-', label='Mean Prediction')
            
            # Plot true confidence intervals and distribution if noise_args is provided
            if noise_args is not None:
                # Get quantile information
                quantile_info = reconstruct_noise_quantiles(noise_args)
                
                if true_means is not None:
                    true_mean = true_means[i, 0]
                    ax.axvline(true_mean, color='blue', linestyle='-', label='True Mean')
                    
                    # Plot confidence intervals based on noise type
                    if noise_type == 'gaussian':
                        # Use theoretical Gaussian intervals
                        cov_matrix = noise_args.get('cov_matrix', np.array([[1.0]]))
                        std_dev = np.sqrt(cov_matrix[0, 0])
                        
                        for n_std, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                            lower = true_mean - n_std * std_dev
                            upper = true_mean + n_std * std_dev
                            ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                        
                        # Plot true distribution curve
                        x_range = np.linspace(true_mean - 4*std_dev, true_mean + 4*std_dev, 1000)
                        pdf = norm.pdf(x_range, true_mean, std_dev)
                        ax.plot(x_range, pdf, 'b-', alpha=0.7, label='True Distribution')
                        
                    elif noise_type == 'student_t':
                        # Use theoretical Student-t intervals
                        cov_matrix = noise_args.get('cov_matrix', np.array([[1.0]]))
                        df = noise_args.get('df', 3)
                        scale = np.sqrt(cov_matrix[0, 0])
                        
                        # Plot confidence intervals using t-distribution quantiles
                        from scipy.stats import t
                        for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                            t_val = t.ppf((1 + conf_level) / 2, df)
                            lower = true_mean - t_val * scale
                            upper = true_mean + t_val * scale
                            ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                        
                        # Plot true distribution curve
                        x_range = np.linspace(true_mean - 4*scale, true_mean + 4*scale, 1000)
                        pdf = t.pdf(x_range, df, loc=true_mean, scale=scale)
                        ax.plot(x_range, pdf, 'b-', alpha=0.7, label='True Distribution')
                        
                    else:
                        # For other noise types, use empirical quantiles
                        if 'sample_quantiles' in noise_args:
                            sample_quantiles = noise_args['sample_quantiles']
                            
                            # Plot empirical confidence intervals
                            for q_level, alpha, label in [('q25', 0.3, 'IQR'), ('q05', 0.2, '90% CI'), ('q01', 0.1, '98% CI')]:
                                if q_level in sample_quantiles:
                                    lower_key = q_level
                                    upper_key = f"q{100 - int(q_level[1:])}"
                                    if upper_key in sample_quantiles:
                                        lower = sample_quantiles[lower_key][0] + true_mean
                                        upper = sample_quantiles[upper_key][0] + true_mean
                                        ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                            
                            # Plot median line
                            if 'q50' in sample_quantiles:
                                median = sample_quantiles['q50'][0] + true_mean
                                ax.axvline(median, color='blue', linestyle='--', alpha=0.7, label='True Median')
                        
                        # Add text annotation about noise type
                        ax.text(0.02, 0.98, f'Noise: {noise_type}', transform=ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set title based on model type
            epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
            if is_affine_model:
                ax.set_title(f"Test Point {i+1}{epoch_str}")
            else:
                ax.set_title(f"Test Point {i+1}: X = {x_subset_np[i, 0]:.2f}{epoch_str}")
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
        # Add a common legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
        
    else:
        # Multi-dimensional case - scatter plots with marginal histograms for each test point
        # Generate all pairs of dimensions to plot
        if output_dim <= 4:
            # For output_dim <= 4, plot all pairs of dimensions
            dims_pairs = list(combinations(range(output_dim), 2))
        else:
            # For output_dim > 4, use only the first 4 dimensions and plot all their pairs
            dims_pairs = list(combinations(range(4), 2))
        
        n_pairs = len(dims_pairs)
        
        # Create figure with appropriate size
        fig = plt.figure(figsize=(5*n_pairs, 6*n_points))
        
        # Create a grid layout: n_points rows, n_pairs columns
        outer_grid = gridspec.GridSpec(n_points, n_pairs, figure=fig)
        
        for i in range(n_points):
            for j, (dim1, dim2) in enumerate(dims_pairs):
                # Create a nested grid for the main plot and marginal histograms
                inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, 
                                                            subplot_spec=outer_grid[i, j], 
                                                            width_ratios=[4, 1], 
                                                            height_ratios=[4, 1],
                                                            wspace=0.05, 
                                                            hspace=0.25)
                
                # Create the main scatter plot
                ax_scatter = plt.Subplot(fig, inner_grid[0, 0])
                fig.add_subplot(ax_scatter)
                
                # Create marginal histogram plots
                ax_histx = plt.Subplot(fig, inner_grid[1, 0])  # bottom histogram
                ax_histy = plt.Subplot(fig, inner_grid[0, 1])  # right histogram
                fig.add_subplot(ax_histx)
                fig.add_subplot(ax_histy)
                
                # Plot sample points on the main scatter plot
                if samples_np[i, :, dim1].shape[0] > 100:
                    sample_indices = np.random.choice(samples_np[i, :, dim1].shape[0], 100, replace=False)
                    ax_scatter.scatter(samples_np[i, sample_indices, dim1], samples_np[i, sample_indices, dim2], 
                              alpha=0.3, s=10, label='Subselected 100/{} Samples'.format(samples_np[i, :, dim1].shape[0]))
                else:
                    ax_scatter.scatter(samples_np[i, :, dim1], samples_np[i, :, dim2], 
                              alpha=0.3, s=10, label='Model Samples')
                
                # Plot mean prediction
                mean_pred_dim1 = samples_np[i, :, dim1].mean()
                mean_pred_dim2 = samples_np[i, :, dim2].mean()
                ax_scatter.scatter(mean_pred_dim1, mean_pred_dim2, 
                          color='g', marker='o', s=100, label='Mean Prediction')
                
                # Plot test point (true value)
                if not is_affine_model:
                    ax_scatter.scatter(y_subset_np[i, dim1], y_subset_np[i, dim2], 
                              color='red', s=100, marker='x', label='True Value')
                
                # Plot true confidence intervals if noise_args is provided
                if noise_args is not None and true_means is not None:
                    true_mean = true_means[i]
                    
                    # Plot true mean
                    ax_scatter.scatter(true_mean[dim1], true_mean[dim2], 
                              color='blue', marker='+', s=100, label='True Mean')
                    
                    # Handle different noise types for multivariate case
                    if noise_type == 'gaussian':
                        # Use theoretical Gaussian confidence ellipses
                        cov_matrix = noise_args.get('cov_matrix')
                        if cov_matrix is not None:
                            # Extract the relevant dimensions from the covariance matrix
                            if output_dim > 2:
                                sub_cov = np.array([[cov_matrix[dim1, dim1], cov_matrix[dim1, dim2]],
                                                    [cov_matrix[dim2, dim1], cov_matrix[dim2, dim2]]])
                                true_mean_2d = np.array([true_mean[dim1], true_mean[dim2]])
                            else:
                                sub_cov = cov_matrix
                                true_mean_2d = true_mean
                            
                            # Plot confidence ellipses
                            from matplotlib.patches import Ellipse
                            
                            def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
                                """Plot an ellipse representing the covariance matrix."""
                                eigvals, eigvecs = np.linalg.eigh(cov)
                                idx = np.argsort(eigvals)[::-1]
                                eigvals = eigvals[idx]
                                eigvecs = eigvecs[:, idx]
                                
                                width, height = 2 * nstd * np.sqrt(eigvals)
                                theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                                
                                ellip = Ellipse(xy=pos, width=width, height=height,
                                                angle=theta, **kwargs)
                                return ellip
                            
                            # Plot confidence ellipses for 1, 2, and 3 standard deviations
                            for nstd, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                                ellipse = plot_cov_ellipse(sub_cov, true_mean_2d, nstd=nstd,
                                                          alpha=alpha, edgecolor='blue', fc='blue',
                                                          label=label if i == 0 and j == 0 else None)
                                ax_scatter.add_patch(ellipse)
                            
                            # Try to add density contours for the true distribution
                            try:
                                x_min, x_max = ax_scatter.get_xlim()
                                y_min, y_max = ax_scatter.get_ylim()
                                X, Y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                
                                rv = multivariate_normal(true_mean_2d, sub_cov)
                                Z = rv.pdf(positions.T)
                                Z = Z.reshape(X.shape)
                                
                                ax_scatter.contour(X, Y, Z, 5, colors='blue', alpha=0.5, linestyles='--')
                            except:
                                pass
                    
                    elif noise_type == 'student_t':
                        # For Student-t, use scaled confidence ellipses
                        cov_matrix = noise_args.get('cov_matrix')
                        df = noise_args.get('df', 3)
                        
                        if cov_matrix is not None:
                            # Extract relevant dimensions
                            if output_dim > 2:
                                sub_cov = np.array([[cov_matrix[dim1, dim1], cov_matrix[dim1, dim2]],
                                                    [cov_matrix[dim2, dim1], cov_matrix[dim2, dim2]]])
                                true_mean_2d = np.array([true_mean[dim1], true_mean[dim2]])
                            else:
                                sub_cov = cov_matrix
                                true_mean_2d = true_mean
                            
                            # Scale covariance for Student-t
                            t_scale = (df + 2) / df if df > 2 else 2.0
                            scaled_cov = sub_cov * t_scale
                            
                            # Plot confidence ellipses
                            from matplotlib.patches import Ellipse
                            
                            def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
                                eigvals, eigvecs = np.linalg.eigh(cov)
                                idx = np.argsort(eigvals)[::-1]
                                eigvals = eigvals[idx]
                                eigvecs = eigvecs[:, idx]
                                
                                width, height = 2 * nstd * np.sqrt(eigvals)
                                theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                                
                                ellip = Ellipse(xy=pos, width=width, height=height,
                                                angle=theta, **kwargs)
                                return ellip
                            
                            # Plot confidence ellipses
                            for nstd, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                                ellipse = plot_cov_ellipse(scaled_cov, true_mean_2d, nstd=nstd,
                                                          alpha=alpha, edgecolor='blue', fc='blue',
                                                          label=label if i == 0 and j == 0 else None)
                                ax_scatter.add_patch(ellipse)
                    
                    else:
                        # For other noise types, use empirical quantiles to create approximate confidence regions
                        if 'sample_quantiles' in noise_args:
                            sample_quantiles = noise_args['sample_quantiles']
                            
                            # Use empirical quantiles to create approximate confidence regions
                            # This is a simplification - in practice, multivariate quantiles are more complex
                            for q_level, alpha, label in [('q25', 0.3, 'IQR Region'), ('q05', 0.2, '90% CI Region')]:
                                if q_level in sample_quantiles:
                                    # Create a simple rectangular region based on marginal quantiles
                                    lower_q = sample_quantiles[q_level]
                                    upper_key = f"q{100 - int(q_level[1:])}"
                                    if upper_key in sample_quantiles:
                                        upper_q = sample_quantiles[upper_key]
                                        
                                        # Create rectangle
                                        from matplotlib.patches import Rectangle
                                        rect = Rectangle((true_mean[dim1] + lower_q[dim1], true_mean[dim2] + lower_q[dim2]),
                                                       upper_q[dim1] - lower_q[dim1], upper_q[dim2] - lower_q[dim2],
                                                       alpha=alpha, edgecolor='blue', facecolor='blue',
                                                       label=label if i == 0 and j == 0 else None)
                                        ax_scatter.add_patch(rect)
                        
                        # Add text annotation about noise type
                        ax_scatter.text(0.02, 0.98, f'Noise: {noise_type}', transform=ax_scatter.transAxes, 
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
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
                        X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        Z = np.reshape(gaussian_kde(xy)(positions).T, X.shape)
                        ax_scatter.contour(X, Y, Z, 5, colors='k', alpha=0.3)
                    except:
                        pass
                except:
                    pass
                
                # Add marginal histograms
                x_min, x_max = ax_scatter.get_xlim()
                y_min, y_max = ax_scatter.get_ylim()
                
                # Plot marginal histograms
                ax_histx.hist(samples_np[i, :, dim1], bins=20, alpha=0.7, density=True, color='gray')
                ax_histy.hist(samples_np[i, :, dim2], bins=20, alpha=0.7, density=True, 
                             orientation='horizontal', color='gray')
                
                # Plot true marginal PDFs if noise_args is provided
                if noise_args is not None and true_means is not None:
                    true_mean = true_means[i]
                    
                    # Plot marginal distributions based on noise type
                    if noise_type == 'gaussian':
                        cov_matrix = noise_args.get('cov_matrix')
                        if cov_matrix is not None:
                            # For dimension 1 (x-axis)
                            std_dev_dim1 = np.sqrt(cov_matrix[dim1, dim1])
                            true_mean_dim1 = true_mean[dim1]
                            x_range = np.linspace(x_min, x_max, 1000)
                            pdf_dim1 = norm.pdf(x_range, true_mean_dim1, std_dev_dim1)
                            ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, label='True PDF')
                            ax_histx.axvline(true_mean_dim1, color='blue', linestyle='-')
                            
                            # For dimension 2 (y-axis)
                            std_dev_dim2 = np.sqrt(cov_matrix[dim2, dim2])
                            true_mean_dim2 = true_mean[dim2]
                            y_range = np.linspace(y_min, y_max, 1000)
                            pdf_dim2 = norm.pdf(y_range, true_mean_dim2, std_dev_dim2)
                            ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7)
                            ax_histy.axhline(true_mean_dim2, color='blue', linestyle='-')
                    
                    elif noise_type == 'student_t':
                        cov_matrix = noise_args.get('cov_matrix')
                        df = noise_args.get('df', 3)
                        
                        if cov_matrix is not None:
                            from scipy.stats import t
                            
                            # For dimension 1 (x-axis)
                            scale_dim1 = np.sqrt(cov_matrix[dim1, dim1])
                            true_mean_dim1 = true_mean[dim1]
                            x_range = np.linspace(x_min, x_max, 1000)
                            pdf_dim1 = t.pdf(x_range, df, loc=true_mean_dim1, scale=scale_dim1)
                            ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, label='True PDF')
                            ax_histx.axvline(true_mean_dim1, color='blue', linestyle='-')
                            
                            # For dimension 2 (y-axis)
                            scale_dim2 = np.sqrt(cov_matrix[dim2, dim2])
                            true_mean_dim2 = true_mean[dim2]
                            y_range = np.linspace(y_min, y_max, 1000)
                            pdf_dim2 = t.pdf(y_range, df, loc=true_mean_dim2, scale=scale_dim2)
                            ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7)
                            ax_histy.axhline(true_mean_dim2, color='blue', linestyle='-')
                    
                    # For other noise types, just plot the mean lines
                    else:
                        ax_histx.axvline(true_mean[dim1], color='blue', linestyle='-', alpha=0.7, label='True Mean')
                        ax_histy.axhline(true_mean[dim2], color='blue', linestyle='-', alpha=0.7)
                    
                    # Plot predicted means
                    ax_histx.axvline(mean_pred_dim1, color='g', linestyle='-')
                    ax_histy.axhline(mean_pred_dim2, color='g', linestyle='-')
                
                # Remove tick labels from marginal plots that overlap with main plot
                ax_histx.set_xticklabels([])
                ax_histy.set_yticklabels([])
                
                # Remove some spines
                ax_histx.spines['right'].set_visible(False)
                ax_histx.spines['top'].set_visible(False)
                ax_histy.spines['right'].set_visible(False)
                ax_histy.spines['top'].set_visible(False)
                
                # Set labels for the main plot based on model type
                epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
                if is_affine_model:
                    ax_scatter.set_title(f"Test Point {i+1}, Dims ({dim1+1}, {dim2+1}){epoch_str}")
                else:
                    ax_scatter.set_title(f"Test Point {i+1}, Dims ({dim1+1}, {dim2+1}): X = {', '.join([f'{x_subset_np[i, k]:.2f}' for k in range(x_subset_np.shape[1])])}{epoch_str}")
                ax_scatter.set_xlabel(f'Dimension {dim1+1}')
                ax_scatter.set_ylabel(f'Dimension {dim2+1}')
                
                # Match the limits
                ax_histx.set_xlim(ax_scatter.get_xlim())
                ax_histy.set_ylim(ax_scatter.get_ylim())
        
        # Add a common legend to the figure (only once, using the last subplot)
        if n_points > 0 and n_pairs > 0:
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

def plot_training_progression_landscape(x_test, y_test, model_checkpoints, epochs, 
                                      n_samples=1000, test_point_idx=0, device='cpu', 
                                      noise_args=None, save_path=None, figsize=(20, 8)):
    """
    Create a landscape plot showing model prediction evolution during training.
    All epochs are shown in one figure with a shared test point, dimensions, and legend.
    
    Parameters:
    -----------
    x_test : torch.Tensor or None
        Test inputs (required for MLPSampler and FGNEncoderSampler, can be None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    model_checkpoints : list of nn.Module
        List of model states at different epochs (loaded models)
    epochs : list of int
        List of epoch numbers corresponding to each model checkpoint
    n_samples : int
        Number of samples to generate for each model
    test_point_idx : int
        Index of the test point to visualize across all epochs
    device : str
        Device to use for prediction ('cpu' or 'cuda')
    noise_args : dict, optional
        Dictionary containing noise parameters for true distribution overlay
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created landscape figure
    """
    if len(model_checkpoints) != len(epochs):
        raise ValueError("Number of model checkpoints must match number of epochs")
    
    if len(model_checkpoints) == 0:
        raise ValueError("At least one model checkpoint is required")
    
    # Limit to maximum 4 epochs, evenly spaced if more than 4
    if len(epochs) > 4:
        # Select 4 evenly spaced indices
        indices = np.linspace(0, len(epochs) - 1, 4, dtype=int)
        selected_checkpoints = [model_checkpoints[i] for i in indices]
        selected_epochs = [epochs[i] for i in indices]
        print(f"Selected {len(selected_epochs)} evenly spaced epochs from {len(epochs)} total: {selected_epochs}")
    else:
        selected_checkpoints = model_checkpoints
        selected_epochs = epochs
    
    # Detect model type from first checkpoint
    model_type = type(selected_checkpoints[0]).__name__
    is_affine_model = model_type == 'SimpleAffineNormal'
    
    # Select the specific test point
    if test_point_idx >= len(y_test):
        test_point_idx = 0  # Fallback to first point
    
    y_point = y_test[test_point_idx:test_point_idx+1]  # Keep batch dimension
    
    # Get input point if needed
    x_point = None
    if not is_affine_model and x_test is not None:
        x_point = x_test[test_point_idx:test_point_idx+1]
    
    # Get output dimension
    output_dim = y_point.shape[1]
    
    # Generate samples for all epochs
    all_samples = []
    for model in selected_checkpoints:
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            if is_affine_model:
                samples = model.forward(n_samples=n_samples, batch_size=1)  # [1, n_samples, output_dim]
            else:
                samples = predict_samples(model, x_point, n_samples=n_samples, device=device)  # [1, n_samples, output_dim]
        
        all_samples.append(samples.cpu().numpy()[0])  # Remove batch dimension: [n_samples, output_dim]
    
    # Calculate true means if noise_args is provided
    true_means = None
    if noise_args is not None:
        if 'mean_function' in noise_args and noise_args['mean_function'] == 'zero':
            true_means = np.zeros(output_dim)
        elif 'gp_functions' in noise_args and not is_affine_model and x_test is not None:
            gp_functions = noise_args['gp_functions']
            y_dim = len(gp_functions)
            true_means = np.zeros(y_dim)
            
            # Evaluate each GP function at the test point
            x_point_np = x_point.cpu().numpy()
            for j, gp_func in enumerate(gp_functions):
                true_means[j] = evaluate_gp_mean(x_point_np, gp_func)[0]
    
    # Create figure layout
    n_epochs = len(selected_epochs)
    
    if output_dim == 1:
        # Single dimension: horizontal layout of histograms
        fig, axes = plt.subplots(1, n_epochs, figsize=figsize, sharey=True)
        if n_epochs == 1:
            axes = [axes]
        
        # Track all samples across epochs for consistent axis limits
        all_samples_flat = np.concatenate([samples[:, 0] for samples in all_samples])
        x_min, x_max = all_samples_flat.min(), all_samples_flat.max()
        x_range = x_max - x_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        
        # Plot for each epoch
        legend_handles = []
        legend_labels = []
        
        for i, (samples, epoch) in enumerate(zip(all_samples, selected_epochs)):
            ax = axes[i]
            
            # Plot histogram of samples
            n, bins, patches = ax.hist(samples[:, 0], bins=30, alpha=0.7, density=True, 
                                     color='lightblue', edgecolor='blue', linewidth=0.5)
            
            # Add kernel density estimation curve for predicted samples
            try:
                from scipy.stats import gaussian_kde
                if samples.shape[0] > 10:  # Only if we have enough samples
                    kde = gaussian_kde(samples[:, 0])
                    x_range = np.linspace(x_min, x_max, 200)
                    kde_values = kde(x_range)
                    
                    ax.plot(x_range, kde_values, color='black', linewidth=2, alpha=0.8, 
                           label='Predicted Density' if i == 0 else None)
            except ImportError:
                pass  # Skip if scipy is not available
            except Exception as e:
                pass  # Skip if KDE fails
            
            # Plot mean prediction
            mean_pred = samples[:, 0].mean()
            line1 = ax.axvline(mean_pred, color='green', linestyle='-', linewidth=2, label='Mean Prediction')
            
            # Plot true value (if not affine model)
            if not is_affine_model:
                line2 = ax.axvline(y_point.cpu().numpy()[0, 0], color='red', linestyle=':', linewidth=2, label='True Value')
            
            # Plot true distribution overlay if noise_args provided
            if noise_args is not None and 'cov_matrix' in noise_args and true_means is not None:
                cov_matrix = noise_args['cov_matrix']
                true_mean = true_means[0] if len(true_means.shape) > 0 else true_means
                std_dev = np.sqrt(cov_matrix[0, 0])
                
                # Plot true mean
                line3 = ax.axvline(true_mean, color='blue', linestyle='-', linewidth=2, label='True Mean')
                
                # Plot confidence intervals
                for n_std, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                    lower = true_mean - n_std * std_dev
                    upper = true_mean + n_std * std_dev
                    span = ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label)
                
                # Plot true distribution curve
                x_vals = np.linspace(x_min, x_max, 1000)
                pdf = norm.pdf(x_vals, true_mean, std_dev)
                line4 = ax.plot(x_vals, pdf, 'b-', alpha=0.8, linewidth=2, label='True Distribution')
            
            # Set title and labels
            ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Prediction Value')
            if i == 0:
                ax.set_ylabel('Density')
            
            # Set consistent limits
            ax.set_xlim(x_min, x_max)
            
            # Store legend info from first subplot
            if i == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        
        # Add overall title
        test_point_info = ""
        if not is_affine_model and x_test is not None:
            x_val = x_point.cpu().numpy()[0]
            if len(x_val) == 1:
                test_point_info = f" (X = {x_val[0]:.3f})"
            else:
                test_point_info = f" (X = [{', '.join([f'{v:.3f}' for v in x_val])}])"
        
        fig.suptitle(f'Model Prediction Evolution During Training', 
                     fontsize=16, fontweight='bold', y=0.95)
        
    else:
        # Multi-dimensional: use subplot grid for different dimension pairs with marginal histograms
        # For simplicity, focus on first two dimensions or all pairs if ≤ 4 dimensions
        if output_dim <= 4:
            dims_pairs = list(combinations(range(output_dim), 2))
        else:
            dims_pairs = [(0, 1), (0, 2), (1, 2)]  # First few pairs for higher dimensions
        
        if len(dims_pairs) == 0:  # Single pair for 2D
            dims_pairs = [(0, 1)]
        
        n_pairs = len(dims_pairs)
        
        # Create figure with nested grids for marginal histograms
        fig = plt.figure(figsize=figsize)
        
        # Create outer grid: epochs as columns, dimension pairs as rows
        outer_grid = gridspec.GridSpec(n_pairs, n_epochs, figure=fig)
        
        # Calculate global limits for consistent scaling
        all_samples_concat = np.concatenate(all_samples, axis=0)  # [n_epochs * n_samples, output_dim]
        dim_mins = all_samples_concat.min(axis=0)
        dim_maxs = all_samples_concat.max(axis=0)
        dim_ranges = dim_maxs - dim_mins
        dim_mins -= 0.1 * dim_ranges
        dim_maxs += 0.1 * dim_ranges
        
        # Store legend info
        legend_handles = []
        legend_labels = []
        
        for pair_idx, (dim1, dim2) in enumerate(dims_pairs):
            for epoch_idx, (samples, epoch) in enumerate(zip(all_samples, selected_epochs)):
                # Create a nested grid for the main plot and marginal histograms
                inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, 
                                                            subplot_spec=outer_grid[pair_idx, epoch_idx], 
                                                            width_ratios=[4, 1], 
                                                            height_ratios=[4, 1],
                                                            wspace=0.05, 
                                                            hspace=0.25)
                
                # Create the main scatter plot
                ax_scatter = plt.Subplot(fig, inner_grid[0, 0])
                fig.add_subplot(ax_scatter)
                
                # Create marginal histogram plots
                ax_histx = plt.Subplot(fig, inner_grid[1, 0])  # bottom histogram
                ax_histy = plt.Subplot(fig, inner_grid[0, 1])  # right histogram
                fig.add_subplot(ax_histx)
                fig.add_subplot(ax_histy)
                
                # Subsample for plotting if too many points
                if samples.shape[0] > 100:
                    sample_indices = np.random.choice(samples.shape[0], 100, replace=False)
                    plot_samples = samples[sample_indices]
                    alpha = 0.6
                    label_suffix = f" (100/{samples.shape[0]})"
                else:
                    plot_samples = samples
                    alpha = 0.4
                    label_suffix = ""
                
                # Plot sample points
                scatter1 = ax_scatter.scatter(plot_samples[:, dim1], plot_samples[:, dim2], 
                                    alpha=alpha, s=8, c='lightblue', edgecolors='blue', linewidth=0.5,
                                    label=f'Model Samples{label_suffix}')
                
                # Plot mean prediction
                mean_pred_dim1 = samples[:, dim1].mean()
                mean_pred_dim2 = samples[:, dim2].mean()
                scatter2 = ax_scatter.scatter(mean_pred_dim1, mean_pred_dim2, 
                                    color='green', marker='o', s=100, 
                                    edgecolors='darkgreen', linewidth=2,
                                    label='Mean Prediction', zorder=5)
                
                # Plot true value (if not affine model)
                if not is_affine_model:
                    true_val_dim1 = y_point.cpu().numpy()[0, dim1]
                    true_val_dim2 = y_point.cpu().numpy()[0, dim2]
                    scatter3 = ax_scatter.scatter(true_val_dim1, true_val_dim2, 
                                        color='red', marker='x', s=150, 
                                        linewidth=3, label='True Value', zorder=6)
                
                # Plot true distribution overlay if available
                if noise_args is not None and 'cov_matrix' in noise_args and true_means is not None:
                    cov_matrix = noise_args['cov_matrix']
                    
                    # Extract relevant 2D covariance and mean
                    if output_dim > 2:
                        sub_cov = np.array([[cov_matrix[dim1, dim1], cov_matrix[dim1, dim2]],
                                           [cov_matrix[dim2, dim1], cov_matrix[dim2, dim2]]])
                        true_mean_2d = np.array([true_means[dim1], true_means[dim2]])
                    else:
                        sub_cov = cov_matrix
                        true_mean_2d = true_means
                    
                    # Plot true mean
                    scatter4 = ax_scatter.scatter(true_mean_2d[0], true_mean_2d[1], 
                                        color='blue', marker='+', s=150, 
                                        linewidth=3, label='True Mean', zorder=6)
                    
                    # Plot confidence ellipses
                    from matplotlib.patches import Ellipse
                    
                    def create_ellipse(cov, pos, nstd=1, **kwargs):
                        """Create confidence ellipse."""
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        idx = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]
                        
                        width, height = 2 * nstd * np.sqrt(eigvals)
                        theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                        
                        return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
                    
                    # Add confidence ellipses
                    for nstd, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                        ellipse = create_ellipse(sub_cov, true_mean_2d, nstd=nstd,
                                               alpha=alpha, edgecolor='blue', facecolor='blue',
                                               label=label)
                        ax_scatter.add_patch(ellipse)
                
                # Add density contours for the predicted samples
                try:
                    from scipy.stats import gaussian_kde
                    xy = np.vstack([samples[:, dim1], samples[:, dim2]])
                    
                    # Only add contours if we have enough samples
                    if samples.shape[0] > 10:
                        z = gaussian_kde(xy)(xy)
                        
                        # Sort points by density for better visualization
                        idx = z.argsort()
                        x_sorted, y_sorted, z_sorted = samples[idx, dim1], samples[idx, dim2], z[idx]
                        
                        # Try to add contour lines
                        try:
                            xmin, xmax = samples[:, dim1].min(), samples[:, dim1].max()
                            ymin, ymax = samples[:, dim2].min(), samples[:, dim2].max()
                            
                            # Expand range slightly for contour grid
                            x_range = xmax - xmin
                            y_range = ymax - ymin
                            xmin -= 0.1 * x_range
                            xmax += 0.1 * x_range
                            ymin -= 0.1 * y_range
                            ymax += 0.1 * y_range
                            
                            # Create grid for contour
                            X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                            positions = np.vstack([X.ravel(), Y.ravel()])
                            Z = np.reshape(gaussian_kde(xy)(positions).T, X.shape)
                            
                            contour_lines = ax_scatter.contour(X, Y, Z, levels=3, colors='black', alpha=0.4, linewidths=1.0)
                            
                            # Add contour labels only for first epoch to avoid clutter
                            if epoch_idx == 0 and pair_idx == 0:
                                ax_scatter.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
                                
                        except Exception as e:
                            # Fallback: just add a few contour levels without grid
                            try:
                                # Simpler approach: just show density with alpha-scaled points
                                scatter_density = ax_scatter.scatter(x_sorted, y_sorted, c=z_sorted, 
                                                           s=2, alpha=0.1, cmap='viridis', zorder=1)
                            except:
                                pass  # Skip if even this fails
                except ImportError:
                    pass  # Skip if scipy is not available
                except Exception as e:
                    pass  # Skip density estimation if it fails
                
                # Set consistent limits
                ax_scatter.set_xlim(dim_mins[dim1], dim_maxs[dim1])
                ax_scatter.set_ylim(dim_mins[dim2], dim_maxs[dim2])
                
                # Add marginal histograms
                # Get the x and y limits from the scatter plot
                x_min, x_max = ax_scatter.get_xlim()
                y_min, y_max = ax_scatter.get_ylim()
                
                # Plot marginal histograms
                ax_histx.hist(samples[:, dim1], bins=20, alpha=0.7, density=True, color='lightblue', edgecolor='blue')
                ax_histy.hist(samples[:, dim2], bins=20, alpha=0.7, density=True, 
                             orientation='horizontal', color='lightblue', edgecolor='blue')
                
                # Plot true marginal PDFs if noise_args is provided
                if noise_args is not None and true_means is not None:
                    cov_matrix = noise_args['cov_matrix']
                    
                    # For dimension 1 (x-axis)
                    std_dev_dim1 = np.sqrt(cov_matrix[dim1, dim1])
                    true_mean_dim1 = true_means[dim1]
                    x_range = np.linspace(x_min, x_max, 1000)
                    pdf_dim1 = norm.pdf(x_range, true_mean_dim1, std_dev_dim1)
                    ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, linewidth=2)
                    ax_histx.axvline(true_mean_dim1, color='blue', linestyle='-', linewidth=2)
                    ax_histx.axvline(mean_pred_dim1, color='g', linestyle='-', linewidth=2)
                    
                    # For dimension 2 (y-axis)
                    std_dev_dim2 = np.sqrt(cov_matrix[dim2, dim2])
                    true_mean_dim2 = true_means[dim2]
                    y_range = np.linspace(y_min, y_max, 1000)
                    pdf_dim2 = norm.pdf(y_range, true_mean_dim2, std_dev_dim2)
                    ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7, linewidth=2)
                    ax_histy.axhline(true_mean_dim2, color='blue', linestyle='-', linewidth=2)
                    ax_histy.axhline(mean_pred_dim2, color='g', linestyle='-', linewidth=2)
                
                # Remove tick labels from marginal plots that overlap with main plot
                ax_histx.set_xticklabels([])
                ax_histy.set_yticklabels([])
                
                # Remove some spines
                ax_histx.spines['right'].set_visible(False)
                ax_histx.spines['top'].set_visible(False)
                ax_histy.spines['right'].set_visible(False)
                ax_histy.spines['top'].set_visible(False)
                
                # Set labels and title
                # if pair_idx == n_pairs - 1:  # Bottom row
                ax_scatter.set_xlabel(f'Dimension {dim1 + 1}')
                if epoch_idx == 0:  # Left column
                    ax_scatter.set_ylabel(f'Dimension {dim2 + 1}')
                
                # Title for top row
                if pair_idx == 0:
                    ax_scatter.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
                
                # Grid
                ax_scatter.grid(True, alpha=0.3)
                
                # Match the limits
                ax_histx.set_xlim(ax_scatter.get_xlim())
                ax_histy.set_ylim(ax_scatter.get_ylim())
                
                # Store legend info from first subplot
                if pair_idx == 0 and epoch_idx == 0:
                    legend_handles, legend_labels = ax_scatter.get_legend_handles_labels()
        
        # Add overall title
        test_point_info = ""
        if not is_affine_model and x_test is not None:
            x_val = x_point.cpu().numpy()[0]
            if len(x_val) == 1:
                test_point_info = f" (X = {x_val[0]:.3f})"
            else:
                test_point_info = f" (X = [{', '.join([f'{v:.3f}' for v in x_val])}])"
        
        dims_info = f"Dimensions: {', '.join([f'({p[0]+1},{p[1]+1})' for p in dims_pairs])}"
        fig.suptitle(f'Model Prediction Evolution During Training', 
                     fontsize=16, fontweight='bold', y=0.95)
    
    # Add shared legend
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=min(len(legend_labels), 6), fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training progression plot to: {save_path}")
    
    return fig

 