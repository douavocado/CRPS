import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
from scipy.stats import multivariate_normal, norm, multivariate_t, gamma as gamma_dist, gaussian_kde, chi2
import matplotlib.gridspec as gridspec
from experiments.data.kernels import evaluate_gp_mean
from experiments.data.generate import reconstruct_noise_quantiles
from itertools import combinations
import os
from scipy.stats import f, chi2, multivariate_normal, t, gaussian_kde

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
        - noise_samples: Full noise samples for empirical estimation (for non-parametric distributions)
        - Additional parameters specific to each noise type
    epoch : int, optional
        Training epoch number to include in plot titles
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract noise_samples from noise_args if available
    noise_samples = None
    if noise_args is not None:
        noise_samples = noise_args.get('noise_samples', None)
    
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
                        # For other noise types, use empirical estimates from noise samples
                        if noise_samples is not None:
                            # Adjust noise samples to the true mean
                            adjusted_noise = noise_samples[:, 0] + true_mean
                            
                            # Plot empirical confidence intervals
                            for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                                lower_p = (1 - conf_level) / 2 * 100
                                upper_p = (1 + conf_level) / 2 * 100
                                lower = np.percentile(adjusted_noise, lower_p)
                                upper = np.percentile(adjusted_noise, upper_p)
                                ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                            
                            # Plot empirical distribution using KDE
                            try:
                                if len(adjusted_noise) > 10:
                                    kde = gaussian_kde(adjusted_noise)
                                    x_range = np.linspace(adjusted_noise.min(), adjusted_noise.max(), 1000)
                                    pdf = kde(x_range)
                                    ax.plot(x_range, pdf, 'b-', alpha=0.7, label='True Distribution (Empirical)')
                            except ImportError:
                                pass
                            except Exception as e:
                                pass
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
                            
                            def plot_cov_ellipse(cov, pos, conf_level=0.95, **kwargs):
                                """Plot an ellipse representing the confidence region of a covariance matrix."""
                                # For 2D Gaussian, use chi-squared distribution with 2 degrees of freedom
                                chi2_val = chi2.ppf(conf_level, df=2)
                                
                                eigvals, eigvecs = np.linalg.eigh(cov)
                                idx = np.argsort(eigvals)[::-1]
                                eigvals = eigvals[idx]
                                eigvecs = eigvecs[:, idx]
                                
                                # Scale by square root of chi-squared quantile for proper confidence ellipse
                                width, height = 2 * np.sqrt(chi2_val * eigvals)
                                theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                                
                                ellip = Ellipse(xy=pos, width=width, height=height,
                                                angle=theta, **kwargs)
                                return ellip
                            
                            # Plot confidence ellipses for proper confidence intervals
                            for conf_level, alpha, label in [(0.68, 0.7, '68% CI'), (0.95, 0.5, '95% CI'), (0.99, 0.3, '99% CI')]:
                                ellipse = plot_cov_ellipse(sub_cov, true_mean_2d, conf_level=conf_level,
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
                        # For Student-t, use confidence ellipses based on Hotelling's T² distribution
                        # NOTE: noise_args['cov_matrix'] contains the scale matrix (the parameter Σ)
                        # This is what we need for Hotelling's T² distribution
                        scale_matrix = noise_args.get('cov_matrix')
                        df = noise_args.get('df', 3)
                        
                        if scale_matrix is not None:
                            # Extract relevant dimensions
                            if output_dim > 2:
                                sub_cov = np.array([[scale_matrix[dim1, dim1], scale_matrix[dim1, dim2]],
                                                    [scale_matrix[dim2, dim1], scale_matrix[dim2, dim2]]])
                                true_mean_2d = np.array([true_mean[dim1], true_mean[dim2]])
                            else:
                                sub_cov = scale_matrix
                                true_mean_2d = true_mean
                            
                            # sub_cov is now the correct scale matrix for Hotelling's T² distribution
                            # No additional scaling needed
                            
                            # Plot confidence ellipses using Hotelling's T² distribution
                            
                            def plot_student_t_ellipse(cov, pos, conf_level=0.95, df=3, **kwargs):
                                """Plot an ellipse representing the confidence region of a multivariate Student-t distribution."""
                                # For 2D multivariate Student-t, use Hotelling's T² distribution
                                # Critical value: c = (p * (df - p + 1))/(df - p) * F(α; p, df - p + 1)
                                p = 2  # 2D case
                                if df > p:
                                    # F-distribution quantile
                                    f_val = f.ppf(conf_level, p, df - p + 1)
                                    # Hotelling's T² critical value
                                    t_squared_val = (p * (df - p + 1)) / (df - p) * f_val
                                else:
                                    # For very low df, use a large critical value
                                    t_squared_val = 20.0
                                
                                eigvals, eigvecs = np.linalg.eigh(cov)
                                idx = np.argsort(eigvals)[::-1]
                                eigvals = eigvals[idx]
                                eigvecs = eigvecs[:, idx]
                                
                                # Scale by square root of Hotelling's T² critical value
                                width, height = 2 * np.sqrt(t_squared_val * eigvals)
                                theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                                
                                ellip = Ellipse(xy=pos, width=width, height=height,
                                                angle=theta, **kwargs)
                                return ellip
                            
                            # Plot confidence ellipses for Student-t distribution
                            for conf_level, alpha, label in [(0.68, 0.7, '68% CI'), (0.95, 0.5, '95% CI'), (0.99, 0.3, '99% CI')]:
                                ellipse = plot_student_t_ellipse(sub_cov, true_mean_2d, conf_level=conf_level, df=df,
                                                               alpha=alpha, edgecolor='blue', fc='blue',
                                                               label=label if i == 0 and j == 0 else None)
                                ax_scatter.add_patch(ellipse)
                    
                    else:
                        # For other noise types, use empirical estimates from noise samples
                        if noise_samples is not None:
                            # Adjust noise samples to the true mean
                            adjusted_noise = noise_samples.copy()
                            adjusted_noise[:, dim1] += true_mean[dim1]
                            adjusted_noise[:, dim2] += true_mean[dim2]
                            
                            # Extract 2D samples for the relevant dimensions
                            samples_2d = adjusted_noise[:, [dim1, dim2]]
                            
                            # Only proceed if we have enough samples for reliable KDE
                            if samples_2d.shape[0] > 20:
                                # Estimate 2D PDF using KDE
                                kde = gaussian_kde(samples_2d.T)
                                
                                # Create grid for contour computation
                                x_min, x_max = samples_2d[:, 0].min(), samples_2d[:, 0].max()
                                y_min, y_max = samples_2d[:, 1].min(), samples_2d[:, 1].max()
                                
                                # Expand range slightly for contour grid
                                x_range = x_max - x_min
                                y_range = y_max - y_min
                                x_min -= 0.2 * x_range
                                x_max += 0.2 * x_range
                                y_min -= 0.2 * y_range
                                y_max += 0.2 * y_range
                                
                                # Create evaluation grid
                                X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                Z = np.reshape(kde(positions), X.shape)
                                
                                # Calculate density values at sample points for quantile estimation
                                sample_densities = kde(samples_2d.T)
                                
                                # Find density thresholds corresponding to confidence levels and plot contours
                                for conf_level, alpha, label in [(0.68, 0.7, '68% CI'), (0.95, 0.5, '95% CI'), (0.99, 0.3, '99% CI')]:
                                    # Find density threshold such that conf_level of samples are above it
                                    density_threshold = np.percentile(sample_densities, (1 - conf_level) * 100)
                                    
                                    # Plot filled contour at this density level
                                    contour_filled = ax_scatter.contourf(X, Y, Z, levels=[density_threshold, Z.max()], 
                                                                        colors=['blue'], alpha=alpha,)
                                    
                                    # Plot contour lines for better visibility
                                    contour_lines = ax_scatter.contour(X, Y, Z, levels=[density_threshold], 
                                                                        colors=['blue'], alpha=min(1.0, alpha + 0.3), 
                                                                        linewidths=1.5)
                            else:
                                print(f"Warning: Not enough samples ({samples_2d.shape[0]}) for reliable KDE, skipping confidence regions")
                        else:
                            print("No noise samples available, omitting confidence regions of true distribution")
                        
                        # Add text annotation about noise type
                        ax_scatter.text(0.02, 0.98, f'Noise: {noise_type}', transform=ax_scatter.transAxes, 
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
                # Add density contours for the predicted samples using quantile-based contours
                xy = np.vstack([samples_np[i, :, dim1], samples_np[i, :, dim2]])
                
                # Only add contours if we have enough samples
                if samples_np.shape[1] > 20:
                    kde = gaussian_kde(xy)
                    z = kde(xy)
                    
                    # Sort points by density for better visualization
                    idx = z.argsort()
                    x_sorted, y_sorted, z_sorted = samples_np[i, idx, dim1], samples_np[i, idx, dim2], z[idx]
                    
                    # Try to add contour lines based on quantiles
                    xmin, xmax = x_sorted.min(), x_sorted.max()
                    ymin, ymax = y_sorted.min(), y_sorted.max()
                    
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
                    Z = np.reshape(kde(positions).T, X.shape)
                    
                    # Calculate density quantiles for contour levels
                    for conf_level, alpha, color in [(0.68, 0.6, 'red'), (0.95, 0.4, 'red'), (0.99, 0.2, 'red')]:
                        # Find density threshold corresponding to confidence level
                        density_threshold = np.percentile(z, (1 - conf_level) * 100)
                        
                        # Plot contour at this density level
                        contour_lines = ax_scatter.contour(X, Y, Z, levels=[density_threshold], 
                                                            colors=[color], alpha=alpha, linewidths=1.0)
                        
                        # Add contour labels only for first plot to avoid clutter
                        if i == 0 and j == 0 and conf_level == 0.95:
                            ax_scatter.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
                            
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
                    
                    else:
                        # For other noise types, use empirical estimates from noise samples
                        if noise_samples is not None:
                            # Adjust noise samples to the true mean
                            adjusted_noise_dim1 = noise_samples[:, dim1] + true_mean[dim1]
                            adjusted_noise_dim2 = noise_samples[:, dim2] + true_mean[dim2]
                            
                            # Plot empirical PDFs using KDE
                            try:
                                
                                # For dimension 1 (x-axis)
                                if len(adjusted_noise_dim1) > 10:
                                    kde_dim1 = gaussian_kde(adjusted_noise_dim1)
                                    x_range = np.linspace(x_min, x_max, 1000)
                                    pdf_dim1 = kde_dim1(x_range)
                                    ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, label='True PDF (Empirical)')
                                
                                # For dimension 2 (y-axis)
                                if len(adjusted_noise_dim2) > 10:
                                    kde_dim2 = gaussian_kde(adjusted_noise_dim2)
                                    y_range = np.linspace(y_min, y_max, 1000)
                                    pdf_dim2 = kde_dim2(y_range)
                                    ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7)
                                    
                            except ImportError:
                                pass
                            except Exception as e:
                                pass
                        
                        # Plot mean lines
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
                    ax_scatter.set_title(f"Test Point {i+1}: Dims ({dim1},{dim2}){epoch_str}")
                else:
                    ax_scatter.set_title(f"Test Point {i+1}: X = {x_subset_np[i, 0]:.2f}, Dims ({dim1},{dim2}){epoch_str}")
                ax_scatter.set_xlabel(f'Dimension {dim1}')
                ax_scatter.set_ylabel(f'Dimension {dim2}')
                ax_scatter.grid(True, alpha=0.3)
                
                # Match the limits
                ax_histx.set_xlim(ax_scatter.get_xlim())
                ax_histy.set_ylim(ax_scatter.get_ylim())
                
                # Add legend only to the first subplot to avoid clutter
                if i == 0 and j == 0:
                    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
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
    Create a training progression landscape plot showing model prediction evolution.
    Visualizes how predictions change across training epochs for a specific test point.
    Now supports various noise types: gaussian, student_t, laplace_symmetric, 
    laplace_asymmetric, gamma, lognormal
    
    Parameters:
    -----------
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    model_checkpoints : list
        List of model checkpoints from different epochs
    epochs : list
        List of epoch numbers corresponding to checkpoints
    n_samples : int
        Number of samples to generate for predictions
    test_point_idx : int
        Index of test point to visualize
    device : str
        Device to use for computation
    noise_args : dict, optional
        Noise arguments for plotting true distribution overlays.
        Should contain 'noise_type', 'cov_matrix', and other distribution parameters.
        For non-parametric distributions, can contain 'noise_samples' for empirical estimation.
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    from itertools import combinations
    import matplotlib.gridspec as gridspec
    from scipy.stats import norm, multivariate_normal, chi2
    
    # Extract noise_samples from noise_args if available
    noise_samples = None
    if noise_args is not None:
        noise_samples = noise_args.get('noise_samples', None)
    
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
    
    # Get noise type and samples for empirical estimation
    noise_type = noise_args.get('noise_type', 'gaussian') if noise_args else 'gaussian'
    
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
        
        # Extend range to include true distribution if available
        if noise_samples is not None:
            true_min, true_max = noise_samples[:, 0].min(), noise_samples[:, 0].max()
            x_min = min(x_min, true_min)
            x_max = max(x_max, true_max)
        
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
                if samples.shape[0] > 10:  # Only if we have enough samples
                    kde = gaussian_kde(samples[:, 0])
                    x_range_plot = np.linspace(x_min, x_max, 200)
                    kde_values = kde(x_range_plot)
                    
                    ax.plot(x_range_plot, kde_values, color='black', linewidth=2, alpha=0.8, 
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
            
            # Plot true distribution overlay based on noise type
            if noise_args is not None and true_means is not None:
                true_mean = true_means[0] if len(true_means.shape) > 0 else true_means
                
                # Plot true mean
                line3 = ax.axvline(true_mean, color='blue', linestyle='-', linewidth=2, label='True Mean')
                
                if noise_type == 'gaussian' and 'cov_matrix' in noise_args:
                    # Use exact Gaussian distribution
                    cov_matrix = noise_args['cov_matrix']
                    std_dev = np.sqrt(cov_matrix[0, 0])
                    
                    # Plot confidence intervals
                    for n_std, alpha, label in [(1, 0.3, '68% CI'), (2, 0.2, '95% CI'), (3, 0.1, '99.7% CI')]:
                        lower = true_mean - n_std * std_dev
                        upper = true_mean + n_std * std_dev
                        span = ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label)
                    
                    # Plot true distribution curve
                    x_vals = np.linspace(x_min, x_max, 1000)
                    pdf = norm.pdf(x_vals, true_mean, std_dev)
                    line4 = ax.plot(x_vals, pdf, 'b-', alpha=0.8, linewidth=2, label='True Distribution')
                    
                elif noise_type == 'student_t' and 'cov_matrix' in noise_args and 'df' in noise_args:
                    # Use exact Student-t distribution
                    from scipy.stats import t
                    cov_matrix = noise_args['cov_matrix']
                    df = noise_args['df']
                    scale = np.sqrt(cov_matrix[0, 0])
                    
                    # Plot confidence intervals using t-distribution quantiles
                    for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                        t_val = t.ppf((1 + conf_level) / 2, df)
                        lower = true_mean - t_val * scale
                        upper = true_mean + t_val * scale
                        ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                    
                    # Plot true distribution curve
                    x_vals = np.linspace(x_min, x_max, 1000)
                    pdf = t.pdf(x_vals, df, loc=true_mean, scale=scale)
                    line4 = ax.plot(x_vals, pdf, 'b-', alpha=0.8, linewidth=2, label='True Distribution')
                    
                else:
                    # Use empirical estimates from noise samples
                    if noise_samples is not None:
                        # Adjust noise samples to the true mean
                        adjusted_noise = noise_samples[:, 0] + true_mean
                        
                        # Plot empirical confidence intervals
                        for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                            lower_p = (1 - conf_level) / 2 * 100
                            upper_p = (1 + conf_level) / 2 * 100
                            lower = np.percentile(adjusted_noise, lower_p)
                            upper = np.percentile(adjusted_noise, upper_p)
                            ax.axvspan(lower, upper, alpha=alpha, color='blue', label=label if i == 0 else None)
                        
                        # Plot empirical distribution using KDE
                        try:
                            if len(adjusted_noise) > 10:
                                kde = gaussian_kde(adjusted_noise)
                                x_vals = np.linspace(x_min, x_max, 1000)
                                pdf = kde(x_vals)
                                line4 = ax.plot(x_vals, pdf, 'b-', alpha=0.8, linewidth=2, label='True Distribution (Empirical)')
                        except ImportError:
                            pass
                        except Exception as e:
                            pass
                    
                    # Add text annotation about noise type
                    ax.text(0.02, 0.98, f'Noise: {noise_type}', transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
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
        
        fig.suptitle(f'Model Prediction Evolution During Training{test_point_info}', 
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
        
        # Extend range to include true distribution if available
        # if noise_samples is not None:
        #     true_mins = noise_samples.min(axis=0)
        #     true_maxs = noise_samples.max(axis=0)
        #     dim_mins = np.minimum(dim_mins, true_mins)
        #     dim_maxs = np.maximum(dim_maxs, true_maxs)
        
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
                                    color='green', marker='o', s=50, 
                                    edgecolors='darkgreen', linewidth=2,
                                    label='Mean Prediction', zorder=5)
                
                # Plot true value (if not affine model)
                # if not is_affine_model:
                #     true_val_dim1 = y_point.cpu().numpy()[0, dim1]
                #     true_val_dim2 = y_point.cpu().numpy()[0, dim2]
                #     scatter3 = ax_scatter.scatter(true_val_dim1, true_val_dim2, 
                #                         color='red', marker='x', s=150, 
                #                         linewidth=3, label='True Value', zorder=6)
                
                # Plot true distribution overlay based on noise type
                if noise_args is not None and true_means is not None:
                    # For the training progression plot, we only have one test point (test_point_idx)
                    # so true_means is a 1D array with shape (output_dim,)
                    if len(true_means.shape) == 1:
                        true_mean = true_means  # true_means is already the mean for the single test point
                    else:
                        true_mean = true_means[0]  # Use first point if somehow we have multiple
                    
                    # Extract relevant 2D mean
                    if output_dim > 2:
                        true_mean_2d = np.array([true_mean[dim1], true_mean[dim2]])
                    else:
                        true_mean_2d = true_mean
                    
                    # Plot true mean
                    scatter4 = ax_scatter.scatter(true_mean_2d[0], true_mean_2d[1], 
                                        color='blue', marker='+', s=150, 
                                        linewidth=3, label='True Mean', zorder=6)
                    
                    if noise_type == 'gaussian' and 'cov_matrix' in noise_args:
                        # Use exact Gaussian confidence ellipses
                        cov_matrix = noise_args['cov_matrix']
                        
                        # Extract relevant 2D covariance
                        if output_dim > 2:
                            sub_cov = np.array([[cov_matrix[dim1, dim1], cov_matrix[dim1, dim2]],
                                               [cov_matrix[dim2, dim1], cov_matrix[dim2, dim2]]])
                        else:
                            sub_cov = cov_matrix
                        
                        # Plot confidence ellipses
                        
                        
                        def create_ellipse(cov, pos, conf_level=0.95, **kwargs):
                            """Create confidence ellipse."""
                            # For 2D Gaussian, use chi-squared distribution with 2 degrees of freedom
                            chi2_val = chi2.ppf(conf_level, df=2)
                            
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            idx = np.argsort(eigvals)[::-1]
                            eigvals = eigvals[idx]
                            eigvecs = eigvecs[:, idx]
                            
                            # Scale by square root of chi-squared quantile for proper confidence ellipse
                            width, height = 2 * np.sqrt(chi2_val * eigvals)
                            theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                            
                            return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
                        
                        # Add confidence ellipses
                        for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                            ellipse = create_ellipse(sub_cov, true_mean_2d, conf_level=conf_level,
                                                   alpha=alpha, edgecolor='blue', fc='blue',
                                                   label=label)
                            ax_scatter.add_patch(ellipse)
                    
                    elif noise_type == 'student_t' and 'cov_matrix' in noise_args and 'df' in noise_args:
                        # Use exact Student-t confidence ellipses based on Hotelling's T² distribution
                        # NOTE: noise_args['moments']['cov'] contains the true covariance matrix = (df/(df-2)) * scale_matrix
                        # For confidence ellipses, we need the scale matrix, not the covariance matrix
                        true_cov_matrix = noise_args['moments']['cov']
                        df = noise_args['df']
                        
                        # Convert from covariance matrix back to scale matrix
                        # true_cov = (df / (df - 2)) * scale_matrix
                        # Therefore: scale_matrix = true_cov * (df - 2) / df
                        if df > 2:
                            scale_matrix = true_cov_matrix * (df - 2) / df
                        else:
                            # For df <= 2, covariance is undefined, use the raw cov_matrix as an approximation
                            scale_matrix = noise_args.get('cov_matrix', true_cov_matrix)
                        
                        # Extract relevant 2D scale matrix
                        if output_dim > 2:
                            sub_cov = np.array([[scale_matrix[dim1, dim1], scale_matrix[dim1, dim2]],
                                               [scale_matrix[dim2, dim1], scale_matrix[dim2, dim2]]])
                        else:
                            sub_cov = scale_matrix
                        
                        # Plot confidence ellipses using Hotelling's T² distribution
                        
                        
                        def create_student_t_ellipse(cov, pos, conf_level=0.95, df=3, **kwargs):
                            """Plot an ellipse representing the confidence region of a multivariate Student-t distribution."""
                            # For 2D multivariate Student-t, use Hotelling's T² distribution
                            # Critical value: c = (p * (df - p + 1))/(df - p) * F(α; p, df - p + 1)
                            p = 2  # 2D case
                            if df > p:
                                # F-distribution quantile
                                f_val = f.ppf(conf_level, p, df - p + 1)
                                # Hotelling's T² critical value
                                t_squared_val = (p * (df - p + 1)) / (df - p) * f_val
                            else:
                                # For very low df, use a large critical value
                                t_squared_val = 20.0
                            
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            idx = np.argsort(eigvals)[::-1]
                            eigvals = eigvals[idx]
                            eigvecs = eigvecs[:, idx]
                            
                            # Scale by square root of Hotelling's T² critical value
                            width, height = 2 * np.sqrt(t_squared_val * eigvals)
                            theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                            
                            return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
                        
                        # Add confidence ellipses for Student-t distribution
                        for conf_level, alpha, label in [(0.68, 0.3, '68% CI'), (0.95, 0.2, '95% CI'), (0.997, 0.1, '99.7% CI')]:
                            ellipse = create_student_t_ellipse(sub_cov, true_mean_2d, conf_level=conf_level, df=df,
                                                             alpha=alpha, edgecolor='blue', fc='blue',
                                                             label=label)
                            ax_scatter.add_patch(ellipse)
                    
                    else:
                        # Use empirical estimates from noise samples with KDE for proper 2D confidence regions
                        # Adjust noise samples to the true mean
                        adjusted_noise = noise_samples.copy()
                        adjusted_noise[:, dim1] += true_mean_2d[0]
                        adjusted_noise[:, dim2] += true_mean_2d[1]
                        
                        if noise_samples is not None:                            
                            # Extract 2D samples for the relevant dimensions
                            samples_2d = adjusted_noise[:, [dim1, dim2]]
                            
                            # Only proceed if we have enough samples for reliable KDE
                            if samples_2d.shape[0] > 20:
                                # Estimate 2D PDF using KDE
                                kde = gaussian_kde(samples_2d.T)
                                
                                # Create grid for contour computation
                                x_min, x_max = samples_2d[:, 0].min(), samples_2d[:, 0].max()
                                y_min, y_max = samples_2d[:, 1].min(), samples_2d[:, 1].max()
                                
                                # Expand range slightly for contour grid
                                x_range = x_max - x_min
                                y_range = y_max - y_min
                                x_min -= 0.2 * x_range
                                x_max += 0.2 * x_range
                                y_min -= 0.2 * y_range
                                y_max += 0.2 * y_range
                                
                                # Create evaluation grid
                                X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                Z = np.reshape(kde(positions), X.shape)
                                
                                # Calculate density values at sample points for quantile estimation
                                sample_densities = kde(samples_2d.T)
                                
                                # Find density thresholds corresponding to confidence levels
                                for conf_level, alpha, label in [(0.68, 0.7, '68% CI'), (0.95, 0.5, '95% CI'), (0.99, 0.3, '99% CI')]:
                                    # Find density threshold such that conf_level of samples are above it
                                    density_threshold = np.percentile(sample_densities, (1 - conf_level) * 100)
                                    
                                    # Plot contour at this density level
                                    contour = ax_scatter.contour(X, Y, Z, levels=[density_threshold], 
                                                                colors=['blue'], alpha=alpha, linewidths=1.0)
                            
                            else:
                                # Skip confidence regions if not enough samples for reliable KDE
                                print(f"Warning: Not enough samples ({samples_2d.shape[0]}) for reliable KDE, skipping ground truth confidence regions")
                                    
                        else:
                            print(f"Warning: No noise samples available, skipping ground truth confidence regions")
                    # Add text annotation about noise type
                    ax_scatter.text(0.02, 0.98, f'Noise: {noise_type}', transform=ax_scatter.transAxes, 
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
                # Add density contours for the predicted samples using quantile-based contours
                try:
                    xy = np.vstack([samples[:, dim1], samples[:, dim2]])
                    
                    # Only add contours if we have enough samples
                    if samples.shape[0] > 20:
                        kde = gaussian_kde(xy)
                        z = kde(xy)
                        
                        # Sort points by density for better visualization
                        idx = z.argsort()
                        x_sorted, y_sorted, z_sorted = samples[idx, dim1], samples[idx, dim2], z[idx]
                        
                        # Try to add contour lines based on quantiles
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
                            Z = np.reshape(kde(positions).T, X.shape)
                            
                            # Calculate density quantiles for contour levels
                            for conf_level, alpha, color in [(0.68, 0.6, 'red'), (0.95, 0.4, 'red'), (0.99, 0.2, 'red')]:
                                # Find density threshold corresponding to confidence level
                                density_threshold = np.percentile(z, (1 - conf_level) * 100)
                                
                                # Plot contour at this density level
                                contour_lines = ax_scatter.contour(X, Y, Z, levels=[density_threshold], 
                                                                  colors=[color], alpha=alpha, linewidths=1.0)
                               
                                # Add contour labels only for first epoch to avoid clutter
                                if epoch_idx == 0 and pair_idx == 0 and conf_level == 0.95:
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
                
                # Plot true marginal PDFs based on noise type
                if noise_args is not None and true_means is not None:
                    # For the training progression plot, we only have one test point (test_point_idx)
                    # so true_means is a 1D array with shape (output_dim,)
                    if len(true_means.shape) == 1:
                        true_mean = true_means  # true_means is already the mean for the single test point
                    else:
                        true_mean = true_means[0]  # Use first point if somehow we have multiple
                    
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
                    
                    else:
                        # For other noise types, use empirical estimates from noise samples
                        if noise_samples is not None:
                            # Adjust noise samples to the true mean
                            adjusted_noise_dim1 = noise_samples[:, dim1] + true_mean[dim1]
                            adjusted_noise_dim2 = noise_samples[:, dim2] + true_mean[dim2]
                            
                            # Plot empirical PDFs using KDE
                            try:
                                
                                # For dimension 1 (x-axis)
                                if len(adjusted_noise_dim1) > 10:
                                    kde_dim1 = gaussian_kde(adjusted_noise_dim1)
                                    x_range = np.linspace(x_min, x_max, 1000)
                                    pdf_dim1 = kde_dim1(x_range)
                                    ax_histx.plot(x_range, pdf_dim1, 'b-', alpha=0.7, label='True PDF (Empirical)')
                                
                                # For dimension 2 (y-axis)
                                if len(adjusted_noise_dim2) > 10:
                                    kde_dim2 = gaussian_kde(adjusted_noise_dim2)
                                    y_range = np.linspace(y_min, y_max, 1000)
                                    pdf_dim2 = kde_dim2(y_range)
                                    ax_histy.plot(pdf_dim2, y_range, 'b-', alpha=0.7)
                                    
                            except ImportError:
                                pass
                            except Exception as e:
                                pass
                        
                        # Plot mean lines
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
                
                # Set labels and title
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
        fig.suptitle(f'Model Prediction Evolution During Training{test_point_info}', 
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

def create_intermediate_sample_plots(model, x_test, y_test, noise_args, epoch, 
                                   save_dir, n_samples=1000, n_points=1, device='cpu'):
    """
    Create and save prediction sample plots for intermediate training states.
    Now supports various noise types: gaussian, student_t, laplace_symmetric, 
    laplace_asymmetric, gamma, lognormal
    
    Parameters:
    -----------
    model : nn.Module
        The model to evaluate
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    noise_args : dict
        Noise arguments for plotting (supports extended noise types)
    epoch : int
        Current training epoch
    save_dir : str
        Directory to save plots
    n_samples : int
        Number of samples to generate
    n_points : int
        Number of test points to plot
    device : str
        Device to use
        
    Returns:
    --------
    str
        Path to saved plot
    """
    # Create the plot
    fig = plot_prediction_samples(
        x_test, y_test, model, 
        n_samples=n_samples, 
        n_points=n_points, 
        device=device,
        noise_args=noise_args,
        epoch=epoch
    )
    
    # Include noise type in filename for better organization
    noise_type = noise_args.get('noise_type', 'gaussian') if noise_args else 'gaussian'
    plot_path = os.path.join(save_dir, f'prediction_samples_epoch_{epoch:03d}_{noise_type}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path

def create_training_progression_plot(model, x_test, y_test, noise_args, 
                                   base_save_dir, sample_epochs, 
                                   model_checkpoints_dir, model_type,
                                   n_samples=1000, test_point_idx=0, 
                                   device='cpu', figsize=None):
    """
    Create a training progression landscape plot using saved intermediate models.
    This function loads model checkpoints from intermediate epochs and creates
    a consolidated plot showing prediction evolution.
    Now properly handles non-Gaussian noise types using empirical estimates and exact PDFs.
    
    Parameters:
    -----------
    model : nn.Module
        The final trained model (used as template for loading checkpoints)
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    noise_args : dict
        Noise arguments for plotting. Should include 'noise_samples' for empirical estimation 
        when exact PDFs are not available. Supports all noise types with proper handling.
    base_save_dir : str
        Base directory where plots will be saved
    sample_epochs : list
        List of epochs where samples were recorded
    model_checkpoints_dir : str
        Directory containing saved model checkpoints
    model_type : str
        Type of model ('MLPSampler', 'FGNEncoderSampler' or 'SimpleAffineNormal')
    n_samples : int
        Number of samples to generate
    test_point_idx : int
        Index of test point to visualize
    device : str
        Device to use
    figsize : tuple, optional
        Figure size. If None, will be calculated dynamically based on layout
        
    Returns:
    --------
    str or None
        Path to saved plot if successful, None if failed
    """
    try:
        import copy
        
        if len(sample_epochs) == 0:
            print("No sample epochs recorded, cannot create progression plot")
            return None
        
        # Load model checkpoints
        model_checkpoints = []
        available_epochs = []
        
        for epoch in sample_epochs:
            checkpoint_path = os.path.join(model_checkpoints_dir, f'model_epoch_{epoch:03d}.pt')
            
            if os.path.exists(checkpoint_path):
                try:
                    # Create a copy of the model
                    checkpoint_model = copy.deepcopy(model)
                    
                    # Load the checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
                    checkpoint_model.to(device)
                    
                    model_checkpoints.append(checkpoint_model)
                    available_epochs.append(epoch)
                    
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint for epoch {epoch}: {e}")
            else:
                print(f"Warning: Checkpoint not found for epoch {epoch}: {checkpoint_path}")
        
        if len(model_checkpoints) == 0:
            print("No valid model checkpoints found, using final model only")
            # Fallback to using just the final model
            model_checkpoints.append(model)
            available_epochs.append(sample_epochs[-1] if sample_epochs else 1)
        
        print(f"Creating training progression plot with {len(available_epochs)} epoch(s): {available_epochs}")
        
        # Calculate dynamic figsize if not provided
        if figsize is None:
            from itertools import combinations
            
            # Get output dimension from y_test
            output_dim = y_test.shape[1]
            n_epochs = len(available_epochs)
            
            # Limit to maximum 4 epochs for plotting (same logic as in plot_training_progression_landscape)
            n_epochs_displayed = min(n_epochs, 4)
            
            if output_dim == 1:
                # Single dimension: horizontal layout of histograms
                figsize = (5 * n_epochs_displayed, 6)
            else:
                # Multi-dimensional: calculate number of dimension pairs
                if output_dim <= 4:
                    n_pairs = len(list(combinations(range(output_dim), 2)))
                else:
                    n_pairs = 3  # Fixed to 3 pairs for higher dimensions: (0,1), (0,2), (1,2)
                
                # Each subplot needs space for scatter plot + marginal histograms
                figsize = (5 * n_epochs_displayed, 6 * n_pairs)
        
        # Create the landscape plot
        noise_type = noise_args.get('noise_type', 'gaussian') if noise_args else 'gaussian'
        save_path = os.path.join(base_save_dir, f'training_progression_landscape_{noise_type}.png')
        
        fig = plot_training_progression_landscape(
            x_test=x_test,
            y_test=y_test,
            model_checkpoints=model_checkpoints,
            epochs=available_epochs,
            n_samples=n_samples,
            test_point_idx=test_point_idx,
            device=device,
            noise_args=noise_args,
            save_path=save_path,
            figsize=figsize
        )
        
        plt.close(fig)  # Clean up memory
        
        return save_path
        
    except Exception as e:
        print(f"Warning: Failed to create training progression plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_training_progression_moments_plot(model, x_test, y_test, noise_args, 
                                            base_save_dir, sample_epochs, 
                                            model_checkpoints_dir, model_type,
                                            moments_to_track=None, n_samples=1000, 
                                            test_point_idx=0, device='cpu', figsize=None,
                                            noise_samples=None):
    """
    Create a training progression plot tracking empirical moments of predicted outputs 
    vs empirical moments of the underlying data distribution.
    
    Parameters:
    -----------
    model : nn.Module
        The final trained model (used as template for loading checkpoints)
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    noise_args : dict
        Noise arguments containing information about the underlying data distribution
    base_save_dir : str
        Base directory where plots will be saved
    sample_epochs : list
        List of epochs where samples were recorded
    model_checkpoints_dir : str
        Directory containing saved model checkpoints
    model_type : str
        Type of model ('MLPSampler', 'FGNEncoderSampler' or 'SimpleAffineNormal')
    moments_to_track : list, optional
        List of moments to track (default: [1, 2, 3, 4])
    n_samples : int
        Number of samples to generate for moment estimation
    test_point_idx : int
        Index of test point to use for input-dependent models
    device : str
        Device to use
    figsize : tuple, optional
        Figure size. If None, will be calculated dynamically
    noise_samples : numpy.ndarray, optional
        Noise samples from the underlying data distribution (required for true moments calculation)
        
    Returns:
    --------
    str or None
        Path to saved plot if successful, None if failed
    """
    try:
        import copy
        from scipy import stats
        
        if moments_to_track is None:
            moments_to_track = [1, 2, 3, 4]
        
        if len(sample_epochs) == 0:
            print("No sample epochs recorded, cannot create moments progression plot")
            return None
        
        # Check if we can calculate true moments (only when mean_function='zero' and noise_samples provided)
        can_calculate_true_moments = (noise_args is not None and 
                                     noise_args.get('mean_function') == 'zero' and
                                     noise_samples is not None)
        
        if not can_calculate_true_moments:
            if noise_args is None or noise_args.get('mean_function') != 'zero':
                print("Warning: Cannot calculate true moments - only available when mean_function='zero'")
            elif noise_samples is None:
                print("Warning: Cannot calculate true moments - noise_samples not provided")
        
        # Load model checkpoints
        model_checkpoints = []
        available_epochs = []
        
        for epoch in sample_epochs:
            checkpoint_path = os.path.join(model_checkpoints_dir, f'model_epoch_{epoch:03d}.pt')
            
            if os.path.exists(checkpoint_path):
                try:
                    # Create a copy of the model
                    checkpoint_model = copy.deepcopy(model)
                    
                    # Load the checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
                    checkpoint_model.to(device)
                    
                    model_checkpoints.append(checkpoint_model)
                    available_epochs.append(epoch)
                    
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint for epoch {epoch}: {e}")
            else:
                print(f"Warning: Checkpoint not found for epoch {epoch}: {checkpoint_path}")
        
        if len(model_checkpoints) == 0:
            print("No valid model checkpoints found, using final model only")
            model_checkpoints.append(model)
            available_epochs.append(sample_epochs[-1] if sample_epochs else 1)
        
        print(f"Creating moments progression plot with {len(available_epochs)} epoch(s): {available_epochs}")
        
        # Determine model type and setup
        is_affine_model = model_type == 'SimpleAffineNormal'
        
        # Get output dimension and select dimensions to plot
        output_dim = y_test.shape[1]
        if output_dim > 4:
            # Select random subset of 4 dimensions
            np.random.seed(42)  # For reproducibility
            selected_dims = np.random.choice(output_dim, 4, replace=False)
            print(f"Output dimension ({output_dim}) > 4, selected dimensions: {selected_dims}")
        else:
            selected_dims = list(range(output_dim))
        
        n_dims_to_plot = len(selected_dims)
        
        # Prepare input for sampling
        if is_affine_model:
            x_input = None
        else:
            if x_test is None:
                raise ValueError("x_test is required for non-affine models")
            # Use specific test point or first point
            x_input = x_test[test_point_idx:test_point_idx+1]
        
        # Generate samples for all epochs and calculate moments
        predicted_moments = {dim: {moment: [] for moment in moments_to_track} for dim in selected_dims}
        
        for model_checkpoint in model_checkpoints:
            model_checkpoint.eval()
            model_checkpoint.to(device)
            
            with torch.no_grad():
                if is_affine_model:
                    samples = model_checkpoint.forward(n_samples=n_samples, batch_size=1)  # [1, n_samples, output_dim]
                else:
                    samples = predict_samples(model_checkpoint, x_input, n_samples=n_samples, device=device)  # [1, n_samples, output_dim]
            
            # Convert to numpy and remove batch dimension
            samples_np = samples.cpu().numpy()[0]  # [n_samples, output_dim]
            
            # Calculate moments for each selected dimension
            for dim in selected_dims:
                dim_samples = samples_np[:, dim]
                
                for moment in moments_to_track:
                    if moment == 1:
                        # First moment: mean
                        moment_value = np.mean(dim_samples)
                    elif moment == 2:
                        # Second central moment: variance
                        moment_value = np.var(dim_samples, ddof=1)
                    elif moment == 3:
                        # Third central moment: skewness
                        moment_value = stats.skew(dim_samples)
                    elif moment == 4:
                        # Fourth central moment: kurtosis
                        moment_value = stats.kurtosis(dim_samples)
                    else:
                        # General central moment
                        moment_value = stats.moment(dim_samples, moment=moment)
                    
                    predicted_moments[dim][moment].append(moment_value)
        
        # Calculate true moments if possible
        true_moments = None
        if can_calculate_true_moments:
            # Use noise samples to calculate true moments
            true_moments = {dim: {} for dim in selected_dims}
            
            for dim in selected_dims:
                dim_noise = noise_samples[:, dim]
                
                for moment in moments_to_track:
                    if moment == 1:
                        true_moments[dim][moment] = np.mean(dim_noise)
                    elif moment == 2:
                        true_moments[dim][moment] = np.var(dim_noise, ddof=1)
                    elif moment == 3:
                        true_moments[dim][moment] = stats.skew(dim_noise)
                    elif moment == 4:
                        true_moments[dim][moment] = stats.kurtosis(dim_noise)
                    else:
                        true_moments[dim][moment] = stats.moment(dim_noise, moment=moment)
        
        # Create plots
        n_moments = len(moments_to_track)
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (6 * n_moments, 4 * n_dims_to_plot)
        
        fig, axes = plt.subplots(n_dims_to_plot, n_moments, figsize=figsize, squeeze=False)
        
        # Moment names for labeling
        moment_names = {
            1: 'Mean (1st moment)',
            2: 'Variance (2nd central moment)',
            3: 'Skewness (3rd central moment)',
            4: 'Kurtosis (4th central moment)'
        }
        
        # Colors for consistency
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot moments for each dimension
        for i, dim in enumerate(selected_dims):
            for j, moment in enumerate(moments_to_track):
                ax = axes[i, j]
                
                # Plot predicted moments evolution
                predicted_values = predicted_moments[dim][moment]
                ax.plot(available_epochs, predicted_values, 
                       marker='o', linewidth=2, markersize=6, 
                       color=colors[i % len(colors)], 
                       label=f'Predicted (Dim {dim})')
                
                # Plot true moment if available
                if true_moments is not None:
                    true_value = true_moments[dim][moment]
                    ax.axhline(y=true_value, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7,
                              label=f'True (Dim {dim})')
                
                # Formatting
                ax.set_xlabel('Epoch')
                ax.set_ylabel(moment_names.get(moment, f'{moment}th moment'))
                ax.set_title(f'Dimension {dim}: {moment_names.get(moment, f"{moment}th moment")}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add text box with final values
                final_pred = predicted_values[-1]
                text_str = f'Final: {final_pred:.4f}'
                if true_moments is not None:
                    true_val = true_moments[dim][moment]
                    error = abs(final_pred - true_val)
                    text_str += f'\nTrue: {true_val:.4f}\nError: {error:.4f}'
                
                ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add overall title
        noise_type = noise_args.get('noise_type', 'unknown') if noise_args else 'unknown'
        test_point_info = ""
        if not is_affine_model and x_test is not None:
            x_val = x_input.cpu().numpy()[0]
            if len(x_val) == 1:
                test_point_info = f" (X = {x_val[0]:.3f})"
            else:
                test_point_info = f" (X = [{', '.join([f'{v:.3f}' for v in x_val])}])"
        
        fig.suptitle(f'Empirical Moments Evolution During Training\n'
                     f'Noise Type: {noise_type}{test_point_info}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        noise_type_str = noise_type.replace('_', '-')
        save_path = os.path.join(base_save_dir, f'training_progression_moments_{noise_type_str}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved moments progression plot to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Warning: Failed to create moments progression plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_training_progression_moment_tensors_plot(model, x_test, y_test, noise_args, 
                                                   base_save_dir, sample_epochs, 
                                                   model_checkpoints_dir, model_type,
                                                   n_samples=1000, test_point_idx=0, 
                                                   device='cpu', figsize=None, moments_to_track=None):
    """
    Create a training progression plot tracking Frobenius norm of differences between
    empirical and ground truth multivariate central moment tensors.
    
    Parameters:
    -----------
    model : nn.Module
        The final trained model (used as template for loading checkpoints)
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    noise_args : dict
        Noise arguments containing ground truth moments in noise_args['moments']
        with keys: 'mean', 'cov', 'third_moment', 'fourth_moment'
    base_save_dir : str
        Base directory where plots will be saved
    sample_epochs : list
        List of epochs where samples were recorded
    model_checkpoints_dir : str
        Directory containing saved model checkpoints
    model_type : str
        Type of model ('MLPSampler', 'FGNEncoderSampler' or 'SimpleAffineNormal')
    n_samples : int
        Number of samples to generate for moment estimation
    test_point_idx : int
        Index of test point to use for input-dependent models
    device : str
        Device to use
    figsize : tuple, optional
        Figure size. If None, will be calculated dynamically
    moments_to_track : list, optional
        List of moments to track (default: [1, 2, 3, 4])
    Returns:
    --------
    str or None
        Path to saved plot if successful, None if failed
    """
    try:
        import copy
        from scipy import stats
        
        if moments_to_track is None:
            moments_to_track = [1, 2, 3, 4]
        
        moment_names = {
            1: 'mean',
            2: 'cov',
            3: 'third_moment',
            4: 'fourth_moment'
        }
        
        if len(sample_epochs) == 0:
            print("No sample epochs recorded, cannot create moment tensors progression plot")
            return None
        
        # Check if ground truth moments are available
        if noise_args is None or 'moments' not in noise_args:
            print("Warning: Cannot create moment tensors plot - ground truth moments not available in noise_args['moments']")
            return None
        
        true_moments = noise_args['moments']
        available_moment_keys = []
        
        # Check which moments are available
        for key in [moment_names[moment] for moment in moments_to_track]:
            if key in true_moments and true_moments[key] is not None:
                available_moment_keys.append(key)
        
        if len(available_moment_keys) == 0:
            print("Warning: No valid ground truth moments found in noise_args['moments']")
            return None
        
        print(f"Tracking Frobenius norms for moments: {available_moment_keys}")
        
        # Load model checkpoints
        model_checkpoints = []
        available_epochs = []
        
        for epoch in sample_epochs:
            checkpoint_path = os.path.join(model_checkpoints_dir, f'model_epoch_{epoch:03d}.pt')
            
            if os.path.exists(checkpoint_path):
                try:
                    # Create a copy of the model
                    checkpoint_model = copy.deepcopy(model)
                    
                    # Load the checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
                    checkpoint_model.to(device)
                    
                    model_checkpoints.append(checkpoint_model)
                    available_epochs.append(epoch)
                    
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint for epoch {epoch}: {e}")
            else:
                print(f"Warning: Checkpoint not found for epoch {epoch}: {checkpoint_path}")
        
        if len(model_checkpoints) == 0:
            print("No valid model checkpoints found, using final model only")
            model_checkpoints.append(model)
            available_epochs.append(sample_epochs[-1] if sample_epochs else 1)
        
        print(f"Creating moment tensors progression plot with {len(available_epochs)} epoch(s): {available_epochs}")
        
        # Determine model type and setup
        is_affine_model = model_type == 'SimpleAffineNormal'
        
        # Get output dimension
        output_dim = y_test.shape[1]
        
        # Prepare input for sampling
        if is_affine_model:
            x_input = None
        else:
            if x_test is None:
                raise ValueError("x_test is required for non-affine models")
            # Use specific test point or first point
            x_input = x_test[test_point_idx:test_point_idx+1]
        
        # Function to compute empirical central moments from samples
        def compute_empirical_moments(samples_np):
            """
            Compute empirical central moments from samples.
            
            Args:
                samples_np: numpy array of shape [n_samples, output_dim]
            
            Returns:
                dict with empirical moment tensors
            """
            empirical_moments = {}
            
            # First central moment: mean
            empirical_mean = np.mean(samples_np, axis=0)
            empirical_moments['mean'] = empirical_mean
            
            # Center the samples
            centered_samples = samples_np - empirical_mean[np.newaxis, :]
            
            # Second central moment: covariance matrix
            empirical_cov = np.cov(centered_samples.T)
            if empirical_cov.ndim == 0:  # Handle 1D case
                empirical_cov = np.array([[empirical_cov]])
            empirical_moments['cov'] = empirical_cov
            
            # Third central moment: E[(X-μ)⊗(X-μ)⊗(X-μ)]
            # Shape: [output_dim, output_dim, output_dim]
            third_moment = np.zeros((output_dim, output_dim, output_dim))
            for i in range(output_dim):
                for j in range(output_dim):
                    for k in range(output_dim):
                        third_moment[i, j, k] = np.mean(
                            centered_samples[:, i] * centered_samples[:, j] * centered_samples[:, k]
                        )
            empirical_moments['third_moment'] = third_moment
            
            # Fourth central moment: E[(X-μ)⊗(X-μ)⊗(X-μ)⊗(X-μ)]
            # Shape: [output_dim, output_dim, output_dim, output_dim]
            fourth_moment = np.zeros((output_dim, output_dim, output_dim, output_dim))
            for i in range(output_dim):
                for j in range(output_dim):
                    for k in range(output_dim):
                        for l in range(output_dim):
                            fourth_moment[i, j, k, l] = np.mean(
                                centered_samples[:, i] * centered_samples[:, j] * 
                                centered_samples[:, k] * centered_samples[:, l]
                            )
            empirical_moments['fourth_moment'] = fourth_moment
            
            return empirical_moments
        
        # Generate samples for all epochs and calculate Frobenius norms
        frobenius_norms = {key: [] for key in available_moment_keys}
        
        for model_checkpoint in model_checkpoints:
            model_checkpoint.eval()
            model_checkpoint.to(device)
            
            with torch.no_grad():
                if is_affine_model:
                    samples = model_checkpoint.forward(n_samples=n_samples, batch_size=1)  # [1, n_samples, output_dim]
                else:
                    samples = predict_samples(model_checkpoint, x_input, n_samples=n_samples, device=device)  # [1, n_samples, output_dim]
            
            # Convert to numpy and remove batch dimension
            samples_np = samples.cpu().numpy()[0]  # [n_samples, output_dim]
            
            # Compute empirical moments
            empirical_moments = compute_empirical_moments(samples_np)
            
            # Calculate Frobenius norms of differences
            for key in available_moment_keys:
                true_moment = true_moments[key]
                empirical_moment = empirical_moments[key]
                
                # Compute Frobenius norm of the difference
                diff = empirical_moment - true_moment
                # flatten diff to a vector
                diff = diff.flatten()
                frobenius_norm = np.linalg.norm(diff)
                frobenius_norms[key].append(frobenius_norm)
        
        # Create plots
        n_moments = len(available_moment_keys)
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (6 * n_moments, 5)
        
        fig, axes = plt.subplots(1, n_moments, figsize=figsize, squeeze=False)
        axes = axes.flatten()  # Ensure axes is always 1D
        
        # Moment names for labeling
        moment_names = {
            'mean': 'Mean Vector',
            'cov': 'Covariance Matrix',
            'third_moment': 'Third Central Moment Tensor',
            'fourth_moment': 'Fourth Central Moment Tensor'
        }
        
        # Colors for consistency
        colors = ['blue', 'green', 'red', 'orange']
        
        # Plot Frobenius norms for each moment type
        for i, key in enumerate(available_moment_keys):
            ax = axes[i]
            
            # Plot Frobenius norm evolution
            frobenius_values = frobenius_norms[key]
            ax.plot(available_epochs, frobenius_values, 
                   marker='o', linewidth=2, markersize=6, 
                   color=colors[i % len(colors)], 
                   label=f'{moment_names[key]} Frobenius Norm')
            
            # Formatting
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Frobenius Norm')
            ax.set_title(f'{moment_names[key]}\nFrobenius Norm vs Ground Truth')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set y-axis to log scale if values span multiple orders of magnitude
            if max(frobenius_values) / min(frobenius_values) > 100:
                ax.set_yscale('log')
            
            # Add text box with final value
            final_value = frobenius_values[-1]
            initial_value = frobenius_values[0]
            improvement = ((initial_value - final_value) / initial_value) * 100
            
            text_str = f'Final: {final_value:.6f}'
            if len(frobenius_values) > 1:
                text_str += f'\nInitial: {initial_value:.6f}'
                text_str += f'\nImprovement: {improvement:.1f}%'
            
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots if any
        for i in range(n_moments, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        noise_type = noise_args.get('noise_type', 'unknown')
        test_point_info = ""
        if not is_affine_model and x_test is not None:
            x_val = x_input.cpu().numpy()[0]
            if len(x_val) == 1:
                test_point_info = f" (X = {x_val[0]:.3f})"
            else:
                test_point_info = f" (X = [{', '.join([f'{v:.3f}' for v in x_val])}])"
        
        fig.suptitle(f'Moment Tensors Frobenius Norm Evolution During Training\n'
                     f'Noise Type: {noise_type}{test_point_info}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the plot
        noise_type_str = noise_type.replace('_', '-')
        save_path = os.path.join(base_save_dir, f'training_progression_moment_tensors_{noise_type_str}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved moment tensors progression plot to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Warning: Failed to create moment tensors progression plot: {e}")
        import traceback
        traceback.print_exc()
        return None

 