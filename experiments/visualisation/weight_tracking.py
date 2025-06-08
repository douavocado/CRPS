import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import seaborn as sns
from .plotting import plot_prediction_samples

class WeightTracker:
    """
    Class to track and visualise the evolution of model weights during training.
    Handles both MLPSampler and SimpleAffineNormal models.
    """
    
    def __init__(self, model, track_weights=True, track_samples_every=5):
        """
        Initialize the weight tracker.
        
        Parameters:
        -----------
        model : nn.Module
            The model to track (MLPSampler or SimpleAffineNormal)
        track_weights : bool
            Whether to track weight evolution
        track_samples_every : int
            Frequency of tracking prediction samples (every N epochs)
        """
        self.model = model
        self.track_weights = track_weights
        self.track_samples_every = track_samples_every
        self.model_type = type(model).__name__
        
        # Storage for weight evolution
        self.weight_history = defaultdict(list)
        self.epoch_history = []
        
        # Storage for intermediate samples
        self.sample_epochs = []
        
    def record_weights(self, epoch):
        """
        Record the current state of model weights.
        
        Parameters:
        -----------
        epoch : int
            Current training epoch
        """
        if not self.track_weights:
            return
            
        self.epoch_history.append(epoch)
        
        if self.model_type == 'SimpleAffineNormal':
            # Track A matrix and b vector
            A_matrix = self.model.A.detach().cpu().numpy()
            b_vector = self.model.b.detach().cpu().numpy()
            
            self.weight_history['A_matrix'].append(A_matrix.copy())
            self.weight_history['b_vector'].append(b_vector.copy())
            
            # Track some derived statistics
            self.weight_history['A_frobenius_norm'].append(np.linalg.norm(A_matrix, 'fro'))
            self.weight_history['b_l2_norm'].append(np.linalg.norm(b_vector))
            
            # Track eigenvalues of covariance matrix (AA^T)
            cov_matrix = A_matrix @ A_matrix.T
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
            self.weight_history['cov_eigenvalues'].append(eigenvals.copy())
            
        elif self.model_type == 'MLPSampler':
            # Track linear layer weights and biases
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_data = param.detach().cpu().numpy()
                    self.weight_history[name].append(param_data.copy())
                    
                    # Track norms for each parameter
                    if 'weight' in name:
                        self.weight_history[f'{name}_frobenius_norm'].append(np.linalg.norm(param_data, 'fro'))
                    elif 'bias' in name or 'input_vector' in name:
                        self.weight_history[f'{name}_l2_norm'].append(np.linalg.norm(param_data))
    
    def should_record_samples(self, epoch):
        """
        Check if we should record prediction samples at this epoch.
        
        Parameters:
        -----------
        epoch : int
            Current training epoch
            
        Returns:
        --------
        bool
            Whether to record samples
        """
        return epoch % self.track_samples_every == 0 or epoch == 1
    
    def record_sample_epoch(self, epoch):
        """
        Record that we saved samples at this epoch.
        
        Parameters:
        -----------
        epoch : int
            Current training epoch
        """
        self.sample_epochs.append(epoch)
    
    def plot_weight_evolution(self, save_dir=None, noise_args=None):
        """
        Create plots showing the evolution of weights during training.
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save plots
        noise_args : dict, optional
            Dictionary containing true parameters for comparison:
            - cov_matrix: True covariance matrix
            - mean_function: True mean function (e.g., 'zero')
            
        Returns:
        --------
        list
            List of matplotlib figures
        """
        if not self.track_weights or len(self.epoch_history) == 0:
            return []
        
        figures = []
        
        if self.model_type == 'SimpleAffineNormal':
            figures.extend(self._plot_affine_weights(save_dir, noise_args))
        elif self.model_type == 'MLPSampler':
            figures.extend(self._plot_mlp_weights(save_dir, noise_args))
        
        return figures
    
    def _plot_affine_weights(self, save_dir=None, noise_args=None):
        """Plot weight evolution for SimpleAffineNormal model."""
        figures = []
        
        # Extract true values if available
        true_cov_matrix = None
        true_mean = None
        true_eigenvals = None
        true_A_example = None
        
        if noise_args is not None:
            if 'cov_matrix' in noise_args:
                true_cov_matrix = noise_args['cov_matrix']
                true_eigenvals = np.linalg.eigvals(true_cov_matrix)
                true_eigenvals = np.sort(true_eigenvals)[::-1]  # Sort in descending order
                
                # Compute an example true A matrix using Cholesky decomposition
                try:
                    true_A_example = np.linalg.cholesky(true_cov_matrix)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, use eigendecomposition
                    eigenvals, eigenvecs = np.linalg.eigh(true_cov_matrix)
                    true_A_example = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
            
            if 'mean_function' in noise_args and noise_args['mean_function'] == 'zero':
                output_dim = self.weight_history['b_vector'][0].shape[0]
                true_mean = np.zeros(output_dim)
        
        # 1. Plot covariance matrix evolution (heatmaps over time)
        A_matrices = self.weight_history['A_matrix']
        n_epochs_to_show = min(len(A_matrices), 7)  # Show at most 7 time points to leave room for true matrix
        indices = np.linspace(0, len(A_matrices)-1, n_epochs_to_show, dtype=int)
        
        fig = plt.figure(figsize=(16, 10))
        
        for i, idx in enumerate(indices):
            plt.subplot(2, 4, i+1)
            # Compute inferred covariance matrix (A @ A.T)
            inferred_cov = A_matrices[idx] @ A_matrices[idx].T
            sns.heatmap(inferred_cov, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
            plt.title(f'Inferred Covariance - Epoch {self.epoch_history[idx]}')
        
        # Add true covariance matrix if available
        if true_cov_matrix is not None:
            plt.subplot(2, 4, 8)
            sns.heatmap(true_cov_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
            plt.title('True Covariance Matrix')
        
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'covariance_matrix_evolution.png'), dpi=150, bbox_inches='tight')
        figures.append(fig)
        
        # 2. Plot b vector evolution
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # b vector over time
        b_vectors = np.array(self.weight_history['b_vector'])
        for dim in range(b_vectors.shape[1]):
            axes[0, 0].plot(self.epoch_history, b_vectors[:, dim], label=f'b[{dim}]', marker='o', markersize=3)
            
            # Add true value line if available
            if true_mean is not None:
                axes[0, 0].axhline(y=true_mean[dim], color='red', linestyle='--', alpha=0.7, 
                                 label=f'True b[{dim}]' if dim == 0 else None)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('b vector components')
        axes[0, 0].set_title('b Vector Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Norms over time
        axes[0, 1].plot(self.epoch_history, self.weight_history['A_frobenius_norm'], 
                       label='||A||_F', marker='o', markersize=3)
        axes[0, 1].plot(self.epoch_history, self.weight_history['b_l2_norm'], 
                       label='||b||_2', marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Norm')
        axes[0, 1].set_title('Parameter Norms Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Covariance eigenvalues over time
        eigenvals = np.array(self.weight_history['cov_eigenvalues'])
        for i in range(eigenvals.shape[1]):
            axes[1, 0].plot(self.epoch_history, eigenvals[:, i], 
                           label=f'λ_{i+1}', marker='o', markersize=3)
            
            # Add true eigenvalue line if available
            if true_eigenvals is not None and i < len(true_eigenvals):
                axes[1, 0].axhline(y=true_eigenvals[i], color='red', linestyle='--', alpha=0.7,
                                 label=f'True λ_{i+1}' if i == 0 else None)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Eigenvalue')
        axes[1, 0].set_title('Covariance Matrix Eigenvalues')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Condition number (ratio of largest to smallest eigenvalue)
        condition_numbers = eigenvals[:, 0] / (eigenvals[:, -1] + 1e-10)
        axes[1, 1].plot(self.epoch_history, condition_numbers, marker='o', markersize=3, color='blue', label='Learned')
        
        # Add true condition number line if available
        if true_eigenvals is not None:
            true_condition_number = true_eigenvals[0] / (true_eigenvals[-1] + 1e-10)
            axes[1, 1].axhline(y=true_condition_number, color='red', linestyle='--', alpha=0.7, 
                             label='True Condition Number')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Condition Number')
        axes[1, 1].set_title('Covariance Matrix Condition Number')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'affine_weights_evolution.png'), dpi=150, bbox_inches='tight')
        figures.append(fig)
        
        return figures
    
    def _plot_mlp_weights(self, save_dir=None, noise_args=None):
        """Plot weight evolution for MLPSampler model."""
        figures = []
        
        # Get all parameter names (excluding norms)
        param_names = [name for name in self.weight_history.keys() 
                      if not name.endswith('_norm') and not name.endswith('_frobenius_norm') and not name.endswith('_l2_norm')]
        
        # 1. Plot parameter norms evolution
        norm_names = [name for name in self.weight_history.keys() 
                     if name.endswith('_norm') or name.endswith('_frobenius_norm') or name.endswith('_l2_norm')]
        
        if norm_names:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for norm_name in norm_names:
                # Clean up the name for the legend
                clean_name = norm_name.replace('feature_extractor.', '').replace('_frobenius_norm', ' (F)').replace('_l2_norm', ' (L2)')
                ax.plot(self.epoch_history, self.weight_history[norm_name], 
                       label=clean_name, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm')
            ax.set_title('Parameter Norms Evolution')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'mlp_norms_evolution.png'), dpi=150, bbox_inches='tight')
            figures.append(fig)
        
        # 2. Plot histograms of weight values at different epochs
        weight_params = [name for name in param_names if 'weight' in name]
        
        if weight_params:
            n_params = len(weight_params)
            n_epochs_to_show = min(len(self.epoch_history), 4)
            epoch_indices = np.linspace(0, len(self.epoch_history)-1, n_epochs_to_show, dtype=int)
            
            fig, axes = plt.subplots(n_params, n_epochs_to_show, 
                                   figsize=(4*n_epochs_to_show, 3*n_params))
            
            if n_params == 1:
                axes = axes.reshape(1, -1)
            if n_epochs_to_show == 1:
                axes = axes.reshape(-1, 1)
            
            for i, param_name in enumerate(weight_params):
                for j, epoch_idx in enumerate(epoch_indices):
                    weights = self.weight_history[param_name][epoch_idx].flatten()
                    
                    axes[i, j].hist(weights, bins=30, alpha=0.7, density=True)
                    axes[i, j].set_title(f'{param_name.replace("feature_extractor.", "")}\nEpoch {self.epoch_history[epoch_idx]}')
                    axes[i, j].grid(True, alpha=0.3)
                    
                    if j == 0:
                        axes[i, j].set_ylabel('Density')
                    if i == n_params - 1:
                        axes[i, j].set_xlabel('Weight Value')
            
            plt.tight_layout()
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'mlp_weight_distributions.png'), dpi=150, bbox_inches='tight')
            figures.append(fig)
        
        # 3. Plot input_vector evolution if it exists
        if 'input_vector' in self.weight_history:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Evolution of input_vector components
            input_vectors = np.array(self.weight_history['input_vector'])
            for dim in range(min(input_vectors.shape[1], 10)):  # Show at most 10 dimensions
                axes[0].plot(self.epoch_history, input_vectors[:, dim], 
                           label=f'dim {dim}', marker='o', markersize=3)
            
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Input Vector Components')
            axes[0].set_title('Input Vector Evolution')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # Norm evolution
            axes[1].plot(self.epoch_history, self.weight_history['input_vector_l2_norm'], 
                        marker='o', markersize=3, color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('L2 Norm')
            axes[1].set_title('Input Vector Norm Evolution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'input_vector_evolution.png'), dpi=150, bbox_inches='tight')
            figures.append(fig)
        
        return figures

def create_intermediate_sample_plots(model, x_test, y_test, noise_args, epoch, 
                                   save_dir, n_samples=1000, n_points=1, device='cpu'):
    """
    Create and save prediction sample plots for intermediate training states.
    
    Parameters:
    -----------
    model : nn.Module
        The model to evaluate
    x_test : torch.Tensor or None
        Test inputs (None for SimpleAffineNormal)
    y_test : torch.Tensor
        Test targets
    noise_args : dict
        Noise arguments for plotting
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
        noise_args=noise_args
    )
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'prediction_samples_epoch_{epoch:03d}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path 