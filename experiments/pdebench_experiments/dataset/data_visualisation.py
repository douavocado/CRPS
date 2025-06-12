import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import warnings

from dataset import CFD2DDataset, get_example_config


class CFDDataVisualiser:
    """
    Comprehensive visualisation tools for CFD2DDataset.
    """
    
    def __init__(self, dataset: CFD2DDataset):
        """
        Initialise visualiser with a dataset.
        
        Args:
            dataset: CFD2DDataset instance
        """
        self.dataset = dataset
        self.cmap_div = 'RdBu_r'  # Diverging colormap for velocity components
        self.cmap_seq = 'viridis'  # Sequential colormap for scalar fields
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_sample_overview(self, sample_idx: int = 0, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create an overview plot of a single sample showing input and target data.
        
        Args:
            sample_idx: Index of sample to visualise
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        sample = self.dataset[sample_idx]
        
        # Extract data
        input_data = sample['input_data']  # (time_lag, n_input_spatial, n_input_vars)
        input_coords = sample['input_coords']  # (n_input_spatial, 2)
        target_data = sample['target_data']  # (time_predict, n_target_spatial)
        target_coords = sample['target_coords']  # (n_target_spatial, 2)
        
        # Get variable names
        var_names = self.dataset.get_variable_names()
        input_vars = var_names['input_variables']
        target_var = var_names['target_variable']
        
        # Create figure
        n_input_vars = len(input_vars)
        n_cols = max(n_input_vars, 2)
        n_rows = 3  # Input variables, spatial sampling, target comparison
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot input variables at last timestep
        for i, var in enumerate(input_vars):
            ax = fig.add_subplot(gs[0, i])
            var_data = input_data[-1, :, i].numpy()  # Last timestep, all spatial points, variable i
            
            scatter = ax.scatter(input_coords[:, 0], input_coords[:, 1], 
                               c=var_data, cmap=self._get_colormap(var), s=20, alpha=0.7)
            ax.set_title(f'Input {var} (t={self.dataset.time_lag-1})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        # Plot spatial sampling comparison
        ax_spatial = fig.add_subplot(gs[1, :2])
        ax_spatial.scatter(input_coords[:, 0], input_coords[:, 1], 
                          c='blue', s=30, alpha=0.6, label=f'Input points ({len(input_coords)})')
        ax_spatial.scatter(target_coords[:, 0], target_coords[:, 1], 
                          c='red', s=30, alpha=0.6, label=f'Target points ({len(target_coords)})')
        ax_spatial.set_title('Spatial Sampling Locations')
        ax_spatial.set_xlabel('x')
        ax_spatial.set_ylabel('y')
        ax_spatial.legend()
        ax_spatial.grid(True, alpha=0.3)
        
        # Plot target data evolution
        for t in range(target_data.shape[0]):
            col_idx = t if target_data.shape[0] <= n_cols else t % n_cols
            row_offset = t // n_cols
            ax = fig.add_subplot(gs[2 + row_offset, col_idx])
            
            target_t = target_data[t, :].numpy()
            scatter = ax.scatter(target_coords[:, 0], target_coords[:, 1], 
                               c=target_t, cmap=self._get_colormap(target_var), s=20, alpha=0.7)
            ax.set_title(f'Target {target_var} (t+{t+1})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        fig.suptitle(f'CFD Sample {sample_idx} Overview\n'
                    f'Input vars: {input_vars}, Target: {target_var}', fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_temporal_evolution(self, sample_idx: int = 0, variable: Optional[str] = None, 
                               figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plot temporal evolution of a variable for a sample.
        
        Args:
            sample_idx: Index of sample to visualise
            variable: Variable to plot (if None, plots all input variables)
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        sample = self.dataset[sample_idx]
        input_data = sample['input_data']
        input_coords = sample['input_coords']
        var_names = self.dataset.get_variable_names()
        
        if variable is None:
            variables_to_plot = var_names['input_variables']
        else:
            variables_to_plot = [variable]
        
        n_vars = len(variables_to_plot)
        n_times = input_data.shape[0]
        
        fig, axes = plt.subplots(n_vars, n_times, figsize=figsize, 
                                squeeze=False if n_vars > 1 or n_times > 1 else True)
        if n_vars == 1 and n_times == 1:
            axes = np.array([[axes]])
        elif n_vars == 1:
            axes = axes.reshape(1, -1)
        elif n_times == 1:
            axes = axes.reshape(-1, 1)
        
        for var_idx, var in enumerate(variables_to_plot):
            input_var_idx = var_names['input_variables'].index(var)
            
            # Find global min/max for consistent colorscale
            var_data_all_times = input_data[:, :, input_var_idx].numpy()
            vmin, vmax = np.percentile(var_data_all_times, [5, 95])
            
            for t in range(n_times):
                ax = axes[var_idx, t]
                var_data = input_data[t, :, input_var_idx].numpy()
                
                scatter = ax.scatter(input_coords[:, 0], input_coords[:, 1], 
                                   c=var_data, cmap=self._get_colormap(var), 
                                   s=15, alpha=0.8, vmin=vmin, vmax=vmax)
                
                ax.set_title(f'{var} (t={t})')
                if var_idx == n_vars - 1:
                    ax.set_xlabel('x')
                if t == 0:
                    ax.set_ylabel('y')
                
                # Add colorbar to rightmost subplot of each row
                if t == n_times - 1:
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        fig.suptitle(f'Temporal Evolution - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_variable_distributions(self, n_samples: int = 100, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot distributions of all variables across multiple samples.
        
        Args:
            n_samples: Number of samples to use for statistics
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        var_names = self.dataset.get_variable_names()
        all_vars = var_names['all_variables']
        
        # Collect data from multiple samples
        var_data_dict = {var: [] for var in all_vars}
        
        n_samples = min(n_samples, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)
        
        print(f"Computing distributions from {n_samples} samples...")
        
        for idx in sample_indices:
            sample = self.dataset[idx]
            input_data = sample['input_data']
            target_data = sample['target_data']
            
            # Input variables
            for var_idx, var in enumerate(var_names['input_variables']):
                var_data_dict[var].extend(input_data[:, :, var_idx].flatten().numpy())
            
            # Target variable
            target_var = var_names['target_variable']
            var_data_dict[target_var].extend(target_data.flatten().numpy())
        
        # Create plots
        n_vars = len(all_vars)
        n_cols = min(4, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(all_vars):
            ax = axes[i]
            data = np.array(var_data_dict[var])
            
            # Histogram
            ax.hist(data, bins=50, alpha=0.7, density=True, color=sns.color_palette()[i % 10])
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'±1σ: {std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8)
            
            ax.set_title(f'{var} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Variable Distributions ({n_samples} samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_spatial_correlation(self, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot spatial correlation patterns between input and target locations.
        
        Args:
            sample_idx: Index of sample to visualise
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        sample = self.dataset[sample_idx]
        input_coords = sample['input_coords'].numpy()
        target_coords = sample['target_coords'].numpy()
        input_data = sample['input_data']
        target_data = sample['target_data']
        
        var_names = self.dataset.get_variable_names()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Spatial coverage
        ax = axes[0, 0]
        ax.scatter(input_coords[:, 0], input_coords[:, 1], c='blue', s=20, alpha=0.6, label='Input points')
        ax.scatter(target_coords[:, 0], target_coords[:, 1], c='red', s=20, alpha=0.6, label='Target points')
        ax.set_title('Spatial Coverage')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distance distribution
        ax = axes[0, 1]
        # Compute distances from each target point to nearest input point
        distances = []
        for target_pt in target_coords:
            dist_to_inputs = np.linalg.norm(input_coords - target_pt[None, :], axis=1)
            distances.append(np.min(dist_to_inputs))
        
        ax.hist(distances, bins=20, alpha=0.7, color='green')
        ax.set_title('Target-to-Input Distance Distribution')
        ax.set_xlabel('Distance to nearest input point')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Input variable correlations (last timestep)
        ax = axes[1, 0]
        if len(var_names['input_variables']) > 1:
            input_last = input_data[-1, :, :].numpy()  # Last timestep
            corr_matrix = np.corrcoef(input_last.T)
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(var_names['input_variables'])))
            ax.set_yticks(range(len(var_names['input_variables'])))
            ax.set_xticklabels(var_names['input_variables'], rotation=45)
            ax.set_yticklabels(var_names['input_variables'])
            ax.set_title('Input Variable Correlations')
            
            # Add correlation values
            for i in range(len(var_names['input_variables'])):
                for j in range(len(var_names['input_variables'])):
                    ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Need multiple input\nvariables for correlation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Input Variable Correlations')
        
        # Plot 4: Temporal variance
        ax = axes[1, 1]
        if input_data.shape[0] > 1:  # Multiple timesteps
            temporal_var = torch.var(input_data, dim=0).numpy()  # Variance across time
            
            for var_idx, var in enumerate(var_names['input_variables']):
                var_variance = temporal_var[:, var_idx]
                ax.scatter(input_coords[:, 0], input_coords[:, 1], 
                          c=var_variance, s=15, alpha=0.7, 
                          label=var, cmap=self.cmap_seq)
            
            ax.set_title('Temporal Variance of Input Variables')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            ax.text(0.5, 0.5, 'Need multiple timesteps\nfor temporal analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Temporal Variance')
        
        fig.suptitle(f'Spatial Analysis - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_data_statistics(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive statistics about the dataset.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        info = self.dataset.get_data_info()
        var_names = self.dataset.get_variable_names()
        norm_stats = self.dataset.get_normalisation_stats()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Dataset structure info
        ax = axes[0, 0]
        info_text = f"""Dataset Structure:
• Samples: {info['n_samples']}
• Temporal windows: {info['n_temporal_windows']}
• Spatial shape: {info['spatial_shape']}
• Time steps: {info['n_timesteps']}
• Time lag: {info['time_lag']}
• Time predict: {info['time_predict']}
• Input variables: {info['n_input_variables']}
• Input spatial points: {info['input_spatial_points']}
• Target spatial points: {info['target_spatial_points']}"""
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Dataset Information')
        ax.axis('off')
        
        # Variable information
        ax = axes[0, 1]
        var_text = f"""Variables:
• All variables: {var_names['all_variables']}
• Input variables: {var_names['input_variables']}
• Target variable: {var_names['target_variable']}

Split: {self.dataset.split}
Normalisation: {self.dataset.normalise}"""
        
        ax.text(0.05, 0.95, var_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Variable Configuration')
        ax.axis('off')
        
        # Normalisation statistics
        ax = axes[0, 2]
        if norm_stats:
            norm_text = "Normalisation Stats:\n"
            for var, stats in norm_stats.items():
                norm_text += f"• {var}:\n"
                norm_text += f"  Mean: {stats['mean']:.4f}\n"
                norm_text += f"  Std: {stats['std']:.4f}\n"
        else:
            norm_text = "Normalisation: Disabled"
        
        ax.text(0.05, 0.95, norm_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Normalisation Statistics')
        ax.axis('off')
        
        # Sample a few data points for statistics
        n_samples_stats = min(50, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), size=n_samples_stats, replace=False)
        
        input_spatial_counts = []
        target_spatial_counts = []
        input_shapes = []
        target_shapes = []
        
        for idx in sample_indices:
            sample = self.dataset[idx]
            input_spatial_counts.append(sample['input_coords'].shape[0])
            target_spatial_counts.append(sample['target_coords'].shape[0])
            input_shapes.append(sample['input_data'].shape)
            target_shapes.append(sample['target_data'].shape)
        
        # Spatial points distribution
        ax = axes[1, 0]
        ax.hist(input_spatial_counts, bins=20, alpha=0.7, label='Input points', color='blue')
        ax.hist(target_spatial_counts, bins=20, alpha=0.7, label='Target points', color='red')
        ax.set_title('Spatial Points Distribution')
        ax.set_xlabel('Number of points')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Data shape consistency
        ax = axes[1, 1]
        unique_input_shapes = list(set(input_shapes))
        unique_target_shapes = list(set(target_shapes))
        
        shape_text = f"""Data Shape Consistency:
Input shapes: {len(unique_input_shapes)} unique
• {unique_input_shapes[:3]}{'...' if len(unique_input_shapes) > 3 else ''}

Target shapes: {len(unique_target_shapes)} unique  
• {unique_target_shapes[:3]}{'...' if len(unique_target_shapes) > 3 else ''}

Samples checked: {n_samples_stats}"""
        
        ax.text(0.05, 0.95, shape_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Shape Analysis')
        ax.axis('off')
        
        # Memory usage estimate
        ax = axes[1, 2]
        
        # Estimate memory usage
        sample = self.dataset[0]
        sample_size = 0
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample_size += value.numel() * value.element_size()
        
        total_memory_mb = (sample_size * len(self.dataset)) / (1024 * 1024)
        
        memory_text = f"""Memory Analysis:
• Sample size: {sample_size / 1024:.1f} KB
• Total dataset: {total_memory_mb:.1f} MB
• Samples in split: {len(self.dataset)}

Storage efficiency:
• Input points/total: {info['input_spatial_points']}/{info['spatial_shape'][0]*info['spatial_shape'][1]:.0f}
• Compression ratio: {100 * info['input_spatial_points']/(info['spatial_shape'][0]*info['spatial_shape'][1]):.1f}%"""
        
        ax.text(0.05, 0.95, memory_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Memory Usage')
        ax.axis('off')
        
        fig.suptitle(f'Dataset Statistics - {self.dataset.split.upper()} Split', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _get_colormap(self, variable: str) -> str:
        """Get appropriate colormap for a variable."""
        velocity_vars = ['Vx', 'Vy', 'velocity_x', 'velocity_y', 'u', 'v']
        
        if any(vel_var in variable.lower() for vel_var in velocity_vars):
            return self.cmap_div  # Diverging for velocity (can be positive/negative)
        else:
            return self.cmap_seq  # Sequential for scalar fields
    
    def save_all_plots(self, output_dir: Union[str, Path], sample_idx: int = 0, 
                      prefix: str = "cfd_viz", n_dist_samples: int = 100):
        """
        Generate and save all visualisation plots.
        
        Args:
            output_dir: Directory to save plots
            sample_idx: Sample index for detailed plots
            prefix: Filename prefix
            n_dist_samples: Number of samples for distribution analysis
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating visualisations for dataset split '{self.dataset.split}'...")
        
        # Overview plot
        print("Creating sample overview...")
        fig = self.plot_sample_overview(sample_idx)
        fig.savefig(output_dir / f"{prefix}_overview_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Temporal evolution
        print("Creating temporal evolution plot...")
        fig = self.plot_temporal_evolution(sample_idx)
        fig.savefig(output_dir / f"{prefix}_temporal_evolution_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Variable distributions
        print(f"Creating variable distributions ({n_dist_samples} samples)...")
        fig = self.plot_variable_distributions(n_dist_samples)
        fig.savefig(output_dir / f"{prefix}_distributions.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Spatial correlation
        print("Creating spatial analysis...")
        fig = self.plot_spatial_correlation(sample_idx)
        fig.savefig(output_dir / f"{prefix}_spatial_analysis_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Dataset statistics
        print("Creating dataset statistics...")
        fig = self.plot_data_statistics()
        fig.savefig(output_dir / f"{prefix}_statistics.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"All plots saved to {output_dir}")


def demo_visualisation(data_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
    """
    Demonstration function showing how to use the visualiser.
    
    Args:
        data_path: Path to HDF5 data file
        output_dir: Directory to save plots (if None, displays interactively)
    """
    # Create dataset with example configuration
    config = get_example_config()
    
    print("Loading dataset...")
    dataset = CFD2DDataset(data_path, config, split='train', normalise=True)
    
    # Create visualiser
    visualiser = CFDDataVisualiser(dataset)
    
    if output_dir is not None:
        # Save all plots
        visualiser.save_all_plots(output_dir, sample_idx=0, n_dist_samples=50)
    else:
        # Display interactively
        print("Displaying plots interactively...")
        
        # Overview
        fig1 = visualiser.plot_sample_overview(0)
        plt.show()
        
        # Temporal evolution
        fig2 = visualiser.plot_temporal_evolution(0)
        plt.show()
        
        # Distributions
        fig3 = visualiser.plot_variable_distributions(50)
        plt.show()
        
        # Spatial analysis
        fig4 = visualiser.plot_spatial_correlation(0)
        plt.show()
        
        # Statistics
        fig5 = visualiser.plot_data_statistics()
        plt.show()


if __name__ == "__main__":
    # Example usage
    data_path = Path("../data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
    output_dir = Path("../visualisations")
    
    if data_path.exists():
        print(f"Running visualisation demo with data from {data_path}")
        demo_visualisation(data_path, output_dir)
    else:
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with the correct path to your HDF5 file.")
        
        # Show example of how to use with custom configuration
        print("\nExample usage:")
        print("""
from data_visualisation import CFDDataVisualiser
from dataset import CFD2DDataset

# Custom configuration
config = {
    'variables': ['Vx', 'Vy', 'density', 'pressure'],
    'input_variables': ['Vx', 'Vy'],
    'target_variable': 'pressure',
    'time_lag': 3,
    'time_predict': 2,
    'input_spatial': 512,
    'target_spatial': 128,
    'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
}

# Create dataset and visualiser
dataset = CFD2DDataset('path/to/your/data.hdf5', config, split='train')
visualiser = CFDDataVisualiser(dataset)

# Generate specific plots
fig = visualiser.plot_sample_overview(sample_idx=0)
fig = visualiser.plot_temporal_evolution(sample_idx=0, variable='Vx')
fig = visualiser.plot_variable_distributions(n_samples=100)

# Save all plots
visualiser.save_all_plots('output_directory/', sample_idx=0)
        """) 