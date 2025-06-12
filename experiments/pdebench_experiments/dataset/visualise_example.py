#!/usr/bin/env python3
"""
Example script demonstrating the CFD dataset visualisation capabilities.

This script shows how to:
1. Load CFD data with different configurations
2. Create visualisations for data exploration
3. Compare different dataset configurations
4. Save visualisations for reports/presentations

Usage:
    python visualise_example.py
    
Make sure you have CFD data available in the data directory.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from dataset import CFD2DDataset, get_example_config, create_cfd_dataloader
from data_visualisation import CFDDataVisualiser


def get_data_path() -> Optional[Path]:
    """
    Find available CFD data files.
    
    Returns:
        Path to CFD data file or None if not found
    """
    # Common locations for CFD data
    possible_paths = [
        Path("../data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5"),
        Path("../data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5"),
        Path("data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5"),
        Path("../pdebench_experiments/data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def create_custom_configs() -> Dict[str, Dict]:
    """
    Create different configurations for demonstration.
    
    Returns:
        Dictionary of configuration examples
    """
    configs = {
        'basic': {
            'variables': ['Vx', 'Vy', 'density', 'pressure'],
            'input_variables': ['Vx', 'Vy', 'density'],
            'target_variable': 'pressure',
            'time_lag': 3,
            'time_predict': 1,
            'input_spatial': 512,
            'target_spatial': 64,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
        },
        
        'velocity_prediction': {
            'variables': ['Vx', 'Vy', 'density', 'pressure'],
            'input_variables': ['density', 'pressure'],
            'target_variable': 'Vx',
            'time_lag': 5,
            'time_predict': 3,
            'input_spatial': 1024,
            'target_spatial': 256,
            'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
        },
        
        'high_resolution': {
            'variables': ['Vx', 'Vy', 'density', 'pressure'],
            'input_variables': ['Vx', 'Vy'],
            'target_variable': 'density',
            'time_lag': 2,
            'time_predict': 2,
            'input_spatial': -1,  # Use all spatial points
            'target_spatial': 1024,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
        },
        
        'sparse_sampling': {
            'variables': ['Vx', 'Vy', 'density', 'pressure'],
            'input_variables': ['Vx', 'Vy', 'density', 'pressure'],
            'target_variable': 'pressure',
            'time_lag': 4,
            'time_predict': 1,
            'input_spatial': 128,
            'target_spatial': 32,
            'split_ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2}
        }
    }
    
    return configs


def demonstrate_single_configuration(data_path: Path, config_name: str, config: Dict, 
                                   output_dir: Optional[Path] = None):
    """
    Demonstrate visualisations for a single configuration.
    
    Args:
        data_path: Path to CFD data
        config_name: Name of the configuration
        config: Configuration dictionary
        output_dir: Directory to save plots (None for interactive display)
    """
    print(f"\n{'='*60}")
    print(f"Demonstrating configuration: {config_name.upper()}")
    print(f"{'='*60}")
    
    # Print configuration details
    print("Configuration details:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Create dataset
        print("Loading dataset...")
        dataset = CFD2DDataset(data_path, config, split='train', normalise=True)
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        
        # Create visualiser
        visualiser = CFDDataVisualiser(dataset)
        
        if output_dir is not None:
            # Save visualisations to files
            config_output_dir = output_dir / config_name
            print(f"Saving visualisations to: {config_output_dir}")
            visualiser.save_all_plots(config_output_dir, sample_idx=0, 
                                     prefix=f"cfd_{config_name}", n_dist_samples=50)
        else:
            # Display interactively
            print("Generating interactive plots...")
            
            # Dataset statistics
            print("1. Dataset statistics...")
            fig_stats = visualiser.plot_data_statistics()
            fig_stats.suptitle(f'Dataset Statistics - {config_name}', fontsize=16, fontweight='bold')
            plt.show()
            
            # Sample overview
            print("2. Sample overview...")
            fig_overview = visualiser.plot_sample_overview(sample_idx=0)
            fig_overview.suptitle(f'Sample Overview - {config_name}', fontsize=16, fontweight='bold')
            plt.show()
            
            # Temporal evolution
            print("3. Temporal evolution...")
            fig_temporal = visualiser.plot_temporal_evolution(sample_idx=0)
            fig_temporal.suptitle(f'Temporal Evolution - {config_name}', fontsize=16, fontweight='bold')
            plt.show()
            
            # Variable distributions
            print("4. Variable distributions...")
            fig_dist = visualiser.plot_variable_distributions(n_samples=30)
            fig_dist.suptitle(f'Variable Distributions - {config_name}', fontsize=16, fontweight='bold')
            plt.show()
            
            # Spatial analysis
            print("5. Spatial analysis...")
            fig_spatial = visualiser.plot_spatial_correlation(sample_idx=0)
            fig_spatial.suptitle(f'Spatial Analysis - {config_name}', fontsize=16, fontweight='bold')
            plt.show()
        
        print(f"Visualisations completed for {config_name}")
        
    except Exception as e:
        print(f"Error with configuration {config_name}: {str(e)}")
        print("Skipping this configuration...")


def compare_configurations(data_path: Path, configs: Dict[str, Dict], 
                         output_dir: Optional[Path] = None):
    """
    Create comparative visualisations between different configurations.
    
    Args:
        data_path: Path to CFD data
        configs: Dictionary of configurations to compare
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("CREATING COMPARATIVE VISUALISATIONS")
    print(f"{'='*60}")
    
    # Create datasets for comparison
    datasets = {}
    visualisers = {}
    
    for config_name, config in configs.items():
        try:
            print(f"Loading dataset for {config_name}...")
            dataset = CFD2DDataset(data_path, config, split='train', normalise=True)
            datasets[config_name] = dataset
            visualisers[config_name] = CFDDataVisualiser(dataset)
            print(f"  {config_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  {config_name}: Failed to load - {str(e)}")
    
    if len(datasets) < 2:
        print("Need at least 2 successful configurations for comparison.")
        return
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    
    # 1. Dataset size comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Number of samples
    ax = axes[0, 0]
    config_names = list(datasets.keys())
    sample_counts = [len(datasets[name]) for name in config_names]
    bars = ax.bar(config_names, sample_counts, color=plt.cm.Set3(np.linspace(0, 1, len(config_names))))
    ax.set_title('Number of Temporal Windows per Configuration')
    ax.set_ylabel('Number of samples')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, sample_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01, 
                f'{count}', ha='center', va='bottom')
    
    # Plot 2: Spatial sampling comparison
    ax = axes[0, 1]
    input_spatial = []
    target_spatial = []
    for name in config_names:
        info = datasets[name].get_data_info()
        input_spatial.append(info['input_spatial_points'])
        target_spatial.append(info['target_spatial_points'])
    
    x = np.arange(len(config_names))
    width = 0.35
    ax.bar(x - width/2, input_spatial, width, label='Input points', alpha=0.8)
    ax.bar(x + width/2, target_spatial, width, label='Target points', alpha=0.8)
    ax.set_title('Spatial Sampling Comparison')
    ax.set_ylabel('Number of points')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45)
    ax.legend()
    ax.set_yscale('log')  # Log scale for better comparison
    
    # Plot 3: Temporal configuration
    ax = axes[1, 0]
    time_lags = []
    time_predicts = []
    for name in config_names:
        info = datasets[name].get_data_info()
        time_lags.append(info['time_lag'])
        time_predicts.append(info['time_predict'])
    
    x = np.arange(len(config_names))
    ax.bar(x - width/2, time_lags, width, label='Time lag', alpha=0.8)
    ax.bar(x + width/2, time_predicts, width, label='Time predict', alpha=0.8)
    ax.set_title('Temporal Configuration Comparison')
    ax.set_ylabel('Number of timesteps')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45)
    ax.legend()
    
    # Plot 4: Variable configuration
    ax = axes[1, 1]
    input_vars = []
    target_vars = []
    for name in config_names:
        var_info = datasets[name].get_variable_names()
        input_vars.append(len(var_info['input_variables']))
        target_vars.append(var_info['target_variable'])
    
    ax.bar(config_names, input_vars, alpha=0.8)
    ax.set_title('Number of Input Variables')
    ax.set_ylabel('Number of variables')
    ax.tick_params(axis='x', rotation=45)
    
    # Add target variable labels
    for i, (name, target_var) in enumerate(zip(config_names, target_vars)):
        ax.text(i, input_vars[i] + 0.1, f'Target: {target_var}', 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Configuration Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    if output_dir is not None:
        comparison_dir = output_dir / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_dir / "configuration_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {comparison_dir}")
    else:
        plt.show()


def demonstrate_dataloader_usage(data_path: Path):
    """
    Demonstrate how to use the dataset with PyTorch DataLoader.
    
    Args:
        data_path: Path to CFD data
    """
    print(f"\n{'='*60}")
    print("DEMONSTRATING DATALOADER USAGE")
    print(f"{'='*60}")
    
    # Create a simple configuration
    config = {
        'variables': ['Vx', 'Vy', 'density', 'pressure'],
        'input_variables': ['Vx', 'Vy'],
        'target_variable': 'pressure',
        'time_lag': 2,
        'time_predict': 1,
        'input_spatial': 256,
        'target_spatial': 64,
        'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
    }
    
    try:
        # Create dataloaders for all splits
        print("Creating dataloaders...")
        train_loader = create_cfd_dataloader(data_path, config, split='train', 
                                           batch_size=4, shuffle=True, num_workers=0)
        val_loader = create_cfd_dataloader(data_path, config, split='val', 
                                         batch_size=4, shuffle=False, num_workers=0)
        test_loader = create_cfd_dataloader(data_path, config, split='test', 
                                          batch_size=4, shuffle=False, num_workers=0)
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Examine a batch
        print("\nExamining a training batch...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)} (first element: {value[0] if len(value) > 0 else 'empty'})")
            
            if batch_idx == 0:  # Only show first batch
                break
        
        print("DataLoader demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error in DataLoader demonstration: {str(e)}")


def main():
    """Main demonstration function."""
    print("CFD Dataset Visualisation Example")
    print("=" * 60)
    
    # Find data file
    data_path = get_data_path()
    if data_path is None:
        print("No CFD data file found!")
        print("\nPlease ensure you have a CFD HDF5 file in one of these locations:")
        print("  - ../data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
        print("  - ../data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
        print("  - data/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
        print("\nOr update the paths in get_data_path() function.")
        return
    
    print(f"✓ Found CFD data: {data_path}")
    
    # Get configurations
    configs = create_custom_configs()
    
    # Ask user what they want to do
    print("\nWhat would you like to demonstrate?")
    print("1. Interactive visualisations for all configurations")
    print("2. Save all visualisations to files")
    print("3. Show configuration comparisons only")
    print("4. Demonstrate DataLoader usage only")
    print("5. Everything (save to files)")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    # Set output directory if saving to files
    output_dir = None
    if choice in ['2', '5']:
        output_dir = Path("../visualisations/data_visualisations")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    
    # Execute based on choice
    if choice == '1':
        # Interactive visualisations
        for config_name, config in configs.items():
            demonstrate_single_configuration(data_path, config_name, config, output_dir=None)
            
            try:
                input(f"\nPress Enter to continue to next configuration, or Ctrl+C to skip...")
            except KeyboardInterrupt:
                print("\nSkipping remaining configurations...")
                break
    
    elif choice == '2':
        # Save all visualisations
        for config_name, config in configs.items():
            demonstrate_single_configuration(data_path, config_name, config, output_dir=output_dir)
    
    elif choice == '3':
        # Configuration comparison only
        compare_configurations(data_path, configs, output_dir=output_dir)
    
    elif choice == '4':
        # DataLoader demonstration only
        demonstrate_dataloader_usage(data_path)
    
    elif choice == '5':
        # Everything (save to files)
        for config_name, config in configs.items():
            demonstrate_single_configuration(data_path, config_name, config, output_dir=output_dir)
        
        compare_configurations(data_path, configs, output_dir=output_dir)
        demonstrate_dataloader_usage(data_path)
    
    else:
        print("Invalid choice. Showing basic example...")
        demonstrate_single_configuration(data_path, 'basic', configs['basic'], output_dir=None)
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETED")
    print(f"{'='*60}")
    
    if output_dir is not None:
        print(f"All visualisations saved to: {output_dir.absolute()}")
    
    print("\nTo use these visualisations in your own code:")
    print("""
from dataset import CFD2DDataset
from data_visualisation import CFDDataVisualiser

# Load dataset
config = {...}  # Your configuration
dataset = CFD2DDataset('path/to/data.hdf5', config, split='train')

# Create visualiser
visualiser = CFDDataVisualiser(dataset)

# Generate plots
fig = visualiser.plot_sample_overview(sample_idx=0)
visualiser.save_all_plots('output_dir/', sample_idx=0)
    """)


if __name__ == "__main__":
    main() 