"""
Visualisation module for discrimination experiments.

This module provides plotting functions to visualise the results of discrimination
experiments, including distribution comparisons, power analysis, and ROC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DiscriminationPlotter:
    """Class for generating visualisation plots for discrimination experiments."""
    
    def __init__(self, results: Dict, config: Dict, output_dir: Path):
        """
        Initialise the plotter.
        
        Args:
            results: Dictionary containing experiment results
            config: Configuration dictionary
            output_dir: Output directory for saving plots
        """
        self.results = results
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = output_dir / 'plots'
        
        # Set up plotting style
        self.setup_plotting_style()
        
        # Extract plot configuration
        self.plot_config = config.get('plotting', {})
        self.dpi = self.plot_config.get('dpi', 300)
        self.figsize = tuple(self.plot_config.get('figsize', [10, 8]))
        self.save_formats = self.plot_config.get('save_formats', ['png'])
        
    def setup_plotting_style(self):
        """Set up the plotting style."""
        style = self.config.get('plotting', {}).get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default colour palette
        sns.set_palette("husl")
    
    def save_figure(self, fig: plt.Figure, filename: str):
        """
        Save figure in specified formats.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
        """
        for fmt in self.save_formats:
            filepath = self.plots_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
    
    def plot_distance_distributions(self):
        """Plot histograms comparing null vs alternative hypothesis distributions."""
        plot_types = self.plot_config.get('plots', [])
        if 'distance_distributions' not in plot_types:
            return
            
        # Get all combinations
        gt_names = [gt['name'] for gt in self.config['ground_truth_distributions']]
        pert_names = [p['name'] for p in self.config['perturbations']]
        dist_names = [d['name'] for d in self.config['distances']]
        
        for gt_name in gt_names:
            for dist_name in dist_names:
                # Create figure for this ground truth + distance combination
                n_perturbations = len(pert_names)
                fig, axes = plt.subplots(1, n_perturbations, figsize=(5*n_perturbations, 6))
                if n_perturbations == 1:
                    axes = [axes]
                
                for i, pert_name in enumerate(pert_names):
                    ax = axes[i]
                    
                    # Get data
                    null_key = f"{gt_name}_{dist_name}"
                    alt_key = f"{gt_name}_{pert_name}_{dist_name}"
                    
                    if (null_key in self.results['null_hypothesis'] and 
                        alt_key in self.results['alternative_hypothesis']):
                        
                        null_distances = self.results['null_hypothesis'][null_key]['distances']
                        alt_distances = self.results['alternative_hypothesis'][alt_key]['distances']
                        
                        # Plot histograms
                        ax.hist(null_distances, bins=50, alpha=0.7, label='Null (H₀)', 
                               density=True, color='skyblue')
                        ax.hist(alt_distances, bins=50, alpha=0.7, label='Alternative (H₁)', 
                               density=True, color='salmon')
                        
                        # Add vertical lines for means
                        ax.axvline(np.mean(null_distances), color='blue', 
                                  linestyle='--', label=f'H₀ mean: {np.mean(null_distances):.3f}')
                        ax.axvline(np.mean(alt_distances), color='red', 
                                  linestyle='--', label=f'H₁ mean: {np.mean(alt_distances):.3f}')
                        
                        ax.set_xlabel('Distance Value')
                        ax.set_ylabel('Density')
                        # Create enhanced perturbation label
                        pert_label = self.create_perturbation_label(pert_name)
                        ax.set_title(f'{pert_label}\n({dist_name})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'Distance Distributions: {gt_name}', fontsize=16)
                plt.tight_layout()
                
                filename = f'distributions_{gt_name}_{dist_name}'
                self.save_figure(fig, filename)
                plt.close(fig)
    
    def plot_power_curves(self):
        """Plot power curves for different significance levels."""
        plot_types = self.plot_config.get('plots', [])
        if 'power_curves' not in plot_types:
            return
            
        thresholds = self.config['analysis']['significance_thresholds']
        
        # Create a comprehensive power analysis plot
        fig, axes = plt.subplots(len(thresholds), 1, figsize=(12, 4*len(thresholds)))
        if len(thresholds) == 1:
            axes = [axes]
        
        for i, alpha in enumerate(thresholds):
            ax = axes[i]
            
            # Collect power data
            power_data = []
            
            for gt_name in [gt['name'] for gt in self.config['ground_truth_distributions']]:
                for pert_name in [p['name'] for p in self.config['perturbations']]:
                    for dist_name in [d['name'] for d in self.config['distances']]:
                        
                        power_key = f"power_{gt_name}_{pert_name}_{dist_name}"
                        
                        if power_key in self.results['statistics']:
                            power_stats = self.results['statistics'][power_key]
                            power_val = power_stats.get(f'power_alpha_{alpha}')
                            effect_size = power_stats.get(f'effect_size_alpha_{alpha}')
                            
                            if power_val is not None and effect_size is not None:
                                power_data.append({
                                    'Ground Truth': gt_name,
                                    'Perturbation': pert_name,
                                    'Distance': dist_name,
                                    'Power': power_val,
                                    'Effect Size': effect_size
                                })
            
            if power_data:
                df = pd.DataFrame(power_data)
                
                # Create grouped bar plot
                pivot_df = df.pivot_table(
                    values='Power', 
                    index=['Ground Truth', 'Perturbation'], 
                    columns='Distance'
                )
                
                pivot_df.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title(f'Statistical Power (α = {alpha})')
                ax.set_ylabel('Power')
                ax.set_xlabel('Ground Truth + Perturbation')
                ax.legend(title='Distance Function', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Add horizontal line at power = 0.8
                ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')

                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        self.save_figure(fig, 'power_curves')
        plt.close(fig)
    
    def plot_roc_curves(self):
        """Plot ROC curves for discrimination performance."""
        plot_types = self.plot_config.get('plots', [])
        if 'roc_curves' not in plot_types:
            return
            
        # Get all combinations
        gt_names = [gt['name'] for gt in self.config['ground_truth_distributions']]
        pert_names = [p['name'] for p in self.config['perturbations']]
        dist_names = [d['name'] for d in self.config['distances']]
        
        for gt_name in gt_names:
            for pert_name in pert_names:
                fig, ax = plt.subplots(figsize=self.figsize)
                
                for dist_name in dist_names:
                    null_key = f"{gt_name}_{dist_name}"
                    alt_key = f"{gt_name}_{pert_name}_{dist_name}"
                    
                    if (null_key in self.results['null_hypothesis'] and 
                        alt_key in self.results['alternative_hypothesis']):
                        
                        null_distances = self.results['null_hypothesis'][null_key]['distances']
                        alt_distances = self.results['alternative_hypothesis'][alt_key]['distances']
                        
                        # Compute ROC curve
                        fpr, tpr, auc = self.compute_roc_curve(null_distances, alt_distances)
                        
                        ax.plot(fpr, tpr, label=f'{dist_name} (AUC = {auc:.3f})', linewidth=2)
                
                # Add diagonal line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
                
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                # Create enhanced perturbation label
                pert_label = self.create_perturbation_label(pert_name)
                ax.set_title(f'ROC Curves: {gt_name} vs {pert_label}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                filename = f'roc_{gt_name}_{pert_name}'
                self.save_figure(fig, filename)
                plt.close(fig)
    
    def compute_roc_curve(self, null_distances: np.ndarray, 
                         alt_distances: np.ndarray) -> tuple:
        """
        Compute ROC curve for binary classification.
        
        Args:
            null_distances: Distances under null hypothesis
            alt_distances: Distances under alternative hypothesis
            
        Returns:
            Tuple of (fpr, tpr, auc)
        """
        # Create labels (0 for null, 1 for alternative)
        y_true = np.concatenate([
            np.zeros(len(null_distances)),
            np.ones(len(alt_distances))
        ])
        
        # Combine distances (higher distance should indicate alternative)
        y_scores = np.concatenate([null_distances, alt_distances])
        
        # Compute ROC curve
        thresholds = np.sort(np.unique(y_scores))
        fpr = []
        tpr = []
        
        for threshold in thresholds:
            # Predict alternative if distance > threshold
            y_pred = (y_scores > threshold).astype(int)
            
            # Compute true/false positive rates
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr.append(tpr_val)
            fpr.append(fpr_val)
        
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return fpr, tpr, auc
    
    def plot_qq_plots(self):
        """Plot Q-Q plots for distribution comparison."""
        plot_types = self.plot_config.get('plots', [])
        if 'qq_plots' not in plot_types:
            return
            
        # Get all combinations
        gt_names = [gt['name'] for gt in self.config['ground_truth_distributions']]
        pert_names = [p['name'] for p in self.config['perturbations']]
        dist_names = [d['name'] for d in self.config['distances']]
        
        for gt_name in gt_names:
            for dist_name in dist_names:
                n_perturbations = len(pert_names)
                fig, axes = plt.subplots(1, n_perturbations, figsize=(5*n_perturbations, 5))
                if n_perturbations == 1:
                    axes = [axes]
                
                for i, pert_name in enumerate(pert_names):
                    ax = axes[i]
                    
                    null_key = f"{gt_name}_{dist_name}"
                    alt_key = f"{gt_name}_{pert_name}_{dist_name}"
                    
                    if (null_key in self.results['null_hypothesis'] and 
                        alt_key in self.results['alternative_hypothesis']):
                        
                        null_distances = self.results['null_hypothesis'][null_key]['distances']
                        alt_distances = self.results['alternative_hypothesis'][alt_key]['distances']
                        
                        # Compute quantiles
                        n_quantiles = min(len(null_distances), len(alt_distances))
                        quantiles = np.linspace(0, 1, n_quantiles)
                        
                        null_quantiles = np.quantile(null_distances, quantiles)
                        alt_quantiles = np.quantile(alt_distances, quantiles)
                        
                        # Plot Q-Q plot
                        ax.scatter(null_quantiles, alt_quantiles, alpha=0.6, s=20)
                        
                        # Add diagonal line
                        min_val = min(null_quantiles.min(), alt_quantiles.min())
                        max_val = max(null_quantiles.max(), alt_quantiles.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                        
                        ax.set_xlabel('Null Hypothesis Quantiles')
                        ax.set_ylabel('Alternative Hypothesis Quantiles')
                        # Create enhanced perturbation label
                        pert_label = self.create_perturbation_label(pert_name)
                        ax.set_title(f'Q-Q Plot: {pert_label}')
                        ax.grid(True, alpha=0.3)
                        
                        # Add correlation coefficient
                        corr = np.corrcoef(null_quantiles, alt_quantiles)[0, 1]
                        ax.text(0.05, 0.95, f'ρ = {corr:.3f}', 
                               transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.suptitle(f'Q-Q Plots: {gt_name} ({dist_name})', fontsize=14)
                plt.tight_layout()
                
                filename = f'qq_{gt_name}_{dist_name}'
                self.save_figure(fig, filename)
                plt.close(fig)
    
    def create_perturbation_label(self, perturbation_name: str) -> str:
        """
        Create a descriptive label for a perturbation including its parameters.
        
        Args:
            perturbation_name: Basic name of the perturbation
            
        Returns:
            Descriptive label with parameters
        """
        try:
            if perturbation_name == 'None':
                return 'None'
                
            # Find the perturbation configuration
            perturbation_config = None
            for pert in self.config['perturbations']:
                if pert['name'] == perturbation_name:
                    perturbation_config = pert
                    break
            
            if perturbation_config is None:
                return perturbation_name
            
            # Extract key parameters for the label
            params = perturbation_config.get('parameters', {})
            
            if perturbation_name == 'location_shift':
                # Extract the shift values
                constant = params.get('perturbation_sampler', {}).get('parameters', {}).get('constant', [])
                if constant is not None and len(constant) > 0:
                    if isinstance(constant, str):  # Handle numpy array string representation
                        # Parse "[0.5 0. ]" format
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', constant)
                        if len(numbers) >= 2:
                            return f"{perturbation_name}_{numbers[0]}_{numbers[1]}"
                    elif isinstance(constant, (list, tuple, np.ndarray)):
                        return f"{perturbation_name}_{constant[0]}_{constant[1]}"
                
            elif perturbation_name == 'scale_perturbation':
                # Extract the coefficient values
                coefficients = params.get('coefficients', [])
                if coefficients is not None and len(coefficients) >= 2:
                    return f"{perturbation_name}_{coefficients[0]}_{coefficients[1]}"
            
            elif perturbation_name == 'heavy_tail_perturbation':
                # Extract the coefficient and df values
                coefficients = params.get('coefficients', [])
                df = params.get('perturbation_sampler', {}).get('parameters', {}).get('df')
                if coefficients is not None and len(coefficients) >= 2 and df is not None:
                    return f"{perturbation_name}_{coefficients[1]}_{df}"
            
        except Exception as e:
            # If anything goes wrong, just return the original name
            print(f"Warning: Failed to create enhanced label for {perturbation_name}: {e}")
        
        # Fallback to original name if parameters couldn't be extracted
        return perturbation_name

    def plot_box_plots(self):
        """Plot box plots of distance distributions."""
        plot_types = self.plot_config.get('plots', [])
        if 'box_plots' not in plot_types:
            return
            
        # Collect all data for comprehensive box plot
        data_list = []
        
        # Add null hypothesis data
        for key, result in self.results['null_hypothesis'].items():
            for distance in result['distances']:
                data_list.append({
                    'Distance': distance,
                    'Ground Truth': result['ground_truth'],
                    'Distance Function': result['distance_function'],
                    'Hypothesis': 'Null',
                    'Perturbation': self.create_perturbation_label('None')
                })
        
        # Add alternative hypothesis data
        for key, result in self.results['alternative_hypothesis'].items():
            for distance in result['distances']:
                # Create enhanced perturbation label with parameters
                perturbation_label = self.create_perturbation_label(result['perturbation'])
                data_list.append({
                    'Distance': distance,
                    'Ground Truth': result['ground_truth'],
                    'Distance Function': result['distance_function'],
                    'Hypothesis': 'Alternative',
                    'Perturbation': perturbation_label
                })
        
        if not data_list:
            return
            
        df = pd.DataFrame(data_list)
        
        # Create box plots grouped by ground truth and distance function
        gt_names = df['Ground Truth'].unique()
        dist_names = df['Distance Function'].unique()
        
        for gt_name in gt_names:
            fig, axes = plt.subplots(1, len(dist_names), figsize=(5*len(dist_names), 6))
            if len(dist_names) == 1:
                axes = [axes]
            
            for i, dist_name in enumerate(dist_names):
                ax = axes[i]
                
                # Filter data
                subset = df[(df['Ground Truth'] == gt_name) & 
                           (df['Distance Function'] == dist_name)]
                
                if len(subset) > 0:
                    # Create box plot
                    sns.boxplot(data=subset, x='Hypothesis', y='Distance', 
                               hue='Perturbation', ax=ax, showfliers=False)
                    
                    ax.set_title(f'{dist_name}')
                    ax.set_xlabel('Hypothesis')
                    ax.set_ylabel('Distance Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels if needed
                    if len(subset['Perturbation'].unique()) > 3:
                        ax.tick_params(axis='x', rotation=45)
            
            plt.suptitle(f'Distance Distributions: {gt_name}', fontsize=14)
            plt.tight_layout()
            
            filename = f'boxplot_{gt_name}'
            self.save_figure(fig, filename)
            plt.close(fig)
    
    def plot_effect_sizes(self):
        """Plot effect sizes across different configurations."""
        # Collect effect size data
        effect_data = []
        
        for key, stats in self.results['statistics'].items():
            if key.startswith('power_'):
                # Extract components from key
                parts = key.replace('power_', '').split('_')
                if len(parts) >= 3:
                    gt_name = parts[0]
                    pert_name = parts[1]
                    dist_name = '_'.join(parts[2:])
                    
                    for stat_key, value in stats.items():
                        if stat_key.startswith('effect_size_alpha_'):
                            alpha = float(stat_key.replace('effect_size_alpha_', ''))
                            # Create enhanced perturbation label
                            pert_label = self.create_perturbation_label(pert_name)
                            effect_data.append({
                                'Ground Truth': gt_name,
                                'Perturbation': pert_label,
                                'Distance': dist_name,
                                'Alpha': alpha,
                                'Effect Size': value
                            })
        
        if not effect_data:
            return
            
        df = pd.DataFrame(effect_data)
        
        # Create effect size plots
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by alpha level
        for alpha in df['Alpha'].unique():
            alpha_data = df[df['Alpha'] == alpha]
            
            pivot_df = alpha_data.pivot_table(
                values='Effect Size',
                index=['Ground Truth', 'Perturbation'],
                columns='Distance'
            )
            
            # Plot as grouped bar chart
            x_pos = np.arange(len(pivot_df.index))
            width = 0.8 / len(df['Alpha'].unique())
            offset = (alpha - 0.05) * 10 * width  # Adjust offset based on alpha
            
            for i, dist_name in enumerate(pivot_df.columns):
                values = pivot_df[dist_name].values
                ax.bar(x_pos + offset, values, width, 
                      label=f'{dist_name} (α={alpha})', alpha=0.8)
        
        ax.set_xlabel('Ground Truth + Perturbation')
        ax.set_ylabel('Effect Size (Cohen\'s d)')
        ax.set_title('Effect Sizes Across Configurations')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{gt}\n{pert}' for gt, pert in pivot_df.index], rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for effect size interpretation
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        
        plt.tight_layout()
        self.save_figure(fig, 'effect_sizes')
        plt.close(fig)
    
    def generate_summary_report(self):
        """Generate a summary report with key findings."""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("DISCRIMINATION EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Experiment configuration
            f.write("CONFIGURATION:\n")
            f.write(f"- Number of trials: {self.config['experiment']['n_trials']}\n")
            f.write(f"- Sample sizes: M={self.config['sample_sizes']['M']}, N={self.config['sample_sizes']['N']}\n")
            f.write(f"- Ground truth distributions: {len(self.config['ground_truth_distributions'])}\n")
            f.write(f"- Perturbations: {len(self.config['perturbations'])}\n")
            f.write(f"- Distance functions: {len(self.config['distances'])}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY:\n")
            f.write(f"- Null hypothesis experiments: {len(self.results['null_hypothesis'])}\n")
            f.write(f"- Alternative hypothesis experiments: {len(self.results['alternative_hypothesis'])}\n\n")
            
            # Power analysis summary
            if 'power_' in str(self.results['statistics']):
                f.write("POWER ANALYSIS:\n")
                
                # Find best performing distance functions
                power_summary = {}
                for key, stats in self.results['statistics'].items():
                    if key.startswith('power_'):
                        for stat_key, value in stats.items():
                            if stat_key.startswith('power_alpha_0.05'):  # Focus on α=0.05
                                parts = key.replace('power_', '').split('_')
                                if len(parts) >= 3:
                                    dist_name = '_'.join(parts[2:])
                                    if dist_name not in power_summary:
                                        power_summary[dist_name] = []
                                    power_summary[dist_name].append(value)
                
                # Compute average power for each distance function
                avg_powers = {dist: np.mean(powers) for dist, powers in power_summary.items()}
                sorted_distances = sorted(avg_powers.items(), key=lambda x: x[1], reverse=True)
                
                f.write("Average statistical power (α=0.05) by distance function:\n")
                for dist_name, avg_power in sorted_distances:
                    f.write(f"  {dist_name}: {avg_power:.3f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            if sorted_distances:
                best_distance = sorted_distances[0][0]
                f.write(f"- Best performing distance function: {best_distance}\n")
                f.write(f"- Consider using {best_distance} for discrimination tasks\n")
            
            f.write("- Review individual plots for detailed analysis\n")
            f.write("- Consider adjusting sample sizes if power is low\n")
            f.write("- Experiment with different perturbation strengths\n")
    
    def generate_all_plots(self):
        """Generate all requested plots."""
        plot_types = self.plot_config.get('plots', [])
        
        print("Generating visualisation plots...")
        
        if 'distance_distributions' in plot_types:
            print("  - Distance distributions...")
            self.plot_distance_distributions()
        
        if 'power_curves' in plot_types:
            print("  - Power curves...")
            self.plot_power_curves()
        
        if 'roc_curves' in plot_types:
            print("  - ROC curves...")
            self.plot_roc_curves()
        
        if 'qq_plots' in plot_types:
            print("  - Q-Q plots...")
            self.plot_qq_plots()
        
        if 'box_plots' in plot_types:
            print("  - Box plots...")
            self.plot_box_plots()
        
        # Generate additional plots
        print("  - Effect sizes...")
        self.plot_effect_sizes()
        
        # Generate summary report
        print("  - Summary report...")
        self.generate_summary_report()
        
        print(f"All plots saved to: {self.plots_dir}") 