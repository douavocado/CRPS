#!/usr/bin/env python3
"""
Discrimination Experiment Script

This script performs discrimination experiments to evaluate the power of different 
distance functions in distinguishing between distributions. It tests both null 
hypothesis scenarios (same distribution) and alternative hypothesis scenarios 
(different distributions with perturbations).

Usage:
    python run.py [--config config.yaml] [--output results/]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional
import importlib
import warnings
from datetime import datetime
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from samplers.samplers import create_sampler, CompositionSampler


class DiscriminationExperiment:
    """Main class for running discrimination experiments."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise the experiment with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_output_directory()
        self.setup_random_seed()
        
        # Storage for results
        self.results = {
            'null_hypothesis': {},
            'alternative_hypothesis': {},
            'statistics': {},
            'metadata': {
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'config_path': config_path
            }
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def setup_logging(self):
        """Set up logging configuration."""
        level = logging.INFO if self.config['advanced']['verbose'] else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{self.config['experiment']['name']}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'raw_data').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory created: {self.output_dir}")
    
    def setup_random_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.config['experiment']['random_seed']
        np.random.seed(seed)
        self.logger.info(f"Random seed set to: {seed}")
    
    def create_sampler_from_config(self, sampler_config: Dict) -> Any:
        """
        Create a sampler from configuration dictionary.
        
        Args:
            sampler_config: Dictionary containing sampler configuration
            
        Returns:
            Sampler instance
        """
        sampler_type = sampler_config['type']
        dimension = sampler_config['dimension']
        parameters = sampler_config.get('parameters', {})
        
        # Convert lists to numpy arrays for matrix parameters
        for key, value in parameters.items():
            if isinstance(value, list) and key in ['mean', 'loc', 'constant']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['cov', 'scale']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['covariances']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['means']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['weights']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['alpha']:
                parameters[key] = np.array(value)
            elif isinstance(value, list) and key in ['low', 'high']:
                parameters[key] = np.array(value)
        
        return create_sampler(sampler_type, dimension, **parameters)
    
    def create_perturbed_sampler(self, ground_truth_sampler: Any, 
                               perturbation_config: Dict) -> Any:
        """
        Create a perturbed sampler using composition.
        
        Args:
            ground_truth_sampler: The base sampler
            perturbation_config: Configuration for the perturbation
            
        Returns:
            Composed sampler
        """
        if perturbation_config['type'] != 'composition':
            raise ValueError("Only composition perturbations are currently supported")
        
        params = perturbation_config['parameters']
        coefficients = np.array(params['coefficients'])
        
        # Create perturbation sampler
        perturbation_sampler = self.create_sampler_from_config({
            'type': params['perturbation_sampler']['type'],
            'dimension': ground_truth_sampler.dimension,
            'parameters': params['perturbation_sampler'].get('parameters', {})
        })
        
        # Create composition sampler
        samplers = [ground_truth_sampler, perturbation_sampler]
        return CompositionSampler(
            dimension=ground_truth_sampler.dimension,
            samplers=samplers,
            coefficients=coefficients
        )
    
    def load_distance_function(self, distance_config: Dict) -> Callable:
        """
        Load a distance function from configuration.
        
        Args:
            distance_config: Dictionary containing distance function configuration
            
        Returns:
            Distance function
        """
        module_name = f"distances.{distance_config['module']}"
        function_name = distance_config['function']
        
        try:
            module = importlib.import_module(module_name)
            distance_func = getattr(module, function_name)
            
            # Create a partial function with fixed parameters
            parameters = distance_config.get('parameters', {})
            if parameters:
                distance_func = partial(distance_func, **parameters)
                
            return distance_func
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load distance function {function_name} from {module_name}: {e}")
            raise
    
    def run_single_trial(self, args: Tuple) -> float:
        """
        Run a single trial of the experiment.
        
        Args:
            args: Tuple containing (sampler1, sampler2, distance_func, M, N)
            
        Returns:
            Distance value
        """
        sampler1, sampler2, distance_func, M, N = args
        
        # Generate samples
        samples1 = sampler1.sample(M)
        samples2 = sampler2.sample(N)
        
        # Compute distance
        distance = distance_func(samples1, samples2)
        
        # Convert to float if necessary
        if hasattr(distance, 'item'):
            distance = distance.item()
        
        return float(distance)
    
    def run_experiment_batch(self, sampler1: Any, sampler2: Any, 
                           distance_func: Callable, n_trials: int) -> np.ndarray:
        """
        Run a batch of experiments with the same setup.
        
        Args:
            sampler1: First sampler
            sampler2: Second sampler  
            distance_func: Distance function
            n_trials: Number of trials to run
            
        Returns:
            Array of distance values
        """
        M = self.config['sample_sizes']['M']
        N = self.config['sample_sizes']['N']
        
        if self.config['advanced']['parallel_processing']:
            # Run in parallel
            n_jobs = self.config['advanced']['n_jobs']
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            
            args_list = [(sampler1, sampler2, distance_func, M, N) for _ in range(n_trials)]
            
            with mp.Pool(n_jobs) as pool:
                distances = list(tqdm(
                    pool.imap(self.run_single_trial, args_list),
                    total=n_trials,
                    desc="Running trials"
                ))
        else:
            # Run sequentially
            distances = []
            for _ in tqdm(range(n_trials), desc="Running trials"):
                distance = self.run_single_trial((sampler1, sampler2, distance_func, M, N))
                distances.append(distance)
        
        return np.array(distances)
    
    def run_null_hypothesis_experiments(self):
        """Run experiments under the null hypothesis (same distribution)."""
        self.logger.info("Running null hypothesis experiments...")
        
        for gt_config in self.config['ground_truth_distributions']:
            gt_name = gt_config['name']
            self.logger.info(f"Testing ground truth distribution: {gt_name}")
            
            # Create ground truth sampler
            gt_sampler = self.create_sampler_from_config(gt_config)
            
            # Test each distance function
            for dist_config in self.config['distances']:
                dist_name = dist_config['name']
                self.logger.info(f"  Using distance function: {dist_name}")
                
                # Load distance function
                distance_func = self.load_distance_function(dist_config)
                
                # Run experiments
                distances = self.run_experiment_batch(
                    gt_sampler, gt_sampler, distance_func,
                    self.config['experiment']['n_trials']
                )
                
                # Store results
                key = f"{gt_name}_{dist_name}"
                self.results['null_hypothesis'][key] = {
                    'distances': distances,
                    'ground_truth': gt_name,
                    'distance_function': dist_name,
                    'n_trials': len(distances)
                }
                
                # Save raw data if requested
                if self.config['experiment']['save_raw_distances']:
                    np.save(
                        self.output_dir / 'raw_data' / f'null_{key}.npy',
                        distances
                    )
    
    def run_alternative_hypothesis_experiments(self):
        """Run experiments under alternative hypotheses (different distributions)."""
        self.logger.info("Running alternative hypothesis experiments...")
        
        for gt_config in self.config['ground_truth_distributions']:
            gt_name = gt_config['name']
            self.logger.info(f"Testing ground truth distribution: {gt_name}")
            
            # Create ground truth sampler
            gt_sampler = self.create_sampler_from_config(gt_config)
            
            for pert_config in self.config['perturbations']:
                pert_name = pert_config['name']
                self.logger.info(f"  Using perturbation: {pert_name}")
                
                # Create perturbed sampler
                perturbed_sampler = self.create_perturbed_sampler(gt_sampler, pert_config)
                
                # Test each distance function
                for dist_config in self.config['distances']:
                    dist_name = dist_config['name']
                    self.logger.info(f"    Using distance function: {dist_name}")
                    
                    # Load distance function
                    distance_func = self.load_distance_function(dist_config)
                    
                    # Run experiments
                    distances = self.run_experiment_batch(
                        gt_sampler, perturbed_sampler, distance_func,
                        self.config['experiment']['n_trials']
                    )
                    
                    # Store results
                    key = f"{gt_name}_{pert_name}_{dist_name}"
                    self.results['alternative_hypothesis'][key] = {
                        'distances': distances,
                        'ground_truth': gt_name,
                        'perturbation': pert_name,
                        'distance_function': dist_name,
                        'n_trials': len(distances)
                    }
                    
                    # Save raw data if requested
                    if self.config['experiment']['save_raw_distances']:
                        np.save(
                            self.output_dir / 'raw_data' / f'alt_{key}.npy',
                            distances
                        )
    
    def compute_statistics(self):
        """Compute summary statistics for all experiments."""
        self.logger.info("Computing summary statistics...")
        
        quantiles = self.config['analysis']['quantiles']
        
        # Process null hypothesis results
        for key, data in self.results['null_hypothesis'].items():
            distances = data['distances']
            
            stats = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'quantiles': {f'q_{int(q*100)}': float(np.quantile(distances, q)) 
                             for q in quantiles}
            }
            
            self.results['statistics'][f'null_{key}'] = stats
        
        # Process alternative hypothesis results
        for key, data in self.results['alternative_hypothesis'].items():
            distances = data['distances']
            
            stats = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'quantiles': {f'q_{int(q*100)}': float(np.quantile(distances, q)) 
                             for q in quantiles}
            }
            
            self.results['statistics'][f'alt_{key}'] = stats
        
        # Compute power analysis if requested
        if self.config['analysis']['power_analysis']:
            self.compute_power_analysis()
    
    def compute_power_analysis(self):
        """Compute power analysis comparing null vs alternative hypotheses."""
        self.logger.info("Computing power analysis...")
        
        thresholds = self.config['analysis']['significance_thresholds']
        
        for gt_name in [gt['name'] for gt in self.config['ground_truth_distributions']]:
            for pert_name in [p['name'] for p in self.config['perturbations']]:
                for dist_name in [d['name'] for d in self.config['distances']]:
                    
                    null_key = f"{gt_name}_{dist_name}"
                    alt_key = f"{gt_name}_{pert_name}_{dist_name}"
                    
                    if (null_key in self.results['null_hypothesis'] and 
                        alt_key in self.results['alternative_hypothesis']):
                        
                        null_distances = self.results['null_hypothesis'][null_key]['distances']
                        alt_distances = self.results['alternative_hypothesis'][alt_key]['distances']
                        
                        power_stats = {}
                        
                        for threshold in thresholds:
                            # Use quantile of null distribution as threshold
                            cutoff = np.quantile(null_distances, 1 - threshold)
                            
                            # Power = P(reject H0 | H1 true) = P(distance > cutoff | H1)
                            power = np.mean(alt_distances > cutoff)
                            power_stats[f'power_alpha_{threshold}'] = float(power)
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(
                                (np.var(null_distances) + np.var(alt_distances)) / 2
                            )
                            effect_size = (np.mean(alt_distances) - np.mean(null_distances)) / pooled_std
                            power_stats[f'effect_size_alpha_{threshold}'] = float(effect_size)
                        
                        self.results['statistics'][f'power_{alt_key}'] = power_stats
    
    def save_results(self):
        """Save all results to files."""
        self.logger.info("Saving results...")
        
        # Save statistics as JSON
        if self.config['experiment']['save_statistics']:
            stats_path = self.output_dir / 'statistics' / 'summary_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(self.results['statistics'], f, indent=2)
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.results['metadata'], f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def run(self):
        """Run the complete discrimination experiment."""
        self.logger.info("Starting discrimination experiments...")
        
        try:
            # Run experiments
            self.run_null_hypothesis_experiments()
            self.run_alternative_hypothesis_experiments()
            
            # Analyse results
            self.compute_statistics()
            
            # Save results
            self.save_results()
            
            # Generate plots if requested
            if self.config['experiment']['generate_plots']:
                self.generate_plots()
            
            self.logger.info("Experiments completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
    
    def generate_plots(self):
        """Generate visualisation plots."""
        self.logger.info("Generating plots...")
        
        try:
            from visualization import DiscriminationPlotter
            plotter = DiscriminationPlotter(self.results, self.config, self.output_dir)
            plotter.generate_all_plots()
        except ImportError:
            self.logger.warning("Visualization module not found. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run discrimination experiments")
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output',
        help='Output directory (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = DiscriminationExperiment(args.config)
    
    # Override output directory if specified
    if args.output:
        experiment.config['experiment']['output_dir'] = args.output
        experiment.setup_output_directory()
    
    # Run experiment
    experiment.run()


if __name__ == "__main__":
    main()
