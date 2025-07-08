import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.data.generate_time_series import generate_multivariate_ar


class MultivariatARDataset(Dataset):
    """
    PyTorch Dataset for multivariate autoregressive (AR) time series data.
    
    This dataset generates synthetic multivariate AR data and provides sliding window
    access for time series prediction tasks.
    
    Supports:
    - Configurable input time steps (l) and output time steps (k)
    - Multiple time series generation
    - Data normalisation
    - Configurable AR parameters
    - Train/validation/test splits
    - Storage of ground truth generation parameters
    """
    
    def __init__(self, 
                 config: Dict,
                 split: str = 'train',
                 normalise: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialise multivariate AR dataset.
        
        Args:
            config: Configuration dictionary containing:
                # Data generation parameters
                - n_timesteps: Total number of timesteps to generate
                - dimension: Dimension of the multivariate time series
                - ar_order: Order of the AR process (default: 1)
                - noise_scale: Scaling factor for noise (default: 1.0)
                - n_series: Number of independent time series to generate (default: 100)
                
                # Dataset parameters  
                - input_timesteps: Number of input timesteps (l) for prediction
                - output_timesteps: Number of output timesteps (k) to predict
                - split_ratios: Dict with 'train', 'val', 'test' ratios (default: 0.7/0.15/0.15)
                
                # Optional AR parameters
                - A_matrices: Custom AR coefficient matrices (optional)
                - noise_cov: Custom noise covariance matrix (optional)
                
            split: Which data split to use ('train', 'val', 'test')
            normalise: Whether to normalise the data
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.split = split
        self.normalise = normalise
        self.random_state = random_state
        
        # Extract configuration parameters with defaults
        self.n_timesteps = config['n_timesteps']
        self.dimension = config['dimension']
        self.ar_order = config.get('ar_order', 1)
        self.noise_scale = config.get('noise_scale', 1.0)
        self.n_series = config.get('n_series', 100)
        
        self.input_timesteps = config['input_timesteps']  # l
        self.output_timesteps = config['output_timesteps']  # k
        self.split_ratios = config.get('split_ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        
        # Optional parameters
        self.A_matrices = config.get('A_matrices', None)
        self.noise_cov = config.get('noise_cov', None)
        
        # Validate configuration
        self._validate_config()
        
        # Generate data
        self._generate_data()
        self._setup_normalisation()
        self._create_data_splits()
        self._create_sliding_windows()
        
        print(f"MultivariatARDataset initialised:")
        print(f"  Split: {split} ({len(self)} samples)")
        print(f"  Time series: {self.n_series} series, {self.n_timesteps} timesteps, {self.dimension}D")
        print(f"  Window size: {self.input_timesteps} input → {self.output_timesteps} output")
        print(f"  AR order: {self.ar_order}, Noise scale: {self.noise_scale}")
        print(f"  Max eigenvalue: {self.generation_metadata['max_eigenvalue']:.4f}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check required parameters
        required_params = ['n_timesteps', 'dimension', 'input_timesteps', 'output_timesteps']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not in config")
        
        # Check parameter values
        if self.n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1")
        if self.dimension < 1:
            raise ValueError("dimension must be >= 1")
        if self.ar_order < 1:
            raise ValueError("ar_order must be >= 1")
        if self.input_timesteps < 1:
            raise ValueError("input_timesteps must be >= 1")
        if self.output_timesteps < 1:
            raise ValueError("output_timesteps must be >= 1")
        if self.n_series < 1:
            raise ValueError("n_series must be >= 1")
        
        # Check if we have enough timesteps for sliding windows
        min_required_timesteps = self.input_timesteps + self.output_timesteps
        if self.n_timesteps < min_required_timesteps:
            raise ValueError(f"n_timesteps ({self.n_timesteps}) must be >= input_timesteps + output_timesteps "
                           f"({min_required_timesteps})")
        
        # Check split ratios
        total_ratio = sum(self.split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate optional matrix parameters
        if self.A_matrices is not None:
            if isinstance(self.A_matrices, np.ndarray):
                if self.A_matrices.shape != (self.dimension, self.dimension):
                    raise ValueError(f"A_matrices must have shape ({self.dimension}, {self.dimension})")
            elif isinstance(self.A_matrices, list):
                if len(self.A_matrices) != self.ar_order:
                    raise ValueError(f"A_matrices list must have length {self.ar_order}")
                for i, A in enumerate(self.A_matrices):
                    if A.shape != (self.dimension, self.dimension):
                        raise ValueError(f"A_matrices[{i}] must have shape ({self.dimension}, {self.dimension})")
        
        if self.noise_cov is not None:
            if self.noise_cov.shape != (self.dimension, self.dimension):
                raise ValueError(f"noise_cov must have shape ({self.dimension}, {self.dimension})")
    
    def _generate_data(self):
        """Generate multivariate AR time series data."""
        print(f"Generating {self.n_series} multivariate AR({self.ar_order}) time series...")
        
        self.data, self.generation_metadata = generate_multivariate_ar(
            n_timesteps=self.n_timesteps,
            dimension=self.dimension,
            ar_order=self.ar_order,
            A_matrices=self.A_matrices,
            noise_cov=self.noise_cov,
            noise_scale=self.noise_scale,
            random_state=self.random_state,
            n_series=self.n_series
        )
        
        # Store generation parameters for reproducibility
        self.ground_truth_params = {
            'generation_metadata': self.generation_metadata,
            'config_used': self.config.copy(),
            'random_state': self.random_state
        }
        
        print(f"Data generated: shape {self.data.shape}")
        if self.generation_metadata.get('stability_ensured', False):
            print(f"  Stability adjustment applied (max eigenvalue: {self.generation_metadata['max_eigenvalue']:.4f})")
    
    def _setup_normalisation(self):
        """Setup normalisation statistics."""
        if not self.normalise:
            self.norm_stats = None
            return
        
        print("Computing normalisation statistics...")
        
        # Compute global normalisation statistics across all series and timesteps
        # data shape: (n_series, n_timesteps, dimension)
        self.norm_stats = {
            'mean': np.mean(self.data, axis=(0, 1)),  # Shape: (dimension,)
            'std': np.std(self.data, axis=(0, 1)) + 1e-8  # Shape: (dimension,)
        }
    
    def _normalise_data(self, data: np.ndarray) -> np.ndarray:
        """Normalise data using computed statistics."""
        if not self.normalise or self.norm_stats is None:
            return data
        return (data - self.norm_stats['mean']) / self.norm_stats['std']
    
    def _denormalise_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalise data using computed statistics."""
        if not self.normalise or self.norm_stats is None:
            return data
        return data * self.norm_stats['std'] + self.norm_stats['mean']
    
    def _create_data_splits(self):
        """Create train/val/test splits."""
        # Split at the series level to ensure temporal coherence within series
        n_train = int(self.n_series * self.split_ratios['train'])
        n_val = int(self.n_series * self.split_ratios['val'])
        n_test = self.n_series - n_train - n_val
        
        # Create split indices
        series_indices = np.arange(self.n_series)
        if self.random_state is not None:
            # Use a different seed for splits to avoid interference with data generation
            np.random.seed(self.random_state + 12345)
            np.random.shuffle(series_indices)
        
        if self.split == 'train':
            self.series_indices = series_indices[:n_train]
        elif self.split == 'val':
            self.series_indices = series_indices[n_train:n_train + n_val]
        elif self.split == 'test':
            self.series_indices = series_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        print(f"Split '{self.split}': using {len(self.series_indices)} series "
              f"(indices {self.series_indices[0]}-{self.series_indices[-1]})")
    
    def _create_sliding_windows(self):
        """Create sliding window indices for each series in the split."""
        # For each series, create all possible sliding windows
        # Window starts at t and includes: input [t:t+l], output [t+l:t+l+k]
        
        max_start_time = self.n_timesteps - self.input_timesteps - self.output_timesteps
        self.window_starts = np.arange(max_start_time + 1)  # +1 because range is exclusive
        
        # Total number of samples = num_series_in_split * num_windows_per_series
        self.n_windows_per_series = len(self.window_starts)
        self.total_samples = len(self.series_indices) * self.n_windows_per_series
        
        print(f"Created {self.n_windows_per_series} sliding windows per series "
              f"(total: {self.total_samples} samples)")
    
    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'input': Input time series of shape (input_timesteps, dimension)
            - 'target': Target time series of shape (output_timesteps, dimension)
            - 'series_idx': Which series this sample came from
            - 'time_start': Starting time index for this window
        """
        # Convert flat index to (series_idx_in_split, window_idx)
        series_idx_in_split = idx // self.n_windows_per_series
        window_idx = idx % self.n_windows_per_series
        
        # Get the actual series index in the full dataset
        actual_series_idx = self.series_indices[series_idx_in_split]
        
        # Get the time window
        time_start = self.window_starts[window_idx]
        time_input_end = time_start + self.input_timesteps
        time_target_end = time_input_end + self.output_timesteps
        
        # Extract the data
        series_data = self.data[actual_series_idx]  # Shape: (n_timesteps, dimension)
        
        input_data = series_data[time_start:time_input_end]  # Shape: (input_timesteps, dimension)
        target_data = series_data[time_input_end:time_target_end]  # Shape: (output_timesteps, dimension)
        
        # Apply normalisation
        input_data = self._normalise_data(input_data)
        target_data = self._normalise_data(target_data)
        
        # Convert to tensors
        return {
            'input': torch.tensor(input_data, dtype=torch.float32),
            'target': torch.tensor(target_data, dtype=torch.float32),
            'series_idx': torch.tensor(actual_series_idx, dtype=torch.long),
            'time_start': torch.tensor(time_start, dtype=torch.long)
        }
    
    def get_normalisation_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get normalisation statistics."""
        return self.norm_stats
    
    def denormalise_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalise a tensor using the computed statistics."""
        if not self.normalise or self.norm_stats is None:
            return tensor
        
        mean = torch.tensor(self.norm_stats['mean'], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(self.norm_stats['std'], dtype=tensor.dtype, device=tensor.device)
        
        return tensor * std + mean
    
    def get_ground_truth_params(self) -> Dict[str, Any]:
        """Get the ground truth generation parameters."""
        return self.ground_truth_params
    
    def get_data_info(self) -> Dict[str, Union[int, Tuple, float]]:
        """Get information about the dataset."""
        return {
            'n_series_total': self.n_series,
            'n_series_in_split': len(self.series_indices),
            'n_timesteps': self.n_timesteps,
            'dimension': self.dimension,
            'ar_order': self.ar_order,
            'input_timesteps': self.input_timesteps,
            'output_timesteps': self.output_timesteps,
            'total_samples': self.total_samples,
            'data_shape': self.data.shape,
            'split': self.split,
            'normalised': self.normalise,
            'max_eigenvalue': self.generation_metadata.get('max_eigenvalue', None),
            'noise_scale': self.noise_scale
        }
    
    def get_full_series(self, series_idx: int, normalised: bool = True) -> np.ndarray:
        """
        Get a full time series by index.
        
        Args:
            series_idx: Index of the series to retrieve
            normalised: Whether to return normalised data
            
        Returns:
            Time series of shape (n_timesteps, dimension)
        """
        if series_idx < 0 or series_idx >= self.n_series:
            raise ValueError(f"series_idx must be in [0, {self.n_series})")
        
        series_data = self.data[series_idx].copy()
        
        if normalised:
            series_data = self._normalise_data(series_data)
        
        return series_data


def create_ar_dataloader(config: Dict,
                        split: str = 'train',
                        batch_size: int = 32,
                        shuffle: bool = True,
                        num_workers: int = 4,
                        normalise: bool = True,
                        random_state: Optional[int] = None) -> DataLoader:
    """
    Create a DataLoader for multivariate AR dataset.
    
    Args:
        config: Dataset configuration dictionary
        split: Data split to use ('train', 'val', 'test')
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data (typically True for train, False for val/test)
        num_workers: Number of worker processes for data loading
        normalise: Whether to normalise the data
        random_state: Random seed for reproducibility
        
    Returns:
        PyTorch DataLoader
    """
    dataset = MultivariatARDataset(
        config=config,
        split=split,
        normalise=normalise,
        random_state=random_state
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


def get_example_config() -> Dict:
    """
    Get an example configuration for the AR dataset.
    
    Returns:
        Example configuration dictionary
    """
    return {
        # Data generation parameters
        'n_timesteps': 200,
        'dimension': 5,
        'ar_order': 2,
        'noise_scale': 0.1,
        'n_series': 500,
        
        # Dataset parameters
        'input_timesteps': 10,  # l
        'output_timesteps': 5,  # k
        'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
        
        # Optional: custom AR matrices (if not provided, random stable ones are generated)
        # 'A_matrices': None,
        # 'noise_cov': None,
    }

