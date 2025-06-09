import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


class CFD2DDataset(Dataset):
    """
    PyTorch Dataset for 2D CFD data from PDEBench.
    
    Supports:
    - Variable downsampling (removing subset of input variables)
    - Target variable selection
    - Temporal prediction with configurable lag and forecast horizon
    - Spatial sampling (random spatial points for input/target)
    
    Data structure expected:
    - HDF5 file with fields: 'Vx', 'Vy', 'density', 'pressure'
    - Each field shape: (n_samples, n_timesteps, height, width)
    - Coordinate fields: 'x-coordinate', 'y-coordinate', 't-coordinate'
    """
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 config: Dict,
                 split: str = 'train',
                 normalise: bool = True):
        """
        Initialise CFD dataset.
        
        Args:
            data_path: Path to HDF5 file containing CFD data
            config: Training configuration dictionary containing:
                - variables: List of variable names to use ['Vx', 'Vy', 'density', 'pressure']
                - input_variables: List of variables to use as inputs (subset of variables)
                - target_variable: Name of target variable to predict
                - time_lag: Number of input timesteps (l)
                - time_predict: Number of prediction timesteps (m)
                - input_spatial: Number of spatial points to use as input (-1 for all)
                - target_spatial: Number of spatial points to predict (default: 1)
                - split_ratios: Dict with 'train', 'val', 'test' ratios
            split: Which data split to use ('train', 'val', 'test')
            normalise: Whether to normalise the data
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.normalise = normalise
        
        # Set default values for config parameters
        self.variables = config.get('variables', ['Vx', 'Vy', 'density', 'pressure'])
        self.input_variables = config.get('input_variables', self.variables)
        self.target_variable = config.get('target_variable', 'pressure')
        self.time_lag = config.get('time_lag', 5)
        self.time_predict = config.get('time_predict', 1)
        self.input_spatial = config.get('input_spatial', -1)  # -1 means all spatial points
        self.target_spatial = config.get('target_spatial', 1)
        self.split_ratios = config.get('split_ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        
        # Validate configuration
        self._validate_config()
        
        # Load and prepare data
        self._load_data()
        self._setup_normalisation()
        self._create_data_splits()
        
        print(f"CFD2DDataset initialised:")
        print(f"  Split: {split} ({len(self)} samples)")
        print(f"  Input variables: {self.input_variables}")
        print(f"  Target variable: {self.target_variable}")
        print(f"  Time lag: {self.time_lag}, Time predict: {self.time_predict}")
        print(f"  Spatial sampling - Input: {self.input_spatial}, Target: {self.target_spatial}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check variables
        if self.target_variable not in self.variables:
            raise ValueError(f"Target variable '{self.target_variable}' not in variables list")
        
        for var in self.input_variables:
            if var not in self.variables:
                raise ValueError(f"Input variable '{var}' not in variables list")
        
        # Check temporal parameters
        if self.time_lag < 1:
            raise ValueError("time_lag must be >= 1")
        if self.time_predict < 1:
            raise ValueError("time_predict must be >= 1")
        
        # Check spatial parameters
        if self.target_spatial < 1:
            raise ValueError("target_spatial must be >= 1")
        
        # Check split ratios
        total_ratio = sum(self.split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    def _load_data(self):
        """Load data from HDF5 file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load coordinate information
            self.x_coords = f['x-coordinate'][()]
            self.y_coords = f['y-coordinate'][()]
            self.t_coords = f['t-coordinate'][()]
            
            self.spatial_shape = (len(self.y_coords), len(self.x_coords))  # (height, width)
            self.n_timesteps = len(self.t_coords) - 1  # Available timesteps for prediction
            
            # Load variable data
            self.data = {}
            sample_shape = None
            
            for var in self.variables:
                if var not in f:
                    raise KeyError(f"Variable '{var}' not found in dataset")
                
                var_data = f[var][()]  # Shape: (n_samples, n_timesteps, height, width)
                self.data[var] = var_data
                
                if sample_shape is None:
                    sample_shape = var_data.shape
                elif var_data.shape != sample_shape:
                    raise ValueError(f"Variable '{var}' has shape {var_data.shape}, "
                                   f"expected {sample_shape}")
            
            self.n_samples, self.data_n_timesteps, self.height, self.width = sample_shape
            
            # Validate temporal requirements
            min_required_timesteps = self.time_lag + self.time_predict
            if self.data_n_timesteps < min_required_timesteps:
                raise ValueError(f"Dataset has {self.data_n_timesteps} timesteps, "
                               f"but need at least {min_required_timesteps} "
                               f"(time_lag={self.time_lag} + time_predict={self.time_predict})")
        
        print(f"Data loaded: {self.n_samples} samples, {self.data_n_timesteps} timesteps, "
              f"{self.height}x{self.width} spatial grid")
    
    def _setup_normalisation(self):
        """Setup normalisation statistics."""
        if not self.normalise:
            self.norm_stats = {}
            return
        
        print("Computing normalisation statistics...")
        self.norm_stats = {}
        
        for var in self.variables:
            data = self.data[var]
            # Compute global stats over all samples, timesteps, and spatial locations
            self.norm_stats[var] = {
                'mean': np.mean(data),  # Scalar global mean
                'std': np.std(data) + 1e-8  # Scalar global std with small epsilon
            }
    
    def _create_data_splits(self):
        """Create train/val/test splits."""
        # Create sample indices for this split
        n_train = int(self.n_samples * self.split_ratios['train'])
        n_val = int(self.n_samples * self.split_ratios['val'])
        n_test = self.n_samples - n_train - n_val
        
        # Use fixed random seed for reproducible splits
        np.random.seed(42)
        all_indices = np.random.permutation(self.n_samples)
        
        if self.split == 'train':
            self.sample_indices = all_indices[:n_train]
        elif self.split == 'val':
            self.sample_indices = all_indices[n_train:n_train + n_val]
        elif self.split == 'test':
            self.sample_indices = all_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # For each sample, create valid temporal indices
        self.temporal_indices = []
        for sample_idx in self.sample_indices:
            # Valid starting points for temporal windows
            max_start = self.data_n_timesteps - self.time_lag - self.time_predict + 1
            for t_start in range(max_start):
                self.temporal_indices.append((sample_idx, t_start))
        
        print(f"Split '{self.split}': {len(self.sample_indices)} samples, "
              f"{len(self.temporal_indices)} temporal windows")
    
    def _normalise_data(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Normalise data using pre-computed statistics."""
        if not self.normalise or variable not in self.norm_stats:
            return data
        
        stats = self.norm_stats[variable]
        return (data - stats['mean']) / stats['std']
    
    def _denormalise_data(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Denormalise data using pre-computed statistics."""
        if not self.normalise or variable not in self.norm_stats:
            return data
        
        stats = self.norm_stats[variable]
        return data * stats['std'] + stats['mean']
    
    def _sample_spatial_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample random spatial points and return their coordinates."""
        if n_points == -1 or n_points >= self.height * self.width:
            # Use all spatial points
            y_idx, x_idx = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
            y_idx = y_idx.flatten()
            x_idx = x_idx.flatten()
        else:
            # Sample random points
            total_points = self.height * self.width
            selected_indices = np.random.choice(total_points, size=n_points, replace=False)
            y_idx = selected_indices // self.width
            x_idx = selected_indices % self.width
        
        # Convert to actual coordinates
        y_coords = self.y_coords[y_idx]
        x_coords = self.x_coords[x_idx]
        
        return np.stack([x_coords, y_coords], axis=-1), (y_idx, x_idx)  # Shape: (n_points, 2)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.temporal_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'input_data': Input variable data, shape (time_lag, n_input_spatial, n_input_vars)
            - 'input_coords': Input spatial coordinates, shape (n_input_spatial, 2)
            - 'target_data': Target variable data, shape (time_predict, n_target_spatial)
            - 'target_coords': Target spatial coordinates, shape (n_target_spatial, 2)
            - 'sample_idx': Original sample index
            - 'time_start': Starting time index
        """
        sample_idx, time_start = self.temporal_indices[idx]
        
        # Extract temporal windows
        input_time_end = time_start + self.time_lag
        target_time_start = input_time_end
        target_time_end = target_time_start + self.time_predict
        
        # Sample spatial points
        input_coords, input_spatial_idx = self._sample_spatial_points(self.input_spatial)
        target_coords, target_spatial_idx = self._sample_spatial_points(self.target_spatial)
        
        # Extract input data for all input variables
        input_data_list = []
        for var in self.input_variables:
            var_data = self.data[var][sample_idx, time_start:input_time_end]  # (time_lag, height, width)
            var_data = self._normalise_data(var_data, var)
            
            # Ensure correct shape - sometimes slicing can add extra dimensions
            if len(var_data.shape) == 4 and var_data.shape[0] == 1:
                var_data = var_data[0]  # Remove extra dimension: (1, time_lag, height, width) -> (time_lag, height, width)
            
            # Sample spatial points
            var_data_sampled = var_data[:, input_spatial_idx[0], input_spatial_idx[1]]  # (time_lag, n_input_spatial)
            input_data_list.append(var_data_sampled)
        
        # Stack input variables: (time_lag, n_input_spatial, n_input_vars)
        input_data = np.stack(input_data_list, axis=-1)
        
        # Extract target data
        target_data = self.data[self.target_variable][sample_idx, target_time_start:target_time_end]  # (time_predict, height, width)
        target_data = self._normalise_data(target_data, self.target_variable)
        
        # Ensure correct shape - sometimes slicing can add extra dimensions
        if len(target_data.shape) == 4 and target_data.shape[0] == 1:
            target_data = target_data[0]  # Remove extra dimension: (1, time_predict, height, width) -> (time_predict, height, width)
        
        # Sample spatial points for target
        target_data_sampled = target_data[:, target_spatial_idx[0], target_spatial_idx[1]]  # (time_predict, n_target_spatial)
        
        return {
            'input_data': torch.FloatTensor(input_data),
            'input_coords': torch.FloatTensor(input_coords),
            'target_data': torch.FloatTensor(target_data_sampled),
            'target_coords': torch.FloatTensor(target_coords),
            'sample_idx': sample_idx,
            'time_start': time_start
        }
    
    def get_normalisation_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return normalisation statistics for all variables."""
        return self.norm_stats
    
    def denormalise_prediction(self, prediction: torch.Tensor, variable: str) -> torch.Tensor:
        """Denormalise a prediction tensor."""
        if not self.normalise or variable not in self.norm_stats:
            return prediction
        
        stats = self.norm_stats[variable]
        # Convert scalar stats to tensors on the same device as prediction
        mean = torch.tensor(stats['mean'], dtype=prediction.dtype, device=prediction.device)
        std = torch.tensor(stats['std'], dtype=prediction.dtype, device=prediction.device)
        
        return prediction * std + mean
    
    def get_variable_names(self) -> Dict[str, List[str]]:
        """Return variable names for inputs and target."""
        return {
            'input_variables': self.input_variables,
            'target_variable': self.target_variable,
            'all_variables': self.variables
        }
    
    def get_data_info(self) -> Dict[str, Union[int, Tuple[int, int]]]:
        """Return dataset information."""
        return {
            'n_samples': len(self.sample_indices),
            'n_temporal_windows': len(self),
            'spatial_shape': self.spatial_shape,
            'n_timesteps': self.data_n_timesteps,
            'time_lag': self.time_lag,
            'time_predict': self.time_predict,
            'n_input_variables': len(self.input_variables),
            'input_spatial_points': len(self.x_coords) * len(self.y_coords) if self.input_spatial == -1 else self.input_spatial,
            'target_spatial_points': self.target_spatial
        }


def create_cfd_dataloader(data_path: Union[str, Path], 
                         config: Dict, 
                         split: str = 'train',
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for CFD dataset.
    
    Args:
        data_path: Path to HDF5 data file
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = CFD2DDataset(data_path, config, split)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Example configuration
def get_example_config() -> Dict:
    """Return an example configuration for the CFD dataset."""
    return {
        'variables': ['Vx', 'Vy', 'density', 'pressure'],
        'input_variables': ['Vx', 'Vy', 'density'],  # Exclude pressure from inputs
        'target_variable': 'pressure',  # Predict pressure
        'time_lag': 5,  # Use 5 previous timesteps as input
        'time_predict': 3,  # Predict 3 future timesteps
        'input_spatial': 1024,  # Use 1024 random spatial points as input
        'target_spatial': 256,  # Predict at 256 spatial points
        'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
    }


if __name__ == "__main__":
    # Example usage
    data_path = "experiments/pdebench_experiments/data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5"
    config = get_example_config()
    
    # Create dataset
    dataset = CFD2DDataset(data_path, config, split='train')
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Create dataloader
    dataloader = create_cfd_dataloader(data_path, config, batch_size=4)
    
    # Test batch loading
    for batch in dataloader:
        print(f"Batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        break
