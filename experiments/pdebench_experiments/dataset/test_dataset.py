#!/usr/bin/env python3
"""
Test script for CFD2DDataset to demonstrate usage and verify functionality.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the CRPS directory to the Python path so we can import experiments
crps_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(crps_dir))

from experiments.pdebench_experiments.dataset.dataset import CFD2DDataset, create_cfd_dataloader, get_example_config


def test_basic_functionality():
    """Test basic dataset functionality."""
    print("=" * 80)
    print("TESTING CFD2D DATASET")
    print("=" * 80)
    
    # Configuration for testing
    config = {
        'variables': ['Vx', 'Vy', 'density', 'pressure'],
        'input_variables': ['Vx', 'Vy'],  # Use only velocity components as input
        'target_variable': 'pressure',   # Predict pressure
        'time_lag': 3,                   # Use 3 previous timesteps
        'time_predict': 2,               # Predict 2 future timesteps  
        'input_spatial': 100,            # Use 100 random spatial points as input
        'target_spatial': 50,            # Predict at 50 spatial points
        'split_ratios': {'train': 0.7, 'val': 0.2, 'test': 0.1}
    }
    
    # Path to the CFD data
    data_path = Path("experiments/pdebench_experiments/data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
    
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("   Please ensure the CFD data is available at this location.")
        return False
    
    try:
        # Test dataset creation for different splits
        for split in ['train', 'val', 'test']:
            print(f"\nTesting {split} split:")
            dataset = CFD2DDataset(data_path, config, split=split, normalise=True)
            
            # Get dataset info
            info = dataset.get_data_info()
            print(f"   Dataset info: {info}")
            
            # Test getting a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample 0 shapes:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: {value.shape}")
                    else:
                        print(f"     {key}: {value}")
            else:
                print(f"No samples in {split} split")
        
        print(f"\nBasic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False


def test_dataloader():
    """Test DataLoader functionality."""
    print(f"\nTesting DataLoader functionality:")
    
    config = get_example_config()
    # Use smaller spatial sampling for faster testing
    config['input_spatial'] = 64
    config['target_spatial'] = 16
    config['time_lag'] = 2
    config['time_predict'] = 1
    
    data_path = Path("experiments/pdebench_experiments/data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
    
    if not data_path.exists():
        print(f"Data file not found, skipping DataLoader test")
        return False
    
    try:
        # Create dataloaders for all splits
        train_loader = create_cfd_dataloader(data_path, config, split='train', batch_size=4, shuffle=True)
        val_loader = create_cfd_dataloader(data_path, config, split='val', batch_size=2, shuffle=False)
        
        print(f"   Train loader: {len(train_loader)} batches")
        print(f"   Val loader: {len(val_loader)} batches")
        
        # Test loading a batch
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i} shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape}")
            
            # Only test first batch to save time
            if i >= 0:
                break
        
        print(f"DataLoader test passed!")
        return True
        
    except Exception as e:
        print(f"DataLoader test failed: {e}")
        return False


def test_configuration_variations():
    """Test different configuration options."""
    print(f"\nTesting configuration variations:")
    
    data_path = Path("experiments/pdebench_experiments/data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
    
    if not data_path.exists():
        print(f"Data file not found, skipping configuration tests")
        return False
    
    test_configs = [
        {
            'name': 'All variables as input',
            'config': {
                'variables': ['Vx', 'Vy', 'density', 'pressure'],
                'input_variables': ['Vx', 'Vy', 'density', 'pressure'],
                'target_variable': 'Vx',
                'time_lag': 1,
                'time_predict': 1,
                'input_spatial': 10,
                'target_spatial': 5,
                'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
            }
        },
        {
            'name': 'Single variable input',
            'config': {
                'variables': ['Vx', 'Vy', 'density', 'pressure'],
                'input_variables': ['density'],
                'target_variable': 'pressure',
                'time_lag': 2,
                'time_predict': 3,
                'input_spatial': -1,  # All spatial points
                'target_spatial': 1,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
            }
        },
        {
            'name': 'Long time series',
            'config': {
                'variables': ['Vx', 'Vy', 'pressure'],
                'input_variables': ['Vx', 'Vy'],
                'target_variable': 'pressure',
                'time_lag': 5,
                'time_predict': 5,
                'input_spatial': 50,
                'target_spatial': 25,
                'split_ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2}
            }
        }
    ]
    
    for test_case in test_configs:
        try:
            print(f"   Testing: {test_case['name']}")
            dataset = CFD2DDataset(data_path, test_case['config'], split='train')
            
            if len(dataset) > 0:
                sample = dataset[0]
                input_shape = sample['input_data'].shape
                target_shape = sample['target_data'].shape
                print(f"     Input shape: {input_shape}, Target shape: {target_shape}")
            
            print(f"{test_case['name']} passed")
            
        except Exception as e:
            print(f"{test_case['name']} failed: {e}")
            return False
    
    print(f"All configuration variations passed!")
    return True


def test_normalisation():
    """Test data normalisation functionality."""
    print(f"\nTesting normalisation functionality:")
    
    data_path = Path("experiments/pdebench_experiments/data/2D/2D_CFD/2D_CFD_rand_Eta1e-08_Zeta1e-08_M0.1_periodic_Train.hdf5")
    
    if not data_path.exists():
        print(f"Data file not found, skipping normalisation tests")
        return False
    
    config = {
        'variables': ['Vx', 'Vy', 'pressure'],
        'input_variables': ['Vx', 'Vy'],
        'target_variable': 'pressure',
        'time_lag': 2,
        'time_predict': 1,
        'input_spatial': 20,
        'target_spatial': 10,
        'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1}
    }
    
    try:
        # Test with normalisation
        dataset_norm = CFD2DDataset(data_path, config, split='train', normalise=True)
        stats = dataset_norm.get_normalisation_stats()
        
        print(f"   Normalisation stats computed for {len(stats)} variables")
        for var, var_stats in stats.items():
            mean_shape = var_stats['mean'].shape
            std_shape = var_stats['std'].shape
            print(f"     {var}: mean shape {mean_shape}, std shape {std_shape}")
        
        # Test denormalisation
        sample = dataset_norm[0]
        target_data = sample['target_data']
        denorm_data = dataset_norm.denormalise_prediction(target_data, 'pressure')
        
        print(f"     Original target shape: {target_data.shape}")
        print(f"     Denormalised shape: {denorm_data.shape}")
        
        # Test without normalisation
        dataset_no_norm = CFD2DDataset(data_path, config, split='train', normalise=False)
        sample_no_norm = dataset_no_norm[0]
        
        print(f"Normalisation test passed!")
        return True
        
    except Exception as e:
        print(f"Normalisation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("TESTING CFD2D DATASET IMPLEMENTATION")
    print("=" * 80)
    
    tests = [
        test_basic_functionality,
        test_dataloader,
        test_configuration_variations,
        test_normalisation
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print(f"=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [func.__name__ for func in tests]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The CFD2DDataset implementation is working correctly.")
    else:
        print("Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 