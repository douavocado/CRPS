"""
Dataset module for PDEBench CFD data.

This module provides PyTorch datasets for loading and preprocessing
2D CFD simulation data for CRPS-based probabilistic training.
"""

from .dataset import CFD2DDataset, create_cfd_dataloader, get_example_config

__all__ = ['CFD2DDataset', 'create_cfd_dataloader', 'get_example_config'] 