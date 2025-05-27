# CRPS - Continuous Ranked Probability Score

This repository contains experiments and models for probabilistic regression using the Continuous Ranked Probability Score (CRPS) as a loss function. CRPS is a proper scoring rule that measures the compatibility of a probability forecast with an observation.

## Models

- `models/mlp_gaussian.py`: Implements a simple MLP that predicts Gaussian distributions with mean and variance outputs.
- `models/mlp_crps_sampler.py`: Implements a sample-based probabilistic MLP using the EpsilonSampler from torchnaut to generate samples directly.

## Experiments

### 1D Regression

- `experiments/1d_regression/crps_samples_example.py`: Demonstrates training a sample-based MLP using the CRPS loss on a synthetic 1D regression problem with non-Gaussian noise.
- `experiments/1d_regression/gaussian_not_well_specified.py`: Experiments with Gaussian models on datasets where the noise is not Gaussian.

### 2D Regression

- `experiments/2d_regression/gaussian_samples.py`: Comprehensive multi-dimensional regression experiments with sample-based CRPS training. This script runs multiple experiments with different random seeds to assess model stability and performance variability.

#### Multi-Run Experimental Structure

The 2D regression experiments automatically run multiple trials with different random seeds to provide robust performance estimates:

**Features:**
- Multiple experiments with configurable random seeds
- Separate output directories for each run
- Complete YAML configuration files storing all parameters and results
- Automatic generation of training history and prediction visualisation plots
- Summary statistics across all runs

**Output Structure:**
```
experiments/2d_regression/
├── figures/
│   ├── run_001_seed_1/
│   │   ├── config.yaml                    # Complete run configuration and results
│   │   ├── training_history.png           # Training/validation loss curves
│   │   ├── prediction_samples.png         # Model predictions vs true values
│   │   └── prediction_samples_more.png    # Extended prediction visualisation
│   ├── run_002_seed_42/
│   │   └── ... (same structure)
│   ├── experiment_summary.yaml            # Summary of all runs
│   └── ... (additional runs)
├── saved_models/
│   ├── run_001_seed_1/
│   │   └── mlp_sampler.pt                 # Trained model weights
│   └── ... (models for each run)
└── gaussian_samples.py
```

**Configuration Files:**
Each `config.yaml` contains:
- Model parameters (hidden_size, latent_dim, n_layers, etc.)
- Training parameters (n_epochs, patience, batch_size, etc.)
- Data generation parameters
- Random seed and run metadata
- Complete results (MSE, CRPS, calibration metrics)

## Installation

The code depends on PyTorch, torchnaut, and additional packages for configuration management. You can install the dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn torchnaut pyyaml
```

## Running the experiments

```bash
# Run the 1D CRPS sampler example
python experiments/1d_regression/crps_samples_example.py

# Run the 2D multi-seed regression experiments
python experiments/2d_regression/gaussian_samples.py
```

The experiments will generate figures in their respective `figures` directories and provide comprehensive performance statistics.

## Key Features

- Direct sample-based training with CRPS
- Implementations for both parametric (Gaussian) and non-parametric (sample-based) models
- Visualisation of prediction intervals and confidence regions
- Multi-dimensional regression capabilities
- Automated multi-run experimentation with statistical analysis
- Complete experiment reproducibility through YAML configuration files
- Performance variability assessment across different random seeds

## Experimental Results

The multi-run structure allows for robust performance evaluation:
- Mean and standard deviation of key metrics across runs
- Calibration analysis for uncertainty quantification
- Training stability assessment
- Model performance variability analysis

## Project Structure
```
CRPS/
├── models/                     # Model implementations
├── experiments/
│   ├── 1d_regression/         # Single-dimensional experiments
│   ├── 2d_regression/         # Multi-dimensional experiments
│   ├── data/                  # Data generation utilities
│   ├── training/              # Training functions and utilities
│   └── visualisation/         # Plotting and analysis tools
└── README.md
```

## Getting Started

1. Install the required dependencies (see Installation section)
2. Run the 1D experiments to familiarise yourself with the basic approach
3. Execute the 2D experiments for comprehensive multi-run analysis
4. Examine the generated YAML configuration files to understand experiment parameters
5. Review the figures directory for visualisation results

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Torchnaut
- PyYAML

## Usage

The experiments are designed to be self-contained and can be run directly. Each experiment generates comprehensive output including:
- Model weights
- Training visualisations
- Performance metrics
- Configuration files for reproducibility

Modify the `base_config` dictionary in the experiment scripts to adjust model architecture, training parameters, or experimental setup.

## Contributing

When adding new experiments, please follow the established structure:
- Use YAML configuration files for parameter storage
- Include comprehensive visualisation
- Implement multi-run capabilities for robust evaluation
- Document experiment purpose and methodology

## License

[Add license information]

## Contact

[Add contact information] 