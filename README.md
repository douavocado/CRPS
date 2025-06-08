# CRPS - Continuous Ranked Probability Score

This repository contains experiments and models for probabilistic regression using the Continuous Ranked Probability Score (CRPS) as a loss function. CRPS is a proper scoring rule that measures the compatibility of a probability forecast with an observation.

## Models

- `models/mlp_gaussian.py`: Implements a simple MLP that predicts Gaussian distributions with mean and variance outputs.
- `models/mlp_crps_sampler.py`: Implements a sample-based probabilistic MLP using the EpsilonSampler from torchnaut to generate samples directly.
- `models/affine_normal.py`: Implements a simple affine transformation of independent normal random variables (y = Ax + b where x ~ N(0,I)), representing pure distributional learning without input dependence.

## Common Utilities

- `common/losses.py`: Contains general loss functions including CRPS loss computation and energy score loss for multivariate samples.

## Experiments

### 1D Regression

- `experiments/1d_regression/crps_samples_example.py`: Demonstrates training a sample-based MLP using the CRPS loss on a synthetic 1D regression problem with non-Gaussian noise.
- `experiments/1d_regression/gaussian_not_well_specified.py`: Experiments with Gaussian models on datasets where the noise is not Gaussian.
- `experiments/1d_regression/gaussian_well_specified.py`: Experiments comparing different models on datasets with Gaussian noise and various mean functions (sinusoidal, linear, sawtooth, Gaussian process, step functions).

### 2D Regression

The 2D regression experiments are now organised by loss type, with comprehensive multi-run experimental capabilities:

#### CRPS Loss Experiments
- `experiments/2d_regression/crps_loss/gaussian_samples.py`: Multi-dimensional regression experiments using CRPS loss with both sample-based MLPs and affine normal models.

#### Energy Loss Experiments  
- `experiments/2d_regression/energy_loss/gaussian_samples.py`: Multi-dimensional regression experiments using energy score loss for comparison with CRPS-based training.

#### Multi-Run Experimental Structure

Both 2D regression experiment types automatically run multiple trials with different random seeds to provide robust performance estimates:

**Features:**
- Multiple experiments with configurable random seeds
- Support for different model types (MLPSampler and SimpleAffineNormal)
- Separate output directories for each run organised by loss type
- Complete YAML configuration files storing all parameters and results
- Automatic generation of training history and prediction visualisation plots
- Summary statistics across all runs
- Comprehensive evaluation metrics including MSE, CRPS, and calibration analysis

**Output Structure:**
```
experiments/2d_regression/
├── crps_loss/
│   ├── figures/
│   │   ├── run_001_seed_1/
│   │   │   ├── config.yaml                    # Complete run configuration and results
│   │   │   ├── training_history.png           # Training/validation loss curves
│   │   │   ├── prediction_samples.png         # Model predictions vs true values
│   │   │   └── prediction_samples_more.png    # Extended prediction visualisation
│   │   ├── run_002_seed_42/
│   │   │   └── ... (same structure)
│   │   └── experiment_summary.yaml            # Summary of all runs
│   ├── saved_models/
│   │   ├── run_001_seed_1/
│   │   │   └── mlp_sampler.pt                 # Trained model weights
│   │   └── ... (models for each run)
│   └── gaussian_samples.py
├── energy_loss/
│   ├── figures/                               # Same structure as crps_loss
│   ├── saved_models/
│   └── gaussian_samples.py
```

## Utility Modules

### Data Generation
- `experiments/data/generate.py`: Comprehensive data generation utilities for multi-dimensional regression with various noise structures and mean functions.
- `experiments/data/kernels.py`: Kernel functions for Gaussian process-based data generation.

### Training
- `experiments/training/train_functions.py`: Unified training functions supporting different model types (MLP samplers, affine normal models) and loss functions (CRPS, energy score, negative log-likelihood).
- `experiments/training/evaluation.py`: Comprehensive evaluation metrics and functions for model assessment including calibration analysis.

### Visualisation
- `experiments/visualisation/plotting.py`: Plotting utilities for training histories, prediction samples, and model comparisons.
- `experiments/visualisation/weight_tracking.py`: Tools for tracking and visualising model weight evolution during training.

## Installation

The code depends on PyTorch, torchnaut, and additional packages for configuration management. You can install the dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn torchnaut pyyaml scipy seaborn
```

## Running the experiments

```bash
# Run the 1D CRPS sampler example
python experiments/1d_regression/crps_samples_example.py

# Run the 1D Gaussian well-specified experiments
python experiments/1d_regression/gaussian_well_specified.py

# Run the 2D multi-seed CRPS-based regression experiments
python experiments/2d_regression/crps_loss/gaussian_samples.py

# Run the 2D multi-seed energy score-based regression experiments
python experiments/2d_regression/energy_loss/gaussian_samples.py
```

The experiments will generate figures in their respective `figures` directories and provide comprehensive performance statistics.

## Key Features

- **Multiple Loss Functions**: Support for CRPS loss, energy score, and negative log-likelihood
- **Diverse Model Types**: Sample-based MLPs, parametric Gaussian models, and pure distributional models (affine normal)
- **Direct sample-based training**: Training with CRPS on raw samples without distributional assumptions
- **Comprehensive Evaluation**: MSE, CRPS, energy score, calibration analysis, and uncertainty quantification
- **Multi-dimensional regression**: Full support for vector-valued outputs with proper multivariate scoring rules
- **Automated multi-run experimentation**: Statistical analysis across different random seeds
- **Complete experiment reproducibility**: YAML configuration files and systematic output organisation
- **Modular architecture**: Separate utilities for data generation, training, evaluation, and visualisation
- **Weight tracking**: Optional monitoring of model parameter evolution during training

## Model Comparison

The repository supports comparison between:
- **MLPSampler**: Neural networks that generate samples directly using torchnaut's EpsilonSampler
- **MLPGaussian**: Traditional neural networks predicting Gaussian parameters (mean and variance)
- **SimpleAffineNormal**: Pure distributional models learning affine transformations of standard normal variables

## Experimental Results

The multi-run structure provides robust performance evaluation:
- **Statistical significance**: Mean and standard deviation of metrics across runs
- **Calibration analysis**: Assessment of uncertainty quantification quality
- **Training stability**: Analysis of convergence behaviour across different initialisations
- **Model performance variability**: Understanding of robustness across random seeds
- **Loss function comparison**: Direct comparison between CRPS and energy score training

## Project Structure
```
CRPS/
├── models/                     # Model implementations
│   ├── mlp_gaussian.py         # Gaussian MLP
│   ├── mlp_crps_sampler.py     # Sample-based MLP
│   └── affine_normal.py        # Affine normal model
├── common/                     # Shared utilities
│   └── losses.py               # Loss function implementations
├── experiments/
│   ├── 1d_regression/          # Single-dimensional experiments
│   ├── 2d_regression/          # Multi-dimensional experiments
│   │   ├── crps_loss/          # CRPS-based training experiments
│   │   └── energy_loss/        # Energy score-based training experiments
│   ├── data/                   # Data generation utilities
│   ├── training/               # Training functions and utilities
│   └── visualisation/          # Plotting and analysis tools
└── README.md
```

## Getting Started

1. **Install dependencies** (see Installation section)
2. **Run 1D experiments** to familiarise yourself with basic functionality:
   ```bash
   python experiments/1d_regression/gaussian_well_specified.py
   ```
3. **Execute 2D experiments** for comprehensive multi-run analysis:
   ```bash
   python experiments/2d_regression/crps_loss/gaussian_samples.py
   ```
4. **Examine configuration files** in the generated `figures` directories to understand experiment parameters
5. **Review visualisations** for model performance analysis
6. **Compare different loss functions** by running both CRPS and energy score experiments

## Advanced Usage

### Custom Experiments
- Modify the `base_config` dictionary in experiment scripts to adjust:
  - Model architecture (hidden sizes, layers, dropout)
  - Training parameters (epochs, learning rates, patience)
  - Data generation (dimensions, noise types, sample sizes)
  - Loss functions and evaluation metrics

### Model Extensions
- Add new model types by implementing the standard interface
- Integrate additional loss functions in `common/losses.py`
- Extend evaluation metrics in `experiments/training/evaluation.py`

### Visualisation Customisation
- Modify plotting functions in `experiments/visualisation/plotting.py`
- Add weight tracking for new model types in `experiments/visualisation/weight_tracking.py`

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Torchnaut
- PyYAML
- SciPy
- seaborn

## License


## Contact
