# CRPS - Continuous Ranked Probability Score

This repository contains experiments and models for probabilistic regression using the Continuous Ranked Probability Score (CRPS) as a loss function. CRPS is a proper scoring rule that measures the compatibility of a probability forecast with an observation.

## Models

- `models/mlp_gaussian.py`: Implements a simple MLP that predicts Gaussian distributions with mean and variance outputs.
- `models/mlp_crps_sampler.py`: Implements a sample-based probabilistic MLP using the EpsilonSampler from torchnaut to generate samples directly.

## Experiments

### 1D Regression

- `experiments/1d_regression/crps_samples_example.py`: Demonstrates training a sample-based MLP using the CRPS loss on a synthetic 1D regression problem with non-Gaussian noise.
- `experiments/1d_regression/gaussian_not_well_specified.py`: Experiments with Gaussian models on datasets where the noise is not Gaussian.

## Installation

The code depends on PyTorch and torchnaut. You can install the dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn torchnaut
```

## Running the experiments

```bash
# Run the CRPS sampler example
python experiments/1d_regression/crps_samples_example.py
```

The experiments will generate figures in the `experiments/1d_regression/figures` directory.

## Key Features

- Direct sample-based training with CRPS
- Implementations for both parametric (Gaussian) and non-parametric (sample-based) models
- Visualisation of prediction intervals

## Project Structure
- `experiments/` - Directory containing experimental data and analysis

## Getting Started
[Add instructions for setting up and running the project]

## Requirements
[Add any specific requirements or dependencies]

## Usage
[Add usage instructions]

## Contributing
[Add contribution guidelines if applicable]

## License
[Add license information]

## Contact
[Add contact information] 