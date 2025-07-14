# Discrimination Experiments

This module provides a comprehensive framework for testing the discrimination ability of different distance functions using statistical power analysis.

## Overview

The discrimination experiments evaluate how well different distance functions can distinguish between:
1. **Null Hypothesis (H₀)**: Two samples from the same distribution
2. **Alternative Hypothesis (H₁)**: Samples from a ground truth distribution vs samples from a perturbed distribution

This framework allows you to empirically estimate:
- Type I error rates (false positive rate)
- Type II error rates (false negative rate)  
- Statistical power (1 - Type II error)
- Effect sizes using Cohen's d

## Files Structure

```
discrimination/
├── run.py                 # Main experiment script
├── config.yaml           # Configuration file
├── visualization.py      # Plotting and analysis module
├── samplers/             
│   └── samplers.py       # Sampling framework
├── distances/            
│   ├── crps_distance.py  # CRPS distance implementation
│   ├── energy_distance.py # Energy distance implementation
│   └── mmd.py           # Maximum Mean Discrepancy
└── results/             # Output directory (created automatically)
    ├── raw_data/        # Raw distance values
    ├── statistics/      # Summary statistics
    └── plots/           # Generated visualizations
```

## Quick Start

### 1. Basic Usage

Run experiments with default configuration:

```bash
cd experiments/discrimination
python run.py
```

### 2. Custom Configuration

Use a custom config file:

```bash
python run.py --config my_config.yaml
```

Specify custom output directory:

```bash
python run.py --output my_results/
```

### 3. Configuration Options

The `config.yaml` file contains all configurable parameters:

#### Ground Truth Distributions
```yaml
ground_truth_distributions:
  - name: "gaussian_2d"
    type: "gaussian"
    dimension: 2
    parameters:
      mean: [0.0, 0.0]
      cov: [[1.0, 0.0], [0.0, 1.0]]
```

#### Perturbations (Alternative Hypotheses)
```yaml
perturbations:
  - name: "location_shift"
    type: "composition"
    parameters:
      coefficients: [1.0, 1.0]  # [ground_truth_coeff, perturbation_coeff]
      perturbation_sampler:
        type: "constant"
        parameters:
          constant: [0.5, 0.0]  # Shift in x-direction
```

#### Distance Functions
```yaml
distances:
  - name: "crps"
    module: "crps_distance"
    function: "crps_distance"
    parameters:
      biased: false
```

## Available Distributions

### Ground Truth Distributions
- **Gaussian**: Multivariate normal distributions
- **Student-t**: Heavy-tailed distributions
- **Mixture Gaussian**: Mixture of Gaussians
- **Uniform**: Uniform distributions on hypercubes
- **Laplacian**: Multivariate Laplacian
- **Dirichlet**: Distributions on simplices

### Perturbation Types
- **Location shift**: Add constant offsets
- **Scale perturbation**: Add scaled noise
- **Heavy-tail perturbation**: Add heavy-tailed noise
- **Custom compositions**: Linear combinations of distributions

## Available Distance Functions

### 1. CRPS Distance
Sum of marginal 1D energy distances (equivalent to multivariate CRPS):
```python
crps_distance(samples_p, samples_q, biased=False)
```

### 2. Energy Distance
Multivariate energy distance with configurable power parameter:
```python
energy_distance(samples_p, samples_q, beta=1.0, biased=False)
```

### 3. Maximum Mean Discrepancy (MMD)
Kernel-based distance with multiple kernel options:
```python
mmd_distance(samples_p, samples_q, kernel_type='gaussian', sigma=1.0, biased=False)
```

## Example: Custom Experiment

### 1. Create Custom Configuration

```yaml
# custom_config.yaml
experiment:
  name: "custom_discrimination"
  n_trials: 500
  random_seed: 123

sample_sizes:
  M: 50
  N: 50

ground_truth_distributions:
  - name: "heavy_tailed"
    type: "student_t"
    dimension: 3
    parameters:
      df: 2.0
      loc: [0.0, 0.0, 0.0]
      scale: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

perturbations:
  - name: "correlation_change"
    type: "composition"
    parameters:
      coefficients: [0.8, 0.2]
      perturbation_sampler:
        type: "gaussian"
        parameters:
          mean: [0.0, 0.0, 0.0]
          cov: [[1.0, 0.7, 0.7], [0.7, 1.0, 0.7], [0.7, 0.7, 1.0]]

distances:
  - name: "energy_l1"
    module: "energy_distance"
    function: "energy_distance"
    parameters:
      beta: 1.0
      biased: false
```

### 2. Run Custom Experiment

```bash
python run.py --config custom_config.yaml --output custom_results/
```

## Output Analysis

### 1. Statistical Results
The framework generates comprehensive statistics in `results/statistics/summary_statistics.json`:

```json
{
  "null_gaussian_2d_crps": {
    "mean": 0.142,
    "std": 0.038,
    "quantiles": {
      "q_5": 0.089,
      "q_95": 0.208
    }
  },
  "power_gaussian_2d_location_shift_crps": {
    "power_alpha_0.05": 0.89,
    "effect_size_alpha_0.05": 2.34
  }
}
```

### 2. Visualizations
Generated plots include:
- **Distance distributions**: Histograms comparing H₀ vs H₁
- **Power curves**: Statistical power across different α levels
- **ROC curves**: Classification performance analysis
- **Q-Q plots**: Distribution comparison plots
- **Box plots**: Distribution summaries
- **Effect sizes**: Cohen's d across configurations

### 3. Raw Data
Raw distance values are saved as NumPy arrays in `results/raw_data/` for further analysis.

## Power Analysis Interpretation

### Statistical Power Values
- **Power < 0.5**: Poor discrimination ability
- **Power 0.5-0.8**: Moderate discrimination ability  
- **Power > 0.8**: Good discrimination ability
- **Power > 0.95**: Excellent discrimination ability

### Effect Size Interpretation (Cohen's d)
- **d < 0.2**: Small effect
- **d = 0.5**: Medium effect
- **d > 0.8**: Large effect

## Advanced Usage

### Parallel Processing
Enable parallel processing for faster experiments:
```yaml
advanced:
  parallel_processing: true
  n_jobs: -1  # Use all available cores
```

### Custom Sampling Framework
Add new distributions by extending the `BaseSampler` class in `samplers/samplers.py`:

```python
class CustomSampler(BaseSampler):
    def __init__(self, dimension: int, **kwargs):
        super().__init__(dimension, **kwargs)
        # Custom initialization
        
    def sample(self, n_samples: int) -> np.ndarray:
        # Custom sampling logic
        return samples
```

### Custom Distance Functions
Add new distance functions by creating modules in `distances/`:

```python
def custom_distance(samples_p, samples_q, **kwargs):
    """Custom distance function."""
    # Implementation
    return distance_value
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `n_trials` or enable `memory_optimization`
3. **Slow Execution**: Enable `parallel_processing` or reduce problem size

### Performance Tips

1. **Start Small**: Begin with small sample sizes and few trials for testing
2. **Use Parallel Processing**: Enable for significant speedup
3. **Profile Memory**: Monitor memory usage for large experiments
4. **Batch Processing**: Process results in batches for very large experiments

## Dependencies

Required packages:
- `numpy`
- `scipy` 
- `matplotlib`
- `seaborn`
- `pandas`
- `pyyaml`
- `tqdm`

Install with:
```bash
pip install numpy scipy matplotlib seaborn pandas pyyaml tqdm
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{discrimination_experiments,
  title={Statistical Discrimination Experiments for Distance Functions},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/discrimination-experiments}}
}
``` 