# AR Time Series Training Framework

This directory contains a comprehensive framework for training probabilistic models on multivariate autoregressive (AR) time series data. The framework supports multiple model architectures, loss functions, and training configurations.

## Overview

The training script (`main.py`) provides a unified interface for:
- **Data Generation**: Synthetic multivariate AR time series with configurable parameters
- **Model Training**: Support for FGN Encoder, MLP Sampler, and Affine Normal models
- **Loss Functions**: CRPS, Energy Score, Variogram Score, Kernel Score, and MMD with combination support
- **Flexible Configuration**: YAML-based configuration with command-line overrides

## Quick Start

### 1. Basic Usage with Default Configuration

```bash
cd CRPS/experiments/ar
python main.py
```

This runs training with default parameters (FGN Encoder model, CRPS loss, 3D AR(2) data).

### 2. Using Configuration Files

```bash
# Train FGN Encoder model
python main.py --config configs/general_config.yaml

# Train with CRPS loss
python main.py --config configs/crps_loss_config.yaml

# Train with Energy Score loss
python main.py --config configs/energy_loss_config.yaml

# Train spatial AR(1) on 3x3 grid
python main.py --config configs/spatial_ar_config.yaml

# Train spatial AR(2) on 4x4 grid with multiple filters
python main.py --config configs/spatial_ar2_config.yaml
```

### 3. Command-Line Configuration Override

```bash
# Override specific parameters
python main.py --config-dict '{"model": {"type": "mlp_sampler"}, "training": {"n_epochs": 200}}'

# Change loss function
python main.py --config-dict '{"loss": {"losses": ["energy_score_loss"]}}'
```

### 4. Generate Example Configuration

```bash
python main.py --save-example-config my_config.yaml
```

### 5. Multi-Seed Experiments

```bash
# Test multi-seed functionality with 3 seeds
python main.py --config configs/multi_seed_test.yaml

# Run CRPS loss comparison with 5 seeds
python main.py --config configs/crps_loss_config.yaml

# Quick multi-seed test via command line
python main.py --config-dict '{"experiment": {"seeds": [1, 2, 3]}, "training": {"n_epochs": 10}}'
```

## Configuration Structure

The configuration is organised into five main sections:

### Data Configuration

Controls AR time series generation and dataset creation:

```yaml
data:
  # Data generation parameters
  n_timesteps: 200        # Total timesteps per series
  dimension: 3            # Multivariate dimension
  ar_order: 2            # AR model order
  noise_scale: 0.1       # Noise scaling factor
  n_series: 500          # Number of series to generate
  
  # Dataset parameters (IMPORTANT: Input must be single timestep, output can be multi-timestep)
  input_timesteps: 1     # Input window size (MUST be 1)
  output_timesteps: 5    # Output prediction horizon (can be > 1 via autoregression)
  split_ratios:          # Train/val/test splits
    train: 0.7
    val: 0.15
    test: 0.15
  
  # Optional custom parameters
  A_matrices: null       # Custom AR coefficient matrices
  noise_cov: null        # Custom noise covariance
  random_state: 42       # Reproducibility seed
```

⚠️ **IMPORTANT LIMITATION**: The current models (FGN Encoder, MLP Sampler, Affine Normal) only support **single timestep inputs**. You must set `input_timesteps: 1` in your configuration. However, **multi-timestep outputs are supported via autoregressive generation** - you can set `output_timesteps > 1` and the model will generate sequences autoregressively.

### Model Configuration

Specifies the probabilistic model architecture:

```yaml
model:
  type: fgn_encoder  # Options: 'fgn_encoder', 'mlp_sampler', 'affine_normal'
  
  # FGN Encoder specific parameters
  fgn_encoder:
    hidden_size: 64
    latent_dim: 16
    n_layers: 3
    dropout_rate: 0.1
    zero_inputs: false
    non_linear: true
    activation_function: relu
  
  # MLP Sampler specific parameters
  mlp_sampler:
    hidden_size: [64, 64, 32]  # Can be list or single int
    latent_dim: 16
    n_layers: 3
    dropout_rate: 0.1
    sample_layer_index: 2
    zero_inputs: false
    non_linear: true
    activation_function: relu
  
  # Affine Normal (baseline) parameters
  affine_normal: {}  # No parameters needed
```

### Loss Function Configuration

Configure single or combined loss functions:

```yaml
loss:
  # List of loss functions to combine
  losses: ['crps_loss_general', 'energy_score_loss']
  
  # Arguments for each loss function
  loss_function_args:
    crps_loss_general: {}
    energy_score_loss:
      norm_dim: true
    variogram_score_loss:
      p: 1.5
      weights: null
    kernel_score_loss:
      kernel_type: gaussian
      sigma: 1.0
    mmd_loss:
      kernel_type: gaussian
      sigma: 1.0
  
  # Weighting coefficients for combination
  coefficients:
    crps_loss_general: 1.0
    energy_score_loss: 0.5
```

### Training Configuration

Control the training process:

```yaml
training:
  n_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  n_samples: 50          # Samples for probabilistic evaluation
  patience: 15           # Early stopping patience
  device: auto           # 'auto', 'cuda', 'cpu'
  num_workers: 4
  shuffle_train: true
  normalise_data: true
  verbose: true
  
  # Optimizer settings
  optimizer: adam        # 'adam', 'sgd', 'rmsprop'
  optimizer_args:
    weight_decay: 0.0
    betas: [0.9, 0.999]
```

### Experiment Configuration

Manage experiment outputs and evaluation:

```yaml
experiment:
  name: my_experiment
  save_dir: results
  save_model: true
  save_results: true
  log_interval: 10
  evaluate_test: true
  # Multi-seed experiment configuration
  seeds: [42, 123, 456, 789, 999]  # List of random seeds to run
  parallel_seeds: false  # Whether to run seeds in parallel (experimental)
```

#### Multi-Seed Experiments

The framework supports running multiple experiments with different random seeds to assess model performance variability:

- **Single Seed**: If `seeds` contains only one value, runs a single experiment
- **Multi-Seed**: If `seeds` contains multiple values, runs separate experiments for each seed
- **Results Structure**: 
  - Each seed creates a subdirectory: `results/experiment_name/seed_42/`, `results/experiment_name/seed_123/`, etc.
  - Aggregated statistics are saved in `results/experiment_name/aggregated_results.json`
  - Individual seed configurations and results are saved in each seed subdirectory

**Example Multi-Seed Usage**:
```bash
# Run with 5 different seeds
python main.py --config configs/multi_seed_test.yaml

# Override seeds via command line
python main.py --config-dict '{"experiment": {"seeds": [1, 2, 3, 4, 5]}}'
```

**Aggregated Results Include**:
- Validation and test loss statistics (mean, std, min, max, median)
- Individual training histories for each seed
- Complete results from each seed experiment

## Available Models

⚠️ **All models only support single timestep inputs. Multi-timestep outputs are generated autoregressively.**

### 1. FGN Encoder Sampler
- Uses conditional layer normalisation with encoded noise
- Flexible architecture with configurable layers and activations
- Good for complex probabilistic modelling
- Input: `[batch_size, input_dim]` Output: `[batch_size, n_samples, output_dim]`

### 2. MLP Sampler
- Multi-layer perceptron with EpsilonSampler for noise injection
- Configurable sampling layer position
- Supports variable hidden layer sizes
- Input: `[batch_size, input_dim]` Output: `[batch_size, n_samples, output_dim]`

### 3. Affine Normal (Baseline)
- Simple affine transformation of multivariate Gaussian
- Represents y = Ax + b where x ~ N(0,I)
- Good baseline for comparison
- Unconditional generation (ignores input): Output: `[batch_size, n_samples, output_dim]`

### Autoregressive Generation for Multi-Timestep Outputs

When `output_timesteps > 1`, the models generate sequences autoregressively:

1. **Step 1**: Use initial input to generate samples for timestep t+1
2. **Step 2**: Use mean of t+1 samples as input to generate timestep t+2  
3. **Step 3**: Continue until all output timesteps are generated

This allows the models to generate coherent multi-timestep sequences despite only being designed for single-timestep inputs.

**Example**: With `input_timesteps: 1` and `output_timesteps: 5`:
- Input: `[batch_size, 1, dim]` → flattened to `[batch_size, dim]`
- Output: `[batch_size, n_samples, 5, dim]` (generated autoregressively)

#### Loss Calculation for Autoregressive Outputs

When `output_timesteps > 1`, the loss is computed **explicitly for each timestep** and then averaged. Specifically, the process is:

1.  **Generate Full Sequence**: The model autoregressively generates the full sequence of samples, resulting in a tensor of shape `[batch, n_samples, timesteps, dim]`.
2.  **Iterate and Compute Loss**: The training loop iterates from `t = 1 to T` (where `T` is `output_timesteps`). In each iteration, it computes the loss between:
    *   The predicted samples for that timestep: `samples[:, :, t, :]`
    *   The ground truth target for that timestep: `targets[:, t, :]`
3.  **Average Losses**: The final loss for the sequence is the average of the losses computed at each timestep.

This approach correctly treats the prediction as a sequence of individual steps and ensures the training signal is an average of the performance across the entire prediction horizon.

## Available Loss Functions

1. **CRPS Loss** (`crps_loss_general`): Continuous Ranked Probability Score
2. **Energy Score** (`energy_score_loss`): Multivariate proper scoring rule
3. **Variogram Score** (`variogram_score_loss`): Spatial dependence-aware scoring
4. **Kernel Score** (`kernel_score_loss`): Kernel-based scoring with multiple kernels
5. **MMD Loss** (`mmd_loss`): Maximum Mean Discrepancy

### Combining Loss Functions

You can combine multiple loss functions with different weights:

```yaml
loss:
  losses: ['crps_loss_general', 'energy_score_loss', 'variogram_score_loss']
  coefficients:
    crps_loss_general: 1.0
    energy_score_loss: 0.5
    variogram_score_loss: 0.3
```

### Separate Training and Evaluation Losses (NEW)

You can now use different loss functions for training and evaluation, which allows you to optimise training convergence while maintaining rigorous evaluation standards:

#### Basic Example: Different Loss Functions

```yaml
loss:
  # Training loss: Use energy score for better gradients
  training_loss:
    loss_instances:
      - name: energy_training
        loss_function: energy_score_loss
        coefficient: 1.0
        loss_args:
          norm_dim: true
  
  # Evaluation loss: Use CRPS for precise assessment
  evaluation_loss:
    loss_instances:
      - name: crps_evaluation
        loss_function: crps_loss_general
        coefficient: 1.0
        loss_args: {}
```

#### Advanced Example: Multiple Loss Combinations

```yaml
loss:
  # Training: Focus on convergence
  training_loss:
    loss_instances:
      - name: energy_primary
        loss_function: energy_score_loss
        coefficient: 1.0
        loss_args:
          norm_dim: true
      - name: variogram_secondary
        loss_function: variogram_score_loss
        coefficient: 0.3
        loss_args:
          p: 1.5
  
  # Evaluation: Comprehensive assessment
  evaluation_loss:
    loss_instances:
      - name: crps_primary
        loss_function: crps_loss_general
        coefficient: 1.0
      - name: energy_secondary
        loss_function: energy_score_loss
        coefficient: 0.5
        loss_args:
          norm_dim: false
      - name: kernel_tertiary
        loss_function: kernel_score_loss
        coefficient: 0.3
        loss_args:
          kernel_type: gaussian
          sigma: 1.0
```

#### Running Separate Loss Experiments

```bash
# Basic separate losses
python main.py --config configs/separate_losses_example.yaml

# Advanced separate losses
python main.py --config configs/advanced_separate_losses_example.yaml
```

#### Benefits of Separate Losses

1. **Training Optimisation**: Use loss functions with better gradient properties (e.g., energy score) for faster convergence
2. **Evaluation Precision**: Use established benchmarks (e.g., CRPS) for accurate model assessment
3. **Research Flexibility**: Compare different loss combinations without retraining
4. **Multi-Scale Assessment**: Use different spatial pooling for training vs. evaluation

**Note**: The framework maintains full backward compatibility - existing configurations will continue to work unchanged.

## Output Structure

Training produces the following outputs in the specified `save_dir`:

```
results/
└── experiment_name/
    ├── config.yaml          # Complete configuration used
    ├── best_model.pth       # Best model state dict
    └── results.json         # Training history and metrics
```

### Results JSON Structure

```json
{
  "training_history": {
    "train_loss": [...],
    "val_loss": [...],
    "epochs": [...]
  },
  "best_val_loss": 0.1234,
  "test_results": {
    "test_loss": 0.1456
  },
  "model_info": {
    "type": "fgn_encoder",
    "parameters": 12345,
    "trainable_parameters": 12345
  }
}
```

## Advanced Usage

### Custom AR Coefficient Matrices

Provide your own AR coefficient matrices:

```yaml
data:
  A_matrices: 
    - [[0.5, 0.1], [0.2, 0.6]]  # A_1 matrix for AR(2)
    - [[0.1, 0.0], [0.0, 0.1]]  # A_2 matrix for AR(2)
```

### Custom Noise Covariance

Specify custom noise covariance structure:

```yaml
data:
  noise_cov: [[1.0, 0.5], [0.5, 1.0]]  # 2x2 covariance matrix
```

### Spatial AR Processes (NEW)

Generate AR processes on 2D spatial grids using convolutional filters:

```yaml
data:
  dimension: 9  # Must equal grid_size height * width
  ar_order: 1
  spatial_args:
    grid_size: [3, 3]  # 3x3 spatial grid
    conv_filters:
      - [[0.1, 0.2, 0.1],    # AR lag 1: smoothing filter
         [0.2, 0.4, 0.2],    # Strong center influence
         [0.1, 0.2, 0.1]]
```

For AR(2) with multiple spatial patterns:

```yaml
data:
  dimension: 16  # 4x4 grid
  ar_order: 2
  spatial_args:
    grid_size: [4, 4]
    conv_filters:
      # AR lag 1: Cross pattern (vertical/horizontal coupling)
      - [[0.0, 0.1, 0.0],
         [0.1, 0.3, 0.1],
         [0.0, 0.1, 0.0]]
      
      # AR lag 2: Diagonal pattern
      - [[0.05, 0.0, 0.05],
         [0.0,  0.1, 0.0],
         [0.05, 0.0, 0.05]]
```

**Spatial AR Features:**
- Treats multivariate data as 2D spatial grids
- Each location depends on neighbors via convolutional filters
- Maintains dimensionality with zero-padding at edges
- Supports any AR order with corresponding filter patterns
- Automatically generates AR coefficient matrices from spatial filters
- Cannot be combined with custom `A_matrices` (spatial filters generate them)

**Spatial AR Requirements:**
- `dimension` must equal `grid_size[0] * grid_size[1]`
- `conv_filters` list length must equal `ar_order`
- All filters must have odd dimensions for proper centering
- Filters can be any 2D numpy-compatible array

### Zero-Input Models

Train models that ignore input (useful for unconditional generation):

```yaml
model:
  fgn_encoder:
    zero_inputs: true
```

### Different Activation Functions

Available activation functions: `relu`, `sigmoid`, `tanh`, `leaky_relu`, `gelu`, `elu`, `softplus`

```yaml
model:
  fgn_encoder:
    activation_function: gelu
```

### Tracking and Visualisation

The framework includes a powerful tracking system that saves model checkpoints during training and generates AR inference visualisation plots:

```yaml
tracking:
  enabled: true                    # Enable/disable tracking
  track_every: 10                  # Save checkpoint every N epochs
  sample_indices: [0, 1, 2]       # Which dataset samples to visualise
  n_samples: 100                   # Number of samples for inference plots
  kde_bandwidth: "auto"            # KDE bandwidth - use "auto" for automatic detection
  contour_levels: [0.65, 0.95, 0.99]  # Confidence levels for contour plots
  max_checkpoints: 10              # Maximum checkpoints to keep (None = keep all)
  generate_plots_after_training: true  # Generate plots after training completes
  create_animation: false          # Create training progression animation
```

#### Automatic KDE Bandwidth Detection

The `kde_bandwidth` parameter supports automatic bandwidth selection for KDE visualisations:

- **`"auto"`**: Automatically computes optimal bandwidth using Silverman's rule of thumb
- **Numeric value**: Use a specific bandwidth (e.g., `0.1`)
- **Fallback**: Invalid values automatically fall back to auto-detection

**Bandwidth Methods Available:**
- **Silverman's Rule**: `h = 1.06 * σ * n^(-1/5)` for 1D, geometric mean for multivariate
- **Scott's Rule**: `h = σ * n^(-1/(d+4))` where d is dimensionality  
- **IQR Method**: Robust to outliers using `min(std, IQR/1.34)`

```yaml
tracking:
  kde_bandwidth: "auto"      # Automatic detection (recommended)
  # or
  kde_bandwidth: 0.15        # Manual specification
```

The automatic detection analyses your sample data characteristics and chooses appropriate bandwidths, eliminating the need for manual tuning while providing optimal visualisation quality.

## Dependencies

The framework requires:
- PyTorch
- NumPy
- PyYAML
- torchnaut (for CRPS/Energy Score functions)
- tqdm (for progress bars)

## Tips for Good Results

1. **Data Normalisation**: Always enable `normalise_data: true` for stable training
2. **Early Stopping**: Use appropriate `patience` values (15-25 epochs)
3. **Sample Count**: Use 50-100 samples for loss evaluation; more samples = more stable but slower
4. **Learning Rate**: Start with 0.001-0.002; adjust based on convergence
5. **Architecture**: FGN Encoder typically performs best, MLP Sampler is more flexible
6. **Loss Functions**: CRPS is generally most reliable; Energy Score for multivariate; combine for robustness

## Troubleshooting

### Common Issues

1. **Single Timestep Input Error**: If you get "Current models only support single timestep inputs", ensure `input_timesteps: 1` in your data configuration. Output timesteps can be > 1 for autoregressive generation
2. **Import Errors**: Ensure you're running from the correct directory and all dependencies are installed
3. **Memory Issues**: Reduce `batch_size`, `n_samples`, or `n_series`
4. **Slow Training**: Reduce `n_samples` or use GPU (`device: cuda`)
5. **Poor Convergence**: Try lower learning rates or different optimisers
6. **NaN Losses**: Check data normalisation and reduce learning rate

### Performance Optimisation

- Use GPU if available: `device: cuda`
- Increase `num_workers` for data loading
- Use appropriate `batch_size` for your hardware
- Consider mixed precision training for large models

## Examples

See the `example_configs/` directory for complete working examples of different model and loss function combinations. 