# Training the ConvCNP Sampler Model

This directory contains the necessary scripts and configurations for training the `ConvCNPSampler` model on the 2D CFD dataset.

## How to Run Training

The training process is managed by `train_convcnp.py` and configured via `config.yml`.

To start the training, execute the following command from the **root of the `CRPS` project directory**:

```bash
python experiments/pdebench_experiments/training/train_convcnp.py --config experiments/pdebench_experiments/training/config.yml
```

### Prerequisites

- Ensure you have all the required packages installed, including `torch`, `pyyaml`, and `tqdm`.
- The script expects the PDEBench 2D CFD dataset to be located at the path specified in `config.yml`.

## Configuration (`config.yml`)

The `config.yml` file is divided into four main sections:

1.  **`data`**: Specifies the dataset path and its configuration. This section is used to initialize the `CFD2DDataset` and `DataLoader`.
    -   `path`: Relative path to the `.hdf5` data file.
    -   `config`: Parameters passed directly to the `CFD2DDataset`, such as variable selection, temporal windowing (`time_lag`, `time_predict`), and spatial sampling.

2.  **`model`**: Defines the architecture of the `ConvCNPSampler`.
    -   Parameters like `latent_dim`, `cnn_hidden_channels`, and `grid_size` control the model's capacity and structure.
    -   Note: `n_input_vars`, `time_lag`, and `time_predict` are automatically inferred from the `data` configuration and do not need to be set here.

3.  **`training`**: Contains hyperparameters for the training process.
    -   `epochs`: Total number of training epochs.
    -   `batch_size`: Number of samples per batch.
    -   `learning_rate`: The learning rate for the Adam optimizer.
    -   `n_crps_samples`: The number of `epsilon` samples used to approximate the CRPS loss during training and evaluation.

4.  **`logging`**: Manages where model checkpoints are saved.
    -   `save_dir`: Directory where the best performing model checkpoint (`best_model.pth`) will be saved. The script automatically saves the model with the lowest validation loss.

## Output

-   **Console Output**: During training, the script will print the progress for each epoch, including the average training and validation loss.
-   **Saved Model**: The best model weights, based on validation loss, will be saved to the path specified in `logging.save_dir` within the configuration file. 