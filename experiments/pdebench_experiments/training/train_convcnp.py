"""
Main training script for the Convolutional Conditional Neural Process (ConvCNP) Sampler.

This script handles the training and evaluation of the ConvCNPSampler model on the
2D CFD dataset from PDEBench. It is configurable via a YAML file.

Usage:
    python train_convcnp.py --config path/to/config.yml
"""
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import os

# Adjusting python path to allow imports from parent directories
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from models.convcnp_sampler import ConvCNPSampler
from experiments.pdebench_experiments.dataset.dataset import create_cfd_dataloader


def train_one_epoch(model, dataloader, optimizer, device, config):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    # Get loss function type from config
    loss_function = config['training'].get('loss_function', 'crps')  # Default to CRPS
    
    for batch in pbar:
        optimizer.zero_grad()
        
        input_data = batch['input_data'].to(device)
        input_coords = batch['input_coords'].to(device)
        target_data = batch['target_data'].to(device)
        target_coords = batch['target_coords'].to(device)
        
        # Select loss function based on configuration
        if loss_function == 'energy_score':
            loss = model.energy_score_loss(
                input_data,
                input_coords,
                target_coords,
                target_data,
                n_samples=config['training']['n_crps_samples']
            )
        elif loss_function == 'crps':
            loss = model.crps_loss(
                input_data,
                input_coords,
                target_coords,
                target_data,
                n_samples=config['training']['n_crps_samples']
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_function}. Choose 'energy_score' or 'crps'.")
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, config):
    """
    Evaluates the model on the validation or test set.
    """
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    # Get loss function type from config
    loss_function = config['training'].get('loss_function', 'crps')  # Default to CRPS

    for batch in pbar:
        input_data = batch['input_data'].to(device)
        input_coords = batch['input_coords'].to(device)
        target_data = batch['target_data'].to(device)
        target_coords = batch['target_coords'].to(device)
        
        # Select loss function based on configuration
        if loss_function == 'energy_score':
            loss = model.energy_score_loss(
                input_data,
                input_coords,
                target_coords,
                target_data,
                n_samples=config['training']['n_crps_samples']
            )
        elif loss_function == 'crps':
            loss = model.crps_loss(
                input_data,
                input_coords,
                target_coords,
                target_data,
                n_samples=config['training']['n_crps_samples']
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_function}. Choose 'energy_score' or 'crps'.")
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)


def main(config_path: str):
    """
    Main training loop.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    data_config = config['data']
    train_loader = create_cfd_dataloader(
        data_path=data_config['path'],
        config=data_config['config'],
        split='train',
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    val_loader = create_cfd_dataloader(
        data_path=data_config['path'],
        config=data_config['config'],
        split='val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    # Initialize model
    model_config = config['model']
    # Derive some model params from data config
    model_config['n_input_vars'] = len(data_config['config']['input_variables'])
    model_config['time_lag'] = data_config['config']['time_lag']
    model_config['time_predict'] = data_config['config']['time_predict']
    
    model = ConvCNPSampler(**model_config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Early stopping parameters
    patience = config['training'].get('patience', 10)  # Default patience of 10 epochs
    epochs_without_improvement = 0
    
    # Get and print loss function
    loss_function = config['training'].get('loss_function', 'crps')
    print(f"\nUsing loss function: {loss_function}")
    print(f"Starting training with early stopping (patience: {patience})...")
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        val_loss = evaluate(model, val_loader, device, config)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {patience})")
                break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvCNP Sampler model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config) 