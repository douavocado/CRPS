"""
Main training script for the Convolutional Conditional Neural Process (ConvCNP) Sampler.

This script handles the training and evaluation of the ConvCNPSampler model on the
2D CFD dataset from PDEBench. It is configurable via a YAML file and supports
checkpoint saving and resuming.

Features:
- Automatic checkpoint saving (best model and latest model)
- Resume training from checkpoint with full state restoration
- Early stopping with configurable patience
- Configurable checkpoint save frequency

Usage:
    # Start fresh training
    python train_convcnp.py --config path/to/config.yml
    
    # Resume from checkpoint (command line - recommended)
    python train_convcnp.py --config path/to/config.yml --resume path/to/checkpoint.pth
    python train_convcnp.py --config path/to/config.yml -r path/to/checkpoint.pth
    
Configuration options for checkpointing:
    training:
        resume_from_checkpoint: "path/to/checkpoint.pth"  # Optional: resume from checkpoint (overridden by --resume)
        checkpoint_save_frequency: 5  # Save latest checkpoint every N epochs (default: 5)
        validate_config_on_resume: true  # Check config compatibility on resume (default: true)
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


def save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_without_improvement, save_dir, config, is_best=False):
    """
    Save a training checkpoint with all necessary information for resuming.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_without_improvement': epochs_without_improvement,
        'config': config,
        'model_config': {
            'n_input_vars': len(config['data']['config']['input_variables']),
            'time_lag': config['data']['config']['time_lag'],
            'time_predict': config['data']['config']['time_predict'],
            **config['model']
        }
    }
    
    # Save latest checkpoint
    latest_path = save_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint if this is the best model
    if is_best:
        best_path = save_dir / "best_checkpoint.pth"
        torch.save(checkpoint, best_path)
        # Also save just the model state dict for compatibility
        best_model_path = save_dir / "best_model.pth"
        torch.save(model.state_dict(), best_model_path)
    
    return latest_path, best_path if is_best else None


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load a training checkpoint and restore model, optimizer, and training state.
    
    Returns:
        tuple: (start_epoch, best_val_loss, epochs_without_improvement, config)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract training state
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    epochs_without_improvement = checkpoint['epochs_without_improvement']
    saved_config = checkpoint.get('config', None)
    
    print(f"Resumed from epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")
    print(f"Epochs without improvement: {epochs_without_improvement}")
    
    return start_epoch, best_val_loss, epochs_without_improvement, saved_config


def main(config_path: str, resume_checkpoint: str = None):
    """
    Main training loop.
    
    Args:
        config_path (str): Path to the configuration file
        resume_checkpoint (str, optional): Path to checkpoint file to resume from.
                                         If provided, overrides config setting.
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

    # Setup directories
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Check for resuming from checkpoint
    # Priority: command line argument > config file setting
    resume_from = resume_checkpoint or config['training'].get('resume_from_checkpoint', None)
    
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            start_epoch, best_val_loss, epochs_without_improvement, saved_config = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            # Optionally validate that the loaded config matches current config
            if saved_config and config['training'].get('validate_config_on_resume', True):
                # Check critical config parameters match
                critical_params = ['model', 'data']
                for param in critical_params:
                    if param in saved_config and saved_config[param] != config[param]:
                        print(f"WARNING: {param} configuration differs from saved checkpoint!")
                        print(f"Current: {config[param]}")
                        print(f"Saved: {saved_config[param]}")
        else:
            print(f"WARNING: Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
    elif resume_checkpoint is not None:
        # If --resume was provided but file doesn't exist, this is an error
        print(f"ERROR: Resume checkpoint specified but file does not exist: {resume_checkpoint}")
        print("Exiting...")
        return
    
    # Early stopping parameters
    patience = config['training'].get('patience', 10)  # Default patience of 10 epochs
    
    # Get and print loss function
    loss_function = config['training'].get('loss_function', 'crps')
    print(f"\nUsing loss function: {loss_function}")
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1} with early stopping (patience: {patience})...")
    else:
        print(f"Starting training with early stopping (patience: {patience})...")
    
    # Training loop
    total_epochs = config['training']['epochs']
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        val_loss = evaluate(model, val_loader, device, config)
        
        print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
        
        # Save checkpoint
        latest_path, best_path = save_checkpoint(
            model, optimizer, epoch, best_val_loss, epochs_without_improvement, 
            save_dir, config, is_best=is_best
        )
        
        if is_best:
            print(f"Saved new best checkpoint to {best_path}")
        
        # Save latest checkpoint every few epochs or at the end
        save_frequency = config['training'].get('checkpoint_save_frequency', 5)
        if (epoch + 1) % save_frequency == 0 or epoch == total_epochs - 1:
            print(f"Saved latest checkpoint to {latest_path}")
            
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {patience})")
            # Save final checkpoint
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, epochs_without_improvement, 
                save_dir, config, is_best=False
            )
            break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Print checkpoint locations for reference
    print(f"\nCheckpoint files saved in: {save_dir}")
    print(f"- Best model: best_checkpoint.pth, best_model.pth")
    print(f"- Latest checkpoint: latest_checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvCNP Sampler model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('-r', '--resume', type=str, help='Resume training from checkpoint file (overrides config setting)')
    args = parser.parse_args()
    main(args.config, resume_checkpoint=args.resume) 