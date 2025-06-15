"""
Utility functions for managing and inspecting training checkpoints.

This module provides helper functions to inspect checkpoint files, compare configurations,
and manage checkpoint directories.

Usage:
    python checkpoint_utils.py --inspect path/to/checkpoint.pth
    python checkpoint_utils.py --list-checkpoints path/to/save_dir
"""

import argparse
import torch
import yaml
from pathlib import Path
from datetime import datetime


def inspect_checkpoint(checkpoint_path):
    """
    Inspect a checkpoint file and print detailed information.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)
        
        # Basic information
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.6f}")
        print(f"Epochs without improvement: {checkpoint.get('epochs_without_improvement', 'Unknown')}")
        
        # Model information
        if 'model_state_dict' in checkpoint:
            model_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"Model parameters: {model_params:,} ({model_params/1e6:.2f}M)")
        
        # Model configuration
        if 'model_config' in checkpoint:
            print("\nModel configuration:")
            for key, value in checkpoint['model_config'].items():
                print(f"  {key}: {value}")
        
        # Training configuration
        if 'config' in checkpoint and 'training' in checkpoint['config']:
            print("\nTraining configuration:")
            for key, value in checkpoint['config']['training'].items():
                if key != 'resume_from_checkpoint':  # Skip this to avoid clutter
                    print(f"  {key}: {value}")
        
        # Optimizer state
        if 'optimizer_state_dict' in checkpoint:
            print(f"\nOptimizer state available: Yes")
            opt_state = checkpoint['optimizer_state_dict']
            if 'param_groups' in opt_state:
                print(f"Learning rate: {opt_state['param_groups'][0].get('lr', 'Unknown')}")
        else:
            print(f"\nOptimizer state available: No")
            
        # File size
        file_size = Path(checkpoint_path).stat().st_size
        print(f"\nFile size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Creation time
        creation_time = datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime)
        print(f"Last modified: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def list_checkpoints(save_dir):
    """
    List all checkpoint files in a directory with basic information.
    
    Args:
        save_dir (str): Directory containing checkpoint files
    """
    save_path = Path(save_dir)
    if not save_path.exists():
        print(f"Directory does not exist: {save_dir}")
        return
    
    checkpoint_files = list(save_path.glob("*.pth"))
    if not checkpoint_files:
        print(f"No checkpoint files found in: {save_dir}")
        return
    
    print(f"Checkpoint files in: {save_dir}")
    print("=" * 80)
    print(f"{'Filename':<25} {'Size (MB)':<10} {'Epoch':<8} {'Val Loss':<12} {'Modified':<20}")
    print("-" * 80)
    
    for ckpt_file in sorted(checkpoint_files):
        try:
            # File info
            file_size = ckpt_file.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(ckpt_file.stat().st_mtime)
            
            # Try to load basic info
            if ckpt_file.name.endswith('.pth'):
                try:
                    checkpoint = torch.load(ckpt_file, map_location='cpu')
                    epoch = checkpoint.get('epoch', 'N/A')
                    val_loss = checkpoint.get('best_val_loss', float('inf'))
                    val_loss_str = f"{val_loss:.4f}" if val_loss != float('inf') else 'N/A'
                except:
                    epoch = 'N/A'
                    val_loss_str = 'N/A'
            else:
                epoch = 'N/A'
                val_loss_str = 'N/A'
            
            print(f"{ckpt_file.name:<25} {file_size:<10.2f} {epoch:<8} {val_loss_str:<12} {mod_time.strftime('%Y-%m-%d %H:%M'):<20}")
            
        except Exception as e:
            print(f"{ckpt_file.name:<25} Error: {e}")


def compare_configs(checkpoint1_path, checkpoint2_path):
    """
    Compare configurations between two checkpoint files.
    
    Args:
        checkpoint1_path (str): Path to first checkpoint
        checkpoint2_path (str): Path to second checkpoint
    """
    try:
        ckpt1 = torch.load(checkpoint1_path, map_location='cpu')
        ckpt2 = torch.load(checkpoint2_path, map_location='cpu')
        
        config1 = ckpt1.get('config', {})
        config2 = ckpt2.get('config', {})
        
        print(f"Comparing configurations:")
        print(f"Checkpoint 1: {checkpoint1_path}")
        print(f"Checkpoint 2: {checkpoint2_path}")
        print("=" * 60)
        
        # Compare model configs
        model_config1 = ckpt1.get('model_config', {})
        model_config2 = ckpt2.get('model_config', {})
        
        print("Model configuration differences:")
        all_keys = set(model_config1.keys()) | set(model_config2.keys())
        differences_found = False
        
        for key in sorted(all_keys):
            val1 = model_config1.get(key, 'MISSING')
            val2 = model_config2.get(key, 'MISSING')
            if val1 != val2:
                print(f"  {key}: {val1} vs {val2}")
                differences_found = True
        
        if not differences_found:
            print("  No differences found")
        
        # Compare training configs
        train_config1 = config1.get('training', {})
        train_config2 = config2.get('training', {})
        
        print("\nTraining configuration differences:")
        all_keys = set(train_config1.keys()) | set(train_config2.keys())
        differences_found = False
        
        for key in sorted(all_keys):
            if key == 'resume_from_checkpoint':  # Skip this field
                continue
            val1 = train_config1.get(key, 'MISSING')
            val2 = train_config2.get(key, 'MISSING')
            if val1 != val2:
                print(f"  {key}: {val1} vs {val2}")
                differences_found = True
        
        if not differences_found:
            print("  No differences found")
            
    except Exception as e:
        print(f"Error comparing configurations: {e}")


def clean_old_checkpoints(save_dir, keep_best=True, keep_latest=True, keep_n_latest=5):
    """
    Clean old checkpoint files, keeping only the most important ones.
    
    Args:
        save_dir (str): Directory containing checkpoint files
        keep_best (bool): Whether to keep best_checkpoint.pth and best_model.pth
        keep_latest (bool): Whether to keep latest_checkpoint.pth
        keep_n_latest (int): Number of latest timestamped checkpoints to keep
    """
    save_path = Path(save_dir)
    if not save_path.exists():
        print(f"Directory does not exist: {save_dir}")
        return
    
    # Files to always keep
    protected_files = set()
    if keep_best:
        protected_files.update(['best_checkpoint.pth', 'best_model.pth'])
    if keep_latest:
        protected_files.add('latest_checkpoint.pth')
    
    # Find all checkpoint files
    all_checkpoints = list(save_path.glob("*.pth"))
    
    # Separate protected and cleanable files
    protected = [f for f in all_checkpoints if f.name in protected_files]
    cleanable = [f for f in all_checkpoints if f.name not in protected_files]
    
    # Sort cleanable by modification time (newest first)
    cleanable.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only the N latest
    to_keep = cleanable[:keep_n_latest]
    to_delete = cleanable[keep_n_latest:]
    
    print(f"Cleaning checkpoint directory: {save_dir}")
    print(f"Protected files: {[f.name for f in protected]}")
    print(f"Keeping latest {keep_n_latest} files: {[f.name for f in to_keep]}")
    
    if to_delete:
        print(f"Deleting {len(to_delete)} old checkpoint files:")
        for file in to_delete:
            print(f"  - {file.name}")
            file.unlink()
        print("Cleanup complete!")
    else:
        print("No files to delete.")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint management utilities")
    parser.add_argument('--inspect', type=str, help='Inspect a checkpoint file')
    parser.add_argument('--list-checkpoints', type=str, help='List checkpoints in directory')
    parser.add_argument('--compare', nargs=2, help='Compare two checkpoint configurations')
    parser.add_argument('--clean', type=str, help='Clean old checkpoints in directory')
    parser.add_argument('--keep-n-latest', type=int, default=5, help='Number of latest checkpoints to keep when cleaning')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_checkpoint(args.inspect)
    elif args.list_checkpoints:
        list_checkpoints(args.list_checkpoints)
    elif args.compare:
        compare_configs(args.compare[0], args.compare[1])
    elif args.clean:
        clean_old_checkpoints(args.clean, keep_n_latest=args.keep_n_latest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 