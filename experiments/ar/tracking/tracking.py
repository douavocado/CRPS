import torch
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np


class ARInferenceTracker:
    """
    Tracker for collecting AR model states during training for later inference visualization.
    
    This tracker saves model checkpoints at specified intervals and collects metadata
    needed to recreate AR inference plots using the plot_ar_inference function.
    """
    
    def __init__(self, 
                 save_dir: str,
                 track_every: int = 10,
                 sample_indices: Optional[List[int]] = None,
                 n_samples: int = 100,
                 kde_bandwidth: float = 0.1,
                 contour_levels: Optional[List[float]] = None,
                 max_checkpoints: Optional[int] = None,
                 enabled: bool = True):
        """
        Initialize the AR inference tracker.
        
        Args:
            save_dir: Directory to save checkpoints and tracking data
            track_every: Save checkpoint every N epochs
            sample_indices: Which dataset samples to track for plotting (if None, will use [0, 1, 2])
            n_samples: Number of samples to use for inference
            kde_bandwidth: KDE bandwidth for plotting
            contour_levels: Contour levels for plotting (if None, will use [0.65, 0.95, 0.99])
            max_checkpoints: Maximum number of checkpoints to keep (None = keep all)
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.save_dir = Path(save_dir)
        self.track_every = track_every
        self.sample_indices = sample_indices or [0, 1, 2]
        self.n_samples = n_samples
        self.kde_bandwidth = kde_bandwidth
        self.contour_levels = contour_levels or [0.65, 0.95, 0.99]
        self.max_checkpoints = max_checkpoints
        
        # Create directories
        self.checkpoints_dir = self.save_dir / 'ar_inference_checkpoints'
        self.plots_dir = self.save_dir / 'ar_inference_plots'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking state
        self.tracked_epochs = []
        self.checkpoint_paths = []
        self.training_metrics = []
        self.tracking_metadata = {}
        
        print(f"ARInferenceTracker initialized:")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Track every: {track_every} epochs")
        print(f"  Sample indices: {self.sample_indices}")
        print(f"  Checkpoints dir: {self.checkpoints_dir}")
    
    def should_track(self, epoch: int) -> bool:
        """Check if we should track this epoch."""
        if not self.enabled:
            return False
        return epoch % self.track_every == 0 or epoch == 1
    
    def track_epoch(self, 
                   epoch: int,
                   model: torch.nn.Module,
                   train_loss: float,
                   val_loss: float,
                   optimizer: torch.optim.Optimizer,
                   additional_info: Optional[Dict] = None) -> str:
        """
        Track model state at the current epoch.
        
        Args:
            epoch: Current epoch number
            model: The model to save
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            optimizer: Optimizer state
            additional_info: Additional information to store
            
        Returns:
            Path to saved checkpoint
        """
        if not self.enabled:
            return ""
        
        # Create checkpoint filename
        checkpoint_filename = f'model_epoch_{epoch:04d}.pt'
        checkpoint_path = self.checkpoints_dir / checkpoint_filename
        
        # Save model checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_class': model.__class__.__name__,
            'tracking_metadata': {
                'sample_indices': self.sample_indices,
                'n_samples': self.n_samples,
                'kde_bandwidth': self.kde_bandwidth,
                'contour_levels': self.contour_levels
            }
        }
        
        # Add additional info if provided
        if additional_info:
            checkpoint_data['additional_info'] = additional_info
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update tracking state
        self.tracked_epochs.append(epoch)
        self.checkpoint_paths.append(str(checkpoint_path))
        self.training_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Clean up old checkpoints if max_checkpoints is set
        if self.max_checkpoints and len(self.checkpoint_paths) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_paths.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
            self.tracked_epochs.pop(0)
            self.training_metrics.pop(0)
        
        print(f"  Tracked epoch {epoch}: {checkpoint_filename}")
        return str(checkpoint_path)
    
    def generate_inference_plots(self, 
                                dataset,
                                model_config: Dict,
                                device: str = 'cpu',
                                show_intermediate_steps: bool = True) -> Dict[str, List[str]]:
        """
        Generate AR inference plots for all tracked epochs.
        
        Args:
            dataset: The AR dataset used for training
            model_config: Model configuration to recreate models
            device: Device to run inference on
            show_intermediate_steps: Whether to show intermediate autoregressive steps
            
        Returns:
            Dictionary mapping epochs to generated plot paths
        """
        if not self.enabled or len(self.checkpoint_paths) == 0:
            return {}
        
        from experiments.ar.visualisation.plotting import plot_ar_inference
        
        plot_paths = {}
        
        print(f"\nGenerating AR inference plots for {len(self.checkpoint_paths)} checkpoints...")
        
        for i, (epoch, checkpoint_path) in enumerate(zip(self.tracked_epochs, self.checkpoint_paths)):
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Recreate model
                model = self._recreate_model(model_config, checkpoint, dataset)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                
                # Generate plot
                plot_filename = f'ar_inference_epoch_{epoch:04d}.png'
                plot_path = self.plots_dir / plot_filename
                
                plot_results = plot_ar_inference(
                    model=model,
                    dataset=dataset,
                    sample_indices=self.sample_indices,
                    n_samples=self.n_samples,
                    figsize=(15, 10),
                    save_path=str(plot_path),
                    device=device,
                    show_intermediate_steps=show_intermediate_steps,
                    contour_levels=self.contour_levels,
                    kde_bandwidth=self.kde_bandwidth,
                    random_state=42
                )
                
                plot_paths[epoch] = str(plot_path)
                print(f"  Generated plot for epoch {epoch}: {plot_filename}")
                
            except Exception as e:
                print(f"  Warning: Failed to generate plot for epoch {epoch}: {e}")
                continue
        
        return plot_paths
    
    def _recreate_model(self, model_config: Dict, checkpoint: Dict, dataset=None) -> torch.nn.Module:
        """Recreate model from configuration with proper input/output sizes."""
        model_type = model_config['type']
        
        # Calculate input and output sizes from dataset if available
        input_size = None
        output_size = None
        if dataset is not None:
            data_info = dataset.get_data_info()
            input_size = data_info['dimension']  # Single timestep input
            output_size = data_info['dimension']  # Single timestep output (for autoregressive)
        
        if model_type == 'fgn_encoder':
            from models.fgn_encoder_sampler import FGNEncoderSampler
            config = model_config['fgn_encoder'].copy()
            # Set sizes if they're None in config
            if config.get('input_size') is None and input_size is not None:
                config['input_size'] = input_size
            if config.get('output_size') is None and output_size is not None:
                config['output_size'] = output_size
            return FGNEncoderSampler(**config)
        
        elif model_type == 'mlp_sampler':
            from models.mlp_crps_sampler import MLPSampler
            config = model_config['mlp_sampler'].copy()
            # Set sizes if they're None in config
            if config.get('input_size') is None and input_size is not None:
                config['input_size'] = input_size
            if config.get('output_size') is None and output_size is not None:
                config['output_size'] = output_size
            return MLPSampler(**config)
        
        elif model_type == 'affine_normal':
            from models.affine_normal import SimpleAffineNormal
            config = model_config['affine_normal'].copy()
            # Set output_dim if it's None in config
            if config.get('output_dim') is None and output_size is not None:
                config['output_dim'] = output_size
            return SimpleAffineNormal(**config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tracking information for saving to results.json.
        
        Returns:
            Dictionary containing tracking summary
        """
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'track_every': self.track_every,
            'sample_indices': self.sample_indices,
            'n_samples': self.n_samples,
            'kde_bandwidth': self.kde_bandwidth,
            'contour_levels': self.contour_levels,
            'max_checkpoints': self.max_checkpoints,
            'tracked_epochs': self.tracked_epochs.copy(),
            'checkpoint_paths': self.checkpoint_paths.copy(),
            'training_metrics': self.training_metrics.copy(),
            'checkpoints_dir': str(self.checkpoints_dir),
            'plots_dir': str(self.plots_dir),
            'total_checkpoints': len(self.checkpoint_paths)
        }
    
    def load_checkpoint(self, epoch: int, device: str = 'cpu') -> Optional[Dict]:
        """
        Load a specific checkpoint by epoch.
        
        Args:
            epoch: Epoch number to load
            device: Device to load to
            
        Returns:
            Loaded checkpoint data or None if not found
        """
        if not self.enabled or epoch not in self.tracked_epochs:
            return None
        
        idx = self.tracked_epochs.index(epoch)
        checkpoint_path = self.checkpoint_paths[idx]
        
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path, map_location=device)
        else:
            print(f"Warning: Checkpoint for epoch {epoch} not found at {checkpoint_path}")
            return None
    
    def cleanup_checkpoints(self, keep_epochs: Optional[List[int]] = None):
        """
        Clean up checkpoints, optionally keeping only specific epochs.
        
        Args:
            keep_epochs: List of epochs to keep (if None, keeps all)
        """
        if not self.enabled:
            return
        
        if keep_epochs is None:
            return
        
        # Find checkpoints to remove
        checkpoints_to_remove = []
        for i, epoch in enumerate(self.tracked_epochs):
            if epoch not in keep_epochs:
                checkpoints_to_remove.append(i)
        
        # Remove checkpoints in reverse order to maintain indices
        for i in reversed(checkpoints_to_remove):
            checkpoint_path = self.checkpoint_paths[i]
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            # Remove from tracking lists
            self.tracked_epochs.pop(i)
            self.checkpoint_paths.pop(i)
            self.training_metrics.pop(i)
        
        print(f"Cleaned up {len(checkpoints_to_remove)} checkpoints, keeping {len(self.tracked_epochs)}")
    
    def create_inference_animation(self, 
                                  dataset,
                                  model_config: Dict,
                                  output_path: str,
                                  device: str = 'cpu',
                                  fps: int = 2) -> str:
        """
        Create an animation showing inference evolution during training.
        
        Args:
            dataset: The AR dataset used for training
            model_config: Model configuration
            output_path: Path for output GIF/MP4
            device: Device to run inference on
            fps: Frames per second for animation
            
        Returns:
            Path to created animation
        """
        if not self.enabled or len(self.checkpoint_paths) == 0:
            return ""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from experiments.ar.visualisation.plotting import plot_ar_inference
            
            # Generate all plots first (without showing)
            plot_data = []
            
            for epoch, checkpoint_path in zip(self.tracked_epochs, self.checkpoint_paths):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model = self._recreate_model(model_config, checkpoint, dataset)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                
                # Generate plot data (don't save individual plots)
                results = plot_ar_inference(
                    model=model,
                    dataset=dataset,
                    sample_indices=self.sample_indices[:1],  # Just first sample for animation
                    n_samples=self.n_samples,
                    figsize=(10, 6),
                    save_path=None,  # Don't save individual plots
                    device=device,
                    show_intermediate_steps=False,
                    contour_levels=self.contour_levels,
                    kde_bandwidth=self.kde_bandwidth,
                    random_state=42
                )
                
                plot_data.append((epoch, results))
                plt.close('all')  # Close any open plots
            
            # Create animation
            # This would require more complex matplotlib animation code
            # For now, return a message about the capability
            print(f"Animation creation not implemented yet. Would create: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to create animation: {e}")
            return ""


def create_tracking_config(track_every: int = 10,
                          sample_indices: Optional[List[int]] = None,
                          n_samples: int = 100,
                          kde_bandwidth: float = 0.1,
                          contour_levels: Optional[List[float]] = None,
                          max_checkpoints: Optional[int] = None,
                          enabled: bool = True,
                          generate_plots_after_training: bool = True,
                          create_animation: bool = False) -> Dict[str, Any]:
    """
    Create a tracking configuration dictionary.
    
    Args:
        track_every: Save checkpoint every N epochs
        sample_indices: Dataset sample indices to track
        n_samples: Number of samples for inference
        kde_bandwidth: KDE bandwidth for plotting
        contour_levels: Contour levels for plotting
        max_checkpoints: Maximum checkpoints to keep
        enabled: Whether tracking is enabled
        generate_plots_after_training: Generate all plots after training completes
        create_animation: Create animation showing training progression
        
    Returns:
        Tracking configuration dictionary
    """
    return {
        'enabled': enabled,
        'track_every': track_every,
        'sample_indices': sample_indices or [0, 1, 2],
        'n_samples': n_samples,
        'kde_bandwidth': kde_bandwidth,
        'contour_levels': contour_levels or [0.65, 0.95, 0.99],
        'max_checkpoints': max_checkpoints,
        'generate_plots_after_training': generate_plots_after_training,
        'create_animation': create_animation
    }
