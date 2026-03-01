"""
Smart Checkpoint Management System

Implements intelligent checkpoint saving with configurable strategies:
- Best model tracking (by validation loss/accuracy)
- Milestone checkpoints (every N epochs)
- Last N checkpoints retention
- Automatic cleanup of old checkpoints

Based on literature best practices for reproducible ML experiments.
"""

import torch
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import glob


class SmartCheckpointManager:
    """
    Intelligent checkpoint management with multiple strategies.
    
    Strategies:
    - 'best': Keep only the best performing checkpoint
    - 'milestone': Save every N epochs + best
    - 'smart': Best + last N + milestones (recommended)
    - 'all': Save all checkpoints (not recommended for long training)
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        strategy: str = 'smart',
        keep_best_n: int = 3,
        keep_last_n: int = 2,
        milestone_every: int = 5,
        metric_name: str = 'val_loss',
        metric_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategy = strategy
        self.keep_best_n = keep_best_n
        self.keep_last_n = keep_last_n
        self.milestone_every = milestone_every
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Tracking
        self.best_checkpoints = []  # List of (metric_value, checkpoint_path)
        self.last_checkpoints = []  # List of checkpoint_paths
        self.milestone_checkpoints = []  # List of checkpoint_paths
        
        # Metadata
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.load_metadata()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_metadata(self):
        """Load existing checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.best_checkpoints = metadata.get('best_checkpoints', [])
                    self.last_checkpoints = metadata.get('last_checkpoints', [])
                    self.milestone_checkpoints = metadata.get('milestone_checkpoints', [])
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint metadata: {e}")
                self.best_checkpoints = []
                self.last_checkpoints = []
                self.milestone_checkpoints = []
    
    def save_metadata(self):
        """Save checkpoint metadata."""
        metadata = {
            'best_checkpoints': self.best_checkpoints,
            'last_checkpoints': self.last_checkpoints,
            'milestone_checkpoints': self.milestone_checkpoints,
            'strategy': self.strategy,
            'metric_name': self.metric_name,
            'metric_mode': self.metric_mode,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def should_save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved based on strategy."""
        if self.strategy == 'all':
            return True
        
        # Always save if it's a milestone
        if epoch % self.milestone_every == 0:
            return True
        
        # Check if it's a new best
        if self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._is_better_metric(metric_value):
                return True
        
        # Always save last checkpoint for 'smart' strategy
        if self.strategy in ['smart', 'milestone']:
            return True
        
        return False
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        additional_state: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save checkpoint with intelligent management.
        
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        if not self.should_save_checkpoint(epoch, metrics):
            return None
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'strategy': self.strategy
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_state:
            checkpoint_data.update(additional_state)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update tracking based on strategy
        self._update_checkpoint_tracking(checkpoint_path, epoch, metrics)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save metadata
        self.save_metadata()
        
        return checkpoint_path
    
    def _is_better_metric(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        if not self.best_checkpoints:
            return True
        
        current_best = self.best_checkpoints[0][0]  # First element is metric value
        
        if self.metric_mode == 'min':
            return metric_value < current_best
        else:  # 'max'
            return metric_value > current_best
    
    def _update_checkpoint_tracking(
        self, 
        checkpoint_path: Path, 
        epoch: int, 
        metrics: Dict[str, float]
    ):
        """Update checkpoint tracking lists."""
        checkpoint_path_str = str(checkpoint_path)
        
        # Update last checkpoints
        self.last_checkpoints.append(checkpoint_path_str)
        if len(self.last_checkpoints) > self.keep_last_n:
            self.last_checkpoints = self.last_checkpoints[-self.keep_last_n:]
        
        # Update milestone checkpoints
        if epoch % self.milestone_every == 0:
            self.milestone_checkpoints.append(checkpoint_path_str)
        
        # Update best checkpoints
        if self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            self.best_checkpoints.append((metric_value, checkpoint_path_str))
            
            # Sort by metric value
            reverse_sort = (self.metric_mode == 'max')
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse_sort)
            
            # Keep only best N
            if len(self.best_checkpoints) > self.keep_best_n:
                self.best_checkpoints = self.best_checkpoints[:self.keep_best_n]
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on strategy."""
        if self.strategy == 'all':
            return  # Don't cleanup if keeping all
        
        # Get all checkpoint files
        all_checkpoints = set(glob.glob(str(self.checkpoint_dir / "checkpoint_*.pt")))
        
        # Get checkpoints to keep
        keep_checkpoints = set()
        
        # Keep best checkpoints
        for _, checkpoint_path in self.best_checkpoints:
            keep_checkpoints.add(checkpoint_path)
        
        # Keep last checkpoints
        for checkpoint_path in self.last_checkpoints:
            keep_checkpoints.add(checkpoint_path)
        
        # Keep milestone checkpoints
        for checkpoint_path in self.milestone_checkpoints:
            keep_checkpoints.add(checkpoint_path)
        
        # Remove checkpoints not in keep list
        for checkpoint_path in all_checkpoints:
            if checkpoint_path not in keep_checkpoints:
                try:
                    Path(checkpoint_path).unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    self.logger.warning(f"Could not remove checkpoint {checkpoint_path}: {e}")
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.best_checkpoints:
            return None
        return Path(self.best_checkpoints[0][1])
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.last_checkpoints:
            return None
        return Path(self.last_checkpoints[-1])
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False
    ) -> Optional[Dict]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load best checkpoint if checkpoint_path is None
            
        Returns:
            Checkpoint data dictionary or None if not found
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.get_best_checkpoint()
            else:
                checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            return None
        
        try:
            # Use weights_only=False for our own trusted checkpoints.
            # PyTorch 2.6 defaults to weights_only=True which blocks numpy types.
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Could not load checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of managed checkpoints."""
        return {
            'strategy': self.strategy,
            'total_checkpoints': len(glob.glob(str(self.checkpoint_dir / "checkpoint_*.pt"))),
            'best_checkpoints': len(self.best_checkpoints),
            'last_checkpoints': len(self.last_checkpoints),
            'milestone_checkpoints': len(self.milestone_checkpoints),
            'best_metric_value': self.best_checkpoints[0][0] if self.best_checkpoints else None,
            'checkpoint_dir': str(self.checkpoint_dir)
        }


def create_checkpoint_manager(
    checkpoint_dir: Path,
    strategy: str = 'smart',
    **kwargs
) -> SmartCheckpointManager:
    """
    Factory function to create checkpoint manager with sensible defaults.
    
    Strategies:
    - 'smart': Best 3 + Last 2 + Every 5 epochs (recommended)
    - 'minimal': Best 1 + Last 1
    - 'research': Best 5 + Last 3 + Every 3 epochs
    """
    if strategy == 'minimal':
        kwargs.setdefault('keep_best_n', 1)
        kwargs.setdefault('keep_last_n', 1)
        kwargs.setdefault('milestone_every', 10)
    elif strategy == 'research':
        kwargs.setdefault('keep_best_n', 5)
        kwargs.setdefault('keep_last_n', 3)
        kwargs.setdefault('milestone_every', 3)
    else:  # 'smart' or default
        kwargs.setdefault('keep_best_n', 3)
        kwargs.setdefault('keep_last_n', 2)
        kwargs.setdefault('milestone_every', 5)
    
    return SmartCheckpointManager(checkpoint_dir, strategy, **kwargs)
