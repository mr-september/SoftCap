"""
Checkpoint Strategies

This module implements different strategies for managing model checkpoints during training.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import glob
import re


class CheckpointStrategy:
    """Base class for checkpoint strategies."""
    
    def __init__(self, checkpoint_dir: Path, **kwargs):
        """
        Initialize the checkpoint strategy.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            **kwargs: Additional strategy-specific arguments
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """Get the full path for a checkpoint file."""
        return self.checkpoint_dir / filename
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str,
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint.
        
        Args:
            state: Model state dictionary
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments for the strategy
            
        Returns:
            List of paths to saved checkpoints
        """
        raise NotImplementedError
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all checkpoints, sorted by some criteria.
        
        Returns:
            List of (path, metadata) tuples, sorted by preference for keeping
        """
        raise NotImplementedError
    
    def cleanup(self, keep: int = 0) -> List[str]:
        """
        Clean up old checkpoints.
        
        Args:
            keep: Number of checkpoints to keep
            
        Returns:
            List of paths to deleted checkpoints
        """
        checkpoints = self.get_checkpoints()
        
        if len(checkpoints) <= keep:
            return []
        
        to_delete = checkpoints[keep:]
        deleted = []
        
        for path, _ in to_delete:
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    deleted.append(str(path))
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        
        return deleted


class BestNStrategy(CheckpointStrategy):
    """Keep the N best checkpoints based on a metric."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        keep_best: int = 3,
        **kwargs
    ):
        """
        Initialize the BestNStrategy.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Name of the metric to use for comparison
            mode: 'min' or 'max' (minimize or maximize the metric)
            keep_best: Number of best checkpoints to keep
            **kwargs: Additional arguments for the base class
        """
        super().__init__(checkpoint_dir, **kwargs)
        self.metric_name = metric_name
        self.mode = mode
        self.keep_best = keep_best
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str,
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint.
        
        Args:
            state: Model state dictionary
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments
            
        Returns:
            List of paths to saved checkpoints
        """
        if not is_best:
            return []
        
        # Add metadata for sorting
        metric = state.get('metrics', {}).get(self.metric_name)
        if metric is None:
            raise ValueError(f"Metric '{self.metric_name}' not found in state")
        
        # Save the checkpoint
        path = self.get_checkpoint_path(filename)
        torch.save(state, path)
        
        # Clean up old checkpoints
        self.cleanup(self.keep_best)
        
        return [str(path)]
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all checkpoints, sorted by the metric.
        
        Returns:
            List of (path, metadata) tuples, sorted by the metric
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob('*.pth'):
            try:
                state = torch.load(path, map_location='cpu')
                metric = state.get('metrics', {}).get(self.metric_name)
                if metric is not None:
                    checkpoints.append((path, {
                        'metric': metric,
                        'epoch': state.get('epoch', 0),
                        'step': state.get('global_step', 0),
                        'is_best': state.get('is_best', False)
                    }))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Sort by metric (ascending for 'min', descending for 'max')
        reverse = self.mode == 'max'
        checkpoints.sort(key=lambda x: x[1]['metric'], reverse=reverse)
        
        return checkpoints


class LastNStrategy(CheckpointStrategy):
    """Keep the N most recent checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 5,
        **kwargs
    ):
        """
        Initialize the LastNStrategy.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last: Number of most recent checkpoints to keep
            **kwargs: Additional arguments for the base class
        """
        super().__init__(checkpoint_dir, **kwargs)
        self.keep_last = keep_last
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str,
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint.
        
        Args:
            state: Model state dictionary
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments
            
        Returns:
            List of paths to saved checkpoints
        """
        path = self.get_checkpoint_path(filename)
        torch.save(state, path)
        
        # Clean up old checkpoints
        self.cleanup(self.keep_last)
        
        return [str(path)]
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all checkpoints, sorted by modification time (newest first).
        
        Returns:
            List of (path, metadata) tuples, sorted by modification time
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob('*.pth'):
            try:
                state = torch.load(path, map_location='cpu')
                mtime = path.stat().st_mtime
                checkpoints.append((path, {
                    'mtime': mtime,
                    'epoch': state.get('epoch', 0),
                    'step': state.get('global_step', 0),
                    'is_best': state.get('is_best', False)
                }))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1]['mtime'], reverse=True)
        
        return checkpoints


class SmartStrategy(CheckpointStrategy):
    """
    Smart checkpointing strategy that combines best and recent checkpoints.
    
    Keeps:
    - The N best checkpoints based on a metric
    - The M most recent checkpoints
    - Milestone checkpoints (every K epochs)
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        keep_best: int = 3,
        keep_last: int = 2,
        milestone_every: int = 5,
        **kwargs
    ):
        """
        Initialize the SmartStrategy.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Name of the metric to use for comparison
            mode: 'min' or 'max' (minimize or maximize the metric)
            keep_best: Number of best checkpoints to keep
            keep_last: Number of most recent checkpoints to keep
            milestone_every: Save a checkpoint every N epochs
            **kwargs: Additional arguments for the base class
        """
        super().__init__(checkpoint_dir, **kwargs)
        self.metric_name = metric_name
        self.mode = mode
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.milestone_every = milestone_every
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str,
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint.
        
        Args:
            state: Model state dictionary
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments
                - is_milestone: Whether this is a milestone checkpoint
                
        Returns:
            List of paths to saved checkpoints
        """
        is_milestone = kwargs.get('is_milestone', False)
        
        # Always save if it's the best, milestone, or final checkpoint
        if not (is_best or is_milestone or filename == 'model_final.pth'):
            return []
        
        # Add metadata for sorting
        metric = state.get('metrics', {}).get(self.metric_name)
        if metric is None and is_best:
            raise ValueError(f"Metric '{self.metric_name}' not found in state")
        
        # Save the checkpoint
        path = self.get_checkpoint_path(filename)
        torch.save(state, path)
        
        # Clean up old checkpoints
        self._cleanup()
        
        return [str(path)]
    
    def _cleanup(self):
        """Clean up old checkpoints based on the strategy."""
        # Get all checkpoints
        all_checkpoints = []
        
        for path in self.checkpoint_dir.glob('*.pth'):
            try:
                state = torch.load(path, map_location='cpu')
                is_best = state.get('is_best', False)
                is_final = path.name == 'model_final.pth'
                is_milestone = (
                    'epoch' in state and 
                    state['epoch'] % self.milestone_every == 0
                )
                
                all_checkpoints.append({
                    'path': path,
                    'is_best': is_best,
                    'is_final': is_final,
                    'is_milestone': is_milestone,
                    'metric': state.get('metrics', {}).get(self.metric_name, 0),
                    'mtime': path.stat().st_mtime,
                    'epoch': state.get('epoch', 0)
                })
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not all_checkpoints:
            return
        
        # Sort by metric (for best checkpoints) and mtime (for recent checkpoints)
        reverse_metric = self.mode == 'max'
        all_checkpoints.sort(key=lambda x: x['metric'], reverse=reverse_metric)
        
        # Keep best checkpoints
        best_checkpoints = set()
        best = sorted(
            [c for c in all_checkpoints if c['is_best']],
            key=lambda x: x['metric'],
            reverse=reverse_metric
        )[:self.keep_best]
        
        # Keep final checkpoint
        final = [c for c in all_checkpoints if c['is_final']]
        
        # Keep milestone checkpoints
        milestones = sorted(
            [c for c in all_checkpoints if c['is_milestone']],
            key=lambda x: x['epoch'],
            reverse=True
        )
        
        # Keep most recent checkpoints
        recent = sorted(
            all_checkpoints,
            key=lambda x: x['mtime'],
            reverse=True
        )[:self.keep_last]
        
        # Combine all checkpoints to keep
        keep_paths = set()
        for checkpoint in best + final + milestones + recent:
            keep_paths.add(checkpoint['path'])
        
        # Delete all other checkpoints
        for checkpoint in all_checkpoints:
            if checkpoint['path'] not in keep_paths:
                try:
                    checkpoint['path'].unlink()
                except Exception as e:
                    print(f"Error deleting {checkpoint['path']}: {e}")
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all checkpoints, sorted by preference for keeping.
        
        Returns:
            List of (path, metadata) tuples
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob('*.pth'):
            try:
                state = torch.load(path, map_location='cpu')
                checkpoints.append((path, {
                    'is_best': state.get('is_best', False),
                    'is_final': path.name == 'model_final.pth',
                    'metric': state.get('metrics', {}).get(self.metric_name, 0),
                    'epoch': state.get('epoch', 0),
                    'step': state.get('global_step', 0),
                    'mtime': path.stat().st_mtime
                }))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Sort by: best first, then final, then milestone, then recent
        reverse_metric = self.mode == 'max'
        
        def sort_key(x):
            path, meta = x
            return (
                not meta['is_best'],  # Best checkpoints first
                not meta['is_final'],  # Then final checkpoint
                not (meta['epoch'] % self.milestone_every == 0),  # Then milestones
                -meta['mtime']  # Then most recent
            )
        
        checkpoints.sort(key=sort_key)
        
        return checkpoints


class ResearchStrategy(CheckpointStrategy):
    """Keep all checkpoints for research purposes."""
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str,
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint.
        
        Args:
            state: Model state dictionary
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments
            
        Returns:
            List of paths to saved checkpoints
        """
        # Add timestamp to filename to avoid overwriting
        import time
        timestamp = int(time.time())
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        path = self.get_checkpoint_path(unique_filename)
        torch.save(state, path)
        
        return [str(path)]
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all checkpoints, sorted by modification time (newest first).
        
        Returns:
            List of (path, metadata) tuples
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob('*.pth'):
            try:
                state = torch.load(path, map_location='cpu')
                checkpoints.append((path, {
                    'mtime': path.stat().st_mtime,
                    'epoch': state.get('epoch', 0),
                    'step': state.get('global_step', 0),
                    'is_best': state.get('is_best', False)
                }))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1]['mtime'], reverse=True)
        
        return checkpoints
