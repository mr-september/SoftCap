"""
Checkpoint Manager

This module provides a unified interface for managing model checkpoints
during training, with support for different checkpointing strategies.
"""

import os
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type, Tuple
import json
import logging

from .strategies import (
    CheckpointStrategy,
    BestNStrategy,
    LastNStrategy,
    SmartStrategy,
    ResearchStrategy
)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints with different strategies.
    """
    
    STRATEGIES = {
        'best': BestNStrategy,
        'last': LastNStrategy,
        'smart': SmartStrategy,
        'research': ResearchStrategy
    }
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        strategy: Union[str, Type[CheckpointStrategy]] = 'smart',
        **strategy_kwargs
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            strategy: Checkpoint strategy name or class
                     ('best', 'last', 'smart', 'research', or a custom class)
            **strategy_kwargs: Additional arguments for the strategy
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy
        if isinstance(strategy, str):
            if strategy not in self.STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {strategy}. "
                    f"Available strategies: {list(self.STRATEGIES.keys())}"
                )
            strategy_class = self.STRATEGIES[strategy]
        elif isinstance(strategy, type) and issubclass(strategy, CheckpointStrategy):
            strategy_class = strategy
        else:
            raise ValueError(
                "strategy must be a string or a CheckpointStrategy subclass"
            )
        
        self.strategy = strategy_class(
            checkpoint_dir=self.checkpoint_dir,
            **strategy_kwargs
        )
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track best checkpoint
        self.best_metric = None
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool = False,
        filename: str = 'checkpoint.pth',
        **kwargs
    ) -> List[str]:
        """
        Save a checkpoint using the current strategy.
        
        Args:
            state: Model state dictionary to save
            is_best: Whether this is the best model so far
            filename: Base filename for the checkpoint
            **kwargs: Additional arguments for the strategy
            
        Returns:
            List of paths to saved checkpoints
        """
        # Add metadata
        state['is_best'] = is_best
        
        # Save the checkpoint using the strategy
        saved_paths = self.strategy.save_checkpoint(
            state=state,
            is_best=is_best,
            filename=filename,
            **kwargs
        )
        
        # Update best checkpoint info
        if is_best:
            self.best_metric = state.get('metrics', {}).get(
                getattr(self.strategy, 'metric_name', 'val_loss'),
                None
            )
            self.best_checkpoint_path = saved_paths[0] if saved_paths else None
            
            # Save best checkpoint info to a JSON file
            best_info = {
                'path': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
                'metric': self.best_metric,
                'epoch': state.get('epoch', 0),
                'global_step': state.get('global_step', 0),
                'timestamp': state.get('timestamp', None)
            }
            
            with open(self.checkpoint_dir / 'best_checkpoint.json', 'w') as f:
                json.dump(best_info, f, indent=2)
        
        return saved_paths
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        load_best: bool = False,
        load_latest: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file. If None, loads the best or latest.
            map_location: Device to map the checkpoint to (default: None, same as saved)
            load_best: If True, load the best checkpoint (overrides checkpoint_path)
            load_latest: If True, load the most recent checkpoint (overrides checkpoint_path)
            **kwargs: Additional arguments for torch.load()
            
        Returns:
            Dictionary containing the checkpoint state
        """
        if load_best and load_latest:
            raise ValueError("Only one of load_best or load_latest can be True")
        
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.get_best_checkpoint_path()
                if checkpoint_path is None:
                    raise FileNotFoundError("No best checkpoint found")
            elif load_latest:
                checkpoints = self.get_checkpoints()
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = checkpoints[0][0]
            else:
                raise ValueError(
                    "Must provide checkpoint_path or set load_best/load_latest"
                )
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, **kwargs)
        
        # Update best checkpoint info if this is the best
        if checkpoint.get('is_best', False):
            self.best_metric = checkpoint.get('metrics', {}).get(
                getattr(self.strategy, 'metric_name', 'val_loss'),
                None
            )
            self.best_checkpoint_path = checkpoint_path
        
        return checkpoint
    
    def get_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """
        Get all available checkpoints.
        
        Returns:
            List of (path, metadata) tuples, sorted by preference for keeping
        """
        return self.strategy.get_checkpoints()
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """
        Get the path to the best checkpoint.
        
        Returns:
            Path to the best checkpoint, or None if not found
        """
        # First try to get from memory
        if self.best_checkpoint_path is not None and self.best_checkpoint_path.exists():
            return self.best_checkpoint_path
        
        # Then try to load from JSON file
        best_info_path = self.checkpoint_dir / 'best_checkpoint.json'
        if best_info_path.exists():
            try:
                with open(best_info_path, 'r') as f:
                    best_info = json.load(f)
                best_path = Path(best_info.get('path', ''))
                if best_path.exists():
                    self.best_checkpoint_path = best_path
                    self.best_metric = best_info.get('metric')
                    return best_path
            except Exception as e:
                self.logger.warning(f"Error loading best checkpoint info: {e}")
        
        # Finally, try to find the best checkpoint by scanning the directory
        checkpoints = self.get_checkpoints()
        for path, metadata in checkpoints:
            if metadata.get('is_best', False):
                self.best_checkpoint_path = path
                self.best_metric = metadata.get('metric')
                return path
        
        return None
    
    def cleanup(self, keep: int = 0) -> List[str]:
        """
        Clean up old checkpoints.
        
        Args:
            keep: Number of checkpoints to keep
            
        Returns:
            List of paths to deleted checkpoints
        """
        return self.strategy.cleanup(keep)
    
    def get_best_metric(self) -> Optional[float]:
        """
        Get the best metric value.
        
        Returns:
            Best metric value, or None if not available
        """
        if self.best_metric is None:
            self.get_best_checkpoint_path()  # Try to load best checkpoint info
        return self.best_metric
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(checkpoint_dir='{self.checkpoint_dir}', strategy={self.strategy.__class__.__name__})"
