"""
Base Trainer

Core training functionality shared across all trainer implementations.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import numpy as np

from .checkpoints.manager import CheckpointManager
from .checkpoints.strategies import SmartStrategy


class BaseTrainer:
    """
    Base trainer class providing core training functionality.
    
    This class implements the basic training loop, validation, and testing procedures
    that are common across different training scenarios.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_strategy: str = "smart",
        gradient_clip: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        **checkpoint_kwargs
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer instance
            device: Device to train on ('cuda' or 'cpu')
            lr_scheduler: Learning rate scheduler (optional)
            metrics: Dictionary of metric functions (name -> function)
            log_dir: Directory to save logs (default: 'runs/experiment_<timestamp>')
            checkpoint_dir: Directory to save checkpoints (default: log_dir/checkpoints)
            checkpoint_strategy: Checkpointing strategy ('best', 'last', 'smart', 'all')
            gradient_clip: Maximum gradient norm for clipping (optional)
            early_stopping_patience: Number of epochs to wait before early stopping (optional)
            **checkpoint_kwargs: Additional arguments for checkpoint manager
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize metrics
        self.metrics = metrics or {}
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path(f"runs/experiment_{int(time.time())}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            strategy=checkpoint_strategy,
            **checkpoint_kwargs
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.epochs_without_improvement = 0
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the trainer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to {log_file}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics}
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for name, metric_fn in self.metrics.items():
                epoch_metrics[name] += metric_fn(output, target).item()
            
            # Update global step
            self.global_step += 1
        
        # Average metrics
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {
            name: value / num_batches
            for name, value in epoch_metrics.items()
        }
        
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def evaluate(
        self,
        data_loader: DataLoader,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.
        
        Args:
            data_loader: DataLoader for evaluation data
            prefix: Prefix for metric names (e.g., 'val' or 'test')
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        metrics = {name: 0.0 for name in self.metrics}
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                for name, metric_fn in self.metrics.items():
                    metrics[name] += metric_fn(output, target).item()
        
        # Average metrics
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {
            f"{prefix}_{name}": value / num_batches
            for name, value in metrics.items()
        }
        
        return {
            f"{prefix}_loss": avg_loss,
            **avg_metrics
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        eval_every: int = 1,
        save_every: int = 1,
        checkpoint_metric: str = 'val_loss',
        checkpoint_mode: str = 'min',
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            epochs: Maximum number of epochs to train for
            eval_every: Evaluate on validation set every N epochs
            save_every: Save checkpoint every N epochs
            checkpoint_metric: Metric to use for checkpointing
            checkpoint_mode: 'min' or 'max' (minimize or maximize the metric)
            **kwargs: Additional arguments for the train_epoch method
            
        Returns:
            Dictionary containing training history and best model state
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch_times': []
        }
        
        # Add metrics to history
        for name in self.metrics:
            history[f'train_{name}'] = []
            if val_loader is not None:
                history[f'val_{name}'] = []
        
        best_metric = float('inf') if checkpoint_mode == 'min' else -float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, **kwargs)
            
            # Update learning rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(train_metrics['loss'])
                else:
                    self.lr_scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log training metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"LR: {current_lr:.6f}"
            )
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['lr'].append(current_lr)
            for name in self.metrics:
                history[f'train_{name}'].append(train_metrics[name])
            
            # Validation phase
            if val_loader is not None and (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(val_loader, prefix='val')
                
                # Log validation metrics
                log_msg = f"Epoch {epoch + 1}/{epochs} - "
                log_msg += f"Val Loss: {val_metrics['val_loss']:.4f}"
                for name in self.metrics:
                    log_msg += f" - Val {name}: {val_metrics[f'val_{name}']:.4f}"
                self.logger.info(log_msg)
                
                # Update history
                history['val_loss'].append(val_metrics['val_loss'])
                for name in self.metrics:
                    history[f'val_{name}'].append(val_metrics[f'val_{name}'])
                
                # Checkpoint if this is the best model so far
                current_metric = val_metrics[checkpoint_metric]
                is_best = (
                    (checkpoint_mode == 'min' and current_metric < best_metric) or
                    (checkpoint_mode == 'max' and current_metric > best_metric)
                )
                
                if is_best:
                    best_metric = current_metric
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self.checkpoint_manager.save_checkpoint(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'metrics': val_metrics,
                            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                            'global_step': self.global_step,
                            'best_metric': best_metric,
                            'checkpoint_metric': checkpoint_metric,
                            'checkpoint_mode': checkpoint_mode
                        },
                        is_best=True,
                        filename=f'model_best.pth'
                    )
                    
                    self.logger.info(f"New best {checkpoint_metric}: {best_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.checkpoint_manager.save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics if val_loader is not None else train_metrics,
                        'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                        'global_step': self.global_step,
                        'best_metric': best_metric,
                        'checkpoint_metric': checkpoint_metric,
                        'checkpoint_mode': checkpoint_mode
                    },
                    is_best=False,
                    filename=f'checkpoint_epoch_{epoch + 1}.pth'
                )
            
            # Update epoch time
            epoch_time = time.time() - start_time
            history['epoch_times'].append(epoch_time)
            
            # Early stopping
            if (self.early_stopping_patience is not None and 
                self.epochs_without_improvement >= self.early_stopping_patience):
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Final evaluation on test set
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader, prefix='test')
            self.logger.info(f"Test Loss: {test_metrics['test_loss']:.4f}")
            for name in self.metrics:
                self.logger.info(f"Test {name}: {test_metrics[f'test_{name}']:.4f}")
            
            history['test_metrics'] = test_metrics
        
        # Save final model
        self.checkpoint_manager.save_checkpoint(
            {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': test_metrics if test_loader is not None else (val_metrics if val_loader is not None else train_metrics),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'global_step': self.global_step,
                'best_metric': best_metric,
                'checkpoint_metric': checkpoint_metric,
                'checkpoint_mode': checkpoint_mode
            },
            is_best=False,
            filename='model_final.pth'
        )
        
        return {
            'history': history,
            'best_metric': best_metric,
            'best_model_path': self.checkpoint_manager.get_best_checkpoint_path()
        }
