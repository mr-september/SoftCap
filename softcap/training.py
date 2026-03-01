"""
SoftCap Training Module

Enhanced training capabilities with comprehensive metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
from collections import defaultdict
import glob
import shutil
from scipy import stats

class EnhancedTrainer:
    """
    Enhanced trainer with comprehensive metrics tracking and analysis.
    
    Integrates all literature-informed metrics during training.
    """
    
    def __init__(
        self,
        activation: nn.Module,
        device: str = "cuda",
        output_dir: Optional[Path] = None,
        enable_checkpoints: bool = True,
        checkpoint_frequency: int = 1  # Save every N epochs
    ):
        self.activation = activation
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("./training_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint setup
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model = None
        self.best_accuracy = 0.0
        self.training_history = {}
        self.batch_history = []  # For batch-level tracking
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        is_best: bool = False
    ) -> str:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'training_history': self.training_history,
            'batch_history': self.batch_history,
            'best_accuracy': self.best_accuracy,
            'activation_name': self.activation.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_history = checkpoint.get('training_history', {})
        self.batch_history = checkpoint.get('batch_history', [])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoint_files[-1])
    
    def load_best_checkpoint(self, checkpoint_dir: Path) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load the best trained model and its training history from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            Tuple of (loaded_model, training_history)
        """
        best_checkpoint_path = checkpoint_dir / "best_model.pt"
        
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(f"Best model checkpoint not found: {best_checkpoint_path}")
        
        # Load checkpoint with PyTorch 2.6 compatibility
        # Use weights_only=False since we trust our own checkpoints
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model with same activation
        from .models import get_model_for_analysis
        model = get_model_for_analysis('standard_mlp', self.activation)
        model = model.to(self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Extract training history
        training_history = checkpoint.get('training_history', {})
        
        print(f"✓ Best model loaded from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")
        
        return model, training_history
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a model with comprehensive metrics tracking.
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training results and metrics
        """
        model = model.to(self.device)
        
        # Setup optimizer and criterion
        optimizer = self._create_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        scheduler = self._create_scheduler(optimizer, config)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'gradient_norms': [],
            'learning_rates': [],
            'batch_losses': []  # New: batch-level loss tracking
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            checkpoint = self.load_checkpoint(resume_from_checkpoint, model, optimizer)
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_accuracy', 0.0)
            history = checkpoint.get('training_history', history)
            print(f"Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, config.get('epochs', 10)):
            # Training phase with batch-level tracking
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, config, epoch
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(
                model, val_loader, criterion
            )
            
            # Update learning rate
            if scheduler:
                scheduler.step(val_metrics['loss'])
            
            # Record metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['gradient_norms'].append(train_metrics.get('gradient_norm', 0))
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Add batch losses to history
            if 'batch_losses' in train_metrics:
                history['batch_losses'].extend(train_metrics['batch_losses'])
            
            # Check for improvement
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
                self.best_model = model.state_dict().copy()
                self.best_accuracy = best_val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if self.enable_checkpoints and (epoch + 1) % self.checkpoint_frequency == 0:
                self.training_history = history  # Update before saving
                self.save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'], 
                    val_metrics['accuracy'], is_best
                )
            
            # Early stopping
            if patience_counter >= config.get('early_stopping_patience', 5):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        
        # Load best model
        if self.best_model:
            model.load_state_dict(self.best_model)
        
        self.training_history = history
        
        return {
            'history': history,
            'best_accuracy': best_val_acc,
            'final_epoch': epoch + 1,
            'model_state': self.best_model
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        epoch: int = 0
    ) -> Dict[str, Any]:
        """Train for one epoch with batch-level tracking."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        batch_losses = []  # Track individual batch losses
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Record batch loss with metadata
            batch_loss_info = {
                'epoch': epoch,
                'batch': batch_idx,
                'loss': loss.item(),
                'timestamp': datetime.now().isoformat()
            }
            batch_losses.append(batch_loss_info)
            self.batch_history.append(batch_loss_info)
            
            # Gradient clipping if specified
            if config.get('gradient_clip_value'):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip_value']
                )
                gradient_norms.append(grad_norm.item())
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print batch progress for long epochs
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total,
            'gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'batch_losses': batch_losses
        }
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _create_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = config.get('optimizer', 'Adam')
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler if specified."""
        scheduler_name = config.get('scheduler')
        
        if scheduler_name == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        elif scheduler_name == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            return None
    
    def train_with_hyperparameter_sweep(
        self,
        datasets: Dict[str, DataLoader],
        architectures: Dict[str, Any],
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train models with hyperparameter sweep.
        
        Args:
            datasets: Dictionary containing train/val/test data loaders
            architectures: Dictionary of architecture configurations
            hyperparameters: Dictionary of hyperparameter ranges
        
        Returns:
            Results from hyperparameter sweep
        """
        from .models import get_model_for_analysis
        
        results = []
        best_config = None
        best_accuracy = 0.0
        
        # Hyperparameter combinations
        learning_rates = hyperparameters.get('learning_rates', [0.001])
        optimizers = hyperparameters.get('optimizers', ['Adam'])
        
        for lr in learning_rates:
            for opt in optimizers:
                print(f"\nTraining with LR={lr}, Optimizer={opt}")
                
                # Create model
                model = get_model_for_analysis('standard_mlp', self.activation)
                
                # Training configuration
                config = {
                    'learning_rate': lr,
                    'optimizer': opt,
                    'epochs': hyperparameters.get('epochs', 6),
                    'early_stopping_patience': hyperparameters.get('early_stopping_patience', 3),
                    'gradient_clip_value': hyperparameters.get('gradient_clip_value', 1.0)
                }
                
                # Train model
                training_results = self.train_model(
                    model, datasets['train'], datasets['validation'], config
                )
                
                # Test evaluation
                test_accuracy = self._evaluate_model(model, datasets['test'])
                
                # Store results
                result = {
                    'config': config,
                    'training_results': training_results,
                    'test_accuracy': test_accuracy,
                    'activation_name': getattr(self.activation, 'name', type(self.activation).__name__)
                }
                results.append(result)
                
                # Track best configuration
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_config = config.copy()
                    self.best_model = model.state_dict().copy()
        
        return {
            'results': results,
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'activation_name': getattr(self.activation, 'name', type(self.activation).__name__)
        }
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def run_comprehensive_analysis(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the best trained model.
        
        Args:
            test_loader: Test data loader for analysis
        
        Returns:
            Comprehensive analysis results
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Run training first.")
        
        # Create model and load best weights
        from .models import get_model_for_analysis
        model = get_model_for_analysis('standard_mlp', self.activation)
        model.load_state_dict(self.best_model)
        model = model.to(self.device)
        
        # Run comprehensive metrics analysis
        analysis_results = run_comprehensive_metrics_analysis(
            model, test_loader, self.activation, self.training_history
        )
        
        return analysis_results


def train_multiple_activations(
    activation_functions: Dict[str, nn.Module],
    datasets: Dict[str, DataLoader],
    architectures: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Train multiple activation functions and compare results.
    
    Args:
        activation_functions: Dictionary of activation functions to test
        datasets: Dictionary containing data loaders
        architectures: Dictionary of architecture configurations
        hyperparameters: Dictionary of hyperparameter ranges
        device: Device to use for training
    
    Returns:
        Comprehensive comparison results
    """
    results = {}
    
    for act_name, activation in activation_functions.items():
        print(f"\n{'='*60}")
        print(f"Training with {act_name} activation")
        print(f"{'='*60}")
        
        # Create trainer with checkpointing enabled
        activation_output_dir = Path("./results") / f"{act_name}_training"
        trainer = EnhancedTrainer(
            activation=activation, 
            device=device,
            output_dir=activation_output_dir,
            enable_checkpoints=True,
            checkpoint_frequency=1  # Save every epoch
        )
        
        # Train with hyperparameter sweep
        training_results = trainer.train_with_hyperparameter_sweep(
            datasets, architectures, hyperparameters
        )
        
        # Run comprehensive analysis
        analysis_results = trainer.run_comprehensive_analysis(datasets['test'])
        
        # Save batch-level history
        batch_history_file = activation_output_dir / "batch_history.json"
        if trainer.batch_history:
            import json
            with open(batch_history_file, 'w') as f:
                json.dump(trainer.batch_history, f, indent=2)
            print(f"📊 Batch history saved: {batch_history_file}")
        
        # Combine results
        results[act_name] = {
            'training': training_results,
            'analysis': analysis_results,
            'best_accuracy': training_results['best_accuracy'],
            'batch_history_file': str(batch_history_file) if trainer.batch_history else None,
            'checkpoint_dir': str(trainer.checkpoint_dir)
        }
        
        print(f"✓ {act_name} completed - Best accuracy: {training_results['best_accuracy']:.4f}")
        print(f"📁 Checkpoints saved to: {trainer.checkpoint_dir}")
    
    return results
