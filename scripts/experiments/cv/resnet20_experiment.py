"""
ResNet-20 CIFAR-100 Experiment

This module implements ResNet-20 classification on CIFAR-100 as part of Thrust 2:
Modern Architecture Viability. Tests SoftCap in residual networks with batch 
normalization and includes A/B testing for BatchNorm dependency analysis.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.base.base_experiment import DomainExperiment
from softcap.parallel_utils import optimize_dataloader


class _CIFARBasicBlock(nn.Module):
    """CIFAR ResNet basic block with swappable activation."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, activation: nn.Module, use_batch_norm: bool = True):
        super().__init__()
        import copy
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        self.act1 = activation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        self.act2 = copy.deepcopy(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            shortcut_layers = [nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=not use_batch_norm)]
            if use_batch_norm:
                shortcut_layers.append(nn.BatchNorm2d(planes))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class _CIFARResNet20(nn.Module):
    """ResNet-20 for CIFAR input (3 stages, 3 blocks/stage)."""

    def __init__(self, activation: nn.Module, num_classes: int = 100, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        self.act0 = activation

        self.layer1 = self._make_layer(16, num_blocks=3, stride=1, activation=activation)
        self.layer2 = self._make_layer(32, num_blocks=3, stride=2, activation=activation)
        self.layer3 = self._make_layer(64, num_blocks=3, stride=2, activation=activation)
        self.fc = nn.Linear(64, num_classes)

    def _clone_activation(self, activation: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(activation)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, activation: nn.Module) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                _CIFARBasicBlock(
                    self.in_planes,
                    planes,
                    s,
                    activation=self._clone_activation(activation),
                    use_batch_norm=self.use_batch_norm,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ResNet20Experiment(DomainExperiment):
    """ResNet-20 CIFAR-100 experiment with SoftCap and baseline activations."""
    
    def __init__(
        self,
        name: str = "resnet20_cifar100",
        activation: str = "softcap",
        batch_size: int = 128,
        learning_rate: float = 0.1,
        num_epochs: int = 100,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Initialize the ResNet-20 CIFAR-100 experiment.
        
        Args:
            name: Experiment name
            activation: Activation function to use
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            use_batch_norm: Whether to use batch normalization
            **kwargs: Additional arguments for parent class
        """
        super().__init__(name=name, **kwargs)
        
        # Experiment parameters
        self.activation = activation.lower()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_batch_norm = use_batch_norm
        
        # Update metadata
        self.metadata.update({
            'activation': activation,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'use_batch_norm': use_batch_norm,
            'dataset': 'CIFAR-100',
            'model': 'ResNet-20',
            'thrust': 'Thrust 2: Modern Architecture Viability'
        })
        
        self.logger.info(f"Initialized ResNet-20 CIFAR-100 experiment with {activation} activation")
    
    def create_model(self) -> nn.Module:
        """Create the ResNet-20 model with the specified activation."""
        return _CIFARResNet20(
            activation=self._get_activation_function(),
            num_classes=100,
            use_batch_norm=self.use_batch_norm,
        )
    
    def _get_activation_function(self) -> nn.Module:
        """Get activation function instance."""
        from softcap.control_activations import (
            get_control_activations,
            get_extended_astar_activations,
            get_standard_experimental_set
        )
        
        # 1. Check Extended Astar Set (The modern standard)
        extended_set = get_extended_astar_activations()
        for name, module in extended_set.items():
            if self.activation.lower() == name.lower():
                return module
            
        # 2. Check Standard Baselines
        baselines = get_control_activations()
        for name, module in baselines.items():
            if self.activation.lower() == name.lower():
                return module
            
        # 3. Minimal compatibility aliases for canonical names
        activation = self.activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation in ['silu', 'swish']:
            return nn.SiLU()
        elif activation in ['parametric_tanh_softcap', 'parametrictanhsoftcap', 'softcap']:
            from softcap.activations import SoftCap
            return SoftCap()
        elif activation in ['parametricsmoothnotchtanhsoftcapv2', 'swishcap']:
            from softcap.activations import SwishCap
            return SwishCap()
        elif activation in ['parametricquinticnotchtanhsoftcap', 'sparsecap']:
            from softcap.activations import SparseCap
            return SparseCap()
            
        raise ValueError(f"Unsupported activation: {self.activation}")
    
    def create_dataloaders(
        self,
        data_dir: str = "data/cifar100",
        val_split: float = 0.1,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for CIFAR-100.
        
        Args:
            data_dir: Directory to store the dataset
            val_split: Fraction of training data to use for validation
            num_workers: Number of worker processes for data loading
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Define transforms with CIFAR-100 optimizations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load datasets
        train_val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        # Split training set into train and validation
        val_size = int(len(train_val_dataset) * val_split)
        train_size = len(train_val_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply test transform to validation set
        val_dataset.dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=False,
            transform=transform_test
        )
        
        # Use plain DataLoader here to avoid platform-specific worker/socket issues.
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        self.logger.info(f"Created CIFAR-100 data loaders: "
                        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on the CIFAR-100 dataset.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            device: Device to use for training
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        model = self.create_model()
        model = model.to(device)
        
        # Use SGD with momentum for ResNet (standard practice)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Learning rate schedule (standard for CIFAR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[50, 75], 
            gamma=0.1
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    self.logger.debug(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Step scheduler
            scheduler.step()
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                self.logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, '
                               f'Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        results = {
            'model': model,
            'history': history,
            'best_val_accuracy': best_val_acc,
            'final_train_accuracy': history['train_acc'][-1],
            'final_val_accuracy': history['val_acc'][-1],
            'convergence_epoch': np.argmax(history['val_acc']),
            'training_parameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.num_epochs,
                'optimizer': 'SGD',
                'scheduler': 'MultiStepLR',
                'use_batch_norm': self.use_batch_norm
            }
        }
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return results
    
    def evaluate(
        self, 
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to use for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'total_samples': test_total
        }
        
        self.logger.info(f"Test evaluation completed. Accuracy: {test_acc:.2f}%")
        
        return results


def run_resnet20_comparison(
    activations: List[str],
    output_dir: str = "experiments/resnet20_comparison",
    device: str = "cuda",
    with_batch_norm: bool = True,
    without_batch_norm: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run ResNet-20 comparison across multiple activation functions.
    
    Args:
        activations: List of activation function names to test
        output_dir: Output directory for results
        device: Device to use for training
        with_batch_norm: Whether to test with batch normalization
        without_batch_norm: Whether to test without batch normalization  
        **kwargs: Additional arguments for ResNet20Experiment
        
    Returns:
        Dictionary containing all experimental results
    """
    results = {}
    
    configurations = []
    if with_batch_norm:
        configurations.append(('with_bn', True))
    if without_batch_norm:
        configurations.append(('without_bn', False))
    
    for config_name, use_bn in configurations:
        results[config_name] = {}
        
        for activation in activations:
            print(f"\n🔄 Training ResNet-20 with {activation} ({config_name})...")
            
            experiment = ResNet20Experiment(
                name=f"resnet20_{activation}_{config_name}",
                activation=activation,
                use_batch_norm=use_bn,
                **kwargs
            )
            
            # Create data loaders
            train_loader, val_loader, test_loader = experiment.create_dataloaders()
            
            # Train model
            training_results = experiment.train(
                train_loader, val_loader, device=device
            )
            
            # Test evaluation
            test_results = experiment.evaluate(
                training_results['model'], test_loader, device=device
            )
            
            # Combine results
            results[config_name][activation] = {
                'training': training_results,
                'testing': test_results,
                'experiment_config': experiment.metadata
            }
            
            print(f"   ✓ Best Val Accuracy: {training_results['best_val_accuracy']:.2f}%")
            print(f"   ✓ Test Accuracy: {test_results['test_accuracy']:.2f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='ResNet-20 CIFAR-100 Experiment (Thrust 3)')
    
    # Activation selection
    act_group = parser.add_mutually_exclusive_group(required=True)
    act_group.add_argument('--activation', type=str, help='Single activation to test')
    act_group.add_argument('--all-activations', action='store_true', 
                          help='Run standard experimental set (Parametric* + controls)')
    act_group.add_argument('--controls-only', action='store_true',
                          help='Run only controls (ReLU, Tanh, GELU, SiLU)')
    act_group.add_argument('--extended-astar', action='store_true', 
                          help='Run full comprehensive Extended Astar set')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--extreme-lr', action='store_true', help='Use extreme LR=0.3 stress test')
    
    # Output and resume
    parser.add_argument('--output-dir', type=str, default='Thrust_3/resnet20_cifar100',
                       help='Output directory (default: Thrust_3/resnet20_cifar100)')
    parser.add_argument('--resume', dest='resume', action='store_true',
                       help='Resume by skipping completed activations (default: on)')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    parser.add_argument('--force', action='store_true', help='Force recompute even if exists')
    
    # BatchNorm A/B testing
    parser.add_argument('--no-batch-norm', action='store_true', help='Disable batch normalization')
    
    # Quick mode
    parser.add_argument('--quick', action='store_true', help='Quick test (10 epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
    
    if args.extreme_lr:
        args.lr = 0.3
    
    # Get activation list
    from softcap.control_activations import get_control_activations, get_standard_experimental_set
    
    if args.extended_astar:
        from softcap.control_activations import get_extended_astar_activations
        activation_dict = get_extended_astar_activations()
        activations = list(activation_dict.keys())
        print(f"Running extended astar set: {activations}")
    elif args.all_activations:
        activation_dict = get_standard_experimental_set()
        activations = list(activation_dict.keys())
        print(f"Running standard experimental set: {activations}")
    elif args.controls_only:
        activation_dict = get_control_activations()
        activations = list(activation_dict.keys())
        print(f"Running controls only: {activations}")
    else:
        activations = [args.activation]
        print(f"Running single activation: {args.activation}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}")
    
    all_results = {}
    
    for activation in activations:
        result_path = output_dir / f"{activation}_results.json"
        
        # Resume support
        if args.resume and not args.force and result_path.exists():
            print(f"\n✓ Skipping {activation} (already exists, use --force to recompute)")
            with open(result_path, 'r') as f:
                all_results[activation] = json.load(f)
            continue
        
        print(f"\n{'='*60}")
        print(f"Training ResNet-20 with {activation}")
        print(f"{'='*60}")
        
        experiment = ResNet20Experiment(
            name=f"resnet20_{activation}",
            activation=activation,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            use_batch_norm=not args.no_batch_norm,
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = experiment.create_dataloaders()
        
        # Train model
        training_results = experiment.train(train_loader, val_loader, device=device)
        
        # Test evaluation
        test_results = experiment.evaluate(training_results['model'], test_loader, device=device)
        
        # Save checkpoint
        checkpoint = {
            'activation': activation,
            'best_val_accuracy': training_results['best_val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'final_train_accuracy': training_results['final_train_accuracy'],
            'hyperparameters': {
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'use_batch_norm': not args.no_batch_norm,
            },
            'training_history': {
                'train_loss': training_results['history']['train_loss'],
                'val_acc': training_results['history']['val_acc'],
            }
        }
        
        with open(result_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        all_results[activation] = checkpoint
        
        print(f"   ✓ Best Val: {training_results['best_val_accuracy']:.2f}%")
        print(f"   ✓ Test Acc: {test_results['test_accuracy']:.2f}%")
        print(f"   ✓ Saved to: {result_path}")
    
    # Generate summary
    print(f"\n{'='*80}")
    print("RESNET-20 CIFAR-100 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Activation':<35} {'Val Acc':>12} {'Test Acc':>12}")
    print(f"{'-'*80}")
    for act_name, result in all_results.items():
        print(f"{act_name:<35} {result['best_val_accuracy']:>11.2f}% {result['test_accuracy']:>11.2f}%")
    
    # Save overall summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'benchmark': 'resnet20_cifar100',
            'num_activations': len(all_results),
            'epochs': args.epochs,
            'lr': args.lr,
            'results': all_results
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
