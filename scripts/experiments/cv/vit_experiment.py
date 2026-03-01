"""
Vision Transformer Experiment for CIFAR-100

This experiment implements Thrust 2 of the SoftCap research program:
Modern Architecture Viability testing with attention-based models.

Tests Vision Transformer Tiny on CIFAR-100 dataset with different activation functions,
focusing on attention pattern analysis and representation quality.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# SoftCap imports
from softcap.activations import (
    SoftCap,
    SwishCap,
    SparseCap,
)
from softcap.models import ViTTiny
from softcap.parallel_utils import optimize_dataloader


class VisionTransformerExperiment:
    """
    Vision Transformer experiment for testing SoftCap activations in attention-based models.
    
    This experiment focuses on:
    1. Attention pattern quality across different activations
    2. Representation learning effectiveness
    3. Training stability in transformer architecture
    4. Gradient flow analysis through attention layers
    """
    
    def __init__(self, device: str = None, results_dir: str = "results/vit_experiment"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment configuration
        self.config = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 0.05,
            'epochs': 100,
            'warmup_epochs': 10,
            'patience': 15,
            'img_size': 32,
            'patch_size': 4,
            'num_classes': 100,
            'embed_dim': 192,
            'depth': 6,
            'num_heads': 3,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'drop_path': 0.1
        }
        
        # Data setup
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        
        # Activation functions to test
        self.activations = {
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU(),
            'SiLU': nn.SiLU(),
            'Tanh': nn.Tanh(),
            'SoftCap': SoftCap(),
            'SwishCap': SwishCap(),
            'SparseCap': SparseCap(),
        }
        
        # Results storage
        self.results = {}
        
    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup CIFAR-100 data loaders with appropriate transforms for ViT."""
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        
        # Standard transform for validation/test
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=val_transform
        )
        
        # Split validation set from test set (8000 val, 2000 test)
        val_size = 8000
        test_size = len(val_dataset) - val_size
        val_dataset, test_dataset = torch.utils.data.random_split(
            val_dataset, [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create optimized data loaders
        train_loader = optimize_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = optimize_dataloader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        test_loader = optimize_dataloader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, activation: nn.Module) -> ViTTiny:
        """Create a ViT-Tiny model with specified activation."""
        return ViTTiny(
            img_size=self.config['img_size'],
            patch_size=self.config['patch_size'],
            num_classes=self.config['num_classes'],
            embed_dim=self.config['embed_dim'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            mlp_ratio=self.config['mlp_ratio'],
            activation=activation,
            dropout=self.config['dropout'],
            drop_path=self.config['drop_path']
        ).to(self.device)
    
    def create_optimizer_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, Any]:
        """Create optimizer and learning rate scheduler."""
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restart
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['epochs'] // 4,
            T_mult=2,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int) -> Dict[str, float]:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Learning rate warmup
        if epoch < self.config['warmup_epochs']:
            lr_scale = (epoch + 1) / self.config['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config['learning_rate'] * lr_scale
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, model: nn.Module) -> Dict[str, float]:
        """Validate model on validation set."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def test_model(self, model: nn.Module) -> Dict[str, float]:
        """Test model on test set."""
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                c = (predicted == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        overall_accuracy = 100.0 * correct / total
        class_accuracies = [100.0 * class_correct[i] / class_total[i] 
                           if class_total[i] > 0 else 0.0 for i in range(100)]
        
        return {
            'accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'mean_class_accuracy': np.mean(class_accuracies)
        }
    
    def analyze_attention_patterns(self, model: nn.Module, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze attention patterns across different layers.
        
        This is a key component of Thrust 2 analysis.
        """
        model.eval()
        attention_maps = {f'layer_{i}': [] for i in range(self.config['depth'])}
        
        with torch.no_grad():
            sample_count = 0
            for data, _ in self.test_loader:
                if sample_count >= num_samples:
                    break
                
                data = data.to(self.device)
                batch_size = min(data.size(0), num_samples - sample_count)
                data = data[:batch_size]
                
                # Get attention maps
                _, attn_maps = model.forward_with_attention_analysis(data)
                
                for layer, attn in attn_maps.items():
                    # Average attention over batch and heads
                    avg_attn = attn.mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                    attention_maps[layer].append(avg_attn)
                
                sample_count += batch_size
        
        # Compute statistics
        attention_stats = {}
        for layer, maps in attention_maps.items():
            all_maps = np.stack(maps, axis=0)  # [num_samples, seq_len, seq_len]
            
            # Attention entropy (measure of attention distribution)
            entropy = []
            for i in range(all_maps.shape[0]):
                attn_map = all_maps[i]
                # Focus on CLS token attention to patches
                cls_attention = attn_map[0, 1:]  # CLS token attention to patches
                cls_attention = cls_attention / (cls_attention.sum() + 1e-8)
                ent = -np.sum(cls_attention * np.log(cls_attention + 1e-8))
                entropy.append(ent)
            
            # Attention distance (how far attention spreads)
            distances = []
            patch_grid_size = int(np.sqrt(all_maps.shape[1] - 1))  # Excluding CLS token
            for i in range(all_maps.shape[0]):
                attn_map = all_maps[i]
                cls_attention = attn_map[0, 1:]  # CLS token attention to patches
                
                # Convert to 2D grid coordinates
                total_distance = 0
                for patch_idx, attn_weight in enumerate(cls_attention):
                    row = patch_idx // patch_grid_size
                    col = patch_idx % patch_grid_size
                    center_row, center_col = patch_grid_size // 2, patch_grid_size // 2
                    distance = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                    total_distance += attn_weight * distance
                distances.append(total_distance)
            
            attention_stats[layer] = {
                'mean_entropy': np.mean(entropy),
                'std_entropy': np.std(entropy),
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'sample_maps': all_maps[:5]  # Store first 5 for visualization
            }
        
        return attention_stats
    
    def extract_representations(self, model: nn.Module, layer_idx: int = -2) -> Dict[str, np.ndarray]:
        """Extract feature representations from specified layer."""
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                feats = model.extract_representations(data, layer_idx)
                features.append(feats.cpu().numpy())
                labels.append(target.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return {'features': features, 'labels': labels}
    
    def train_single_activation(self, activation_name: str, activation: nn.Module) -> Dict[str, Any]:
        """Train model with single activation function."""
        print(f"\n=== Training ViT-Tiny with {activation_name} ===")
        
        # Create model
        model = self.create_model(activation)
        optimizer, scheduler = self.create_optimizer_scheduler(model)
        
        # Training tracking
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            train_metrics = self.train_epoch(model, optimizer, epoch)
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(model)
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['accuracy'])
            
            # Learning rate scheduling (after warmup)
            if epoch >= self.config['warmup_epochs']:
                scheduler.step()
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final test evaluation
        test_metrics = self.test_model(model)
        
        # Attention analysis
        attention_stats = self.analyze_attention_patterns(model)
        
        # Feature extraction
        representations = self.extract_representations(model)
        
        return {
            'activation_name': activation_name,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_metrics['accuracy'],
            'test_class_accuracies': test_metrics['class_accuracies'],
            'mean_class_accuracy': test_metrics['mean_class_accuracy'],
            'attention_patterns': attention_stats,
            'representations': representations,
            'final_epoch': len(train_losses)
        }
    
    def run_full_comparison(self) -> Dict[str, Any]:
        """Run full comparison across all activation functions."""
        print("Starting Vision Transformer Experiment")
        print(f"Device: {self.device}")
        print(f"Configuration: {self.config}")
        
        results = {}
        
        for activation_name, activation in self.activations.items():
            try:
                result = self.train_single_activation(activation_name, activation)
                results[activation_name] = result
                
                # Save intermediate results
                self.save_results(results)
                
            except Exception as e:
                print(f"Error training {activation_name}: {str(e)}")
                results[activation_name] = {'error': str(e)}
        
        # Generate comprehensive analysis
        analysis = self.analyze_results(results)
        results['analysis'] = analysis
        
        # Save final results
        self.save_results(results)
        self.generate_plots(results)
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across all activation functions."""
        successful_results = {k: v for k, v in results.items() 
                            if isinstance(v, dict) and 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful training runs'}
        
        # Performance comparison
        performance_comparison = {}
        for name, result in successful_results.items():
            performance_comparison[name] = {
                'test_accuracy': result['test_accuracy'],
                'best_val_accuracy': result['best_val_accuracy'],
                'mean_class_accuracy': result['mean_class_accuracy'],
                'training_epochs': result['final_epoch']
            }
        
        # Attention pattern analysis
        attention_analysis = {}
        for name, result in successful_results.items():
            if 'attention_patterns' in result:
                patterns = result['attention_patterns']
                
                # Average entropy across layers
                avg_entropy = np.mean([stats['mean_entropy'] 
                                     for stats in patterns.values()])
                
                # Average attention distance
                avg_distance = np.mean([stats['mean_distance'] 
                                      for stats in patterns.values()])
                
                attention_analysis[name] = {
                    'average_entropy': avg_entropy,
                    'average_distance': avg_distance,
                    'layer_entropies': {layer: stats['mean_entropy'] 
                                       for layer, stats in patterns.items()},
                    'layer_distances': {layer: stats['mean_distance'] 
                                       for layer, stats in patterns.items()}
                }
        
        # Best performing activations
        best_test_acc = max(performance_comparison.values(), 
                           key=lambda x: x['test_accuracy'])
        best_activation = [name for name, metrics in performance_comparison.items() 
                          if metrics['test_accuracy'] == best_test_acc['test_accuracy']][0]
        
        return {
            'performance_comparison': performance_comparison,
            'attention_analysis': attention_analysis,
            'best_activation': best_activation,
            'best_test_accuracy': best_test_acc['test_accuracy'],
            'summary_statistics': {
                'mean_test_accuracy': np.mean([m['test_accuracy'] 
                                             for m in performance_comparison.values()]),
                'std_test_accuracy': np.std([m['test_accuracy'] 
                                           for m in performance_comparison.values()]),
                'activation_ranking': sorted(performance_comparison.items(), 
                                           key=lambda x: x[1]['test_accuracy'], 
                                           reverse=True)
            }
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = self._convert_for_json(value)
            else:
                json_results[key] = value
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"vit_experiment_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON types for serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def generate_plots(self, results: Dict[str, Any]):
        """Generate comprehensive visualization plots."""
        successful_results = {k: v for k, v in results.items() 
                            if isinstance(v, dict) and 'error' not in v and k != 'analysis'}
        
        if not successful_results:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Training curves
        ax1 = plt.subplot(3, 3, 1)
        for name, result in successful_results.items():
            epochs = range(1, len(result['train_losses']) + 1)
            plt.plot(epochs, result['train_losses'], label=f'{name} (train)', alpha=0.7)
            plt.plot(epochs, result['val_losses'], label=f'{name} (val)', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = plt.subplot(3, 3, 2)
        for name, result in successful_results.items():
            epochs = range(1, len(result['train_accuracies']) + 1)
            plt.plot(epochs, result['train_accuracies'], label=f'{name} (train)', alpha=0.7)
            plt.plot(epochs, result['val_accuracies'], label=f'{name} (val)', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Final performance comparison
        ax3 = plt.subplot(3, 3, 3)
        names = list(successful_results.keys())
        test_accs = [successful_results[name]['test_accuracy'] for name in names]
        val_accs = [successful_results[name]['best_val_accuracy'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        plt.bar(x - width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        plt.bar(x + width/2, val_accs, width, label='Best Val Accuracy', alpha=0.8)
        plt.xlabel('Activation Function')
        plt.ylabel('Accuracy (%)')
        plt.title('Final Performance Comparison')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Attention entropy by layer
        ax4 = plt.subplot(3, 3, 4)
        if 'analysis' in results and 'attention_analysis' in results['analysis']:
            for name, analysis in results['analysis']['attention_analysis'].items():
                layers = list(analysis['layer_entropies'].keys())
                entropies = list(analysis['layer_entropies'].values())
                layer_nums = [int(layer.split('_')[1]) for layer in layers]
                plt.plot(layer_nums, entropies, marker='o', label=name, alpha=0.7)
        plt.xlabel('Layer')
        plt.ylabel('Average Attention Entropy')
        plt.title('Attention Entropy by Layer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Attention distance by layer
        ax5 = plt.subplot(3, 3, 5)
        if 'analysis' in results and 'attention_analysis' in results['analysis']:
            for name, analysis in results['analysis']['attention_analysis'].items():
                layers = list(analysis['layer_distances'].keys())
                distances = list(analysis['layer_distances'].values())
                layer_nums = [int(layer.split('_')[1]) for layer in layers]
                plt.plot(layer_nums, distances, marker='s', label=name, alpha=0.7)
        plt.xlabel('Layer')
        plt.ylabel('Average Attention Distance')
        plt.title('Attention Distance by Layer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Training convergence comparison
        ax6 = plt.subplot(3, 3, 6)
        training_epochs = [successful_results[name]['final_epoch'] for name in names]
        plt.bar(names, training_epochs, alpha=0.8)
        plt.xlabel('Activation Function')
        plt.ylabel('Training Epochs')
        plt.title('Training Convergence (Epochs to Best)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 7. Attention entropy vs accuracy scatter
        ax7 = plt.subplot(3, 3, 7)
        if 'analysis' in results and 'attention_analysis' in results['analysis']:
            for name in successful_results.keys():
                if name in results['analysis']['attention_analysis']:
                    entropy = results['analysis']['attention_analysis'][name]['average_entropy']
                    accuracy = successful_results[name]['test_accuracy']
                    plt.scatter(entropy, accuracy, label=name, s=100, alpha=0.7)
        plt.xlabel('Average Attention Entropy')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Attention Entropy vs Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Performance summary box plot
        ax8 = plt.subplot(3, 3, 8)
        if len(successful_results) >= 3:  # Only if we have enough data
            all_test_accs = [result['test_accuracy'] for result in successful_results.values()]
            all_val_accs = [result['best_val_accuracy'] for result in successful_results.values()]
            
            plt.boxplot([all_test_accs, all_val_accs], labels=['Test', 'Validation'])
            plt.ylabel('Accuracy (%)')
            plt.title('Performance Distribution')
            plt.grid(True, alpha=0.3)
        
        # 9. Sample attention map visualization
        ax9 = plt.subplot(3, 3, 9)
        if successful_results and 'attention_patterns' in list(successful_results.values())[0]:
            # Show attention map from best performing activation
            if 'analysis' in results:
                best_activation = results['analysis']['best_activation']
                if best_activation in successful_results:
                    attention_patterns = successful_results[best_activation]['attention_patterns']
                    # Show attention from middle layer
                    middle_layer = f'layer_{self.config["depth"]//2}'
                    if middle_layer in attention_patterns:
                        sample_map = attention_patterns[middle_layer]['sample_maps'][0]
                        im = plt.imshow(sample_map, cmap='Blues', aspect='auto')
                        plt.title(f'Attention Map ({best_activation}, {middle_layer})')
                        plt.xlabel('Token Position')
                        plt.ylabel('Token Position')
                        plt.colorbar(im, shrink=0.8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"vit_experiment_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to: {plot_file}")


def run_vit_comparison():
    """Main function to run the Vision Transformer comparison."""
    # Initialize experiment
    experiment = VisionTransformerExperiment(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        results_dir="results/vit_experiment"
    )
    
    # Run full comparison
    results = experiment.run_full_comparison()
    
    # Print summary
    if 'analysis' in results and 'summary_statistics' in results['analysis']:
        summary = results['analysis']['summary_statistics']
        print("\n" + "="*60)
        print("VISION TRANSFORMER EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Best Activation: {results['analysis']['best_activation']}")
        print(f"Best Test Accuracy: {results['analysis']['best_test_accuracy']:.2f}%")
        print(f"Mean Test Accuracy: {summary['mean_test_accuracy']:.2f}%")
        print(f"Std Test Accuracy: {summary['std_test_accuracy']:.2f}%")
        print("\nActivation Ranking:")
        for i, (name, metrics) in enumerate(summary['activation_ranking'], 1):
            print(f"{i:2d}. {name:25s}: {metrics['test_accuracy']:.2f}%")
        print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_vit_comparison()