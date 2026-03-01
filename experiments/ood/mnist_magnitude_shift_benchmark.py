#!/usr/bin/env python3
"""MNIST Magnitude Shift Benchmark

Goal:
- Test how well activations preserve distance ratios under input magnitude scaling
- Real-data OOD test: train on MNIST, test on scaled inputs

Task:
- Train MNIST classifier on standard inputs (scale=1.0)  
- At test time, apply scale factors: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Measure accuracy decay as function of scale

Theoretical Motivation:
- Isotropic activations should preserve relative representations under scaling
- Anisotropic activations (ReLU) map all scaled inputs to same half-space → lose info

References:
- Mu et al. (2018) — Representation anisotropy in word embeddings
- Ethayarajh (2019) — Anisotropy in contextualized representations
- Tu et al. (2025) — Isotropic Deep Learning

Usage:
    # Quick test
    python mnist_magnitude_shift_benchmark.py --controls-only --n-seeds 1 --epochs 5
    
    # Full sweep  
    python mnist_magnitude_shift_benchmark.py --comprehensive --n-seeds 5 --epochs 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Add project root to path
# Locate repo root (works at any depth in the tree)
_repo_p = Path(__file__).resolve().parent
while _repo_p != _repo_p.parent and not (_repo_p / 'softcap').is_dir():
    _repo_p = _repo_p.parent
project_root = _repo_p
sys.path.insert(0, str(project_root))

from softcap.control_activations import (
    get_control_activations,
    get_standard_experimental_set,
    get_extended_astar_activations,
)
from softcap.initialization import get_recommended_init, apply_initialization
from softcap.models import SimpleCNN


# =============================================================================
# Model
# =============================================================================


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, list]:
    """Train model and return history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    return history


@torch.no_grad()
def evaluate_at_scale(
    model: nn.Module,
    test_loader: DataLoader,
    scale_factor: float,
    device: torch.device,
) -> float:
    """Evaluate model on scaled inputs."""
    model.eval()
    correct = 0
    total = 0
    
    for batch_x, batch_y in test_loader:
        batch_x = batch_x * scale_factor  # Apply magnitude scaling
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        pred = logits.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
    
    return correct / total


# =============================================================================
# Visualization
# =============================================================================

def plot_scale_decay_curve(scale_factors, accuracies, activation_name, output_path):
    """Plot accuracy vs scale factor."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scale_factors, accuracies, marker='o', linewidth=2, markersize=8)
    ax.axvline(1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Training Scale')
    ax.axhline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Random Chance')
    ax.set_xlabel('Scale Factor', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'MNIST Magnitude Shift: {activation_name}', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Benchmark Logic
# =============================================================================

def run_benchmark_for_activation(
    activation_name: str,
    activation_fn: nn.Module,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, any]:
    """Run complete benchmark for a single activation and seed."""
    
    print(f"\n{'='*60}")
    print(f"MNIST Magnitude Shift: {activation_name} (seed={seed})")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    activation_dir = output_dir / activation_name
    activation_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpointing
    seed_metrics_path = activation_dir / f'metrics_seed{seed}.json'
    if args.resume and seed_metrics_path.exists():
        with open(seed_metrics_path, 'r') as f:
            print(f"  ✓ Skipped (resume): {seed_metrics_path.name}")
            return json.load(f)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SimpleCNN(activation_fn).to(device)
    
    init_method = get_recommended_init(activation_name)
    print(f"Device: {device}, Init: {init_method}")
    
    apply_initialization(model, init_method, activation_name)
    
    print("Training...")
    history = train_model(model, train_loader, device, args.epochs, args.lr)
    
    print("Evaluating at different scales...")
    scale_factors = [float(s) for s in args.scale_factors.split(',')]
    accuracies = []
    
    for scale in scale_factors:
        acc = evaluate_at_scale(model, test_loader, scale, device)
        accuracies.append(acc)
        print(f"  Scale {scale:.2f}: Accuracy = {acc:.4f}")
    
    # Robustness metric: average accuracy drop from scale=1.0
    idx_1 = scale_factors.index(1.0) if 1.0 in scale_factors else 0
    baseline_acc = accuracies[idx_1]
    avg_decay = np.mean([baseline_acc - acc for i, acc in enumerate(accuracies) if i != idx_1])
    
    results = {
        'activation': activation_name,
        'seed': seed,
        'final_train_accuracy': history['accuracy'][-1],
        'scale_factors': scale_factors,
        'test_accuracies': accuracies,
        'baseline_accuracy': baseline_acc,
        'avg_accuracy_decay': avg_decay,
        'training_history': history,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'init': init_method,
        }
    }
    
    with open(seed_metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if seed == 0:
        plot_scale_decay_curve(scale_factors, accuracies, activation_name,
                               activation_dir / 'scale_decay_curve.png')
    
    return results


class MNISTMagnitudeShiftBenchmark:
    """Orchestrator for MNIST magnitude shift benchmark."""
    
    @staticmethod
    def run(args: argparse.Namespace) -> Dict[str, any]:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\nMNIST Magnitude Shift Benchmark")
        print(f"Output: {output_dir}")
        
        if args.controls_only:
            activations = get_control_activations()
            print("Mode: Controls only")
        elif args.comprehensive:
            activations = get_extended_astar_activations()
            controls = get_control_activations()
            for name, fn in controls.items():
                if name not in activations:
                    activations[name] = fn
            print(f"Mode: Comprehensive ({len(activations)} activations)")
        elif args.all_activations:
            activations = get_standard_experimental_set()
            print("Mode: Standard set")
        elif args.activation:
            all_acts = {**get_standard_experimental_set(), **get_extended_astar_activations()}
            if args.activation not in all_acts:
                raise ValueError(f"Unknown activation: {args.activation}")
            activations = {args.activation: all_acts[args.activation]}
            print(f"Mode: Single ({args.activation})")
        else:
            # Default to comprehensive mode
            activations = get_extended_astar_activations()
            controls = get_control_activations()
            for name, fn in controls.items():
                if name not in activations:
                    activations[name] = fn
            print(f"Mode: Comprehensive ({len(activations)} activations) [default]")
        
        all_results = {}
        for act_name, act_fn in activations.items():
            seed_results = []
            for seed in range(args.n_seeds):
                seed_results.append(run_benchmark_for_activation(act_name, act_fn, seed, output_dir, args))
            all_results[act_name] = seed_results
        
        summary = {
            'benchmark': 'mnist_magnitude_shift',
            'timestamp': timestamp,
            'n_seeds': args.n_seeds,
            'num_activations': len(all_results),
            'results': {
                name: {
                    'baseline_accuracy': float(np.mean([r['baseline_accuracy'] for r in sr])),
                    'avg_accuracy_decay': float(np.mean([r['avg_accuracy_decay'] for r in sr])),
                    'decay_std': float(np.std([r['avg_accuracy_decay'] for r in sr])),
                }
                for name, sr in all_results.items()
            }
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}\nBENCHMARK SUMMARY\n{'='*80}")
        print(f"{'Activation':<35} {'Baseline':>10} {'Decay':>10}")
        print('-'*60)
        for name, res in sorted(summary['results'].items(), key=lambda x: x[1]['avg_accuracy_decay']):
            print(f"{name:<35} {res['baseline_accuracy']:>10.4f} {res['avg_accuracy_decay']:>10.4f}")
        
        print(f"\nSaved to: {output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description='MNIST Magnitude Shift Benchmark')
    
    act_group = parser.add_mutually_exclusive_group(required=False)
    act_group.add_argument('--activation', type=str, help='Single activation')
    act_group.add_argument('--all-activations', action='store_true', help='Standard set')
    act_group.add_argument('--comprehensive', action='store_true', help='Extended a* + controls (DEFAULT)')
    act_group.add_argument('--controls-only', action='store_true', help='Controls only')
    
    parser.add_argument('--scale-factors', type=str, default='0.1,0.25,0.5,1.0,2.0,4.0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='mechanistic_interpretability/latent_geometry/ood/magnitude_shift/standard')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    MNISTMagnitudeShiftBenchmark.run(args)


if __name__ == '__main__':
    main()
