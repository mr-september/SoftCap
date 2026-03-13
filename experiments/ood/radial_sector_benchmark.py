#!/usr/bin/env python3
# Copyright 2026 Larry Cai and Jie Tang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Radial Angular Sector Benchmark (Moderate OOD)
----------------------------------------------
Tests generalization of angular features across moderate scale shifts.
Unlike Spiral, the label depends ONLY on angle, not radius.
A correct model should represent "angle" independently of "magnitude".

Task:
- 8 Angular Sectors (Classes 0/1 alternating)
- Train: Annulus r in [1.0, 2.0]
- Test: Rings at r in {2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0} (max 5x scale)

This is a cleaner test of "Neural Refraction" than Spiral because the ground truth
decision boundary is a straight line ray from the origin, which should be
learnable by ReLU networks. Failures here indicate specific failures of
scale invariance, not just inability to extrapolate a spiral function.
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
import sys
# Locate repo root (works at any depth in the tree)
_repo_p = Path(__file__).resolve().parent
while _repo_p != _repo_p.parent and not (_repo_p / 'softcap').is_dir():
    _repo_p = _repo_p.parent
project_root = _repo_p
sys.path.insert(0, str(project_root))

from softcap.activations import (
    get_default_activations,
)
from softcap.control_activations import (
    get_control_activations,
    get_extended_astar_activations,
    get_standard_experimental_set,
)
from softcap.models import SimpleMLP
from softcap.initialization import get_recommended_init, apply_initialization

def get_sector_label(X, n_sectors=8):
    theta = np.arctan2(X[:, 1], X[:, 0])
    # Map [-pi, pi] to [0, 2pi]
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    
    sector_idx = np.floor(theta / (2*np.pi / n_sectors)).astype(int)
    return sector_idx % 2 # Alternating binary labels

def generate_data(n_samples, r_min, r_max, seed):
    rng = np.random.default_rng(seed)
    if float(r_min) == float(r_max):
        r = np.full((n_samples,), float(r_min), dtype=np.float32)
    else:
        r = rng.uniform(r_min, r_max, n_samples).astype(np.float32)
    theta = rng.uniform(0, 2*np.pi, n_samples)
    
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float32)
    y = get_sector_label(X)
    return X, y

def train_model(model, device, X, y, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

def evaluate(model, device, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return (preds == y).mean()


@torch.no_grad()
def _predict_grid(model: nn.Module, device: torch.device, grid_points: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(grid_points).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def plot_task_and_decision_boundary(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    activation_name: str,
    output_dir: Path,
    device: torch.device,
    train_r_max: float,
    test_radii: List[float],
    grid_range: float = 10.5,
    grid_resolution: int = 420,
    n_sectors: int = 8,
):
    """Save (1) a task visualization and (2) predicted-vs-GT decision boundary panels."""
    xx, yy = np.meshgrid(
        np.linspace(-grid_range, grid_range, grid_resolution),
        np.linspace(-grid_range, grid_range, grid_resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    Z_true = get_sector_label(grid_points, n_sectors=n_sectors).reshape(xx.shape)
    Z_pred = _predict_grid(model, device, grid_points).reshape(xx.shape)

    # Task visualization (ground truth only)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax.contourf(xx, yy, Z_true, levels=1, alpha=0.35, colors=['#ff7f0e', '#1f77b4'])
    ax.contour(xx, yy, Z_true, levels=1, colors='black', linewidths=1.2, alpha=0.75)
    train_circle = plt.Circle((0, 0), train_r_max, fill=False, color='red', linestyle='--', linewidth=1.2, alpha=0.85)
    ax.add_patch(train_circle)
    for r in sorted(set(float(r) for r in test_radii)):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', linewidth=0.8, alpha=0.65))
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.set_title('Radial Angular Sector — Ground Truth', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'task_visualization.png', dpi=220, bbox_inches='tight')
    plt.close()

    # Predicted vs ground truth (legacy 1x2 panel)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    ax = axes[0]
    ax.contourf(xx, yy, Z_pred, levels=1, alpha=0.35, colors=['#ff7f0e', '#1f77b4'])
    ax.contour(xx, yy, Z_pred, levels=1, colors='black', linewidths=1.2, alpha=0.75)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='#ff7f0e', s=8, alpha=0.45)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='#1f77b4', s=8, alpha=0.45)
    ax.add_patch(plt.Circle((0, 0), train_r_max, fill=False, color='red', linestyle='--', linewidth=1.2, alpha=0.85))
    for r in sorted(set(float(r) for r in test_radii)):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', linewidth=0.8, alpha=0.65))
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.set_title(f'{activation_name} — Predicted', fontsize=11)
    ax.axis('off')

    ax = axes[1]
    ax.contourf(xx, yy, Z_true, levels=1, alpha=0.35, colors=['#ff7f0e', '#1f77b4'])
    ax.contour(xx, yy, Z_true, levels=1, colors='black', linewidths=1.2, alpha=0.75)
    ax.add_patch(plt.Circle((0, 0), train_r_max, fill=False, color='red', linestyle='--', linewidth=1.2, alpha=0.85))
    for r in sorted(set(float(r) for r in test_radii)):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', linewidth=0.8, alpha=0.65))
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth', fontsize=11)
    ax.axis('off')

    plt.suptitle(f'Radial Angular Sector OOD: {activation_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_boundary.png', dpi=220, bbox_inches='tight')
    plt.close()

    # Clean Predicted (1x1 panel, matching spiral logic)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.contourf(xx, yy, Z_pred, levels=1, alpha=0.35, colors=['#ff7f0e', '#1f77b4'])
    ax.contour(xx, yy, Z_pred, levels=1, colors='black', linewidths=1.2, alpha=0.75)
    ax.add_patch(plt.Circle((0, 0), train_r_max, fill=False, color='red', linestyle='--', linewidth=1.2, alpha=0.85))
    for r in sorted(set(float(r) for r in test_radii)):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', linewidth=0.8, alpha=0.65))
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(output_dir / 'decision_boundary_clean_pred.png', dpi=220, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_extrapolation_curve(test_radii: List[float], accuracies: List[float], activation_name: str, output_path: Path, train_radius_max: float = 2.0):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(test_radii, accuracies, marker='o', linewidth=2, markersize=6)
    ax.axvline(train_radius_max, color='red', linestyle='--', linewidth=1.2, alpha=0.75, label=f'Train r≤{train_radius_max}')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Random Chance')
    ax.set_xlabel('Test Radius')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Radial Angular Sector: {activation_name}')
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()

def run_benchmark(activation_name, activation_fn, seed, output_dir, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    act_dir = output_dir / activation_name
    act_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = act_dir / f'metrics_seed{seed}.json'
    if args.resume and (not args.force) and metrics_path.exists():
        with open(metrics_path, 'r') as f:
            print(f"  ✓ Skipped (resume): {activation_name} seed={seed}")
            return json.load(f)
    
    # Train Data: r in [1, 2]
    X_train, y_train = generate_data(8000, 1.0, 2.0, seed)
    
    model = SimpleMLP(activation_fn, hidden_dim=128).to(device)
    apply_initialization(model, get_recommended_init(activation_name), activation_name)
    
    train_model(model, device, X_train, y_train, epochs=args.epochs)
    
    # Eval Data
    test_radii = [float(r) for r in args.test_radii.split(',')]
    accuracies = []
    
    for r in test_radii:
        # Ring test sets
        X_test, y_test = generate_data(4000, r, r, seed + 100 + int(10 * r))
        acc = evaluate(model, device, X_test, y_test)
        accuracies.append(acc)
        print(f"  {activation_name} r={r}: {acc:.4f}")
        
    auc = np.trapz(accuracies, test_radii) / (test_radii[-1] - test_radii[0])
    
    results = {
        'activation': activation_name,
        'seed': seed,
        'final_train_accuracy': float(evaluate(model, device, X_train, y_train)),
        'test_radii': test_radii,
        'test_accuracies': accuracies,
        'auc_over_radius': auc
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    if seed == 0 and args.make_plots:
        plot_task_and_decision_boundary(
            model=model,
            X_train=X_train,
            y_train=y_train,
            activation_name=activation_name,
            output_dir=act_dir,
            device=device,
            train_r_max=2.0,
            test_radii=test_radii,
            grid_range=max(test_radii) + 0.5,
        )
        plot_extrapolation_curve(test_radii, accuracies, activation_name, act_dir / 'extrapolation_curve.png')
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='runs/ood/angular')
    parser.add_argument('--test-radii', type=str, default='2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0')
    parser.add_argument('--force', action='store_true', help='Recompute all seeds even if cached')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    parser.add_argument('--make-plots', dest='make_plots', action='store_true', help='Generate seed0 plots per activation')
    parser.add_argument('--no-make-plots', dest='make_plots', action='store_false')
    parser.set_defaults(make_plots=True)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--controls-only', action='store_true')
    group.add_argument('--comprehensive', action='store_true')
    group.add_argument('--paper-set', action='store_true',
                       help='Run canonical 10-activation paper set: 6 Cap configs (a=1 and a=a*, fixed) + 4 controls')

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.controls_only:
        activations = get_control_activations()
    elif args.comprehensive:
        activations = get_control_activations()
        extended = get_extended_astar_activations()
        for name in ['TanhSoftCap_astar_fixed', 'SmoothNotchV2_1_fixed', 'QuinticNotch_1_learnable']:
            if name in extended:
                activations[name] = extended[name]
    elif args.paper_set:
        extended = get_extended_astar_activations()
        paper_cap_names = [
            'TanhSoftCap_1_fixed', 'TanhSoftCap_astar_fixed',
            'SmoothNotchV2_1_fixed', 'SmoothNotchV2_astar_fixed',
            'QuinticNotch_1_fixed', 'QuinticNotch_astar_fixed',
        ]
        activations = {}
        for name in paper_cap_names:
            if name in extended:
                activations[name] = extended[name]
            else:
                print(f"WARNING: {name} not found in get_extended_astar_activations()")
        activations.update(get_control_activations())
                
    print(f"Running Radial Sector Benchmark on {len(activations)} activations...")
    
    all_results = {}
    for name, fn in activations.items():
        seed_results = []
        for seed in range(args.n_seeds):
            res = run_benchmark(name, fn, seed, output_dir, args)
            seed_results.append(res)
        all_results[name] = seed_results
        
    # Summary
    summary = {
        '_meta': {
            'benchmark': 'radial_angular_sector',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_seeds': args.n_seeds,
            'epochs': args.epochs,
            'test_radii': [float(r) for r in args.test_radii.split(',')],
        }
    }
    for name, seeds in all_results.items():
        summary[name] = {
            'auc_mean': np.mean([s['auc_over_radius'] for s in seeds]),
            'auc_std': np.std([s['auc_over_radius'] for s in seeds])
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\nBenchmark Complete.")

if __name__ == '__main__':
    main()
