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

"""Dead-Zone Sparsity Analysis

Measures the fraction of Layer-1 and Layer-2 activations that fall in the exact-zero
dead zone (x <= -a) for notch variants, and near-zero region for smooth variants,
after training a simple MNIST MLP.

This produces the histograms/bar charts for §4.1 of the paper.

Usage:
    python mechanistic_interpretability/sparse_coding/dead_zone_sparsity_analysis.py
    python mechanistic_interpretability/sparse_coding/dead_zone_sparsity_analysis.py --quick
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Locate repo root (works at any depth in the tree)
_repo_p = Path(__file__).resolve().parent
while _repo_p != _repo_p.parent and not (_repo_p / 'softcap').is_dir():
    _repo_p = _repo_p.parent
project_root = _repo_p
sys.path.insert(0, str(project_root))

from softcap.activations import (
    SoftCap,
    SwishCap,
    SparseCap,
)


# a* values from variance map
A_STAR = {
    'TanhSoftCap': 2.890625,
    'SmoothNotchV2': 2.43359375,
    'QuinticNotch': 2.14,
}


class ProbeableMLP(nn.Module):
    """Simple MLP that exposes intermediate activations for dead-zone analysis."""
    def __init__(self, activation_fn: nn.Module, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.act = activation_fn
        self._pre_act_1 = None
        self._pre_act_2 = None
        self._post_act_1 = None
        self._post_act_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        self._pre_act_1 = self.fc1(x)
        self._post_act_1 = self.act(self._pre_act_1)
        self._pre_act_2 = self.fc2(self._post_act_1)
        self._post_act_2 = self.act(self._pre_act_2)
        return self.fc3(self._post_act_2)


def get_a_value(act_module: nn.Module) -> float:
    """Extract the current a value from an activation."""
    if hasattr(act_module, 'a'):
        return act_module.a.item()
    return float('nan')


def compute_dead_zone_stats(pre_act: torch.Tensor, a_val: float, has_dead_zone: bool) -> Dict[str, float]:
    """Compute dead-zone occupancy statistics."""
    flat = pre_act.detach().cpu().numpy().flatten()
    n = len(flat)

    stats = {
        'total_units': n,
        'mean_pre_act': float(np.mean(flat)),
        'std_pre_act': float(np.std(flat)),
    }

    if has_dead_zone:
        dead_mask = flat <= -a_val
        stats['dead_zone_frac'] = float(np.sum(dead_mask) / n)
        stats['dead_zone_count'] = int(np.sum(dead_mask))
    else:
        stats['dead_zone_frac'] = 0.0
        stats['dead_zone_count'] = 0

    # Near-zero fraction (|output| < 0.01 after activation)
    stats['near_zero_threshold'] = 0.01

    # Exact zero fraction for ReLU-like (pre_act <= 0)
    stats['relu_zero_frac'] = float(np.sum(flat <= 0) / n)

    return stats


def build_activations(quick: bool = False) -> Dict[str, Dict[str, Any]]:
    """Build activation configs for analysis."""
    configs = {}

    # Controls
    configs['ReLU'] = {
        'factory': lambda: nn.ReLU(),
        'has_dead_zone': False,
        'is_relu': True,
    }
    configs['GELU'] = {
        'factory': lambda: nn.GELU(),
        'has_dead_zone': False,
        'is_relu': False,
    }
    configs['Tanh'] = {
        'factory': lambda: nn.Tanh(),
        'has_dead_zone': False,
        'is_relu': False,
    }
    configs['SiLU'] = {
        'factory': lambda: nn.SiLU(),
        'has_dead_zone': False,
        'is_relu': False,
    }

    # SoftCap (TanhSoftCap) — hard zero for x <= 0
    for a_label, a_val in [('1', 1.0), ('astar', A_STAR['TanhSoftCap'])]:
        for mode in ['fixed', 'learnable']:
            name = f'TanhSoftCap_{a_label}_{mode}'
            def _factory(a=a_val, freeze=(mode == 'fixed')):
                act = SoftCap(a_init=a)
                if freeze:
                    act.a.requires_grad = False
                return act
            configs[name] = {
                'factory': _factory,
                'has_dead_zone': False,  # SoftCap has ReLU-like zero, not interval dead zone
                'is_relu': True,  # hard zero at x=0
            }

    # SmoothNotchV2 — asymptotic, no dead zone
    for a_label, a_val in [('1', 1.0), ('astar', A_STAR['SmoothNotchV2'])]:
        for mode in ['fixed', 'learnable']:
            name = f'SmoothNotchV2_{a_label}_{mode}'
            def _factory(a=a_val, freeze=(mode == 'fixed')):
                act = SwishCap(a_init=a)
                if freeze:
                    act.a.requires_grad = False
                return act
            configs[name] = {
                'factory': _factory,
                'has_dead_zone': False,
                'is_relu': False,
            }

    # QuinticNotch — dead zone for x <= -a
    for a_label, a_val in [('1', 1.0), ('astar', A_STAR['QuinticNotch'])]:
        for mode in ['fixed', 'learnable']:
            name = f'QuinticNotch_{a_label}_{mode}'
            def _factory(a=a_val, freeze=(mode == 'fixed')):
                act = SparseCap(a_init=a)
                if freeze:
                    act.a.requires_grad = False
                return act
            configs[name] = {
                'factory': _factory,
                'has_dead_zone': True,
                'is_relu': False,
            }

    if quick:
        # Subset for testing
        keep = ['ReLU', 'GELU', 'SmoothNotchV2_astar_fixed',
                'QuinticNotch_1_fixed', 'QuinticNotch_astar_fixed']
        configs = {k: v for k, v in configs.items() if k in keep}

    return configs


def run_analysis(
    configs: Dict[str, Dict[str, Any]],
    seeds: list,
    epochs: int,
    device: torch.device,
    output_dir: Path,
):
    """Run the dead-zone sparsity analysis."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    use_cuda = device.type == 'cuda'
    nw = 4 if use_cuda else 0
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=nw, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=nw, pin_memory=use_cuda)

    all_results = {}

    for act_name, act_cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Activation: {act_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if use_cuda:
                torch.cuda.manual_seed_all(seed)

            act_fn = act_cfg['factory']()
            model = ProbeableMLP(act_fn).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = F.cross_entropy(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate test accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    correct += (logits.argmax(1) == batch_y).sum().item()
                    total += batch_y.size(0)
            test_acc = correct / total

            # Collect dead-zone stats over test set
            all_pre_act_1 = []
            all_pre_act_2 = []
            all_post_act_1 = []
            all_post_act_2 = []
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    _ = model(batch_x)
                    all_pre_act_1.append(model._pre_act_1.cpu())
                    all_pre_act_2.append(model._pre_act_2.cpu())
                    all_post_act_1.append(model._post_act_1.cpu())
                    all_post_act_2.append(model._post_act_2.cpu())

            pre_act_1 = torch.cat(all_pre_act_1, dim=0)
            pre_act_2 = torch.cat(all_pre_act_2, dim=0)
            post_act_1 = torch.cat(all_post_act_1, dim=0)
            post_act_2 = torch.cat(all_post_act_2, dim=0)

            a_val = get_a_value(model.act)
            has_dz = act_cfg['has_dead_zone']

            l1_stats = compute_dead_zone_stats(pre_act_1, a_val, has_dz)
            l2_stats = compute_dead_zone_stats(pre_act_2, a_val, has_dz)

            # Near-zero fraction of outputs
            l1_near_zero = (post_act_1.abs() < 0.01).float().mean().item()
            l2_near_zero = (post_act_2.abs() < 0.01).float().mean().item()
            l1_exact_zero = (post_act_1 == 0).float().mean().item()
            l2_exact_zero = (post_act_2 == 0).float().mean().item()

            result = {
                'seed': seed,
                'test_accuracy': round(test_acc, 4),
                'a_value': round(a_val, 4) if not np.isnan(a_val) else None,
                'layer1': {
                    **l1_stats,
                    'output_near_zero_frac': round(l1_near_zero, 4),
                    'output_exact_zero_frac': round(l1_exact_zero, 4),
                },
                'layer2': {
                    **l2_stats,
                    'output_near_zero_frac': round(l2_near_zero, 4),
                    'output_exact_zero_frac': round(l2_exact_zero, 4),
                },
            }
            seed_results.append(result)
            print(f"  Seed {seed}: test_acc={test_acc:.4f}, "
                  f"L1 dead_zone={l1_stats['dead_zone_frac']:.3f}, "
                  f"L2 dead_zone={l2_stats['dead_zone_frac']:.3f}, "
                  f"L1 near_zero={l1_near_zero:.3f}, "
                  f"L1 exact_zero={l1_exact_zero:.3f}")

        # Aggregate across seeds
        mean_test_acc = np.mean([r['test_accuracy'] for r in seed_results])
        mean_l1_dz = np.mean([r['layer1']['dead_zone_frac'] for r in seed_results])
        mean_l2_dz = np.mean([r['layer2']['dead_zone_frac'] for r in seed_results])
        mean_l1_nz = np.mean([r['layer1']['output_near_zero_frac'] for r in seed_results])
        mean_l2_nz = np.mean([r['layer2']['output_near_zero_frac'] for r in seed_results])
        mean_l1_ez = np.mean([r['layer1']['output_exact_zero_frac'] for r in seed_results])
        mean_l2_ez = np.mean([r['layer2']['output_exact_zero_frac'] for r in seed_results])

        all_results[act_name] = {
            'seeds': seed_results,
            'aggregate': {
                'mean_test_accuracy': round(float(mean_test_acc), 4),
                'mean_l1_dead_zone_frac': round(float(mean_l1_dz), 4),
                'mean_l2_dead_zone_frac': round(float(mean_l2_dz), 4),
                'mean_l1_near_zero_frac': round(float(mean_l1_nz), 4),
                'mean_l2_near_zero_frac': round(float(mean_l2_nz), 4),
                'mean_l1_exact_zero_frac': round(float(mean_l1_ez), 4),
                'mean_l2_exact_zero_frac': round(float(mean_l2_ez), 4),
            },
            'has_dead_zone': act_cfg['has_dead_zone'],
        }
        print(f"  → Mean: test_acc={mean_test_acc:.4f}, "
              f"L1_dz={mean_l1_dz:.3f}, L2_dz={mean_l2_dz:.3f}, "
              f"L1_ez={mean_l1_ez:.3f}, L2_ez={mean_l2_ez:.3f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'dead_zone_sparsity_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'seeds': seeds,
            'results': all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Activation':<35} {'Test Acc':>8} {'L1 DZ%':>7} {'L1 EZ%':>7} {'L1 NZ%':>7} {'L2 DZ%':>7} {'L2 EZ%':>7}")
    print(f"{'-'*90}")
    for name, res in sorted(all_results.items()):
        agg = res['aggregate']
        print(f"{name:<35} {agg['mean_test_accuracy']:>8.4f} "
              f"{agg['mean_l1_dead_zone_frac']*100:>6.1f}% "
              f"{agg['mean_l1_exact_zero_frac']*100:>6.1f}% "
              f"{agg['mean_l1_near_zero_frac']*100:>6.1f}% "
              f"{agg['mean_l2_dead_zone_frac']*100:>6.1f}% "
              f"{agg['mean_l2_exact_zero_frac']*100:>6.1f}%")
    print(f"{'='*90}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Dead-Zone Sparsity Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick subset run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seeds')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--output-dir', type=str,
                        default='mechanistic_interpretability/sparse_coding/dead_zone_analysis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    configs = build_activations(quick=args.quick)
    print(f"Activations to analyze: {len(configs)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    run_analysis(
        configs=configs,
        seeds=args.seeds,
        epochs=args.epochs,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
