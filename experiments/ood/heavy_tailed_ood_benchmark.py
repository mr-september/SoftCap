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

"""Heavy-Tailed Input OOD Benchmark

Goal:
  Test whether activation functions maintain accuracy AND calibration when test
  inputs are drawn from a distribution heavier-tailed than training.

Design:
  - Task: Binary classification of D=16 isotropic Gaussian blobs (well-separated).
  - Train: N=5000 clean Gaussian samples.
  - Test: Gaussian samples with a fraction p replaced by Cauchy-contaminated
    versions carrying the same label but extreme-magnitude inputs.
  - Sweep: p ∈ {0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0}

Rationale for this task:
  Bounded activations (Cap family, Tanh) saturate at ±a/±1 for extreme inputs,
  preserving directional class signal. Unbounded activations (ReLU, GELU, SiLU)
  produce logit magnitudes proportional to input magnitude → overconfident /
  miscalibrated when inputs are extreme. The key differentiating metric vs mere
  accuracy is calibration: Cap variants produce lower but more accurate
  confidence estimates on contaminated samples.

Why Cauchy contamination:
  Cauchy is the canonical "problematic heavy tail" (undefined mean/variance).
  Framing as a contamination fraction (0%→100% Cauchy) gives a clean
  single-parameter sweep that is easy to interpret and compare.

Metrics (first- and second-order):
  Primary:
    - accuracy at each contamination fraction p
    - auc_over_p (summary statistic, normalised)
  Calibration (second-order / paper differentiator):
    - mean_max_conf_all     : mean(max softmax prob) over full test set
    - mean_max_conf_outlier : mean(max softmax prob) on contaminated samples only
    - ece_all               : Expected Calibration Error, full test set
    - ece_outlier           : ECE on contaminated samples only
  Representation:
    - mean_logit_gap_outlier : mean |logit_0 - logit_1| on contaminated samples
                               (unbounded → explodes; bounded → stays finite)

a-regime selection (reasoned):
  - Fixed a=1 (default, pre-scale) and fixed a=a* (variance-preserving init).
  - Learnable variants excluded: we want activation-function comparison, not
    optimisation-of-a comparison. Fixed a isolates the structural property.
  - Both a=1 and a=a* included to surface the effect of output scale.

Init: get_recommended_init() for each activation (kaiming for all Cap variants).

Usage:
    python heavy_tailed_ood_benchmark.py --controls-only --n-seeds 1 --epochs 10
    python heavy_tailed_ood_benchmark.py --paper-set --n-seeds 5 --epochs 100
    python heavy_tailed_ood_benchmark.py --all-activations --n-seeds 5 --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
from softcap.models import SimpleMLP
from softcap.initialization import get_recommended_init, apply_initialization


# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

MU = 0.5   # class centre offset per dimension (D-dimensional unit diagonal)
INPUT_DIM = 16


def _gaussian_blobs(n: int, dim: int, mu: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """n/2 samples per class from N(±mu*1_D, I)."""
    n0 = n // 2
    n1 = n - n0
    X0 = rng.standard_normal((n0, dim)).astype(np.float32) - mu
    X1 = rng.standard_normal((n1, dim)).astype(np.float32) + mu
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1, dtype=np.int64)
    return X, y


def _contaminate(X: np.ndarray, y: np.ndarray, p: float, rng: np.random.Generator, cauchy_scale: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace fraction p of samples with Cauchy-contaminated versions.

    The contaminated samples have the same labels but inputs drawn from
    Cauchy(class_centre, cauchy_scale) instead of Gaussian. Returns
    (X_mixed, y_mixed, outlier_mask).
    """
    n, dim = X.shape
    outlier_mask = np.zeros(n, dtype=bool)
    if p <= 0.0:
        return X.copy(), y.copy(), outlier_mask

    n_out = max(1, int(round(p * n)))
    idx = rng.choice(n, size=n_out, replace=False)
    outlier_mask[idx] = True

    X_out = X.copy()
    for i in idx:
        class_centre = (+MU if y[i] == 1 else -MU) * np.ones(dim, dtype=np.float32)
        # Cauchy: ratio of two standard normals (standard construction)
        numer = rng.standard_normal(dim).astype(np.float32)
        denom = np.abs(rng.standard_normal(1).astype(np.float32)) + 1e-8
        X_out[i] = class_centre + cauchy_scale * (numer / denom)

    return X_out, y.copy(), outlier_mask


# ──────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform-width bins on confidence)."""
    if len(probs) == 0:
        return float('nan')
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Last bin inclusive on right so confidence == 1.0 (float32 saturation
        # for unbounded activations with large logit gap) is captured correctly.
        if i == n_bins - 1:
            in_bin = conf >= lo
        else:
            in_bin = (conf >= lo) & (conf < hi)
        if in_bin.sum() == 0:
            continue
        frac = in_bin.sum() / len(conf)
        avg_conf = conf[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece += frac * abs(avg_conf - avg_acc)
    return float(ece)


@torch.no_grad()
def _get_probs_and_logits(model: nn.Module, X: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    logits = model(X_t).cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, logits


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _train(model: nn.Module, X: np.ndarray, y: np.ndarray,
           device: torch.device, epochs: int, lr: float, batch_size: int) -> Dict:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'accuracy': []}
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (model(bx).argmax(1) == by).sum().item()
            total += by.size(0)
        history['loss'].append(total_loss / len(loader))
        history['accuracy'].append(correct / total)
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss={history['loss'][-1]:.4f}  acc={history['accuracy'][-1]:.4f}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Per-activation benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_one(activation_name: str, activation_fn: nn.Module, seed: int,
            output_dir: Path, args: argparse.Namespace) -> Dict:
    print(f"\n{'='*60}\nHeavy-Tailed OOD: {activation_name} (seed={seed})\n{'='*60}")

    act_dir = output_dir / activation_name
    act_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = act_dir / f'metrics_seed{seed}.json'

    if args.resume and not args.force and metrics_path.exists():
        with open(metrics_path) as f:
            print(f"  ✓ Skipped (resume): {metrics_path.name}")
            return json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ── Training data ────────────────────────────────────────────────────────
    X_train, y_train = _gaussian_blobs(args.train_samples, INPUT_DIM, MU, rng)

    model = SimpleMLP(activation_fn, input_dim=INPUT_DIM,
                      hidden_dim=args.hidden_dim, output_dim=2, num_layers=4).to(device)
    init = get_recommended_init(activation_name)
    apply_initialization(model, init, activation_name)
    print(f"  Device={device}  init={init}")

    print("  Training…")
    history = _train(model, X_train, y_train, device,
                     args.epochs, args.lr, args.batch_size)
    train_acc = history['accuracy'][-1]

    # ── Evaluation ───────────────────────────────────────────────────────────
    contamination_fractions = [float(p) for p in args.contamination_fracs.split(',')]
    n_test = args.test_samples

    # Fixed test seed per activation+seed (reproducible but independent of train)
    test_rng_seed = seed + 10000

    results_per_p = []
    for p in contamination_fractions:
        test_rng = np.random.default_rng(test_rng_seed + int(p * 10000))
        X_clean, y_clean = _gaussian_blobs(n_test, INPUT_DIM, MU, test_rng)
        X_test, y_test, outlier_mask = _contaminate(
            X_clean, y_clean, p, test_rng, cauchy_scale=args.cauchy_scale)

        probs, logits = _get_probs_and_logits(model, X_test, device)
        preds = probs.argmax(axis=1)

        accuracy = float((preds == y_test).mean())
        ece_all = _ece(probs, y_test)
        mean_max_conf_all = float(probs.max(axis=1).mean())

        n_out = int(outlier_mask.sum())
        if n_out > 0:
            probs_out = probs[outlier_mask]
            logits_out = logits[outlier_mask]
            y_out = y_test[outlier_mask]
            acc_outlier = float((probs_out.argmax(1) == y_out).mean())
            ece_outlier = _ece(probs_out, y_out)
            mean_max_conf_outlier = float(probs_out.max(axis=1).mean())
            # mean absolute logit gap: proxy for "how extreme are the logits"
            mean_logit_gap_outlier = float(np.abs(logits_out[:, 0] - logits_out[:, 1]).mean())
        else:
            acc_outlier = accuracy
            ece_outlier = ece_all
            mean_max_conf_outlier = mean_max_conf_all
            mean_logit_gap_outlier = float(np.abs(logits[:, 0] - logits[:, 1]).mean())

        results_per_p.append({
            'p': p,
            'n_outliers': n_out,
            'accuracy': accuracy,
            'acc_outlier': acc_outlier,
            'ece_all': ece_all,
            'ece_outlier': ece_outlier,
            'mean_max_conf_all': mean_max_conf_all,
            'mean_max_conf_outlier': mean_max_conf_outlier,
            'mean_logit_gap_outlier': mean_logit_gap_outlier,
        })
        print(f"  p={p:.2f}: acc={accuracy:.4f}  acc_out={acc_outlier:.4f}  "
              f"conf_out={mean_max_conf_outlier:.3f}  logit_gap={mean_logit_gap_outlier:.2f}  "
              f"ece_out={ece_outlier:.3f}")

    # AUC over contamination fractions (trapezoidal, normalised to [0,1])
    accs = [r['accuracy'] for r in results_per_p]
    ps = [r['p'] for r in results_per_p]
    auc = float(np.trapz(accs, ps) / (ps[-1] - ps[0])) if ps[-1] > ps[0] else accs[0]

    record = {
        'activation': activation_name,
        'seed': seed,
        'final_train_accuracy': float(train_acc),
        'auc_over_contamination': auc,
        'contamination_results': results_per_p,
        'training_history': history,
        'hyperparameters': {
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'hidden_dim': args.hidden_dim,
            'input_dim': INPUT_DIM, 'train_samples': args.train_samples,
            'test_samples': args.test_samples, 'cauchy_scale': args.cauchy_scale,
            'mu': MU, 'init': init,
        },
    }
    with open(metrics_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation (per-activation)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_activation_summary(seed_records: List[Dict], activation_name: str, act_dir: Path):
    """4-panel summary: accuracy, confidence, logit-gap, ECE vs contamination p."""
    if not seed_records:
        return

    ps = [r['p'] for r in seed_records[0]['contamination_results']]
    metrics = {
        'accuracy': [], 'acc_outlier': [],
        'mean_max_conf_outlier': [], 'mean_logit_gap_outlier': [],
        'ece_outlier': [],
    }
    for rec in seed_records:
        for key in metrics:
            metrics[key].append([r[key] for r in rec['contamination_results']])

    def _stats(vals):
        arr = np.array(vals)
        return arr.mean(0), arr.std(0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Accuracy
    ax = axes[0]
    mu, sd = _stats(metrics['accuracy'])
    mu_out, sd_out = _stats(metrics['acc_outlier'])
    ax.plot(ps, mu, 'o-', label='All samples', linewidth=2)
    ax.fill_between(ps, mu - sd, mu + sd, alpha=0.2)
    ax.plot(ps, mu_out, 's--', label='Contaminated only', linewidth=1.5)
    ax.fill_between(ps, mu_out - sd_out, mu_out + sd_out, alpha=0.15)
    ax.set_xlabel('Contamination fraction p')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{activation_name}\nAccuracy vs Contamination')
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Confidence on outliers
    ax = axes[1]
    mu, sd = _stats(metrics['mean_max_conf_outlier'])
    ax.plot(ps, mu, 'o-', color='darkorange', linewidth=2)
    ax.fill_between(ps, mu - sd, mu + sd, alpha=0.2, color='darkorange')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Contamination fraction p')
    ax.set_ylabel('Mean max softmax prob')
    ax.set_title('Mean Confidence (Outliers)')
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)

    # Logit gap on outliers
    ax = axes[2]
    mu, sd = _stats(metrics['mean_logit_gap_outlier'])
    ax.plot(ps, mu, 'o-', color='firebrick', linewidth=2)
    ax.fill_between(ps, mu - sd, mu + sd, alpha=0.2, color='firebrick')
    ax.set_xlabel('Contamination fraction p')
    ax.set_ylabel('Mean |logit₀ − logit₁|')
    ax.set_title('Logit Gap (Outliers)')
    ax.grid(True, alpha=0.3)

    # ECE on outliers
    ax = axes[3]
    mu, sd = _stats(metrics['ece_outlier'])
    ax.plot(ps, mu, 'o-', color='steelblue', linewidth=2)
    ax.fill_between(ps, mu - sd, mu + sd, alpha=0.2, color='steelblue')
    ax.set_xlabel('Contamination fraction p')
    ax.set_ylabel('ECE')
    ax.set_title('Calibration Error (Outliers)')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Heavy-Tailed OOD — {activation_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(act_dir / 'summary.png', dpi=180, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-activation comparison figures
# ──────────────────────────────────────────────────────────────────────────────

# Paper-representative display names
_DISPLAY = {
    'TanhSoftCap_1_fixed':        'SoftCap (a=1)',
    'TanhSoftCap_astar_fixed':    'SoftCap (a=a*)',
    'SmoothNotchV2_1_fixed':     'SwishCap (a=1)',
    'SmoothNotchV2_astar_fixed': 'SwishCap (a=a*)',
    'QuinticNotch_1_fixed':      'SparseCap (a=1)',
    'QuinticNotch_astar_fixed':  'SparseCap (a=a*)',
    'ReLU': 'ReLU', 'Tanh': 'Tanh', 'GELU': 'GELU', 'SiLU': 'SiLU',
}

_PALETTE = {
    'SoftCap (a=1)':        '#1f77b4',
    'SoftCap (a=a*)':       '#1f77b4',
    'SwishCap (a=1)':       '#2ca02c',
    'SwishCap (a=a*)':      '#2ca02c',
    'SparseCap (a=1)':      '#9467bd',
    'SparseCap (a=a*)':     '#9467bd',
    'ReLU':  '#d62728',
    'Tanh':  '#ff7f0e',
    'GELU':  '#8c564b',
    'SiLU':  '#e377c2',
}

_STYLE = {  # (a=1 → solid, a=a* → dashed)
    'SoftCap (a=1)':        '-',
    'SoftCap (a=a*)':       '--',
    'SwishCap (a=1)':       '-',
    'SwishCap (a=a*)':      '--',
    'SparseCap (a=1)':      '-',
    'SparseCap (a=a*)':     '--',
    'ReLU': '-', 'Tanh': '-', 'GELU': '-', 'SiLU': '-',
}


def _make_comparison_figures(summary: Dict, output_dir: Path):
    """4-panel comparison figure across all activations."""
    acts = [k for k in summary['results'] if k not in ('benchmark', 'timestamp')]
    ps = summary.get('contamination_fractions', [])
    if not ps:
        return

    metrics_keys = [
        ('accuracy',                  'Accuracy',              (0.4, 1.02),  'Accuracy vs Contamination'),
        ('mean_max_conf_outlier',     'Mean Confidence\n(Outliers)', (0.45, 1.02), 'Confidence on Contaminated Samples'),
        ('mean_logit_gap_outlier',    'Mean Logit Gap\n(Outliers)',  None,         'Logit Gap on Contaminated Samples'),
        ('ece_outlier',               'ECE (Outliers)',         (0, None),    'Calibration Error on Contaminated Samples'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, (metric, ylabel, ylim, title) in zip(axes, metrics_keys):
        for act_name in acts:
            display = _DISPLAY.get(act_name, act_name)
            res = summary['results'][act_name]
            mu = np.array(res.get(f'{metric}_mean', []))
            sd = np.array(res.get(f'{metric}_std', []))
            if len(mu) == 0:
                continue
            color = _PALETTE.get(display, '#999999')
            ls = _STYLE.get(display, '-')
            ax.plot(ps, mu, marker='o', markersize=4, linewidth=1.8,
                    color=color, linestyle=ls, label=display)
            if sd.sum() > 0:
                ax.fill_between(ps, mu - sd, mu + sd, alpha=0.12, color=color)

        if ylim:
            ax.set_ylim(*[v for v in ylim if v is not None] if None not in ylim else [])
        ax.set_xlabel('Contamination fraction p', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if metric == 'accuracy':
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.12))
    plt.suptitle('Heavy-Tailed Input OOD Benchmark — All Activations', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.savefig(output_dir / 'comparison_all.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Bar chart: AUC over contamination
    fig, ax = plt.subplots(figsize=(12, 5))
    names_sorted = sorted(acts, key=lambda a: -summary['results'][a].get('auc_over_contamination_mean', 0))
    bar_labels = [_DISPLAY.get(n, n) for n in names_sorted]
    bar_vals = [summary['results'][n].get('auc_over_contamination_mean', 0) for n in names_sorted]
    bar_errs = [summary['results'][n].get('auc_over_contamination_std', 0) for n in names_sorted]
    colors = [_PALETTE.get(l, '#999999') for l in bar_labels]
    ax.barh(range(len(bar_vals)), bar_vals, xerr=bar_errs, color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.set_xlabel('AUC (accuracy) over contamination fraction')
    ax.set_title('Heavy-Tailed OOD: AUC Summary (↑ better)')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_barplot.png', dpi=200, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _get_activations_for_mode(args) -> Dict[str, nn.Module]:
    """Return the activation dict for the requested run mode."""
    if args.controls_only:
        return get_control_activations()

    if args.paper_set:
        # Three Cap variants × 2 a-regimes + 4 controls = 10 activations
        from softcap.activations import SoftCap, SwishCap, SparseCap
        A_STAR = {'TanhSoftCap': 2.89, 'SmoothNotchV2': 2.434, 'QuinticNotch': 2.14}

        def _make_fixed(cls, a):
            m = cls(a_init=a); m.a.requires_grad_(False); return m

        acts = {
            'TanhSoftCap_1_fixed':        _make_fixed(SoftCap, 1.0),
            'TanhSoftCap_astar_fixed':    _make_fixed(SoftCap, A_STAR['TanhSoftCap']),
            'SmoothNotchV2_1_fixed':     _make_fixed(SwishCap, 1.0),
            'SmoothNotchV2_astar_fixed': _make_fixed(SwishCap, A_STAR['SmoothNotchV2']),
            'QuinticNotch_1_fixed':      _make_fixed(SparseCap, 1.0),
            'QuinticNotch_astar_fixed':  _make_fixed(SparseCap, A_STAR['QuinticNotch']),
        }
        acts.update(get_control_activations())
        return acts

    if args.comprehensive:
        acts = get_extended_astar_activations()
        acts.update(get_control_activations())
        return acts

    if args.all_activations:
        return get_standard_experimental_set()

    if args.activation:
        all_acts = {**get_standard_experimental_set(), **get_extended_astar_activations()}
        if args.activation not in all_acts:
            raise ValueError(f'Unknown activation: {args.activation}')
        return {args.activation: all_acts[args.activation]}

    # Default: paper set
    args.paper_set = True
    return _get_activations_for_mode(args)


class HeavyTailedOODBenchmark:
    @staticmethod
    def run(args: argparse.Namespace):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        activations = _get_activations_for_mode(args)
        contamination_fractions = [float(p) for p in args.contamination_fracs.split(',')]

        print(f'\nHeavy-Tailed Input OOD Benchmark')
        print(f'Activations: {len(activations)}  Seeds: {args.n_seeds}')
        print(f'Contamination fractions: {contamination_fractions}')
        print(f'Output: {output_dir}\n')

        all_records: Dict[str, List[Dict]] = {}
        for act_name, act_fn in activations.items():
            seed_records = []
            for seed in range(args.n_seeds):
                seed_records.append(run_one(act_name, act_fn, seed, output_dir, args))
            all_records[act_name] = seed_records
            act_dir = output_dir / act_name
            if args.n_seeds > 1:
                _plot_activation_summary(seed_records, act_name, act_dir)

        # Build summary with per-metric mean/std over seeds, per contamination fraction
        summary_results = {}
        for act_name, seed_records in all_records.items():
            metric_names = [
                'accuracy', 'acc_outlier',
                'mean_max_conf_all', 'mean_max_conf_outlier',
                'mean_logit_gap_outlier', 'ece_all', 'ece_outlier',
            ]
            per_metric = {m: [] for m in metric_names}
            aucs = [r.get('auc_over_contamination', float('nan')) for r in seed_records]

            for rec in seed_records:
                for m in metric_names:
                    per_metric[m].append([
                        row.get(m, float('nan'))
                        for row in rec.get('contamination_results', [])
                    ])

            entry = {
                'auc_over_contamination_mean': float(np.nanmean(aucs)),
                'auc_over_contamination_std':  float(np.nanstd(aucs)),
            }
            for m in metric_names:
                arr = np.array(per_metric[m])  # (n_seeds, n_fracs)
                entry[f'{m}_mean'] = arr.mean(0).tolist()
                entry[f'{m}_std'] = arr.std(0).tolist()
            summary_results[act_name] = entry

        summary = {
            'benchmark': 'heavy_tailed_ood',
            'timestamp': timestamp,
            'n_seeds': args.n_seeds,
            'num_activations': len(all_records),
            'contamination_fractions': contamination_fractions,
            'cauchy_scale': args.cauchy_scale,
            'input_dim': INPUT_DIM,
            'mu': MU,
            'results': summary_results,
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        _make_comparison_figures(summary, output_dir)

        # Print table
        print(f"\n{'='*80}\nBENCHMARK SUMMARY (sorted by AUC over contamination)\n{'='*80}")
        print(f"{'Activation':<35} {'AUC':>8} {'Conf@p=1 (out)':>15} {'ECE@p=1 (out)':>14} {'LogitGap@p=1':>13}")
        print('-' * 90)
        for name, res in sorted(summary_results.items(), key=lambda x: -x[1]['auc_over_contamination_mean']):
            auc = res['auc_over_contamination_mean']
            conf_last = res['mean_max_conf_outlier_mean'][-1] if res.get('mean_max_conf_outlier_mean') else float('nan')
            ece_last  = res['ece_outlier_mean'][-1]          if res.get('ece_outlier_mean') else float('nan')
            lgap_last = res['mean_logit_gap_outlier_mean'][-1] if res.get('mean_logit_gap_outlier_mean') else float('nan')
            print(f"{name:<35} {auc:>8.4f} {conf_last:>15.4f} {ece_last:>14.4f} {lgap_last:>13.2f}")

        print(f'\nSaved to: {output_dir}')
        return all_records


def main():
    parser = argparse.ArgumentParser(description='Heavy-Tailed Input OOD Benchmark')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--paper-set',       action='store_true', default=False,
                      help='3 Cap × 2 a-regimes + 4 controls (=10, DEFAULT)')
    mode.add_argument('--controls-only',   action='store_true')
    mode.add_argument('--all-activations', action='store_true')
    mode.add_argument('--comprehensive',   action='store_true')
    mode.add_argument('--activation',      type=str)

    parser.add_argument('--contamination-fracs', type=str,
                        default='0.0,0.05,0.10,0.20,0.35,0.50,0.75,1.0')
    parser.add_argument('--cauchy-scale',  type=float, default=3.0,
                        help='Scale of Cauchy noise (default 3.0 = moderately extreme)')
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--test-samples',  type=int, default=2000)
    parser.add_argument('--hidden-dim',    type=int, default=128)
    parser.add_argument('--epochs',        type=int, default=100)
    parser.add_argument('--batch-size',    type=int, default=256)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--n-seeds',       type=int, default=5)
    parser.add_argument('--output-dir',    type=str,
                        default='mechanistic_interpretability/latent_geometry/ood/heavy_tailed/standard')
    parser.add_argument('--resume',        dest='resume', action='store_true', default=True)
    parser.add_argument('--no-resume',     dest='resume', action='store_false')
    parser.add_argument('--force',         action='store_true', default=False)

    args = parser.parse_args()
    HeavyTailedOODBenchmark.run(args)


if __name__ == '__main__':
    main()
