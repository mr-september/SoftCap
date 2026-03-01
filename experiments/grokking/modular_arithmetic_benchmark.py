#!/usr/bin/env python3
"""Modular Arithmetic Grokking Benchmark (Thrust 4)

Goal
- Replicate grokking phenomenon on modular arithmetic tasks
- Test SoftCap's effect on phase transition dynamics (memorization → generalization)
- Explore extrapolation to unseen operand ranges, moduli, and operations

Modes:
- standard: Classic grokking setup (80/20 train/test split on (a+b) mod p)
- modulus:  Train on mod 97, test on mod 101, 103, 107
- range:    Train on operands [0,50], test on [51,96]
- operation: Train on addition, test on subtraction

Outputs
- Per-activation directory under --output-dir:
    metrics_seed<seed>.json
    training_curves.png
- Aggregate summary.json with mean/std across seeds.

WSL example
  wsl -e bash -lc 'cd /mnt/e/SoftCap; source softcap_env/bin/activate; \\
    python mechanistic_interpretability/grokking/modular_arithmetic_benchmark.py --all-activations --mode standard'
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import platform
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

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
    get_astar_activations,
    get_extended_astar_activations,
)
from softcap.activations import get_default_activations
from softcap.initialization import get_recommended_init




def initialize_linear_layers(model: nn.Module, scheme: str) -> None:
    """Initialize nn.Linear layers using a named scheme."""
    scheme = (scheme or "xavier").lower()
    if scheme not in {"xavier", "kaiming", "orthogonal"}:
        raise ValueError(f"Unknown init scheme: {scheme}")

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if scheme == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "kaiming":
                # Match the initialization-sensitivity setup (linear nonlinearity).
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="linear")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class SimpleMLP(nn.Module):
    """Simple MLP for modular arithmetic; follows grokking paper architecture."""

    def __init__(self, num_classes: int, embed_dim: int, hidden_dim: int, activation_fn: nn.Module):
        super().__init__()
        # Embeddings for each operand (a, b)
        self.embed_a = nn.Embedding(num_classes, embed_dim)
        self.embed_b = nn.Embedding(num_classes, embed_dim)
        
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = type(activation_fn)() if hasattr(activation_fn, '__class__') else activation_fn
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea = self.embed_a(a)
        eb = self.embed_b(b)
        x = torch.cat([ea, eb], dim=-1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_hidden(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, hidden) where hidden is the post-activation output of fc2."""
        ea = self.embed_a(a)
        eb = self.embed_b(b)
        x = torch.cat([ea, eb], dim=-1)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        logits = self.fc3(h2)
        return logits, h2


def save_pair_representations(
    model: SimpleMLP,
    *,
    modulus: int,
    device: torch.device,
    output_dir: Path,
    seed: int,
    batch_size: int = 2048,
) -> None:
    """Save composed pair representations for all (a,b) in [0,modulus).

    Operand embeddings are not discriminative (they become Fourier-like even when the
    model fails to generalize). This saves a composed representation: the hidden state
    after (a,b) are combined and passed through fc1+act1+fc2+act2.
    """

    model.eval()

    a = torch.arange(modulus, device=device)
    b = torch.arange(modulus, device=device)
    grid_a, grid_b = torch.meshgrid(a, b, indexing='ij')
    flat_a = grid_a.reshape(-1)
    flat_b = grid_b.reshape(-1)

    h_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, flat_a.numel(), batch_size):
            aa = flat_a[i : i + batch_size]
            bb = flat_b[i : i + batch_size]
            _, h = model.forward_with_hidden(aa, bb)
            h_chunks.append(h.detach().cpu().numpy().astype(np.float32, copy=False))

    h_all = np.concatenate(h_chunks, axis=0).reshape(modulus, modulus, -1)
    np.save(output_dir / f"pair_hidden_seed{seed}.npy", h_all)

    # Canonical full-grid labels (useful across modes).
    a_np = np.arange(modulus, dtype=np.int64)
    b_np = np.arange(modulus, dtype=np.int64)
    y_add = (a_np[:, None] + b_np[None, :]) % modulus
    y_sub = (a_np[:, None] - b_np[None, :]) % modulus
    np.save(output_dir / "pair_y_add.npy", y_add.astype(np.int64, copy=False))
    np.save(output_dir / "pair_y_sub.npy", y_sub.astype(np.int64, copy=False))

    # Readout parameters (often where Fourier structure shows up in grokking).
    np.save(
        output_dir / f"readout_W_seed{seed}.npy",
        model.fc3.weight.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    np.save(
        output_dir / f"readout_b_seed{seed}.npy",
        model.fc3.bias.detach().cpu().numpy().astype(np.float32, copy=False),
    )

    meta = {
        'schema_version': '1',
        'modulus': int(modulus),
        'seed': int(seed),
        'repr': 'post_act_fc2',
        'shape': [int(modulus), int(modulus), int(h_all.shape[-1])],
        'batch_size': int(batch_size),
    }
    with open(output_dir / f"pair_repr_meta_seed{seed}.json", 'w') as f:
        json.dump(meta, f, indent=2)


def generate_modular_data(
    modulus: int,
    operation: str = 'add',
    train_fraction: float = 0.8,
    operand_min: int = 0,
    operand_max: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate modular arithmetic dataset.
    
    Returns:
        train_a, train_b, train_y, test_a, test_b, test_y as numpy arrays
    """
    rng = np.random.default_rng(seed)
    
    if operand_max is None:
        operand_max = modulus
    
    # Generate all pairs in range
    pairs = []
    for a in range(operand_min, operand_max):
        for b in range(operand_min, operand_max):
            if operation == 'add':
                y = (a + b) % modulus
            elif operation == 'sub':
                y = (a - b) % modulus
            elif operation == 'mul':
                y = (a * b) % modulus
            else:
                raise ValueError(f"Unknown operation: {operation}")
            pairs.append((a, b, y))
    
    pairs = np.array(pairs)
    rng.shuffle(pairs)
    
    split_idx = int(len(pairs) * train_fraction)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    return (
        train_pairs[:, 0].astype(np.int64),
        train_pairs[:, 1].astype(np.int64),
        train_pairs[:, 2].astype(np.int64),
        test_pairs[:, 0].astype(np.int64),
        test_pairs[:, 1].astype(np.int64),
        test_pairs[:, 2].astype(np.int64),
    )


def generate_range_extrapolation_data(
    modulus: int,
    train_max: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Generate data for range extrapolation mode.
    
    Train: operands in [0, train_max)
    Test: operands in [train_max, modulus)
    """
    rng = np.random.default_rng(seed)
    
    train_pairs = []
    for a in range(train_max):
        for b in range(train_max):
            y = (a + b) % modulus
            train_pairs.append((a, b, y))
    
    test_pairs = []
    for a in range(train_max, modulus):
        for b in range(train_max, modulus):
            y = (a + b) % modulus
            test_pairs.append((a, b, y))
    
    train_pairs = np.array(train_pairs)
    test_pairs = np.array(test_pairs)
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    
    return (
        train_pairs[:, 0].astype(np.int64),
        train_pairs[:, 1].astype(np.int64),
        train_pairs[:, 2].astype(np.int64),
        test_pairs[:, 0].astype(np.int64),
        test_pairs[:, 1].astype(np.int64),
        test_pairs[:, 2].astype(np.int64),
    )


def generate_operation_extrapolation_data(
    modulus: int,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Generate data for operation extrapolation mode.
    
    Train: addition
    Test: subtraction on same pairs
    """
    rng = np.random.default_rng(seed)
    
    pairs = []
    for a in range(modulus):
        for b in range(modulus):
            pairs.append((a, b))
    
    pairs = np.array(pairs)
    rng.shuffle(pairs)
    
    split_idx = int(len(pairs) * train_fraction)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    # Train on addition
    train_a = train_pairs[:, 0].astype(np.int64)
    train_b = train_pairs[:, 1].astype(np.int64)
    train_y = ((train_a + train_b) % modulus).astype(np.int64)
    
    # Test on subtraction
    test_a = test_pairs[:, 0].astype(np.int64)
    test_b = test_pairs[:, 1].astype(np.int64)
    test_y = ((test_a - test_b) % modulus).astype(np.int64)
    
    return train_a, train_b, train_y, test_a, test_b, test_y


def train_model(
    model: nn.Module,
    train_a: torch.Tensor,
    train_b: torch.Tensor,
    train_y: torch.Tensor,
    test_a: torch.Tensor,
    test_b: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    a_weight_decay: Optional[float] = None,
    eval_interval: int = 10,
    enable_early_stopping: bool = True,
) -> Dict[str, list]:
    """Train model and return history with grokking detection.
    
    Early stopping criteria (if enabled):
      - Success: test_acc > 0.99 for 3 consecutive evals → stop
      - Failure: after epoch 2000, if test_acc < 0.1 and sparsity > 0.95 → stop
      - Collapse: sparsity > 0.98 for 10 consecutive evals → stop
    """

    # AdamW applies weight decay as a parameter-space shrinkage. For SoftCap-style
    # activations with learnable scalar parameters (e.g., `a`), this can be a major
    # confounder. Optionally override WD for parameters named '*.a'.
    if a_weight_decay is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        a_params = []
        non_a_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith('.a'):
                a_params.append(param)
            else:
                non_a_params.append(param)

        param_groups = []
        if non_a_params:
            param_groups.append({'params': non_a_params, 'weight_decay': float(weight_decay)})
        if a_params:
            param_groups.append({'params': a_params, 'weight_decay': float(a_weight_decay)})

        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'sparsity': [],
        'weight_norm': [],
    }
    
    # Early stopping trackers
    consecutive_success = 0
    consecutive_collapse = 0

    model.train()
    for epoch in range(epochs):
        # Full batch training
        optimizer.zero_grad()
        
        # Forward pass capturing intermediates for sparsity check
        # We need to hook or modify forward to get activations.
        # Alternatively, since it's a simple MLP, we can just re-run forward for metrics.
        # Efficient approach: Just run forward normally for training.
        logits = model(train_a, train_b)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
        
        # Evaluate periodically
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get metrics and sparsity
                # We need to manually run layers to get activations for sparsity
                ea = model.embed_a(train_a)
                eb = model.embed_b(train_b)
                x = torch.cat([ea, eb], dim=-1)
                
                # Layer 1
                pre_act1 = model.fc1(x)
                act1 = model.act1(pre_act1)
                
                # Check sparsity (fraction of zeros)
                # For smooth functions like GELU, we use a small threshold
                # For Hard functions, exact zero works.
                # Let's use a strict threshold < 1e-6 to be safe for float precision
                is_zero = (act1.abs() < 1e-6).float()
                sparsity = is_zero.mean().item()
                
                # Weight norm (L2)
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()).sqrt().item()

                # Accuracies
                logits_train = model.fc3(model.act2(model.fc2(act1)))
                train_pred = logits_train.argmax(dim=1)
                train_acc = (train_pred == train_y).float().mean().item()
                
                test_pred = model(test_a, test_b).argmax(dim=1)
                test_acc = (test_pred == test_y).float().mean().item()

            model.train()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['sparsity'].append(sparsity)
            history['weight_norm'].append(l2_norm)
            
            if epoch % (eval_interval * 10) == 0:
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}, Sparsity={sparsity:.4f}")
            
            # Early stopping checks
            if enable_early_stopping:
                # Success: stable high test accuracy
                if test_acc > 0.99:
                    consecutive_success += 1
                    if consecutive_success >= 3:
                        print(f"  Early stop: grokked at epoch {epoch} (test_acc={test_acc:.4f})")
                        break
                else:
                    consecutive_success = 0
                
                # Collapse: persistent dead network (high sparsity)
                if sparsity > 0.98:
                    consecutive_collapse += 1
                    if consecutive_collapse >= 10:
                        print(f"  Early stop: network collapsed at epoch {epoch} (sparsity={sparsity:.4f})")
                        break
                else:
                    consecutive_collapse = 0
                
                # Clear failure: after 2000 epochs, if acc is terrible and network is dead
                if epoch >= 2000 and test_acc < 0.1 and sparsity > 0.95:
                    print(f"  Early stop: failed to grok by epoch {epoch} (test_acc={test_acc:.4f}, sparsity={sparsity:.4f})")
                    break
    
    return history


def detect_grokking(history: Dict[str, list], threshold: float = 0.95) -> Optional[int]:
    """Detect epoch where grokking occurred (test acc crosses threshold after train acc = 1)."""
    train_saturated = False
    for i, (epoch, train_acc, test_acc) in enumerate(zip(
        history['epoch'], history['train_acc'], history['test_acc']
    )):
        if train_acc > 0.99:
            train_saturated = True
        if train_saturated and test_acc >= threshold:
            return epoch
    return None


def plot_training_curves(
    history: Dict[str, list],
    activation_name: str,
    output_path: Path,
    grokking_epoch: Optional[int] = None,
):
    """Plot training curves showing grokking dynamics and sparsity."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    epochs = history['epoch']
    
    # 1. Loss & Weight Norm
    ax1.semilogy(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, history['weight_norm'], 'g--', alpha=0.5, label='Weight L2')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log)')
    ax1_twin.set_ylabel('Weight Norm')
    ax1.set_title(f'{activation_name}: Loss & Regularization')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 2. Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Acc')
    ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='Threshold')
    if grokking_epoch is not None:
        ax2.axvline(grokking_epoch, color='green', linestyle='--', alpha=0.7, label=f'Grok @ {grokking_epoch}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{activation_name}: Generalization')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Sparsity
    ax3.plot(epochs, history['sparsity'], 'm-', linewidth=2, label='Act Sparsity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity (Fraction Zeros)')
    ax3.set_title(f'{activation_name}: Activation Sparsity')
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_benchmark_for_activation(
    activation_name: str,
    activation_fn: nn.Module,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, any]:
    """Run complete benchmark for a single activation function and seed."""
    activation_dir = output_dir / activation_name
    activation_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics_path = activation_dir / f"metrics_seed{seed}.json"

    if args.resume and seed_metrics_path.exists() and not args.force:
        with open(seed_metrics_path, 'r') as f:
            print(f"  [OK] Skipped (resume): {seed_metrics_path.name}")
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Running Grokking Benchmark: {activation_name} (seed={seed}, mode={args.mode})")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data based on mode
    modulus = args.modulus
    
    if args.mode == 'standard':
        train_a, train_b, train_y, test_a, test_b, test_y = generate_modular_data(
            modulus=modulus, operation='add', train_fraction=0.8, seed=seed
        )
    elif args.mode == 'range':
        train_a, train_b, train_y, test_a, test_b, test_y = generate_range_extrapolation_data(
            modulus=modulus, train_max=modulus // 2, seed=seed
        )
    elif args.mode == 'operation':
        train_a, train_b, train_y, test_a, test_b, test_y = generate_operation_extrapolation_data(
            modulus=modulus, train_fraction=0.8, seed=seed
        )
    elif args.mode == 'modulus':
        # Train on base modulus
        train_a, train_b, train_y, _, _, _ = generate_modular_data(
            modulus=modulus, operation='add', train_fraction=1.0, seed=seed
        )
        # Test on different moduli (will be handled separately)
        test_a, test_b, test_y = None, None, None
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Convert to tensors
    train_a_t = torch.tensor(train_a, dtype=torch.long, device=device)
    train_b_t = torch.tensor(train_b, dtype=torch.long, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.long, device=device)
    
    if test_a is not None:
        test_a_t = torch.tensor(test_a, dtype=torch.long, device=device)
        test_b_t = torch.tensor(test_b, dtype=torch.long, device=device)
        test_y_t = torch.tensor(test_y, dtype=torch.long, device=device)
    else:
        # For modulus mode, we need a placeholder
        test_a_t = train_a_t[:100]
        test_b_t = train_b_t[:100]
        test_y_t = train_y_t[:100]

    # Create model
    model = SimpleMLP(
        num_classes=modulus,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        activation_fn=activation_fn,
    ).to(device)

    # Initialize weights (configurable)
    init_choice = (getattr(args, 'weight_init', 'xavier') or 'xavier').lower()
    if init_choice == 'auto':
        init_choice = get_recommended_init(activation_name)
    initialize_linear_layers(model, init_choice)

    print(f"Training on {len(train_a)} pairs...")
    history = train_model(
        model, train_a_t, train_b_t, train_y_t,
        test_a_t, test_b_t, test_y_t,
        device, args.epochs, args.lr, args.weight_decay,
        getattr(args, 'a_weight_decay', None),
        eval_interval=args.eval_interval,
    )

    grokking_epoch = detect_grokking(history, threshold=0.95)
    
    # For modulus mode, evaluate on multiple test moduli
    modulus_test_results = {}
    if args.mode == 'modulus':
        test_moduli = [101, 103, 107]
        model.eval()
        with torch.no_grad():
            for test_mod in test_moduli:
                # Generate test data for this modulus
                _, _, _, ta, tb, ty = generate_modular_data(
                    modulus=test_mod, operation='add', train_fraction=0.0, seed=seed
                )
                # Clamp to vocab size
                ta = np.clip(ta, 0, modulus - 1)
                tb = np.clip(tb, 0, modulus - 1)
                ty = (ta + tb) % test_mod  # Recompute for correct labels
                
                ta_t = torch.tensor(ta, dtype=torch.long, device=device)
                tb_t = torch.tensor(tb, dtype=torch.long, device=device)
                ty_t = torch.tensor(ty % modulus, dtype=torch.long, device=device)  # Clamp labels
                
                pred = model(ta_t, tb_t).argmax(dim=1)
                acc = (pred == ty_t).float().mean().item()
                modulus_test_results[f'test_acc_mod{test_mod}'] = acc
                print(f"  Test accuracy on mod {test_mod}: {acc:.4f}")

    results = {
        'schema_version': '1',
        'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'command': ' '.join(sys.argv) if sys.argv else None,
        'code': {
            'repo_root': str(Path(__file__).resolve().parents[2]),
            'git_commit': None,
        },
        'environment': {
            'python': platform.python_version(),
            'torch': getattr(torch, '__version__', 'unknown'),
            'device': str(device),
        },
        'benchmark': 'grokking_modular_arithmetic',
        'mode': args.mode,
        'activation': activation_name,
        'seed': seed,
        'modulus': modulus,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'grokking_epoch': grokking_epoch,
        'final_sparsity': history['sparsity'][-1],
        'epochs_trained': args.epochs,
        'training_history': history,
        **modulus_test_results,
        'standardization': {
            'a_regime': (
                'fixed_astar' if activation_name.endswith('_astar_fixed') or activation_name.endswith('_astar') else
                'learnable_init_astar' if activation_name.endswith('_astar_learnable') else
                'unknown'
            ),
            'a_init': (
                'astar' if '_astar_' in activation_name or activation_name.endswith('_astar') else
                'unknown'
            ),
            'a_value': (float(getattr(activation_fn, 'a').detach().cpu().item()) if hasattr(activation_fn, 'a') else None),
            'weight_init': init_choice,
        },
        'hyperparameters': {
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'a_weight_decay': getattr(args, 'a_weight_decay', None),
            'weight_init': init_choice,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
        },
    }

    with open(seed_metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Optional artifacts for mechanistic analysis (kept lightweight).
    if getattr(args, 'save_embeddings', False):
        np.save(activation_dir / f"embed_a_seed{seed}.npy", model.embed_a.weight.detach().cpu().numpy())
        np.save(activation_dir / f"embed_b_seed{seed}.npy", model.embed_b.weight.detach().cpu().numpy())

    if getattr(args, 'save_pair_reprs', False):
        save_pair_representations(
            model,
            modulus=modulus,
            device=device,
            output_dir=activation_dir,
            seed=seed,
            batch_size=int(getattr(args, 'pair_repr_batch', 2048)),
        )

    if getattr(args, 'save_checkpoint', False):
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'activation': activation_name,
                'seed': seed,
                'modulus': modulus,
                'embed_dim': int(args.embed_dim),
                'hidden_dim': int(args.hidden_dim),
                'weight_init': init_choice,
                'hyperparameters': results.get('hyperparameters', {}),
                'standardization': results.get('standardization', {}),
            },
            activation_dir / f"checkpoint_seed{seed}.pt",
        )

    if seed == 0:
        plot_training_curves(
            history, activation_name,
            activation_dir / 'training_curves.png',
            grokking_epoch,
        )

    return results


class GrokkingBenchmark:
    """Orchestrator class for Grokking benchmarks."""
    
    @staticmethod
    def run(args: argparse.Namespace) -> Dict[str, any]:
        """Run benchmark for selected activations."""

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print("\nGrokking Modular Arithmetic Benchmark (Thrust 4)")
        print(f"Mode: {args.mode}")
        print(f"Output directory: {output_dir}")

        # Get activations
        if args.controls_only:
            activations = get_control_activations()
            print("\nRunning controls-only mode (ReLU, Tanh, GELU, SiLU)")
        elif args.all_activations:
            activations = get_standard_experimental_set()
            print("\nRunning standard experimental set (Parametric* core + controls)")
        elif args.astar:
            activations = get_astar_activations()
            activations.update(get_control_activations())  # Include baselines
            print("\nRunning canonical a* variants + controls")
        elif args.extended_astar:
            activations = get_extended_astar_activations()
            print("\nRunning extended a* conditions (canonical variants fixed/learnable)")
        elif args.activation:
            # Check in both standard and a* sets
            all_acts = get_standard_experimental_set()
            all_acts.update(get_astar_activations())
            all_acts.update(get_extended_astar_activations())
            if args.activation not in all_acts:
                raise ValueError(f"Unknown activation: {args.activation}. Available: {list(all_acts.keys())}")
            activations = {args.activation: all_acts[args.activation]}
            print(f"\nRunning single activation: {args.activation}")
        else:
            raise ValueError("Must specify --activation, --all-activations, --controls-only, or --astar")

        all_results: Dict[str, List[Dict[str, any]]] = {}
        for act_name, act_fn in activations.items():
            seed_results = []
            for seed in range(args.n_seeds):
                seed_results.append(run_benchmark_for_activation(
                    act_name, act_fn, seed, output_dir, args
                ))
            all_results[act_name] = seed_results

        # Generate summary
        summary = {
            'schema_version': '1',
            'created_at': timestamp,
            'command': ' '.join(sys.argv) if sys.argv else None,
            'code': {
                'repo_root': str(Path(__file__).resolve().parents[2]),
                'git_commit': None,
            },
            'environment': {
                'python': platform.python_version(),
                'torch': getattr(torch, '__version__', 'unknown'),
                'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            },
            'benchmark': 'grokking_modular_arithmetic',
            'mode': args.mode,
            'timestamp': timestamp,
            'n_seeds': int(args.n_seeds),
            'seeds': list(range(int(args.n_seeds))),
            'num_activations': len(all_results),
            'modulus': args.modulus,
            'results': {
                name: {
                    'final_train_acc': float(np.mean([r['final_train_acc'] for r in seed_results])),
                    'final_train_acc_std': float(np.std([r['final_train_acc'] for r in seed_results])),
                    'final_test_acc': float(np.mean([r['final_test_acc'] for r in seed_results])),
                    'final_test_acc_std': float(np.std([r['final_test_acc'] for r in seed_results])),
                    'grokking_epochs': [r['grokking_epoch'] for r in seed_results],
                    'mean_grokking_epoch': float(np.mean([r['grokking_epoch'] for r in seed_results if r['grokking_epoch'] is not None])) if any(r['grokking_epoch'] for r in seed_results) else None,
                    'standardization': (seed_results[0].get('standardization') if seed_results else None),
                }
                for name, seed_results in all_results.items()
            },
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"{'Activation':<35} {'Train Acc':>12} {'Test Acc':>12} {'Grok Epoch':>12}")
        print(f"{'-'*80}")
        for name in sorted(all_results.keys()):
            res = summary['results'][name]
            grok = res['mean_grokking_epoch']
            grok_str = f"{grok:.0f}" if grok else "N/A"
            print(
                f"{name:<35} {res['final_train_acc']:>12.4f} "
                f"{res['final_test_acc']:>12.4f} "
                f"{grok_str:>12}"
            )

        print(f"\nResults saved to: {output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Grokking Modular Arithmetic Benchmark (Thrust 4)')

    act_group = parser.add_mutually_exclusive_group(required=True)
    act_group.add_argument('--activation', type=str, help='Single activation to test')
    act_group.add_argument('--all-activations', action='store_true', help='Run the standard experimental set')
    act_group.add_argument('--controls-only', action='store_true', help='Run only control activations')
    act_group.add_argument('--astar', action='store_true', help='Run canonical a* variants (SoftCap/SwishCap/SparseCap with optimal a*)')
    act_group.add_argument('--extended-astar', action='store_true', dest='extended_astar', 
                          help='Run extended a* conditions (canonical variants a* fixed/learnable)')

    # Mode
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'modulus', 'range', 'operation'],
                       help='Benchmark mode (default: standard)')

    # Hyperparams
    parser.add_argument('--epochs', type=int, default=10000, help='Training epochs (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1.0, help='Weight decay (default: 1.0, important for grokking)')
    parser.add_argument(
        '--a-weight-decay',
        type=float,
        default=None,
        help='Override weight decay for activation parameter(s) named "*.a" (default: None = use --weight-decay)',
    )
    parser.add_argument(
        '--weight-init',
        type=str,
        default='xavier',
        choices=['xavier', 'kaiming', 'orthogonal', 'auto'],
        help='Weight initialization for Linear layers (default: xavier). Use auto for activation-specific init.',
    )
    parser.add_argument('--modulus', type=int, default=97, help='Modulus for arithmetic (default: 97)')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimension (default: 128)')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval (default: 100)')

    # Paths
    parser.add_argument('--output-dir', type=str, default='mechanistic_interpretability/grokking',
                       help='Output directory')

    # Repro / statistics
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of random seeds (default: 3)')
    parser.add_argument('--force', action='store_true', help='Recompute even if metrics exist')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume by skipping existing metrics')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)

    # Quick mode for testing
    parser.add_argument('--quick', action='store_true', help='Quick test run (500 epochs, 1 seed)')

    # Optional artifacts for mechanistic probes
    parser.add_argument('--save-embeddings', action='store_true', help='Save embedding matrices (embed_a/embed_b) as .npy per seed')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save model state_dict checkpoint per seed (.pt)')
    parser.add_argument('--save-pair-reprs', action='store_true', help='Save composed pair representations (post-fc2 hidden state over full (a,b) grid) per seed')
    parser.add_argument('--pair-repr-batch', type=int, default=2048, help='Batch size used when saving pair representations (default: 2048)')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 500
        args.n_seeds = 1
        args.eval_interval = 50

    sns.set(style='whitegrid')
    GrokkingBenchmark.run(args)


if __name__ == '__main__':
    main()
