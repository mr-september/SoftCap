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

"""Train Sparse Autoencoder (SAE).

Track A (SAE): tests the "activation as sparsity prior" hypothesis.

This file supports two data sources:
- mock: synthetic activations (fast sanity check)
- grokking_pair_hidden: reuse saved hidden activations from grokking runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
import sys
import platform
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


# Add project root to path
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
from experiments.representation.sae import SparseAutoencoder, TopKActivation

def generate_mock_activations(n_samples=10000, d_model=256, sparsity=0.8, seed=42):
    """Generate synthetic 'neuron activations' with some structure."""
    torch.manual_seed(seed)
    # Create sparse ground truth
    ground_truth = torch.randn(n_samples, d_model * 4) * (torch.rand(n_samples, d_model * 4) > sparsity).float()
    # Project to d_model
    projection = torch.randn(d_model * 4, d_model) / (d_model ** 0.5)
    activations = ground_truth @ projection
    return activations


def load_grokking_pair_hidden_dataset(
    grokking_dir: Path,
    max_samples: Optional[int] = None,
    seed_filter: Optional[List[int]] = None,
) -> torch.Tensor:
    """Load `pair_hidden_seed*.npy` activations from a grokking run directory."""
    if not grokking_dir.exists():
        raise FileNotFoundError(f"grokking_dir not found: {grokking_dir}")

    files = sorted(grokking_dir.glob('pair_hidden_seed*.npy'))
    if seed_filter is not None:
        wanted = set(int(s) for s in seed_filter)
        filtered = []
        for f in files:
            # pair_hidden_seed{N}.npy
            stem = f.stem
            try:
                seed = int(stem.split('seed', 1)[1])
            except Exception:
                continue
            if seed in wanted:
                filtered.append(f)
        files = filtered

    if not files:
        raise FileNotFoundError(
            f"No pair_hidden_seed*.npy found under {grokking_dir}. "
            "Expected outputs from grokking runs with --save-pair-reprs."
        )

    arrays = [np.load(str(f)) for f in files]
    data = np.concatenate(arrays, axis=0)
    if max_samples is not None and data.shape[0] > max_samples:
        data = data[:max_samples]
    return torch.tensor(data, dtype=torch.float32)


def build_activation(
    activation_name: str,
    k: int = 32,
    a_regime: str = 'not_applicable',
    a_value: Optional[float] = None,
    freeze_a: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiate an activation module and return (module, standardization_dict).
    
    Supports canonical names ('SoftCap', 'SwishCap', 'SparseCap') and legacy
    class names ('ParametricTanhSoftCap', etc.) for backward compatibility.
    """
    # --- Config Label Parser ---
    # Parse experiment config labels: {Family}_{a_setting}_{mode}
    # Example: SmoothNotchV2_1_fixed, SmoothNotchV2_astar_learnable
    CONFIG_LABEL_MAP = {
        'SwishCap': ('SwishCap', 2.434),
        'SparseCap': ('SparseCap', 2.14),
        'SoftCap': ('SoftCap', 2.89),
        # Backward-compatible labels used by older result folders.
        'SmoothNotchV2': ('SwishCap', 2.434),
        'QuinticNotch': ('SparseCap', 2.14),
        'TanhSoftCap': ('SoftCap', 2.89),
    }
    
    parsed_activation = activation_name
    parsed_a_value = a_value
    parsed_freeze_a = freeze_a
    parsed_a_regime = a_regime
    
    for family, (cls_name, astar) in CONFIG_LABEL_MAP.items():
        if activation_name.startswith(family + '_'):
            suffix = activation_name[len(family) + 1:]  # e.g., "1_fixed", "astar_learnable"
            parts = suffix.split('_')
            if len(parts) >= 2:
                a_setting, mode = parts[0], parts[1]
                # Parse a_value
                if a_setting == 'astar':
                    parsed_a_value = astar
                    parsed_a_regime = 'fixed_astar' if mode == 'fixed' else 'learnable_init_astar'
                elif a_setting.replace('.', '', 1).isdigit():
                    parsed_a_value = float(a_setting)
                    parsed_a_regime = 'fixed_a1' if mode == 'fixed' else 'learnable_init_1'
                else:
                    parsed_a_value = 1.0  # default
                    parsed_a_regime = 'unknown'
                # Parse freeze_a
                parsed_freeze_a = (mode == 'fixed')
                parsed_activation = cls_name
                break
    
    # Update effective values
    activation_name = parsed_activation
    a_value = parsed_a_value
    freeze_a = parsed_freeze_a
    a_regime = parsed_a_regime
    
    standardization: Dict[str, Any] = {
        'a_regime': a_regime,
        'a_value': a_value,
        'a_init': 'not_applicable',
        'weight_init': 'unknown',
    }

    if activation_name == 'TopK':
        return TopKActivation(k=int(k)), standardization

    if activation_name == 'Identity':
        return nn.Identity(), standardization

    # Controls (non-parametric).
    if activation_name == 'ReLU':
        return nn.ReLU(), standardization
    if activation_name == 'GELU':
        return nn.GELU(), standardization
    if activation_name == 'SiLU':
        return nn.SiLU(), standardization
    if activation_name == 'Tanh':
        return nn.Tanh(), standardization

    # Canonical parametric family (new names + deprecated aliases for backward compat).
    if activation_name in ('SoftCap', 'ParametricTanhSoftCap'):
        a_init = float(a_value if a_value is not None else 1.0)
        act = SoftCap(a_init=a_init)
        standardization.update({'a_regime': a_regime, 'a_value': None, 'a_init': str(a_init)})
        if freeze_a and hasattr(act, 'a'):
            act.a.requires_grad_(False)
        return act, standardization
    if activation_name in ('SparseCap', 'ParametricQuinticNotchTanhSoftCap'):
        a_init = float(a_value if a_value is not None else 1.0)
        act = SparseCap(a_init=a_init)
        standardization.update({'a_regime': a_regime, 'a_value': None, 'a_init': str(a_init)})
        if freeze_a and hasattr(act, 'a'):
            act.a.requires_grad_(False)
        return act, standardization
    if activation_name in ('SwishCap', 'ParametricSmoothNotchTanhSoftCapV2'):
        a_init = float(a_value if a_value is not None else 1.0)
        act = SwishCap(a_init=a_init)
        standardization.update({'a_regime': a_regime, 'a_value': None, 'a_init': str(a_init)})
        if freeze_a and hasattr(act, 'a'):
            act.a.requires_grad_(False)
        return act, standardization

    raise ValueError(f"Unknown activation or unsupported a-regime wiring: {activation_name}")

def train_sae(
    activation_name: str,
    d_model: int = 256,
    d_hidden: int = 1024,
    l1_coeff: float = 3e-4,
    k: int = 32,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    a_weight_decay: Optional[float] = None,
    a_regime: str = 'not_applicable',
    a_value: Optional[float] = None,
    freeze_a: bool = False,
    seed: int = 0,
    output_dir: Optional[str] = None,
    save_artifacts: bool = False,
    command: Optional[str] = None,
    provenance: Optional[dict] = None,
    data_source: str = 'mock',
    grokking_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    input_center: bool = True,
    init_strategy: str = 'kaiming',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training SAE with {activation_name} (init={init_strategy}) on {device} (seed={seed})")

    # Minimal determinism controls (good enough for this toy/mock-data setup).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Data (determine d_model before building the SAE).
    if data_source == 'mock':
        data = generate_mock_activations(d_model=d_model, seed=seed)
    elif data_source == 'grokking_pair_hidden':
        if grokking_dir is None:
            raise ValueError('data_source=grokking_pair_hidden requires grokking_dir')
        data = load_grokking_pair_hidden_dataset(Path(grokking_dir), max_samples=max_samples)
        d_model = int(data.shape[-1])
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    if input_center:
        data = data - data.mean(dim=0, keepdim=True)

    # Setup Activation
    act_fn, act_standardization = build_activation(
        activation_name=activation_name,
        k=k,
        a_regime=a_regime,
        a_value=a_value,
        freeze_a=freeze_a,
    )
    use_l1 = activation_name != 'TopK'
    if activation_name == 'TopK':
        print(f"Using TopK-{k} activation. L1 regularization disabled.")

    model = SparseAutoencoder(d_model, d_hidden, act_fn, init_strategy=init_strategy).to(device)

    # Optimizer: allow a-specific weight decay to mirror grokking provenance issues.
    if a_weight_decay is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        a_params = []
        base_params = []
        for n, p in model.named_parameters():
            if n.endswith('.a'):
                a_params.append(p)
            else:
                base_params.append(p)
        optimizer = optim.AdamW(
            [
                {'params': base_params, 'weight_decay': float(weight_decay)},
                {'params': a_params, 'weight_decay': float(a_weight_decay)},
            ],
            lr=lr,
        )
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    
    history = {'loss': [], 'recon': [], 'l1': [], 'l0': [], 'zero_frac': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_l1 = 0
        epoch_l0 = 0
        epoch_zero_frac = 0
        
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            x_recon, f = model(x)
            
            # SAE Loss = MSE + L1 * coeff
            recon_loss = F.mse_loss(x_recon, x)
            
            l1_loss = f.abs().sum(dim=1).mean() if use_l1 else torch.tensor(0.0, device=device)
            loss = recon_loss + (l1_coeff * l1_loss if use_l1 else 0.0)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_l1 += l1_loss.item()
            
            # Track L0 (sparsity)
            l0 = (f.abs() > 1e-5).float().sum(dim=1).mean().item()
            epoch_l0 += l0

            zero_frac = (f == 0).float().mean().item()
            epoch_zero_frac += zero_frac
            
        avg_loss = epoch_loss / len(loader)
        avg_recon = epoch_recon / len(loader)
        avg_l0 = epoch_l0 / len(loader)
        avg_l1 = (epoch_l1 / len(loader)) if use_l1 else 0.0
        avg_zero_frac = epoch_zero_frac / len(loader)
        
        history['loss'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['l1'].append(avg_l1)
        history['l0'].append(avg_l0)
        history['zero_frac'].append(avg_zero_frac)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, "
                f"L0={avg_l0:.1f}, ZeroFrac={avg_zero_frac:.3f}"
            )
            
    if save_artifacts and output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = {
            'schema_version': '1',
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'command': command,
            'code': {
                'repo_root': str(project_root),
                'git_commit': None,
            },
            'environment': {
                'python': platform.python_version(),
                'torch': getattr(torch, '__version__', 'unknown'),
                'device': str(device),
            },
            'standardization': act_standardization,
            'activation_name': activation_name,
            'seeds': [seed],
            'n_seeds': 1,
            'hyperparameters': {
                'data_source': data_source,
                'grokking_dir': str(grokking_dir) if grokking_dir else None,
                'max_samples': max_samples,
                'input_center': bool(input_center),
                'd_model': int(d_model),
                'd_hidden': d_hidden,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'weight_decay': float(weight_decay),
                'a_weight_decay': float(a_weight_decay) if a_weight_decay is not None else None,
                'l1_coeff': float(l1_coeff),
                'k': int(k),
                'uses_l1': bool(use_l1),
                'freeze_a': bool(freeze_a),
            },
            'results': {
                'final': {
                    'loss': float(history['loss'][-1]) if history['loss'] else None,
                    'recon': float(history['recon'][-1]) if history['recon'] else None,
                    'l1': float(history['l1'][-1]) if history['l1'] else None,
                    'l0': float(history['l0'][-1]) if history['l0'] else None,
                    'zero_frac': float(history['zero_frac'][-1]) if history.get('zero_frac') else None,
                },
            },
            'provenance': provenance or {
                'is_pre_standardization': True,
                'pre_standardization_reason': 'SAE track is still being standardized; weight_init/a_regime not yet wired through.',
            },
        }

        with open(out / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(out / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    return history

def main():
    parser = argparse.ArgumentParser()
    # Get list of available activations for help text
    available_acts = list(get_default_activations().keys()) + ['TopK', 'Identity']
    
    parser.add_argument('--activation', type=str, default='ReLU', 
                        help=f'Activation function. Options: {available_acts}')
    parser.add_argument('--data-source', type=str, default='mock',
                        choices=['mock', 'grokking_pair_hidden'],
                        help='Input dataset source (mock synthetic or grokking hidden activations).')
    parser.add_argument('--grokking-dir', type=str, default=None,
                        help='Directory containing pair_hidden_seed*.npy (required for grokking_pair_hidden).')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Optional cap on number of input examples (useful for quick smoke tests).')
    parser.add_argument('--no-input-center', action='store_true',
                        help='Disable centering inputs by dataset mean.')
    parser.add_argument('--init-strategy', type=str, default='kaiming',
                        choices=['kaiming', 'orthogonal', 'xavier'],
                        help='Initialization strategy for SAE encoder weights.')
    parser.add_argument('--a-regime', type=str, default='not_applicable',
                        choices=['fixed_a1', 'fixed_astar', 'learnable_init_1', 'learnable_init_astar', 'not_applicable', 'unknown'],
                        help='Record/drive the a regime for parametric activations.')
    parser.add_argument('--a', type=float, default=None,
                        help='a value (fixed-a) or a_init (parametric). If omitted, defaults to 1.0 for explicit constructors.')
    parser.add_argument('--freeze-a', action='store_true',
                        help='If activation has learnable parameter a, freeze it (no gradients).')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='AdamW weight decay for all parameters (default 0.0).')
    parser.add_argument('--a-weight-decay', type=float, default=None,
                        help='Override weight decay for parameters named *.a (mirrors grokking a_wd confounder).')
    
    # Updated args
    parser.add_argument('--l1-coeff', type=float, default=3e-4, dest='l1_coeff',
                        help='L1 regularization coefficient (ignored for TopK)')
    parser.add_argument('--k', type=int, default=32,
                        help='K for TopK activation')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs (default: 50)')
    
    # New args
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--save-artifacts', action='store_true', help='Save training artifacts (summary.json, history.json)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='List of random seeds to run')

    args = parser.parse_args()
    
    for seed in args.seeds:
        # Construct specific output dir for this seed if parent dir provided
        current_output_dir = None
        if args.output_dir:
            path = Path(args.output_dir)
            if len(args.seeds) > 1:
                path = path / f"seed_{seed}"
            current_output_dir = str(path)
            
            # Simple skip logic
            if (path / 'summary.json').exists():
                print(f"Skipping seed {seed} - results exist at {path}")
                continue

        train_sae(
            args.activation,
            l1_coeff=args.l1_coeff,
            k=args.k,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            a_weight_decay=args.a_weight_decay,
            a_regime=args.a_regime,
            a_value=args.a,
            freeze_a=bool(args.freeze_a),
            data_source=args.data_source,
            grokking_dir=args.grokking_dir,
            max_samples=args.max_samples,
            input_center=not bool(args.no_input_center),
            init_strategy=args.init_strategy,
            seed=seed,
            output_dir=current_output_dir,
            save_artifacts=args.save_artifacts,
            command=" ".join(sys.argv)
        )

if __name__ == '__main__':
    main()
