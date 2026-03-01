"""
Muon Optimizer Experiment Runner

Canonical runner for the Muon × SoftCap research track.
Tests whether bounded activations (SoftCap) can substitute for QK-Clip by
naturally constraining pre-softmax attention scores.

Configuration Matrix (full sweep mode):
- Activations: 5 (controls: ReLU, GELU; SoftCap: SoftCap, SwishCap, SparseCap)
- Inits: 4 (kaiming, orthogonal, xavier, softcap_optimal)
- a-Policies: 4 (a=1, a=a*, a=2.5, learnable)
- LR: 0.02 (Muon canonical), WD: 0.01
- Seeds: 2

Verification mode: 5 configs × 2 seeds = 10 runs (see README.md for config table).

Usage:
    # Verification run (5 configs × 2 seeds)
    python scripts/experiments/run_muon.py --verification --epochs 30 --seeds 2
    
    # Quick sanity check
    python scripts/experiments/run_muon.py --quick --epochs 3 --seeds 1
    
    # Resume from checkpoint
    python scripts/experiments/run_muon.py --resume
    
    # SoftCap variants only
    python scripts/experiments/run_muon.py --softcap-only

References:
    - Bernstein et al. (2025): Muon Optimizer arXiv:2502.16982
    - MuonClip / Kimi K2: QK-Clip for attention stability
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from softcap.activations import (
    SoftCap,
    SwishCap,
    SparseCap,
)
from softcap.optimizers.muon import Muon, create_muon_optimizer_groups, MuonOptimizerScheduler
from softcap.checkpoint_manager import SmartCheckpointManager
from softcap.initialization import kaiming_softcap_normal_, apply_initialization

# Import the research ViT model
from scripts.experiments.cv.models.vit_muon import ViTMuonResearch


# ============================================================================
# CONFIGURATION
# ============================================================================

# Optimal a* values from outputs/theory/variance_map_results.json
# Derived by scripts/theory/plot_variance_map.py (Monte Carlo binary search
# for Var(f(x)) = 1 where x ~ N(0,1)).
A_STAR_VALUES = {
    'SoftCap': 2.890625,
    'SwishCap': 2.43359375,
    'SparseCap': 2.14,
}

# Sweep configuration
DEFAULT_CONFIG = {
    # Architecture
    'img_size': 32,
    'patch_size': 4,
    'num_classes': 100,
    'embed_dim': 192,
    'depth': 6,
    'num_heads': 3,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'drop_path': 0.1,
    
    # Training
    'epochs': 30,  # Preliminary: reduced epochs
    'batch_size': 128,
    'warmup_epochs': 5,
    'patience': 10,
    
    # Muon-specific
    'muon_momentum': 0.95,
    'ns_steps': 5,  # Newton-Schulz steps (optimal from literature)
    'momentum_warmup_steps': 300,
    
    # Data
    'data_root': './data',
}


def get_activation_configs() -> List[Dict[str, Any]]:
    """
    Get activation configurations to sweep.
    
    Returns list of dicts with:
    - name: str
    - factory: callable(a) -> nn.Module
    - is_softcap: bool
    - a_star: float (optimal a for variance preservation)
    """
    configs = [
        # Controls (unbounded baselines)
        {
            'name': 'ReLU',
            'factory': lambda a: nn.ReLU(),
            'is_softcap': False,
            'a_star': 1.0,  # Not applicable
        },
        {
            'name': 'GELU',
            'factory': lambda a: nn.GELU(),
            'is_softcap': False,
            'a_star': 1.0,  # Not applicable
        },
        # SoftCap variants (bounded)
        {
            'name': 'SoftCap',
            'factory': lambda a: SoftCap(a_init=a),
            'is_softcap': True,
            'a_star': 2.890625,
        },
        {
            'name': 'SwishCap',
            'factory': lambda a: SwishCap(a_init=a),
            'is_softcap': True,
            'a_star': 2.43359375,
        },
        {
            'name': 'SparseCap',
            'factory': lambda a: SparseCap(a_init=a),
            'is_softcap': True,
            'a_star': 2.14,
        },
    ]
    return configs


def get_init_configs() -> List[Dict[str, Any]]:
    """Get initialization configurations to sweep."""
    return [
        {'name': 'kaiming', 'method': 'kaiming'},
        {'name': 'orthogonal', 'method': 'orthogonal'},
        {'name': 'xavier', 'method': 'xavier'},
        {'name': 'softcap_optimal', 'method': 'softcap_optimal'},
    ]


def get_a_policies() -> List[Dict[str, Any]]:
    """Get 'a' parameter policies to sweep."""
    return [
        {'name': 'a=1.0', 'a_value': 1.0, 'learnable': False},
        {'name': 'a=a*', 'a_value': 'a_star', 'learnable': False},  # Use a_star from activation
        {'name': 'a=2.5', 'a_value': 2.5, 'learnable': False},
        {'name': 'a=learnable', 'a_value': 1.0, 'learnable': True},
    ]


def get_lr_configs() -> List[float]:
    """Get learning rates to sweep.
    
    Using LR=0.02 only (Muon paper canonical value).
    Muon optimizer uses higher LR than Adam/AdamW due to orthogonalization.
    """
    return [0.02]  # Canonical Muon LR from Bernstein et al. (2025)


def get_wd_configs() -> List[float]:
    """Get weight decay values to sweep.
    
    Using WD=0.01 only (standard AdamW value).
    WD is orthogonal to attention stability (core hypothesis).
    """
    return [0.01]  # Standard weight decay


def _fmt_float_tag(x: Optional[float]) -> str:
    """Format a float for stable, filesystem-friendly run tags."""
    if x is None:
        return "default"
    s = f"{x:.6g}"
    return s.replace('-', 'm').replace('.', 'p')


# ============================================================================
# DATA LOADING
# ============================================================================

def setup_data(config: Dict[str, Any], device: str) -> Tuple[DataLoader, DataLoader]:
    """Setup CIFAR-100 data loaders."""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=config['data_root'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root=config['data_root'],
        train=False,
        download=True,
        transform=val_transform
    )
    
    # DataLoaders with CUDA acceleration
    use_cuda = torch.cuda.is_available()
    # WSL/DrvFS can fail on multiprocessing socket listeners; force single-worker IO.
    nw = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=nw,
        pin_memory=False,
        persistent_workers=nw > 0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
        persistent_workers=nw > 0,
    )
    
    return train_loader, val_loader


# ============================================================================
# MODEL CREATION AND INITIALIZATION
# ============================================================================


def create_model(
    config: Dict[str, Any],
    activation_config: Dict[str, Any],
    a_policy: Dict[str, Any],
    init_config: Dict[str, Any],
    device: str,
    qk_score_clamp: Optional[float] = None,
) -> ViTMuonResearch:
    """Create ViT model with specified configuration."""
    
    # Determine 'a' value
    if a_policy['a_value'] == 'a_star':
        a_value = activation_config['a_star']
    else:
        a_value = a_policy['a_value']
    
    # Create activation for Q/K projections
    qk_activation = activation_config['factory'](a_value)
    
    # Freeze or unfreeze the 'a' parameter based on policy
    if hasattr(qk_activation, 'a'):
        qk_activation.a.requires_grad_(a_policy['learnable'])
    
    # Create model
    model = ViTMuonResearch(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        mlp_activation=nn.GELU(),  # MLP uses GELU (standard)
        qk_activation=qk_activation,  # Q/K uses our test activation
        dropout=config['dropout'],
        drop_path=config['drop_path'],
        track_attention_stats=True,
        qk_clip_threshold=qk_score_clamp,
    )
    
    # Apply initialization
    apply_initialization(model, init_config['method'], activation_config['name'])
    
    model = model.to(device)
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    muon_opt: Muon,
    adamw_opt: optim.AdamW,
    scheduler: MuonOptimizerScheduler,
    device: str,
    config: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    max_qk_scores_raw = []
    max_qk_scores_post_clip = []
    clip_fractions = []
    grad_norms = []
    layer_wise_qk_raw = {}  # Store lists of raw max QK per layer
    layer_wise_qk_post_clip = {}  # Store lists of post-clip max QK per layer
    layer_wise_clip_fraction = {}  # Store clip fraction per layer
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'max_qk_score': float('inf'),
                'max_qk_score_post_clip': float('inf'),
                'grad_norm': float('inf'),
                'diverged': True,
            }
        
        # Backward pass
        loss.backward()
        
        # Record gradient norm BEFORE optimizer step
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Check for exploding gradients
        if total_norm > 1e6:
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'max_qk_score': float('inf'),
                'max_qk_score_post_clip': float('inf'),
                'grad_norm': total_norm,
                'diverged': True,
            }
        
        # Optimizer steps
        muon_opt.step()
        adamw_opt.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Track max QK score (key stability metric)
        max_qk = model.get_max_qk_score()
        if not np.isnan(max_qk):
            max_qk_scores_raw.append(max_qk)
            
        # Track layer-wise stats
        all_stats = model.get_all_attention_stats()
        for layer_name, stats in all_stats.items():
            if layer_name not in layer_wise_qk_raw:
                layer_wise_qk_raw[layer_name] = []
            if layer_name not in layer_wise_qk_post_clip:
                layer_wise_qk_post_clip[layer_name] = []
            if layer_name not in layer_wise_clip_fraction:
                layer_wise_clip_fraction[layer_name] = []
            if stats.get('max_qk_score') is not None:
                layer_wise_qk_raw[layer_name].append(stats['max_qk_score'])
            if stats.get('max_qk_score_post_clip') is not None:
                layer_wise_qk_post_clip[layer_name].append(stats['max_qk_score_post_clip'])
            if stats.get('clip_fraction') is not None:
                layer_wise_clip_fraction[layer_name].append(stats['clip_fraction'])

        max_qk_post_clip = model.get_max_qk_score_post_clip()
        if not np.isnan(max_qk_post_clip):
            max_qk_scores_post_clip.append(max_qk_post_clip)

        valid_layer_clip_fractions = [
            stats.get('clip_fraction') for stats in all_stats.values()
            if stats.get('clip_fraction') is not None
        ]
        if valid_layer_clip_fractions:
            clip_fractions.append(float(np.mean(valid_layer_clip_fractions)))
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    avg_max_qk_raw = np.mean(max_qk_scores_raw) if max_qk_scores_raw else float('nan')
    avg_max_qk_post_clip = np.mean(max_qk_scores_post_clip) if max_qk_scores_post_clip else float('nan')
    avg_clip_fraction = np.mean(clip_fractions) if clip_fractions else float('nan')
    avg_grad_norm = np.mean(grad_norms) if grad_norms else float('nan')
    
    # Compute layer-wise max QK (average over batches)
    layer_wise_avg_raw = {}
    if layer_wise_qk_raw:
        for layer_name, scores in layer_wise_qk_raw.items():
            layer_wise_avg_raw[layer_name] = np.mean(scores) if scores else float('nan')

    layer_wise_avg_post_clip = {}
    if layer_wise_qk_post_clip:
        for layer_name, scores in layer_wise_qk_post_clip.items():
            layer_wise_avg_post_clip[layer_name] = np.mean(scores) if scores else float('nan')

    layer_wise_avg_clip_fraction = {}
    if layer_wise_clip_fraction:
        for layer_name, scores in layer_wise_clip_fraction.items():
            layer_wise_avg_clip_fraction[layer_name] = np.mean(scores) if scores else float('nan')
            
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'max_qk_score': avg_max_qk_raw,
        'max_qk_score_raw': avg_max_qk_raw,
        'max_qk_score_post_clip': avg_max_qk_post_clip,
        'clip_fraction': avg_clip_fraction,
        'grad_norm': avg_grad_norm,
        'layer_wise_max_qk': layer_wise_avg_raw,
        'layer_wise_max_qk_raw': layer_wise_avg_raw,
        'layer_wise_max_qk_post_clip': layer_wise_avg_post_clip,
        'layer_wise_clip_fraction': layer_wise_avg_clip_fraction,
        'diverged': False,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            if not torch.isfinite(loss):
                return {
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'diverged': True,
                }
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100.0 * correct / total,
        'diverged': False,
    }


# ============================================================================
# SWEEP ORCHESTRATION
# ============================================================================


def generate_config_id(
    activation: str,
    init: str,
    a_policy: str,
    lr: float,
    wd: float,
    seed: int,
    epochs: int,
    qk_score_clamp: Optional[float] = None,
) -> str:
    """Generate unique config ID for checkpointing."""
    clamp_tag = "none" if qk_score_clamp is None else f"{qk_score_clamp:.6g}"
    key = f"{activation}_{init}_{a_policy}_lr{lr}_wd{wd}_seed{seed}_ep{epochs}_qkclamp{clamp_tag}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def run_single_experiment(
    config: Dict[str, Any],
    activation_config: Dict[str, Any],
    init_config: Dict[str, Any],
    a_policy: Dict[str, Any],
    lr: float,
    wd: float,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    stop_at_epoch: Optional[int] = None,
    # New args for checkpointing
    config_id: str = None,
    output_dir: Path = None,
    resume: bool = False,
    qk_score_clamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""

    def _ensure_history_schema_compatibility(hist: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill newly added history keys when resuming older checkpoints."""
        base_len = len(hist.get('train_loss', []))
        list_defaults = {
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'max_qk_score': [],
            'max_qk_score_raw': [],
            'max_qk_score_post_clip': [],
            'clip_fraction': [],
            'grad_norm': [],
            'layer_wise_max_qk': [],
            'layer_wise_max_qk_raw': [],
            'layer_wise_max_qk_post_clip': [],
            'layer_wise_clip_fraction': [],
        }

        for key in list_defaults:
            if key not in hist:
                fill_value = {} if key.startswith('layer_wise_') else float('nan')
                hist[key] = [fill_value for _ in range(base_len)]
            elif isinstance(hist[key], list) and len(hist[key]) < base_len:
                deficit = base_len - len(hist[key])
                fill_value = {} if key.startswith('layer_wise_') else float('nan')
                hist[key].extend([fill_value for _ in range(deficit)])

        hist.setdefault('diverged', False)
        hist.setdefault('divergence_epoch', None)
        hist.setdefault('stopped_at_epoch', None)
        return hist
    
    # Initialize Checkpoint Manager
    ckpt_manager = None
    if output_dir is not None and config_id is not None:
        # Save to per-config subdirectory
        ckpt_dir = output_dir / 'checkpoints' / config_id
        ckpt_manager = SmartCheckpointManager(
            checkpoint_dir=ckpt_dir,
            metric_name='val_acc',
            metric_mode='max',
            strategy='smart'
        )
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model = create_model(
        config,
        activation_config,
        a_policy,
        init_config,
        device,
        qk_score_clamp=qk_score_clamp,
    )
    
    # Create optimizers
    # Muon requires specific parameter grouping (handled in create_muon_optimizer_groups)
    muon_opt, adamw_opt = create_muon_optimizer_groups(
        model,
        muon_lr=lr,
        adamw_lr=lr / 20,  # AdamW typically needs lower LR
        muon_momentum=config['muon_momentum'],
        weight_decay=wd,
        ns_steps=config['ns_steps'],
        warmup_steps=config['momentum_warmup_steps'],
    )
    
    # Create scheduler
    # IMPORTANT: The scheduler depends on the TOTAL planned epochs, not the stop_at_epoch.
    # This allows resuming later with the correct decay curve.
    total_steps = config['epochs'] * len(train_loader)
    scheduler = MuonOptimizerScheduler(
        muon_opt,
        adamw_opt,
        total_steps=total_steps,
        warmup_steps=config['warmup_epochs'] * len(train_loader),
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'max_qk_score': [],
        'max_qk_score_raw': [],
        'max_qk_score_post_clip': [],
        'clip_fraction': [],
        'grad_norm': [],
        'layer_wise_max_qk': [], # List of dicts (raw, backward-compatible)
        'layer_wise_max_qk_raw': [],
        'layer_wise_max_qk_post_clip': [],
        'layer_wise_clip_fraction': [],
        'diverged': False,
        'divergence_epoch': None,
        'stopped_at_epoch': None,
        # Metadata for intervention test reconstruction
        'activation_name': activation_config['name'],
        'init_name': init_config['name'],
        'a_value': a_policy.get('value'),  # May be None for 'learnable'
        'seed': seed,
        'qk_score_clamp': qk_score_clamp,
    }
    
    start_epoch = 0
    best_val_acc = 0.0
    
    # Resume from checkpoint if available
    if resume and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint()
        if checkpoint:
            print(f"    [INFO] Resuming from epoch {checkpoint['epoch']}")
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load optimizer states if available (Muon has internal state!)
            if 'muon_optimizer_state_dict' in checkpoint:
                 muon_opt.load_state_dict(checkpoint['muon_optimizer_state_dict'])
            if 'adamw_optimizer_state_dict' in checkpoint:
                 adamw_opt.load_state_dict(checkpoint['adamw_optimizer_state_dict'])
            # Restore scheduler and Muon global step
            if 'scheduler_current_step' in checkpoint:
                scheduler.current_step = checkpoint['scheduler_current_step']
            if 'muon_global_step' in checkpoint:
                muon_opt.global_step = checkpoint['muon_global_step']
            
            # Restore history
            if 'history' in checkpoint:
                history = _ensure_history_schema_compatibility(checkpoint['history'])
                start_epoch = len(history['train_loss'])
                if history['val_acc']:
                    best_val_acc = max(history['val_acc'])
            else:
                start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        # Check stop condition (Resumability logic)
        if stop_at_epoch is not None and epoch >= stop_at_epoch:
            print(f"    [INFO] Pausing at epoch {epoch} (requested stop_at_epoch={stop_at_epoch})")
            history['stopped_at_epoch'] = epoch
            
            # Save checkpoint before stopping
            if ckpt_manager is not None:
                 ckpt_manager.save_checkpoint(
                     model=model,
                     optimizer=None,
                     scheduler=None,
                     epoch=epoch,
                     metrics={'val_acc': history['val_acc'][-1] if history['val_acc'] else 0.0},
                     additional_state={
                         'muon_optimizer_state_dict': muon_opt.state_dict(),
                         'adamw_optimizer_state_dict': adamw_opt.state_dict(),
                         'scheduler_current_step': scheduler.current_step,
                         'muon_global_step': muon_opt.global_step,
                         'history': history,
                     }
                 )
            break
            
        # Train
        train_metrics = train_epoch(
            model, train_loader, muon_opt, adamw_opt, scheduler, device, config, epoch
        )
        
        if train_metrics['diverged']:
            history['diverged'] = True
            history['divergence_epoch'] = epoch
            break
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['max_qk_score'].append(train_metrics['max_qk_score'])
        history['max_qk_score_raw'].append(train_metrics['max_qk_score_raw'])
        history['max_qk_score_post_clip'].append(train_metrics['max_qk_score_post_clip'])
        history['clip_fraction'].append(train_metrics.get('clip_fraction', float('nan')))
        history['grad_norm'].append(train_metrics['grad_norm'])
        history['layer_wise_max_qk'].append(train_metrics.get('layer_wise_max_qk', {}))
        history['layer_wise_max_qk_raw'].append(train_metrics.get('layer_wise_max_qk_raw', {}))
        history['layer_wise_max_qk_post_clip'].append(train_metrics.get('layer_wise_max_qk_post_clip', {}))
        history['layer_wise_clip_fraction'].append(train_metrics.get('layer_wise_clip_fraction', {}))
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        if val_metrics['diverged']:
            history['diverged'] = True
            history['divergence_epoch'] = epoch
            break
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Track best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
        # Logging
        print(f"    Ep {epoch+1}/{config['epochs']} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.2f}% (Val: {val_metrics['accuracy']:.2f}%) | "
              f"QK(raw/post): {train_metrics['max_qk_score_raw']:.2f}/{train_metrics['max_qk_score_post_clip']:.2f} | "
              f"ClipFrac: {train_metrics.get('clip_fraction', float('nan')):.4f} | "
              f"Grad: {train_metrics['grad_norm']:.2f}")
              
        # Save checkpoint (manager's 'smart' strategy handles retention)
        if ckpt_manager is not None:
             ckpt_manager.save_checkpoint(
                 model=model,
                 optimizer=None,
                 scheduler=None,
                 epoch=epoch,
                 metrics={'val_acc': val_metrics['accuracy']},
                 additional_state={
                     'muon_optimizer_state_dict': muon_opt.state_dict(),
                     'adamw_optimizer_state_dict': adamw_opt.state_dict(),
                     'scheduler_current_step': scheduler.current_step,
                     'muon_global_step': muon_opt.global_step,
                     'history': history,
                 }
             )

    # Final metrics
    history['best_val_acc'] = best_val_acc
    history['final_train_acc'] = history['train_acc'][-1] if history['train_acc'] else 0.0
    history['final_val_acc'] = history['val_acc'][-1] if history['val_acc'] else 0.0
    history['max_recorded_qk'] = max(history['max_qk_score']) if history['max_qk_score'] else float('nan')
    history['max_recorded_qk_raw'] = max(history['max_qk_score_raw']) if history['max_qk_score_raw'] else float('nan')
    history['max_recorded_qk_post_clip'] = max(history['max_qk_score_post_clip']) if history['max_qk_score_post_clip'] else float('nan')
    history['mean_clip_fraction'] = np.mean(history['clip_fraction']) if history['clip_fraction'] else float('nan')
    history['max_clip_fraction'] = max(history['clip_fraction']) if history['clip_fraction'] else float('nan')
    history['max_recorded_grad'] = max(history['grad_norm']) if history['grad_norm'] else float('nan')
    
    return history


def run_intervention_test(
    output_dir: Path,
    device: str,
    fraction: float = 0.01,
    scale: float = 5.0,
    qk_score_clamp: Optional[float] = None,
):
    """
    Run intervention test on trained models.
    
    Loads checkpoints from recent runs and evaluates with injected Q/K corruption.
    Tests hypothesis: If bounded activations prevent extremes architecturally,
    injecting extremes should degrade their performance.
    """
    output_dir = Path(output_dir)
    intervention_dir = output_dir / 'intervention_test'
    intervention_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("INTERVENTION TEST: Causal Ablation via Artificial Extremes")
    print("="*80)
    print(f"Config: {fraction*100:.1f}% of Q/K values scaled by {scale}×")
    print(f"Output: {intervention_dir}")
    print()
    
    intervention_config = {
        'targets': ['q', 'k'],
        'fraction': fraction,
        'scale': scale,
    }
    
    # Find all checkpoints (searches all subdirs for checkpoint_*.pt)
    checkpoints_dir = output_dir / 'checkpoints'
    if not checkpoints_dir.exists():
        print(f"ERROR: No checkpoints directory found at {checkpoints_dir}")
        print("Please run training first:")
        print("  python scripts/experiments/run_muon.py --verification --epochs 10 --lr 0.1 --seeds 3")
        return
    
    # PRIORITIZE: Only use FINAL checkpoints (epoch 9) from recent 10-epoch verification runs
    # This gives us one representative checkpoint per (activation, init, seed) combination
    checkpoint_files = []
    
    for config_dir in sorted(checkpoints_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        metadata_file = config_dir / 'checkpoint_metadata.json'
        if not metadata_file.exists():
            continue
        
        # Load metadata to check if it's from a recent run
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                last_updated = metadata.get('last_updated', '')
                
                # Only include checkpoints from 2026-02-17 onwards (most recent 10-epoch runs)
                if last_updated >= '2026-02-17':
                    # Look for epoch 9 checkpoint (final epoch of 10-epoch runs)
                    epoch_9_ckpts = list(config_dir.glob('checkpoint_epoch_0009_*.pt'))
                    if epoch_9_ckpts:
                        checkpoint_files.append(epoch_9_ckpts[0])  # Take first match
        except Exception as e:
            continue
    
    if not checkpoint_files:
        print(f"ERROR: No recent final-epoch checkpoints found in {checkpoints_dir}")
        print("Looking for epoch 0009 checkpoints from 2026-02-17+")
        print("Please run training first:")
        print("  python scripts/experiments/run_muon.py --verification --epochs 10 --lr 0.1 --seeds 3")
        return
    
    # Load results.json to map config_id to activation metadata (for backward compatibility)
    # This handles old checkpoints that don't have metadata embedded
    config_metadata_map = {}
    results_files = output_dir.glob('*_results.json')
    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                for r in results:
                    config_metadata_map[r['config_id']] = {
                        'activation': r['activation'],
                        'init': r['init'],
                        'a_policy': r['a_policy'],
                        'seed': r['seed'],
                        'lr': r.get('lr'),
                        'wd': r.get('wd'),
                    }
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")
    
    print(f"Found {len(checkpoint_files)} checkpoint(s)")
    
    # Setup data (will eval on validation set)
    config = DEFAULT_CONFIG.copy()
    _, val_loader = setup_data(config, device)
    
    results = []
    
    for idx, ckpt_path in enumerate(sorted(checkpoint_files), 1):
        print(f"\n{'='*70}")
        print(f"Checkpoint {idx}/{len(checkpoint_files)}: {ckpt_path.name}")
        
        # Load checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
        
        # Extract activation name and config from history (new checkpoints)
        # or from config_metadata_map (old checkpoints)
        history = ckpt.get('history', {})
        
        # Get config_id from checkpoint directory name
        config_id = ckpt_path.parent.name
        
        # Try to get metadata from history first, then fall back to results.json mapping
        activation_name = history.get('activation_name')
        init_name = history.get('init_name')
        a_value = history.get('a_value')
        seed = history.get('seed')
        stored_qk_clamp = history.get('qk_score_clamp')
        
        if activation_name is None and config_id in config_metadata_map:
            # Fallback to results.json mapping for old checkpoints
            metadata = config_metadata_map[config_id]
            activation_name = metadata['activation']
            init_name = metadata['init']
            # Need to parse a_value from a_policy string (e.g., "a=a*" or "a=1.0")
            a_policy_str = metadata['a_policy']
            if 'a*' in a_policy_str:
                # Look up optimal a value for this activation
                act_conf_temp = next((a for a in get_activation_configs() if a['name'] == activation_name), None)
                if act_conf_temp and 'optimal_a' in act_conf_temp:
                    a_value = act_conf_temp['optimal_a']
                else:
                    a_value = 1.0  # Default fallback
            elif '=' in a_policy_str:
                try:
                    a_value = float(a_policy_str.split('=')[1])
                except:
                    a_value = 1.0
            else:
                a_value = 1.0  # Default
            
            seed = metadata['seed']
            print(f"  Note: Using results.json metadata for old checkpoint")
        
        if activation_name is None or activation_name == 'Unknown':
            print(f"  ✗ Cannot determine activation for config_id: {config_id}")
            continue
        
        print(f"  Activation: {activation_name}")
        print(f"  Init: {init_name}, a={a_value}, seed={seed}")
        
        # Reconstruct model (clean, no intervention)
        activation_configs = get_activation_configs()
        act_conf = next((a for a in activation_configs if a['name'] == activation_name), None)
        if act_conf is None:
            print(f"  ✗ Unknown activation: {activation_name}")
            continue
        
        # Create clean model
        a_val = a_value if a_value is not None else 1.0
        qk_activation = act_conf['factory'](a_val) if act_conf.get('is_softcap') else act_conf['factory'](1.0)
        
        model_clean = ViTMuonResearch(
            **{k: config[k] for k in ['img_size', 'patch_size', 'num_classes', 'embed_dim', 
                                       'depth', 'num_heads', 'mlp_ratio', 'dropout', 'drop_path']},
            qk_activation=qk_activation,
            track_attention_stats=True,
            qk_clip_threshold=qk_score_clamp,
            intervention_config=None,  # Clean
        ).to(device)
        
        # Create intervention model (same architecture, different forward behavior)
        qk_activation_int = act_conf['factory'](a_val) if act_conf.get('is_softcap') else act_conf['factory'](1.0)
        
        model_intervention = ViTMuonResearch(
            **{k: config[k] for k in ['img_size', 'patch_size', 'num_classes', 'embed_dim',
                                       'depth', 'num_heads', 'mlp_ratio', 'dropout', 'drop_path']},
            qk_activation=qk_activation_int,
            track_attention_stats=True,
            qk_clip_threshold=qk_score_clamp,
            intervention_config=intervention_config,  # Corruption enabled
        ).to(device)
        
        # Load weights into both
        try:
            model_clean.load_state_dict(ckpt['model_state_dict'])
            model_intervention.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            print(f"  ✗ Failed to load weights: {e}")
            continue
        
        # Evaluate both
        print("  Evaluating...")
        clean_metrics = validate(model_clean, val_loader, device)
        intervention_metrics = validate(model_intervention, val_loader, device)
        
        acc_clean = clean_metrics['accuracy']
        acc_intervention = intervention_metrics['accuracy']
        degradation = acc_clean - acc_intervention
        degradation_pct = (degradation / acc_clean * 100) if acc_clean > 0 else 0
        
        print(f"  Clean Accuracy:        {acc_clean:.2f}%")
        print(f"  Intervention Accuracy: {acc_intervention:.2f}%")
        print(f"  Degradation:           {degradation:.2f}% ({degradation_pct:.1f}% relative)")
        
        # Store results
        results.append({
            'checkpoint': str(ckpt_path),
            'activation': activation_name,
            'init': init_name,
            'a_value': a_value,
            'seed': seed,
            'clean_accuracy': float(acc_clean),
            'intervention_accuracy': float(acc_intervention),
            'degradation_abs': float(degradation),
            'degradation_pct': float(degradation_pct),
            'intervention_config': intervention_config,
        })
    
    # Save results
    results_path = intervention_dir / 'intervention_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("INTERVENTION TEST SUMMARY")
    print("="*80)
    print(f"{'Activation':<40} {'Clean':>10} {'Corrupt':>10} {'Degrad':>10}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['degradation_pct'], reverse=True):
        print(f"{r['activation']:<40} {r['clean_accuracy']:>10.2f}% {r['intervention_accuracy']:>10.2f}% "
              f"{r['degradation_pct']:>10.1f}%")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    bounded_results = [
        r for r in results
        if any(x in r['activation'] for x in ['TanhSoftCap', 'SmoothNotch', 'QuinticNotch'])
    ]
    unbounded_results = [r for r in results if r['activation'] in ['ReLU', 'GELU']]
    
    if bounded_results and unbounded_results:
        avg_degrad_bounded = np.mean([r['degradation_pct'] for r in bounded_results])
        avg_degrad_unbounded = np.mean([r['degradation_pct'] for r in unbounded_results])
        
        print(f"Bounded activations avg degradation:   {avg_degrad_bounded:.1f}%")
        print(f"Unbounded activations avg degradation: {avg_degrad_unbounded:.1f}%")
        
        if avg_degrad_bounded > avg_degrad_unbounded * 1.5:
            print("\n✓ SUPPORTS HYPOTHESIS: Bounded activations degrade more when extremes are injected")
            print("  → They rely on preventing extremes architecturally")
        elif avg_degrad_unbounded > avg_degrad_bounded * 1.5:
            print("\n✗ CONTRADICTS HYPOTHESIS: Unbounded activations are more sensitive to injected extremes")
        else:
            print("\n~ INCONCLUSIVE: Similar degradation across activation types")
    
    print(f"\nResults saved to: {results_path}")


def run_preliminary_sweep(
    output_dir: Path,
    device: str,
    epochs: int = 30,
    num_seeds: int = 2,
    softcap_only: bool = False,
    quick: bool = False,
    resume: bool = True,
    verification: bool = False,
    stop_at_epoch: Optional[int] = None,
    lr_override: Optional[float] = None,
    wd_override: Optional[float] = None,
    qk_score_clamp: Optional[float] = None,
    embed_dim_override: Optional[int] = None,
    depth_override: Optional[int] = None,
    num_heads_override: Optional[int] = None,
):
    """Run the full preliminary sweep."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_label = "verification" if verification else ("quick" if quick else "sweep")
    run_tag = (
        f"{run_label}_e{epochs}"
        f"_lr{_fmt_float_tag(lr_override)}"
        f"_wd{_fmt_float_tag(wd_override)}"
        f"_qkclamp{_fmt_float_tag(qk_score_clamp)}"
        f"_emb{embed_dim_override if embed_dim_override is not None else DEFAULT_CONFIG['embed_dim']}"
        f"_d{depth_override if depth_override is not None else DEFAULT_CONFIG['depth']}"
        f"_h{num_heads_override if num_heads_override is not None else DEFAULT_CONFIG['num_heads']}"
    )
    
    # Load completed configs if resuming
    completed_file = output_dir / f"{run_tag}_completed_configs.json"
    completed_configs = set()
    if resume and completed_file.exists():
        with open(completed_file, 'r') as f:
            completed_configs = set(json.load(f))
        print(f"Resuming: {len(completed_configs)} configs already completed")
    
    # Setup data
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = epochs
    if embed_dim_override is not None:
        config['embed_dim'] = int(embed_dim_override)
    if depth_override is not None:
        config['depth'] = int(depth_override)
    if num_heads_override is not None:
        config['num_heads'] = int(num_heads_override)
    
    print(f"Setting up data loaders...")
    train_loader, val_loader = setup_data(config, device)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Get sweep configurations
    activation_configs = get_activation_configs()
    
    init_configs = get_init_configs()
    a_policies = get_a_policies()
    lr_configs = get_lr_configs()
    wd_configs = get_wd_configs()

    if lr_override is not None:
        lr_configs = [lr_override]
    if wd_override is not None:
        wd_configs = [wd_override]

    seeds = list(range(num_seeds))
    
    # Filter configs based on mode
    if verification:
        print(f"\n{'!'*80}")
        print("VERIFICATION MODE: Running targeted Apples-to-Apples Comparison")
        print(f"{'!'*80}\n")
        
        # Define the exact Verification Set
        # 1. TanhSoftCap + softcap_optimal + a=a*
        # 2. SmoothNotchV2 + softcap_optimal + a=a*
        # 3. QuinticNotch + softcap_optimal + a=a*
        # 4. ReLU + kaiming
        # 5. GELU + kaiming
        
        target_activations = {
            'SoftCap': {'init': 'softcap_optimal', 'policy': 'a=a*'},
            'SwishCap': {'init': 'softcap_optimal', 'policy': 'a=a*'},
            'SparseCap': {'init': 'softcap_optimal', 'policy': 'a=a*'},
            'ReLU': {'init': 'kaiming', 'policy': 'a=1.0'},
            'GELU': {'init': 'kaiming', 'policy': 'a=1.0'},
        }
        
        # Override lists to generate only these combinations
        # We construct a custom list of (activation_config, init_name, policy_name) tuples to iterate
        
        # Helper to find config objects
        act_lookup = {c['name']: c for c in activation_configs}
        
        # We will use a custom iterator logic instead of the nested loops below?
        # To minimize code changes, we'll filter the lists dynamically or create a special list.
        # Actually, let's just create a list of "Tasks" and iterate that.
        
        verification_tasks = []
        for act_name, specs in target_activations.items():
            if act_name not in act_lookup:
                print(f"Warning: {act_name} not found in activation configs")
                continue
            verification_tasks.append({
                'activation': act_lookup[act_name],
                'init': specs['init'],
                'policy': specs['policy']
            })
            
    elif softcap_only:
        activation_configs = [c for c in activation_configs if c.get('is_softcap', False)]
    elif quick:
        # Quick sanity check
        activation_configs = activation_configs[:2]
        init_configs = init_configs[:2]
        a_policies = a_policies[:2]
        lr_configs = lr_configs[:1]
        wd_configs = wd_configs[:1]
        seeds = seeds[:1]
    
    # Logic for iteration
    # If verification, we have a specific list of tasks.
    # Otherwise, we have the Cartesian product.
    
    # Common function to run a specific combination
    def execute_run(act_conf, init_name, policy_name, lr, wd, seed, run_idx, total_runs):
        init_conf = next(i for i in init_configs if i['name'] == init_name)
        policy_conf = next(p for p in a_policies if p['name'] == policy_name)
        
        # Generate config ID (NOW INCLUDES EPOCHS for clean break)
        config_id = generate_config_id(
            act_conf['name'],
            init_conf['name'],
            policy_conf['name'],
            lr, wd, seed, epochs,
            qk_score_clamp=qk_score_clamp,
        )
        
        if config_id in completed_configs:
            return None
            
        print(f"[{run_idx}/{total_runs}] "
              f"{act_conf['name']} | {init_conf['name']} | "
              f"{policy_conf['name']} | lr={lr} | wd={wd} | seed={seed}")
              
        try:
            hist = run_single_experiment(
                config, act_conf, init_conf, policy_conf,
                lr, wd, seed, train_loader, val_loader, device,
                stop_at_epoch=stop_at_epoch,
                config_id=config_id,
                output_dir=output_dir,
                resume=resume,
                qk_score_clamp=qk_score_clamp,
            )
            
            # Store result
            history_epochs = len(hist.get('val_acc', [])) or len(hist.get('train_loss', []))
            epochs_run = hist.get('divergence_epoch', 0) if hist['diverged'] else (history_epochs or hist.get('stopped_at_epoch', epochs))
            res = {
                'config_id': config_id,
                'activation': act_conf['name'],
                'init': init_conf['name'],
                'a_policy': policy_conf['name'],
                'lr': lr,
                'wd': wd,
                'seed': seed,
                'epochs_planned': epochs, # Traceability
                'epochs_run': epochs_run,
                'is_softcap': act_conf['is_softcap'],
                'diverged': hist['diverged'],
                'divergence_epoch': hist.get('divergence_epoch'),
                'best_val_acc': hist['best_val_acc'],
                'final_val_acc': hist['final_val_acc'],
                'max_qk_score': hist['max_recorded_qk'],
                'max_qk_score_raw': hist['max_recorded_qk_raw'],
                'max_qk_score_post_clip': hist['max_recorded_qk_post_clip'],
                'mean_clip_fraction': hist.get('mean_clip_fraction', float('nan')),
                'max_clip_fraction': hist.get('max_clip_fraction', float('nan')),
                'max_grad_norm': hist['max_recorded_grad'],
                'train_loss_final': hist['train_loss'][-1] if hist['train_loss'] else None,
                # Full History for Analysis
                'history': {
                    'train_loss': hist['train_loss'],
                    'train_acc': hist['train_acc'],
                    'val_loss': hist['val_loss'],
                    'val_acc': hist['val_acc'],
                    'max_qk_score': hist['max_qk_score'],
                    'max_qk_score_raw': hist['max_qk_score_raw'],
                    'max_qk_score_post_clip': hist['max_qk_score_post_clip'],
                    'clip_fraction': hist.get('clip_fraction', []),
                    'grad_norm': hist['grad_norm'],
                }
            }
            
            status = "✓ OK" if not hist['diverged'] else "✗ DIVERGED"
            print(f"    {status} | Val Acc: {hist['best_val_acc']:.2f}% | "
                f"Max QK(raw/post): {hist['max_recorded_qk_raw']:.2f}/{hist['max_recorded_qk_post_clip']:.2f}")
                  
            return res
            
        except Exception as e:
            print(f"    ✗ ERROR: {str(e)}")
            traceback.print_exc()
            return None

    # Calculate total runs & plan execution
    tasks = []
    
    if verification:
        # Verification tasks: 5 variants * 2 seeds * 1 LR * 1 WD = 10 runs total
        target_lr = lr_configs[0]
        target_wd = wd_configs[0]
        
        print(f"Verification Mode: Fixed LR={target_lr}, WD={target_wd}")
        
        for task in verification_tasks:
            for seed in seeds:
                tasks.append((task['activation'], task['init'], task['policy'], target_lr, target_wd, seed))
    else:
        # Cartesian product
        init_lookup = {i['name']: i for i in init_configs}
        policy_lookup = {p['name']: p for p in a_policies}
        
        for act_conf in activation_configs:
            # Filter inits/policies if verification mode was handled differently (it's handled above now)
            # Standard sweep logic:
            current_inits = init_configs
            current_policies = a_policies
            
            for init_conf in current_inits:
                for policy_conf in current_policies:
                     # Skip non-applicable a-policies for controls
                    if not act_conf['is_softcap'] and policy_conf['name'] != 'a=1.0':
                        continue
                        
                    for lr in lr_configs:
                        for wd in wd_configs:
                            for seed in seeds:
                                tasks.append((act_conf, init_conf['name'], policy_conf['name'], lr, wd, seed))
                                
    total_runs = len(tasks)
    remaining_tasks = [
        t for t in tasks
        if generate_config_id(
            t[0]['name'], t[1], t[2], t[3], t[4], t[5], epochs,
            qk_score_clamp=qk_score_clamp,
        )
        not in completed_configs
    ]

    print(f"\n{'='*80}")
    run_label = "verification" if verification else ("quick" if quick else "sweep")
    print(f"Muon Optimizer Run ({run_label})")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    if stop_at_epoch:
        print(f"Stop at Epoch: {stop_at_epoch} (Resumable Mode)")
    print(f"Total configurations: {total_runs}")
    print(f"Already completed: {total_runs - len(remaining_tasks)}")
    print(f"Remaining: {len(remaining_tasks)}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Results storage
    all_results = []
    results_file = output_dir / f"{run_tag}_results.json"
    
    # Load existing results if resuming
    if resume and results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)

    # Run loop
    start_time = time.time()
    run_count = 0 
    
    for task in remaining_tasks:
        run_count += 1
        res = execute_run(*task, run_count, len(remaining_tasks))
        if res:
            # Replace any existing entry for this exact config_id (e.g., stop-at-epoch then resume)
            replaced = False
            for i, existing in enumerate(all_results):
                if existing.get('config_id') == res['config_id']:
                    all_results[i] = res
                    replaced = True
                    break
            if not replaced:
                all_results.append(res)
            completed_configs.add(res['config_id'])
            
            # Save incrementally
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            with open(completed_file, 'w') as f:
                json.dump(list(completed_configs), f)
    
    # Final save
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(completed_file, 'w') as f:
        json.dump(list(completed_configs), f)
    
    # Generate summary
    generate_summary(all_results, output_dir, run_tag=run_tag)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Sweep Complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")


def generate_summary(results: List[Dict[str, Any]], output_dir: Path, run_tag: str):
    """Generate summary statistics from sweep results."""
    
    summary = {
        'total_runs': len(results),
        'successful_runs': len([r for r in results if not r.get('diverged', True) and 'error' not in r]),
        'diverged_runs': len([r for r in results if r.get('diverged', False)]),
        'error_runs': len([r for r in results if 'error' in r]),
    }
    
    # Group by activation
    by_activation = {}
    for r in results:
        act = r['activation']
        if act not in by_activation:
            by_activation[act] = []
        by_activation[act].append(r)
    
    summary['by_activation'] = {}
    for act, runs in by_activation.items():
        successful = [r for r in runs if not r.get('diverged', True) and 'error' not in r]
        summary['by_activation'][act] = {
            'total': len(runs),
            'successful': len(successful),
            'diverged': len([r for r in runs if r.get('diverged', False)]),
            'mean_val_acc': np.mean([r['best_val_acc'] for r in successful]) if successful else 0,
            'std_val_acc': np.std([r['best_val_acc'] for r in successful]) if successful else 0,
            'max_val_acc': max([r['best_val_acc'] for r in successful]) if successful else 0,
            'mean_max_qk': np.mean([r['max_qk_score'] for r in successful if not np.isnan(r.get('max_qk_score', float('nan')))]) if successful else 0,
            'mean_max_qk_raw': np.mean([r['max_qk_score_raw'] for r in successful if not np.isnan(r.get('max_qk_score_raw', float('nan')))]) if successful else 0,
            'mean_max_qk_post_clip': np.mean([r['max_qk_score_post_clip'] for r in successful if not np.isnan(r.get('max_qk_score_post_clip', float('nan')))]) if successful else 0,
            'mean_clip_fraction': np.mean([r['mean_clip_fraction'] for r in successful if not np.isnan(r.get('mean_clip_fraction', float('nan')))]) if successful else 0,
            'max_clip_fraction': max([r['max_clip_fraction'] for r in successful if not np.isnan(r.get('max_clip_fraction', float('nan')))], default=0),
        }
    
    # Key findings
    summary['key_findings'] = {
        'softcap_stability': 'Pending analysis',
        'best_init_for_softcap': 'Pending analysis',
        'qk_score_comparison': 'Pending analysis',
    }
    
    summary_file = output_dir / f"{run_tag}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Print quick summary
    print("\n--- Quick Summary ---")
    for act, stats in summary['by_activation'].items():
        print(f"{act}: {stats['successful']}/{stats['total']} successful, "
              f"Mean Val Acc: {stats['mean_val_acc']:.2f}%, "
              f"Max QK(raw/post): {stats['mean_max_qk_raw']:.2f}/{stats['mean_max_qk_post_clip']:.2f}, "
              f"ClipFrac(mean/max): {stats['mean_clip_fraction']:.4f}/{stats['max_clip_fraction']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Muon Optimizer Experiments')
    parser.add_argument('--output-dir', type=str,
                        default='mechanistic_interpretability/optimizer_dynamics/muon/runs',
                        help='Output directory')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: cuda if available)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (single value)')
    parser.add_argument('--wd', type=float, default=None,
                        help='Override weight decay (single value)')
    parser.add_argument('--qk-score-clamp', type=float, default=None,
                        help='Clamp pre-softmax attention scores to this max (proxy QK-clip baseline)')
    parser.add_argument('--embed-dim', type=int, default=None,
                        help='Override ViT embedding dim for replication axis')
    parser.add_argument('--depth', type=int, default=None,
                        help='Override ViT depth for replication axis')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Override ViT num heads for replication axis')
    parser.add_argument('--intervention-test', action='store_true',
                        help='Run intervention test (inject artificial extremes into Q/K)')
    parser.add_argument('--intervention-fraction', type=float, default=0.01,
                        help='Fraction of Q/K values to corrupt (default: 0.01 = 1%%)')
    parser.add_argument('--intervention-scale', type=float, default=5.0,
                        help='Multiplication factor for corrupted values (default: 5.0)')
    parser.add_argument('--softcap-only', action='store_true',
                        help='Run SoftCap variants only (skip controls)')
    parser.add_argument('--verification', action='store_true',
                        help='Run targeted verification set (Top 2 SoftCap + Controls)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity check mode')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Start fresh (do not resume)')
    parser.add_argument('--stop-at-epoch', type=int, default=None,
                        help='Pause execution at this epoch (for resumable runs)')
    
    args = parser.parse_args()
    
    if args.intervention_test:
        run_intervention_test(
            output_dir=Path(args.output_dir),
            device=args.device,
            fraction=args.intervention_fraction,
            scale=args.intervention_scale,
            qk_score_clamp=args.qk_score_clamp,
        )
    else:
        run_preliminary_sweep(
            output_dir=Path(args.output_dir),
            device=args.device,
            epochs=args.epochs,
            num_seeds=args.seeds,
            softcap_only=args.softcap_only,
            quick=args.quick,
            resume=args.resume,
            verification=args.verification,
            stop_at_epoch=args.stop_at_epoch,
            lr_override=args.lr,
            wd_override=args.wd,
            qk_score_clamp=args.qk_score_clamp,
            embed_dim_override=args.embed_dim,
            depth_override=args.depth,
            num_heads_override=args.num_heads,
        )


if __name__ == '__main__':
    main()
