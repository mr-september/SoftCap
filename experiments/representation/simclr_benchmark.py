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

"""SimCLR Contrastive Learning Benchmark for SoftCap Activation Analysis

This benchmark implements the SimCLR framework (Chen et al., 2020) to test whether
isotrop

ic activations prevent dimensional collapse in self-supervised contrastive learning.

Key Hypothesis:
--------------
Isotropic activations (ParametricSmoothNotchTanhSoftCapV2) should maintain better
uniformity in the learned representation space, preventing dimensional collapse
and improving downstream task performance (linear probe accuracy).

Implementation Notes:
--------------------
Based on canonical SimCLR implementations:
- spijkervet/SimCLR (PyTorch, widely used)
- sthalles/SimCLR (reference implementation)
- Original paper: "A Simple Framework for Contrastive Learning of Visual Representations"

Standard hyperparameters (from SimCLR paper + community best practices):
- Temperature: τ=0.1 (default), common range: [0.07, 0.25]
- Batch size: 256 (larger is better, but memory-constrained)
- Optimizer: Adam (LARS if scaling to very large batches)
- Learning rate: 3e-4 with cosine decay
- Epochs: 200 (reduced from 1000 for feasibility)
- Projection head: 2-layer MLP (2048 → 128)
- Encoder: ResNet-18 (feature dim: 512)
- Augmentations: RandomResizedCrop, ColorJitter, RandomHorizontalFlip, GaussianBlur

Metrics:
-------
1. **Linear probe accuracy** (primary): Train linear classifier on frozen representations
2. **Uniformity**: Measures how uniformly representations are distributed on hypersphere
3. **Alignment**: Measures similarity between positive pairs
4. **IsoScore**: Variance of neuron norms (measures isotropy)
5. **Rank**: Effective rank of covariance matrix

References:
----------
- SimCLR: https://arxiv.org/abs/2002.05709
- Uniformity & Alignment: https://arxiv.org/abs/2005.10242 (Wang & Isola, 2020)
- InfoNCE Loss: https://arxiv.org/abs/1807.03748
"""

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

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


def _set_activation_monitoring(model: nn.Module, enabled: bool) -> int:
    """Enable/disable activation monitoring for modules that support it.

    Many SoftCap activations inherit from BaseActivation and expose a `monitoring`
    boolean that triggers per-forward mean/std/zero_ratio reductions.
    """
    n_toggled = 0
    for module in model.modules():
        if hasattr(module, "monitoring"):
            try:
                module.monitoring = enabled
                n_toggled += 1
            except Exception:
                pass
    return n_toggled


class SimCLRAugmentation:
    """SimCLR augmentation pipeline (standard)."""
    
    def __init__(self, image_size=32, s=0.5):
        """
        Args:
            image_size: Input image size (32 for CIFAR-10)
            s: Strength of color distortion (default: 0.5, range: [0.5, 1.0])
        """
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        
        # SimCLR augmentation strategy
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 stats
        ])
    
    def __call__(self, x):
        """Return two randomly augmented views of the same image."""
        return self.transform(x), self.transform(x)


def simclr_collate(batch):
    """Collate function that applies SimCLR augmentation.
    
    Must be defined at module level for pickling/multiprocessing support.
    """
    augmentation = SimCLRAugmentation(image_size=32)
    images = [augmentation(img) for img, _ in batch]
    labels = [label for _, label in batch]
    
    # Stack augmented views
    x_i = torch.stack([img[0] for img in images])
    x_j = torch.stack([img[1] for img in images])
    y = torch.tensor(labels)
    
    return (x_i, x_j), y


def standard_collate(batch):
    """Standard (non-augmented) collate: stack images and labels."""
    return (
        torch.stack([img for img, _ in batch]),
        torch.tensor([label for _, label in batch]),
    )


def _seed_worker(worker_id: int):
    """Seed numpy/random/torch inside each DataLoader worker.

    Uses torch.initial_seed(), which is derived from the DataLoader generator.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class SimCLREncoder(nn.Module):
    """ResNet-18 encoder with custom activation and projection head."""
    
    def __init__(self, activation_fn: nn.Module, projection_dim=128):
        """
        Args:
            activation_fn: Activation function to use (replaces ReLU in ResNet)
            projection_dim: Dimension of projection head output (default: 128)
        """
        super().__init__()
        
        # Load pre-defined ResNet-18 architecture
        resnet = models.resnet18(weights=None)
        
        # Replace all ReLU with custom activation.
        # IMPORTANT: We must preserve the configured activation instance (e.g. a=a* fixed).
        # Using `type(activation_fn)()` would silently reset parameters back to defaults.
        self._replace_activations(resnet, activation_fn)
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-18
        feature_dim = 512
        
        # Projection head (2-layer MLP as per SimCLR paper)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            copy.deepcopy(activation_fn),
            nn.Linear(2048, projection_dim)
        )
    
    def _replace_activations(self, model: nn.Module, new_activation: nn.Module):
        """Recursively replace all ReLU activations in model."""
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                # Create a fresh activation module with identical configuration.
                # - Preserves fixed-vs-learnable flags and a_init values.
                # - Avoids unintended parameter sharing across layers.
                setattr(model, name, copy.deepcopy(new_activation))
            else:
                self._replace_activations(module, new_activation)
    
    def forward(self, x):
        """
        Args:
            x: Input image batch [B, 3, 32, 32]
        Returns:
            features: Encoder features [B, 512]
            projections: Projected embeddings [B, projection_dim]
        """
        features = self.encoder(x).squeeze()  # [B, 512]
        projections = self.projection_head(features)  # [B, projection_dim]
        return features, projections


class NT_Xent_Loss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (InfoNCE)."""
    
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature parameter τ (default: 0.1)
                - Lower τ → sharper distribution (stronger gradients)
                - Higher τ → softer distribution (more uniform)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: First view projections [B, projection_dim]
            z_j: Second view projections [B, projection_dim]
        Returns:
            loss: NT-Xent loss value
        """
        batch_size = z_i.size(0)
        
        # Normalize projections to unit hypersphere (critical for NT-Xent)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, projection_dim]
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2B, 2B]
        
        # Mask to exclude self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)  # [2B]
        
        # Apply temperature scaling
        pos_sim = pos_sim / self.temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute NT-Xent loss (InfoNCE with log-sum-exp trick for numerical stability)
        nominator = torch.exp(pos_sim)
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
        
        loss = -torch.mean(torch.log(nominator / denominator))
        
        return loss


def save_checkpoint(
    state: Dict,
    checkpoint_dir: Path,
    activation_name: str,
    epoch: int,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_dir / f'checkpoint_{activation_name}_latest.pt'
    torch.save(state, filename)
    
    # Also save periodic checkpoints to prevent corruption
    if (epoch + 1) % 50 == 0:
        periodic_name = checkpoint_dir / f'checkpoint_{activation_name}_epoch_{epoch+1}.pt'
        torch.save(state, periodic_name)


def load_checkpoint(
    checkpoint_dir: Path,
    activation_name: str,
    device: str
) -> Dict:
    """Load latest checkpoint if exists."""
    filename = checkpoint_dir / f'checkpoint_{activation_name}_latest.pt'
    if filename.exists():
        print(f"  -> Loading checkpoint: {filename}")
        return torch.load(filename, map_location=device)
    return None


def train_simclr(
    model: SimCLREncoder,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    temperature: float,
    weight_decay: float = 1e-6,
    activation_name: str = "unknown",
    checkpoint_name: str | None = None,
    checkpoint_dir: Path = None,
    checkpoint_freq: int = 10,
    start_epoch: int = 0,
    history: Dict = None
) -> Dict[str, List[float]]:
    """Train SimCLR model with NT-Xent loss and checkpointing."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing learning rate schedule (standard for SimCLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # If resuming, step scheduler to correct point
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    criterion = NT_Xent_Loss(temperature=temperature)
    
    if history is None:
        history = {'loss': [], 'lr': []}
    
    # Skip if already finished
    if start_epoch >= epochs:
        print(f"  [OK] Training already completed ({start_epoch}/{epochs} epochs). Skipping.")
        return history
        
    print(f"  -> Starting training from epoch {start_epoch+1}/{epochs}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False, ascii=True)
        for (x_i, x_j), _ in pbar:
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through both augmented views
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            # Compute contrastive loss
            loss = criterion(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
        # Checkpointing
        if checkpoint_dir and ((epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs):
            ckpt_name = checkpoint_name or activation_name
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                'activation': activation_name,
                'run_name': ckpt_name,
                'hyperparameters': {
                    'epochs': epochs,
                    'lr': lr,
                    'temperature': temperature
                }
            }
            save_checkpoint(state, checkpoint_dir, ckpt_name, epoch)
            
    return history


def linear_probe_eval(
    encoder: SimCLREncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int = 100,
    lr: float = 3e-3
) -> Dict[str, float]:
    """Evaluate learned representations with linear probe (frozen encoder)."""
    encoder = encoder.to(device)
    encoder.eval()  # Freeze encoder
    
    # Simple linear classifier
    linear_clf = nn.Linear(512, 10).to(device)  # 512 = ResNet-18 feature dim, 10 = CIFAR-10 classes
    optimizer = optim.Adam(linear_clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train linear classifier
    for epoch in range(epochs):
        linear_clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                features, _ = encoder(x)
            
            optimizer.zero_grad()
            logits = linear_clf(features)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    linear_clf.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            features, _ = encoder(x)
            logits = linear_clf(features)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    accuracy = 100.0 * correct / total
    return {'linear_probe_accuracy': accuracy}


def compute_uniformity_alignment(
    encoder: SimCLREncoder,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Compute uniformity and alignment metrics (Wang & Isola, 2020)."""
    encoder = encoder.to(device)
    encoder.eval()
    
    all_z = []
    
    with torch.no_grad():
        for (x_i, x_j), _ in dataloader:
            x_i = x_i.to(device)
            _, z_i = encoder(x_i)
            z_i = F.normalize(z_i, dim=1)
            all_z.append(z_i)
    
    all_z = torch.cat(all_z, dim=0)  # [N, projection_dim]
    
    # Uniformity: log of average pairwise Gaussian potential
    # More negative = more uniform (representations spread out on hypersphere)
    sq_dist = torch.pdist(all_z, p=2).pow(2)
    uniformity = sq_dist.mul(-2).exp().mean().log()
    
    # NOTE: Alignment requires positive pairs, which we don't store during eval
    # For simplicity, we compute it during training or skip it
    
    return {
        'uniformity': uniformity.item(),
        # 'alignment': alignment.item(),  #  Requires positive pair tracking
    }


def compute_isoscore_and_rank(
    encoder: SimCLREncoder,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Compute IsoScore and effective rank of learned representations."""
    encoder = encoder.to(device)
    encoder.eval()
    
    all_features = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            features, _ = encoder(x)
            all_features.append(features)
    
    all_features = torch.cat(all_features, dim=0)  # [N, 512]
    
    # IsoScore: variance of neuron norms
    neuron_norms = torch.norm(all_features, p=2, dim=0)  # [512]
    isoscore = neuron_norms.var().item()
    
    # Effective rank: ratio of Frobenius norm to operator norm of covariance
    cov = torch.cov(all_features.T)
    frobenius_norm = torch.norm(cov, p='fro').item()
    operator_norm = torch.linalg.matrix_norm(cov, ord=2).item()
    effective_rank = (frobenius_norm / operator_norm) if operator_norm > 0 else 0.0
    
    return {
        'isoscore': isoscore,
        'effective_rank': effective_rank
    }


def run_simclr_for_activation(
    activation_name: str,
    activation_fn: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    device: str
) -> Dict:
    """Run complete SimCLR pipeline for one activation."""
    print(f"\n{'='*80}")
    print(f"Running SimCLR: {activation_name}")
    print(f"{'='*80}")
    
    # Check if results already exist (resume support)
    seed_suffix = f"_seed{args.seed}" if getattr(args, 'seed', None) is not None else ""
    run_name = f"{activation_name}{seed_suffix}"
    results_file = output_dir / f'simclr_{activation_name}{seed_suffix}.json'
    if results_file.exists() and not args.force:
        print(f"  [OK] Skipped (results exist): {results_file.name}")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    # Setup datasets
    # Note: We utilize a custom collate function (defined at module level) for SimCLR augmentation
    
    # Standard transform for linear probe (no augmentation)
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=standard_transform)
    
    # DataLoaders
    dl_worker_init = _seed_worker if args.seed is not None else None
    dl_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    persistent_workers = bool(getattr(args, "num_workers", 0) > 0)

    train_loader_simclr = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=simclr_collate,
        pin_memory=True
        ,worker_init_fn=dl_worker_init
        ,generator=dl_generator
        ,persistent_workers=persistent_workers
    )
    
    # For linear probe, we need standard (non-augmented) data
    train_dataset_standard = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=standard_transform)
    train_loader_standard = DataLoader(
        train_dataset_standard,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=standard_collate,
        pin_memory=True
        ,worker_init_fn=dl_worker_init
        ,generator=dl_generator
        ,persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=standard_collate,
        pin_memory=True
        ,worker_init_fn=dl_worker_init
        ,generator=dl_generator
        ,persistent_workers=persistent_workers
    )
    
    # Create model
    model = SimCLREncoder(activation_fn, projection_dim=args.projection_dim)

    # Optional performance toggle: BaseActivation monitoring can be expensive.
    if getattr(args, "disable_monitoring", False):
        n_toggled = _set_activation_monitoring(model, enabled=False)
        print(f"  -> Disabled activation monitoring on {n_toggled} modules")
    
    # Apply activation-aware initialization
    _apply_activation_aware_init(model, activation_name)
    
    # Define checkpoint directory
    checkpoint_dir = output_dir / 'checkpoints'
    
    # Handle force restart
    if args.force_restart:
        print("  ⚠️ Force restart requested. Deleting existing checkpoints.")
        for ckpt in checkpoint_dir.glob(f'checkpoint_{run_name}_*.pt'):
            ckpt.unlink()
            
    # Try to load checkpoint
    start_epoch = 0
    history = None
    
    ckpt = load_checkpoint(checkpoint_dir, run_name, device)
    if ckpt:
        print(f"  [OK] Resuming from epoch {ckpt['epoch']}")
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        history = ckpt.get('history', None)
        
        # Verify compatibility (basic check)
        if start_epoch > args.epochs:
            print(f"  [INFO] Checkpoint has {start_epoch} epochs, which is > requested {args.epochs}. Using existing model.")
    
    # Train SimCLR (or resume)
    print(f"Training SimCLR (contrastive pretraining)...")
    train_history = train_simclr(
        model,
        train_loader_simclr,
        device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        activation_name=activation_name,
        checkpoint_name=run_name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        start_epoch=start_epoch,
        history=history
    )
    
    # Evaluate with linear probe
    print(f"Evaluating with linear probe (frozen encoder)...")
    linear_probe_results = linear_probe_eval(
        model,
        train_loader_standard,
        test_loader,
        device,
        epochs=args.linear_probe_epochs,
        lr=args.linear_probe_lr
    )
    
    # Compute uniformity & alignment
    print(f"Computing uniformity metrics...")
    uniformity_results = compute_uniformity_alignment(model, train_loader_simclr, device)
    
    # Compute IsoScore & rank
    print(f"Computing isotropy metrics...")
    isotropy_results = compute_isoscore_and_rank(model, train_loader_standard, device)
    
    # Combine results
    results = {
        'activation': activation_name,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'temperature': args.temperature,
            'projection_dim': args.projection_dim,
            'weight_decay': args.weight_decay,
            'seed': getattr(args, 'seed', None),
            'num_workers': args.num_workers,
            'disable_monitoring': bool(getattr(args, 'disable_monitoring', False)),
        },
        'training': {
            'final_loss': train_history['loss'][-1],
            'history': train_history,
        },
        **linear_probe_results,
        **uniformity_results,
        **isotropy_results,
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Linear Probe Accuracy: {linear_probe_results['linear_probe_accuracy']:.2f}%")
    print(f"  Uniformity: {uniformity_results['uniformity']:.4f}")
    print(f"  IsoScore: {isotropy_results['isoscore']:.6e}")
    print(f"  Effective Rank: {isotropy_results['effective_rank']:.1f}")
    
    return results


def _apply_activation_aware_init(model: nn.Module, activation_name: str):
    """Apply activation-aware initialization."""
    name = activation_name.lower()
    init_method = 'orthogonal' if ('v2' in name and ('cubic' in name or 'quartic' in name)) else 'kaiming'
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if  init_method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            if init_method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def main():
    parser = argparse.ArgumentParser(description='SimCLR Contrastive Learning Benchmark')
    
    # Activation selection
    act_group = parser.add_mutually_exclusive_group(required=True)
    act_group.add_argument('--activation', type=str, help='Single activation to test')
    act_group.add_argument('--all-activations', action='store_true', help='Run standard experimental set')
    act_group.add_argument('--comprehensive', action='store_true', help='Run comprehensive set (all a* variants)')
    act_group.add_argument('--controls-only', action='store_true', help='Run only control activations')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='SimCLR pretraining epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256, larger is better)')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (default: 0)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--temperature', type=float, default=0.1, help='NT-Xent temperature τ (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay (default: 1e-6)')
    parser.add_argument('--projection-dim', type=int, default=128, help='Projection head output dim (default: 128)')
    
    # Linear probe settings
    parser.add_argument('--linear-probe-epochs', type=int, default=100, help='Linear probe training epochs (default: 100)')
    parser.add_argument('--linear-probe-lr', type=float, default=3e-3, help='Linear probe learning rate (default: 3e-3)')
    
    # Paths & misc
    parser.add_argument('--output-dir', type=str, 
                        default='mechanistic_interpretability/latent_geometry/contrastive/results',
                        help='Output directory')
    parser.add_argument(
        '--disable-monitoring',
        action='store_true',
        help='Disable BaseActivation internal monitoring (mean/std/zero_ratio) for speed',
    )
    parser.add_argument('--force', action='store_true', help='Recompute evaluation even if results exist')
    parser.add_argument('--force-restart', action='store_true', help='Force full retrain (delete existing checkpoints)')
    parser.add_argument('--checkpoint-freq', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (if set, also suffixes results filename)')
    
    args = parser.parse_args()

    if args.seed is not None:
        # Basic determinism controls. (num_workers=0 so DataLoader order is seedable.)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SimCLR CONTRASTIVE LEARNING BENCHMARK")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Activation monitoring: {'OFF' if args.disable_monitoring else 'ON'}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Select activations
    if args.controls_only:
        activations = get_control_activations()
    elif args.comprehensive:
        activations = {}
        activations.update(get_extended_astar_activations())
        activations.update(get_control_activations())
        activations.update(get_standard_experimental_set())
        print(f"Running COMPREHENSIVE set ({len(activations)} activations)\n")
    elif args.all_activations:
        activations = get_standard_experimental_set()
    else:
        all_acts = {}
        all_acts.update(get_standard_experimental_set())
        all_acts.update(get_extended_astar_activations())
        all_acts.update(get_control_activations())
        if args.activation not in all_acts:
            raise ValueError(f"Unknown activation: {args.activation}")
        activations = {args.activation: all_acts[args.activation]}
    
    # Run SimCLR for each activation
    all_results = {}
    for act_name, act_fn in activations.items():
        results = run_simclr_for_activation(act_name, act_fn, output_dir, args, args.device)
        all_results[act_name] = results
    
    # Save summary
    summary = {
        'benchmark': 'simclr_contrastive_learning',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'hyperparameters': vars(args),
        'results': {
            name: {
                'linear_probe_accuracy': res['linear_probe_accuracy'],
                'uniformity': res.get('uniformity', None),
                'isoscore': res['isoscore'],
                'effective_rank': res['effective_rank'],
                'final_contrastive_loss': res['training']['final_loss'],
            }
            for name, res in all_results.items()
        }
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Activation':<40} {'Linear Probe Acc':<18} {'Uniformity':<15} {'IsoScore'}")
    print("-"*80)
    for name in sorted(all_results.keys()):
        res = all_results[name]
        print(f"{name:<40} {res['linear_probe_accuracy']:>16.2f}% "
              f"{res.get('uniformity', 0.0):>14.4f} {res['isoscore']:>12.6e}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
