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
ResNet-18 / CIFAR-100 Scaling Benchmark

Trains ResNet-18 on CIFAR-100 with swappable activation functions.
All hyperparameters are FIXED across runs — only the activation changes.

Cap-aware training:
    - Learnable scalar `a` parameters get a separate optimizer group with
      lower LR (LR × 0.1) and no weight decay, preventing the collapse
      observed in legacy ResNet-20/CIFAR-100 runs (SoftCap → 1.06%).
    - Controls (ReLU/GELU/SiLU) have no `a` parameters, so their training
      is identical to standard practice.

Usage (from repo root, in WSL venv):
    # Full benchmark (all activations × 3 seeds):
    python benchmarks/vision/train_resnet18_cifar100.py

    # Quick sanity check (10 epochs):
    python benchmarks/vision/train_resnet18_cifar100.py --quick

    # Mini-sweep (a=a* variants, 50 epochs, 1 seed):
    python benchmarks/vision/train_resnet18_cifar100.py --minisweep

    # With warmup (optional, helps Cap convergence):
    python benchmarks/vision/train_resnet18_cifar100.py --warmup 5

Expected baselines:
    ReLU  → ~77.5 ± 0.3%
    GELU  → ~77.5 ± 0.3%
    SiLU  → ~77.5 ± 0.3%

Hardware requirements:
    ~2 GB VRAM per run (ResNet-18 + CIFAR-100 + AMP)
    ~50 min per 200-epoch run on RTX 2080 Ti with AMP
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.seeding import seed_everything
from benchmarks.shared.logging_utils import BenchmarkLogger
from benchmarks.shared.activation_registry import (
    get_vision_activations,
    get_vision_minisweep_activations,
    clone_activation,
)
from benchmarks.shared.efficiency import (
    apply_channels_last,
    make_efficient_dataloader,
    maybe_compile,
    get_amp_context,
)
from softcap.initialization import apply_initialization, get_recommended_init

# ---------------------------------------------------------------------------
# Fixed hyperparameters (DO NOT CHANGE across activation runs)
# ---------------------------------------------------------------------------
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEEDS = [42, 123, 456]

# Learnable `a` parameter training — separate group prevents the collapse
# observed in legacy ResNet-20/CIFAR-100 runs (SoftCap → 1.06% at LR=0.1).
A_LR_FACTOR = 0.1       # a_lr = LR * 0.1 = 0.01
A_WEIGHT_DECAY = 0.0    # WD on `a` pushes it toward 0 (conceptually wrong)

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar100_loaders(batch_size: int, data_dir: str = "data"):
    """Standard CIFAR-100 train/test loaders with canonical augmentation."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = make_efficient_dataloader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = make_efficient_dataloader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp,
                    label_smoothing=0.0):
    """Train for one epoch. Returns (avg_loss, accuracy, epoch_time)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_amp_context(enabled=use_amp):
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets,
                                               label_smoothing=label_smoothing)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    elapsed = time.time() - t0
    return total_loss / total, 100.0 * correct / total, elapsed


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Evaluate on test set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with get_amp_context(enabled=use_amp):
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def run_single(
    act_name: str,
    act_fn: nn.Module,
    init_method: str,
    seed: int,
    epochs: int,
    device: torch.device,
    log_dir: str,
    data_dir: str,
    use_amp: bool = True,
    use_compile: bool = False,
    warmup_epochs: int = 0,
    label_smoothing: float = 0.0,
):
    """Run a single training experiment (one activation × one seed)."""
    seed_everything(seed)

    run_name = f"{act_name}_seed{seed}"
    logger = BenchmarkLogger(log_dir, run_name)
    logger.set_metadata(
        activation=act_name,
        seed=seed,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        init_method=init_method,
        use_amp=use_amp,
        use_compile=use_compile,
        warmup_epochs=warmup_epochs,
        label_smoothing=label_smoothing,
        a_lr_factor=A_LR_FACTOR,
        a_weight_decay=A_WEIGHT_DECAY,
        dataset="CIFAR-100",
    )

    # Data
    train_loader, test_loader = get_cifar100_loaders(BATCH_SIZE, data_dir)

    # Model
    from benchmarks.vision.resnet18 import resnet18

    model = resnet18(clone_activation(act_fn), num_classes=100)
    apply_initialization(model, init_method, act_name)
    model = apply_channels_last(model)
    model = model.to(device)
    model = maybe_compile(model, enabled=use_compile)

    # Disable per-forward monitoring in benchmarked activations (avoid CUDA syncs)
    for m in model.modules():
        if hasattr(m, "monitoring"):
            m.monitoring = False

    # Build optimizer with separate param group for learnable `a`
    activation_a_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.a') and param.numel() == 1:
            activation_a_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
    ]
    if activation_a_params:
        param_groups.append({
            "params": activation_a_params,
            "lr": LR * A_LR_FACTOR,
            "weight_decay": A_WEIGHT_DECAY,
        })
        print(f"  \u2192 {len(activation_a_params)} learnable `a` param(s) "
              f"at lr={LR * A_LR_FACTOR}, wd={A_WEIGHT_DECAY}")

    optimizer = optim.SGD(param_groups, momentum=MOMENTUM)

    # LR scheduler (with optional warmup)
    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            label_smoothing=label_smoothing,
        )
        test_loss, test_acc = evaluate(model, test_loader, device, use_amp)
        scheduler.step()

        logger.log_epoch(epoch, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": epoch_time,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            # Save best checkpoint
            ckpt_path = Path(log_dir) / f"{run_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, ckpt_path)

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"  [{run_name}] Epoch {epoch}/{epochs}  "
                f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%  "
                f"best={best_acc:.2f}%  ({epoch_time:.1f}s)"
            )

    # Save final summary
    logger.save_summary({
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
    })
    logger.close()
    print(f"  ✓ {run_name} done — best test acc: {best_acc:.2f}% (epoch {best_epoch})")
    return {"best_test_acc": best_acc, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 / CIFAR-100 activation benchmark"
    )
    parser.add_argument("--activation", type=str, default=None,
                        help="Run a single activation (e.g. 'ReLU', 'SoftCap')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: run all 3)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 10 epochs, 1 seed")
    parser.add_argument("--minisweep", action="store_true",
                        help="Run a-star mini-sweep: 50 epochs, 1 seed")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (experimental speedup)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Linear warmup epochs (0=disabled, try 5 for Cap)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0=disabled, try 0.1)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Dataset root directory")
    parser.add_argument("--log-dir", type=str, default="benchmarks/vision/results",
                        help="Output log directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = not args.no_amp and torch.cuda.is_available()

    # Determine activations & seeds
    if args.minisweep:
        activations = get_vision_minisweep_activations()
        seeds = [42]
        epochs = args.epochs or 50
        print(f"\n=== Mini-sweep mode: {len(activations)} activations × 1 seed × {epochs} epochs ===\n")
    elif args.quick:
        activations = get_vision_activations()
        seeds = [42]
        epochs = args.epochs or 10
        print(f"\n=== Quick mode: {len(activations)} activations × 1 seed × {epochs} epochs ===\n")
    else:
        activations = get_vision_activations()
        seeds = SEEDS
        epochs = args.epochs or EPOCHS
        print(f"\n=== Full benchmark: {len(activations)} activations × {len(seeds)} seeds × {epochs} epochs ===\n")
    if args.warmup > 0:
        print(f"  Warmup: {args.warmup} epochs")
    if args.label_smoothing > 0:
        print(f"  Label smoothing: {args.label_smoothing}")
    # Filter to single activation if requested
    if args.activation:
        if args.activation not in activations:
            print(f"Unknown activation: {args.activation}")
            print(f"Available: {list(activations.keys())}")
            sys.exit(1)
        activations = {args.activation: activations[args.activation]}

    if args.seed is not None:
        seeds = [args.seed]

    # Run
    all_results = {}
    for act_name, (act_fn, init_method) in activations.items():
        print(f"\n{'='*60}")
        print(f"Activation: {act_name}  |  Init: {init_method}  |  Seeds: {seeds}")
        print(f"{'='*60}")
        act_results = []
        for seed in seeds:
            result = run_single(
                act_name=act_name,
                act_fn=act_fn,
                init_method=init_method,
                seed=seed,
                epochs=epochs,
                device=device,
                log_dir=args.log_dir,
                data_dir=args.data_dir,
                use_amp=use_amp,
                use_compile=args.compile,
                warmup_epochs=args.warmup,
                label_smoothing=args.label_smoothing,
            )
            act_results.append(result)
        all_results[act_name] = act_results

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for act_name, results in all_results.items():
        accs = [r["best_test_acc"] for r in results]
        import numpy as np
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
        print(f"  {act_name:20s}  {mean_acc:.2f} ± {std_acc:.2f}%")

    # Sanity check
    if "ReLU" in all_results:
        relu_accs = [r["best_test_acc"] for r in all_results["ReLU"]]
        relu_mean = sum(relu_accs) / len(relu_accs)
        if relu_mean < 73.0:
            print("\n\u26a0\ufe0f  WARNING: ReLU baseline is below 73%. Something may be wrong.")
            print("    Expected ~77.5%. Debug before proceeding.\n")

    print("\nResults saved to:", args.log_dir)


if __name__ == "__main__":
    main()
