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
Safe efficiency optimizations for benchmark training.

These are techniques that speed up training without affecting model
accuracy or optimization dynamics. Every technique here is standard
in the ML community and accepted in academic papers.

References:
    - NanoGPT speedrun community (2024-2025): torch.compile, AMP, channels-last
    - PyTorch docs: torch.cuda.amp, torch.compile, DataLoader best practices
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional


def apply_channels_last(model: nn.Module) -> nn.Module:
    """Convert a conv-net to channels-last memory format.

    On modern GPUs this can give 10-20% speedup for convolutions with no
    accuracy impact whatsoever. The weights stay in the same logical order;
    only the physical memory layout changes.

    Only useful for models with Conv2d layers (vision).
    """
    return model.to(memory_format=torch.channels_last)


def make_efficient_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create an optimized DataLoader with standard best-practice settings.

    - ``pin_memory``: Speeds up CPU→GPU transfers via page-locked memory.
    - ``persistent_workers``: Avoids re-spawning worker processes each epoch.
    - ``num_workers``: Overlaps data loading with compute.

    None of these affect numerical results.
    """
    # persistent_workers requires num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=use_persistent,
        drop_last=drop_last,
    )


def maybe_compile(model: nn.Module, enabled: bool = True) -> nn.Module:
    """Optionally apply torch.compile for 10-30% speedup.

    torch.compile is a no-op on PyTorch < 2.0. On 2.x+ it JIT-compiles
    the model graph, fusing ops and reducing Python overhead. The compiled
    model produces identical numerical results to the eager-mode model.

    Disable for debugging (stack traces are cleaner without compile).
    """
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception:
        # Compilation can fail on some model graphs; fall back gracefully
        return model


def get_amp_context(enabled: bool = True, dtype=None):
    """Return an AMP autocast context manager.

    Mixed precision (FP16 on Turing, BF16 on Ampere+) is standard in modern
    ML training. On RTX 2080 Ti (Turing), this uses FP16 with dynamic loss
    scaling. Accuracy impact is negligible when gradient scaling is used.

    Args:
        enabled: Whether to enable AMP.
        dtype: Override autocast dtype (default: auto-selected by PyTorch).
    """
    if dtype is None:
        # RTX 2080 Ti = Turing → no native BF16, use FP16
        dtype = torch.float16
    return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=dtype)
