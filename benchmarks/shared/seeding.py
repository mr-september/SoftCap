"""
Deterministic seeding for reproducible benchmarks.

Sets all sources of randomness to the same seed so that
same-seed runs produce bitwise-identical results.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Disables non-deterministic cuDNN algorithms. This costs ~10-20% wall-clock
    time but guarantees bitwise reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.x deterministic algorithms flag
    torch.use_deterministic_algorithms(True, warn_only=True)
