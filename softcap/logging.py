"""
Shared logging and seeding utilities for SoftCap benchmarks.
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch


def seed_everything(seed: int):
    """Sets seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BenchmarkLogger:
    """Simple logger for benchmark results and telemetry."""
    
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.summary_file = self.run_dir / "summary.json"
        self.metadata: Dict[str, Any] = {}

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        data = {"epoch": epoch}
        data.update(metrics)
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def save_summary(self, summary: Dict[str, Any]):
        data = {"metadata": self.metadata, "summary": summary}
        with open(self.summary_file, "w") as f:
            json.dump(data, f, indent=2)

    def close(self):
        pass
