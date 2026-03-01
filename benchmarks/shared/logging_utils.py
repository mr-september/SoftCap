"""
Logging utilities for scaling benchmarks.

Provides JSON-based metric logging and optional TensorBoard integration.
Each experiment run produces a structured JSON log file.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkLogger:
    """Append-only JSON-lines logger for a single experiment run.

    Also writes to TensorBoard if a SummaryWriter is provided.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str,
        tb_writer=None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.log_path = self.log_dir / f"{run_name}.jsonl"
        self.tb_writer = tb_writer
        self._metadata: Dict[str, Any] = {}
        self._start_time = time.time()

    def set_metadata(self, **kwargs) -> None:
        """Store run-level metadata (written in the summary)."""
        self._metadata.update(kwargs)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch."""
        record = {"epoch": epoch, "wall_time": time.time() - self._start_time}
        record.update(metrics)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"{self.run_name}/{k}", v, epoch)

    def save_summary(self, final_metrics: Dict[str, Any]) -> None:
        """Write a single JSON summary file for the entire run."""
        summary = {
            "run_name": self.run_name,
            "total_wall_time_s": time.time() - self._start_time,
            **self._metadata,
            **final_metrics,
        }
        summary_path = self.log_dir / f"{self.run_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()


def load_run_summaries(log_dir: str) -> List[Dict[str, Any]]:
    """Load all *_summary.json files from a log directory."""
    summaries = []
    log_path = Path(log_dir)
    for p in sorted(log_path.glob("*_summary.json")):
        with open(p) as f:
            summaries.append(json.load(f))
    return summaries
