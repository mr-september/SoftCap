"""Lightweight base class for legacy experiment scripts.

This file restores the minimal API expected by older experiment entrypoints
that subclass `DomainExperiment` (e.g., ResNet-20 CIFAR scripts).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict


class DomainExperiment:
    """Minimal base experiment with logger + metadata helpers."""

    def __init__(self, name: str, output_dir: str | None = None, **_: Any) -> None:
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("experiments") / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"experiment.{name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

            fh = logging.FileHandler(log_dir / "experiment.log")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        self.metadata: Dict[str, Any] = {"name": name}

