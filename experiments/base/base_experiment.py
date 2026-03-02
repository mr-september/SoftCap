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

