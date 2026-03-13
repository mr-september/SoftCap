from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_muon_verification_defaults_match_release_suite():
    from scripts.experiments.run_muon import get_verification_specs

    specs = get_verification_specs()

    default_enabled = {name for name, spec in specs.items() if spec.get("default_enabled")}
    assert default_enabled == {"SoftCap", "SwishCap", "SparseCap", "ReLU", "GELU"}
    assert specs["ReLU6"]["default_enabled"] is False
    assert specs["HardTanh"]["default_enabled"] is False
