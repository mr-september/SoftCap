#!/usr/bin/env python3
"""Canonical activation naming checks for the public API."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from softcap import get_standard_experimental_set


def _load_activations_text() -> str:
    activations_path = PROJECT_ROOT / "softcap" / "activations.py"
    return activations_path.read_text(encoding="utf-8")


def test_legacy_aliases_map_to_canonical_classes():
    text = _load_activations_text()
    assert "class ParametricTanhSoftCap(SoftCap):" in text
    assert "class ParametricSmoothNotchTanhSoftCapV2(SwishCap):" in text
    assert "class ParametricQuinticNotchTanhSoftCap(SparseCap):" in text


def test_standard_set_exposes_canonical_names():
    activations = get_standard_experimental_set()
    assert "SoftCap" in activations
    assert "SwishCap" in activations
    assert "SparseCap" in activations
