#!/usr/bin/env python3
"""Canonical activation API and deprecation policy tests."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_top_level_exports_are_canonical_only():
    from softcap import SoftCap, SwishCap, SparseCap

    assert SoftCap is not None
    assert SwishCap is not None
    assert SparseCap is not None


def test_legacy_alias_not_top_level_importable():
    with pytest.raises(ImportError):
        from softcap import ParametricTanhSoftCap  # noqa: F401


def test_legacy_aliases_warn_on_instantiation():
    from softcap.activations import (
        ParametricSmoothNotchTanhSoftCapV2,
        ParametricTanhSoftCap,
        ParametricQuinticNotchTanhSoftCap,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        ParametricTanhSoftCap()
        ParametricSmoothNotchTanhSoftCapV2()
        ParametricQuinticNotchTanhSoftCap()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) >= 3


def test_removed_cubic_quartic_classes_not_present():
    import softcap.activations as activations

    removed = [
        "ParametricCubicNotchTanhSoftCapV2",
        "ParametricCubicNotchTanhSoftCapV3",
        "ParametricCubicNotchTanhSoftCapV4",
        "ParametricQuarticNotchTanhSoftCapV2",
        "ParametricQuarticNotchTanhSoftCapV3",
    ]
    for name in removed:
        assert not hasattr(activations, name), f"Unexpected legacy class still present: {name}"


def test_default_activation_helpers_keep_release_surface_small():
    from softcap.activations import get_default_activations, get_full_default_activations
    from softcap.compatibility import get_all_activations

    assert set(get_default_activations()) == {
        "SoftCap",
        "SwishCap",
        "SparseCap",
        "ReLU",
        "Tanh",
        "GELU",
        "SiLU",
    }
    assert {"ReLU6", "HardTanh"}.issubset(set(get_full_default_activations()))
    assert {"ReLU6", "HardTanh"}.issubset(set(get_all_activations()))
