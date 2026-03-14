from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_canonical_activation_suites_are_stable():
    from softcap.control_activations import (
        get_baseline_controls,
        get_bounded_controls,
        get_full_control_activations,
        get_full_experimental_set,
        get_core_activations,
        get_control_activations,
        get_standard_experimental_set,
    )

    assert set(get_core_activations()) == {"SoftCap", "SwishCap", "SparseCap"}
    assert set(get_baseline_controls()) == {"ReLU", "Tanh", "GELU", "SiLU"}
    assert set(get_bounded_controls()) == {"ReLU6", "HardTanh"}
    assert set(get_control_activations()) == {"ReLU", "Tanh", "GELU", "SiLU"}
    assert set(get_full_control_activations()) == {"ReLU", "Tanh", "GELU", "SiLU", "ReLU6", "HardTanh"}
    assert set(get_standard_experimental_set()) == {
        "SoftCap",
        "SwishCap",
        "SparseCap",
        "ReLU",
        "Tanh",
        "GELU",
        "SiLU",
    }
    assert set(get_full_experimental_set()) == {
        "SoftCap",
        "SwishCap",
        "SparseCap",
        "ReLU",
        "Tanh",
        "GELU",
        "SiLU",
        "ReLU6",
        "HardTanh",
    }


def test_astar_suite_exposes_fixed_canonical_variants():
    from softcap.control_activations import get_astar_activations

    astar = get_astar_activations()
    assert set(astar) == {
        "SoftCap_astar_fixed",
        "SwishCap_astar_fixed",
        "SparseCap_astar_fixed",
    }
    for activation in astar.values():
        assert hasattr(activation, "a")
        assert activation.a.requires_grad is False


def test_extended_astar_suite_keeps_canonical_and_legacy_aliases():
    from softcap.control_activations import get_extended_astar_activations

    extended = get_extended_astar_activations()

    required = {
        "SoftCap_1_fixed",
        "SoftCap_astar_learnable",
        "SwishCap_1_learnable",
        "SparseCap_astar_fixed",
        "TanhSoftCap_1_fixed",
        "SmoothNotchV2_astar_fixed",
        "QuinticNotch_1_learnable",
    }
    assert required.issubset(set(extended))

    assert extended["SoftCap_1_fixed"].a.requires_grad is False
    assert extended["SoftCap_astar_learnable"].a.requires_grad is True
