"""Canonical activation suites used across SoftCap experiments.

Public-facing code should prefer the stable release helpers in this module:

- ``get_core_activations()``: the three Cap-family activations
- ``get_control_activations()``: the matched release controls
- ``get_standard_experimental_set()``: core family + release controls

Appendix-only follow-up controls (currently ``ReLU6`` and ``HardTanh``) are
available via explicit opt-in helpers so the default surface stays aligned with
the simpler publication-facing suite.

Legacy helpers such as ``get_astar_activations()`` and
``get_extended_astar_activations()`` are kept for backward compatibility with
older scripts, but new code should avoid building new public surfaces around
those older regime-specific names.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch.nn as nn

from .activations import (
    GELUWithMetrics,
    HardTanhWithMetrics,
    ReLU6WithMetrics,
    ReLUWithMetrics,
    SiLUWithMetrics,
    SoftCap,
    SparseCap,
    SwishCap,
    TanhWithMetrics,
)


A_STAR_VALUES = {
    "SoftCap": 2.890625,
    "SwishCap": 2.43359375,
    "SparseCap": 2.14,
}


def _clone(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def _make_cap_variant(factory: type[nn.Module], *, a_value: float, learnable: bool) -> nn.Module:
    activation = factory(a_init=float(a_value))
    if hasattr(activation, "a"):
        activation.a.requires_grad_(learnable)
    return activation


def get_core_activations() -> Dict[str, nn.Module]:
    return {
        "SoftCap": SoftCap(),
        "SwishCap": SwishCap(),
        "SparseCap": SparseCap(),
    }


def get_baseline_controls() -> Dict[str, nn.Module]:
    return {
        "ReLU": ReLUWithMetrics(),
        "Tanh": TanhWithMetrics(),
        "GELU": GELUWithMetrics(),
        "SiLU": SiLUWithMetrics(),
    }


def get_bounded_controls() -> Dict[str, nn.Module]:
    return {
        "ReLU6": ReLU6WithMetrics(),
        "HardTanh": HardTanhWithMetrics(),
    }


def get_full_control_activations() -> Dict[str, nn.Module]:
    controls = get_baseline_controls()
    controls.update(get_bounded_controls())
    return controls


def get_control_activations(*, include_bounded: bool = False) -> Dict[str, nn.Module]:
    if include_bounded:
        return get_full_control_activations()
    return get_baseline_controls()


def get_full_experimental_set() -> Dict[str, nn.Module]:
    activations = get_core_activations()
    activations.update(get_full_control_activations())
    return activations


def get_standard_experimental_set(*, include_bounded: bool = False) -> Dict[str, nn.Module]:
    activations = get_core_activations()
    activations.update(get_control_activations(include_bounded=include_bounded))
    return activations


def get_named_activation_suite(name: str) -> Dict[str, nn.Module]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"core", "family", "cap_family"}:
        return get_core_activations()
    if normalized in {"baseline_controls", "baselines"}:
        return get_baseline_controls()
    if normalized in {"bounded_controls", "bounded"}:
        return get_bounded_controls()
    if normalized in {"controls", "release_controls"}:
        return get_control_activations()
    if normalized in {"full_controls", "all_controls", "appendix_controls"}:
        return get_full_control_activations()
    if normalized in {"standard", "paper", "release", "canonical"}:
        return get_standard_experimental_set()
    if normalized in {"full", "comprehensive", "appendix", "supplement"}:
        return get_full_experimental_set()
    raise ValueError(
        "Unknown activation suite "
        f"{name!r}. Expected one of: core, baseline_controls, bounded_controls, controls, "
        "full_controls, standard, full."
    )


def get_astar_activations() -> Dict[str, nn.Module]:
    """Compatibility helper for the fixed-a* Cap-family suite."""

    return {
        "SoftCap_astar_fixed": _make_cap_variant(
            SoftCap,
            a_value=A_STAR_VALUES["SoftCap"],
            learnable=False,
        ),
        "SwishCap_astar_fixed": _make_cap_variant(
            SwishCap,
            a_value=A_STAR_VALUES["SwishCap"],
            learnable=False,
        ),
        "SparseCap_astar_fixed": _make_cap_variant(
            SparseCap,
            a_value=A_STAR_VALUES["SparseCap"],
            learnable=False,
        ),
    }


def get_extended_astar_activations() -> Dict[str, nn.Module]:
    """Compatibility helper for legacy regime-specific suite names.

    This returns the historical fixed/learnable a-regime variants under both
    canonical names (``SoftCap_*``) and the older artifact aliases
    (``TanhSoftCap_*``, ``SmoothNotchV2_*``, ``QuinticNotch_*``).
    """

    specs = {
        "SoftCap": (SoftCap, A_STAR_VALUES["SoftCap"], "TanhSoftCap"),
        "SwishCap": (SwishCap, A_STAR_VALUES["SwishCap"], "SmoothNotchV2"),
        "SparseCap": (SparseCap, A_STAR_VALUES["SparseCap"], "QuinticNotch"),
    }

    activations: Dict[str, nn.Module] = {}
    for canonical_name, (factory, a_star, legacy_prefix) in specs.items():
        variants = {
            "1_fixed": _make_cap_variant(factory, a_value=1.0, learnable=False),
            "1_learnable": _make_cap_variant(factory, a_value=1.0, learnable=True),
            "astar_fixed": _make_cap_variant(factory, a_value=a_star, learnable=False),
            "astar_learnable": _make_cap_variant(factory, a_value=a_star, learnable=True),
        }
        for suffix, module in variants.items():
            activations[f"{canonical_name}_{suffix}"] = _clone(module)
            activations[f"{legacy_prefix}_{suffix}"] = _clone(module)
    return activations


def validate_controls_present(
    activation_dict: Dict[str, nn.Module],
    *,
    include_bounded: bool = False,
) -> bool:
    required_controls = set(get_control_activations(include_bounded=include_bounded).keys())
    present_controls = set(activation_dict.keys())

    missing = required_controls - present_controls
    if missing:
        print(f"Missing required control activations: {missing}")
        print(
            "Use get_control_activations("
            f"include_bounded={include_bounded}) to ensure the intended controls are present."
        )
        return False

    return True


def ensure_controls_in_plan(
    activations: Dict[str, nn.Module],
    *,
    include_bounded: bool = False,
) -> Dict[str, nn.Module]:
    complete_set = activations.copy()
    controls = get_control_activations(include_bounded=include_bounded)
    for name, control in controls.items():
        if name not in complete_set:
            complete_set[name] = control
    return complete_set


def get_baseline_controls_with_bounded() -> Dict[str, nn.Module]:
    """Backward-compatible alias for the full control set."""

    return get_full_control_activations()


get_thrust_0_activations = get_standard_experimental_set
get_thrust_1_activations = get_standard_experimental_set
get_thrust_2_activations = get_standard_experimental_set
