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

"""Standard activation sets used across SoftCap experiments.

Canonical policy (v1 publication): expose only the three parametric SoftCap
variants plus four control activations.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .activations import (
    GELUWithMetrics,
    ReLUWithMetrics,
    SiLUWithMetrics,
    TanhWithMetrics,
)


def get_control_activations() -> Dict[str, nn.Module]:
    return {
        "ReLU": ReLUWithMetrics(),
        "Tanh": TanhWithMetrics(),
        "GELU": GELUWithMetrics(),
        "SiLU": SiLUWithMetrics(),
    }


def get_standard_experimental_set() -> Dict[str, nn.Module]:
    from .activations import (
        SoftCap,
        SwishCap,
        SparseCap,
    )

    activations: Dict[str, nn.Module] = {
        "SoftCap": SoftCap(),
        "SwishCap": SwishCap(),
        "SparseCap": SparseCap(),
    }
    activations.update(get_control_activations())
    return activations


def get_astar_activations() -> Dict[str, nn.Module]:
    """Deprecated legacy API kept for compatibility.

    Returns the canonical set; legacy cubic/quartic a* variants were purged.
    """

    return get_standard_experimental_set()


def get_extended_astar_activations() -> Dict[str, nn.Module]:
    """Deprecated legacy API kept for compatibility.

    Returns the canonical set; legacy V1/cubic/quartic families were purged.
    """

    return get_standard_experimental_set()


def validate_controls_present(activation_dict: Dict[str, nn.Module]) -> bool:
    required_controls = {"ReLU", "Tanh", "GELU", "SiLU"}
    present_controls = set(activation_dict.keys())

    missing = required_controls - present_controls
    if missing:
        print(f"Missing required control activations: {missing}")
        print("Use get_control_activations() to ensure all controls are present.")
        return False

    return True


def ensure_controls_in_plan(activations: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
    complete_set = activations.copy()
    controls = get_control_activations()
    for name, control in controls.items():
        if name not in complete_set:
            complete_set[name] = control
    return complete_set


def get_baseline_controls() -> Dict[str, nn.Module]:
    return get_control_activations()


get_thrust_0_activations = get_standard_experimental_set
get_thrust_1_activations = get_standard_experimental_set
get_thrust_2_activations = get_standard_experimental_set