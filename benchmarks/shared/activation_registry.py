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

"""
Activation registry for scaling benchmarks.

Provides a single function to construct the full set of activations
(Definitive SoftCap variants + controls) for the scaling benchmarks,
using the recommended init and a-values from the research.
"""

import copy
from typing import Dict, Tuple
import torch.nn as nn

# Import from the softcap library
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from softcap.activations import (
    SoftCap,
    SwishCap,
    SparseCap,
    ReLUWithMetrics,
    GELUWithMetrics,
    SiLUWithMetrics,
)
from softcap.initialization import get_recommended_init


# a* values from variance-map analysis (control_activations.py)
A_STAR = {
    "SoftCap": 2.89,
    "SwishCap": 2.434,
    "SparseCap": 2.14,
}


def _make_activation(cls, a_init: float, learnable: bool = True) -> nn.Module:
    """Construct a parametric activation, optionally freezing `a`."""
    act = cls(a_init=a_init)
    if not learnable:
        act.a.requires_grad = False
    return act


def get_vision_activations() -> Dict[str, Tuple[nn.Module, str]]:
    """Return activations for the vision benchmark (ResNet-18 / CIFAR-100).

    For vision we train from scratch, so init matters. The dict maps
    ``name -> (activation_instance, recommended_init_method)``.

    Primary conditions (for full benchmark):
        - Controls: ReLU, GELU, SiLU
        - SoftCap (a=1, learnable)
        - SwishCap (a=1, learnable)
        - SparseCap (a=1, learnable)
    """
    return {
        # Controls
        "ReLU":      (ReLUWithMetrics(),  get_recommended_init("ReLU")),
        "GELU":      (GELUWithMetrics(),  get_recommended_init("GELU")),
        "SiLU":      (SiLUWithMetrics(),  get_recommended_init("SiLU")),
        # Definitive SoftCap variants
        "SoftCap":   (_make_activation(SoftCap, 1.0),
                      get_recommended_init("SoftCap")),
        "SwishCap":  (_make_activation(SwishCap, 1.0),
                      get_recommended_init("SwishCap")),
        "SparseCap": (_make_activation(SparseCap, 1.0),
                      get_recommended_init("SparseCap")),
    }


def get_vision_minisweep_activations() -> Dict[str, Tuple[nn.Module, str]]:
    """Extra conditions for a quick mini-sweep (a=a* variants).

    Run these AFTER verifying baselines pass. One seed, 50 epochs.
    """
    return {
        "SoftCap_astar":   (_make_activation(SoftCap, A_STAR["SoftCap"]),
                            "kaiming"),
        "SwishCap_astar":  (_make_activation(SwishCap, A_STAR["SwishCap"]),
                            "kaiming"),
        "SparseCap_astar": (_make_activation(SparseCap, A_STAR["SparseCap"]),
                            "orthogonal"),
    }


def get_nlp_activations() -> Dict[str, nn.Module]:
    """Return activations for the NLP benchmark (DistilBERT / GLUE).

    For NLP we fine-tune from pretrained weights, so init of the base
    model doesn't change — only the activation function in the FFN
    layers is swapped. No separate init method needed.

    Primary conditions:
        - GELU (default baseline, since DistilBERT was trained with GELU)
        - SoftCap (a=1, learnable)
        - SwishCap (a=1, learnable)
        - SparseCap (a=1, learnable)
    """
    return {
        "GELU":      GELUWithMetrics(),
        "SoftCap":   _make_activation(SoftCap, 1.0),
        "SwishCap":  _make_activation(SwishCap, 1.0),
        "SparseCap": _make_activation(SparseCap, 1.0),
    }


def get_nlp_minisweep_activations() -> Dict[str, nn.Module]:
    """Extra a=a* conditions for NLP mini-sweep (SST-2 only, 1 seed)."""
    return {
        "SoftCap_astar":   _make_activation(SoftCap, A_STAR["SoftCap"]),
        "SwishCap_astar":  _make_activation(SwishCap, A_STAR["SwishCap"]),
        "SparseCap_astar": _make_activation(SparseCap, A_STAR["SparseCap"]),
    }


def clone_activation(act: nn.Module) -> nn.Module:
    """Deep-copy an activation so each layer gets independent parameters."""
    return copy.deepcopy(act)
