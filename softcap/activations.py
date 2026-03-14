"""SoftCap activation functions.

Canonical public names (v1 publication):
- SoftCap   — parametric tanh-based half-rectifier
- SwishCap  — C1 smooth notch + tanh positive branch
- SparseCap — C2 hard-zero quintic notch + tanh positive branch

Legacy class names (ParametricTanhSoftCap, ParametricSmoothNotchTanhSoftCapV2,
ParametricQuinticNotchTanhSoftCap) are kept as deprecated aliases that emit
DeprecationWarning on instantiation.  All other legacy families (Linear,
SmoothNotch V1, Cubic, Quartic) were removed.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseActivation(nn.Module):
    """Base class with lightweight activation telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self.monitoring = True
        self._setup_monitoring()

    def _setup_monitoring(self) -> None:
        self.register_buffer("input_mean", torch.tensor(0.0))
        self.register_buffer("input_std", torch.tensor(0.0))
        self.register_buffer("output_mean", torch.tensor(0.0))
        self.register_buffer("output_std", torch.tensor(0.0))
        self.register_buffer("gradient_mean", torch.tensor(0.0))
        self.register_buffer("gradient_std", torch.tensor(0.0))
        self.register_buffer("zero_ratio", torch.tensor(0.0))
        self.register_buffer("n_forwards", torch.tensor(0))
        self.register_buffer("n_backwards", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.monitoring:
            with torch.no_grad():
                self.input_mean = 0.9 * self.input_mean + 0.1 * x.mean()
                self.input_std = 0.9 * self.input_std + 0.1 * x.std()
                self.n_forwards += 1

        output = self.activation_function(x)

        if self.monitoring:
            with torch.no_grad():
                self.output_mean = 0.9 * self.output_mean + 0.1 * output.mean()
                self.output_std = 0.9 * self.output_std + 0.1 * output.std()
                self.zero_ratio = 0.9 * self.zero_ratio + 0.1 * (output == 0).float().mean()

        return output

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, float]:
        return {
            "input_mean": self.input_mean.item(),
            "input_std": self.input_std.item(),
            "output_mean": self.output_mean.item(),
            "output_std": self.output_std.item(),
            "gradient_mean": self.gradient_mean.item(),
            "gradient_std": self.gradient_std.item(),
            "zero_ratio": self.zero_ratio.item(),
            "n_forwards": self.n_forwards.item(),
            "n_backwards": self.n_backwards.item(),
        }

    def reset_metrics(self) -> None:
        self.input_mean.zero_()
        self.input_std.zero_()
        self.output_mean.zero_()
        self.output_std.zero_()
        self.gradient_mean.zero_()
        self.gradient_std.zero_()
        self.zero_ratio.zero_()
        self.n_forwards.zero_()
        self.n_backwards.zero_()


class SoftCap(BaseActivation):
    """SoftCap: ``f(x)=0`` for ``x<=0`` else ``a*tanh(x)``."""

    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.name = "SoftCap"
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def _safe_a(self) -> torch.Tensor:
        return torch.clamp(self.a, min=1e-3)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        output = torch.zeros_like(x)
        pos = x > 0
        output[pos] = a * torch.tanh(x[pos])
        return output

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        derivative = torch.zeros_like(x)
        pos = x > 0
        t = torch.tanh(x[pos])
        derivative[pos] = a * (1 - t * t)
        return derivative


class SwishCap(BaseActivation):
    """SwishCap: C1 smooth notch + tanh positive branch scaled by ``a``."""

    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.name = "SwishCap"
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def _safe_a(self) -> torch.Tensor:
        return torch.clamp(self.a, min=1e-3)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        s = torch.sigmoid(a * x)
        neg = 2 * a * x * s
        pos = a * torch.tanh(x)
        return torch.where(x <= 0, neg, pos)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        s = torch.sigmoid(a * x)
        neg_grad = 2 * a * s * (1 + a * x * (1 - s))
        t = torch.tanh(x)
        pos_grad = a * (1 - t * t)
        return torch.where(x <= 0, neg_grad, pos_grad)


class SparseCap(BaseActivation):
    """SparseCap: minimum-degree C2 hard-zero quintic notch with parametric scale ``a``."""

    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.name = "SparseCap"
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def _safe_a(self) -> torch.Tensor:
        return torch.clamp(self.a, min=1e-3)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        notch = (x > -a) & (x <= 0.0)
        pos = x > 0.0

        output = torch.zeros_like(x)
        x_q = x[notch]
        x_plus_a = x_q + a
        output[notch] = x_q * (x_plus_a ** 3) * (a - 3.0 * x_q) / (a ** 3)
        output[pos] = a * torch.tanh(x[pos])
        return output

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        a = self._safe_a()
        notch = (x > -a) & (x <= 0.0)
        pos = x > 0.0

        derivative = torch.zeros_like(x)
        x_q = x[notch]
        x_plus_a = x_q + a
        derivative[notch] = (x_plus_a ** 2) * (a * a - 2.0 * a * x_q - 15.0 * x_q ** 2) / (a ** 3)
        derivative[pos] = a * (1.0 - torch.tanh(x[pos]) ** 2)
        return derivative


class ReLUWithMetrics(BaseActivation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ReLU"

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()


class TanhWithMetrics(BaseActivation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Tanh"

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(x)
        return 1 - t * t


class GELUWithMetrics(BaseActivation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "GELU"

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # Match PyTorch's GELU implementations.
        approximate = getattr(self, "approximate", "none")
        if approximate == "tanh":
            # gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
            c = math.sqrt(2.0 / math.pi)
            u = c * (x + 0.044715 * x**3)
            t = torch.tanh(u)
            du_dx = c * (1.0 + 3.0 * 0.044715 * x**2)
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx

        # Exact GELU: gelu(x) = x * Phi(x), Phi = Normal CDF.
        # d/dx gelu(x) = Phi(x) + x * phi(x), phi = Normal PDF.
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        phi = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
        return Phi + x * phi


class SiLUWithMetrics(BaseActivation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "SiLU"

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(x)
        return s * (1 + x * (1 - s))


class ReLU6WithMetrics(BaseActivation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ReLU6"

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return ((x > 0) & (x < 6)).float()


class HardTanhWithMetrics(BaseActivation):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.name = "HardTanh"
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return ((x > self.min_val) & (x < self.max_val)).float()


def get_default_activations() -> Dict[str, nn.Module]:
    """Return the release-facing default activation set."""

    return {
        "SoftCap": SoftCap(),
        "SwishCap": SwishCap(),
        "SparseCap": SparseCap(),
        "ReLU": ReLUWithMetrics(),
        "Tanh": TanhWithMetrics(),
        "GELU": GELUWithMetrics(),
        "SiLU": SiLUWithMetrics(),
    }


def get_full_default_activations() -> Dict[str, nn.Module]:
    """Return the full built-in activation catalog, including appendix controls."""

    activations = get_default_activations()
    activations.update(
        {
            "ReLU6": ReLU6WithMetrics(),
            "HardTanh": HardTanhWithMetrics(),
        }
    )
    return activations


def get_baseline_activations() -> Dict[str, nn.Module]:
    return {
        "ReLU": ReLUWithMetrics(),
        "Tanh": TanhWithMetrics(),
        "GELU": GELUWithMetrics(),
        "SiLU": SiLUWithMetrics(),
    }


def get_modern_activations() -> Dict[str, nn.Module]:
    return {
        "GELU": GELUWithMetrics(),
        "SiLU": SiLUWithMetrics(),
    }


def analyze_activation_properties(
    activation: nn.Module,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    num_points: int = 1000,
) -> Dict[str, Any]:
    x = torch.linspace(x_range[0], x_range[1], num_points)
    with torch.no_grad():
        y = activation(x)
        dy_dx = activation.derivative(x) if hasattr(activation, "derivative") else None

    properties: Dict[str, Any] = {
        "name": getattr(activation, "name", type(activation).__name__),
        "output_range": (y.min().item(), y.max().item()),
        "output_mean": y.mean().item(),
        "output_std": y.std().item(),
        "sparsity_ratio": (y == 0).float().mean().item(),
        "monotonic": torch.all(torch.diff(y) >= 0).item(),
        "bounded": torch.isfinite(y).all().item(),
        "zero_centered": abs(y.mean().item()) < 0.1,
        "symmetric": torch.allclose(y, -torch.flip(y, [0]), atol=0.1),
    }

    if dy_dx is not None:
        properties.update(
            {
                "gradient_range": (dy_dx.min().item(), dy_dx.max().item()),
                "gradient_mean": dy_dx.mean().item(),
                "gradient_std": dy_dx.std().item(),
                "vanishing_gradient_risk": (dy_dx < 0.01).float().mean().item(),
                "exploding_gradient_risk": (dy_dx > 10).float().mean().item(),
            }
        )
    return properties


def compare_activation_functions(
    activations: Dict[str, nn.Module],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    num_points: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    return {
        name: analyze_activation_properties(act, x_range=x_range, num_points=num_points)
        for name, act in activations.items()
    }


__all__ = [
    "BaseActivation",
    # Canonical public names
    "SoftCap",
    "SwishCap",
    "SparseCap",
    # Controls
    "ReLUWithMetrics",
    "ReLU6WithMetrics",
    "TanhWithMetrics",
    "HardTanhWithMetrics",
    "GELUWithMetrics",
    "SiLUWithMetrics",
    # Helpers
    "get_default_activations",
    "get_full_default_activations",
    "get_baseline_activations",
    "get_modern_activations",
    "analyze_activation_properties",
    "compare_activation_functions",
    # Deprecated aliases (kept for backward compatibility)
    "ParametricTanhSoftCap",
    "ParametricSmoothNotchTanhSoftCapV2",
    "ParametricQuinticNotchTanhSoftCap",
]


# ---------------------------------------------------------------------------
# Deprecated aliases — emit DeprecationWarning on instantiation
# ---------------------------------------------------------------------------

class ParametricTanhSoftCap(SoftCap):
    """Deprecated alias for :class:`SoftCap`. Use ``SoftCap`` instead."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "ParametricTanhSoftCap is deprecated and will be removed in a future release. "
            "Use SoftCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ParametricSmoothNotchTanhSoftCapV2(SwishCap):
    """Deprecated alias for :class:`SwishCap`. Use ``SwishCap`` instead."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "ParametricSmoothNotchTanhSoftCapV2 is deprecated and will be removed in a future release. "
            "Use SwishCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ParametricQuinticNotchTanhSoftCap(SparseCap):
    """Deprecated alias for :class:`SparseCap`. Use ``SparseCap`` instead."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "ParametricQuinticNotchTanhSoftCap is deprecated and will be removed in a future release. "
            "Use SparseCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
