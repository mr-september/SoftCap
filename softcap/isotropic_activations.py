"""Isotropic activation utilities and optional isotropic wrappers."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_isotropic(
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    epsilon: float = 1e-2,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def isotropic_activation(x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= epsilon
        unit_vector = x / magnitude.clamp(min=epsilon)
        activated_magnitude = activation_fn(magnitude)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)

    return isotropic_activation


class IsotropicTanh(nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.name = "IsotropicTanh"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        return torch.where(identity_mask, x, torch.tanh(magnitude) * unit_vector)


class IsotropicLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01, epsilon: float = 1e-2):
        super().__init__()
        self.negative_slope = negative_slope
        self.epsilon = epsilon
        self.name = f"IsotropicLeakyReLU(alpha={negative_slope})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        activated_magnitude = F.leaky_relu(magnitude, negative_slope=self.negative_slope)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)


class IsotropicReLU(nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.name = "IsotropicReLU"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        activated_magnitude = F.relu(magnitude)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)


class IsotropicSoftCap(nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.name = "IsotropicSoftCap"
        from .activations import SoftCap

        self._base_activation = SoftCap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        activated_magnitude = self._base_activation(magnitude)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)


class IsotropicSwishCap(nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.name = "IsotropicSwishCap"
        from .activations import SwishCap

        self._base_activation = SwishCap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        activated_magnitude = self._base_activation(magnitude)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)


class IsotropicSparseCap(nn.Module):
    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.name = "IsotropicSparseCap"
        from .activations import SparseCap

        self._base_activation = SparseCap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.norm(x, dim=-1, keepdim=True)
        identity_mask = magnitude <= self.epsilon
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        activated_magnitude = self._base_activation(magnitude)
        return torch.where(identity_mask, x, activated_magnitude * unit_vector)


def get_isotropic_activations(include_softcap: bool = False) -> Dict[str, nn.Module]:
    activations: Dict[str, nn.Module] = {
        "IsotropicTanh": IsotropicTanh(),
        "IsotropicLeakyReLU": IsotropicLeakyReLU(),
        "IsotropicReLU": IsotropicReLU(),
    }

    if include_softcap:
        activations.update(
            {
                "IsotropicSoftCap": IsotropicSoftCap(),
                "IsotropicSwishCap": IsotropicSwishCap(),
                "IsotropicSparseCap": IsotropicSparseCap(),
            }
        )

    return activations


__all__ = [
    "make_isotropic",
    "IsotropicTanh",
    "IsotropicLeakyReLU",
    "IsotropicReLU",
    # Canonical public names
    "IsotropicSoftCap",
    "IsotropicSwishCap",
    "IsotropicSparseCap",
    "get_isotropic_activations",
    # Deprecated aliases
    "IsotropicParametricTanhSoftCap",
    "IsotropicParametricSmoothNotchTanhSoftCapV2",
    "IsotropicParametricQuinticNotchTanhSoftCap",
]


# ---------------------------------------------------------------------------
# Deprecated aliases — emit DeprecationWarning on instantiation
# ---------------------------------------------------------------------------

import warnings as _warnings


class IsotropicParametricTanhSoftCap(IsotropicSoftCap):
    """Deprecated alias for :class:`IsotropicSoftCap`."""

    def __init__(self, *args, **kwargs) -> None:
        _warnings.warn(
            "IsotropicParametricTanhSoftCap is deprecated. Use IsotropicSoftCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class IsotropicParametricSmoothNotchTanhSoftCapV2(IsotropicSwishCap):
    """Deprecated alias for :class:`IsotropicSwishCap`."""

    def __init__(self, *args, **kwargs) -> None:
        _warnings.warn(
            "IsotropicParametricSmoothNotchTanhSoftCapV2 is deprecated. Use IsotropicSwishCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class IsotropicParametricQuinticNotchTanhSoftCap(IsotropicSparseCap):
    """Deprecated alias for :class:`IsotropicSparseCap`."""

    def __init__(self, *args, **kwargs) -> None:
        _warnings.warn(
            "IsotropicParametricQuinticNotchTanhSoftCap is deprecated. Use IsotropicSparseCap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)