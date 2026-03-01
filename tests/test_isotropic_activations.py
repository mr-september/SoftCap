#!/usr/bin/env python3
"""
Tests for isotropic activation implementations from Bird (2024).

Tests:
1. Radial symmetry property (same magnitude -> same output magnitude)
2. Direction preservation (output direction matches input direction)
3. Batch processing (correct shapes)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
from softcap.isotropic_activations import IsotropicTanh, IsotropicLeakyReLU, IsotropicReLU


_ACTIVATIONS = [
    (IsotropicTanh(), "IsotropicTanh"),
    (IsotropicLeakyReLU(), "IsotropicLeakyReLU"),
    (IsotropicReLU(), "IsotropicReLU"),
]


@pytest.fixture(params=_ACTIVATIONS, ids=[n for _, n in _ACTIVATIONS])
def activation_and_name(request):
    """Parametrize over all isotropic activations."""
    return request.param


def test_radial_symmetry(activation_and_name):
    """Vectors of same magnitude must produce same output magnitude."""
    activation, name = activation_and_name

    x1 = torch.tensor([[1.0, 0.0]])
    x2 = torch.tensor([[0.0, 1.0]])
    x3 = torch.tensor([[0.707, 0.707]])

    mag1 = torch.norm(activation(x1))
    mag2 = torch.norm(activation(x2))
    mag3 = torch.norm(activation(x3))

    max_diff = max(abs(mag1 - mag2), abs(mag1 - mag3), abs(mag2 - mag3))
    assert max_diff < 1e-3, f"{name}: radial symmetry violated (max_diff={max_diff:.6f})"


def test_direction_preservation(activation_and_name):
    """Output direction must match input direction."""
    activation, name = activation_and_name

    x = torch.tensor([[3.0, 4.0]])
    y = activation(x)

    x_unit = x / torch.norm(x)
    y_unit = y / torch.norm(y)

    cos_sim = torch.sum(x_unit * y_unit).item()
    assert cos_sim > 0.999, f"{name}: direction not preserved (cos_sim={cos_sim:.6f})"


def test_batch_processing(activation_and_name):
    """Batch processing must preserve shapes."""
    activation, name = activation_and_name

    batch = torch.randn(32, 10)
    output = activation(batch)
    assert output.shape == batch.shape, f"{name}: shape mismatch {output.shape} != {batch.shape}"
