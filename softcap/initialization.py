"""
SoftCap Initialization Module

Provides specialized weight initialization strategies for SoftCap activation functions,
derived from theoretical analysis of variance propagation similar to Self-Normalizing
Neural Networks (SELU).

Theoretical Background
======================
Standard initialization schemes (Xavier, He/Kaiming) are derived for specific activation
functions (Tanh/Sigmoid, ReLU respectively). SoftCap variants have different statistical
properties:

1. **Half-Sparsity**: Like ReLU, SoftCap outputs 0 for negative inputs, halving the
   expected variance from half the neurons. This suggests He-style (variance = 2/n)
   rather than Xavier-style (variance = 1/n) initialization.

2. **Bounded Positive Side**: Unlike ReLU (unbounded), SoftCap uses tanh(x), which
   saturates. This reduces the variance expansion that He initialization expects.

3. **Notch Variants**: Cubic/Smooth notch variants have a "dip" in the negative region
   that contributes to variance differently than hard zeros.

This module provides:
- `calculate_softcap_gain()`: Compute the appropriate gain for SoftCap variants.
- `kaiming_softcap_normal_()`: In-place normal initialization for SoftCap.
- `kaiming_softcap_uniform_()`: In-place uniform initialization for SoftCap.
- `init_softcap_model()`: Helper to initialize an entire model for SoftCap.

Usage
=====
```python
from softcap.initialization import kaiming_softcap_normal_, init_softcap_model

# Initialize a single layer
kaiming_softcap_normal_(layer.weight, variant='tanh_softcap')

# Initialize an entire model
init_softcap_model(model, variant='cubic_notch', a=1.2)
```

References
==========
- He et al. (2015): "Delving Deep into Rectifiers" (Kaiming Init)
- Klambauer et al. (2017): "Self-Normalizing Neural Networks" (SELU)
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal

# Empirically derived gains for SoftCap variants at a=1.0.
# Measured by scripts/theory/compute_gains_and_propagation.py (500k MC samples).
# Gain := std(f(x)) / std(x) for x ~ N(0,1).
SOFTLU_GAINS = {
    'tanh_softcap': 0.3459,
    'smooth_notch_v2': 0.5267,    # V2 (different negative-side geometry)
    'quintic_notch': 0.4012,
    'relu': 0.7071,               # sqrt(0.5)
    'tanh': 1.0,
    'selu': 1.0,
}


def calculate_softcap_gain(
    variant: str = 'tanh_softcap',
    param_a: float = 1.0
) -> float:
    """
    Calculate the gain (scaling factor) for SoftCap weight initialization.

    The gain compensates for the expected variance change introduced by the
    activation function. For SoftCap:
        - Negative inputs produce 0 (or near-0 for notch variants).
        - Positive inputs are bounded by tanh saturation.

    Args:
        variant: Which SoftCap variant ('tanh_softcap', 'cubic_notch', etc.).
        param_a: The learnable 'a' parameter of parametric variants. Affects scaling.

    Returns:
        The gain factor to multiply the standard deviation by.

    Example:
        >>> gain = calculate_softcap_gain('tanh_softcap', param_a=1.0)
        >>> std = gain / math.sqrt(fan_in)
    """
    base_gain = SOFTLU_GAINS.get(variant, 0.7)

    # Heuristic parametric adjustment. The true gain-vs-a relationship is
    # nonlinear and variant-specific — this approximation can be 40%+ off.
    # For the critical a=a* case, use gain=1.0 directly (since a* is defined
    # as the value where Var(f(x))=1 for x~N(0,1), giving gain=1 by construction).
    adjusted_gain = base_gain * math.sqrt(param_a)

    return adjusted_gain


def kaiming_softcap_normal_(
    tensor: torch.Tensor,
    variant: str = 'tanh_softcap',
    param_a: float = 1.0,
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    gain: Optional[float] = None,
) -> torch.Tensor:
    """
    Fill the input Tensor with values using a Kaiming-style normal distribution,
    adjusted for SoftCap activation functions.

    Args:
        tensor: An n-dimensional torch.Tensor (weight matrix).
        variant: SoftCap variant name for gain calculation (ignored if gain is set).
        param_a: The 'a' parameter value (ignored if gain is set).
        mode: 'fan_in' (default) preserves variance in the forward pass.
        gain: If provided, use this gain directly instead of the heuristic formula.
              Use gain=1.0 when param_a=a* (variance-preserving by construction).

    Returns:
        The input tensor (modified in-place).
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    if gain is None:
        gain = calculate_softcap_gain(variant, param_a)

    std = gain / math.sqrt(fan)

    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_softcap_uniform_(
    tensor: torch.Tensor,
    variant: str = 'tanh_softcap',
    param_a: float = 1.0,
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
) -> torch.Tensor:
    """
    Fill the input Tensor with values using a Kaiming-style uniform distribution,
    adjusted for SoftCap activation functions.

    This is the SoftCap equivalent of `torch.nn.init.kaiming_uniform_`.

    Args:
        tensor: An n-dimensional torch.Tensor (weight matrix).
        variant: SoftCap variant name for gain calculation.
        param_a: The 'a' parameter value of parametric SoftCap.
        mode: 'fan_in' or 'fan_out'.

    Returns:
        The input tensor (modified in-place).
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = calculate_softcap_gain(variant, param_a)
    std = gain / math.sqrt(fan)

    # For uniform distribution, bound = sqrt(3) * std
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def init_softcap_model(
    model: nn.Module,
    variant: str = 'tanh_softcap',
    param_a: float = 1.0,
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    init_type: Literal['normal', 'uniform'] = 'normal',
) -> nn.Module:
    """
    Initialize all Linear and Conv2d layers in a model for SoftCap.

    Biases are initialized to zero by default.

    Args:
        model: The nn.Module to initialize.
        variant: SoftCap variant name.
        param_a: The 'a' parameter for parametric variants.
        mode: 'fan_in' or 'fan_out'.
        init_type: 'normal' (default) or 'uniform'.

    Returns:
        The model (modified in-place).

    Example:
        >>> model = SimpleMLP(hidden_dim=256, num_layers=10)
        >>> init_softcap_model(model, variant='cubic_notch', param_a=1.2)
    """
    init_fn = kaiming_softcap_normal_ if init_type == 'normal' else kaiming_softcap_uniform_

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init_fn(module.weight, variant=variant, param_a=param_a, mode=mode)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


def derive_optimal_a_for_variance_preservation(
    activation_fn,
    target_variance: float = 1.0,
    input_std: float = 1.0,
    n_samples: int = 100000,
    search_range: tuple = (0.5, 3.0),
    tolerance: float = 0.01,
) -> float:
    """
    Numerically derive the optimal parameter 'a' that achieves target output variance.

    This function performs a binary search over 'a' values to find the setting
    where Var(activation(x)) = target_variance for x ~ N(0, input_std^2).

    This is the core numerical verification method for the "Self-Normalizing SoftCap"
    hypothesis. If we find an 'a' such that unit-variance inputs produce unit-variance
    outputs, we have a potential fixed point.

    Args:
        activation_fn: A callable that takes 'a' and returns an activation function.
                       Example: lambda a: SparseCap(a_init=a)
        target_variance: The desired output variance (default: 1.0 for normalization).
        input_std: Standard deviation of input Gaussian (default: 1.0).
        n_samples: Number of samples for Monte Carlo estimation.
        search_range: (min_a, max_a) to search within.
        tolerance: Acceptable deviation from target_variance.

    Returns:
        The optimal 'a' value (or the closest found within the search range).

    Example:
        >>> from softcap.activations import SparseCap
        >>> optimal_a = derive_optimal_a_for_variance_preservation(
        ...     activation_fn=lambda a: SparseCap(a_init=a)
        ... )
        >>> print(f"Optimal a for unit variance: {optimal_a:.4f}")
    """
    low, high = search_range

    # Generate input samples once
    x = torch.randn(n_samples) * input_std

    for _ in range(50):  # Max 50 iterations for binary search
        mid = (low + high) / 2.0
        act = activation_fn(mid)

        with torch.no_grad():
            y = act.activation_function(x)
            output_var = y.var().item()

        if abs(output_var - target_variance) < tolerance:
            return mid
        elif output_var < target_variance:
            low = mid  # Need higher 'a' to increase variance
        else:
            high = mid  # Need lower 'a' to decrease variance

    # Return best effort
    return (low + high) / 2.0


# =============================================================================
# Consolidated Init Dispatch
# =============================================================================

# Recommended initialization per activation, derived from the init-sensitivity
# study (Fashion-MNIST, DeepMLP 15-layer, 5 seeds × 15 epochs).
# Source: mechanistic_interpretability/initialization/initialization_sensitivity/
#
# Key observations:
# - Deep networks (15+ layers) show large init sensitivity (10%+ spread).
# - Shallow networks (ConvNet) show <0.7% spread → init barely matters.
# - The table below uses the DeepMLP recommendations as the conservative
#   default, since they discriminate where it matters most.
_INIT_LOOKUP = {
    # Canonical public names (v1 publication)
    'SoftCap':  'kaiming',
    'SwishCap': 'kaiming',
    'SparseCap': 'orthogonal',
    # Deprecated aliases — kept for robustness against old call sites
    'ParametricTanhSoftCap': 'kaiming',
    'ParametricSmoothNotchTanhSoftCapV2': 'kaiming',  # V2
    'ParametricQuinticNotchTanhSoftCap': 'orthogonal',

    # Controls
    'ReLU': 'xavier',
    'ReLU6': 'xavier',
    'Tanh': 'xavier',
    'HardTanh': 'xavier',
    'GELU': 'xavier',
    'SiLU': 'orthogonal',
}


def get_recommended_init(activation_name: str) -> str:
    """Return the recommended weight initialization for a given activation.

    Consolidates the init-sensitivity study results from both DeepMLP
    (high-sensitivity regime) and ConvNet (low-sensitivity regime).

    Uses exact name match first, then substring fallback for robustness
    against naming variations (e.g. 'SmoothNotchV2_astar_fixed' will
    match the 'smoothnotchv2' substring rule).

    Args:
        activation_name: Activation function name (case-sensitive for
            exact match, case-insensitive for substring fallback).

    Returns:
        One of 'xavier', 'kaiming', 'orthogonal'.
    """
    # Exact match first (covers the canonical names + deprecated variants)
    if activation_name in _INIT_LOOKUP:
        return _INIT_LOOKUP[activation_name]

    # Substring fallback for minor canonical naming variants.
    name = (activation_name or '').lower()

    # Canonical public names (new) and legacy names
    if 'softcap' in name or 'swishcap' in name:
        return 'kaiming'
    if 'sparsecap' in name:
        return 'orthogonal'
    if 'parametrictanhsoftcap' in name or 'parametricsmoothnotchtanhsoftcapv2' in name:
        return 'kaiming'
    if 'parametricquinticnotchtanhsoftcap' in name:
        return 'orthogonal'

    # Controls
    if name in {'relu', 'relu6', 'tanh', 'hardtanh', 'gelu'}:
        return 'xavier'
    if name in {'silu'}:
        return 'orthogonal'

    # Default: kaiming is the safest for SoftCap family
    return 'kaiming'


def apply_initialization(
    model: 'torch.nn.Module',
    init_method: str,
    activation_name: str = None,
) -> None:
    """Apply weight initialization to all Linear and Conv2d layers.

    Supports 'kaiming', 'orthogonal', 'xavier', and 'softcap_optimal'.
    Skips LayerNorm-related and embedding-like parameters.

    Args:
        model: The model to initialize.
        init_method: One of 'kaiming', 'orthogonal', 'xavier', 'softcap_optimal'.
        activation_name: Optional, used only for logging/validation.

    Raises:
        ValueError: If init_method is not recognized.
    """
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Skip LayerNorm-related and embedding-like params
            if 'norm' in name.lower() or 'embed' in name.lower():
                continue

            if init_method == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif init_method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_method == 'softcap_optimal':
                # At a=a*, the gain is exactly 1.0 by construction:
                # a* is defined as the value where Var(f(x))=1 for x~N(0,1),
                # giving std = 1/sqrt(fan_in).
                # Verified by scripts/theory/compute_gains_and_propagation.py.
                kaiming_softcap_normal_(module.weight, gain=1.0)
            else:
                raise ValueError(
                    f"Unknown init method: {init_method!r}. "
                    f"Valid options: 'kaiming', 'orthogonal', 'xavier', 'softcap_optimal'."
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

