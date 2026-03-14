"""SoftCap public package API.

The top-level package stays intentionally thin and resolves most exports
lazy-on-access so importing ``softcap`` does not eagerly pull in the full
research stack.
"""

from __future__ import annotations

from importlib import import_module


__version__ = "1.0.0"
__author__ = "Larry Cai, Jie Tang"
__description__ = "Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity"


_LAZY_EXPORTS = {
    # Core engine
    "SoftCapAnalysisEngine": ("softcap.core", "SoftCapAnalysisEngine"),
    "run_softcap_analysis": ("softcap.core", "run_softcap_analysis"),
    "analyze": ("softcap.core", "run_softcap_analysis"),
    "SoftCapEngine": ("softcap.core", "SoftCapAnalysisEngine"),
    # Canonical activation names
    "SoftCap": ("softcap.activations", "SoftCap"),
    "SwishCap": ("softcap.activations", "SwishCap"),
    "SparseCap": ("softcap.activations", "SparseCap"),
    # Controls
    "ReLUWithMetrics": ("softcap.activations", "ReLUWithMetrics"),
    "TanhWithMetrics": ("softcap.activations", "TanhWithMetrics"),
    "GELUWithMetrics": ("softcap.activations", "GELUWithMetrics"),
    "SiLUWithMetrics": ("softcap.activations", "SiLUWithMetrics"),
    # Activation helpers
    "get_default_activations": ("softcap.activations", "get_default_activations"),
    "get_baseline_activations": ("softcap.activations", "get_baseline_activations"),
    "get_modern_activations": ("softcap.activations", "get_modern_activations"),
    "analyze_activation_properties": ("softcap.activations", "analyze_activation_properties"),
    "compare_activation_functions": ("softcap.activations", "compare_activation_functions"),
    # Models
    "SimpleMLP": ("softcap.models", "SimpleMLP"),
    "DeepMLP": ("softcap.models", "DeepMLP"),
    "ConvNet": ("softcap.models", "ConvNet"),
    "create_model": ("softcap.models", "create_model"),
    "get_default_architectures": ("softcap.models", "get_default_architectures"),
    "get_model_for_analysis": ("softcap.models", "get_model_for_analysis"),
    "SimpleClassifier": ("softcap.models", "SimpleClassifier"),
    # Control activation utilities
    "get_core_activations": ("softcap.control_activations", "get_core_activations"),
    "get_baseline_controls": ("softcap.control_activations", "get_baseline_controls"),
    "get_bounded_controls": ("softcap.control_activations", "get_bounded_controls"),
    "get_full_control_activations": ("softcap.control_activations", "get_full_control_activations"),
    "get_control_activations": ("softcap.control_activations", "get_control_activations"),
    "get_named_activation_suite": ("softcap.control_activations", "get_named_activation_suite"),
    "get_full_experimental_set": ("softcap.control_activations", "get_full_experimental_set"),
    "get_standard_experimental_set": ("softcap.control_activations", "get_standard_experimental_set"),
    "ensure_controls_in_plan": ("softcap.control_activations", "ensure_controls_in_plan"),
    "validate_controls_present": ("softcap.control_activations", "validate_controls_present"),
    # Isotropic wrappers
    "make_isotropic": ("softcap.isotropic_activations", "make_isotropic"),
    "IsotropicTanh": ("softcap.isotropic_activations", "IsotropicTanh"),
    "IsotropicLeakyReLU": ("softcap.isotropic_activations", "IsotropicLeakyReLU"),
    "IsotropicReLU": ("softcap.isotropic_activations", "IsotropicReLU"),
    "IsotropicSoftCap": ("softcap.isotropic_activations", "IsotropicSoftCap"),
    "IsotropicSwishCap": ("softcap.isotropic_activations", "IsotropicSwishCap"),
    "IsotropicSparseCap": ("softcap.isotropic_activations", "IsotropicSparseCap"),
    "get_isotropic_activations": ("softcap.isotropic_activations", "get_isotropic_activations"),
    # Metrics
    "IsotropyAnalyzer": ("softcap.metrics", "IsotropyAnalyzer"),
    "SparsityAnalyzer": ("softcap.metrics", "SparsityAnalyzer"),
    "GradientHealthAnalyzer": ("softcap.metrics", "GradientHealthAnalyzer"),
    "InitializationAnalyzer": ("softcap.metrics", "InitializationAnalyzer"),
    "NumericalStabilityAnalyzer": ("softcap.metrics", "NumericalStabilityAnalyzer"),
    "comprehensive_analysis": ("softcap.metrics", "comprehensive_analysis"),
    "run_comprehensive_metrics_analysis": ("softcap.metrics", "run_comprehensive_metrics_analysis"),
    # Synthetic benchmarks
    "run_synthetic_benchmarks": ("softcap.synthetic_benchmarks", "run_synthetic_benchmarks"),
    "SyntheticBenchmarks": ("softcap.synthetic_benchmarks", "SyntheticBenchmarks"),
    "DecisionBoundaryVisualizer": ("softcap.synthetic_benchmarks", "DecisionBoundaryVisualizer"),
    "SyntheticDataset": ("softcap.synthetic_benchmarks", "SyntheticDataset"),
    # Initialization
    "kaiming_softcap_normal_": ("softcap.initialization", "kaiming_softcap_normal_"),
    "kaiming_softcap_uniform_": ("softcap.initialization", "kaiming_softcap_uniform_"),
    "init_softcap_model": ("softcap.initialization", "init_softcap_model"),
    "calculate_softcap_gain": ("softcap.initialization", "calculate_softcap_gain"),
    "derive_optimal_a_for_variance_preservation": ("softcap.initialization", "derive_optimal_a_for_variance_preservation"),
    "get_recommended_init": ("softcap.initialization", "get_recommended_init"),
    "apply_initialization": ("softcap.initialization", "apply_initialization"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def get_all_activations(*args, **kwargs):
    from .compatibility import get_all_activations as _impl

    return _impl(*args, **kwargs)


def load_mnist_data(*args, **kwargs):
    from .compatibility import load_mnist_data as _impl

    return _impl(*args, **kwargs)


def get_experiment_config(*args, **kwargs):
    from .compatibility import get_experiment_config as _impl

    return _impl(*args, **kwargs)


__all__ = [
    # Core engine
    "SoftCapAnalysisEngine",
    "run_softcap_analysis",
    "analyze",
    "SoftCapEngine",
    # Canonical activation names (v1 public)
    "SoftCap",
    "SwishCap",
    "SparseCap",
    # Controls
    "ReLUWithMetrics",
    "TanhWithMetrics",
    "GELUWithMetrics",
    "SiLUWithMetrics",
    # Activation helpers
    "get_default_activations",
    "get_baseline_activations",
    "get_modern_activations",
    "analyze_activation_properties",
    "compare_activation_functions",
    # Models
    "SimpleMLP",
    "DeepMLP",
    "ConvNet",
    "create_model",
    "get_default_architectures",
    "get_model_for_analysis",
    "SimpleClassifier",
    # Control activation utilities
    "get_core_activations",
    "get_baseline_controls",
    "get_bounded_controls",
    "get_full_control_activations",
    "get_control_activations",
    "get_named_activation_suite",
    "get_full_experimental_set",
    "get_standard_experimental_set",
    "ensure_controls_in_plan",
    "validate_controls_present",
    # Isotropic wrappers
    "make_isotropic",
    "IsotropicTanh",
    "IsotropicLeakyReLU",
    "IsotropicReLU",
    "IsotropicSoftCap",
    "IsotropicSwishCap",
    "IsotropicSparseCap",
    "get_isotropic_activations",
    # Metrics
    "IsotropyAnalyzer",
    "SparsityAnalyzer",
    "GradientHealthAnalyzer",
    "InitializationAnalyzer",
    "NumericalStabilityAnalyzer",
    "comprehensive_analysis",
    "run_comprehensive_metrics_analysis",
    # Synthetic benchmarks
    "run_synthetic_benchmarks",
    "SyntheticBenchmarks",
    "DecisionBoundaryVisualizer",
    "SyntheticDataset",
    # Initialization
    "kaiming_softcap_normal_",
    "kaiming_softcap_uniform_",
    "init_softcap_model",
    "calculate_softcap_gain",
    "derive_optimal_a_for_variance_preservation",
    "get_recommended_init",
    "apply_initialization",
    # Compatibility stubs
    "get_all_activations",
    "load_mnist_data",
    "get_experiment_config",
]
