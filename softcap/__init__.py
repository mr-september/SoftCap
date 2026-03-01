"""SoftCap public package API."""

from .core import SoftCapAnalysisEngine, run_softcap_analysis
from .activations import (
    SoftCap,
    SwishCap,
    SparseCap,
    ReLUWithMetrics,
    TanhWithMetrics,
    GELUWithMetrics,
    SiLUWithMetrics,
    get_default_activations,
    get_baseline_activations,
    get_modern_activations,
    analyze_activation_properties,
    compare_activation_functions,
)
from .models import (
    SimpleMLP,
    DeepMLP,
    ConvNet,
    create_model,
    get_default_architectures,
    get_model_for_analysis,
    SimpleClassifier,
)
from .metrics import (
    IsotropyAnalyzer,
    SparsityAnalyzer,
    GradientHealthAnalyzer,
    InitializationAnalyzer,
    NumericalStabilityAnalyzer,
    comprehensive_analysis,
    run_comprehensive_metrics_analysis,
)
from .control_activations import (
    get_control_activations,
    get_standard_experimental_set,
    ensure_controls_in_plan,
    validate_controls_present,
)
from .isotropic_activations import (
    make_isotropic,
    IsotropicTanh,
    IsotropicLeakyReLU,
    IsotropicReLU,
    IsotropicSoftCap,
    IsotropicSwishCap,
    IsotropicSparseCap,
    get_isotropic_activations,
)
from .synthetic_benchmarks import (
    run_synthetic_benchmarks,
    SyntheticBenchmarks,
    DecisionBoundaryVisualizer,
    SyntheticDataset,
    SimpleClassifier,
)
from .initialization import (
    kaiming_softcap_normal_,
    kaiming_softcap_uniform_,
    init_softcap_model,
    calculate_softcap_gain,
    derive_optimal_a_for_variance_preservation,
    get_recommended_init,
    apply_initialization,
)

__version__ = "1.0.0"
__author__ = "Larry Cai, Jie Tang"
__description__ = "Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity"

analyze = run_softcap_analysis
SoftCapEngine = SoftCapAnalysisEngine


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
    # Control activation utilities
    "get_control_activations",
    "get_standard_experimental_set",
    "ensure_controls_in_plan",
    "validate_controls_present",
    # Isotropic wrappers (canonical names)
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
    # Initialization
    "get_recommended_init",
    "apply_initialization",
    # Compatibility stubs
    "get_all_activations",
    "load_mnist_data",
    "get_experiment_config",
]