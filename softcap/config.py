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
SoftCap Configuration Module

Default configurations and hyperparameters for experiments.
"""

from typing import Dict, Any, List


def get_default_hyperparameters() -> Dict[str, Any]:
    """Get default hyperparameter grid for comprehensive evaluation.
    
    Learning rates include aggressive ranges to test SoftCap's bounded properties.
    Multiple seeds ensure statistical robustness.
    """
    return {
        'learning_rates': [1.0, 0.1, 0.01, 0.001],  # Aggressive range for SoftCap testing
        'optimizers': ['Adam', 'SGD'],
        'batch_sizes': [64, 128, 256],  # Larger batch sizes to reduce spikiness
        'epochs': 10,  # Extended training for convergence
        'early_stopping_patience': 5,
        'gradient_clip_value': 1.0,
        'weight_decay': 1e-4,
        'initialization_methods': ['default', 'xavier', 'kaiming'],
        'seeds': [42, 123, 456, 789, 999],  # 5 seeds for statistical robustness
        'scheduler': 'ReduceLROnPlateau',  # Add LR scheduling
        'scheduler_patience': 3,
        'scheduler_factor': 0.5
    }


def get_training_config() -> Dict[str, Any]:
    """Get training configuration with comprehensive logging and checkpointing."""
    return {
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
        'early_stopping_patience': 5,
        'gradient_clip_value': 1.0,
        'enable_profiler': False,
        'detect_anomaly': False,
        # Enhanced logging
        'log_interval': 100,  # Log every 100 batches
        'detailed_metrics': True,
        'save_training_curves': True,
        # Smart checkpointing
        'checkpoint_strategy': 'smart',  # best, last, milestone
        'checkpoint_every_n_epochs': 5,
        'keep_best_n_checkpoints': 3,
        'keep_last_n_checkpoints': 2
    }


def get_analysis_config() -> Dict[str, Any]:
    """
    Get default analysis configuration.
    
    Returns:
        Analysis configuration dictionary
    """
    return {
        'isotropy_analysis': True,
        'sparsity_analysis': True,
        'gradient_health_analysis': True,
        'initialization_analysis': True,
        'deep_network_analysis': True,
        'visualization': True,
        'statistical_testing': True,
        'export_csv': True,
        'export_json': True,
        'generate_report': True
    }


def get_experiment_config() -> Dict[str, Any]:
    """
    Get complete experiment configuration.
    
    Returns:
        Complete experiment configuration
    """
    return {
        'hyperparameters': get_default_hyperparameters(),
        'training': get_training_config(),
        'analysis': get_analysis_config(),
        'output_settings': {
            'save_models': True,
            'save_plots': True,
            'plot_dpi': 300,
            'figure_format': 'png'
        }
    }
