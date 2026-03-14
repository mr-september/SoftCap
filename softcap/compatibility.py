"""
Compatibility functions for main.py and other legacy imports.

This module provides wrapper functions to maintain backward compatibility
with the old interface while using the new modular architecture.
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from .activations import get_default_activations, get_full_default_activations
from .data import get_mnist_loaders
from .config import get_experiment_config as _get_experiment_config
from .models import get_model_for_analysis as _get_model_for_analysis


def get_all_activations() -> Dict[str, torch.nn.Module]:
    """
    Get all available activation functions for analysis.
    
    Returns:
        Dictionary mapping activation names to instances
    """
    return get_full_default_activations()


def load_mnist_data(batch_size: int = 64) -> Dict[str, DataLoader]:
    """
    Load MNIST dataset for experiments.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Dictionary containing train and test data loaders
    """
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    # Provide both 'validation' and 'val' keys for compatibility with different callers
    return {
        'train': train_loader,
        'validation': val_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_experiment_config() -> Dict[str, Any]:
    """
    Get default experiment configuration.
    
    Returns:
        Dictionary containing experiment configuration
    """
    return _get_experiment_config()


def get_model_for_analysis(model_type: str, activation_fn: torch.nn.Module) -> torch.nn.Module:
    """
    Create a model for analysis with the specified activation function.
    
    Args:
        model_type: Type of model to create ('standard_mlp', 'deep_mlp', etc.)
        activation_fn: Activation function to use
        
    Returns:
        Neural network model
    """
    return _get_model_for_analysis(model_type, activation_fn)
