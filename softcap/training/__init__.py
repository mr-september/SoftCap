"""
SoftCap Training Module

This module provides comprehensive training utilities for the SoftCap project,
including base training functionality, hyperparameter search, and analysis.
"""

from .base_trainer import BaseTrainer
from .grid_trainer import GridTrainer
from .checkpoints.manager import CheckpointManager
from .checkpoints.strategies import (
    BestNStrategy,
    LastNStrategy,
    SmartStrategy,
    ResearchStrategy
)

__all__ = [
    'BaseTrainer',
    'GridTrainer',
    'CheckpointManager',
    'BestNStrategy',
    'LastNStrategy',
    'SmartStrategy',
    'ResearchStrategy'
]
