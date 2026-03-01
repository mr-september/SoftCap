"""
Base Experiment Class

This module defines the base experiment class that all domain-specific experiments
should inherit from. It provides a consistent interface for running experiments
and handling common functionality like logging, checkpointing, and result tracking.
"""

import os
import json
import logging
import time
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from softcap.training import BaseTrainer, GridTrainer
from softcap.training.checkpoints import CheckpointManager


class BaseExperiment(ABC):
    """
    Base class for all experiments in the SoftCap research framework.
    
    This class provides a standard interface for running experiments across
    different domains (CV, NLP, RL, etc.) with consistent logging, checkpointing,
    and result tracking.
    """
    
    def __init__(
        self,
        name: str,
        root_dir: Union[str, Path] = "experiments",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the base experiment.
        
        Args:
            name: Name of the experiment (used for logging and directories)
            root_dir: Root directory for all experiments
            device: Device to run the experiment on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            **kwargs: Additional experiment-specific arguments
        """
        self.name = name
        self.root_dir = Path(root_dir) / name
        self.device = device
        self.seed = seed
        
        # Set random seeds for reproducibility
        self._set_seeds(seed)
        
        # Setup experiment directory structure
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Experiment metadata
        self.metadata = {
            'name': name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device,
            'seed': seed,
            'git_hash': self._get_git_hash(),
            'command': ' '.join(os.sys.argv),
            **kwargs
        }
        
        # Track metrics and results
        self.metrics = {}
        self.results = {}
        
        # Training components (initialized in setup())
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.trainer = None
        
        self.logger.info(f"Initialized experiment: {name}")
        self.logger.info(f"Device: {device}, Seed: {seed}")
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_directories(self) -> None:
        """Create necessary directories for the experiment."""
        self.checkpoint_dir = self.root_dir / 'checkpoints'
        self.log_dir = self.root_dir / 'logs'
        self.results_dir = self.root_dir / 'results'
        self.visualizations_dir = self.root_dir / 'visualizations'
        
        for d in [self.checkpoint_dir, self.log_dir, 
                 self.results_dir, self.visualizations_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure logging for the experiment."""
        # Create logger
        self.logger = logging.getLogger(f"experiment.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to {log_file}")
    
    def _get_git_hash(self) -> str:
        """Get the current git commit hash for reproducibility."""
        try:
            import subprocess
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(__file__))
            ).decode('ascii').strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"
    
    @abstractmethod
    def setup(self, **kwargs) -> None:
        """
        Setup the experiment (load data, create model, etc.).
        
        This method should be implemented by subclasses to handle domain-specific
        setup tasks like loading datasets and creating models.
        """
        pass
    
    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Run the training process for the experiment.
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save_metadata(self) -> None:
        """Save experiment metadata to a JSON file."""
        metadata_file = self.root_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_file}")
    
    def save_results(self) -> None:
        """Save experiment results to a JSON file."""
        results_file = self.results_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'results': self.results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        self.logger.info(f"Saved results to {results_file}")
    
    def save_model(self, filename: str = 'model.pt') -> None:
        """
        Save the model to a file.
        
        Args:
            filename: Name of the file to save the model to
        """
        if self.model is None:
            self.logger.warning("No model to save")
            return
        
        model_path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved model to {model_path}")
    
    def load_model(self, filename: str = 'model.pt') -> None:
        """
        Load a model from a file.
        
        Args:
            filename: Name of the file to load the model from
        """
        if self.model is None:
            self.logger.error("Cannot load model: model not initialized")
            return
        
        model_path = self.checkpoint_dir / filename
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.logger.info(f"Loaded model from {model_path}")
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the complete experiment (setup, train, evaluate).
        
        Returns:
            Dictionary containing all results and metrics
        """
        self.logger.info(f"Starting experiment: {self.name}")
        start_time = time.time()
        
        try:
            # Setup experiment
            self.logger.info("Setting up experiment...")
            self.setup(**kwargs)
            
            # Save metadata
            self.save_metadata()
            
            # Run training
            self.logger.info("Starting training...")
            train_results = self.train(**kwargs)
            self.metrics.update(train_results)
            
            # Evaluate on test set
            self.logger.info("Running evaluation...")
            eval_results = self.evaluate(**kwargs)
            self.metrics.update(eval_results)
            
            # Save final results
            self.save_results()
            
            # Calculate total runtime
            runtime = time.time() - start_time
            self.logger.info(f"Experiment completed in {runtime:.2f} seconds")
            
            return {
                'metrics': self.metrics,
                'results': self.results,
                'runtime': runtime
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class DomainExperiment(BaseExperiment):
    """
    Base class for domain-specific experiments.
    
    This class extends BaseExperiment with additional functionality specific to
    different domains (CV, NLP, RL, etc.).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract domain from module path, handle __main__ case
        module_parts = self.__class__.__module__.split('.')
        if len(module_parts) >= 2:
            self.domain = module_parts[-2]
        else:
            self.domain = 'default'  # Fallback when run as __main__
        
        # Domain-specific directories
        self.domain_dir = self.root_dir / self.domain
        self.domain_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def create_model(self, **kwargs) -> torch.nn.Module:
        """
        Create the model for this experiment.
        
        Returns:
            A PyTorch model
        """
        pass
    
    @abstractmethod
    def create_dataloaders(self, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        pass
    
    def setup(self, **kwargs) -> None:
        """Setup the experiment (load data, create model, etc.)."""
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders(**kwargs)
        
        # Create model
        self.model = self.create_model(**kwargs).to(self.device)
        
        # Log model architecture
        self.logger.info(f"Model architecture:\n{self.model}")
        
        # Log dataset sizes
        if self.train_loader:
            self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        if self.test_loader:
            self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
