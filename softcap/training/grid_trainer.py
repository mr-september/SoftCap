"""
Grid Search Trainer

This module provides functionality for hyperparameter grid search using the BaseTrainer.
"""

import os
import time
import json
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .checkpoints.manager import CheckpointManager


class GridTrainer:
    """
    Performs hyperparameter grid search using BaseTrainer.
    """
    
    def __init__(
        self,
        model_factory: Callable[..., torch.nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[torch.nn.Module] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: Optional[Union[str, Path]] = None,
        checkpoint_strategy: str = 'smart',
        num_seeds: int = 1,
        seed: int = 42,
        **checkpoint_kwargs
    ):
        """
        Initialize the grid trainer.
        
        Args:
            model_factory: Function that creates a model given hyperparameters
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            criterion: Loss function (default: CrossEntropyLoss)
            metrics: Dictionary of metric functions (name -> function)
            device: Device to train on ('cuda' or 'cpu')
            log_dir: Directory to save logs and checkpoints
            checkpoint_strategy: Checkpoint strategy ('best', 'last', 'smart', 'research')
            num_seeds: Number of random seeds to use for each configuration
            seed: Base random seed
            **checkpoint_kwargs: Additional arguments for CheckpointManager
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.metrics = metrics or {}
        self.device = device
        self.num_seeds = num_seeds
        self.base_seed = seed
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path(f"grid_search_{int(time.time())}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_metric = None
        self.best_model_path = None
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.log_dir / 'checkpoints',
            strategy=checkpoint_strategy,
            **checkpoint_kwargs
        )
    
    def _setup_logging(self):
        """Configure logging for the grid trainer."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / 'grid_search.log'
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
        
        self.logger.info(f"Grid search logs will be saved to {log_file}")
    
    def _generate_seeds(self, num_seeds: int) -> List[int]:
        """Generate random seeds for the grid search."""
        if num_seeds == 1:
            return [self.base_seed]
        
        np_rng = np.random.RandomState(self.base_seed)
        return [int(x) for x in np_rng.randint(0, 2**32 - 1, size=num_seeds)]
    
    def _create_optimizer(
        self,
        model: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float,
        **optimizer_kwargs
    ) -> torch.optim.Optimizer:
        """Create an optimizer instance."""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                **optimizer_kwargs
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                **optimizer_kwargs
            )
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                **optimizer_kwargs
            )
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: Optional[str],
        **scheduler_kwargs
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create a learning rate scheduler."""
        if scheduler_name is None:
            return None
        
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name == 'steplr':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                **scheduler_kwargs
            )
        elif scheduler_name == 'multisteplr':
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                **scheduler_kwargs
            )
        elif scheduler_name == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs
            )
        elif scheduler_name == 'cosineannealinglr':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **scheduler_kwargs
            )
        elif scheduler_name == 'cosineannealingwarmrestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _train_single_config(
        self,
        config: Dict[str, Any],
        seed: int,
        run_id: str
    ) -> Dict[str, Any]:
        """Train a single configuration with a given seed."""
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        model = self.model_factory(**config.get('model_args', {})).to(self.device)
        
        # Create optimizer
        optimizer = self._create_optimizer(
            model=model,
            **config.get('optimizer_args', {'optimizer_name': 'adam', 'learning_rate': 0.001})
        )
        
        # Create learning rate scheduler
        lr_scheduler = self._create_lr_scheduler(
            optimizer=optimizer,
            **config.get('scheduler_args', {'scheduler_name': None})
        )
        
        # Create trainer
        trainer = BaseTrainer(
            model=model,
            criterion=self.criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=self.metrics,
            device=self.device,
            log_dir=str(self.log_dir / 'runs' / run_id),
            checkpoint_dir=str(self.checkpoint_manager.checkpoint_dir / run_id)
        )
        
        # Train the model
        result = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            **config.get('training_args', {})
        )
        
        # Add configuration and seed to results
        result['config'] = config
        result['seed'] = seed
        result['run_id'] = run_id
        
        return result
    
    def _generate_run_id(self, config: Dict[str, Any], seed: int) -> str:
        """Generate a unique run ID for a configuration and seed."""
        # Create a hash of the configuration
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{config_hash}_s{seed}"
    
    def search(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = 'val_loss',
        mode: str = 'min',
        num_workers: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter grid search.
        
        Args:
            param_grid: Dictionary of parameter names mapped to lists of values to try
            metric: Metric to optimize
            mode: 'min' or 'max' (minimize or maximize the metric)
            num_workers: Number of parallel workers to use (not implemented)
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary containing search results and best configuration
        """
        self.logger.info("Starting hyperparameter grid search")
        self.logger.info(f"Parameter grid: {param_grid}")
        self.logger.info(f"Optimizing metric: {metric} (mode: {mode})")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Total configurations: {len(param_combinations)}")
        self.logger.info(f"Seeds per configuration: {self.num_seeds}")
        self.logger.info(f"Total runs: {len(param_combinations) * self.num_seeds}")
        
        # Generate seeds for each configuration
        seeds = self._generate_seeds(self.num_seeds)
        
        # Track best configuration
        best_metric = float('inf') if mode == 'min' else -float('inf')
        best_config = None
        best_run_id = None
        
        # Save parameter grid
        with open(self.log_dir / 'param_grid.json', 'w') as f:
            json.dump(param_grid, f, indent=2)
        
        # Results storage
        all_results = []
        
        # Perform grid search
        for i, params in enumerate(param_combinations):
            config = dict(zip(param_names, params))
            
            for seed in seeds:
                run_id = self._generate_run_id(config, seed)
                
                self.logger.info("-" * 80)
                self.logger.info(f"Run {len(all_results) + 1}: {run_id}")
                self.logger.info(f"Configuration: {config}")
                self.logger.info(f"Seed: {seed}")
                
                try:
                    # Train the model
                    start_time = time.time()
                    result = self._train_single_config(
                        config=config,
                        seed=seed,
                        run_id=run_id
                    )
                    
                    # Record training time
                    result['training_time'] = time.time() - start_time
                    
                    # Get the metric value
                    metric_value = result['history'][metric][-1]
                    result['metric'] = metric_value
                    
                    # Update best configuration
                    is_better = (
                        (mode == 'min' and metric_value < best_metric) or
                        (mode == 'max' and metric_value > best_metric)
                    )
                    
                    if is_better:
                        best_metric = metric_value
                        best_config = config
                        best_run_id = run_id
                        
                        self.logger.info(
                            f"New best {metric}: {best_metric:.6f} "
                            f"(config: {best_config}, seed: {seed})"
                        )
                    
                    # Save results
                    all_results.append(result)
                    
                    # Save results to CSV after each run
                    self._save_results_to_csv(all_results)
                    
                except Exception as e:
                    self.logger.error(f"Error in run {run_id}: {str(e)}", exc_info=True)
        
        # Save final results
        self.results = all_results
        self.best_config = best_config
        self.best_metric = best_metric
        self.best_run_id = best_run_id
        
        # Generate summary
        summary = self._generate_summary()
        
        self.logger.info("-" * 80)
        self.logger.info("Grid search completed")
        self.logger.info(f"Best {metric}: {best_metric:.6f}")
        self.logger.info(f"Best configuration: {best_config}")
        self.logger.info(f"Best run ID: {best_run_id}")
        
        return {
            'results': all_results,
            'best_config': best_config,
            'best_metric': best_metric,
            'best_run_id': best_run_id,
            'summary': summary
        }
    
    def _save_results_to_csv(self, results: List[Dict[str, Any]]) -> str:
        """Save results to a CSV file."""
        if not results:
            return ""
        
        # Flatten the results
        flat_results = []
        
        for result in results:
            flat = {
                'run_id': result.get('run_id', ''),
                'seed': result.get('seed', -1),
                'metric': result.get('metric', float('nan')),
                'training_time': result.get('training_time', 0.0)
            }
            
            # Add configuration parameters
            config = result.get('config', {})
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    flat[f'config.{key}'] = value
                else:
                    flat[f'config.{key}'] = str(value)
            
            # Add metrics from history
            history = result.get('history', {})
            for key, values in history.items():
                if isinstance(values, (list, tuple)) and values:
                    flat[f'final_{key}'] = values[-1]
            
            flat_results.append(flat)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(flat_results)
        csv_path = self.log_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the grid search results."""
        if not self.results:
            return {}
        
        # Group results by configuration
        config_results = {}
        
        for result in self.results:
            config_key = json.dumps(result['config'], sort_keys=True)
            if config_key not in config_results:
                config_results[config_key] = []
            config_results[config_key].append(result['metric'])
        
        # Calculate statistics for each configuration
        summary = {}
        
        for config_key, metrics in config_results.items():
            config = json.loads(config_key)
            metrics = np.array(metrics)
            
            summary[config_key] = {
                'config': config,
                'mean': float(np.mean(metrics)),
                'std': float(np.std(metrics)),
                'min': float(np.min(metrics)),
                'max': float(np.max(metrics)),
                'count': len(metrics)
            }
        
        # Sort configurations by mean metric value
        sorted_configs = sorted(
            summary.values(),
            key=lambda x: x['mean'],
            reverse=(self.best_metric is not None and self.best_metric > 0)
        )
        
        # Save summary to file
        summary_path = self.log_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(sorted_configs, f, indent=2)
        
        return sorted_configs
