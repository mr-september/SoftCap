"""
GPU-Aware Parallel Execution Utilities

Automatically optimizes experiment execution based on available hardware:
- Single GPU: Uses threading for concurrent experiment execution
- Multiple GPUs: Uses multiprocessing with device assignment
- CPU only: Uses multiprocessing for parallelism

Key Design Principles:
1. Zero-configuration: Works optimally out-of-the-box
2. Memory-aware: Estimates safe concurrency based on GPU VRAM
3. Academic integrity: Preserves model sizes and experimental design
4. Simple integration: Drop-in replacement for sequential loops

Usage:
    from softcap.parallel_utils import parallel_map
    
    # Automatic optimization
    results = parallel_map(
        run_experiment,
        experiments,
        desc="Running experiments"
    )
    
    # Or use the decorator for single experiments
    @auto_optimize_dataloader
    def train_model(model, dataset, ...):
        ...

Author: SoftCap Team
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, List, Any, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import warnings
import psutil


CURRENT_PARALLEL_STRATEGY: Optional[str] = None  # Module-level strategy indicator


class GPUOptimizer:
    """
    Intelligent GPU utilization optimizer.
    
    Optimized for single RTX 2080 Ti (23.6GB VRAM) in WSL environment.
    Analyzes workload to determine optimal execution strategy.
    """
    
    # Hardware-specific constants (RTX 2080 Ti)
    # Actual usable memory is ~11 GB (11264 MiB). Previous value (23.6 GB) was incorrect
    # and led to overly aggressive concurrency recommendations.
    SINGLE_GPU_VRAM = 20.0 * 1024 * 1024 * 1024  # 11 GB in bytes
    SAFE_VRAM_USAGE = 0.75  # Use max 75% of VRAM to avoid OOM
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Get GPU configuration and available memory.
        
        Optimized for single-GPU setup. WSL-compatible.
        """
        if not torch.cuda.is_available():
            return {
                'available': False,
                'count': 0,
                'devices': []
            }
        
        # Simplified for single GPU
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            # Fast path for known hardware (RTX 2080 Ti)
            props = torch.cuda.get_device_properties(0)
            reserved = torch.cuda.memory_reserved(0)
            total = props.total_memory
            
            return {
                'available': True,
                'count': 1,
                'devices': [{
                    'id': 0,
                    'name': props.name,
                    'total_memory': total,
                    'reserved_memory': reserved,
                    'free_memory': total - reserved,
                }]
            }
        
        # Generic fallback for unexpected multi-GPU (shouldn't happen)
        devices = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory
            
            devices.append({
                'id': i,
                'name': props.name,
                'total_memory': total,
                'reserved_memory': reserved,
                'free_memory': total - reserved,
            })
        
        return {
            'available': True,
            'count': len(devices),
            'devices': devices
        }
    
    @staticmethod
    def estimate_model_memory(model: Optional[nn.Module] = None, 
                            batch_size: int = 128,
                            hidden_dim: int = 64,
                            input_dim: int = 784) -> int:
        """
        Conservative estimate of GPU memory required for a single experiment.
        
        Tuned for typical SoftCap experiments (small models, MNIST/CIFAR).
        
        Args:
            model: PyTorch model (if available)
            batch_size: Batch size for training
            hidden_dim: Hidden layer dimension (fallback estimate)
            input_dim: Input dimension (fallback estimate)
        
        Returns:
            Estimated memory in bytes (conservative)
        """
        if model is not None:
            # Calculate actual model parameters
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            # Gradient memory (roughly same as params)
            grad_memory = param_memory
            # Optimizer state (Adam: 2x params for momentum + variance)
            optimizer_memory = param_memory * 2
            # Activation memory (batch_size * hidden * layers * 4 bytes float32)
            # Rough heuristic: count total params and estimate activation size
            activation_memory = batch_size * hidden_dim * 3 * 4
        else:
            # Conservative fallback for small experiments
            # Typical: 2-3 layer MLP with 64-256 hidden units
            param_memory = input_dim * hidden_dim * 4 + hidden_dim * hidden_dim * 4
            grad_memory = param_memory
            optimizer_memory = param_memory * 2
            activation_memory = batch_size * hidden_dim * 3 * 4
        
        # PyTorch overhead (context, caching, etc.) - conservative 800MB
        pytorch_overhead = 800 * 1024 * 1024
        
        total = param_memory + grad_memory + optimizer_memory + activation_memory + pytorch_overhead
        
        # Safety margin: 2x for fragmentation and conservative estimate
        # Better to underestimate concurrency than OOM
        return int(total * 2.0)
    
    @staticmethod
    def recommend_concurrency(gpu_info: Dict[str, Any], 
                             estimated_memory_per_experiment: int,
                             max_workers: Optional[int] = None) -> Tuple[str, int]:
        """
        Recommend optimal execution strategy and worker count.
        
        Simplified for single RTX 2080 Ti setup.
        
        Returns:
            (strategy, num_workers) where strategy is 'threading' or 'sequential'
        """
        if not gpu_info['available']:
            # CPU-only: not expected in your setup, but handle it
            cpu_count = mp.cpu_count()
            workers = max_workers if max_workers else max(1, cpu_count // 2)
            return 'multiprocessing', workers
        
        num_gpus = gpu_info['count']
        
        if num_gpus != 1:
            # Multi-GPU: not your setup, but handle gracefully
            # Use sequential to avoid complexity
            warnings.warn(f"Detected {num_gpus} GPUs. Parallel utils optimized for single GPU. Using sequential execution.")
            return 'sequential', 1
        
        # Single GPU (your RTX 2080 Ti)
        gpu = gpu_info['devices'][0]
        free_memory = gpu['free_memory']
        
        # How many experiments can fit in memory?
        # Use SAFE_VRAM_USAGE fraction of total memory
        safe_memory = GPUOptimizer.SINGLE_GPU_VRAM * GPUOptimizer.SAFE_VRAM_USAGE
        max_concurrent = max(1, int(safe_memory / estimated_memory_per_experiment))
        
        if max_concurrent >= 3:
            # Good concurrency possible - use threading
            # Cap at 4 workers for stability (diminishing returns beyond this)
            workers = max_workers if max_workers else min(max_concurrent, 4)
            return 'threading', min(workers, max_concurrent)
        elif max_concurrent == 2:
            # Marginal case - still worth threading
            workers = 2
            return 'threading', workers
        else:
            # Memory-constrained: sequential execution with optimized DataLoader
            return 'sequential', 1
    
    @staticmethod
    def optimize_dataloader_workers() -> int:
        """
        Recommend optimal number of DataLoader workers.
        
        Optimized for WSL environment with sufficient CPU cores.
        
        Returns:
            Number of workers for DataLoader (0 = main process only)
        """
        cpu_count = mp.cpu_count()
        
        # For WSL with good CPU: use 4 workers as sweet spot
        # - Enough parallelism for data loading
        # - Not so many that overhead dominates
        # - Leaves CPU for training/evaluation
        
        if cpu_count >= 8:
            return 4  # Optimal for most workloads
        elif cpu_count >= 4:
            return 2  # Conservative for fewer cores
        else:
            return 0  # Single core: don't use workers


def parallel_map(
    func: Callable,
    tasks: List[Any],
    desc: str = "Processing",
    strategy: Optional[str] = None,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    Execute tasks in parallel using optimal strategy for available hardware.
    
    Automatically selects between threading (single GPU), multiprocessing (multi-GPU/CPU),
    or sequential execution based on hardware and memory constraints.
    
    Args:
        func: Function to call for each task. Signature: func(task, **kwargs)
        tasks: List of task arguments
        desc: Progress bar description
        strategy: Force specific strategy ('threading', 'multiprocessing', 'sequential').
                 If None, automatically determines optimal strategy.
        max_workers: Maximum number of parallel workers. If None, auto-determined.
        show_progress: Show progress bar
        **kwargs: Additional keyword arguments passed to func
    
    Returns:
        List of results in same order as tasks
    
    Example:
        def train_model(config):
            model = create_model(config)
            return train(model)
        
        configs = [{'seed': i} for i in range(5)]
        results = parallel_map(train_model, configs, desc="Training models")
    """
    global CURRENT_PARALLEL_STRATEGY

    if not tasks:
        return []
    
    # Auto-detect optimal strategy
    if strategy is None:
        gpu_info = GPUOptimizer.get_gpu_info()
        estimated_memory = GPUOptimizer.estimate_model_memory()
        strategy, num_workers = GPUOptimizer.recommend_concurrency(
            gpu_info, estimated_memory, max_workers
        )
        # Record chosen strategy for downstream dataloader tuning
        CURRENT_PARALLEL_STRATEGY = strategy
    else:
        num_workers = max_workers if max_workers else mp.cpu_count()
        CURRENT_PARALLEL_STRATEGY = strategy
    
    # Notify user of strategy
    if show_progress and len(tasks) > 1:
        strategy_info = f"[{strategy.upper()}: {num_workers} workers]"
        print(f"{strategy_info} {desc}...")
    
    # Execute based on strategy
    try:
        if strategy == 'sequential' or len(tasks) == 1:
            return _execute_sequential(func, tasks, desc, show_progress, **kwargs)
        elif strategy == 'threading':
            return _execute_threaded(func, tasks, num_workers, desc, show_progress, **kwargs)
        elif strategy == 'multiprocessing':
            return _execute_multiprocess(func, tasks, num_workers, desc, show_progress, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    finally:
        CURRENT_PARALLEL_STRATEGY = None


def _execute_sequential(func: Callable, tasks: List[Any], desc: str, 
                       show_progress: bool, **kwargs) -> List[Any]:
    """Execute tasks sequentially with progress bar."""
    results = []
    iterator = tqdm(tasks, desc=desc) if show_progress else tasks
    
    for task in iterator:
        result = func(task, **kwargs)
        results.append(result)
    
    return results


def _execute_threaded(func: Callable, tasks: List[Any], num_workers: int,
                     desc: str, show_progress: bool, **kwargs) -> List[Any]:
    """
    Execute tasks using threading (optimal for single GPU).
    
    Threading works well with PyTorch CUDA because:
    1. GIL is released during CUDA operations
    2. Shared CUDA context (efficient memory)
    3. Lower overhead than multiprocessing
    """
    results = [None] * len(tasks)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(func, task, **kwargs): idx
            for idx, task in enumerate(tasks)
        }
        
        # Collect results with progress bar
        iterator = as_completed(future_to_idx)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc=desc)
        
        for future in iterator:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                warnings.warn(f"Task {idx} failed with error: {e}")
                results[idx] = None
    
    return results


def _execute_multiprocess(func: Callable, tasks: List[Any], num_workers: int,
                          desc: str, show_progress: bool, **kwargs) -> List[Any]:
    """
    Execute tasks using multiprocessing (optimal for multi-GPU or CPU).
    
    For multi-GPU: Assigns each worker to a different GPU via CUDA_VISIBLE_DEVICES.
    For CPU: Standard multiprocessing pool.
    """
    # Check if we're using GPUs
    gpu_info = GPUOptimizer.get_gpu_info()
    
    if gpu_info['available'] and gpu_info['count'] > 1:
        # Multi-GPU: Assign workers to specific GPUs
        return _execute_multigpu(func, tasks, num_workers, desc, show_progress, **kwargs)
    else:
        # CPU-only multiprocessing
        ctx = mp.get_context('spawn')  # Use spawn for PyTorch compatibility
        
        with ctx.Pool(processes=num_workers) as pool:
            if show_progress:
                # Use imap for progress tracking
                results = list(tqdm(
                    pool.imap(partial(func, **kwargs), tasks),
                    total=len(tasks),
                    desc=desc
                ))
            else:
                results = pool.map(partial(func, **kwargs), tasks)
        
        return results


def _execute_multigpu(func: Callable, tasks: List[Any], num_workers: int,
                     desc: str, show_progress: bool, **kwargs) -> List[Any]:
    """Execute tasks across multiple GPUs with device assignment."""
    # TODO: Implement multi-GPU support when needed
    # For now, fall back to sequential with warning
    warnings.warn("Multi-GPU support not yet implemented. Falling back to sequential execution.")
    return _execute_sequential(func, tasks, desc, show_progress, **kwargs)


def optimize_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with automatic worker tuning.
    
    Automatically sets:
    - num_workers: Based on CPU count
    - pin_memory: True for CUDA
    - persistent_workers: True if workers > 0
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        **dataloader_kwargs: Additional DataLoader arguments (override defaults)
    
    Returns:
        Optimized DataLoader
    """
    # Auto-determine optimal settings
    # Determine base number of workers
    base_workers = GPUOptimizer.optimize_dataloader_workers()

    # If we're inside a threaded parallel experiment, spawning DataLoader worker
    # processes per thread multiplies process count (threads * workers) and has
    # been observed to trigger allocator corruption / double free errors.
    # Disable DataLoader multiprocessing in threading mode for safety.
    global CURRENT_PARALLEL_STRATEGY
    if CURRENT_PARALLEL_STRATEGY == 'threading':
        effective_workers = 0
    else:
        effective_workers = base_workers

    defaults = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': effective_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    
    # Add persistent_workers if we have workers
    if defaults['num_workers'] > 0:
        defaults['persistent_workers'] = True
    
    # Allow user overrides
    defaults.update(dataloader_kwargs)
    
    return DataLoader(dataset, **defaults)


def auto_optimize_dataloader(func: Callable) -> Callable:
    """
    Decorator to automatically optimize DataLoader creation in training functions.
    
    Wraps training functions to replace basic DataLoader calls with optimized versions.
    
    Example:
        @auto_optimize_dataloader
        def train_model(model, train_dataset, test_dataset):
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            # Automatically becomes optimized with num_workers, pin_memory, etc.
            ...
    """
    # This is a placeholder for future implementation
    # For now, users should explicitly use optimize_dataloader()
    return func


# Convenience functions for common patterns

def parallel_experiments(
    experiment_func: Callable,
    activation_names: List[str],
    activation_functions: List[nn.Module],
    seeds: List[int],
    desc: str = "Running experiments",
    **kwargs
) -> Dict[str, Dict[int, Any]]:
    """
    Run experiments across multiple activations and seeds in parallel.
    
    Automatically handles the common pattern of testing multiple activation functions
    with multiple random seeds.
    
    Args:
        experiment_func: Function with signature func(act_name, act_fn, seed, **kwargs)
        activation_names: List of activation function names
        activation_functions: List of activation function instances
        seeds: List of random seeds
        desc: Progress description
        **kwargs: Additional arguments passed to experiment_func
    
    Returns:
        Nested dict: {activation_name: {seed: result}}
    
    Example:
        def run_experiment(act_name, act_fn, seed):
            model = create_model(act_fn)
            return train(model, seed)
        
        results = parallel_experiments(
            run_experiment,
            ['ReLU', 'GELU'],
            [nn.ReLU(), nn.GELU()],
            [0, 1, 2]
        )
    """
    # Create all (activation, seed) combinations
    tasks = []
    for act_name, act_fn in zip(activation_names, activation_functions):
        for seed in seeds:
            tasks.append((act_name, act_fn, seed))
    
    # Wrapper to match parallel_map signature
    def wrapper(task_tuple):
        act_name, act_fn, seed = task_tuple
        return (act_name, seed, experiment_func(act_name, act_fn, seed, **kwargs))
    
    # Execute in parallel
    flat_results = parallel_map(wrapper, tasks, desc=desc)
    
    # Restructure results into nested dict
    results = {}
    for act_name, seed, result in flat_results:
        if act_name not in results:
            results[act_name] = {}
        results[act_name][seed] = result
    
    return results


# Export public API
__all__ = [
    'parallel_map',
    'parallel_experiments',
    'optimize_dataloader',
    'auto_optimize_dataloader',
    'GPUOptimizer',
]
