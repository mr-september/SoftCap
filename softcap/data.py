"""
SoftCap Data Module

Handles dataset loading and preprocessing for experiments.
"""

import torch
from torch.utils.data import DataLoader, random_split
try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torchvision = None
    transforms = None
from typing import Dict, Tuple, Any
from pathlib import Path
import numpy as np


def get_mnist_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    validation_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST data loaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of training data to use for validation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split training data into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_default_datasets() -> Dict[str, Any]:
    """
    Get default datasets for experiments.
    
    Returns:
        Dictionary containing train, validation, and test data loaders
    """
    train_loader, val_loader, test_loader = get_mnist_loaders()
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
        'dataset_name': 'MNIST',
        'input_shape': (1, 28, 28),
        'num_classes': 10
    }


def load_synthetic_dataset(dataset_name: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic dataset from saved .npy files.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'spiral', 'moons')
        data_dir: Directory containing the dataset files
    
    Returns:
        Tuple of (X, y) arrays
    """
    data_dir = Path(data_dir)
    X_file = data_dir / f"{dataset_name}_X.npy"
    y_file = data_dir / f"{dataset_name}_y.npy"
    
    if not X_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"Dataset files not found for {dataset_name} in {data_dir}")
    
    X = np.load(X_file)
    y = np.load(y_file)
    
    return X, y
