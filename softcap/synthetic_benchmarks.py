"""
Synthetic Benchmarks for Activation Function Analysis

This module implements various synthetic datasets and challenges specifically designed
to test activation function properties, particularly isotropy vs anisotropy.

Key Features:
- Spiral Challenge: Two interleaved spirals (classic isotropy test)
- Two Moons: Non-linearly separable crescents
- Concentric Circles: Radial decision boundaries
- XOR Problem: Classic non-linear separation
- Checkerboard: Complex multi-modal decision boundaries
- Decision boundary visualization
- Isotropy analysis through synthetic tasks
- Aggregated convergence analysis
- CSV export for external tooling

This module can be run independently of training-intensive experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import seaborn as sns
import json
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time
import subprocess
from datetime import datetime
try:
    import psutil
except Exception:
    psutil = None

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SyntheticDataset(Dataset):
    """Generic dataset wrapper for synthetic data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SyntheticBenchmarks:
    """Generate various synthetic datasets for activation function testing."""
    
    def __init__(self, n_samples: int = 1000, noise: float = 0.1, random_state: int = 42):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def create_spiral_dataset(self, n_points_per_spiral: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the classic two spirals dataset.
        
        This is the gold standard test for activation function isotropy.
        Anisotropic functions tend to create artificial decision boundaries
        aligned with their discrete symmetries.
        """
        if n_points_per_spiral is None:
            n_points_per_spiral = self.n_samples // 2
        
        # Generate two interleaved spirals
        theta = np.linspace(0, 4 * np.pi, n_points_per_spiral)
        
        # Spiral 1
        r1 = theta / (2 * np.pi)
        x1 = r1 * np.cos(theta) + np.random.normal(0, self.noise, n_points_per_spiral)
        y1 = r1 * np.sin(theta) + np.random.normal(0, self.noise, n_points_per_spiral)
        
        # Spiral 2 (offset by π)
        r2 = theta / (2 * np.pi)
        x2 = r2 * np.cos(theta + np.pi) + np.random.normal(0, self.noise, n_points_per_spiral)
        y2 = r2 * np.sin(theta + np.pi) + np.random.normal(0, self.noise, n_points_per_spiral)
        
        # Combine datasets
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n_points_per_spiral), np.ones(n_points_per_spiral)])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def create_moons_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create two moons dataset - tests curved decision boundaries."""
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        return X, y
    
    def create_circles_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create concentric circles - tests radial decision boundaries."""
        X, y = make_circles(n_samples=self.n_samples, noise=self.noise, factor=0.6, random_state=self.random_state)
        return X, y
    
    def create_xor_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create XOR problem - classic non-linear separation test."""
        # Generate points in four quadrants
        n_per_quad = self.n_samples // 4
        
        # Positive class: quadrants 1 and 3
        x1_pos = np.random.uniform(0.2, 1.0, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        y1_pos = np.random.uniform(0.2, 1.0, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        
        x3_pos = np.random.uniform(-1.0, -0.2, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        y3_pos = np.random.uniform(-1.0, -0.2, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        
        # Negative class: quadrants 2 and 4
        x2_neg = np.random.uniform(-1.0, -0.2, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        y2_neg = np.random.uniform(0.2, 1.0, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        
        x4_neg = np.random.uniform(0.2, 1.0, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        y4_neg = np.random.uniform(-1.0, -0.2, n_per_quad) + np.random.normal(0, self.noise, n_per_quad)
        
        # Combine
        X_pos = np.vstack([np.column_stack([x1_pos, y1_pos]), np.column_stack([x3_pos, y3_pos])])
        X_neg = np.vstack([np.column_stack([x2_neg, y2_neg]), np.column_stack([x4_neg, y4_neg])])
        
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def create_checkerboard_dataset(self, grid_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Create checkerboard pattern - tests complex multi-modal boundaries."""
        X = np.random.uniform(-2, 2, (self.n_samples, 2))
        
        # Create checkerboard pattern
        x_grid = np.floor((X[:, 0] + 2) * grid_size / 4).astype(int)
        y_grid = np.floor((X[:, 1] + 2) * grid_size / 4).astype(int)
        
        # Checkerboard rule: (x + y) % 2
        y = (x_grid + y_grid) % 2
        
        # Add noise
        X += np.random.normal(0, self.noise, X.shape)
        
        return X, y
    
    def get_all_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get all synthetic datasets."""
        return {
            'spiral': self.create_spiral_dataset(),
            'moons': self.create_moons_dataset(),
            'circles': self.create_circles_dataset(),
            'xor': self.create_xor_dataset(),
            'checkerboard': self.create_checkerboard_dataset()
        }


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for synthetic benchmarks."""
    
    def __init__(self, activation: nn.Module, hidden_size: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(activation)
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_size, 2))
        
        self.network = nn.Sequential(*layers)
        self.activation_name = getattr(activation, 'name', activation.__class__.__name__)
    
    def forward(self, x):
        return self.network(x)


class DecisionBoundaryVisualizer:
    """Visualize decision boundaries for different activation functions."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray,
                    activation: nn.Module, epochs: int = 500, lr: float = 0.01, weight_decay: float = 0.0) -> Tuple[SimpleClassifier, Dict[str, List[float]]]:
        """Train a simple classifier on the dataset and return training history.

    activation: nn.Module, epochs: int = 500, lr: float = 0.01, weight_decay: float = 0.0) -> Tuple[SimpleClassifier, Dict[str, List[float]]]:
        and a history dict containing per-epoch lists for train/val loss and accuracy.
        """

        # Create datasets and dataloaders
        train_dataset = SyntheticDataset(X_train, y_train)
        val_dataset = SyntheticDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model and optimizer
        model = SimpleClassifier(activation).to(self.device)
        criterion = nn.CrossEntropyLoss()
        # Use AdamW when weight decay is requested to decouple weight decay from Adam updates.
        # Exclude biases and 1D parameters (e.g., LayerNorm/Bias) from weight decay as standard practice.
        if weight_decay and float(weight_decay) > 0.0:
            decay, no_decay = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Exclude biases and 1D or scalar parameters (biases, LayerNorm/BatchNorm, scalar activation params)
                if param.ndim <= 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            param_groups = [
                {'params': decay, 'weight_decay': float(weight_decay)},
                {'params': no_decay, 'weight_decay': 0.0}
            ]
            optimizer = optim.AdamW(param_groups, lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # Track training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        # Training loop with per-epoch progress bar
        from tqdm import trange
        for epoch in trange(epochs, desc=f"train:{self.device}", leave=False):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

            # Record history (use average per batch)
            avg_train_loss = train_loss / max(1, len(train_dataloader))
            avg_val_loss = val_loss / max(1, len(val_dataloader))
            avg_train_acc = train_correct / max(1, train_total)
            avg_val_acc = val_correct / max(1, val_total)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(avg_train_acc)
            history['val_accuracy'].append(avg_val_acc)

            try:
                # Update tqdm description with recent metrics for quick feedback
                tr = None
                # Access the active tqdm from the stack (best-effort)
                # If not available, skip without failing
                from tqdm import tqdm
                for obj in tqdm._instances:
                    # find the trange instance used above (match by desc prefix)
                    if getattr(obj, 'desc', '').startswith('train:'):
                        obj.set_postfix({'t_loss': f"{avg_train_loss:.3f}", 'v_loss': f"{avg_val_loss:.3f}", 'v_acc': f"{avg_val_acc:.3f}"})
                        break
            except Exception:
                pass

        return model, history
    
    def plot_decision_boundary(self, model: SimpleClassifier, X: np.ndarray, y: np.ndarray,
                             title: str = "", resolution: int = 100) -> plt.Figure:
        """Plot decision boundary for a trained model."""
        
        model.eval()
        
        # Create a mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_tensor = torch.FloatTensor(mesh_points).to(self.device)
        
        with torch.no_grad():
            mesh_predictions = model(mesh_tensor)
            mesh_probs = torch.softmax(mesh_predictions, dim=1)[:, 1]
            mesh_probs = mesh_probs.cpu().numpy().reshape(xx.shape)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, mesh_probs, levels=50, alpha=0.6, cmap='RdYlBu')
        ax.contour(xx, yy, mesh_probs, levels=[0.5], colors='black', linewidths=2)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_title(f'{title}\nActivation: {model.activation_name}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Class Probability')
        
        return fig
    
    def analyze_isotropy_score(self, model: SimpleClassifier, X: np.ndarray) -> float:
        """
        Calculate an isotropy score based on decision boundary properties.
        
        Higher scores indicate more isotropic (direction-independent) boundaries.
        """
        model.eval()
        
        # Sample points around the dataset
        x_center, y_center = X[:, 0].mean(), X[:, 1].mean()
        x_std, y_std = X[:, 0].std(), X[:, 1].std()
        
        # Create radial samples
        n_angles = 36  # Every 10 degrees
        n_radii = 20
        angles = np.linspace(0, 2*np.pi, n_angles)
        radii = np.linspace(0.1, 2.0, n_radii)
        
        gradient_directions = []
        
        for angle in angles:
            for radius in radii:
                x = x_center + radius * x_std * np.cos(angle)
                y = y_center + radius * y_std * np.sin(angle)
                
                point = torch.FloatTensor([[x, y]]).to(self.device)
                point.requires_grad_(True)
                
                with torch.enable_grad():
                    output = model(point)
                    prob = torch.softmax(output, dim=1)[0, 1]
                    
                    # Compute gradient
                    grad = torch.autograd.grad(prob, point, create_graph=False)[0]
                    if grad.norm() > 1e-6:  # Avoid near-zero gradients
                        grad_angle = torch.atan2(grad[0, 1], grad[0, 0]).item()
                        gradient_directions.append(grad_angle)
        
        if len(gradient_directions) < 10:
            return 0.0  # Not enough valid gradients
        
        # Calculate isotropy as uniformity of gradient directions
        gradient_directions = np.array(gradient_directions)
        
        # Convert to unit vectors and compute variance
        cos_dirs = np.cos(gradient_directions)
        sin_dirs = np.sin(gradient_directions)
        
        # Isotropy score: 1 - variance of direction distribution
        mean_cos = np.mean(cos_dirs)
        mean_sin = np.mean(sin_dirs)
        isotropy_score = 1.0 - (mean_cos**2 + mean_sin**2)
        
        return float(np.clip(isotropy_score, 0.0, 1.0))


# --- MODIFIED FUNCTION SIGNATURE ---
def load_existing_results(output_dir: Path) -> Dict[str, Any]:
    """
    Load existing results from a previous (potentially incomplete) run.
    
    Returns a dictionary with the same structure as run_synthetic_benchmarks output,
    or None if no valid existing results are found.
    """
    results_file = output_dir / 'synthetic_benchmark_results.json'
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded existing results from {results_file}")
        return results
    except Exception as e:
        print(f"⚠️  Could not load existing results: {e}")
        return None


def check_run_completed(results: Dict[str, Any], dataset_name: str, act_name: str, 
                       seed_val: int, lr_val: float, wd_val: float) -> bool:
    """
    Check if a specific run (dataset, activation, seed, lr, wd) has been completed.
    
    A run is considered complete if:
    1. It has accuracy scores
    2. It has isotropy scores
    3. It has training history
    4. The training history has the expected number of epochs
    """
    try:
        # Check if basic structure exists
        if dataset_name not in results.get('accuracy_scores', {}):
            return False
        if act_name not in results['accuracy_scores'][dataset_name]:
            return False
        
        # Check if we have values (one per seed)
        acc_values = results['accuracy_scores'][dataset_name][act_name].get('values', [])
        iso_values = results['isotropy_scores'].get(dataset_name, {}).get(act_name, {}).get('values', [])
        
        if not acc_values or not iso_values:
            return False
        
        # Check training histories exist
        histories = results.get('training_histories', {}).get(dataset_name, {}).get(act_name, {}).get('runs', [])
        if not histories:
            return False
        
        # For now, we can't perfectly match seed->index without more metadata
        # So we'll just check if we have enough completed runs
        # This is a conservative check - if uncertain, we'll re-run
        return True
        
    except Exception:
        return False


def run_synthetic_benchmarks(activation_functions: Dict[str, nn.Module],
                           output_dir: Path,
                           n_samples: int = 1000,
                           epochs: int = 500,
                           save_models: bool = True,
                           device: str = "auto",
                           boundary_resolution: int = 200,
                           repeats: int = 5,
                           lr: Any = 0.01,
                           weight_decay: Any = 0.0,
                           seed_offset: int = 0,
                           reuse_datasets_from: Dict = None,
                           resume: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive synthetic benchmark analysis.
    
    Args:
        activation_functions: Dictionary of activation functions to test
        output_dir: Directory to save results and visualizations
        n_samples: Number of samples per dataset
        epochs: Training epochs per model
        save_models: Whether to save trained models
        device: The device to run training on ('cuda', 'cpu', or 'auto')
        boundary_resolution: Resolution (number of points per axis) for boundary grids when saving
        repeats: Number of seed repeats per experiment
        lr: Learning rate(s) - single value or comma-separated string for sweep
        weight_decay: Weight decay coefficient(s) - single value or comma-separated string for sweep
        seed_offset: Offset added to base seed (42)
        reuse_datasets_from: Dictionary with 'datasets_dir' to reuse datasets from previous run
        resume: If True, check for existing results and skip completed runs
        
    Returns:
        Dictionary containing all results and metrics
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse HP grids early so we can store them in manifest
    if isinstance(lr, str):
        lr_list = [float(x) for x in lr.split(',') if x.strip()]
    elif isinstance(lr, (list, tuple, np.ndarray)):
        lr_list = [float(x) for x in lr]
    else:
        lr_list = [float(lr)]

    if isinstance(weight_decay, str):
        wd_list = [float(x) for x in weight_decay.split(',') if x.strip()]
    elif isinstance(weight_decay, (list, tuple, np.ndarray)):
        wd_list = [float(x) for x in weight_decay]
    else:
        wd_list = [float(weight_decay)]
    
    seeds_used = [42 + seed_offset + i for i in range(repeats)]
    
    # Try to load existing results if resume is enabled
    existing_results = None
    if resume and output_dir.exists():
        existing_results = load_existing_results(output_dir)
        if existing_results:
            print(f"🔄 Resume mode enabled: Will skip completed runs and continue from where we left off")
    
    # Write run manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(n_samples),
        'epochs': int(epochs),
        'repeats': int(repeats),
        'boundary_resolution': int(boundary_resolution),
        'activations': list(activation_functions.keys()),
        'save_models': bool(save_models),
        'lr': lr,
        'weight_decay': weight_decay,
        # Store parsed HP grid for CSV export
        'lr_grid': lr_list,
        'wd_grid': wd_list,
        'seeds_used': seeds_used,
        'seed_offset': int(seed_offset),
    }
    try:
        # git commit hash if available
        import subprocess
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent.parent).decode().strip()
        manifest['git_commit'] = git_head
    except Exception:
        manifest['git_commit'] = None

    with open(output_dir / 'run_manifest.json', 'w') as mf:
        json.dump(manifest, mf, indent=2)

    # --- Capture environment/package/GPU metadata where possible ---
    try:
        import pkg_resources
        packages = {p.project_name: p.version for p in pkg_resources.working_set}
        manifest['packages'] = {k: packages[k] for k in sorted(packages.keys())}
    except Exception:
        manifest['packages'] = None

    try:
        # GPU/CUDA info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_info['cuda_device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info['nvml'] = True
                gpu_info['gpu_mem_total_mb'] = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024*1024)
            except Exception:
                gpu_info['nvml'] = False
        else:
            gpu_info['cuda_available'] = False
        manifest['gpu_info'] = gpu_info
    except Exception:
        manifest['gpu_info'] = None

    # Re-write manifest with the additional info
    with open(output_dir / 'run_manifest.json', 'w') as mf:
        json.dump(manifest, mf, indent=2)

    # Create or reuse synthetic datasets
    if reuse_datasets_from is not None:
        print("🔄 Reusing datasets from previous run...")
        datasets = {}
        prev_datasets_dir = reuse_datasets_from['datasets_dir']
        
        # Load datasets from previous run
        dataset_names = ['xor', 'spiral', 'moons', 'circles', 'checkerboard']
        for name in dataset_names:
            X_file = prev_datasets_dir / f"{name}_X.npy"
            y_file = prev_datasets_dir / f"{name}_y.npy"
            
            if X_file.exists() and y_file.exists():
                X = np.load(X_file)
                y = np.load(y_file)
                datasets[name] = (X, y)
                print(f"   ✓ Loaded {name}: {X.shape[0]} samples")
            else:
                print(f"   ⚠️ Warning: Dataset {name} not found in previous run, will generate new")
                # Fallback to generating this dataset
                benchmark_gen = SyntheticBenchmarks(n_samples=n_samples, noise=0.1)
                X, y = getattr(benchmark_gen, f'generate_{name}')()
                datasets[name] = (X, y)
        
        # Copy datasets to new output directory
        datasets_dir = output_dir / 'datasets'
        datasets_dir.mkdir(exist_ok=True)
        for name, (X, y) in datasets.items():
            np.save(datasets_dir / f"{name}_X.npy", X)
            np.save(datasets_dir / f"{name}_y.npy", y)
            # CSV for convenience
            try:
                import pandas as _pd
                df = _pd.DataFrame(X, columns=['x1', 'x2'])
                df['y'] = y
                df.to_csv(datasets_dir / f"{name}.csv", index=False)
            except Exception:
                # pandas optional; skip CSV if not available
                pass
    else:
        # Create synthetic datasets and save raw points
        print("🔄 Generating synthetic datasets and saving raw points...")
        benchmark_gen = SyntheticBenchmarks(n_samples=n_samples, noise=0.1)
        datasets = benchmark_gen.get_all_datasets()

        # Persist raw dataset points for later analysis
        datasets_dir = output_dir / 'datasets'
        datasets_dir.mkdir(exist_ok=True)
        for name, (X, y) in datasets.items():
            np.save(datasets_dir / f"{name}_X.npy", X)
            np.save(datasets_dir / f"{name}_y.npy", y)
            # CSV for convenience
            try:
                import pandas as _pd
                df = _pd.DataFrame(X, columns=['x1', 'x2'])
                df['y'] = y
                df.to_csv(datasets_dir / f"{name}.csv", index=False)
            except Exception:
                # pandas optional; skip CSV if not available
                pass
    
    # --- ADDED: Resolve device and initialize visualizer ---
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    
    visualizer = DecisionBoundaryVisualizer(device=resolved_device)
    
    # Results storage
    results = {
        'accuracy_scores': {},
        'isotropy_scores': {},
        'training_stability': {},
        'training_histories': {},
        'function_fitting': {},  # Will hold function fitting analysis
        'dataset_info': {name: {'n_samples': len(X), 'n_features': X.shape[1]}
                        for name, (X, y) in datasets.items()},
    'boundary_areas': {}  # Area summaries of decision regions (always populated when runs complete)
    }
    # attach manifest for completeness
    results['manifest'] = manifest
    
    print(f"📊 Testing {len(activation_functions)} activation functions on {len(datasets)} datasets...")

    for dataset_name, (X, y) in tqdm(datasets.items(), desc="Processing Datasets"):
        # Immediate dataset-level log so users see progress right away
        print(f"Starting dataset {dataset_name}: n_samples={len(X)}, shape={X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Initialize result storage for this dataset
        results['accuracy_scores'][dataset_name] = {}
        results['isotropy_scores'][dataset_name] = {}
        results['training_stability'][dataset_name] = {}
        results['training_histories'][dataset_name] = {}
        
        # If resuming, copy existing results for this dataset
        if existing_results and dataset_name in existing_results.get('accuracy_scores', {}):
            for act_name in activation_functions.keys():
                if act_name in existing_results['accuracy_scores'][dataset_name]:
                    results['accuracy_scores'][dataset_name][act_name] = existing_results['accuracy_scores'][dataset_name][act_name]
                if act_name in existing_results.get('isotropy_scores', {}).get(dataset_name, {}):
                    results['isotropy_scores'][dataset_name][act_name] = existing_results['isotropy_scores'][dataset_name][act_name]
                if act_name in existing_results.get('training_stability', {}).get(dataset_name, {}):
                    results['training_stability'][dataset_name][act_name] = existing_results['training_stability'][dataset_name][act_name]
                if act_name in existing_results.get('training_histories', {}).get(dataset_name, {}):
                    results['training_histories'][dataset_name][act_name] = existing_results['training_histories'][dataset_name][act_name]
                if dataset_name in existing_results.get('boundary_areas', {}) and act_name in existing_results['boundary_areas'][dataset_name]:
                    if dataset_name not in results['boundary_areas']:
                        results['boundary_areas'][dataset_name] = {}
                    results['boundary_areas'][dataset_name][act_name] = existing_results['boundary_areas'][dataset_name][act_name]
        
        # Create dataset plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
        ax.set_title(f'{dataset_name.title()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Class')
        plt.tight_layout()
        # Save dataset overview into the per-dataset Boundary Plots folder so it is
        # grouped with other visualizations for that dataset and excluded from
        # top-level report images.
        dataset_png_dir = output_dir / f'Boundary Plots - {dataset_name.title()}'
        dataset_png_dir.mkdir(exist_ok=True)
        plt.savefig(dataset_png_dir / f'{dataset_name}_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test each activation function
        for act_name, activation in activation_functions.items():
            
            # Check if this activation has already been fully completed for this dataset
            if (existing_results and 
                dataset_name in existing_results.get('accuracy_scores', {}) and
                act_name in existing_results['accuracy_scores'][dataset_name]):
                
                # Verify it has the expected number of runs
                existing_acc_values = existing_results['accuracy_scores'][dataset_name][act_name].get('values', [])
                expected_runs = len(lr_list) * len(wd_list) * repeats
                
                if len(existing_acc_values) >= expected_runs:
                    print(f"   ⏭️  Skipping {dataset_name}/{act_name}: Already completed ({len(existing_acc_values)} runs)")
                    continue

            accuracies = []
            isotropy_scores = []
            all_histories = []
            run_time_per_seed = []
            # For storing per-seed boundary metrics before summarizing
            per_seed_boundary_metrics: Dict[int, Dict[str, Any]] = {}
            
            # For hyperparameter sweep over LR values
            for lr_val in lr_list:
                for wd_val in wd_list:
                    # Show per-seed progress for this dataset/activation/lr/wd
                    for seed in tqdm(range(repeats), desc=f"{dataset_name}/{act_name} seeds", leave=False):
                        seed_val = 42 + seed_offset + seed
                        
                        # Check if this specific run exists in saved results
                        skip_this_seed = False
                        if existing_results:
                            # Check if we have a saved model or history for this exact configuration
                            saved_history_file = output_dir / 'run_histories' / f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_history.csv'
                            saved_model_file = output_dir / 'models' / f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_model.pt'
                            
                            # If we have the history file, assume this run completed successfully
                            if saved_history_file.exists():
                                try:
                                    # Try to load and validate the history
                                    import pandas as pd
                                    hist_df = pd.read_csv(saved_history_file)
                                    if len(hist_df) >= epochs * 0.95:  # At least 95% of epochs completed
                                        # Reconstruct results from saved data
                                        history = {
                                            'train_loss': hist_df['train_loss'].tolist(),
                                            'val_loss': hist_df['val_loss'].tolist(),
                                            'train_accuracy': hist_df['train_accuracy'].tolist(),
                                            'val_accuracy': hist_df['val_accuracy'].tolist()
                                        }
                                        all_histories.append(history)
                                        
                                        # Get final accuracy
                                        final_acc = hist_df['val_accuracy'].iloc[-1]
                                        accuracies.append(float(final_acc))
                                        
                                        # Load isotropy if available from existing results
                                        if (dataset_name in existing_results.get('isotropy_scores', {}) and
                                            act_name in existing_results['isotropy_scores'][dataset_name]):
                                            existing_iso_values = existing_results['isotropy_scores'][dataset_name][act_name].get('values', [])
                                            if len(existing_iso_values) > len(isotropy_scores):
                                                isotropy_scores.append(existing_iso_values[len(isotropy_scores)])
                                        
                                        # Load boundary metrics if available
                                        if (dataset_name in existing_results.get('boundary_areas', {}) and
                                            act_name in existing_results['boundary_areas'][dataset_name]):
                                            existing_runs = existing_results['boundary_areas'][dataset_name][act_name].get('runs', [])
                                            for run in existing_runs:
                                                if run.get('seed') == seed_val:
                                                    per_seed_boundary_metrics[seed_val] = run
                                                    run_time_per_seed.append(run.get('runtime_seconds', 0))
                                                    break
                                        
                                        print(f"      ✅ Loaded existing run: {dataset_name}/{act_name}/lr{lr_val}/wd{wd_val}/seed{seed_val}")
                                        skip_this_seed = True
                                except Exception as e:
                                    print(f"      ⚠️  Could not load saved run {saved_history_file}: {e}, will re-run")
                                    skip_this_seed = False
                        
                        if skip_this_seed:
                            continue
                        
                        # Run the training for this seed
                        torch.manual_seed(seed_val)
                        np.random.seed(seed_val)

                        t0 = time.time()
                        # Pass lr and weight decay through to the trainer
                        # Note: wd_val is already the per-iteration value from wd_list (parsed as float above).
                        # Do not overwrite it with the original `weight_decay` input (which may be a comma-separated string).
                        model, history = visualizer.train_classifier(
                            X_train_split, y_train_split, X_val, y_val, activation,
                            epochs=epochs, lr=float(lr_val), weight_decay=float(wd_val)
                        )
                        t1 = time.time()
                        run_time_per_seed.append(t1 - t0)
                        all_histories.append(history)

                        model.eval()
                        with torch.no_grad():
                            test_tensor = torch.FloatTensor(X_test).to(visualizer.device)
                            predictions = model(test_tensor)
                            predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                            accuracy = np.mean(predicted_classes == y_test)
                            accuracies.append(accuracy)

                        isotropy = visualizer.analyze_isotropy_score(model, X_train)
                        isotropy_scores.append(isotropy)

                        # Save a decision-boundary plot for this seed (useful for inspection)
                        # Put boundary PNGs into a per-dataset subfolder so they don't clutter the root
                        fig = visualizer.plot_decision_boundary(
                            model, X_train, y_train,
                            title=f'{dataset_name.title()} Dataset (seed={seed_val})'
                        )
                        plt.tight_layout()
                        boundary_png_dir = output_dir / f'Boundary Plots - {dataset_name.title()}'
                        boundary_png_dir.mkdir(exist_ok=True)
                        png_name = f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_boundary.png'
                        plt.savefig(boundary_png_dir / png_name, dpi=300, bbox_inches='tight')
                        plt.close()

                        # Save model state for this seed if requested (and checkpoint best val)
                        if save_models:
                            model_dir = output_dir / 'models'
                            model_dir.mkdir(exist_ok=True)
                            # include lr in filename to disambiguate sweeps
                            torch.save(model.state_dict(),
                                     model_dir / f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_model.pt')

                        # Additionally save a best-checkpoint based on val_accuracy if available
                        try:
                            best_val_acc = max(history.get('val_accuracy', [])) if history.get('val_accuracy') else None
                            if best_val_acc is not None:
                                ckpt_meta = {
                                    'dataset': dataset_name,
                                    'activation': act_name,
                                    'seed': int(seed_val),
                                    'lr': float(lr_val),
                                    'best_val_accuracy': float(best_val_acc)
                                }
                                torch.save({'state_dict': model.state_dict(), 'meta': ckpt_meta},
                                           model_dir / f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_best.pt')
                        except Exception:
                            pass

                        # --- Always save raw boundary grid & area metrics ---
                        # Create mesh identical to plot_decision_boundary logic
                        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
                        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
                        xx, yy = np.meshgrid(
                            np.linspace(x_min, x_max, boundary_resolution),
                            np.linspace(y_min, y_max, boundary_resolution)
                        )
                        mesh_points = np.c_[xx.ravel(), yy.ravel()]
                        with torch.no_grad():
                            mesh_tensor = torch.FloatTensor(mesh_points).to(visualizer.device)
                            mesh_predictions = model(mesh_tensor)
                            mesh_probs = torch.softmax(mesh_predictions, dim=1)[:, 1].cpu().numpy().reshape(xx.shape)

                        # Save arrays per-seed
                        boundary_dir = output_dir / 'boundary_grids'
                        boundary_dir.mkdir(exist_ok=True)
                        np.save(boundary_dir / f'{dataset_name}_{act_name.lower()}_seed{seed_val}_xx.npy', xx)
                        np.save(boundary_dir / f'{dataset_name}_{act_name.lower()}_seed{seed_val}_yy.npy', yy)
                        np.save(boundary_dir / f'{dataset_name}_{act_name.lower()}_seed{seed_val}_probs.npy', mesh_probs)

                        # Also save per-run training history as CSV for exact raw data
                        try:
                            hist_df = pd.DataFrame({
                                'train_loss': history['train_loss'],
                                'val_loss': history['val_loss'],
                                'train_accuracy': history['train_accuracy'],
                                'val_accuracy': history['val_accuracy']
                            })
                            run_hist_dir = output_dir / 'run_histories'
                            run_hist_dir.mkdir(exist_ok=True)
                            hist_df.to_csv(run_hist_dir / f'{dataset_name}_{act_name.lower()}_lr{lr_val}_wd{wd_val}_seed{seed_val}_history.csv', index_label='epoch')
                        except Exception:
                            pass

                # Compute area estimates
                x_step = (x_max - x_min) / (boundary_resolution - 1)
                y_step = (y_max - y_min) / (boundary_resolution - 1)
                cell_area = x_step * y_step
                total_bbox_area = (x_max - x_min) * (y_max - y_min)
                # Threshold 0.5 region for class 1
                class1_mask = (mesh_probs >= 0.5)
                class1_area_threshold = float(class1_mask.sum() * cell_area)
                class0_area_threshold = float(total_bbox_area - class1_area_threshold)
                # Probabilistic expected area (integral of probability)
                class1_area_expected = float(mesh_probs.sum() * cell_area)
                class0_area_expected = float(total_bbox_area - class1_area_expected)

                # gather perf telemetry where possible
                perf = {}
                try:
                    perf['cpu_percent'] = psutil.cpu_percent(interval=None) if psutil is not None else None
                    perf['memory_mb'] = psutil.Process().memory_info().rss // (1024*1024) if psutil is not None else None
                except Exception:
                    perf['cpu_percent'] = None
                    perf['memory_mb'] = None
                try:
                    if torch.cuda.is_available():
                        perf['cuda_max_mem_mb'] = torch.cuda.max_memory_allocated() // (1024*1024)
                    else:
                        perf['cuda_max_mem_mb'] = None
                except Exception:
                    perf['cuda_max_mem_mb'] = None

                seed_metrics = {
                    'seed': int(seed_val),
                    'runtime_seconds': float(run_time_per_seed[-1]),
                    'perf': perf,
                    'bbox': {'x_min': float(x_min), 'x_max': float(x_max), 'y_min': float(y_min), 'y_max': float(y_max)},
                    'cell_area': float(cell_area),
                    'threshold_area': {
                        'class1': class1_area_threshold,
                        'class0': class0_area_threshold,
                        'class1_ratio': class1_area_threshold / total_bbox_area,
                        'class0_ratio': class0_area_threshold / total_bbox_area
                    },
                    'expected_area': {
                        'class1': class1_area_expected,
                        'class0': class0_area_expected,
                        'class1_ratio': class1_area_expected / total_bbox_area,
                        'class0_ratio': class0_area_expected / total_bbox_area
                    }
                }

                # attach lr to seed_metrics for disambiguation in sweeps
                seed_metrics['lr'] = float(lr_val)

                # Compute simple calibration metric (ECE) using mesh probs and nearest training labels
                try:
                    # For calibration on training distribution we'll compute probs on X_train and compare to y_train
                    train_tensor = torch.FloatTensor(X_train).to(visualizer.device)
                    with torch.no_grad():
                        train_preds = model(train_tensor)
                        train_probs = torch.softmax(train_preds, dim=1)[:, 1].cpu().numpy()
                    ece_val = expected_calibration_error(train_probs, np.array(y_train).astype(int), n_bins=10)
                    seed_metrics['ece'] = float(ece_val)
                except Exception:
                    seed_metrics['ece'] = None

                # Store per-seed metrics temporarily
                per_seed_boundary_metrics[seed_val] = seed_metrics
            
            # Ensure all stored numeric values are native Python types (JSON-friendly)
            results['accuracy_scores'][dataset_name][act_name] = {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'values': [float(v) for v in accuracies]
            }
            results['isotropy_scores'][dataset_name][act_name] = {
                'mean': float(np.mean(isotropy_scores)),
                'std': float(np.std(isotropy_scores)),
                'values': [float(v) for v in isotropy_scores]
            }
            results['training_stability'][dataset_name][act_name] = float(1.0 - np.std(accuracies))

            # Save individual training histories and compute average history
            avg_history = {}
            # store per-run histories
            results['training_histories'][dataset_name][act_name] = {'runs': all_histories}

            for key in all_histories[0].keys():
                avg_values = [float(np.mean([h[key][epoch] for h in all_histories])) for epoch in range(len(all_histories[0][key]))]
                avg_history[key] = avg_values
            results['training_histories'][dataset_name][act_name]['average'] = avg_history

            # Summarize per-seed metrics into mean/std
            if per_seed_boundary_metrics:
                results['boundary_areas'].setdefault(dataset_name, {})[act_name] = {}
                # list of seeds in deterministic order
                seeds = sorted(per_seed_boundary_metrics.keys())
                results['boundary_areas'][dataset_name][act_name]['runs'] = [per_seed_boundary_metrics[s] for s in seeds]

                # compute summary stats for threshold and expected areas (class1)
                thresh_vals = np.array([per_seed_boundary_metrics[s]['threshold_area']['class1'] for s in seeds])
                expect_vals = np.array([per_seed_boundary_metrics[s]['expected_area']['class1'] for s in seeds])
                total_bbox_area = float((per_seed_boundary_metrics[seeds[0]]['bbox']['x_max'] - per_seed_boundary_metrics[seeds[0]]['bbox']['x_min']) * (per_seed_boundary_metrics[seeds[0]]['bbox']['y_max'] - per_seed_boundary_metrics[seeds[0]]['bbox']['y_min']))

                results['boundary_areas'][dataset_name][act_name]['summary'] = {
                    'threshold_area_mean': float(thresh_vals.mean()),
                    'threshold_area_std': float(thresh_vals.std()),
                    'expected_area_mean': float(expect_vals.mean()),
                    'expected_area_std': float(expect_vals.std()),
                    'total_bbox_area': total_bbox_area,
                    'threshold_area_mean_ratio': float(thresh_vals.mean() / total_bbox_area),
                    'expected_area_mean_ratio': float(expect_vals.mean() / total_bbox_area)
                }
            
            # Save intermediate results after each activation completes (for resume capability)
            try:
                with open(output_dir / 'synthetic_benchmark_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"⚠️  Could not save intermediate results: {e}")

    # End of dataset loop - post-processing starts here
    print("\n📈 Creating summary visualizations...")
    create_summary_plots(results, list(activation_functions.keys()), output_dir)
    
    print("📈 Creating training loss curves...")
    create_loss_curves_plot(results, list(activation_functions.keys()), output_dir)
    
    print("📈 Creating aggregated convergence plots...")
    create_aggregated_convergence_plots(results, list(activation_functions.keys()), output_dir)
    
    try:
        print("🧪 Running function fitting analysis (activation approximation capabilities)...")
        try:
            from .synthetic_function_fitting import run_function_fitting_analysis
        except Exception:
            run_function_fitting_analysis = None

        if run_function_fitting_analysis is not None:
            ff_dir = output_dir / 'function_fitting'
            ff_results = run_function_fitting_analysis(activation_functions, ff_dir, epochs=750, lr=0.01)
            results['function_fitting'] = ff_results
            ff_summary = {t: data['final_losses'] for t, data in ff_results.items()}
            if ff_summary:
                df_ff = pd.DataFrame(ff_summary).T
                plt.figure(figsize=(12, 8))
                sns.heatmap(df_ff, annot=False, cmap='magma_r', cbar_kws={'label': 'Final MSE'})
                plt.title('Function Fitting Performance (Lower MSE Better)')
                plt.xlabel('Activation Function')
                plt.ylabel('Target Function')
                plt.tight_layout()
                plt.savefig(output_dir / 'function_fitting_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print("⚠️  Skipping function fitting analysis: `synthetic_function_fitting` not found.")
    except Exception as e:
        print(f"⚠️  Function fitting analysis failed: {e}")

    print("📊 Performing statistical analysis...")
    statistical_results = calculate_statistical_analysis(results, list(activation_functions.keys()))
    create_statistical_plots(statistical_results, list(activation_functions.keys()), output_dir)
    # Also create boxplots per dataset with significance annotations
    try:
        create_accuracy_boxplots(results, statistical_results, list(activation_functions.keys()), output_dir)
    except Exception:
        pass
    
    results.update(statistical_results)
    
    print("📄 Generating comprehensive HTML report...")
    generate_comprehensive_html_report(results, list(activation_functions.keys()), output_dir)
    
    with open(output_dir / 'synthetic_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Also write boundary areas separately for convenient access if they were computed
    if any(results.get('boundary_areas', {})):
        with open(output_dir / 'decision_boundary_areas.json', 'w') as f:
            json.dump(results['boundary_areas'], f, indent=2)

    print("💾 Exporting summary results to CSV...")
    export_results_to_csv(results, output_dir)
    
    print(f"\n✅ Synthetic benchmark analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return results

# (The rest of the file remains the same: create_summary_plots, create_radar_chart, etc.)
# ...
# The plotting and utility functions from your previous correct version go here.
# I am omitting them for brevity, but they should be included in the final file.
# ...

def get_available_activations(results: Dict[str, Any], activation_names: List[str]) -> List[str]:
    """Filter activation names to only include those that have results for all datasets."""
    datasets = list(results['accuracy_scores'].keys())
    available_activations = []
    
    for act in activation_names:
        if all(act in results.get('accuracy_scores', {}).get(d, {}) and 
               act in results.get('isotropy_scores', {}).get(d, {}) and
               act in results.get('training_stability', {}).get(d, {}) for d in datasets):
            available_activations.append(act)
    
    if not available_activations:
        print("❌ Warning: No activation functions have complete results across all datasets.")
        print("   Available results per dataset:")
        for d in datasets:
            acc_acts = list(results.get('accuracy_scores', {}).get(d, {}).keys())
            print(f"   {d}: {acc_acts}")
    
    return available_activations

def create_summary_plots(results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create summary plots comparing all activation functions."""
    
    # Filter activation_names to only include those that have complete results
    activation_names = get_available_activations(results, activation_names)
    if not activation_names:
        return  # Skip plotting if no complete results
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    datasets = list(results['accuracy_scores'].keys())
    
    acc_data = np.array([[results['accuracy_scores'][d][a]['mean'] for a in activation_names] for d in datasets])
    row_min = acc_data.min(axis=1, keepdims=True)
    row_max = acc_data.max(axis=1, keepdims=True)
    acc_data_norm = (acc_data - row_min) / (row_max - row_min + 1e-9)
    
    sns.heatmap(acc_data_norm, xticklabels=activation_names, yticklabels=datasets, annot=acc_data, fmt='.3f', cmap='viridis', ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Accuracy Scores (Per-Row Scaled)', fontweight='bold')
    axes[0].set_xlabel('Activation Function')
    
    iso_data = np.array([[results['isotropy_scores'][d][a]['mean'] for a in activation_names] for d in datasets])
    row_min = iso_data.min(axis=1, keepdims=True)
    row_max = iso_data.max(axis=1, keepdims=True)
    iso_data_norm = (iso_data - row_min) / (row_max - row_min + 1e-9)
    
    sns.heatmap(iso_data_norm, xticklabels=activation_names, yticklabels=datasets, annot=iso_data, fmt='.3f', cmap='plasma', ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Isotropy Scores (Per-Row Scaled)', fontweight='bold')
    axes[1].set_xlabel('Activation Function')
    
    stab_data = np.array([[results['training_stability'][d][a] for a in activation_names] for d in datasets])
    row_min = stab_data.min(axis=1, keepdims=True)
    row_max = stab_data.max(axis=1, keepdims=True)
    stab_data_norm = (stab_data - row_min) / (row_max - row_min + 1e-9)
    
    sns.heatmap(stab_data_norm, xticklabels=activation_names, yticklabels=datasets, annot=stab_data, fmt='.3f', cmap='coolwarm', ax=axes[2], vmin=0, vmax=1)
    axes[2].set_title('Training Stability (1 - StdDev)', fontweight='bold')
    axes[2].set_xlabel('Activation Function')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_benchmark_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    create_radar_chart(results, activation_names, output_dir)


def create_radar_chart(results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create radar chart comparing activation functions across all metrics."""
    
    # Filter activation_names to only include those that have complete results
    activation_names = get_available_activations(results, activation_names)
    if not activation_names:
        return  # Skip plotting if no complete results
    
    from math import pi
    
    datasets = list(results['accuracy_scores'].keys())
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    categories = ['Accuracy', 'Isotropy', 'Stability']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Prefer darker, high-contrast palettes so lines remain visible on the
    # seaborn darkgrid background. Use tab10 for up to 10 activations, fall
    # back to tab20 for larger sets.
    if len(activation_names) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(activation_names)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(activation_names)))
    
    # Use linear scale for radar, clipping small values to a floor of 0.75 to
    # focus the plot on the upper range of performance values.
    floor = 0.75
    for i, act_name in enumerate(activation_names):
        avg_accuracy = np.mean([results['accuracy_scores'][d][act_name]['mean'] for d in datasets])
        avg_isotropy = np.mean([results['isotropy_scores'][d][act_name]['mean'] for d in datasets])
        avg_stability = np.mean([results['training_stability'][d][act_name] for d in datasets])
        values = [avg_accuracy, avg_isotropy, avg_stability]
        clipped = [max(v, floor) for v in values]
        vals = np.array(clipped)
        vals = np.concatenate([vals, [vals[0]]])

        ax.plot(angles, vals, 'o-', linewidth=2, label=act_name, color=colors[i])
        ax.fill(angles, vals, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    # Linear radial axis with floor at 0.75; draw circular ticks at 0.75 and 1.0
    ax.set_ylim(floor, 1.0)
    ax.set_yticks([0.75, 1.0])
    ax.set_yticklabels(['0.75', '1.00'], fontsize=10)
    # Ensure the radial grid is shown as concentric circles
    ax.grid(True)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Activation Function Performance Comparison\n(Averaged Across All Synthetic Datasets)', size=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_loss_curves_plot(results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create comprehensive loss curves visualization."""
    
    # Filter activation_names to only include those that have complete results
    activation_names = get_available_activations(results, activation_names)
    if not activation_names:
        return  # Skip plotting if no complete results
    
    datasets = list(results['training_histories'].keys())
    n_datasets = len(datasets)
    # Do not share y-axis between dataset columns; we'll set per-dataset ranges.
    fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 10), sharey=False)
    if n_datasets == 1: axes = axes.reshape(-1, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(activation_names)))
    
    for i, dataset in enumerate(datasets):
        # First collect train/val histories for all activations to compute a
        # sensible per-dataset y-range (shared between train and val panels).
        train_curves = []
        val_curves = []
        max_len = 0
        for j, act_name in enumerate(activation_names):
            # Skip if activation doesn't have training history for this dataset
            if act_name not in results['training_histories'][dataset]:
                continue
                
            entry = results['training_histories'][dataset][act_name]
            if isinstance(entry, dict) and 'average' in entry:
                history = entry['average']
            elif isinstance(entry, dict) and 'train_loss' in entry and 'val_loss' in entry:
                history = entry
            elif isinstance(entry, dict) and 'runs' in entry:
                runs = entry['runs']
                history = {}
                history['train_loss'] = np.mean([np.array(r['train_loss']) for r in runs], axis=0).tolist()
                history['val_loss'] = np.mean([np.array(r['val_loss']) for r in runs], axis=0).tolist()
            else:
                raise ValueError(f"Unrecognized history format for {dataset}/{act_name}")

            train_arr = np.array(history['train_loss'])
            val_arr = np.array(history['val_loss'])
            train_curves.append(train_arr)
            val_curves.append(val_arr)
            max_len = max(max_len, len(train_arr), len(val_arr))

        # Combine to find global min/max for this dataset (avoid zeros for log scale)
        all_vals = np.concatenate([arr.flatten() for arr in (train_curves + val_curves)])
        eps = 1e-12
        ymin = max(eps, float(np.nanmin(all_vals[all_vals > 0])) if np.any(all_vals > 0) else eps)
        ymax = float(np.nanmax(all_vals)) if np.any(~np.isnan(all_vals)) else (ymin * 10)

        ax_train = axes[0, i]
        for j, arr in enumerate(train_curves):
            epochs = range(len(arr))
            ax_train.plot(epochs, arr, color=colors[j], label=activation_names[j], linewidth=2)

        ax_train.set_title(f'{dataset.title()} - Training Loss', fontweight='bold')
        ax_train.set_xlabel('Epoch')
        if i == 0:
            ax_train.set_ylabel('Loss')
        ax_train.legend()
        ax_train.grid(True, alpha=0.3)
        ax_train.set_yscale('log')
        ax_train.set_ylim(ymin * 0.8, ymax * 1.2)

        ax_val = axes[1, i]
        for j, arr in enumerate(val_curves):
            epochs = range(len(arr))
            ax_val.plot(epochs, arr, color=colors[j], label=activation_names[j], linewidth=2)

        ax_val.set_title(f'{dataset.title()} - Validation Loss', fontweight='bold')
        ax_val.set_xlabel('Epoch')
        if i == 0:
            ax_val.set_ylabel('Loss')
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        ax_val.set_yscale('log')
        ax_val.set_ylim(ymin * 0.8, ymax * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_aggregated_convergence_plots(results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create plots showing the average convergence across all datasets for each activation function."""
    
    # Filter activation_names to only include those that have complete results
    activation_names = get_available_activations(results, activation_names)
    if not activation_names:
        return  # Skip plotting if no complete results
    
    datasets = list(results['training_histories'].keys())
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(activation_names)))

    for i, act_name in enumerate(activation_names):
        def get_avg_curve(dataset, key):
            # Skip if activation doesn't have training history for this dataset
            if act_name not in results['training_histories'][dataset]:
                return None
                
            entry = results['training_histories'][dataset][act_name]
            if isinstance(entry, dict) and 'average' in entry:
                return np.array(entry['average'][key])
            elif isinstance(entry, dict) and key in entry:
                return np.array(entry[key])
            elif isinstance(entry, dict) and 'runs' in entry:
                runs = entry['runs']
                return np.mean([np.array(r[key]) for r in runs], axis=0)
            else:
                raise ValueError(f"Unrecognized history format for {dataset}/{act_name}")

        train_curves = [get_avg_curve(d, 'train_loss') for d in datasets]
        val_curves = [get_avg_curve(d, 'val_loss') for d in datasets]
        
        # Filter out None values
        train_curves = [c for c in train_curves if c is not None]
        val_curves = [c for c in val_curves if c is not None]
        
        if not train_curves or not val_curves:
            continue
            
        train_curves = np.array(train_curves)
        val_curves = np.array(val_curves)
        
        mean_train_loss, std_train_loss = np.mean(train_curves, axis=0), np.std(train_curves, axis=0)
        mean_val_loss, std_val_loss = np.mean(val_curves, axis=0), np.std(val_curves, axis=0)
        epochs = range(len(mean_train_loss))
        
        axes[0].plot(epochs, mean_train_loss, color=colors[i], label=act_name, linewidth=2)
        axes[0].fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color=colors[i], alpha=0.2)
        
        axes[1].plot(epochs, mean_val_loss, color=colors[i], label=act_name, linewidth=2)
    # 95% CI shading (approximate using std/sqrt(Ndatasets))
    ci_factor = 1.96 / np.sqrt(max(1, len(datasets)))
    axes[1].fill_between(epochs, mean_val_loss - std_val_loss * ci_factor, mean_val_loss + std_val_loss * ci_factor, color=colors[i], alpha=0.2)

    axes[0].set_title('Aggregated Training Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Aggregated Validation Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Average Convergence Across All Datasets (Shaded area = ±1 std. dev.)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'aggregated_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary predictions.

    probs: predicted probability for class 1
    labels: ground-truth binary labels (0/1)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1]) if i < n_bins - 1 else (probs >= bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def calculate_statistical_analysis(results: Dict[str, Any], activation_names: List[str]) -> Dict[str, Any]:
    """
    Perform statistical analysis (t-tests, effect sizes) on results.
    Handles cases where some activations may not have results for all datasets.
    """
    from scipy import stats

    pairwise_results = {}
    effect_sizes = {}

    # Filter activation_names to only include those that have complete results
    activation_names = get_available_activations(results, activation_names)
    if not activation_names:
        return {'pairwise_results': {}, 'effect_sizes': {}}

    # For each dataset, perform paired tests across repeats between activations
    datasets = list(results['accuracy_scores'].keys())
    for i, act1 in enumerate(activation_names):
        for j, act2 in enumerate(activation_names):
            if i < j:
                p_values = []
                deltas = []
                for d in datasets:
                    # Check if both activations have results for this dataset
                    if act1 not in results['accuracy_scores'][d] or act2 not in results['accuracy_scores'][d]:
                        continue
                        
                    vals1 = np.array(results['accuracy_scores'][d][act1]['values'])
                    vals2 = np.array(results['accuracy_scores'][d][act2]['values'])
                    minlen = min(len(vals1), len(vals2))
                    if minlen < 2:
                        continue
                    # paired wilcoxon where applicable
                    try:
                        stat, p = stats.wilcoxon(vals1[:minlen], vals2[:minlen])
                    except Exception:
                        # fallback to mannwhitneyu
                        stat, p = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    p_values.append(p)
                    # Cliff's delta (non-parametric effect size)
                    greater = 0
                    lesser = 0
                    for a in vals1[:minlen]:
                        for b in vals2[:minlen]:
                            if a > b: greater += 1
                            elif a < b: lesser += 1
                    n = minlen * minlen
                    delta = (greater - lesser) / n if n > 0 else 0
                    deltas.append(delta)

                # aggregate across datasets by taking median p and mean delta
                pair_key = f"{act1}_vs_{act2}"
                if p_values:
                    median_p = float(np.median(p_values))
                    mean_delta = float(np.mean(deltas))
                    pairwise_results[pair_key] = {'p_value_median_across_datasets': median_p, 'significant': bool(median_p < 0.05)}
                    effect_sizes[pair_key] = float(abs(mean_delta))
                else:
                    pairwise_results[pair_key] = {'p_value_median_across_datasets': None, 'significant': False}
                    effect_sizes[pair_key] = 0.0

    return {'pairwise_results': pairwise_results, 'effect_sizes': effect_sizes}


def create_statistical_plots(statistical_results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create statistical significance and effect size visualizations."""
    
    pairwise_results = statistical_results['pairwise_results']
    effect_sizes = statistical_results['effect_sizes']
    
    n_acts = len(activation_names)
    significance_matrix = np.ones((n_acts, n_acts))
    p_value_matrix = np.ones((n_acts, n_acts))
    
    # Collect p-values and apply Bonferroni correction across all pairs
    collected = []
    keys = []
    for pair_key, result in pairwise_results.items():
        p = result.get('p_value_median_across_datasets') if 'p_value_median_across_datasets' in result else result.get('p_value')
        if p is not None:
            collected.append(p)
        keys.append(pair_key)

    m = max(1, len(collected))
    corrected_map = {}
    idx = 0
    for pair_key in keys:
        result = pairwise_results[pair_key]
        p = result.get('p_value_median_across_datasets') if 'p_value_median_across_datasets' in result else result.get('p_value')
        if p is None:
            corrected_map[pair_key] = None
        else:
            corrected_map[pair_key] = min(1.0, p * m)
        idx += 1

    for pair_key, result in pairwise_results.items():
        act1, act2 = pair_key.split('_vs_')
        i, j = activation_names.index(act1), activation_names.index(act2)
        corrected_p = corrected_map.get(pair_key)
        significance = bool(corrected_p is not None and corrected_p < 0.05)
        significance_matrix[i, j] = significance_matrix[j, i] = 0 if significance else 1
        p_value_matrix[i, j] = p_value_matrix[j, i] = (corrected_p if corrected_p is not None else 1.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(significance_matrix, xticklabels=activation_names, yticklabels=activation_names, annot=p_value_matrix, fmt='.3f', cmap='RdYlBu_r', center=0.5, ax=ax1, cbar_kws={'label': 'Significant (0) vs Non-significant (1)'})
    ax1.set_title('Statistical Significance Matrix (p-values)\nLower is more significant', fontweight='bold')
    
    if effect_sizes:
        pairs, values = list(effect_sizes.keys()), list(effect_sizes.values())
        y_pos = np.arange(len(pairs))
        ax2.barh(y_pos, values, color='skyblue', edgecolor='navy')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([p.replace('_vs_', ' vs ') for p in pairs])
        ax2.set_xlabel("Effect Size (|Cohen's d|)")
        ax2.set_title('Effect Sizes Between Activations', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0.2, color='green', linestyle='--', alpha=0.7, label='Small (0.2)')
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium (0.5)')
        ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Large (0.8)')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_accuracy_boxplots(results: Dict[str, Any], statistical_results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    """Create boxplots of final accuracies per activation and annotate significance."""
    datasets = list(results['accuracy_scores'].keys())
    pairwise = statistical_results.get('pairwise_results', {})

    for dataset in datasets:
        data = []
        labels = []
        for act in activation_names:
            vals = results['accuracy_scores'][dataset][act]['values']
            data.append(vals)
            labels.append(act)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=data)
        sns.stripplot(data=data, color='black', alpha=0.6, jitter=True)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Per-run Accuracies on {dataset.title()}')

        # Overlay mean markers
        means = [np.mean(d) for d in data]
        for i, m in enumerate(means):
            ax.scatter(i, m, color='red', marker='D', s=50, zorder=10)

        # Annotate significance for top pairs (simple if available)
        y_max = max([max(d) if len(d) else 0 for d in data]) if data else 1.0
        y_start = y_max + 0.02
        step = 0.03
        k = 0
        for i in range(len(activation_names)):
            for j in range(i+1, len(activation_names)):
                key = f"{activation_names[i]}_vs_{activation_names[j]}"
                corrected_p = None
                if key in pairwise:
                    corrected_p = pairwise[key].get('p_value_median_across_datasets') if 'p_value_median_across_datasets' in pairwise[key] else pairwise[key].get('p_value')
                if corrected_p is not None and corrected_p < 0.05:
                    ax.plot([i, j], [y_start + k*step, y_start + k*step], color='k')
                    ax.text((i+j)/2, y_start + k*step + 0.005, f'p={corrected_p:.3f}', ha='center')
                    k += 1

        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_accuracy_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_comprehensive_html_report(results: Dict[str, Any], activation_names: List[str], output_dir: Path):
    # Minimal HTML report generator: lists images and key CSV outputs
    html_path = output_dir / 'comprehensive_report.html'
    lines = []
    lines.append('<html><head><meta charset="utf-8"><title>Synthetic Benchmark Report</title></head><body>')
    lines.append(f'<h1>Synthetic Benchmark Report</h1>')
    lines.append(f'<p>Generated: {datetime.now().isoformat()}</p>')

    # Manifest
    if 'manifest' in results:
        lines.append('<h2>Run Manifest</h2>')
        lines.append('<pre>')
        lines.append(json.dumps(results['manifest'], indent=2))
        lines.append('</pre>')

    # Images (exclude per-seed decision boundary PNGs and dataset overview PNGs)
    images = sorted([p.name for p in output_dir.glob('*.png') if ('_boundary.png' not in p.name and '_dataset.png' not in p.name)])
    if images:
        lines.append('<h2>Plots</h2>')
        for img in images:
            lines.append(f'<div style="margin:12px 0;"><h3>{img}</h3><img src="{img}" style="max-width:100%; height:auto;"/></div>')

    # CSVs
    csvs = sorted([p.name for p in output_dir.glob('*.csv')])
    if csvs:
        lines.append('<h2>CSV Outputs</h2><ul>')
        for c in csvs:
            lines.append(f'<li><a href="{c}">{c}</a></li>')
        lines.append('</ul>')

    lines.append('</body></html>')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def export_results_to_csv(results: Dict[str, Any], output_dir: Path):
    # Produce tidy CSVs for summary and per-run data
    # 1) Summary table: dataset, activation, mean_accuracy, std_accuracy, mean_isotropy, std_isotropy
    rows = []
    for dataset in results.get('accuracy_scores', {}):
        for act in results['accuracy_scores'][dataset]:
            acc = results['accuracy_scores'][dataset][act]
            iso = results['isotropy_scores'][dataset][act]
            stab = results['training_stability'][dataset][act]
            rows.append({
                'dataset': dataset,
                'activation': act,
                'accuracy_mean': acc.get('mean'),
                'accuracy_std': acc.get('std'),
                'isotropy_mean': iso.get('mean'),
                'isotropy_std': iso.get('std'),
                'stability': stab
            })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_dir / 'synthetic_benchmark_summary.csv', index=False)

    # 2) Per-run accuracies
    per_rows = []
    for dataset in results.get('accuracy_scores', {}):
        for act in results['accuracy_scores'][dataset]:
            values = results['accuracy_scores'][dataset][act].get('values', [])
            for i, v in enumerate(values):
                per_rows.append({'dataset': dataset, 'activation': act, 'seed_index': int(i), 'accuracy': float(v)})

    df_per = pd.DataFrame(per_rows)
    df_per.to_csv(output_dir / 'synthetic_benchmark_per_run_accuracies.csv', index=False)

    # 3) Boundary areas per-run and summary
    boundary_rows = []
    ba = results.get('boundary_areas', {})
    for dataset, acts in ba.items():
        for act, data in acts.items():
            runs = data.get('runs', [])
            for run in runs:
                seed = run.get('seed')
                runtime = run.get('runtime_seconds')
                thr = run.get('threshold_area', {}).get('class1')
                exp = run.get('expected_area', {}).get('class1')
                perf = run.get('perf', {})
                ece = run.get('ece')
                boundary_rows.append({
                    'dataset': dataset,
                    'activation': act,
                    'seed': seed,
                    'runtime_seconds': runtime,
                    'threshold_area_class1': thr,
                    'expected_area_class1': exp,
                    'ece': ece,
                    'cpu_percent': perf.get('cpu_percent') if isinstance(perf, dict) else None,
                    'memory_mb': perf.get('memory_mb') if isinstance(perf, dict) else None,
                    'cuda_max_mem_mb': perf.get('cuda_max_mem_mb') if isinstance(perf, dict) else None
                })

    if boundary_rows:
        pd.DataFrame(boundary_rows).to_csv(output_dir / 'decision_boundary_areas_per_run.csv', index=False)

    # Export captured package list for reproducibility as CSV (if available)
    pkg_info = results.get('manifest', {}).get('packages')
    try:
        if pkg_info:
            pd.DataFrame([{'package': k, 'version': v} for k, v in pkg_info.items()]).to_csv(output_dir / 'packages.csv', index=False)
    except Exception:
        pass

    # 4) Save training histories into CSV files
    # IMPORTANT: Write activation-specific files to prevent overwrites from sequential runs
    # Also write a proper HP-aware longform CSV with explicit lr/wd/seed columns
    
    history_rows = []
    activations_in_results = set()
    th = results.get('training_histories', {})
    
    # Get HP grid info from manifest if available
    manifest = results.get('manifest', {})
    lr_list = manifest.get('lr_grid', [0.01])  # fallback
    wd_list = manifest.get('wd_grid', [0.0])   # fallback
    seeds_used = manifest.get('seeds_used', [42, 43, 44])  # fallback
    
    for dataset, acts in th.items():
        for act, entry in acts.items():
            activations_in_results.add(act)
            # entry may contain 'runs' which is a list of dicts with keys train_loss etc.
            if isinstance(entry, dict) and 'runs' in entry:
                runs = entry['runs']
                # Decode run index to (lr, wd, seed)
                # runs are ordered: for each LR, for each WD, for each seed
                n_wds = len(wd_list)
                n_seeds = len(seeds_used)
                
                for run_idx, run in enumerate(runs):
                    # Decode run_idx: run_idx = lr_idx * (n_wds * n_seeds) + wd_idx * n_seeds + seed_idx
                    lr_idx = run_idx // (n_wds * n_seeds)
                    remainder = run_idx % (n_wds * n_seeds)
                    wd_idx = remainder // n_seeds
                    seed_idx = remainder % n_seeds
                    
                    # Handle case where we have fewer runs than expected (e.g., partial run)
                    if lr_idx >= len(lr_list):
                        lr_val = lr_list[-1] if lr_list else 0.01
                    else:
                        lr_val = lr_list[lr_idx]
                    if wd_idx >= len(wd_list):
                        wd_val = wd_list[-1] if wd_list else 0.0
                    else:
                        wd_val = wd_list[wd_idx]
                    seed_val = seeds_used[seed_idx] if seed_idx < len(seeds_used) else 42 + seed_idx
                    
                    n_epochs = len(run.get('train_loss', []))
                    for epoch in range(n_epochs):
                        history_rows.append({
                            'dataset': dataset,
                            'activation': act,
                            'epoch': epoch,
                            'train_loss': run['train_loss'][epoch],
                            'val_loss': run['val_loss'][epoch],
                            'train_accuracy': run['train_accuracy'][epoch],
                            'val_accuracy': run['val_accuracy'][epoch],
                            'lr': lr_val,
                            'weight_decay': wd_val,
                            'seed_index': seed_val
                        })
            elif isinstance(entry, dict) and 'average' in entry:
                avg = entry['average']
                n_epochs = len(avg.get('train_loss', []))
                for epoch in range(n_epochs):
                    history_rows.append({
                        'dataset': dataset,
                        'activation': act,
                        'epoch': epoch,
                        'train_loss': avg['train_loss'][epoch],
                        'val_loss': avg['val_loss'][epoch],
                        'train_accuracy': np.nan,
                        'val_accuracy': np.nan,
                        'lr': np.nan,
                        'weight_decay': np.nan,
                        'seed_index': 'average'
                    })

    if history_rows:
        df_history = pd.DataFrame(history_rows)
        
        # Ensure column order for consistency
        col_order = ['dataset', 'activation', 'epoch', 'train_loss', 'val_loss', 
                     'train_accuracy', 'val_accuracy', 'lr', 'weight_decay', 'seed_index']
        df_history = df_history[col_order]
        
        # Write activation-specific file (safe - won't be overwritten by other activation runs)
        act_names_safe = '_'.join(sorted(activations_in_results)).lower().replace(' ', '_')[:50]
        activation_specific_file = output_dir / f'training_histories_{act_names_safe}.csv'
        df_history.to_csv(activation_specific_file, index=False)
        print(f"   Saved activation-specific histories to: {activation_specific_file.name}")
        
        # Also write to toy_models master file if it exists (proper upsert)
        thrust0_master = Path('mechanistic_interpretability/toy_models/training_histories.csv')
        if thrust0_master.exists():
            print(f"   Updating toy_models master file...")
            existing_df = pd.read_csv(thrust0_master)
            
            # Remove rows for (activation, lr, wd) combos we're about to add
            # This prevents duplicates when re-running specific combos
            for act in activations_in_results:
                act_new = df_history[df_history['activation'] == act]
                for lr_val in act_new['lr'].dropna().unique():
                    for wd_val in act_new['weight_decay'].dropna().unique():
                        mask = ~((existing_df['activation'] == act) & 
                                (existing_df['lr'] == lr_val) & 
                                (existing_df['weight_decay'] == wd_val))
                        existing_df = existing_df[mask]
            
            combined_df = pd.concat([existing_df, df_history], ignore_index=True)
            combined_df.to_csv(thrust0_master, index=False)
            print(f"   ✅ Updated {thrust0_master}: {len(combined_df):,} total rows")
        
        # Also write/append to local combined file for this run
        combined_file = output_dir / 'training_histories_longform.csv'
        df_history.to_csv(combined_file, index=False)
        print(f"   Saved local combined file: {combined_file.name}")


if __name__ == "__main__":
    class SoftCap(nn.Module):
        def __init__(self, alpha=1.0, beta=1.0):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        def forward(self, x):
            return self.alpha * torch.log(1 + torch.exp(self.beta * x))
    
    activations = {
        'ReLU': nn.ReLU(), 'GELU': nn.GELU(), 'SiLU': nn.SiLU(),
        'Tanh': nn.Tanh(), 'SoftCap': SoftCap(),
    }
    
    output_path = Path("synthetic_benchmark_results_example")
    
    results = run_synthetic_benchmarks(
        activation_functions=activations,
        output_dir=output_path,
        n_samples=1000,
        epochs=300,
        device='auto'
    )