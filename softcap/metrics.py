"""
SoftCap Enhanced Metrics Module

Implements all literature-informed metrics for comprehensive activation function analysis:
- Isotropy metrics (RII, DDS, SBD)
- Sparsity efficiency analysis
- Gradient health assessment
- Computational efficiency metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports for new metrics (Papers 1 & 3)
try:
    from isoscore import IsoScore
    ISOSCORE_AVAILABLE = True
except ImportError:
    ISOSCORE_AVAILABLE = False
    warnings.warn("IsoScore package not available. Install with: pip install isoscore")

try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available for advanced metrics")


class IsotropyAnalyzer:
    """
    Analyzes isotropy properties of neural network representations.
    
    Based on George Bird's isotropic deep learning theory, this class implements
    metrics to detect representational collapse and anisotropic biases.
    """
    
    def __init__(self):
        self.name = "IsotropyAnalyzer"
    
    def compute_isotropy_index(self, representations: torch.Tensor) -> float:
        """
        Compute Representational Isotropy Index (RII).
        
        Measures angular uniformity in representation space.
        Higher values indicate more isotropic (uniform) distributions.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Isotropy index (higher = more isotropic)
        """
        # Normalize representations to unit sphere
        normalized = F.normalize(representations, dim=-1)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(normalized, normalized.t())
        
        # Remove diagonal (self-similarities)
        n = similarities.size(0)
        mask = torch.eye(n, device=similarities.device).bool()
        similarities = similarities[~mask]
        
        # Measure uniformity - lower variance in similarities indicates higher isotropy
        isotropy_score = 1.0 / (1.0 + similarities.var().item())
        
        return isotropy_score
    
    def compute_directional_diversity_score(self, representations: torch.Tensor) -> float:
        """
        Compute Directional Diversity Score (DDS).
        
        Measures how well representations span the available space dimensions.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Directional diversity score (higher = more diverse directions)
        """
        # Normalize representations
        normalized = F.normalize(representations, dim=-1)
        
        # Compute covariance matrix
        cov_matrix = torch.cov(normalized.t())
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.clamp(eigenvals, min=1e-8)  # Avoid log(0)
        
        # Shannon entropy of eigenvalue distribution
        eigenvals_normalized = eigenvals / eigenvals.sum()
        entropy = -torch.sum(eigenvals_normalized * torch.log(eigenvals_normalized))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(eigenvals))
        dds = entropy.item() / max_entropy if max_entropy > 0 else 0.0
        
        return dds
    
    def compute_spherical_bias_detection(self, representations: torch.Tensor) -> float:
        """
        Compute Spherical Bias Detection (SBD) metric.
        
        Detects clustering or preferential directions in representation space.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Spherical bias score (lower = more isotropic)
        """
        # Normalize to unit sphere
        normalized = F.normalize(representations, dim=-1)
        
        # Use K-means to detect clustering
        n_samples = normalized.size(0)
        k = min(8, n_samples // 10)  # Adaptive number of clusters
        
        if k < 2:
            return 0.0
        
        try:
            # Convert to numpy for sklearn
            data_np = normalized.detach().cpu().numpy()
            
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_np)
            
            # Silhouette score - higher means more clustered (less isotropic)
            silhouette = silhouette_score(data_np, labels)
            
            # Convert to bias score (lower = more isotropic)
            bias_score = max(0.0, silhouette)
            
        except Exception:
            bias_score = 0.0
        
        return bias_score
    
    def compute_geometric_isotropy_score(self, representations: torch.Tensor) -> float:
        """
        Compute geometric isotropy score based on polytope analysis.
        
        Inspired by "Softplus & Convex Polytopes" and geometric deep learning literature.
        Analyzes the geometric structure of representation polytopes.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Geometric isotropy score (higher = more isotropic geometry)
        """
        if representations.size(0) < 4:  # Need at least 4 points for geometric analysis
            return 0.0
        
        # Normalize to unit sphere
        normalized = F.normalize(representations, dim=-1)
        
        # Compute convex hull volume approximation
        try:
            # Sample random subsets for volume estimation
            n_samples = min(100, normalized.size(0))
            indices = torch.randperm(normalized.size(0))[:n_samples]
            sample_points = normalized[indices]
            
            # Compute pairwise distances
            distances = torch.cdist(sample_points, sample_points)
            
            # Geometric isotropy: uniform distribution should have consistent distances
            distance_variance = distances[distances > 0].var().item()
            geometric_score = 1.0 / (1.0 + distance_variance)
            
        except Exception:
            geometric_score = 0.0
        
        return geometric_score
    
    def compute_representation_polytope_metrics(self, representations: torch.Tensor) -> Dict[str, float]:
        """
        Analyze the polytope structure of representations.
        
        Based on insights from quantization and polytope analysis literature.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Dictionary of polytope-based metrics
        """
        if representations.size(0) < 3:
            return {'polytope_volume': 0.0, 'polytope_regularity': 0.0, 'vertex_distribution': 0.0}
        
        # Ensure all tensors are on the same device as input
        device = representations.device
        
        # Normalize representations
        normalized = F.normalize(representations, dim=-1)
        
        # Compute centroid
        centroid = normalized.mean(dim=0)
        
        # Distances from centroid
        centroid_distances = torch.norm(normalized - centroid.unsqueeze(0), dim=1)
        
        # Polytope volume approximation (using distance variance)
        distance_std = centroid_distances.std().item()
        distance_mean = centroid_distances.mean().item()
        polytope_regularity = 1.0 / (1.0 + distance_std / (distance_mean + 1e-8))
        
        # Vertex distribution uniformity
        # Check how evenly distributed the points are on the sphere
        pairwise_distances = torch.cdist(normalized, normalized)
        # Ensure eye tensor is on the same device
        eye_tensor = torch.eye(pairwise_distances.size(0), device=device) * 1e6
        min_distances = torch.topk(pairwise_distances + eye_tensor, 
                                  k=2, dim=1, largest=False)[0][:, 1]  # Exclude self-distance
        vertex_distribution = 1.0 / (1.0 + min_distances.std().item())
        
        return {
            'polytope_volume': polytope_regularity,
            'polytope_regularity': polytope_regularity,
            'vertex_distribution': vertex_distribution
        }
    
    def analyze_representations(self, representations: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive isotropy analysis of representations.
        
        Args:
            representations: Tensor of shape (n_samples, n_features)
        
        Returns:
            Dictionary containing isotropy metrics
        """
        if representations.numel() == 0:
            return {
                'isotropy_index': 0.0,
                'directional_diversity': 0.0,
                'spherical_bias': 1.0,
                'geometric_isotropy': 0.0,
                'overall_isotropy': 0.0
            }
        
        # Ensure 2D tensor
        if representations.dim() > 2:
            representations = representations.view(-1, representations.size(-1))
        
        # Compute individual metrics
        isotropy_index = self.compute_isotropy_index(representations)
        directional_diversity = self.compute_directional_diversity_score(representations)
        spherical_bias = self.compute_spherical_bias_detection(representations)
        geometric_isotropy = self.compute_geometric_isotropy_score(representations)
        
        # Polytope analysis
        polytope_metrics = self.compute_representation_polytope_metrics(representations)
        
        # Overall isotropy score (higher = more isotropic)
        overall_isotropy = (isotropy_index + directional_diversity + (1.0 - spherical_bias) + geometric_isotropy) / 4.0
        
        result = {
            'isotropy_index': isotropy_index,
            'directional_diversity': directional_diversity,
            'spherical_bias': spherical_bias,
            'geometric_isotropy': geometric_isotropy,
            'overall_isotropy': overall_isotropy,
            # Legacy key mappings for backward compatibility
            'discretization_score': directional_diversity,
            'symmetry_bias': spherical_bias
        }
        
        # Add polytope metrics
        result.update(polytope_metrics)
        
        return result


class SparsityAnalyzer:
    """
    Analyzes sparsity properties and computational efficiency.
    
    Based on literature insights about activation sparsity in modern networks,
    particularly focusing on SoftCap's predictable sparsity patterns.
    """
    
    def __init__(self):
        self.name = "SparsityAnalyzer"
    
    def compute_sparsity_ratio(self, activations: torch.Tensor, threshold: float = 1e-6) -> float:
        """
        Compute the sparsity ratio of activations.
        
        Args:
            activations: Tensor of activation values
            threshold: Threshold below which values are considered zero
        
        Returns:
            Sparsity ratio (fraction of near-zero activations)
        """
        if activations.numel() == 0:
            return 0.0
        
        near_zero = torch.abs(activations) < threshold
        sparsity_ratio = near_zero.float().mean().item()
        
        return sparsity_ratio
    
    def compute_intrinsic_sparsity(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        Compute intrinsic sparsity metrics inspired by ProSparse research.
        
        Args:
            activations: Tensor of activation values
        
        Returns:
            Dictionary of intrinsic sparsity metrics
        """
        if activations.numel() == 0:
            return {'intrinsic_sparsity': 0.0, 'effective_sparsity': 0.0, 'sparsity_concentration': 0.0}
        
        # Flatten activations
        flat_acts = activations.view(-1)
        
        # Intrinsic sparsity (based on distribution shape)
        abs_acts = torch.abs(flat_acts)
        mean_abs = abs_acts.mean()
        std_abs = abs_acts.std()
        
        # Higher ratio indicates more concentrated around zero
        intrinsic_sparsity = (std_abs / (mean_abs + 1e-8)).item()
        
        # Effective sparsity (entropy-based)
        # Normalize to probability distribution
        abs_acts_norm = abs_acts / (abs_acts.sum() + 1e-8)
        entropy = -torch.sum(abs_acts_norm * torch.log(abs_acts_norm + 1e-8))
        max_entropy = np.log(len(abs_acts_norm))
        effective_sparsity = 1.0 - (entropy.item() / max_entropy) if max_entropy > 0 else 0.0
        
        # Sparsity concentration (how concentrated the non-zero values are)
        non_zero_mask = abs_acts > 1e-6
        if non_zero_mask.sum() > 0:
            non_zero_vals = abs_acts[non_zero_mask]
            concentration = (non_zero_vals.max() / (non_zero_vals.mean() + 1e-8)).item()
        else:
            concentration = 0.0
        
        return {
            'intrinsic_sparsity': intrinsic_sparsity,
            'effective_sparsity': effective_sparsity,
            'sparsity_concentration': concentration
        }
    
    def predict_sparsity_without_training(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """
        Predict activation sparsity without full training (SparseInfer approach).
        
        Args:
            model: Neural network model
            sample_input: Sample input for forward pass
        
        Returns:
            Dictionary of predicted sparsity metrics
        """
        model.eval()
        activations = {}
        
        def capture_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU)):
                hooks.append(module.register_forward_hook(capture_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze captured activations
        sparsity_predictions = {}
        for name, acts in activations.items():
            sparsity_ratio = self.compute_sparsity_ratio(acts)
            intrinsic_metrics = self.compute_intrinsic_sparsity(acts)
            
            sparsity_predictions[f'{name}_sparsity'] = sparsity_ratio
            sparsity_predictions[f'{name}_intrinsic'] = intrinsic_metrics['intrinsic_sparsity']
        
        # Overall prediction
        if sparsity_predictions:
            avg_sparsity = np.mean([v for k, v in sparsity_predictions.items() if 'sparsity' in k])
            sparsity_predictions['predicted_overall_sparsity'] = avg_sparsity
        else:
            sparsity_predictions['predicted_overall_sparsity'] = 0.0
        
        return sparsity_predictions
    
    def compute_sparsity_efficiency(self, activations: torch.Tensor, inputs: torch.Tensor) -> float:
        """
        Measure how well sparsity patterns can be predicted for optimization.
        
        Args:
            activations: Output activations
            inputs: Input data
        
        Returns:
            Sparsity prediction accuracy
        """
        if activations.numel() == 0 or inputs.numel() == 0:
            return 0.0
        
        try:
            # Flatten tensors
            acts_flat = activations.view(activations.size(0), -1)
            inputs_flat = inputs.view(inputs.size(0), -1)
            
            # Create sparsity labels (binary: sparse or not)
            sparsity_threshold = 1e-6
            sparse_labels = (torch.abs(acts_flat).mean(dim=1) < sparsity_threshold).float()
            
            # Simple logistic regression to predict sparsity from inputs
            if sparse_labels.std() > 0:  # Ensure we have both classes
                X = inputs_flat.detach().cpu().numpy()
                y = sparse_labels.detach().cpu().numpy()
                
                clf = LogisticRegression(random_state=42, max_iter=100)
                clf.fit(X, y)
                accuracy = clf.score(X, y)
            else:
                accuracy = 0.5  # Random baseline
                
        except Exception:
            accuracy = 0.0
        
        return accuracy
    
    def estimate_computational_savings(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        Estimate FLOPs and memory savings from sparsity.
        
        Args:
            activations: Activation tensor
        
        Returns:
            Dictionary containing savings estimates
        """
        if activations.numel() == 0:
            return {'flop_savings': 0.0, 'memory_savings': 0.0, 'theoretical_speedup': 1.0}
        
        sparsity_ratio = self.compute_sparsity_ratio(activations)
        
        # Estimate FLOP savings (assuming sparse operations)
        flop_savings = sparsity_ratio * 0.8  # Conservative estimate
        
        # Estimate memory savings
        memory_savings = sparsity_ratio * 0.9  # Sparse storage efficiency
        
        # Theoretical speedup
        theoretical_speedup = 1.0 / (1.0 - flop_savings + 1e-8)
        
        return {
            'flop_savings': flop_savings,
            'memory_savings': memory_savings,
            'theoretical_speedup': theoretical_speedup
        }
    
    def compute_advanced_efficiency_metrics(self, activations: torch.Tensor, 
                                          model_params: Optional[int] = None) -> Dict[str, float]:
        """
        Compute advanced computational efficiency metrics.
        
        Based on insights from efficiency-focused literature including
        sparse inference and computational optimization research.
        
        Args:
            activations: Activation tensor
            model_params: Optional number of model parameters
        
        Returns:
            Dictionary of advanced efficiency metrics
        """
        if activations.numel() == 0:
            return {'efficiency_score': 0.0, 'compute_intensity': 0.0, 'memory_efficiency': 0.0}
        
        # Basic sparsity metrics
        sparsity_ratio = self.compute_sparsity_ratio(activations)
        intrinsic_metrics = self.compute_intrinsic_sparsity(activations)
        
        # Compute intensity (how concentrated the non-zero values are)
        abs_acts = torch.abs(activations.view(-1))
        non_zero_mask = abs_acts > 1e-6
        
        if non_zero_mask.sum() > 0:
            non_zero_vals = abs_acts[non_zero_mask]
            # Compute intensity: higher values indicate more concentrated computation
            compute_intensity = (non_zero_vals.max() / (non_zero_vals.mean() + 1e-8)).item()
            
            # Memory efficiency: how well sparsity can be exploited for memory savings
            # Based on clustering of non-zero values
            sorted_vals, _ = torch.sort(non_zero_vals, descending=True)
            top_10_percent = int(0.1 * len(sorted_vals)) or 1
            top_vals_ratio = sorted_vals[:top_10_percent].sum() / sorted_vals.sum()
            memory_efficiency = top_vals_ratio.item()
        else:
            compute_intensity = 0.0
            memory_efficiency = 1.0  # Perfect efficiency if everything is zero
        
        # Overall efficiency score combining multiple factors
        efficiency_components = [
            sparsity_ratio,  # Higher sparsity = better efficiency
            intrinsic_metrics['effective_sparsity'],  # Entropy-based efficiency
            memory_efficiency,  # Memory access efficiency
            1.0 / (1.0 + compute_intensity * 0.1)  # Normalize compute intensity
        ]
        
        efficiency_score = np.mean(efficiency_components)
        
        return {
            'efficiency_score': efficiency_score,
            'compute_intensity': compute_intensity,
            'memory_efficiency': memory_efficiency,
            'sparsity_quality': intrinsic_metrics['effective_sparsity']
        }
    
    def analyze_inference_efficiency(self, model: nn.Module, data_loader, 
                                   device: torch.device = None) -> Dict[str, Any]:
        """
        Analyze inference efficiency characteristics.
        
        Based on insights from inference optimization literature.
        
        Args:
            model: Neural network model
            data_loader: DataLoader for analysis
            device: Device for computation
        
        Returns:
            Dictionary containing inference efficiency metrics
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        inference_times = []
        memory_usage = []
        activation_stats = []
        
        # Measure inference characteristics
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 5:  # Limit to 5 batches for efficiency
                    break
                
                data = data.to(device)
                
                # Measure inference time
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                
                if start_time:
                    start_time.record()
                
                # Forward pass
                output = model(data)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)  # milliseconds
                    inference_times.append(inference_time)
                
                # Memory usage (approximate)
                if device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated(device) / 1024**2  # MB
                    memory_usage.append(memory_used)
                
                # Activation statistics
                activation_stats.append({
                    'mean_activation': output.mean().item(),
                    'activation_sparsity': self.compute_sparsity_ratio(output),
                    'activation_range': (output.max() - output.min()).item()
                })
        
        # Compute efficiency metrics
        results = {
            'mean_inference_time': np.mean(inference_times) if inference_times else 0.0,
            'inference_time_std': np.std(inference_times) if inference_times else 0.0,
            'mean_memory_usage': np.mean(memory_usage) if memory_usage else 0.0,
            'memory_efficiency': 1.0 / (1.0 + np.std(memory_usage)) if memory_usage else 1.0
        }
        
        # Activation efficiency
        if activation_stats:
            mean_sparsity = np.mean([s['activation_sparsity'] for s in activation_stats])
            activation_stability = 1.0 / (1.0 + np.std([s['mean_activation'] for s in activation_stats]))
            
            results.update({
                'activation_sparsity': mean_sparsity,
                'activation_stability': activation_stability,
                'inference_efficiency_score': (mean_sparsity + activation_stability) / 2.0
            })
        
        return results
    
    def analyze_model(self, model: nn.Module, data_loader) -> Dict[str, Any]:
        """
        Comprehensive sparsity analysis of a model.
        
        Args:
            model: Neural network model
            data_loader: DataLoader for analysis
        
        Returns:
            Dictionary containing sparsity metrics
        """
        model.eval()
        all_activations = []
        
        # Get model device
        device = next(model.parameters()).device
        
        # Collect activations from a few batches
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 3:  # Limit to 3 batches for efficiency
                    break
                
                # Move data to the same device as model
                data = data.to(device)
                
                # Get model output (assuming it's an activation)
                output = model(data)
                all_activations.append(output)
        
        if not all_activations:
            return {'error': 'No activations collected'}
        
        # Concatenate all activations
        combined_activations = torch.cat(all_activations, dim=0)
        
        # Compute metrics
        sparsity_ratio = self.compute_sparsity_ratio(combined_activations)
        intrinsic_metrics = self.compute_intrinsic_sparsity(combined_activations)
        computational_savings = self.estimate_computational_savings(combined_activations)
        
        return {
            'sparsity_ratio': sparsity_ratio,
            **intrinsic_metrics,
            **computational_savings
        }


class InitializationAnalyzer:
    """
    Analyzes initialization quality and its impact on training dynamics.
    
    Based on literature about proper initialization (He, Xavier, etc.) and
    its relationship to activation function choice.
    """
    
    def __init__(self):
        self.name = "InitializationAnalyzer"
    
    def analyze_weight_distribution(self, model: nn.Module) -> Dict[str, float]:
        """
        Analyze the distribution of initial weights.
        
        Args:
            model: Neural network model
        
        Returns:
            Dictionary of weight distribution statistics
        """
        all_weights = []
        layer_stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights = param.data.view(-1)
                all_weights.append(weights)
                
                # Per-layer statistics
                layer_stats[f'{name}_mean'] = weights.mean().item()
                layer_stats[f'{name}_std'] = weights.std().item()
                layer_stats[f'{name}_min'] = weights.min().item()
                layer_stats[f'{name}_max'] = weights.max().item()
        
        if all_weights:
            combined_weights = torch.cat(all_weights)
            
            overall_stats = {
                'overall_mean': combined_weights.mean().item(),
                'overall_std': combined_weights.std().item(),
                'overall_min': combined_weights.min().item(),
                'overall_max': combined_weights.max().item(),
                'weight_range': (combined_weights.max() - combined_weights.min()).item()
            }
            
            return {**overall_stats, **layer_stats}
        else:
            return {'error': 'No weight parameters found'}
    
    def compute_initial_gradient_flow(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """
        Measure gradient flow quality at initialization.
        
        Args:
            model: Neural network model
            sample_input: Sample input for gradient computation
        
        Returns:
            Initial gradient flow score
        """
        model.train()
        
        # Forward pass
        output = model(sample_input)
        loss = output.sum()  # Simple loss
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        
        if gradients:
            combined_grads = torch.cat(gradients)
            grad_norm = torch.norm(combined_grads).item()
            
            # Clear gradients
            model.zero_grad()
            
            return grad_norm
        else:
            return 0.0
    
    def assess_activation_distribution_quality(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """
        Assess the quality of activation distributions at initialization.
        
        Args:
            model: Neural network model
            sample_input: Sample input
        
        Returns:
            Dictionary of activation distribution quality metrics
        """
        model.eval()
        activations = {}
        
        def capture_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(capture_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activation distributions
        quality_metrics = {}
        for name, acts in activations.items():
            acts_flat = acts.view(-1)
            
            # Distribution statistics
            mean_act = acts_flat.mean().item()
            std_act = acts_flat.std().item()
            
            # Saturation detection (for bounded activations)
            saturation_ratio = ((torch.abs(acts_flat) > 0.9).float().mean()).item()
            
            # Dead neuron detection
            dead_ratio = ((torch.abs(acts_flat) < 1e-6).float().mean()).item()
            
            quality_metrics[f'{name}_mean'] = mean_act
            quality_metrics[f'{name}_std'] = std_act
            quality_metrics[f'{name}_saturation'] = saturation_ratio
            quality_metrics[f'{name}_dead_ratio'] = dead_ratio
        
        return quality_metrics
    
    def compute_initialization_quality_score(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """
        Compute an overall initialization quality score.
        
        Args:
            model: Neural network model
            sample_input: Sample input
        
        Returns:
            Initialization quality score (higher = better)
        """
        # Weight distribution quality
        weight_stats = self.analyze_weight_distribution(model)
        weight_quality = 1.0 / (1.0 + abs(weight_stats.get('overall_mean', 0.0)))
        
        # Gradient flow quality
        grad_flow = self.compute_initial_gradient_flow(model, sample_input)
        grad_quality = min(1.0, grad_flow / 10.0)  # Normalize
        
        # Activation distribution quality
        act_stats = self.assess_activation_distribution_quality(model, sample_input)
        
        # Penalize high saturation and dead neurons
        saturation_penalty = np.mean([v for k, v in act_stats.items() if 'saturation' in k])
        dead_penalty = np.mean([v for k, v in act_stats.items() if 'dead_ratio' in k])
        
        act_quality = 1.0 - 0.5 * (saturation_penalty + dead_penalty)
        
        # Overall score
        overall_quality = (weight_quality + grad_quality + act_quality) / 3.0
        
        return max(0.0, overall_quality)


class NumericalStabilityAnalyzer:
    """
    Analyzes numerical stability during training.
    
    Monitors for gradient explosion, vanishing gradients, loss instability,
    and other numerical issues that can derail training.
    """
    
    def __init__(self):
        self.name = "NumericalStabilityAnalyzer"
    
    def monitor_numerical_stability(self, model: nn.Module, loss_history: List[float], 
                                  gradient_norms: List[float]) -> Dict[str, float]:
        """
        Monitor various aspects of numerical stability during training.
        
        Args:
            model: Neural network model
            loss_history: List of loss values over training
            gradient_norms: List of gradient norms over training
        
        Returns:
            Dictionary of stability metrics
        """
        metrics = {}
        
        # Loss stability analysis
        if len(loss_history) > 1:
            loss_array = np.array(loss_history)
            
            # Loss variance (lower = more stable)
            loss_variance = np.var(loss_array)
            metrics['loss_variance'] = loss_variance
            
            # Loss trend stability (detect oscillations)
            if len(loss_history) > 5:
                recent_losses = loss_array[-5:]
                loss_stability = 1.0 / (1.0 + np.std(recent_losses))
                metrics['loss_stability'] = loss_stability
        
        # Gradient flow stability
        if len(gradient_norms) > 1:
            grad_array = np.array(gradient_norms)
            
            # Gradient explosion detection
            grad_explosion_threshold = 10.0
            explosion_ratio = (grad_array > grad_explosion_threshold).mean()
            metrics['gradient_explosion_ratio'] = explosion_ratio
            
            # Vanishing gradient detection
            vanishing_threshold = 1e-6
            vanishing_ratio = (grad_array < vanishing_threshold).mean()
            metrics['vanishing_gradient_ratio'] = vanishing_ratio
            
            # Gradient flow stability
            if len(gradient_norms) > 5:
                recent_grads = grad_array[-5:]
                grad_stability = 1.0 / (1.0 + np.std(recent_grads))
                metrics['gradient_flow_stability'] = grad_stability
        
        # Edge of Stability detection (loss increases then decreases)
        if len(loss_history) > 10:
            loss_array = np.array(loss_history)
            # Look for patterns where loss increases then decreases
            diff = np.diff(loss_array)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            stability_oscillations = sign_changes / len(diff) if len(diff) > 0 else 0
            metrics['stability_oscillations'] = stability_oscillations
        
        # Overall stability score
        stability_components = []
        if 'loss_stability' in metrics:
            stability_components.append(metrics['loss_stability'])
        if 'gradient_flow_stability' in metrics:
            stability_components.append(metrics['gradient_flow_stability'])
        
        if stability_components:
            metrics['stability_score'] = np.mean(stability_components)
        else:
            metrics['stability_score'] = 0.0
        
        return metrics


class GradientHealthAnalyzer:
    """
    Advanced gradient health analysis.
    
    Implements comprehensive gradient analysis including EGF (Effective Gradient Flow),
    gradient health scores, and deep network analysis.
    """
    
    def __init__(self):
        self.name = "GradientHealthAnalyzer"
    
    def compute_effective_gradient_flow(self, model: nn.Module, input_batch: torch.Tensor) -> float:
        """
        Measure how effectively gradients flow through the network.
        
        Based on the ratio of output gradients to input gradients.
        
        Args:
            model: Neural network model
            input_batch: Input batch for analysis
        
        Returns:
            Effective Gradient Flow score
        """
        model.train()
        input_batch = input_batch.to(next(model.parameters()).device)
        input_batch.requires_grad_(True)
        
        # Forward pass
        output = model(input_batch)
        loss = output.sum()  # Simple loss for gradient computation
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Measure gradient flow
        input_grad_norm = torch.norm(input_batch.grad) if input_batch.grad is not None else 0.0
        
        # Get final layer gradients
        final_layer_grad_norm = 0.0
        for param in reversed(list(model.parameters())):
            if param.grad is not None:
                final_layer_grad_norm = torch.norm(param.grad)
                break
        
        # Compute EGF as ratio
        if input_grad_norm > 1e-8:
            efg = final_layer_grad_norm / input_grad_norm
        else:
            efg = 0.0
        
        return efg.item() if isinstance(efg, torch.Tensor) else efg
    
    def analyze_deep_network(self, model: nn.Module, data_loader) -> Dict[str, Any]:
        """
        Comprehensive analysis of gradient flow in deep networks.
        
        Args:
            model: Deep neural network model
            data_loader: DataLoader for analysis
        
        Returns:
            Dictionary containing gradient health metrics
        """
        model.train()
        gradient_norms = []
        layer_gradients = {}
        
        # Get model device
        device = next(model.parameters()).device
        
        # Analyze gradients over a few batches
        for i, (data, target) in enumerate(data_loader):
            if i >= 3:  # Limit analysis to 3 batches
                break
            
            # Move data and target to the same device as model
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Collect gradient norms per layer
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    if name not in layer_gradients:
                        layer_gradients[name] = []
                    layer_gradients[name].append(param_norm.item())
            
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)
        
        # Compute EGF score using sample data
        sample_data, _ = next(iter(data_loader))
        sample_data = sample_data.to(device)
        efg_score = self.compute_effective_gradient_flow(model, sample_data)
        
        # Analyze gradient health
        results = {
            'mean_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'gradient_norm_std': np.std(gradient_norms) if gradient_norms else 0.0,
            'gradient_norm_stability': 1.0 / (1.0 + np.std(gradient_norms)) if gradient_norms else 0.0,
            'efg_score': efg_score  # Add EGF score with expected key name
        }
        
        # Per-layer analysis
        for name, norms in layer_gradients.items():
            results[f'{name}_mean_grad'] = np.mean(norms)
            results[f'{name}_std_grad'] = np.std(norms)
        
        return results


class IsotropyClassificationTradeoffAnalyzer:
    """
    Analyzes the isotropy-classification performance trade-off.
    
    Based on "Isotropy, Clusters, and Classifiers" (2024) which demonstrates
    that isotropy and clustering objectives are mathematically incompatible.
    
    This analyzer tests whether SoftCap can achieve better Pareto trade-offs:
    maintaining higher isotropy while achieving competitive classification.
    
    Reference: https://arxiv.org/abs/2407.16638
    """
    
    def __init__(self):
        self.name = "IsotropyClassificationTradeoffAnalyzer"
        self.isoscore_available = ISOSCORE_AVAILABLE
    
    def compute_isoscore(self, embeddings: torch.Tensor) -> float:
        """
        Compute IsoScore metric from the official package.
        
        IsoScore measures isotropy as the inverse condition number of the
        embedding covariance matrix. Higher values indicate more isotropic distributions.
        
        Args:
            embeddings: Tensor of shape (n_samples, n_features)
        
        Returns:
            IsoScore value (higher = more isotropic)
        """
        if not self.isoscore_available:
            warnings.warn("IsoScore package not installed. Falling back to approximation.")
            return self._approximate_isoscore(embeddings)
        
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            iso_scorer = IsoScore()
            score = iso_scorer.score(embeddings_np)
            return float(score)
        except Exception as e:
            warnings.warn(f"IsoScore computation failed: {e}. Using approximation.")
            return self._approximate_isoscore(embeddings)
    
    def _approximate_isoscore(self, embeddings: torch.Tensor) -> float:
        """
        Approximate IsoScore using eigenvalue-based isotropy measure.
        
        Computes ratio of minimum to maximum eigenvalue of covariance matrix.
        This approximates the inverse condition number.
        """
        # Center embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.cov(centered.T)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvalsh(cov)
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        
        # Isotropy = min_eigenval / max_eigenval (inverse condition number)
        isoscore = (eigenvals.min() / eigenvals.max()).item()
        
        return isoscore
    
    def compute_silhouette_score(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute silhouette score measuring cluster quality.
        
        Higher silhouette scores indicate better-separated clusters,
        which correlates with classification performance.
        
        Args:
            embeddings: Tensor of shape (n_samples, n_features)
            labels: Ground truth labels of shape (n_samples,)
        
        Returns:
            Silhouette score in range [-1, 1] (higher = better clustering)
        """
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Need at least 2 clusters
            if len(np.unique(labels_np)) < 2:
                return 0.0
            
            score = silhouette_score(embeddings_np, labels_np)
            return float(score)
        except Exception as e:
            warnings.warn(f"Silhouette score computation failed: {e}")
            return 0.0
    
    def compute_tradeoff_metrics(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute both isotropy and clustering metrics to analyze trade-off.
        
        Args:
            embeddings: Tensor of shape (n_samples, n_features)
            labels: Ground truth labels of shape (n_samples,)
        
        Returns:
            Dictionary containing:
            - isoscore: Isotropy measure (higher = more isotropic)
            - silhouette: Cluster quality (higher = better clustering)
            - isotropy_index: Alternative isotropy measure (RII)
            - tradeoff_efficiency: Geometric mean of isoscore and silhouette
        """
        isoscore = self.compute_isoscore(embeddings)
        silhouette = self.compute_silhouette_score(embeddings, labels)
        
        # Also compute RII for comparison
        isotropy_analyzer = IsotropyAnalyzer()
        rii = isotropy_analyzer.compute_isotropy_index(embeddings)
        
        # Compute trade-off efficiency (geometric mean)
        # Shift silhouette to [0, 1] range for meaningful geometric mean
        silhouette_shifted = (silhouette + 1.0) / 2.0
        tradeoff_efficiency = np.sqrt(isoscore * silhouette_shifted)
        
        return {
            'isoscore': isoscore,
            'silhouette': silhouette,
            'isotropy_index': rii,
            'tradeoff_efficiency': tradeoff_efficiency
        }


class QuantizationContinuityAnalyzer:
    """
    Analyzes whether representations are quantized (discrete) or continuous.
    
    Inspired by Bird (2024) "Emergence of Quantised Representations Isolated to 
    Anisotropic Functions", which shows anisotropic activation functions spontaneously 
    create quantized representations, while isotropic functions maintain continuous 
    distributions.
    
    **Novel Contribution**: This QC Score metric is our quantitative operationalization
    of Bird's qualitative observations. Bird uses PPPM (Privileged Plane Projective 
    Method) for visual analysis; we use histogram entropy for automated measurement.
    
    The QC Score measures how discrete/quantized a neuron's activations are:
    - 1.0 = fully quantized (discrete levels, low entropy)
    - 0.0 = fully continuous (smooth distribution, high entropy)
    
    Hypothesis (from Bird 2024): Anisotropic functions (ReLU, Tanh) develop high QC 
    scores, isotropic functions (IsotropicTanh, IsotropicReLU) maintain low QC scores.
    
    Reference: Bird (2024) https://arxiv.org/abs/2410.03712
    Implementation: Novel (not from paper)
    """
    
    def __init__(self, n_bins: int = 50):
        """
        Args:
            n_bins: Number of bins for histogram-based quantization detection
        """
        self.name = "QuantizationContinuityAnalyzer"
        self.n_bins = n_bins
    
    def compute_qc_score(self, activations: torch.Tensor) -> float:
        """
        Compute Quantization-Continuity (QC) Score for neuron activations.
        
        The QC Score is computed as:
        1. Build histogram of activation values
        2. Compute entropy of histogram (normalized)
        3. QC = 1 - normalized_entropy
        
        High entropy = continuous distribution (low QC score)
        Low entropy = quantized/discrete distribution (high QC score)
        
        Args:
            activations: Tensor of shape (n_samples,) or (n_samples, n_neurons)
                        Activation values for one or more neurons
        
        Returns:
            QC score in [0, 1] where 1.0 = fully quantized, 0.0 = fully continuous
        """
        if activations.ndim > 1:
            # Average over multiple neurons
            scores = [self._compute_single_qc_score(activations[:, i]) 
                     for i in range(activations.shape[1])]
            return float(np.mean(scores))
        else:
            return self._compute_single_qc_score(activations)
    
    def _compute_single_qc_score(self, activations: torch.Tensor) -> float:
        """Compute QC score for a single neuron's activations."""
        activations_np = activations.detach().cpu().numpy()
        
        # Remove any NaN/Inf values
        activations_np = activations_np[np.isfinite(activations_np)]
        
        if len(activations_np) < 10:
            return 0.0  # Not enough data
        
        # Compute histogram
        counts, _ = np.histogram(activations_np, bins=self.n_bins, density=True)
        counts = counts + 1e-10  # Avoid log(0)
        
        # Normalize to probability distribution
        probs = counts / counts.sum()
        
        # Compute entropy
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(self.n_bins)
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0.0
        
        # QC score = 1 - normalized_entropy
        # (high entropy = continuous = low QC score)
        qc_score = 1.0 - normalized_entropy
        
        return float(qc_score)
    
    def compute_layer_qc_scores(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """
        Compute QC scores for all layers in a network.
        
        Args:
            model: Neural network model
            data_loader: DataLoader for collecting activations
        
        Returns:
            Dictionary mapping layer names to their QC scores
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Hook to collect activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # Register hooks on all activation layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU)) or \
               'SoftCap' in module.__class__.__name__:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass to collect activations
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                model(data)
                if batch_idx >= 5:  # Sample first few batches
                    break
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute QC scores
        qc_scores = {}
        for name, acts in activations.items():
            # Flatten to (n_samples, n_neurons)
            if acts.ndim > 2:
                acts = acts.reshape(acts.size(0), -1)
            qc_scores[name] = self.compute_qc_score(acts)
        
        return qc_scores
    
    def analyze_quantization_evolution(self, activation_history: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze how quantization evolves during training.
        
        Args:
            activation_history: List of activation tensors at different training steps
        
        Returns:
            Dictionary containing:
            - qc_scores: List of QC scores over time
            - quantization_trend: 'increasing', 'decreasing', or 'stable'
            - final_qc_score: Final quantization level
        """
        qc_scores = [self.compute_qc_score(acts) for acts in activation_history]
        
        # Compute trend
        if len(qc_scores) >= 2:
            trend_slope = (qc_scores[-1] - qc_scores[0]) / len(qc_scores)
            if trend_slope > 0.01:
                trend = 'increasing'  # Becoming more quantized
            elif trend_slope < -0.01:
                trend = 'decreasing'  # Becoming more continuous
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        return {
            'qc_scores': qc_scores,
            'quantization_trend': trend,
            'final_qc_score': qc_scores[-1] if qc_scores else 0.0,
            'mean_qc_score': float(np.mean(qc_scores)) if qc_scores else 0.0
        }


def comprehensive_analysis(model: nn.Module, data_loader, activation_fn=None, 
                         training_history: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Perform comprehensive analysis using all metrics.
    
    Args:
        model: Neural network model
        data_loader: DataLoader for analysis
        activation_fn: Activation function used in the model
        training_history: Optional training history for curve analysis
    
    Returns:
        Comprehensive analysis results
    """
    results = {}
    
    # Isotropy analysis
    isotropy_analyzer = IsotropyAnalyzer()
    sparsity_analyzer = SparsityAnalyzer()
    init_analyzer = InitializationAnalyzer()
    stability_analyzer = NumericalStabilityAnalyzer()
    gradient_analyzer = GradientHealthAnalyzer()
    
    # Get sample data and move to model's device
    sample_data, _ = next(iter(data_loader))
    device = next(model.parameters()).device
    sample_data = sample_data.to(device)
    
    # Collect model representations
    model.eval()
    with torch.no_grad():
        representations = model(sample_data)
    
    # Run analyses
    results['isotropy'] = isotropy_analyzer.analyze_representations(representations)
    results['sparsity'] = sparsity_analyzer.analyze_model(model, data_loader)
    results['initialization'] = init_analyzer.compute_initialization_quality_score(model, sample_data)
    results['gradient_health'] = gradient_analyzer.analyze_deep_network(model, data_loader)
    
    # Stability analysis (if training history available)
    if training_history:
        # Handle actual training history structure from checkpoints
        loss_history = training_history.get('train_loss', training_history.get('losses', []))
        grad_norms = training_history.get('gradient_norms', [])
        results['stability'] = stability_analyzer.monitor_numerical_stability(model, loss_history, grad_norms)
    
    return results


def run_comprehensive_metrics_analysis(model: nn.Module, data_loader, activation_fn=None, 
                                      training_history: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Wrapper function for comprehensive_analysis to maintain backward compatibility.
    
    This function is used by main.py and training.py modules.
    
    Args:
        model: Neural network model
        data_loader: DataLoader for analysis
        activation_fn: Activation function used in the model
        training_history: Optional training history for curve analysis
    
    Returns:
        Comprehensive analysis results
    """
    return comprehensive_analysis(model, data_loader, activation_fn, training_history)
