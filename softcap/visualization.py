"""
SoftCap Visualization Module

Publication-quality visualization tools for activation function analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


class ActivationVisualizer:
    """Creates publication-quality plots for activation function analysis."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib for publication-quality plots."""
        plt.style.use('seaborn-v0_8-paper')
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def plot_activation_function(
        self,
        activation_fn,
        x_range: Tuple[float, float] = (-3, 3),
        num_points: int = 1000,
        title: str = "Activation Function",
        filename: str = "activation_plot.png"
    ):
        """Plot a single activation function."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        with torch.no_grad():
            y = activation_fn(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x.numpy(), y.numpy(), linewidth=2, label=getattr(activation_fn, 'name', 'Activation'))
        plt.xlabel('Input (x)')
        plt.ylabel('Output f(x)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_activation_derivative(
        self,
        activation_fn,
        x_range: Tuple[float, float] = (-3, 3),
        num_points: int = 1000,
        title: str = "Activation Function Derivative",
        filename: str = "activation_derivative_plot.png"
    ):
        """Plot activation function derivative."""
        if not hasattr(activation_fn, 'derivative'):
            print(f"Warning: {type(activation_fn).__name__} has no derivative method")
            return
        
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        with torch.no_grad():
            dy_dx = activation_fn.derivative(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x.numpy(), dy_dx.numpy(), linewidth=2, 
                label=f"{getattr(activation_fn, 'name', 'Activation')} Derivative")
        plt.xlabel('Input (x)')
        plt.ylabel("f'(x)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_activation_functions(
        self,
        activation_dict: Dict[str, Any],
        x_range: Tuple[float, float] = (-3, 3),
        num_points: int = 1000,
        filename_base: str = "activation_comparison"
    ):
        """Create comprehensive comparison plots."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Function comparison
        ax1 = axes[0, 0]
        for name, activation in activation_dict.items():
            with torch.no_grad():
                y = activation(x)
            ax1.plot(x.numpy(), y.numpy(), linewidth=2, label=name)
        ax1.set_xlabel('Input (x)')
        ax1.set_ylabel('Output f(x)')
        ax1.set_title('Activation Functions')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Derivative comparison
        ax2 = axes[0, 1]
        for name, activation in activation_dict.items():
            if hasattr(activation, 'derivative'):
                with torch.no_grad():
                    dy_dx = activation.derivative(x)
                ax2.plot(x.numpy(), dy_dx.numpy(), linewidth=2, label=f"{name} Derivative")
        ax2.set_xlabel('Input (x)')
        ax2.set_ylabel("f'(x)")
        ax2.set_title('Activation Function Derivatives')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Zoomed view around origin
        ax3 = axes[1, 0]
        x_zoom = torch.linspace(-1, 1, 200)
        for name, activation in activation_dict.items():
            with torch.no_grad():
                y_zoom = activation(x_zoom)
            ax3.plot(x_zoom.numpy(), y_zoom.numpy(), linewidth=2, label=name)
        ax3.set_xlabel('Input (x)')
        ax3.set_ylabel('Output f(x)')
        ax3.set_title('Activation Functions (Zoomed)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Properties comparison (if available)
        ax4 = axes[1, 1]
        properties = []
        names = []
        for name, activation in activation_dict.items():
            if hasattr(activation, 'get_metrics'):
                metrics = activation.get_metrics()
                properties.append([
                    metrics.get('zero_ratio', 0),
                    metrics.get('output_mean', 0),
                    metrics.get('output_std', 1)
                ])
                names.append(name)
        
        if properties:
            properties = np.array(properties)
            x_pos = np.arange(len(names))
            width = 0.25
            
            ax4.bar(x_pos - width, properties[:, 0], width, label='Sparsity Ratio', alpha=0.8)
            ax4.bar(x_pos, properties[:, 1], width, label='Output Mean', alpha=0.8)
            ax4.bar(x_pos + width, properties[:, 2], width, label='Output Std', alpha=0.8)
            
            ax4.set_xlabel('Activation Functions')
            ax4.set_ylabel('Metric Value')
            ax4.set_title('Activation Properties')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(names, rotation=45)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No metrics available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Activation Properties')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_isotropy_analysis(self, isotropy_results: Dict[str, Any], filename: str = "isotropy_analysis.png"):
        """Plot isotropy analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract results by activation
        activations = list(isotropy_results.keys())
        isotropy_indices = [isotropy_results[act].get('isotropy_index', 0) for act in activations]
        discretization_scores = [isotropy_results[act].get('discretization_score', 0) for act in activations]
        symmetry_biases = [isotropy_results[act].get('symmetry_bias', 0) for act in activations]
        
        # Plot 1: Isotropy Index
        axes[0, 0].bar(activations, isotropy_indices, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Representational Isotropy Index')
        axes[0, 0].set_ylabel('RII Score (Higher = More Isotropic)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Discretization Score
        axes[0, 1].bar(activations, discretization_scores, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Discretization Detection Score')
        axes[0, 1].set_ylabel('DDS Score (Lower = Less Discretized)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Symmetry Bias
        axes[1, 0].bar(activations, symmetry_biases, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Symmetry Bias Detection')
        axes[1, 0].set_ylabel('SBD Score (Higher = More Symmetric)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Combined isotropy score
        combined_scores = [(rii + sbd - dds) / 2 for rii, sbd, dds in 
                          zip(isotropy_indices, symmetry_biases, discretization_scores)]
        axes[1, 1].bar(activations, combined_scores, alpha=0.7, color='gold')
        axes[1, 1].set_title('Combined Isotropy Score')
        axes[1, 1].set_ylabel('Combined Score (Higher = More Isotropic)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sparsity_analysis(self, sparsity_results: Dict[str, Any], filename: str = "sparsity_analysis.png"):
        """Plot sparsity analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        activations = list(sparsity_results.keys())
        sparsity_ratios = [sparsity_results[act].get('sparsity_ratio', 0) for act in activations]
        flops_savings = [sparsity_results[act].get('computational_savings', {}).get('flops_savings', 0) 
                        for act in activations]
        memory_savings = [sparsity_results[act].get('computational_savings', {}).get('memory_savings', 0) 
                         for act in activations]
        
        # Plot 1: Sparsity Ratios
        axes[0, 0].bar(activations, sparsity_ratios, alpha=0.7, color='purple')
        axes[0, 0].set_title('Activation Sparsity Ratios')
        axes[0, 0].set_ylabel('Sparsity Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: FLOPs Savings
        axes[0, 1].bar(activations, flops_savings, alpha=0.7, color='orange')
        axes[0, 1].set_title('Estimated FLOPs Savings')
        axes[0, 1].set_ylabel('FLOPs Savings Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory Savings
        axes[1, 0].bar(activations, memory_savings, alpha=0.7, color='teal')
        axes[1, 0].set_title('Estimated Memory Savings')
        axes[1, 0].set_ylabel('Memory Savings Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Efficiency Score (combined)
        efficiency_scores = [(sr + fs + ms) / 3 for sr, fs, ms in 
                           zip(sparsity_ratios, flops_savings, memory_savings)]
        axes[1, 1].bar(activations, efficiency_scores, alpha=0.7, color='brown')
        axes[1, 1].set_title('Combined Efficiency Score')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gradient_health(self, gradient_results: Dict[str, Any], filename: str = "gradient_health.png"):
        """Plot gradient health analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        activations = list(gradient_results.keys())
        
        # Extract gradient health metrics (using actual keys from GradientHealthAnalyzer)
        efg_scores = [gradient_results[act].get('efg_score', 0) for act in activations]
        # Use gradient norm std as proxy for vanishing risk (higher std = more unstable)
        vanishing_risks = [gradient_results[act].get('gradient_norm_std', 0) for act in activations]
        # Use inverse of gradient stability as exploding risk (lower stability = higher risk)
        exploding_risks = [1.0 - gradient_results[act].get('gradient_norm_stability', 1.0) for act in activations]
        # Use actual mean gradient norm
        mean_norms = [gradient_results[act].get('mean_gradient_norm', 0) for act in activations]
        
        # Plot 1: Effective Gradient Flow
        axes[0, 0].bar(activations, efg_scores, alpha=0.7, color='blue')
        axes[0, 0].set_title('Effective Gradient Flow')
        axes[0, 0].set_ylabel('EGF Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Gradient Health Risks
        x_pos = np.arange(len(activations))
        width = 0.35
        axes[0, 1].bar(x_pos - width/2, vanishing_risks, width, label='Vanishing Risk', alpha=0.7, color='red')
        axes[0, 1].bar(x_pos + width/2, exploding_risks, width, label='Exploding Risk', alpha=0.7, color='orange')
        axes[0, 1].set_title('Gradient Health Risks')
        axes[0, 1].set_ylabel('Number of Layers at Risk')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(activations, rotation=45)
        axes[0, 1].legend()
        
        # Plot 3: Mean Gradient Norms
        axes[1, 0].bar(activations, mean_norms, alpha=0.7, color='green')
        axes[1, 0].set_title('Mean Gradient Norms')
        axes[1, 0].set_ylabel('Mean Gradient Norm')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Overall Gradient Health Score
        health_scores = [efg / (1 + vr + er) for efg, vr, er in zip(efg_scores, vanishing_risks, exploding_risks)]
        axes[1, 1].bar(activations, health_scores, alpha=0.7, color='purple')
        axes[1, 1].set_title('Overall Gradient Health Score')
        axes[1, 1].set_ylabel('Health Score (Higher = Better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, performance_results: Dict[str, Any], filename: str = "performance_comparison.png"):
        """Plot performance comparison across activation functions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        activations = list(performance_results.keys())
        
        # Extract performance metrics (assuming they exist in results)
        accuracies = [performance_results[act].get('best_accuracy', 0) for act in activations]
        convergence_epochs = [performance_results[act].get('convergence_epoch', 0) for act in activations]
        training_stability = [performance_results[act].get('training_stability', 0) for act in activations]
        final_losses = [performance_results[act].get('final_loss', 1) for act in activations]
        
        # Plot 1: Best Accuracy
        axes[0, 0].bar(activations, accuracies, alpha=0.7, color='blue')
        axes[0, 0].set_title('Best Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Convergence Speed
        axes[0, 1].bar(activations, convergence_epochs, alpha=0.7, color='orange')
        axes[0, 1].set_title('Convergence Speed')
        axes[0, 1].set_ylabel('Epochs to Convergence')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Training Stability
        axes[1, 0].bar(activations, training_stability, alpha=0.7, color='green')
        axes[1, 0].set_title('Training Stability')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Final Loss
        axes[1, 1].bar(activations, final_losses, alpha=0.7, color='red')
        axes[1, 1].set_title('Final Training Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_dashboard(results: Dict[str, Any], output_dir: Path, filename: str = "summary_dashboard.png"):
    """Create a comprehensive summary dashboard of all results."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create a complex grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    activations = list(results.keys())
    
    # Main performance comparison (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    accuracies = [results[act].get('training', {}).get('best_accuracy', 0) for act in activations]
    bars = ax1.bar(activations, accuracies, alpha=0.8, color='steelblue')
    ax1.set_title('Best Validation Accuracy', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Isotropy scores (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    isotropy_scores = [results[act].get('isotropy', {}).get('isotropy_index', 0) for act in activations]
    ax2.bar(activations, isotropy_scores, alpha=0.8, color='lightcoral')
    ax2.set_title('Isotropy Index (Higher = More Isotropic)', fontsize=14)
    ax2.set_ylabel('RII Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Sparsity comparison (second row, left)
    ax3 = fig.add_subplot(gs[1, :2])
    sparsity_ratios = [results[act].get('sparsity', {}).get('sparsity_ratio', 0) for act in activations]
    ax3.bar(activations, sparsity_ratios, alpha=0.8, color='lightgreen')
    ax3.set_title('Activation Sparsity Ratios', fontsize=14)
    ax3.set_ylabel('Sparsity Ratio')
    ax3.tick_params(axis='x', rotation=45)
    
    # Gradient health (second row, right)
    ax4 = fig.add_subplot(gs[1, 2:])
    efg_scores = [results[act].get('gradient_health', {}).get('efg_score', 0) for act in activations]
    ax4.bar(activations, efg_scores, alpha=0.8, color='gold')
    ax4.set_title('Effective Gradient Flow', fontsize=14)
    ax4.set_ylabel('EGF Score')
    ax4.tick_params(axis='x', rotation=45)
    
    # Radar chart for comprehensive comparison (bottom half)
    ax5 = fig.add_subplot(gs[2:, :])
    
    # Prepare data for radar chart
    categories = ['Accuracy', 'Isotropy', 'Sparsity', 'Gradient Health', 'Efficiency']
    
    # Normalize all metrics to 0-1 scale for fair comparison
    accuracy_norm = np.array(accuracies)
    isotropy_norm = np.array(isotropy_scores)
    sparsity_norm = np.array(sparsity_ratios)
    gradient_norm = np.array(efg_scores)
    
    # Compute efficiency as combination of sparsity and computational savings
    efficiency_scores = []
    for act in activations:
        comp_savings = results[act].get('sparsity', {}).get('computational_savings', {})
        eff = (comp_savings.get('flops_savings', 0) + comp_savings.get('memory_savings', 0)) / 2
        efficiency_scores.append(eff)
    efficiency_norm = np.array(efficiency_scores)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(activations)))
    
    for i, act in enumerate(activations):
        values = [
            accuracy_norm[i],
            isotropy_norm[i],
            sparsity_norm[i],
            gradient_norm[i],
            efficiency_norm[i]
        ]
        values += values[:1]  # Complete the circle
        
        ax5.plot(angles, values, 'o-', linewidth=2, label=act, color=colors[i])
        ax5.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_ylim(0, 1)
    ax5.set_title('Comprehensive Activation Function Comparison', fontsize=16, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax5.grid(True)
    
    plt.suptitle('SoftCap Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary dashboard saved to {output_dir / filename}")
