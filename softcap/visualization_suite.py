"""
Comprehensive Visualization Suite for SoftCap Research

This module provides a complete visualization framework for activation function analysis,
including basic plots, statistical analysis, synthetic benchmarks, and publication-quality figures.

Key Features:
- Basic performance visualizations (accuracy, loss curves)
- Statistical analysis (significance testing, effect sizes)
- Synthetic function fitting analysis
- Interactive dashboards
- Publication-quality figures
- Automated HTML index generation

Usage:
    from softcap.visualization_suite import create_visualization_suite
    
    viz_suite = create_visualization_suite(output_dir='./results/visualizations')
    viz_suite.generate_all_visualizations(results_dict)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class BasicVisualizationModule:
    """Basic visualization plots for standard metrics."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_comparison(self, results: Dict[str, float], 
                               title: str = "Activation Function Accuracy Comparison"):
        """Create bar plot of accuracy comparison."""
        activations = list(results.keys())
        accuracies = list(results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(activations, accuracies, alpha=0.8)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn([acc for acc in accuracies])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Activation Function')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self, history: Dict[str, Any]):
        """Plot training and validation loss curves."""
        if not isinstance(history, dict) or 'train_loss' not in history:
            print("Warning: Invalid history format for plotting loss curves")
            return
            
        plt.figure(figsize=(12, 6))
        
        train_loss = history['train_loss']
        val_loss = history.get('val_loss', {})
        
        # Get common activations
        activations = list(train_loss.keys())
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        for act in activations:
            plt.plot(train_loss[act], label=act, alpha=0.8, linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot validation loss if available
        if val_loss:
            plt.subplot(1, 2, 2)
            for act in activations:
                if act in val_loss:
                    plt.plot(val_loss[act], label=act, alpha=0.8, linewidth=2)
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class SyntheticBenchmarkModule:
    """Synthetic benchmark analysis and visualization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_function_fitting_analysis(self, activation_functions: Dict[str, nn.Module]):
        """Run comprehensive function fitting analysis."""
        from .synthetic_function_fitting import FunctionFittingVisualizer
        
        visualizer = FunctionFittingVisualizer()
        visualizer.plot_function_fits(activation_functions, self.output_dir)
        
        # Create summary
        self._create_function_fit_summary()
    
    def _create_function_fit_summary(self):
        """Create summary heatmap of function fitting results."""
        results = {}
        for f in self.output_dir.glob('function_fit_*_results.json'):
            func_name = f.stem.replace('function_fit_', '').replace('_results', '')
            try:
                with open(f, 'r') as fh:
                    results[func_name] = json.load(fh)
            except:
                continue
        
        if not results:
            return
            
        df = pd.DataFrame(results).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.1e', cmap='viridis_r', 
                   cbar_kws={'label': 'Mean Squared Error (log scale)'})
        plt.title('Function Approximation Performance (Lower is Better)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'function_fit_summary.png', dpi=300, bbox_inches='tight')
        plt.close()


class StatisticalVisualizationModule:
    """Statistical analysis visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_statistical_significance(self, pairwise_results: Dict[str, Any]):
        """Create heatmap of statistical significance between activations."""
        if not pairwise_results:
            print("Warning: No pairwise results provided for statistical significance plot")
            return
        
        # Handle different input formats
        if isinstance(list(pairwise_results.values())[0], dict):
            # Format: {'act1': {'act2': p_value, ...}, ...}
            activations = list(pairwise_results.keys())
            n_acts = len(activations)
            
            sig_matrix = np.ones((n_acts, n_acts))
            p_value_matrix = np.ones((n_acts, n_acts))
            
            for i, act1 in enumerate(activations):
                for j, act2 in enumerate(activations):
                    if act2 in pairwise_results[act1]:
                        p_val = pairwise_results[act1][act2]
                        sig_matrix[i, j] = 0 if p_val < 0.05 else 1
                        p_value_matrix[i, j] = p_val
        else:
            # Format: {'act1_vs_act2': {'p_value': val, 'significant': bool}, ...}
            activations = set()
            for comparison in pairwise_results.keys():
                if '_vs_' in comparison:
                    act1, act2 = comparison.split('_vs_')
                    activations.update([act1, act2])
            
            activations = sorted(list(activations))
            n_acts = len(activations)
            
            sig_matrix = np.ones((n_acts, n_acts))
            p_value_matrix = np.ones((n_acts, n_acts))
            
            for comparison, result in pairwise_results.items():
                if '_vs_' in comparison:
                    act1, act2 = comparison.split('_vs_')
                    i, j = activations.index(act1), activations.index(act2)
                    
                    p_val = result.get('p_value', 1.0)
                    is_sig = result.get('significant', False)
                    
                    sig_matrix[i, j] = 0 if is_sig else 1
                    sig_matrix[j, i] = 0 if is_sig else 1
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        colors = ['red', 'white']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        sns.heatmap(sig_matrix, 
                   xticklabels=activations,
                   yticklabels=activations,
                   annot=p_value_matrix, fmt='.4f',
                   cmap=cmap, cbar_kws={'label': 'Significant Difference'},
                   square=True)
        
        plt.title('Statistical Significance Matrix (p-values)\nRed = Significant Difference (p < 0.05)')
        plt.xlabel('Activation Function')
        plt.ylabel('Activation Function')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'statistical_significance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_effect_sizes(self, effect_sizes: Dict[str, float]):
        """Plot effect sizes for activation function comparisons."""
        if not effect_sizes:
            print("Warning: No effect sizes provided")
            return
        
        comparisons = list(effect_sizes.keys())
        effects = list(effect_sizes.values())
        
        colors = ['green' if abs(e) < 0.2 else 'orange' if abs(e) < 0.5 else 'red' 
                 for e in effects]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(comparisons)), effects, color=colors, alpha=0.7)
        
        plt.title('Effect Sizes for Activation Function Comparisons')
        plt.xlabel('Comparison')
        plt.ylabel('Effect Size (Cohen\'s d)')
        plt.xticks(range(len(comparisons)), comparisons, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        plt.axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.8, color='red', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()


class ConsolidatedPlotsModule:
    """Consolidated plotting functions for aggregated benchmark data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_boxplots_by_dataset(self, df: pd.DataFrame, lr_value: float = None):
        """Create boxplots of accuracy by dataset and activation."""
        if lr_value is not None:
            df = df[df['lr'] == lr_value].copy()
            title_suffix = f" (lr={lr_value})"
            filename = f"accuracy_boxplots_lr{lr_value}.png"
        else:
            title_suffix = ""
            filename = "accuracy_boxplots_all.png"
        
        datasets = sorted(df['dataset'].unique())
        n_datasets = len(datasets)
        
        if n_datasets <= 3:
            fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6))
            if n_datasets == 1:
                axes = [axes]
        else:
            rows = (n_datasets + 2) // 3
            fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
            axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            ax = axes[idx]
            df_dataset = df[df['dataset'] == dataset]
            sns.boxplot(data=df_dataset, x='activation', y='accuracy', ax=ax, palette='Set2')
            ax.set_title(f'{dataset.capitalize()} - Accuracy{title_suffix}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Activation', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_ylim([0, 1.05])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused axes
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {self.output_dir / filename}")
    
    def plot_accuracy_heatmap(self, df: pd.DataFrame, lr_value: float = None):
        """Create heatmap of activation vs dataset accuracy."""
        if lr_value is not None:
            df = df[df['lr'] == lr_value].copy()
            title_suffix = f" (lr={lr_value})"
            filename = f"accuracy_heatmap_lr{lr_value}.png"
        else:
            title_suffix = ""
            filename = "accuracy_heatmap_all.png"
        
        pivot = df.groupby(['activation', 'dataset'])['accuracy'].mean().reset_index()
        pivot_table = pivot.pivot(index='activation', columns='dataset', values='accuracy')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})
        plt.title(f'Accuracy Heatmap: Activation × Dataset{title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Activation', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {self.output_dir / filename}")
    
    def plot_performance_radar(self, df: pd.DataFrame, lr_value: float = None):
        """Create radar plot for top performers across datasets."""
        if lr_value is not None:
            df = df[df['lr'] == lr_value].copy()
            title_suffix = f" (lr={lr_value})"
            filename = f"performance_radar_lr{lr_value}.png"
        else:
            title_suffix = ""
            filename = "performance_radar_all.png"
        
        datasets = sorted(df['dataset'].unique())
        activations = sorted(df['activation'].unique())
        
        # Calculate average accuracy per activation
        avg_acc = df.groupby('activation')['accuracy'].mean().reset_index()
        avg_acc = avg_acc.sort_values('accuracy', ascending=False)
        top_activations = avg_acc.head(10)['activation'].tolist()  # Top 10 to avoid clutter
        
        N = len(datasets)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_activations)))
        
        for i, activation in enumerate(top_activations):
            values = []
            for dataset in datasets:
                acc = df[(df['activation'] == activation) & (df['dataset'] == dataset)]['accuracy'].mean()
                values.append(acc if not np.isnan(acc) else 0)
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=activation, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=12)
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        ax.set_title(f'Performance Radar Chart{title_suffix}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {self.output_dir / filename}")
    
    def plot_overview_dashboard(self, df: pd.DataFrame):
        """Create overview dashboard with multiple plots."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Average accuracy by activation
        ax1 = fig.add_subplot(gs[0, :2])
        avg_acc = df.groupby('activation')['accuracy'].mean().reset_index()
        avg_acc = avg_acc.sort_values('accuracy', ascending=False)
        bars = ax1.bar(range(len(avg_acc)), avg_acc['accuracy'], alpha=0.8, color='steelblue')
        ax1.set_xticks(range(len(avg_acc)))
        ax1.set_xticklabels(avg_acc['activation'], rotation=45, ha='right')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Accuracy Across All Datasets', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Accuracy distribution by dataset
        ax2 = fig.add_subplot(gs[0, 2])
        datasets = df['dataset'].unique()
        for dataset in datasets:
            data = df[df['dataset'] == dataset]['accuracy']
            ax2.hist(data, alpha=0.5, label=dataset.capitalize(), bins=20)
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Accuracy Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Activation performance heatmap
        ax3 = fig.add_subplot(gs[1, :])
        pivot = df.groupby(['activation', 'dataset'])['accuracy'].mean().reset_index()
        pivot_table = pivot.pivot(index='activation', columns='dataset', values='accuracy')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, cbar=False)
        ax3.set_title('Accuracy Heatmap: Activation × Dataset', fontsize=14, fontweight='bold')
        
        # 4. Radar chart for top activations
        ax4 = fig.add_subplot(gs[2, :], projection='polar')
        top_acts = avg_acc.head(5)['activation'].tolist()
        N = len(datasets)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_acts)))
        for i, act in enumerate(top_acts):
            values = [df[(df['activation'] == act) & (df['dataset'] == d)]['accuracy'].mean() for d in datasets]
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=act, color=colors[i])
            ax4.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels([d.capitalize() for d in datasets])
        ax4.set_ylim([0, 1])
        ax4.set_title('Top 5 Activations Radar', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax4.grid(True)
        
        plt.suptitle('Consolidated Benchmarks Overview Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / 'overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {self.output_dir / 'overview_dashboard.png'}")


class ComprehensiveVisualizationSuite:
    """Main visualization suite combining all modules."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        self.basic = BasicVisualizationModule(output_dir / 'basic')
        self.statistical = StatisticalVisualizationModule(output_dir / 'statistical')
        self.synthetic = SyntheticBenchmarkModule(output_dir / 'synthetic')
        self.consolidated = ConsolidatedPlotsModule(output_dir / 'consolidated')
    
    def generate_all_visualizations(self, results: Dict[str, Any]):
        """Generate all visualization types from results."""
        print("Generating comprehensive visualization suite...")
        
        # Basic visualizations
        if 'metrics' in results:
            print("- Creating basic visualizations...")
            self.basic.plot_accuracy_comparison(results['metrics'])
        
        if 'history' in results:
            self.basic.plot_loss_curves(results['history'])
        
        # Statistical visualizations
        if 'pairwise_results' in results:
            print("- Creating statistical visualizations...")
            self.statistical.plot_statistical_significance(results['pairwise_results'])
        
        if 'effect_sizes' in results:
            self.statistical.plot_effect_sizes(results['effect_sizes'])
        
        # Synthetic benchmark visualizations
        if 'activation_functions' in results:
            print("- Running synthetic benchmarks...")
            self.synthetic.run_function_fitting_analysis(results['activation_functions'])
        
        # Consolidated visualizations
        if 'consolidated_df' in results:
            print("- Creating consolidated visualizations...")
            df = results['consolidated_df']
            self.consolidated.plot_accuracy_boxplots_by_dataset(df)
            self.consolidated.plot_accuracy_heatmap(df)
            self.consolidated.plot_performance_radar(df)
            self.consolidated.plot_overview_dashboard(df)
        
        # Create HTML index
        self._create_html_index()
        
        print(f"All visualizations saved to: {self.output_dir}")
    
    def _create_html_index(self):
        """Create HTML index of all visualizations."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SoftCap Visualization Suite</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 1px solid #ccc; }
                .section { margin: 20px 0; }
                .plot { margin: 10px 0; }
                img { max-width: 800px; border: 1px solid #ddd; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>SoftCap Activation Function Analysis</h1>
            
            <div class="section">
                <h2>Basic Performance Analysis</h2>
                <div class="plot">
                    <h3>Accuracy Comparison</h3>
                    <img src="basic/accuracy_comparison.png" alt="Accuracy Comparison">
                </div>
                <div class="plot">
                    <h3>Training Curves</h3>
                    <img src="basic/loss_curves.png" alt="Loss Curves">
                </div>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                <div class="plot">
                    <h3>Statistical Significance</h3>
                    <img src="statistical/statistical_significance.png" alt="Statistical Significance">
                </div>
                <div class="plot">
                    <h3>Effect Sizes</h3>
                    <img src="statistical/effect_sizes.png" alt="Effect Sizes">
                </div>
            </div>
            
            <div class="section">
                <h2>Synthetic Function Fitting</h2>
                <div class="plot">
                    <h3>Function Fitting Summary</h3>
                    <img src="synthetic/function_fit_summary.png" alt="Function Fitting Summary">
                </div>
            </div>
            
            <div class="section">
                <h2>Consolidated Benchmarks</h2>
                <div class="plot">
                    <h3>Overview Dashboard</h3>
                    <img src="consolidated/overview_dashboard.png" alt="Overview Dashboard">
                </div>
                <div class="plot">
                    <h3>Accuracy Boxplots</h3>
                    <img src="consolidated/accuracy_boxplots_all.png" alt="Accuracy Boxplots">
                </div>
                <div class="plot">
                    <h3>Accuracy Heatmap</h3>
                    <img src="consolidated/accuracy_heatmap_all.png" alt="Accuracy Heatmap">
                </div>
                <div class="plot">
                    <h3>Performance Radar</h3>
                    <img src="consolidated/performance_radar_all.png" alt="Performance Radar">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'index.html', 'w') as f:
            f.write(html_content)


def create_visualization_suite(output_dir: Path) -> ComprehensiveVisualizationSuite:
    """Factory function to create comprehensive visualization suite."""
    return ComprehensiveVisualizationSuite(output_dir)
    
    def generate_all_visualizations(self, results: Dict[str, Any]):
        """Generate all visualization types from results."""
        print("Generating comprehensive visualization suite...")
        
        # Basic visualizations
        if 'metrics' in results:
            print("- Creating basic visualizations...")
            self.basic.plot_accuracy_comparison(results['metrics'])
        
        if 'history' in results:
            self.basic.plot_loss_curves(results['history'])
        
        # Statistical visualizations
        if 'pairwise_results' in results:
            print("- Creating statistical visualizations...")
            self.statistical.plot_statistical_significance(results['pairwise_results'])
        
        if 'effect_sizes' in results:
            self.statistical.plot_effect_sizes(results['effect_sizes'])
        
        # Synthetic benchmark visualizations
        if 'activation_functions' in results:
            print("- Running synthetic benchmarks...")
            self.synthetic.run_function_fitting_analysis(results['activation_functions'])
        
        # Create HTML index
        self._create_html_index()
        
        print(f"All visualizations saved to: {self.output_dir}")
    
    def _create_html_index(self):
        """Create HTML index of all visualizations."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SoftCap Visualization Suite</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 1px solid #ccc; }
                .section { margin: 20px 0; }
                .plot { margin: 10px 0; }
                img { max-width: 800px; border: 1px solid #ddd; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>SoftCap Activation Function Analysis</h1>
            
            <div class="section">
                <h2>Basic Performance Analysis</h2>
                <div class="plot">
                    <h3>Accuracy Comparison</h3>
                    <img src="basic/accuracy_comparison.png" alt="Accuracy Comparison">
                </div>
                <div class="plot">
                    <h3>Training Curves</h3>
                    <img src="basic/loss_curves.png" alt="Loss Curves">
                </div>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                <div class="plot">
                    <h3>Statistical Significance</h3>
                    <img src="statistical/statistical_significance.png" alt="Statistical Significance">
                </div>
                <div class="plot">
                    <h3>Effect Sizes</h3>
                    <img src="statistical/effect_sizes.png" alt="Effect Sizes">
                </div>
            </div>
            
            <div class="section">
                <h2>Synthetic Function Fitting</h2>
                <div class="plot">
                    <h3>Function Fitting Summary</h3>
                    <img src="synthetic/function_fit_summary.png" alt="Function Fitting Summary">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'index.html', 'w') as f:
            f.write(html_content)


def create_visualization_suite(output_dir: Path) -> ComprehensiveVisualizationSuite:
    """Factory function to create comprehensive visualization suite."""
    return ComprehensiveVisualizationSuite(output_dir)
