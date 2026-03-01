"""
Consolidated Training Analysis Module

Provides comprehensive training analysis, loss curve analysis, convergence detection,
and recommendations for further training.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import torch
import torch.nn as nn

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingAnalyzer:
    """Comprehensive training analysis and monitoring."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self) -> Dict[str, Any]:
        """Load training results and batch history."""
        data = {}
        
        # Load main results
        results_file = self.results_dir / "complete_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data['complete_results'] = json.load(f)
        
        # Load batch histories from individual training runs
        batch_histories = {}
        for activation_dir in self.results_dir.glob("*_training"):
            batch_file = activation_dir / "batch_history.json"
            if batch_file.exists():
                activation_name = activation_dir.name.replace("_training", "")
                with open(batch_file, 'r') as f:
                    batch_histories[activation_name] = json.load(f)
        
        data['batch_histories'] = batch_histories
        return data
    
    def analyze_convergence_potential(self, training_history: Dict[str, List]) -> Dict[str, Any]:
        """Analyze if a model has potential for further improvement."""
        val_losses = training_history.get('val_loss', [])
        val_accs = training_history.get('val_acc', [])
        
        if len(val_losses) < 3:
            return {'recommendation': 'insufficient_data', 'confidence': 0.0}
        
        # Analyze recent trend (last 20% of training)
        recent_window = max(3, len(val_losses) // 5)
        recent_losses = val_losses[-recent_window:]
        recent_accs = val_accs[-recent_window:] if val_accs else []
        
        # Calculate trends
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        acc_trend = np.polyfit(range(len(recent_accs)), recent_accs, 1)[0] if recent_accs else 0
        
        # Stability analysis
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        stability = loss_std / loss_mean if loss_mean > 0 else float('inf')
        
        # Generate recommendation
        if loss_trend < -0.001 and stability < 0.1:
            recommendation = 'continue_training'
            confidence = 0.8
        elif loss_trend > 0.001 or stability > 0.2:
            recommendation = 'early_stopping'
            confidence = 0.7
        else:
            recommendation = 'monitor_closely'
            confidence = 0.5
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'loss_trend': loss_trend,
            'acc_trend': acc_trend,
            'stability': stability,
            'recent_loss_mean': loss_mean,
            'recent_loss_std': loss_std
        }
    
    def create_loss_visualization(self, batch_histories: Dict[str, List]) -> None:
        """Create comprehensive loss visualizations."""
        if not batch_histories:
            print("No batch histories available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Analysis', fontsize=16)
        
        # Raw loss curves
        ax1 = axes[0, 0]
        for activation, history in batch_histories.items():
            if 'train_loss' in history:
                ax1.plot(history['train_loss'], label=activation, alpha=0.7)
        ax1.set_title('Raw Training Loss')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Smoothed loss curves
        ax2 = axes[0, 1]
        for activation, history in batch_histories.items():
            if 'train_loss' in history:
                smoothed = self._smooth_curve(history['train_loss'])
                ax2.plot(smoothed, label=activation, alpha=0.8, linewidth=2)
        ax2.set_title('Smoothed Training Loss')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss distribution
        ax3 = axes[1, 0]
        loss_data = []
        labels = []
        for activation, history in batch_histories.items():
            if 'train_loss' in history:
                loss_data.extend(history['train_loss'])
                labels.extend([activation] * len(history['train_loss']))
        
        if loss_data:
            df = pd.DataFrame({'Loss': loss_data, 'Activation': labels})
            sns.boxplot(data=df, x='Activation', y='Loss', ax=ax3)
            ax3.set_title('Loss Distribution by Activation')
            ax3.tick_params(axis='x', rotation=45)
        
        # Convergence analysis
        ax4 = axes[1, 1]
        convergence_data = {}
        for activation, history in batch_histories.items():
            if 'val_loss' in history:
                analysis = self.analyze_convergence_potential(history)
                convergence_data[activation] = analysis['confidence']
        
        if convergence_data:
            activations = list(convergence_data.keys())
            confidences = list(convergence_data.values())
            bars = ax4.bar(activations, confidences, alpha=0.7)
            ax4.set_title('Convergence Confidence')
            ax4.set_ylabel('Confidence Score')
            ax4.tick_params(axis='x', rotation=45)
            
            # Color bars by confidence level
            for bar, conf in zip(bars, confidences):
                if conf > 0.7:
                    bar.set_color('green')
                elif conf > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_curve(self, data: List[float], window: int = 50) -> List[float]:
        """Apply moving average smoothing to data."""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training recommendations."""
        recommendations = {}
        
        if 'batch_histories' not in data:
            return {'error': 'No batch histories available for analysis'}
        
        for activation, history in data['batch_histories'].items():
            analysis = self.analyze_convergence_potential(history)
            
            # Generate specific recommendations
            rec = {
                'analysis': analysis,
                'actions': []
            }
            
            if analysis['recommendation'] == 'continue_training':
                rec['actions'].append('Continue training with current settings')
                rec['actions'].append('Monitor for overfitting')
            elif analysis['recommendation'] == 'early_stopping':
                rec['actions'].append('Consider early stopping')
                rec['actions'].append('Try different hyperparameters')
            else:
                rec['actions'].append('Monitor closely for 5-10 more epochs')
                rec['actions'].append('Consider reducing learning rate')
            
            # Specific recommendations for SoftCap
            if 'softcap' in activation.lower():
                if analysis['stability'] > 0.15:
                    rec['actions'].append('Try higher learning rate (SoftCap is bounded)')
                    rec['actions'].append('Consider longer training (SoftCap benefits from extended training)')
            
            recommendations[activation] = rec
        
        return recommendations
    
    def create_comprehensive_report(self, data: Dict[str, Any]) -> None:
        """Generate comprehensive analysis report."""
        recommendations = self.generate_recommendations(data)
        
        report_content = "# Training Analysis Report\n\n"
        report_content += f"Generated: {pd.Timestamp.now()}\n\n"
        
        report_content += "## Summary\n\n"
        if 'complete_results' in data:
            results = data['complete_results']
            report_content += f"- Total activations analyzed: {len(results)}\n"
            
            # Find best performing activation
            best_acc = 0
            best_activation = None
            for act, metrics in results.items():
                if isinstance(metrics, dict) and 'test_acc' in metrics:
                    if metrics['test_acc'] > best_acc:
                        best_acc = metrics['test_acc']
                        best_activation = act
            
            if best_activation:
                report_content += f"- Best performing activation: {best_activation} ({best_acc:.4f})\n"
        
        report_content += "\n## Detailed Recommendations\n\n"
        
        for activation, rec in recommendations.items():
            report_content += f"### {activation}\n\n"
            
            analysis = rec['analysis']
            report_content += f"- **Recommendation**: {analysis['recommendation']}\n"
            report_content += f"- **Confidence**: {analysis['confidence']:.2f}\n"
            report_content += f"- **Loss trend**: {analysis['loss_trend']:.6f}\n"
            report_content += f"- **Stability**: {analysis['stability']:.4f}\n\n"
            
            report_content += "**Actions:**\n"
            for action in rec['actions']:
                report_content += f"- {action}\n"
            report_content += "\n"
        
        # Save report
        with open(self.output_dir / 'training_analysis_report.md', 'w') as f:
            f.write(report_content)
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete training analysis pipeline."""
        print("Loading training data...")
        data = self.load_training_data()
        
        print("Creating visualizations...")
        if 'batch_histories' in data:
            self.create_loss_visualization(data['batch_histories'])
        
        print("Generating recommendations...")
        recommendations = self.generate_recommendations(data)
        
        print("Creating comprehensive report...")
        self.create_comprehensive_report(data)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        return {
            'data': data,
            'recommendations': recommendations,
            'output_dir': str(self.output_dir)
        }


def analyze_training_results(results_dir: Path) -> Dict[str, Any]:
    """Convenience function to run training analysis."""
    analyzer = TrainingAnalyzer(results_dir)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results-dir', type=str, default='./training_output',
                       help='Directory containing training results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        exit(1)
    
    analyze_training_results(results_dir)
