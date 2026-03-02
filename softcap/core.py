# Copyright 2026 Larry Cai and Jie Tang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SoftCap Core Analysis Engine

This module provides the main analysis framework for SoftCap research,
integrating all literature-informed metrics and experimental protocols.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import warnings
import time

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

class SoftCapAnalysisEngine:
    """
    Main analysis engine for comprehensive SoftCap evaluation.
    
    Integrates all literature-informed metrics:
    - Isotropy analysis (RII, DDS, SBD)
    - Sparsity efficiency metrics
    - Gradient health assessment
    - Performance evaluation
    - Computational efficiency analysis
    """
    
    def __init__(self, output_dir: Optional[str] = None, device: Optional[str] = None):
        """Initialize the analysis engine."""
        self.device = device or self._get_best_device()
        self.output_dir = Path(output_dir) if output_dir else self._create_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric trackers
        self.results = defaultdict(dict)
        self.experiment_log = []
        
        # Configure matplotlib for publication-quality plots
        self._setup_plotting()
    
    def _get_best_device(self) -> str:
        """Select the best available device."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA device: {device_name}")
            return "cuda"
        else:
            print("Using CPU device")
            return "cpu"
    
    def _create_output_dir(self) -> Path:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_suffix = "CUDA" if "cuda" in self.device else "CPU"
        return Path(f"results/run_{timestamp}_{device_suffix}")
    
    def _setup_plotting(self):
        """Configure matplotlib for publication-quality plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
    
    def run_comprehensive_analysis(
        self,
        activation_functions: Dict[str, nn.Module],
        datasets: Dict[str, Any],
        architectures: Dict[str, Any],
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the complete 7-step analysis pipeline with enhanced metrics.
        
        Returns:
            Comprehensive results dictionary with all metrics and analyses.
        """
        print("=" * 60)
        print("SoftCap Comprehensive Analysis Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Activation Function Visualization
        print("\n--- Step 1: Activation Function Visualization ---")
        self._step1_visualize_activations(activation_functions)
        
        # Step 2: Standard Performance Comparison (Enhanced)
        print("\n--- Step 2: Enhanced Performance Comparison ---")
        self._step2_performance_comparison(
            activation_functions, datasets, architectures, hyperparameters
        )
        
        # Step 3: Deep Network Gradient Flow Analysis
        print("\n--- Step 3: Deep Network Gradient Flow Analysis ---")
        self._step3_deep_gradient_analysis(activation_functions)
        
        # Step 4: Initialization Analysis
        print("\n--- Step 4: Initialization Analysis ---")
        self._step4_initialization_analysis(activation_functions)
        
        # Step 5: Isotropy Analysis (NEW)
        print("\n--- Step 5: Isotropy Analysis ---")
        self._step5_isotropy_analysis(activation_functions)
        
        # Step 6: Comprehensive Reporting
        print("\n--- Step 6: Comprehensive Data Export ---")
        self._step6_export_data()
        
        # Step 7: Final Report Generation
        print("\n--- Step 7: Final Report Generation ---")
        self._step7_generate_report()
        
        total_time = time.time() - start_time
        print(f"\n--- Analysis Complete ---")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return self.results
    
    def _step1_visualize_activations(self, activation_functions: Dict[str, nn.Module]):
        """Step 1: Generate publication-quality activation function plots."""
        from .visualization import ActivationVisualizer
        
        visualizer = ActivationVisualizer()
        
        # Create comparison plots
        visualizer.compare_activation_functions(
            activation_functions,
            self.output_dir,
            filename_base="activation_comparison"
        )
        
        print("✓ Activation function visualizations saved")
    
    def _step2_performance_comparison(
        self,
        activation_functions: Dict[str, nn.Module],
        datasets: Dict[str, Any],
        architectures: Dict[str, Any],
        hyperparameters: Dict[str, Any]
    ):
        """Step 2: Enhanced performance comparison with isotropy metrics."""
        from .training import EnhancedTrainer
        from .metrics import IsotropyAnalyzer, SparsityAnalyzer
        
        for act_name, activation in activation_functions.items():
            print(f"  Analyzing {act_name}...")
            
            # Train models with different configurations
            trainer = EnhancedTrainer(
                activation=activation,
                device=self.device,
                output_dir=self.output_dir / f"training_{act_name}"
            )
            
            # Run training with hyperparameter sweep
            training_results = trainer.train_with_hyperparameter_sweep(
                datasets, architectures, hyperparameters
            )
            
            # Analyze isotropy properties
            isotropy_analyzer = IsotropyAnalyzer()
            isotropy_results = isotropy_analyzer.analyze_model(
                trainer.best_model, datasets['test']
            )
            
            # Analyze sparsity properties
            sparsity_analyzer = SparsityAnalyzer()
            sparsity_results = sparsity_analyzer.analyze_model(
                trainer.best_model, datasets['test']
            )
            
            # Store results
            self.results[act_name] = {
                'training': training_results,
                'isotropy': isotropy_results,
                'sparsity': sparsity_results
            }
        
        print("✓ Enhanced performance comparison completed")
    
    def _step3_deep_gradient_analysis(self, activation_functions: Dict[str, nn.Module]):
        """Step 3: Deep network gradient flow analysis."""
        from .models import DeepMLP
        from .metrics import GradientHealthAnalyzer
        
        gradient_analyzer = GradientHealthAnalyzer()
        
        for act_name, activation in activation_functions.items():
            print(f"  Analyzing deep gradients for {act_name}...")
            
            # Create deep network (15 layers)
            model = DeepMLP(
                input_dim=784,
                hidden_dim=64,
                output_dim=10,
                num_layers=15,
                activation=activation
            ).to(self.device)
            
            # Analyze gradient flow
            gradient_results = gradient_analyzer.analyze_deep_network(
                model, self._get_sample_data()
            )
            
            if act_name not in self.results:
                self.results[act_name] = {}
            self.results[act_name]['deep_gradients'] = gradient_results
        
        print("✓ Deep gradient analysis completed")
    
    def _step4_initialization_analysis(self, activation_functions: Dict[str, nn.Module]):
        """Step 4: Initialization gradient analysis."""
        from .metrics import InitializationAnalyzer
        
        init_analyzer = InitializationAnalyzer()
        
        for act_name, activation in activation_functions.items():
            print(f"  Analyzing initialization for {act_name}...")
            
            init_results = init_analyzer.analyze_initialization_gradients(
                activation, self._get_sample_data()
            )
            
            if act_name not in self.results:
                self.results[act_name] = {}
            self.results[act_name]['initialization'] = init_results
        
        print("✓ Initialization analysis completed")
    
    def _step5_isotropy_analysis(self, activation_functions: Dict[str, nn.Module]):
        """Step 5: Comprehensive isotropy analysis (NEW)."""
        from .metrics import IsotropyAnalyzer
        
        isotropy_analyzer = IsotropyAnalyzer()
        
        for act_name, activation in activation_functions.items():
            print(f"  Analyzing isotropy for {act_name}...")
            
            # Get trained model from previous steps
            if act_name in self.results and 'training' in self.results[act_name]:
                model = self.results[act_name]['training']['best_model']
                
                # Comprehensive isotropy analysis
                isotropy_results = isotropy_analyzer.comprehensive_analysis(
                    model, self._get_sample_data()
                )
                
                self.results[act_name]['isotropy_detailed'] = isotropy_results
        
        print("✓ Isotropy analysis completed")
    
    def _step6_export_data(self):
        """Step 6: Export all data to structured formats."""
        # Export to CSV
        self._export_to_csv()
        
        # Export to JSON for detailed analysis
        self._export_to_json()
        
        print("✓ Data export completed")
    
    def _step7_generate_report(self):
        """Step 7: Generate comprehensive analysis report."""
        from .reporting import ReportGenerator
        
        report_generator = ReportGenerator(self.results, self.output_dir)
        report_generator.generate_comprehensive_report()
        
        print("✓ Comprehensive report generated")
    
    def _get_sample_data(self) -> torch.Tensor:
        """Get sample data for analysis."""
        # Return a batch of sample data for analysis
        return torch.randn(32, 784).to(self.device)
    
    def _export_to_csv(self):
        """Export results to CSV format."""
        # Flatten results for CSV export
        csv_data = []
        
        for act_name, act_results in self.results.items():
            for metric_category, metrics in act_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        csv_data.append({
                            'activation': act_name,
                            'category': metric_category,
                            'metric': metric_name,
                            'value': value
                        })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.output_dir / 'comprehensive_results.csv', index=False)
    
    def _export_to_json(self):
        """Export detailed results to JSON format."""
        import json
        
        # Convert torch tensors to lists for JSON serialization
        json_results = self._convert_tensors_to_lists(dict(self.results))
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _convert_tensors_to_lists(self, obj):
        """Recursively convert torch tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_lists(item) for item in obj]
        else:
            return obj


# Convenience function for easy usage
def run_softcap_analysis(
    activation_functions: Optional[Dict[str, nn.Module]] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete SoftCap analysis.
    
    Args:
        activation_functions: Dict of activation functions to test
        output_dir: Output directory for results
        device: Device to use for computation
    
    Returns:
        Comprehensive analysis results
    """
    # Default activation functions if none provided
    if activation_functions is None:
        from .activations import get_default_activations
        activation_functions = get_default_activations()
    
    # Default datasets and configurations
    from .data import get_default_datasets
    from .models import get_default_architectures
    from .config import get_default_hyperparameters
    
    datasets = get_default_datasets()
    architectures = get_default_architectures()
    hyperparameters = get_default_hyperparameters()
    
    # Run analysis
    engine = SoftCapAnalysisEngine(output_dir=output_dir, device=device)
    return engine.run_comprehensive_analysis(
        activation_functions, datasets, architectures, hyperparameters
    )
