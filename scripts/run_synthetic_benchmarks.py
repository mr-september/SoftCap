#!/usr/bin/env python3
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
Run Synthetic Benchmarks for Activation Function Analysis

This script runs the synthetic benchmark suite independently of the main training pipeline.
It tests activation functions on various synthetic datasets designed to evaluate isotropy
and decision boundary properties.

Usage:
    python run_synthetic_benchmarks.py                    # Default settings (core Parametric* + controls)
    python run_synthetic_benchmarks.py --quick            # Quick test with fewer samples
    python run_synthetic_benchmarks.py --detailed         # Full analysis with model saving
    python run_synthetic_benchmarks.py --output custom    # Custom output directory
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime
import json
import torch
import torch.nn as nn

# Add project root to path if necessary
try:
    from softcap.synthetic_benchmarks import run_synthetic_benchmarks
    from softcap.control_activations import get_standard_experimental_set
except ImportError:
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    from softcap.synthetic_benchmarks import run_synthetic_benchmarks
    from softcap.control_activations import get_standard_experimental_set


def get_activation_functions() -> dict:
    """Get canonical activation functions (3 parametric + controls)."""
    activations = get_standard_experimental_set()
    activations['LeakyReLU'] = nn.LeakyReLU(0.01)
    for name, activation in activations.items():
        if not hasattr(activation, 'name'):
            activation.name = name
    return activations


def get_control_only_activations() -> dict:
    """Get only the essential control activations for direct comparison."""
    activations = {
        # Essential control activations only
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(), 
        'SiLU': nn.SiLU(),
        'Tanh': nn.Tanh(),
    }
    
    for name, activation in activations.items():
        if not hasattr(activation, 'name'):
            activation.name = name
            
    return activations


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Activation Function Synthetic Benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    python run_synthetic_benchmarks.py                   # Standard analysis (core set)
    python run_synthetic_benchmarks.py --quick           # Quick test (500 samples, 200 epochs)
    python run_synthetic_benchmarks.py --detailed        # Detailed analysis (2000 samples, 1000 epochs)
    python run_synthetic_benchmarks.py --activations ReLU GELU SiLU  # Test specific activations
        """
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default=None,
        help='Output directory for results (default: synthetic_benchmarks_YYYYMMDD_HHMMSS)'
    )
    
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=1000,
        help='Number of samples per synthetic dataset (default: 1000)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=500,
        help='Training epochs per model (default: 500)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test mode preset (500 samples, 200 epochs)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Detailed analysis mode preset (2000 samples, 1000 epochs)'
    )
    
    parser.add_argument(
        '--activations',
        nargs='+',
        default=None,
        help='Specific activation functions to test (e.g., --activations ReLU GELU)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training (default: auto detects CUDA)'
    )

    parser.add_argument(
        '--lr',
        default='0.01',
        help='Base learning rate or comma-separated list for grid sweep (e.g. "0.01,0.001")'
    )

    parser.add_argument(
        '--weight-decay',
        default='0.0',
        help='Weight decay (L2) coefficient or comma-separated list for grid sweep (e.g. "1e-4,1e-5"). Default: 0.0 (no weight decay)'
    )

    parser.add_argument(
        '--sweep',
        action='store_true',
        help='If set, treat --lr as a comma-separated sweep; otherwise use only the first value'
    )

    parser.add_argument(
        '--softcap-hp-search',
        action='store_true',
        help='Run a focused hyperparameter grid search for the SoftCap family (lrs: 0.1,0.01,0.001; wds: 0,1e-4,1e-5; repeats=3)'
    )
    
    parser.add_argument(
        '--seed-offset',
        type=int,
        default=0,
        help='Seed offset for reproducibility. Final seed = 42 + seed_offset. Use with --repeats=1 to run single seeds.'
    )
    
    parser.add_argument(
        '--repeats',
        type=int,
        default=None,
        help='Number of random seed repeats per experiment. Overrides default values (5 for normal runs, 3 for HP search).'
    )
    
    parser.add_argument(
        '--weight-decay-filter',
        type=str,
        default=None,
        help='Filter to specific weight decay values (comma-separated). Only applies with --softcap-hp-search. Example: "0.0,1e-4"'
    )
    
    parser.add_argument(
        '--reuse-datasets-from',
        type=str,
        default=None,
        help='Path to previous run directory to reuse datasets and seeds from (e.g., "synthetic_benchmarks_20250923_000502")'
    )
    
    parser.add_argument(
        '--controls-only',
        action='store_true',
        help='Test only the essential control activations (ReLU, GELU, SiLU, Tanh) at specified learning rates and weight decay'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume functionality - start fresh and overwrite existing results'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    
    print("🌀 Activation Function Synthetic Benchmarks")
    print("=" * 50)
    
    args = parse_arguments()
    
    if args.quick:
        args.n_samples = 500
        args.epochs = 200
        print("⚡ Quick mode enabled: 500 samples, 200 epochs")
    elif args.detailed:
        args.n_samples = 2000
        args.epochs = 1000
        print("🔬 Detailed mode enabled: 2000 samples, 1000 epochs")
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"synthetic_benchmarks_{timestamp}"
    
    output_dir = Path(args.output_dir)
    
    # Determine which activations to use
    if args.controls_only:
        activations = get_control_only_activations()
        print("🎯 Controls-only mode: Testing ReLU, GELU, SiLU, Tanh")
    else:
        all_activations = get_activation_functions()

        if args.activations:
            requested = set(args.activations)
            available = set(all_activations.keys())
            invalid = requested - available
            if invalid:
                print(f"❌ Error: Invalid activation function(s) requested: {', '.join(invalid)}")
                print(f"✅ Available options are: {', '.join(available)}")
                return 1
            activations = {name: all_activations[name] for name in requested}
        else:
            activations = all_activations
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"🎯 Testing activations: {', '.join(activations.keys())}")
    print(f"🖥️  Using device: {device.upper()}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Samples per dataset: {args.n_samples}")
    print(f"🔄 Training epochs: {args.epochs}")
    print("-" * 50 + "\n")
    
    # Handle dataset reuse from previous run
    reuse_info = None
    if args.reuse_datasets_from:
        reuse_dir = Path(args.reuse_datasets_from)
        if not reuse_dir.exists():
            print(f"❌ Error: Reuse directory does not exist: {reuse_dir}")
            return 1
        
        # Check for existing datasets and manifest
        datasets_dir = reuse_dir / "softcap_hyperparam_search" / "datasets" if (reuse_dir / "softcap_hyperparam_search").exists() else reuse_dir / "datasets"
        manifest_file = reuse_dir / "softcap_hyperparam_search" / "run_manifest.json" if (reuse_dir / "softcap_hyperparam_search").exists() else reuse_dir / "run_manifest.json"
        
        if not datasets_dir.exists():
            print(f"❌ Error: No datasets directory found in: {reuse_dir}")
            return 1
        if not manifest_file.exists():
            print(f"❌ Error: No manifest file found in: {reuse_dir}")
            return 1
        
        # Load previous manifest for seed and configuration info  
        with open(manifest_file, 'r') as f:
            prev_manifest = json.load(f)
        
        reuse_info = {
            'datasets_dir': datasets_dir,
            'prev_manifest': prev_manifest,
            'n_samples': prev_manifest.get('n_samples', 1000),
            'repeats': prev_manifest.get('repeats', 3)
        }
        
        print(f"🔄 Reusing datasets from: {datasets_dir}")
        print(f"📊 Previous run: {prev_manifest.get('n_samples', 1000)} samples, {prev_manifest.get('repeats', 3)} repeats")
        
        # Override current settings to match previous run for reproducibility
        args.n_samples = reuse_info['n_samples']
        if args.repeats is None:
            args.repeats = reuse_info['repeats']
    
    start_time = time.time()
    
    try:
        # Pass repeats and base learning rate through to the benchmark runner
        # Prepare lr argument: allow sweep if requested, otherwise pass single value
        lr_arg = args.lr
        if not args.sweep and isinstance(lr_arg, str) and ',' in lr_arg:
            lr_arg = lr_arg.split(',')[0]

        # Focused SoftCap hyperparameter search mode
        if args.softcap_hp_search:
            print("🔎 SoftCap hyperparameter search requested")
            # Respect --activations if specified, otherwise filter for SoftCap family
            if args.activations:
                # User explicitly specified which activations to search
                softcap_activations = activations  # Already filtered above
                print(f"   - Using user-specified activations: {', '.join(softcap_activations.keys())}")
            else:
                # Default: all SoftCap variants
                softcap_activations = {name: act for name, act in all_activations.items() if 'SoftCap' in name}
            if not softcap_activations:
                print("❌ No activations found for HP search. Aborting.")
                return 1

            # Hard-coded grid for the focused search as requested
            hp_lr = "0.1,0.01,0.001"
            hp_wd = "0.0,0.0001,0.00001"
            
            # Apply weight decay filter if specified
            if args.weight_decay_filter:
                filtered_wd = args.weight_decay_filter
                print(f"   - LR grid: {hp_lr}")
                print(f"   - Weight decay grid: {filtered_wd} (filtered)")
                hp_wd = filtered_wd
            else:
                print(f"   - LR grid: {hp_lr}")
                print(f"   - Weight decay grid: {hp_wd}")

            # Use custom repeats if provided, otherwise default to 3 for HP search
            hp_repeats = args.repeats if args.repeats is not None else 3

            # Run the focused search: repeats=3 for each combination
            results = run_synthetic_benchmarks(
                activation_functions=softcap_activations,
                output_dir=output_dir / "softcap_hyperparam_search",
                n_samples=args.n_samples,
                epochs=args.epochs,
                save_models=True,
                device=device,
                repeats=hp_repeats,
                lr=hp_lr,
                weight_decay=hp_wd,
                seed_offset=args.seed_offset,
                reuse_datasets_from=reuse_info,
                resume=not args.no_resume,
            )
            # Ensure downstream reporting uses the same restricted set
            activations = softcap_activations
            
            # Automatically generate hyperparameter analysis reports
            print("\n🔍 Generating hyperparameter analysis reports...")
            try:
                import subprocess
                hp_search_dir = output_dir / "softcap_hyperparam_search"
                hp_reports_dir = hp_search_dir / "hp_reports"
                
                # Run the hp reports generation script
                script_path = Path(__file__).parent / "scripts" / "analysis" / "generate_hp_reports.py"
                cmd = [
                    sys.executable, str(script_path),
                    "--input-dir", str(hp_search_dir),
                    "--output-dir", str(hp_reports_dir)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
                if result.returncode == 0:
                    print(f"✅ Hyperparameter analysis reports saved to: {hp_reports_dir}")
                    print("   - Training curves for all activation functions")
                    print("   - Heatmaps showing performance across hyperparameter combinations")
                    print("   - Boxplots comparing final accuracies")
                    print("   - HTML index for easy browsing")
                else:
                    print(f"⚠️ Warning: HP reports generation failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"   Error: {result.stderr.strip()}")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not generate HP analysis reports: {e}")
                print("   You can manually run: python scripts/analysis/generate_hp_reports.py")

        else:
            # Use custom repeats if provided, otherwise default to 5 for normal runs
            normal_repeats = args.repeats if args.repeats is not None else 5
            
            results = run_synthetic_benchmarks(
                activation_functions=activations,
                output_dir=output_dir,
                n_samples=args.n_samples,
                epochs=args.epochs,
                save_models=True,
                device=device,
                repeats=normal_repeats,
                lr=lr_arg,
                weight_decay=args.weight_decay,
                seed_offset=args.seed_offset,
                reuse_datasets_from=reuse_info,
                resume=not args.no_resume,
                )
        
        print("\n" + "=" * 60)
        print("🎉 SYNTHETIC BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        # Robust summary: handle missing activations in results (e.g., when a focused run
        # only produced SoftCap-family results). Avoid KeyError by inspecting the result keys
        # and computing averages only over datasets that actually contain a given activation.
        datasets = list(results.get('accuracy_scores', {}).keys())
        if not datasets:
            print("No dataset results found. Nothing to summarize.")
            return 1

        # Determine which activations were actually produced in the results
        observed_acts = set()
        for d in datasets:
            observed_acts.update(results.get('accuracy_scores', {}).get(d, {}).keys())

        requested_act_names = list(activations.keys()) if activations else []
        missing = [a for a in requested_act_names if a not in observed_acts]
        if missing:
            print(f"⚠️ The following requested activations were not present in results and will be skipped: {', '.join(missing)}")

        # Prefer the requested order, but fall back to observed activations if none match
        act_names = [a for a in requested_act_names if a in observed_acts] or sorted(observed_acts)

        print("\n📈 Average Performance Across All Datasets:")
        print("-" * 55)
        print(f"{'Activation':<25} | {'Avg Accuracy':<15} | {'Avg Isotropy':<15}")
        print("-" * 55)

        summary_data = {}
        for act_name in act_names:
            acc_sum = 0.0
            acc_count = 0
            iso_sum = 0.0
            iso_count = 0
            for d in datasets:
                acc_entry = results.get('accuracy_scores', {}).get(d, {}).get(act_name)
                if acc_entry and 'mean' in acc_entry:
                    acc_sum += acc_entry['mean']
                    acc_count += 1
                iso_entry = results.get('isotropy_scores', {}).get(d, {}).get(act_name)
                if iso_entry and 'mean' in iso_entry:
                    iso_sum += iso_entry['mean']
                    iso_count += 1

            if acc_count == 0 and iso_count == 0:
                print(f"Skipping {act_name}: no data found in any dataset")
                continue

            avg_accuracy = (acc_sum / acc_count) if acc_count else float('nan')
            avg_isotropy = (iso_sum / iso_count) if iso_count else float('nan')
            summary_data[act_name] = {'accuracy': avg_accuracy, 'isotropy': avg_isotropy}
            print(f"{act_name:<25} | {avg_accuracy:<15.4f} | {avg_isotropy:<15.4f}")

        print("-" * 55)
        
        best_accuracy_act = max(summary_data, key=lambda k: summary_data[k]['accuracy'])
        best_isotropy_act = max(summary_data, key=lambda k: summary_data[k]['isotropy'])
        
        print(f"\n🏆 Best Average Accuracy: {best_accuracy_act} ({summary_data[best_accuracy_act]['accuracy']:.4f})")
        print(f"🏆 Best Average Isotropy:  {best_isotropy_act} ({summary_data[best_isotropy_act]['isotropy']:.4f})")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed_time/60:.2f} minutes")
        
        print(f"\n📁 Detailed results saved to: {output_dir}")
        print("       - comprehensive_report.html: Main interactive report with top-level visuals (note: per-seed decision boundary PNGs and dataset overview PNGs are excluded from the root report and are stored in per-dataset 'Boundary Plots - <Dataset>' subfolders).")
        print("       - synthetic_benchmark_summary.csv: Tidy data for external tools.")
        print("       - synthetic_benchmark_results.json: Raw JSON data dump.")
        print("       - PNG images for summary and diagnostics (e.g., synthetic_benchmark_summary.png, aggregated_convergence_curves.png, activation_performance_radar.png).")
        print("         - Note: The activation_performance_radar uses a clipped linear range focusing on [0.75, 1.0] for clarity; boundary and dataset overview PNGs are grouped under 'Boundary Plots - <Dataset>'.")

        return 0
        
    except Exception as e:
        print(f"\n❌ An error occurred during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())