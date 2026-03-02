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
Create hyperparameter impact analysis from available data.

Usage:
  python create_hyperparameter_analysis.py --data-dir /path/to/condensed

Expected inputs in data-dir:
  - synthetic_benchmark_results.json (for manifest ranges)
  - synthetic_benchmark_per_run_accuracies.csv
  - overall_activation_performance.csv, final_convergence_analysis.csv,
    learning_curve_improvements.csv (optional; used for dashboard)
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def create_hyperparameter_impact_analysis(data_dir: Path):
    print("Creating hyperparameter impact analysis...")

    results_path = data_dir / 'synthetic_benchmark_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"Missing required file: {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Prefer manifest values; if manifest lacks hp ranges, fallback to per-run CSV unique lr/wd values
    manifest = data.get('manifest', {})
    lr_values = []
    wd_values = []
    if manifest and ('lr' in manifest or 'learning_rates' in manifest):
        lr_val = manifest.get('lr', manifest.get('learning_rates'))
        if isinstance(lr_val, (list, tuple)):
            lr_values = [float(x) for x in lr_val]
        else:
            lr_values = [float(x) for x in str(lr_val).split(',') if x]
    if manifest and ('weight_decay' in manifest or 'weight_decay_values' in manifest):
        wd_val = manifest.get('weight_decay', manifest.get('weight_decay_values'))
        if isinstance(wd_val, (list, tuple)):
            wd_values = [float(x) for x in wd_val]
        else:
            wd_values = [float(x) for x in str(wd_val).split(',') if x]

    print(f"Learning rates tested: {lr_values}")
    print(f"Weight decay values tested: {wd_values}")

    per_run_path = data_dir / 'synthetic_benchmark_per_run_accuracies.csv'
    if not per_run_path.exists():
        raise FileNotFoundError(f"Missing required file: {per_run_path}")
    per_run_df = pd.read_csv(per_run_path)

    # Derive HP combinations: prefer manifest-derived combinations but fallback to unique pairs from per-run CSV
    repeats = 3
    hp_combinations = [{'lr': lr, 'weight_decay': wd} for lr in lr_values for wd in wd_values]
    per_run_df['hp_combo_id'] = per_run_df['seed_index'] // repeats
    per_run_df['repeat_id'] = per_run_df['seed_index'] % repeats
    # If hp_combinations is empty or does not align with per_run hp_combo_id indices, fallback to unique (lr, weight_decay)
    try:
        if not hp_combinations or per_run_df['hp_combo_id'].max() >= len(hp_combinations):
            raise IndexError('Manifest-derived hp_combinations do not match per-run hp_combo_id indices')
        per_run_df['lr'] = per_run_df['hp_combo_id'].map(lambda x: hp_combinations[x]['lr'])
        per_run_df['weight_decay'] = per_run_df['hp_combo_id'].map(lambda x: hp_combinations[x]['weight_decay'])
    except Exception:
        # Derive unique hp combinations from data
        unique_pairs = per_run_df[['lr', 'weight_decay']].drop_duplicates().sort_values(['lr', 'weight_decay'])
        combos = [(row['lr'], row['weight_decay']) for _, row in unique_pairs.iterrows()]
        hp_combinations = [{'lr': lr, 'weight_decay': wd} for lr, wd in combos]
        # Rebuild hp_combo_id mapping using (lr, wd) mapping
        combo_map = { (c['lr'], c['weight_decay']): idx for idx, c in enumerate(hp_combinations) }
        per_run_df['hp_combo_id'] = per_run_df.apply(lambda r: combo_map.get((r['lr'], r['weight_decay']), 0), axis=1)

    hp_analysis = []
    for (dataset, activation), group in per_run_df.groupby(['dataset', 'activation']):
        lr_effects = group.groupby('lr')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        lr_effects['dataset'] = dataset
        lr_effects['activation'] = activation
        lr_effects['hyperparameter'] = 'learning_rate'
        lr_effects = lr_effects.rename(columns={'lr': 'value', 'mean': 'accuracy_mean', 'std': 'accuracy_std'})

        wd_effects = group.groupby('weight_decay')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        wd_effects['dataset'] = dataset
        wd_effects['activation'] = activation
        wd_effects['hyperparameter'] = 'weight_decay'
        wd_effects = wd_effects.rename(columns={'weight_decay': 'value', 'mean': 'accuracy_mean', 'std': 'accuracy_std'})

        hp_analysis.extend(lr_effects.to_dict('records'))
        hp_analysis.extend(wd_effects.to_dict('records'))

    hp_analysis_df = pd.DataFrame(hp_analysis)
    hp_analysis_df.to_csv(data_dir / 'hyperparameter_effects_analysis.csv', index=False)
    print(f"Created hyperparameter_effects_analysis.csv with {len(hp_analysis_df)} rows")

    # Best hyperparameter combinations per dataset/activation
    best_hp_combos = []
    for (dataset, activation), group in per_run_df.groupby(['dataset', 'activation']):
        hp_combo_perf = group.groupby(['lr', 'weight_decay'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        best_combo = hp_combo_perf.loc[hp_combo_perf['mean'].idxmax()]
        best_hp_combos.append({
            'dataset': dataset,
            'activation': activation,
            'best_lr': best_combo['lr'],
            'best_weight_decay': best_combo['weight_decay'],
            'best_accuracy_mean': best_combo['mean'],
            'best_accuracy_std': best_combo['std'],
            'n_repeats': best_combo['count'],
        })
    best_hp_df = pd.DataFrame(best_hp_combos)
    best_hp_df.to_csv(data_dir / 'best_hyperparameter_combinations.csv', index=False)
    print(f"Created best_hyperparameter_combinations.csv with {len(best_hp_df)} rows")

    # Sensitivity analysis
    sensitivity_analysis = []
    for (dataset, activation), group in per_run_df.groupby(['dataset', 'activation']):
        lr_groups = group.groupby('lr')['accuracy'].mean()
        wd_groups = group.groupby('weight_decay')['accuracy'].mean()
        lr_sensitivity = lr_groups.max() - lr_groups.min()
        wd_sensitivity = wd_groups.max() - wd_groups.min()

        sensitivity_analysis.append({
            'dataset': dataset,
            'activation': activation,
            'lr_sensitivity': lr_sensitivity,
            'wd_sensitivity': wd_sensitivity,
            'total_sensitivity': lr_sensitivity + wd_sensitivity,
            'most_sensitive_to': 'learning_rate' if lr_sensitivity > wd_sensitivity else 'weight_decay',
            'baseline_accuracy': group['accuracy'].mean(),
            'accuracy_range': group['accuracy'].max() - group['accuracy'].min(),
        })

    sensitivity_df = pd.DataFrame(sensitivity_analysis).sort_values('total_sensitivity', ascending=False)
    sensitivity_df.to_csv(data_dir / 'hyperparameter_sensitivity_analysis.csv', index=False)
    print(f"Created hyperparameter_sensitivity_analysis.csv with {len(sensitivity_df)} rows")

    return hp_analysis_df, best_hp_df, sensitivity_df


def create_final_summary_dashboard(data_dir: Path) -> pd.DataFrame:
    print("Creating final summary dashboard...")

    overall_path = data_dir / 'overall_activation_performance.csv'
    conv_path = data_dir / 'final_convergence_analysis.csv'
    curves_path = data_dir / 'learning_curve_improvements.csv'

    for p in [overall_path, conv_path, curves_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    overall_perf = pd.read_csv(overall_path)
    convergence = pd.read_csv(conv_path)
    learning_curves = pd.read_csv(curves_path)

    dashboard_entries = []
    for activation in overall_perf['activation']:
        perf_data = overall_perf[overall_perf['activation'] == activation].iloc[0]
        conv_data = convergence[convergence['activation'] == activation]
        avg_convergence_consistency = conv_data['convergence_consistency'].mean()
        avg_overfitting = conv_data['overfitting_indicator'].mean()
        learning_data = learning_curves[learning_curves['activation'] == activation]
        avg_total_improvement = learning_data['avg_total_improvement'].mean()

        dashboard_entries.append({
            'activation': activation,
            'overall_rank': overall_perf[overall_perf['activation'] == activation].index[0] + 1,
            'avg_accuracy': perf_data['avg_accuracy_across_datasets'],
            'consistency_score': perf_data['accuracy_consistency'],
            'avg_isotropy': perf_data.get('avg_isotropy', np.nan),
            'avg_stability': perf_data.get('avg_stability', np.nan),
            'convergence_consistency': avg_convergence_consistency,
            'overfitting_tendency': avg_overfitting,
            'learning_improvement': avg_total_improvement,
            'best_dataset': perf_data['best_dataset'],
            'worst_dataset': perf_data['worst_dataset'],
            'datasets_tested': perf_data['datasets_tested'],
        })

    dashboard_df = pd.DataFrame(dashboard_entries).sort_values('overall_rank')
    dashboard_df.to_csv(data_dir / 'executive_summary_dashboard.csv', index=False)
    print(f"Created executive_summary_dashboard.csv with {len(dashboard_df)} rows")
    return dashboard_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create hyperparameter impact analyses")
    p.add_argument('--data-dir', type=str, default='.', help='Directory containing input files; outputs written here')
    p.add_argument('--skip-dashboard', action='store_true', help='Skip creating executive_summary_dashboard.csv')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    _hp_analysis_df, _best_hp_df, sensitivity_df = create_hyperparameter_impact_analysis(data_dir)

    if not args.skip_dashboard:
        try:
            _dashboard_df = create_final_summary_dashboard(data_dir)
        except FileNotFoundError as e:
            # Dashboard depends on other condensed outputs; don't fail the whole run.
            print(f"Warning: dashboard skipped due to missing inputs: {e}")

    print("\n=== Additional Summary Files Created ===")
    print("6. hyperparameter_effects_analysis.csv - LR and WD effects")
    print("7. best_hyperparameter_combinations.csv - Optimal HP settings")
    print("8. hyperparameter_sensitivity_analysis.csv - Sensitivity to HP changes")
    print("9. executive_summary_dashboard.csv - Executive summary (if inputs available)")


if __name__ == "__main__":
    main()
