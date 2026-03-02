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
Create condensed summary CSV files from synthetic benchmark results.

This version is location-agnostic: pass --data-dir to point at a folder
containing the required inputs (e.g. training_histories_longform.csv,
synthetic_benchmark_summary.csv). Outputs are written alongside inputs.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def create_condensed_training_metrics(data_dir: Path) -> pd.DataFrame:
    """Create condensed training metrics with key milestone epochs."""
    print("Creating condensed training metrics...")

    # Try multiple possible locations for training histories
    possible_files = [
        data_dir / 'training_histories_longform.csv',
        data_dir / 'raw_data' / 'merged_run_histories.csv',
        data_dir / 'merged_run_histories.csv'
    ]
    
    df_path = None
    for path in possible_files:
        if path.exists():
            df_path = path
            break
    
    if df_path is None:
        raise FileNotFoundError(f"Missing required training histories file. Looked in: {possible_files}")
    
    df = pd.read_csv(df_path)

    # Define key milestone epochs (early, mid, late training)
    key_epochs = [0, 9, 24, 49, 99, 199, 299, 399, 499]
    milestone_df = df[df['epoch'].isin(key_epochs)].copy()

    condensed_metrics = []
    for (dataset, activation, epoch), group in milestone_df.groupby(['dataset', 'activation', 'epoch']):
        metrics = {
            'dataset': dataset,
            'activation': activation,
            'epoch': epoch,
            'train_loss_mean': group['train_loss'].mean(),
            'train_loss_std': group['train_loss'].std(),
            'val_loss_mean': group['val_loss'].mean(),
            'val_loss_std': group['val_loss'].std(),
            'train_accuracy_mean': group['train_accuracy'].mean(),
            'train_accuracy_std': group['train_accuracy'].std(),
            'val_accuracy_mean': group['val_accuracy'].mean(),
            'val_accuracy_std': group['val_accuracy'].std(),
            'n_runs': len(group),
        }
        condensed_metrics.append(metrics)

    condensed_df = pd.DataFrame(condensed_metrics)
    out_path = data_dir / 'condensed_training_milestones.csv'
    condensed_df.to_csv(out_path, index=False)
    print(f"Created {out_path.name} with {len(condensed_df)} rows")
    return condensed_df


def create_performance_ranking_summary(data_dir: Path):
    """Create performance ranking and statistical summary."""
    print("Creating performance ranking summary...")

    summary_path = data_dir / 'synthetic_benchmark_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required file: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    rankings = []
    for dataset in summary_df['dataset'].unique():
        dataset_data = summary_df[summary_df['dataset'] == dataset].copy()
        dataset_data = dataset_data.sort_values('accuracy_mean', ascending=False)

        for i, (_, row) in enumerate(dataset_data.iterrows()):
            ranking_entry = {
                'dataset': dataset,
                'activation': row['activation'],
                'rank': i + 1,
                'accuracy_mean': row['accuracy_mean'],
                'accuracy_std': row['accuracy_std'],
                'accuracy_lower_ci': row['accuracy_mean'] - 1.96 * row['accuracy_std'],
                'accuracy_upper_ci': row['accuracy_mean'] + 1.96 * row['accuracy_std'],
                'isotropy_mean': row.get('isotropy_mean', np.nan),
                'stability': row.get('stability', np.nan),
            }
            rankings.append(ranking_entry)

    ranking_df = pd.DataFrame(rankings)
    ranking_df.to_csv(data_dir / 'performance_rankings_by_dataset.csv', index=False)
    print(f"Created performance_rankings_by_dataset.csv with {len(ranking_df)} rows")

    overall_summary = []
    for activation in summary_df['activation'].unique():
        activation_data = summary_df[summary_df['activation'] == activation]
        summary_entry = {
            'activation': activation,
            'avg_accuracy_across_datasets': activation_data['accuracy_mean'].mean(),
            'accuracy_consistency': 1.0 / (activation_data['accuracy_std'].mean() + 1e-8),
            'avg_isotropy': activation_data['isotropy_mean'].mean() if 'isotropy_mean' in activation_data else np.nan,
            'avg_stability': activation_data['stability'].mean() if 'stability' in activation_data else np.nan,
            'datasets_tested': len(activation_data),
            'best_dataset': activation_data.loc[activation_data['accuracy_mean'].idxmax(), 'dataset'],
            'worst_dataset': activation_data.loc[activation_data['accuracy_mean'].idxmin(), 'dataset'],
        }
        overall_summary.append(summary_entry)

    overall_df = pd.DataFrame(overall_summary).sort_values('avg_accuracy_across_datasets', ascending=False)
    overall_df.to_csv(data_dir / 'overall_activation_performance.csv', index=False)
    print(f"Created overall_activation_performance.csv with {len(overall_df)} rows")

    return ranking_df, overall_df


def create_convergence_analysis(data_dir: Path) -> pd.DataFrame:
    """Create convergence quality analysis."""
    print("Creating convergence analysis...")

    # Try multiple possible locations
    possible_files = [
        data_dir / 'training_histories_longform.csv',
        data_dir / 'raw_data' / 'merged_run_histories.csv',
        data_dir / 'merged_run_histories.csv'
    ]
    
    df_path = None
    for path in possible_files:
        if path.exists():
            df_path = path
            break
    
    if df_path is None:
        raise FileNotFoundError(f"Missing required training histories file. Looked in: {possible_files}")

    df = pd.read_csv(df_path)
    final_epoch = df['epoch'].max()
    final_results = df[df['epoch'] == final_epoch].copy()

    convergence_analysis = []
    for (dataset, activation), group in final_results.groupby(['dataset', 'activation']):
        val_acc_final = group['val_accuracy']
        train_loss_final = group['train_loss']
        val_loss_final = group['val_loss']

        analysis_entry = {
            'dataset': dataset,
            'activation': activation,
            'final_val_accuracy_mean': val_acc_final.mean(),
            'final_val_accuracy_std': val_acc_final.std(),
            'final_train_loss_mean': train_loss_final.mean(),
            'final_train_loss_std': train_loss_final.std(),
            'final_val_loss_mean': val_loss_final.mean(),
            'final_val_loss_std': val_loss_final.std(),
            'convergence_consistency': 1.0 / (val_acc_final.std() + 1e-8),
            'overfitting_indicator': (train_loss_final.mean() - val_loss_final.mean()),
            'n_runs': len(group),
        }
        convergence_analysis.append(analysis_entry)

    convergence_df = pd.DataFrame(convergence_analysis)
    convergence_df.to_csv(data_dir / 'final_convergence_analysis.csv', index=False)
    print(f"Created final_convergence_analysis.csv with {len(convergence_df)} rows")
    return convergence_df


def create_learning_curve_summary(data_dir: Path) -> pd.DataFrame:
    """Create a learning curve summary showing improvement rates."""
    print("Creating learning curve summary...")

    # Try multiple possible locations
    possible_files = [
        data_dir / 'training_histories_longform.csv',
        data_dir / 'raw_data' / 'merged_run_histories.csv',
        data_dir / 'merged_run_histories.csv'
    ]
    
    df_path = None
    for path in possible_files:
        if path.exists():
            df_path = path
            break
    
    if df_path is None:
        raise FileNotFoundError(f"Missing required training histories file. Looked in: {possible_files}")

    df = pd.read_csv(df_path)
    learning_curves = []
    for (dataset, activation, run), group in df.groupby(['dataset', 'activation', 'run']):
        group_sorted = group.sort_values('epoch')
        initial_val_acc = group_sorted.iloc[0]['val_accuracy']
        final_val_acc = group_sorted.iloc[-1]['val_accuracy']

        epochs_of_interest = [50, 100, 200, 300, 400, 499]
        improvements = {}
        for target_epoch in epochs_of_interest:
            epoch_data = group_sorted[group_sorted['epoch'] <= target_epoch]
            if len(epoch_data) > 0:
                acc_at_epoch = epoch_data.iloc[-1]['val_accuracy']
                improvement = acc_at_epoch - initial_val_acc
                improvements[f'improvement_by_epoch_{target_epoch}'] = improvement

        curve_entry = {
            'dataset': dataset,
            'activation': activation,
            'run': run,
            'initial_val_accuracy': initial_val_acc,
            'final_val_accuracy': final_val_acc,
            'total_improvement': final_val_acc - initial_val_acc,
            **improvements,
        }
        learning_curves.append(curve_entry)

    curves_df = pd.DataFrame(learning_curves)

    aggregated_curves = []
    for (dataset, activation), group in curves_df.groupby(['dataset', 'activation']):
        agg_entry = {
            'dataset': dataset,
            'activation': activation,
            'avg_total_improvement': group['total_improvement'].mean(),
            'total_improvement_std': group['total_improvement'].std(),
            'avg_final_accuracy': group['final_val_accuracy'].mean(),
            'n_runs': len(group),
        }
        for col in group.columns:
            if col.startswith('improvement_by_epoch_'):
                agg_entry[f'avg_{col}'] = group[col].mean()
                agg_entry[f'{col}_std'] = group[col].std()
        aggregated_curves.append(agg_entry)

    agg_curves_df = pd.DataFrame(aggregated_curves)
    agg_curves_df.to_csv(data_dir / 'learning_curve_improvements.csv', index=False)
    print(f"Created learning_curve_improvements.csv with {len(agg_curves_df)} rows")
    return agg_curves_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create condensed summaries for synthetic benchmarks")
    p.add_argument('--data-dir', type=str, default='.', help='Directory containing input CSVs; outputs written here')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Writing condensed summaries in: {data_dir}")
    _ = create_condensed_training_metrics(data_dir)
    _ranking_df, _overall_df = create_performance_ranking_summary(data_dir)
    _ = create_convergence_analysis(data_dir)
    _ = create_learning_curve_summary(data_dir)

    print("\n=== Summary of Created Files ===")
    print("1. condensed_training_milestones.csv - Key training milestones with aggregated metrics")
    print("2. performance_rankings_by_dataset.csv - Performance rankings and confidence intervals")
    print("3. overall_activation_performance.csv - Cross-dataset activation performance")
    print("4. final_convergence_analysis.csv - Convergence quality and stability")
    print("5. learning_curve_improvements.csv - Learning rate and improvement analysis")


if __name__ == "__main__":
    main()
