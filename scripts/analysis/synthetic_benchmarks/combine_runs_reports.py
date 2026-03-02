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
Combine multiple synthetic benchmark runs into richer, cross-run reports.

Inputs: one or more run directories. Expected to find per-run assets at:
  - <run>/synthetic_benchmark_per_run_accuracies.csv (or under condensed/ or hp_reports/condensed/)
  - <run>/run_histories/**/*.csv (training_history_*.csv or *_history.csv)

Outputs (under --output-dir):
  - combined_per_run_accuracies.csv
  - plots/combined_accuracy_boxplot_by_activation.png (hue=dataset)
  - plots/combined_accuracy_boxplot_by_activation_and_run.png (hue=run)
  - stats/combined_heatmap_<dataset>_<activation>.png (accuracy mean across runs)
  - plots/curves/curves_<dataset>_<activation>.png (mean±SE across runs & replicates, per HP combo)
  - index.html linking everything

Note: This script complements aggregate_runs.py by producing richer visuals across runs.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_file(run_dir: Path, relative_candidates: list[str]) -> Path | None:
    for rel in relative_candidates:
        p = run_dir / rel
        if p.exists():
            return p
    # bounded recursive search fallback
    matches = list(run_dir.glob('**/' + relative_candidates[0].split('/')[-1]))
    return matches[0] if matches else None


def _find_results_json(run_dir: Path) -> Path | None:
    cands = [
        run_dir / 'synthetic_benchmark_results.json',
        run_dir / 'softcap_hyperparam_search' / 'synthetic_benchmark_results.json',
        run_dir / 'condensed' / 'synthetic_benchmark_results.json',
        run_dir / 'hp_reports' / 'condensed' / 'synthetic_benchmark_results.json',
    ]
    for c in cands:
        if c.exists():
            return c
    rec = list(run_dir.glob('**/synthetic_benchmark_results.json'))
    return rec[0] if rec else None


def gather_per_run_accuracies(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    candidates = [
        'synthetic_benchmark_per_run_accuracies.csv',
        'condensed/synthetic_benchmark_per_run_accuracies.csv',
        'hp_reports/condensed/synthetic_benchmark_per_run_accuracies.csv',
        'softcap_hyperparam_search/synthetic_benchmark_per_run_accuracies.csv',
    ]
    for rd in run_dirs:
        found = None
        for c in candidates:
            fp = rd / c
            if fp.exists():
                found = fp
                break
        if not found:
            # last-chance recursive search
            rec = list(rd.glob('**/synthetic_benchmark_per_run_accuracies.csv'))
            found = rec[0] if rec else None
        if not found:
            print(f"Warning: {rd} missing per-run accuracies; skipping")
            continue
        df = pd.read_csv(found)
        df['run'] = rd.name
        # Try to enrich with lr/weight_decay if missing but seed_index is present
        if 'lr' not in df.columns and 'weight_decay' not in df.columns and 'seed_index' in df.columns:
            rj = _find_results_json(rd)
            if rj is not None:
                try:
                    import json
                    manifest = json.loads(rj.read_text(encoding='utf-8')).get('manifest', {})
                    lr_values = [float(x) for x in str(manifest.get('lr', '0.01')).split(',')]
                    wd_values = [float(x) for x in str(manifest.get('weight_decay', '0.0')).split(',')]
                    repeats = int(manifest.get('repeats', 3)) if str(manifest.get('repeats', '')).strip() else 3
                    hp_combos = [{'lr': lr, 'weight_decay': wd} for lr in lr_values for wd in wd_values]
                    df['hp_combo_id'] = (df['seed_index'] // repeats).astype(int)
                    df['lr'] = df['hp_combo_id'].map(lambda i: hp_combos[i]['lr'] if 0 <= i < len(hp_combos) else np.nan)
                    df['weight_decay'] = df['hp_combo_id'].map(lambda i: hp_combos[i]['weight_decay'] if 0 <= i < len(hp_combos) else np.nan)
                except Exception as e:
                    print(f"Warning: could not map lr/wd for {rd}: {e}")
        frames.append(df)
    if not frames:
        raise RuntimeError('No per-run accuracies found in provided runs')
    return pd.concat(frames, ignore_index=True)


def get_unique_colors(n: int, cmap_name: str = 'Set1'):
    if n <= 9:
        cmap = plt.colormaps.get_cmap('Set1')
    elif n <= 16:
        cmap = plt.colormaps.get_cmap('Dark2')
    else:
        cmap = plt.colormaps.get_cmap('viridis')
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def load_training_histories(run_dirs: list[Path]):
    entries = {}
    # Look for per-run training histories in various locations, including longform files
    patterns = ["**/training_history_*.csv", "**/*_history.csv"]
    for rd in run_dirs:
        # First, try explicit run_histories directory
        candidates = []
        rh_dir = rd / 'run_histories'
        if rh_dir.exists():
            for pat in patterns:
                for csv in rh_dir.glob(pat):
                    candidates.append(csv)
        # Fallback: search recursively for common naming patterns
        for pat in patterns:
            for csv in rd.glob(f'**/{pat}'):
                candidates.append(csv)
        # Fallback: aggregated longform training histories
        for csv in rd.glob('**/training_histories_longform.csv'):
            candidates.append(csv)

        for csv in candidates:
                try:
                    df = pd.read_csv(csv)
                except Exception as e:
                    print(f"Skip {csv}: {e}")
                    continue
                name = csv.name
                dataset = activation = None
                lr = wd = None
                seed = None
                if name.startswith('training_history_'):
                    parts = name.replace('training_history_', '').replace('.csv', '').split('_')
                    if len(parts) >= 4:
                        dataset, activation = parts[0], parts[1]
                        lr = float(parts[2].replace('lr', ''))
                        wd_str = parts[3].replace('wd', '')
                        wd = float(wd_str)
                elif name.endswith('_history.csv'):
                    parts = name.replace('_history.csv', '').split('_')
                    if len(parts) >= 5:
                        dataset, activation = parts[0], parts[1]
                        lr = float(parts[2].replace('lr', ''))
                        wd = float(parts[3].replace('wd', ''))
                        if len(parts) >= 6 and parts[4].startswith('seed'):
                            try:
                                seed = int(parts[4].replace('seed', ''))
                            except:
                                seed = None
                if dataset is None or activation is None or lr is None or wd is None:
                    # Fallback: check if longform CSV; then split by dataset/activation group
                    if name == 'training_histories_longform.csv':
                        try:
                            df_long = pd.read_csv(csv)
                        except Exception as e:
                            print(f"Skip {csv} (longform parse error): {e}")
                            continue
                        for (dset, act), g in df_long.groupby(['dataset', 'activation']):
                            entries.setdefault(f"{dset}_{act}", []).append({
                                'data': g.sort_values('epoch'),
                                'lr': None,
                                'wd': None,
                                'seed': None,
                                'file': str(csv),
                            })
                        continue
                    else:
                        continue
                key = f"{dataset}_{activation}"
                entries.setdefault(key, []).append({
                    'data': df,
                    'lr': lr,
                    'wd': wd,
                    'seed': seed,
                    'file': str(csv),
                })
    return entries


def aggregate_curves(data_list: list[dict]):
    # group by HP combo
    hp_groups = {}
    for e in data_list:
        hp_groups.setdefault((e['lr'], e['wd']), []).append(e)
    out = []
    for (lr, wd), group in hp_groups.items():
        if not group:
            continue
        max_epochs = max((len(g['data']) for g in group if not g['data'].empty), default=0)
        if max_epochs == 0:
            continue
        tl, vl, ta, va = [], [], [], []
        for g in group:
            df = g['data']
            if df.empty:
                continue
            # Support missing columns by using .get and filling with NaNs if missing
            def col_or_nan(column_name):
                if column_name in df.columns:
                    return list(df[column_name]) + [np.nan] * (max_epochs - len(df))
                else:
                    return [np.nan] * max_epochs

            tl.append(col_or_nan('train_loss'))
            vl.append(col_or_nan('val_loss'))
            ta.append(col_or_nan('train_accuracy'))
            va.append(col_or_nan('val_accuracy'))
        if not tl:
            continue
        tl, vl, ta, va = map(np.array, (tl, vl, ta, va))
        def mean_se(arr):
            mean = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(1, np.sum(~np.isnan(arr), axis=0)))
            return mean, se
        tlm, tls = mean_se(tl)
        vlm, vls = mean_se(vl)
        tam, tas = mean_se(ta)
        vam, vas = mean_se(va)
        out.append({
            'lr': lr,
            'wd': wd,
            'n_reps': len(group),
            'epochs': np.arange(1, max_epochs + 1),
            'train_loss_mean': tlm, 'train_loss_se': tls,
            'val_loss_mean': vlm, 'val_loss_se': vls,
            'train_acc_mean': tam, 'train_acc_se': tas,
            'val_acc_mean': vam, 'val_acc_se': vas,
        })
    return out


def plot_curves(training_data: dict, out_dir: Path):
    curves_dir = out_dir / 'plots' / 'curves'
    curves_dir.mkdir(parents=True, exist_ok=True)
    for key, data_list in training_data.items():
        dataset, activation = key.split('_', 1)
        aggr = aggregate_curves(data_list)
        if not aggr:
            continue
        fig, ((ax_tl, ax_vl), (ax_ta, ax_va)) = plt.subplots(2, 2, figsize=(16, 12))
        colors = get_unique_colors(len(aggr))
        for i, e in enumerate(aggr):
            label = f"lr={e['lr']}, wd={e['wd']} (n={e['n_reps']})"
            c = colors[i]
            epochs = e['epochs']
            ax_tl.plot(epochs, e['train_loss_mean'], color=c, linewidth=2, label=label)
            ax_tl.fill_between(epochs, e['train_loss_mean'] - e['train_loss_se'], e['train_loss_mean'] + e['train_loss_se'], color=c, alpha=0.2)
            ax_vl.plot(epochs, e['val_loss_mean'], color=c, linewidth=2, label=label)
            ax_vl.fill_between(epochs, e['val_loss_mean'] - e['val_loss_se'], e['val_loss_mean'] + e['val_loss_se'], color=c, alpha=0.2)
            # Only plot accuracy curves if they exist (avoid KeyError)
            if 'train_acc_mean' in e and not np.all(np.isnan(e['train_acc_mean'])):
                ax_ta.plot(epochs, e['train_acc_mean'], color=c, linewidth=2, label=label)
                ax_ta.fill_between(epochs, e['train_acc_mean'] - e['train_acc_se'], e['train_acc_mean'] + e['train_acc_se'], color=c, alpha=0.2)
            else:
                # Hide axis if no data
                ax_ta.set_visible(False)
            if 'val_acc_mean' in e and not np.all(np.isnan(e['val_acc_mean'])):
                ax_va.plot(epochs, e['val_acc_mean'], color=c, linewidth=2, label=label)
                ax_va.fill_between(epochs, e['val_acc_mean'] - e['val_acc_se'], e['val_acc_mean'] + e['val_acc_se'], color=c, alpha=0.2)
            else:
                # Hide axis if no data
                ax_va.set_visible(False)
        for ax in (ax_tl, ax_vl, ax_ta, ax_va):
            ax.grid(True, alpha=0.3)
        ax_tl.set_title(f"{dataset.title()} - {activation} - Training Loss (Mean ± SE)")
        ax_vl.set_title(f"{dataset.title()} - {activation} - Validation Loss (Mean ± SE)")
        ax_ta.set_title(f"{dataset.title()} - {activation} - Training Accuracy (Mean ± SE)")
        ax_va.set_title(f"{dataset.title()} - {activation} - Validation Accuracy (Mean ± SE)")
        ax_tl.set_xlabel('Epoch'); ax_tl.set_ylabel('Loss'); ax_tl.set_yscale('log')
        ax_vl.set_xlabel('Epoch'); ax_vl.set_ylabel('Loss'); ax_vl.set_yscale('log')
        ax_ta.set_xlabel('Epoch'); ax_ta.set_ylabel('Accuracy')
        ax_va.set_xlabel('Epoch'); ax_va.set_ylabel('Accuracy')
        ax_tl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out = curves_dir / f"curves_{dataset}_{activation.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wrote {out}")


def plot_boxplots(per_run_df: pd.DataFrame, out_dir: Path):
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Basic: accuracy by activation across datasets
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=per_run_df, x='activation', y='accuracy', hue='dataset')
    plt.title('Accuracy by Activation Across Datasets (All Runs Combined)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = plots_dir / 'combined_accuracy_boxplot_by_activation.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    # With run as hue
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=per_run_df, x='activation', y='accuracy', hue='run')
    plt.title('Accuracy by Activation (Colored by Run)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    path2 = plots_dir / 'combined_accuracy_boxplot_by_activation_and_run.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {path1}\nWrote {path2}")
    
    # Per-learning-rate boxplots (by dataset and by run) if lr column present
    if 'lr' in per_run_df.columns:
        for lr_val in sorted(per_run_df['lr'].dropna().unique()):
            subset = per_run_df[per_run_df['lr'] == lr_val]
            if subset.empty:
                continue
            # By dataset
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=subset, x='activation', y='accuracy', hue='dataset')
            plt.title(f'Accuracy by Activation (lr={lr_val}, All Datasets)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            path_lr_ds = plots_dir / f'combined_accuracy_boxplot_by_activation_lr{lr_val}.png'
            plt.savefig(path_lr_ds, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wrote {path_lr_ds}")
            # By run
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=subset, x='activation', y='accuracy', hue='run')
            plt.title(f'Accuracy by Activation (lr={lr_val}, Colored by Run)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            path_lr_run = plots_dir / f'combined_accuracy_boxplot_by_activation_and_run_lr{lr_val}.png'
            plt.savefig(path_lr_run, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wrote {path_lr_run}")


def plot_heatmaps(per_run_df: pd.DataFrame, out_dir: Path):
    stats_dir = out_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)
    # Ensure lr/wd numeric if present
    if 'lr' in per_run_df.columns:
        per_run_df['lr'] = per_run_df['lr'].astype(float)
    if 'weight_decay' in per_run_df.columns:
        per_run_df['weight_decay'] = per_run_df['weight_decay'].astype(float)
    # If HP columns are missing, skip
    if not {'lr', 'weight_decay'} <= set(per_run_df.columns):
        print('Heatmaps skipped: missing lr/weight_decay in per-run accuracies')
        return
    for dataset in sorted(per_run_df['dataset'].unique()):
        for activation in sorted(per_run_df['activation'].unique()):
            df = per_run_df[(per_run_df['dataset'] == dataset) & (per_run_df['activation'] == activation)]
            if df.empty:
                continue
            pivot = df.groupby(['lr', 'weight_decay'])['accuracy'].mean().reset_index()
            heat = pivot.pivot_table(values='accuracy', index='lr', columns='weight_decay', aggfunc='mean')
            if heat.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.heatmap(heat, annot=True, fmt='.3f', cmap='viridis')
            plt.title(f'{dataset.title()} - {activation} - Accuracy Heatmap (Combined Runs)')
            plt.tight_layout()
            out = stats_dir / f'combined_heatmap_{dataset}_{activation.lower()}.png'
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wrote {out}")


def write_index(out_dir: Path):
    html = [
        "<html><head><meta charset='utf-8'><title>Combined Runs Reports</title></head><body>",
        "<h1>Combined Runs Reports</h1>",
        "<ul>",
        "  <li><a href='combined_per_run_accuracies.csv'>combined_per_run_accuracies.csv</a></li>",
        "  <li>Plots: <a href='plots/combined_accuracy_boxplot_by_activation.png'>accuracy by activation (by dataset)</a>, "
        "<a href='plots/combined_accuracy_boxplot_by_activation_and_run.png'>accuracy by activation (by run)</a></li>",
        "  <li>Per-LR Plots: see plots/ for combined_accuracy_boxplot_by_activation_lr*.png (one per learning rate)</li>",
        "  <li>Curves: see plots/curves/</li>",
        "  <li>Heatmaps: see stats/ (combined across runs)</li>",
        "</ul>",
        "</body></html>",
    ]
    (out_dir / 'index.html').write_text("\n".join(html), encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description='Combine multiple runs into richer reports')
    p.add_argument('run_dirs', nargs='+', help='Paths to run directories to combine')
    p.add_argument('--output-dir', type=str, default='synthetic_benchmarks_aggregate', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracies-based combined views
    per_run_df = gather_per_run_accuracies(run_dirs)
    per_run_csv = out_dir / 'combined_per_run_accuracies.csv'
    per_run_df.to_csv(per_run_csv, index=False)
    print(f"Wrote {per_run_csv}")
    plot_boxplots(per_run_df, out_dir)
    plot_heatmaps(per_run_df, out_dir)

    # Training/validation curves (aggregate across runs)
    training_data = load_training_histories(run_dirs)
    if training_data:
        plot_curves(training_data, out_dir)
    else:
        print('No training histories found; skipping curves')

    write_index(out_dir)
    print(f"Combined reports written to: {out_dir}")


if __name__ == '__main__':
    main()
