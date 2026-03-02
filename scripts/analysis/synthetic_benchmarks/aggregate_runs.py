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
Aggregate multiple synthetic benchmark runs into a consolidated summary and visuals.

Inputs: one or more run directories. Each run directory may be structured as:
  - <run_dir>/synthetic_benchmark_results.json (top-level)
  - or <run_dir>/softcap_hyperparam_search/synthetic_benchmark_results.json (nested)

Outputs (written to --output-dir):
  - combined_summary.csv: per-run x dataset x activation summary
  - comparative_accuracy_by_run.png: simple comparison across runs
  - index.html: quick preview page
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_results_json(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / 'synthetic_benchmark_results.json',
        run_dir / 'softcap_hyperparam_search' / 'synthetic_benchmark_results.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search recursively but bounded
    matches = list(run_dir.glob('**/synthetic_benchmark_results.json'))
    return matches[0] if matches else None


def load_summary_from_json(results_json: Path) -> pd.DataFrame:
    with open(results_json, 'r') as f:
        data = json.load(f)
    # Expect structure similar to: { 'accuracy_scores': {dataset: {act: {mean, std}}}, 'isotropy_scores': ... }
    records = []
    acc = data.get('accuracy_scores', {})
    iso = data.get('isotropy_scores', {})
    for dataset, acts in acc.items():
        for act, metrics in acts.items():
            row = {
                'dataset': dataset,
                'activation': act,
                'accuracy_mean': metrics.get('mean', np.nan),
                'accuracy_std': metrics.get('std', np.nan),
            }
            iso_m = iso.get(dataset, {}).get(act, {})
            row['isotropy_mean'] = iso_m.get('mean', np.nan)
            row['isotropy_std'] = iso_m.get('std', np.nan)
            records.append(row)
    return pd.DataFrame(records)


def aggregate_runs(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        results_json = find_results_json(run_dir)
        if not results_json:
            print(f"Skipping {run_dir}: no synthetic_benchmark_results.json found")
            continue
        df = load_summary_from_json(results_json)
        df['run'] = run_dir.name
        frames.append(df)
    if not frames:
        raise RuntimeError("No valid runs found to aggregate")
    return pd.concat(frames, ignore_index=True)


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compare avg accuracy across datasets per activation and run
    comp = df.groupby(['run', 'activation'])['accuracy_mean'].mean().reset_index()
    plt.figure(figsize=(12, 7))
    sns.barplot(data=comp, x='activation', y='accuracy_mean', hue='run')
    plt.title('Average Accuracy Across Datasets by Activation and Run')
    plt.ylabel('Avg Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = out_dir / 'comparative_accuracy_by_run.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {plot_path}")


def write_index(out_dir: Path):
    html = [
        "<html><head><meta charset='utf-8'><title>Combined Synthetic Benchmark Report</title></head><body>",
        "<h1>Combined Synthetic Benchmark Report</h1>",
        "<ul>",
        "  <li><a href='combined_summary.csv'>combined_summary.csv</a></li>",
        "  <li><a href='comparative_accuracy_by_run.png'>comparative_accuracy_by_run.png</a></li>",
        "</ul>",
        "</body></html>",
    ]
    (out_dir / 'index.html').write_text("\n".join(html), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Aggregate multiple synthetic benchmark runs')
    p.add_argument('run_dirs', nargs='+', help='Paths to run directories to aggregate')
    p.add_argument('--output-dir', type=str, default='synthetic_benchmarks_aggregate', help='Output directory for aggregate report')
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate_runs(run_dirs)
    combined_csv = out_dir / 'combined_summary.csv'
    df.to_csv(combined_csv, index=False)
    print(f"Wrote {combined_csv}")

    make_plots(df, out_dir)
    write_index(out_dir)
    print(f"Aggregate report written to: {out_dir}")


if __name__ == '__main__':
    main()
