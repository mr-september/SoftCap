#!/usr/bin/env python3
"""
Merge one or more new synthetic benchmark runs into an existing aggregate safely.

Workflow:
  1. Copy new run(s) into the aggregate's raw_data/ directory
  2. Recompute combined reports (combine_runs_reports.py) and aggregated summary (aggregate_runs.py)
  3. Reorganize the combined output (reorganize_aggregate_results.py)

This script is designed to be conservative - it creates a backup of the aggregate dir before making changes.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import sys
import json
import hashlib
from typing import Dict


def parse_args():
    p = argparse.ArgumentParser(description='Merge one or more run directories into an existing aggregate')
    p.add_argument('--aggregate-dir', type=str, required=True, help='Path to existing aggregate directory (e.g., Thrust_0/synthetic_benchmarks_aggregate)')
    p.add_argument('new_runs', nargs='+', help='Paths to new run directories to merge')
    p.add_argument('--backup', action='store_true', help='Create backup of aggregated directory before modifying')
    p.add_argument('--reorganize', action='store_true', help='Run reorganization to create a final clean structure')
    p.add_argument('--dedupe-datasets', action='store_true', help='Deduplicate datasets by content before copying (default: false)')
    p.add_argument('--merge-db', action='store_true', help='Merge decision boundary areas into a combined JSON')
    p.add_argument('--merge-boundary-grids', action='store_true', help='Copy and consolidate run boundary grids into combined raw_data')
    p.add_argument('--copy-model-artifacts', action='store_true', help='Copy saved model artifacts (pt/pth/ckpt/h5/pb) from run directories into consolidated output')
    p.add_argument('--python', type=str, default=sys.executable, help='Python interpreter to run helper scripts')
    return p.parse_args()


def copy_runs_into_aggregate(aggregate_dir: Path, new_runs: list[Path]):
    raw_data_dir = aggregate_dir / 'raw_data'
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    for nr in new_runs:
        if not nr.exists():
            print(f"Warning: new run directory not found: {nr}; skipping")
            continue
        dest = raw_data_dir / nr.name
        if dest.exists():
            print(f"Warning: {dest} already exists in aggregate raw_data; skipping copy")
            continue
        print(f"Copy: {nr} -> {dest}")
        shutil.copytree(nr, dest)
    print("Copy complete")


def dedupe_and_copy_datasets(raw_runs: list[Path], combined_raw: Path) -> Dict[str, str]:
    """Copy unique dataset files from runs into combined_raw/datasets, dedupe by SHA256.

    Returns a mapping from original path to final dest path for logging and verification.
    """
    combined_datasets = combined_raw / 'datasets'
    combined_datasets.mkdir(parents=True, exist_ok=True)
    hash_map = {}  # sha256 -> dest_filename
    path_map = {}
    for r in raw_runs:
        ds_dir = r / 'datasets'
        if not ds_dir.exists():
            ds_dir = r / 'consolidated_datasets' / 'datasets'
        if not ds_dir.exists():
            continue
        for f in ds_dir.glob('*'):
            if not f.is_file():
                continue
            # compute SHA256
            h = hashlib.sha256()
            with open(f, 'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            hex_ = h.hexdigest()
            if hex_ in hash_map:
                # Already copied; skip if same content
                dest_name = hash_map[hex_]
                path_map[str(f)] = str(combined_datasets / dest_name)
                print(f"Skipping duplicate dataset {f.name} -> already present as {dest_name}")
                continue
            # Compute destination filename; handle collisions
            dest_filename = f.name
            dest_path = combined_datasets / dest_filename
            if dest_path.exists():
                # If file exists but different content, append short hash
                existing_h = hashlib.sha256(open(dest_path, 'rb').read()).hexdigest()
                if existing_h != hex_:
                    dest_filename = f"{f.stem}_{hex_[:8]}{f.suffix}"
                    dest_path = combined_datasets / dest_filename
            shutil.copy2(f, dest_path)
            hash_map[hex_] = dest_filename
            path_map[str(f)] = str(dest_path)
            print(f"Copied dataset {f.name} -> datasets/{dest_filename}")
    return path_map


def run_command(cmd: list[str], cwd: Path = Path('.')):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError('Command failed')
    else:
        print(res.stdout)
        print(res.stderr)


def main():
    args = parse_args()
    aggregate_dir = Path(args.aggregate_dir).resolve()
    new_runs = [Path(p).resolve() for p in args.new_runs]
    if not aggregate_dir.exists():
        raise FileNotFoundError(f"Aggregate directory not found: {aggregate_dir}")
    if args.backup:
        backup_dir = aggregate_dir.parent / f"{aggregate_dir.name}_backup"
        if backup_dir.exists():
            print(f"Backup dir already exists: {backup_dir}; skipping backup creation")
        else:
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(aggregate_dir, backup_dir)

    # Copy new runs into aggregate raw_data
    copy_runs_into_aggregate(aggregate_dir, new_runs)

    # Recompute combined reports using combine_runs_reports.py and aggregate_runs.py
    combine_script = Path(__file__).resolve().parents[2] / 'analysis' / 'synthetic_benchmarks' / 'combine_runs_reports.py'
    aggregate_script = Path(__file__).resolve().parents[2] / 'analysis' / 'synthetic_benchmarks' / 'aggregate_runs.py'
    reorganize_script = Path(__file__).resolve().parents[2] / 'analysis' / 'reorganize_aggregate_results.py'

    # Build run directories list: existing run_*/ directories in aggregate raw_data
    raw_runs = [p for p in (aggregate_dir / 'raw_data').iterdir() if p.is_dir() and p.name.startswith('run_')]
    print(f"Found {len(raw_runs)} run directories to combine")
    run_dirs_args = [str(p) for p in raw_runs]

    # Seed index consistency check across runs (warn on mismatch)
    def check_seed_index_consistency(raw_runs):
        print('\n🔎 Checking seed_index consistency across runs...')
        per_run_info = {}
        candidates = ['synthetic_benchmark_per_run_accuracies.csv', 'condensed/synthetic_benchmark_per_run_accuracies.csv', 'hp_reports/condensed/synthetic_benchmark_per_run_accuracies.csv', 'softcap_hyperparam_search/synthetic_benchmark_per_run_accuracies.csv']
        for r in raw_runs:
            found = None
            for c in candidates:
                fp = r / c
                if fp.exists():
                    found = fp
                    break
            if not found:
                rec = list(r.glob('**/synthetic_benchmark_per_run_accuracies.csv'))
                found = rec[0] if rec else None
            if not found:
                print(f"  ⚠️  {r.name}: no per-run accuracy CSV found; skipping seed check")
                continue
            try:
                import pandas as pd
                df = pd.read_csv(found)
            except Exception as e:
                print(f"  ⚠️  {r.name}: could not read {found}: {e}")
                continue
            if 'seed_index' not in df.columns:
                print(f"  ⚠️  {r.name}: 'seed_index' column missing; cannot validate seeds")
                continue
            unique_seeds = sorted(df['seed_index'].dropna().unique().tolist())
            per_run_info[r.name] = {
                'min': int(min(unique_seeds)) if unique_seeds else None,
                'max': int(max(unique_seeds)) if unique_seeds else None,
                'count': len(unique_seeds)
            }
            # try to read run_manifest.json if present to confirm expected repeats
            rm = r / 'run_manifest.json'
            if not rm.exists():
                rm = r / 'softcap_hyperparam_search' / 'run_manifest.json'
            if rm.exists():
                try:
                    with open(rm, 'r') as fh:
                        rmd = json.load(fh)
                        per_run_info[r.name]['manifest_repeats'] = rmd.get('repeats', rmd.get('n_repeats', None))
                except Exception:
                    per_run_info[r.name]['manifest_repeats'] = None
        # Compare across runs
        if not per_run_info:
            print('  ⚠️  No seed_index info available in any run')
            return
        # determine if all runs have same count
        counts = set(info['count'] for info in per_run_info.values())
        if len(counts) > 1:
            print('  ⚠️ Seed index count mismatch across runs:')
            for rn, info in per_run_info.items():
                print(f"    {rn}: {info['count']} seed indexes (range {info['min']}..{info['max']})")
        else:
            count_val = next(iter(counts))
            print(f"  ✅ All runs have the same number of unique seed_index values: {count_val}")
        return per_run_info

    per_run_seed_info = check_seed_index_consistency(raw_runs)

    # Create combined output in a new folder
    combined_output = aggregate_dir.parent / f"{aggregate_dir.name}_combined_with_new"
    combined_output.mkdir(parents=True, exist_ok=True)

    # Use combine_runs_reports.py to produce combined per-run CSVs and visuals
    cmd = [args.python, str(combine_script), *run_dirs_args, '--output-dir', str(combined_output)]
    try:
        run_command(cmd)
    except Exception as e:
        print(f"⚠️ Warning: combine_runs_reports.py failed: {e}")
        print("Continuing - combined CSV outputs may exist; attempting to proceed with available files.")

    # Use aggregate_runs.py to create summary CSVs
    cmd = [args.python, str(aggregate_script), *run_dirs_args, '--output-dir', str(combined_output)]
    try:
        run_command(cmd)
    except Exception as e:
        print(f"⚠️ Warning: aggregate_runs.py failed: {e}")
        print("Continuing - some summary outputs may not be fully generated.")

    # Copy raw_data merged contents into combined_output/raw_data for archiving
    # (create a merged raw_data folder, copying existing ones in)
    combined_raw = combined_output / 'raw_data'
    combined_raw.mkdir(parents=True, exist_ok=True)
    for r in raw_runs:
        dest = combined_raw / r.name
        if dest.exists():
            print(f"Skipping existing {dest}")
            continue
        print(f"Copying raw_data run {r} -> {dest}")
        shutil.copytree(r, dest)

    # Optionally, collect model artifacts from runs into combined_output/raw_data/models
    if args.copy_model_artifacts:
        artifacts_out = combined_output / 'raw_data' / 'models'
        artifacts_out.mkdir(parents=True, exist_ok=True)
        patterns = ['**/*.pt', '**/*.pth', '**/*.ckpt', '**/*.h5', '**/*.pb', '**/*weights*', '**/model_*', '**/best*']
        for r in raw_runs:
            found_any = False
            run_out = artifacts_out / r.name
            for p in patterns:
                for f in r.glob(p):
                    if f.is_file():
                        run_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, run_out / f.name)
                        print(f"Copied model artifact for {r.name}: {f} -> {run_out / f.name}")
                        found_any = True
            if not found_any:
                print(f"No recognized model artifacts found for run: {r.name}")

    # Deduplicate and copy dataset files into combined_output/raw_data/datasets (optional)
    datasets_map = {}
    if args.dedupe_datasets:
        print('\n📁 Deduplicating and copying datasets into combined output')
        datasets_map = dedupe_and_copy_datasets(raw_runs, combined_raw)
    else:
        print('\n📁 Dataset deduplication skipped (use --dedupe-datasets to enable)')

    # Merge decision boundary areas JSONs from runs into combined_output/raw_data/merged_decision_boundary_areas.json
    def merge_decision_boundary_areas(raw_runs, out_path):
        import math
        merged = {}
        for r in raw_runs:
            db = r / 'decision_boundary_areas.json'
            if not db.exists():
                db = r / 'hp_reports' / 'decision_boundary_areas.json'
            if not db.exists():
                continue
            try:
                with open(db, 'r') as fh:
                    d = json.load(fh)
            except Exception as e:
                print(f"  ⚠️  Could not read decision boundary areas from {db}: {e}")
                continue
            for dataset, acts in d.items():
                merged.setdefault(dataset, {})
                for act, val in acts.items():
                    merged[dataset].setdefault(act, {'runs': []})
                    # Append run entries if 'runs' present
                    if 'runs' in val and isinstance(val['runs'], list):
                        merged[dataset][act]['runs'].extend(val['runs'])
        # Recompute summary aggregates
        for dataset, acts in merged.items():
            for act, val in acts.items():
                runs = val.get('runs', [])
                if not runs:
                    continue
                # Recompute numeric summary metrics: total_bbox_area, threshold_area_mean/expected_area_mean and ratios
                total_bbox_areas = []
                threshold_sums = []
                expected_sums = []
                threshold_ratios = []
                expected_ratios = []
                for rentry in runs:
                    bbox = rentry.get('bbox', {})
                    tb = None
                    if bbox and 'x_min' in bbox and 'x_max' in bbox and 'y_min' in bbox and 'y_max' in bbox:
                        tb = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                    if tb is not None:
                        total_bbox_areas.append(tb)
                    thr = rentry.get('threshold_area', {})
                    exp = rentry.get('expected_area', {})
                    # compute total threshold/expected across classes if available
                    thr_total = None
                    exp_total = None
                    if 'class1' in thr and 'class0' in thr:
                        thr_total = thr.get('class1', 0.0) + thr.get('class0', 0.0)
                    if 'class1' in exp and 'class0' in exp:
                        exp_total = exp.get('class1', 0.0) + exp.get('class0', 0.0)
                    if thr_total is not None:
                        threshold_sums.append(thr_total)
                    if exp_total is not None:
                        expected_sums.append(exp_total)
                    # ratios
                    if 'class1_ratio' in thr:
                        threshold_ratios.append(thr.get('class1_ratio'))
                    if 'class1_ratio' in exp:
                        expected_ratios.append(exp.get('class1_ratio'))
                # Create summary
                summary = {}
                import numpy as np
                if total_bbox_areas:
                    summary['total_bbox_area'] = float(np.mean(total_bbox_areas))
                if threshold_sums:
                    summary['threshold_area_mean'] = float(np.mean(threshold_sums))
                    summary['threshold_area_std'] = float(np.std(threshold_sums, ddof=1) if len(threshold_sums) > 1 else 0.0)
                if expected_sums:
                    summary['expected_area_mean'] = float(np.mean(expected_sums))
                    summary['expected_area_std'] = float(np.std(expected_sums, ddof=1) if len(expected_sums) > 1 else 0.0)
                if threshold_ratios:
                    summary['threshold_area_mean_ratio'] = float(np.mean(threshold_ratios))
                if expected_ratios:
                    summary['expected_area_mean_ratio'] = float(np.mean(expected_ratios))
                merged[dataset][act]['summary'] = summary
        # Save
        if merged:
            with open(out_path, 'w') as fh:
                json.dump(merged, fh, indent=2)
            print(f"Wrote merged decision boundary areas to {out_path}")

    merged_db_out = combined_output / 'raw_data' / 'merged_decision_boundary_areas.json'
    if args.merge_db:
        merge_decision_boundary_areas(raw_runs, merged_db_out)
    else:
        print('\n📁 Decision boundary areas merging skipped (use --merge-db to enable)')

    # Merge boundary grid directories if requested
    if args.merge_boundary_grids:
        grid_out = combined_output / 'raw_data' / 'consolidated_boundary_grids'
        grid_out.mkdir(parents=True, exist_ok=True)
        for r in raw_runs:
            src_grid = r / 'boundary_grids'
            if not src_grid.exists():
                continue
            for f in src_grid.glob('**/*'):
                if f.is_dir():
                    continue
                dest = grid_out / r.name / f.relative_to(src_grid)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        print(f"Copied boundary grids into {grid_out}")
    else:
        print('\n📁 Boundary grid merge skipped (use --merge-boundary-grids to enable)')

    # Create a merged_synthetic_benchmark_results.json based on combined_summary.csv
    combined_summary_csv = combined_output / 'combined_summary.csv'
    if combined_summary_csv.exists():
        import pandas as pd
        df = pd.read_csv(combined_summary_csv)
        merged_results = {
            'accuracy_scores': {},
            'isotropy_scores': {},
            'stability_scores': {},
            'metadata': {
                'merged_from': [str(p) for p in raw_runs],
            }
        }
        for _, row in df.iterrows():
            ds = row['dataset']
            act = row['activation']
            merged_results['accuracy_scores'].setdefault(ds, {})[act] = {
                'mean': float(row.get('accuracy_mean', float('nan'))),
                'std': float(row.get('accuracy_std', float('nan'))),
            }
            # When additional isotropy/stability columns are present, copy them
            if 'isotropy_mean' in row:
                merged_results['isotropy_scores'].setdefault(ds, {})[act] = {'mean': float(row.get('isotropy_mean', float('nan'))), 'std': float(row.get('isotropy_std', float('nan')) if 'isotropy_std' in row else float('nan'))}
            if 'stability' in row:
                merged_results['stability_scores'].setdefault(ds, {})[act] = {'mean': float(row.get('stability', float('nan'))), 'std': float(row.get('stability', float('nan')))}

        output_json = combined_output / 'raw_data' / 'merged_synthetic_benchmark_results.json'
        with open(output_json, 'w') as f:
            json.dump(merged_results, f, indent=2)
        print(f"Wrote merged_synthetic_benchmark_results.json to {output_json}")

    # Create a merged_run_manifest.json from individual run manifests (if present)
    merged_manifest = {
        'runs': {},
        'source_runs': [],
        'activations': [],
        'datasets': [],
        'n_samples': None,
        'epochs': None,
    }
    for r in raw_runs:
        mf = r / 'run_manifest.json'
        if mf.exists():
            try:
                with open(mf, 'r') as f:
                    d = json.load(f)
            except Exception:
                d = {}
        else:
            d = {}
        merged_manifest['runs'][r.name] = d
        merged_manifest['source_runs'].append(str(r))
        # Extend activations/datasets if present
        if 'activations' in d:
            for a in d['activations']:
                if a not in merged_manifest['activations']:
                    merged_manifest['activations'].append(a)
        if 'datasets' in d:
            for ds in d['datasets']:
                if ds not in merged_manifest['datasets']:
                    merged_manifest['datasets'].append(ds)
        if merged_manifest['n_samples'] is None and 'n_samples' in d:
            merged_manifest['n_samples'] = d['n_samples']
        if merged_manifest['epochs'] is None and 'epochs' in d:
            merged_manifest['epochs'] = d['epochs']

    # write the merged manifest to combined raw_data
    merged_manifest_path = combined_output / 'raw_data' / 'merged_run_manifest.json'
    with open(merged_manifest_path, 'w') as f:
        json.dump(merged_manifest, f, indent=2)
    print(f"Wrote merged_run_manifest.json to {merged_manifest_path}")

    # Merge training histories longform CSVs into merged_run_histories.csv
    merged_histories = None
    for r in raw_runs:
        th = r / 'training_histories_longform.csv'
        if not th.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(th)
        except Exception as e:
            print(f"Warning: Could not read training histories from {th}: {e}")
            continue
        if merged_histories is None:
            merged_histories = df
        else:
            merged_histories = pd.concat([merged_histories, df], ignore_index=True)
    if merged_histories is not None:
        hist_out = combined_output / 'raw_data' / 'merged_run_histories.csv'
        merged_histories.to_csv(hist_out, index=False)
        print(f"Wrote merged_run_histories.csv to {hist_out}")

    # Reorganize if requested
    if args.reorganize:
        cmd = [args.python, str(reorganize_script), '--source-dir', str(combined_output), '--backup']
        run_command(cmd)

    print("Merge complete; review the combined output and, if satisfied, replace the aggregate directory as needed.")
    # Print a small summary of operations
    print('\n✅ Merge summary:')
    print(f"  - Runs processed: {len(raw_runs)}")
    if args.dedupe_datasets:
        print(f"  - Unique datasets copied: {len(datasets_map)}")
    if per_run_seed_info:
        print('  - Seed index info by run:')
        for rn, info in per_run_seed_info.items():
            manifest_repeats = info.get('manifest_repeats', 'N/A')
            print(f"    {rn}: seeds {info['count']} (range {info['min']}..{info['max']}), manifest_repeats={manifest_repeats}")


if __name__ == '__main__':
    main()
