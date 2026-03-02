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
Ingest an aggregated folder (e.g., synthetic_benchmarks_full_cubic_controls) as a new run into an aggregate and merge everything.
Creates a temporary run_* directory from an aggregate folder, calls merge_runs_into_aggregate.py, and (optionally) replaces the aggregate dir with the reorganized result.
"""
import argparse
import shutil
import subprocess
from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description='Ingest an aggregate folder into an aggregate and merge')
    p.add_argument('--aggregate-dir', type=str, required=True, help='Path to target aggregate directory (e.g., Thrust_0/synthetic_benchmarks_aggregate)')
    p.add_argument('--aggregate-folder', type=str, required=True, help='Path to the aggregate folder to ingest (e.g., synthetic_benchmarks_full_cubic_controls)')
    p.add_argument('--backup', action='store_true', help='Create backup of aggregate dir before merging')
    p.add_argument('--dedupe-datasets', action='store_true', help='Deduplicate datasets by content before copying into combined')
    p.add_argument('--merge-db', action='store_true', help='Merge decision boundary areas JSONs')
    p.add_argument('--merge-boundary-grids', action='store_true', help='Merge boundary grid directories')
    p.add_argument('--replace', action='store_true', help='If set, replace the target aggregate with the reorganized result')
    p.add_argument('--split-aggregate', action='store_true', help='Split the ingested aggregate into run_ directories per (hp_combo_id|lr+weight_decay) and seed_index')
    p.add_argument('--include-models', action='store_true', help='Attempt to include model artifact files into created run directories (heuristic copy)')
    p.add_argument('--python', type=str, default=sys.executable, help='Python interpreter to run helper scripts')
    return p.parse_args()


def create_run_from_aggregate(aggregate_folder: Path, tmp_base: Path, split_aggregate: bool = False, include_models: bool = False) -> list[Path]:
    """Create a new run directory from an aggregate folder and return the path to it.

    The created directory is named run_ingested_{slug}_{timestamp} under tmp_base.
    """
    # Create a place to collect created run dirs
    created_runs: list[Path] = []
    slug = aggregate_folder.name.replace(' ', '_').replace('-', '_')
    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    if not split_aggregate:
        run_name = f"run_ingested_{slug}_{now}"
        run_dir = tmp_base / run_name
        if run_dir.exists():
            raise FileExistsError(f"Run dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    # Copy typical files that a run contains (common variable used in both paths)
    files_to_copy = [
        'synthetic_benchmark_per_run_accuracies.csv',
        'training_histories_longform.csv',
        'synthetic_benchmark_results.json',
        'decision_boundary_areas.json',
        'run_manifest.json',
        'packages.csv',
    ]
    if not split_aggregate:
        # copy typical files that a run contains
        for f in files_to_copy:
            src = aggregate_folder / f
            if src.exists():
                print(f"Copying file {src} -> {run_dir / src.name}")
                shutil.copy2(src, run_dir / src.name)
    # Copy datasets and boundary_grids if they exist
    ds = aggregate_folder / 'datasets'
    if not ds.exists():
        # also accept 'consolidated_datasets/datasets' or plain 'consolidated_datasets'
        ds_alt = aggregate_folder / 'consolidated_datasets'
        if ds_alt.exists():
            ds = ds_alt
    if not split_aggregate and ds.exists():
        dest_ds = run_dir / 'datasets'
        shutil.copytree(ds, dest_ds)
        print(f"Copied datasets -> {dest_ds}")
    bg = aggregate_folder / 'boundary_grids'
    if not bg.exists():
        bg_alt = aggregate_folder / 'consolidated_boundary_grids'
        if bg_alt.exists():
            bg = bg_alt
    if not split_aggregate and bg.exists():
        dest_bg = run_dir / 'boundary_grids'
        shutil.copytree(bg, dest_bg)
        print(f"Copied boundary_grids -> {dest_bg}")
    # If no run_manifest.json but the aggregate has one under a different name (rare), attempt to create a run_manifest
    if not split_aggregate and not (run_dir / 'run_manifest.json').exists():
        # Try to build a minimal run_manifest from the aggregate run_manifest.json
        src_rm = aggregate_folder / 'run_manifest.json'
        if src_rm.exists():
            try:
                with open(src_rm, 'r') as fh:
                    d = json.load(fh)
            except Exception:
                d = {}
            # build run manifest keys used by combine/aggregate scripts
            run_manifest = {}
            for k in ('timestamp', 'n_samples', 'epochs', 'repeats', 'boundary_resolution', 'activations', 'lr', 'weight_decay', 'git_commit'):
                if k in d:
                    run_manifest[k] = d[k]
            # Place activation/dataset info, default logs
            with open(run_dir / 'run_manifest.json', 'w') as fh:
                json.dump(run_manifest, fh, indent=2)
            print(f"Created minimal run_manifest.json from aggregate manifest: {run_dir / 'run_manifest.json'}")
        # continue to allow return below

    if not split_aggregate:
        created_runs.append(run_dir)
        return created_runs

    # split_aggregate == True path: create one run directory per unique hp/seed combination
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError('Missing pandas dependency; required for splitting aggregates: ' + str(e))
    # locate the per-run accuracies CSV in the aggregate folder
    csv_candidates = [aggregate_folder / 'synthetic_benchmark_per_run_accuracies.csv', aggregate_folder / 'per_run' / 'synthetic_benchmark_per_run_accuracies.csv']
    found = None
    for c in csv_candidates:
        if c.exists():
            found = c
            break
    if not found:
        # try recursing
        rec = list(aggregate_folder.glob('**/synthetic_benchmark_per_run_accuracies.csv'))
        if rec:
            found = rec[0]
    if not found:
        raise FileNotFoundError('Per-run CSV required for splitting not found in aggregate folder: ' + str(aggregate_folder))
    df = pd.read_csv(found)
    # determine grouping keys
    if 'hp_combo_id' in df.columns:
        group_cols = ['hp_combo_id', 'seed_index'] if 'seed_index' in df.columns else ['hp_combo_id']
    else:
        # fallback to lr + weight_decay
        group_cols = ['lr', 'weight_decay'] if 'lr' in df.columns and 'weight_decay' in df.columns else ['activation']
        if 'seed_index' in df.columns:
            group_cols.append('seed_index')
    print(f"Splitting aggregate into runs grouped by: {group_cols}")
    grouped = df.groupby(group_cols)
    created_runs = []
    for idx, group in enumerate(grouped):
        key, rows = group
        run_suffix = f"{idx:03d}"
        run_name = f"run_ingested_{slug}_{now}_{run_suffix}"
        run_dir = tmp_base / run_name
        if run_dir.exists():
            print(f"Warning: run dir already exists (skipping): {run_dir}")
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        # Write the subset per-run CSV
        per_csv = run_dir / 'synthetic_benchmark_per_run_accuracies.csv'
        rows.to_csv(per_csv, index=False)
        print(f"Created run: {run_dir} with {len(rows)} rows -> {per_csv}")
        # If training histories exist in aggregate, try to split and copy subset
        th_candidates = [aggregate_folder / 'training_histories_longform.csv', aggregate_folder / 'merged_run_histories.csv']
        for t in th_candidates:
            if t.exists():
                try:
                    df_th = pd.read_csv(t)
                    # determine the seed index col to filter by
                    if 'seed_index' in rows.columns and 'seed_index' in df_th.columns:
                        seeds = rows['seed_index'].unique().tolist()
                        df_th_sub = df_th[df_th['seed_index'].isin(seeds)]
                    else:
                        df_th_sub = df_th
                    if not df_th_sub.empty:
                        (run_dir / 'training_histories_longform.csv').write_text(df_th_sub.to_csv(index=False))
                except Exception:
                    pass
        # copy small files
        for f in files_to_copy:
            src = aggregate_folder / f
            if src.exists():
                shutil.copy2(src, run_dir / src.name)
        # Copy datasets (optional) — copy the whole datasets tree, dedupe later in merge step if enabled
        if ds.exists():
            dest_ds = run_dir / 'datasets'
            shutil.copytree(ds, dest_ds)
        if bg.exists():
            dest_bg = run_dir / 'boundary_grids'
            shutil.copytree(bg, dest_bg)
        # Build a run_manifest minimal
        manifest = {
            'created_from_aggregate': str(aggregate_folder.name),
            'group_key': key if not isinstance(key, tuple) else list(key),
            'repeats': 1,
        }
        # try to copy some meta fields from aggregate run_manifest.json
        src_rm = aggregate_folder / 'run_manifest.json'
        if src_rm.exists():
            try:
                with open(src_rm, 'r') as fh:
                    d = json.load(fh)
                    for k in ('n_samples', 'epochs', 'boundary_resolution'):
                        if k in d:
                            manifest[k] = d[k]
            except Exception:
                pass
        # Normalize values in manifest for JSON serialization (convert numpy scalars if present)
        def norm_val(v):
            # if it's a tuple/list, normalize each entry
            if isinstance(v, (list, tuple)):
                return [norm_val(i) for i in v]
            # attempt to use pandas/numpy scalars .item()
            if hasattr(v, 'item'):
                try:
                    return v.item()
                except Exception:
                    pass
            return v

        manifest['group_key'] = norm_val(manifest['group_key'])
        with open(run_dir / 'run_manifest.json', 'w') as fh:
            json.dump(manifest, fh, indent=2)
        # optionally try to copy model artifacts heuristically
        if include_models:
            patterns = ['**/*.pt', '**/*.pth', '**/*.ckpt', '**/*.h5', '**/*.pb', '**/model_*', '**/best*']
            found_models = []
            for p in patterns:
                for f in aggregate_folder.glob(p):
                    if f.is_file():
                        found_models.append(f)
            if found_models:
                destm = run_dir / 'models'
                destm.mkdir(parents=True, exist_ok=True)
                for f in found_models:
                    shutil.copy2(f, destm / f.name)
        created_runs.append(run_dir)
    return created_runs


def run_merge(aggregate_dir: Path, new_run_paths: list[Path], args: argparse.Namespace):
    # Build path to merge script
    merge_script = Path(__file__).resolve().parent / 'merge_runs_into_aggregate.py'
    cmd = [args.python, str(merge_script), '--aggregate-dir', str(aggregate_dir)]
    # Flags
    if args.backup:
        cmd.append('--backup')
    if args.dedupe_datasets:
        cmd.append('--dedupe-datasets')
    if args.merge_db:
        cmd.append('--merge-db')
    if args.merge_boundary_grids:
        cmd.append('--merge-boundary-grids')
    if args.include_models:
        cmd.append('--copy-model-artifacts')
    # We want reorganize to be performed so pass as a flag
    cmd.append('--reorganize')
    # pass all run dirs
    for nr in new_run_paths:
        cmd.append(str(nr))
    print(f"Running merge helper: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError('merge_runs_into_aggregate invocation failed')
    else:
        print('Merge helper completed successfully')


def finalize_replace(aggregate_dir: Path, reorganized_dir: Path, args: argparse.Namespace):
    """Replace the original aggregate with reorganized_dir safely (backing up)"""
    if not reorganized_dir.exists():
        raise FileNotFoundError('Reorganized dir not found: ' + str(reorganized_dir))
    backup_dir = aggregate_dir.parent / f"{aggregate_dir.name}_backup_before_replace"
    if not backup_dir.exists():
        print(f"Backing up original aggregate: {aggregate_dir} -> {backup_dir}")
        shutil.copytree(aggregate_dir, backup_dir)
    # Rename original to old name and move new
    final_aggregate_new = aggregate_dir.parent / f"{aggregate_dir.name}_replaced_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    print(f"Renaming original aggregate to: {final_aggregate_new}")
    aggregate_dir.rename(final_aggregate_new)
    print(f"Moving reorganized result to: {aggregate_dir}")
    shutil.copytree(reorganized_dir, aggregate_dir)
    print('Replace complete. The original aggregate is preserved as: ' + str(final_aggregate_new))


def main():
    args = parse_args()
    aggregate_dir = Path(args.aggregate_dir).resolve()
    aggregate_folder = Path(args.aggregate_folder).resolve()
    if not aggregate_dir.exists():
        raise FileNotFoundError(f"Aggregate directory not found: {aggregate_dir}")
    if not aggregate_folder.exists():
        raise FileNotFoundError(f"Aggregate folder to ingest not found: {aggregate_folder}")

    # Create tmp base
    tmp_base = Path(__file__).resolve().parent.parent / 'tmp_ingests'
    tmp_base.mkdir(parents=True, exist_ok=True)

    run_dirs = create_run_from_aggregate(aggregate_folder, tmp_base, args.split_aggregate, args.include_models)

    try:
        run_merge(aggregate_dir, run_dirs, args)
    except Exception as e:
        print(f"Error during merge: {e}")
        raise

    # Reorganized output will be created under aggregate_dir.parent / '<aggregate_name>_combined_with_new_reorganized'
    reorganized_dir = aggregate_dir.parent / f"{aggregate_dir.name}_combined_with_new_reorganized"

    # Run a quick smoke test to validate the merge output
    smoke_test = Path(__file__).resolve().parent / 'smoke_tests' / 'check_cubic_notch_merge.py'
    if smoke_test.exists():
        print('\n🔎 Running smoke test to validate merge...')
        cmd = [args.python, str(smoke_test)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            print('Smoke test failed; please inspect logs and combined outputs.')
        else:
            print('Smoke test passed')

    if args.replace:
        print('\n⚠️ Replace option set; replacing aggregate with reorganized results (with backup).')
        finalize_replace(aggregate_dir, reorganized_dir, args)

    print('\n✅ Ingest and merge complete. Reorganized output: ' + str(reorganized_dir))


if __name__ == '__main__':
    main()
