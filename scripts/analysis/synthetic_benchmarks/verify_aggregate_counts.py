#!/usr/bin/env python3
"""
Verify that Thrust_0/synthetic_benchmarks_aggregate_combined_with_new_reorganized
contains the union of data from Thrust_0/synthetic_benchmarks_aggregate and
synthetic_benchmarks_full_cubic_controls.

Checks performed:
- Row membership for `synthetic_benchmark_per_run_accuracies.csv` using keys (dataset, activation, seed_index)
- Row membership for `decision_boundary_areas_per_run.csv` using keys (dataset, activation, seed_index)
- Presence of raw `datasets/` files
- Presence of run directories in `merged_run_manifest.json` that correspond to raw_data dirs

Usage:
    python verify_aggregate_counts.py

Outputs a JSON report to stdout and a human summary.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
from typing import Set, Tuple, Dict


ROOT = Path(__file__).resolve().parents[3]  # e:/SoftCap/scripts/analysis/.. => e:/SoftCap

AGG_COMBINED = ROOT / "Thrust_0" / "synthetic_benchmarks_aggregate_combined_with_new_reorganized"
AGG_RAW = AGG_COMBINED / "raw_data"
AGG_MERGED = AGG_COMBINED

SRC_CONTROLS = ROOT / "synthetic_benchmarks_full_cubic_controls"
SRC_CONTROLS_RUN = SRC_CONTROLS  # files are in the folder root in this dataset

# paths we will inspect
SRC_PER_RUN_ACC = SRC_CONTROLS / "synthetic_benchmark_per_run_accuracies.csv"
AGG_PER_RUN_ACC = AGG_COMBINED / "synthetic_benchmark_per_run_accuracies.csv"

# Ingested raw runs in aggregate
INGESTED_RUN_PATTERN = "run_ingested_synthetic_benchmarks_full_cubic_controls_"


def read_per_run_accuracies(path: Path) -> Tuple[Set[Tuple[str, str, str]], int]:
    rows = set()
    if not path.exists():
        return rows, 0
    with open(path, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        # Expect columns like dataset, activation, seed_index
        for r in rdr:
            k = (r.get("dataset"), r.get("activation"), r.get("seed_index"))
            rows.add(k)
    return rows, len(rows)


def read_decision_boundary_csv(path: Path) -> Tuple[Set[Tuple[str, str, str]], int]:
    rows = set()
    if not path.exists():
        return rows, 0
    with open(path, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            k = (r.get("dataset"), r.get("activation"), r.get("seed_index"))
            rows.add(k)
    return rows, len(rows)


def read_decision_boundary_json(path: Path) -> Tuple[Set[Tuple[str, str, str]], int]:
    rows = set()
    if not path.exists():
        return rows, 0
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return rows, 0
    # data: dataset -> activation -> runs[]
    for dataset, dataset_v in data.items():
        for act, act_v in dataset_v.items():
            if isinstance(act_v, dict) and "runs" in act_v:
                for run in act_v["runs"]:
                    seed = str(run.get("seed")) if run.get("seed") is not None else ""
                    rows.add((dataset, act, seed))
    return rows, len(rows)


def list_ingested_runs(agg_raw: Path) -> Dict[str, Path]:
    d = {}
    if not agg_raw.exists():
        return d
    for sub in agg_raw.iterdir():
        if sub.is_dir() and sub.name.startswith(INGESTED_RUN_PATTERN):
            d[sub.name] = sub
    return d


def compare_sets(src: Set[Tuple[str, str, str]], target: Set[Tuple[str, str, str]]) -> Dict[str, int]:
    in_target = src.intersection(target)
    missing = src - target
    return {
        "src_total": len(src),
        "target_total": len(target),
        "in_target": len(in_target),
        "missing": len(missing),
    }


def check_datasets(src_datasets_dir: Path, agg_datasets_dir: Path) -> Dict[str, int]:
    result = {"src_files": 0, "agg_files": 0, "common_files": 0, "missing_files": 0}
    if not src_datasets_dir.exists():
        return result
    src_files = set([p.name for p in src_datasets_dir.iterdir() if p.is_file()])
    agg_files = set([p.name for p in agg_datasets_dir.iterdir() if p.is_file()]) if agg_datasets_dir.exists() else set()
    result["src_files"] = len(src_files)
    result["agg_files"] = len(agg_files)
    result["common_files"] = len(src_files.intersection(agg_files))
    result["missing_files"] = len(src_files.difference(agg_files))
    if result["missing_files"] > 0:
        result["missing_list"] = list(sorted(src_files.difference(agg_files)))
    return result


def main() -> int:
    report = {}

    # 1) Read source per-run accuracy
    src_rows, src_count = read_per_run_accuracies(SRC_PER_RUN_ACC)
    report["source_per_run_acc_count"] = src_count

    # 2) Read ingested per-run ACCs
    ingested_runs = list_ingested_runs(AGG_RAW)
    # Also read all runs under AGG_RAW to sum union
    all_runs = {p.name: p for p in AGG_RAW.iterdir() if p.is_dir()}
    ingested_rows_union = set()
    ingested_counts = {}
    for name, path in ingested_runs.items():
        f = path / "synthetic_benchmark_per_run_accuracies.csv"
        rows, n = read_per_run_accuracies(f)
        ingested_counts[name] = n
        ingested_rows_union.update(rows)
    report["ingested_runs"] = {k: v for k, v in ingested_counts.items()}
    report["ingested_rows_union_count"] = len(ingested_rows_union)

    # 3) Read aggregate merged per_run
    merged_rows, merged_count = read_per_run_accuracies(AGG_PER_RUN_ACC)
    report["merged_per_run_acc_count"] = merged_count

    # 3b) Union of per-run accuracies across all raw_data runs
    all_runs_union = set()
    all_run_counts = {}
    for name, path in all_runs.items():
        f = path / "synthetic_benchmark_per_run_accuracies.csv"
        rows, n = read_per_run_accuracies(f)
        all_run_counts[name] = n
        all_runs_union.update(rows)
    report["all_runs_union_per_run_acc_count"] = len(all_runs_union)
    report["per_run_acc_all_runs_counts"] = {k: v for k, v in all_run_counts.items()}

    # 4) Compare source -> ingested, source -> merged
    report["per_run_acc_source_in_ingested"] = compare_sets(src_rows, ingested_rows_union)
    report["per_run_acc_source_in_merged"] = compare_sets(src_rows, merged_rows)
    report["per_run_acc_all_runs_in_merged"] = compare_sets(all_runs_union, merged_rows)

    # 5) Decision boundary csv (if present) comparison
    src_boundary_csv = SRC_CONTROLS / "decision_boundary_areas_per_run.csv"
    src_boundary_json = SRC_CONTROLS / "decision_boundary_areas.json"
    src_boundary_rows_csv, src_boundary_count_csv = read_decision_boundary_csv(src_boundary_csv)
    src_boundary_rows_json, src_boundary_count_json = read_decision_boundary_json(src_boundary_json)
    # prefer csv if present else JSON
    src_boundary_rows = src_boundary_rows_csv or src_boundary_rows_json
    src_boundary_count = src_boundary_count_csv if src_boundary_count_csv > 0 else src_boundary_count_json
    report["source_db_per_run_count"] = src_boundary_count

    # ingested boundary rows union
    ingested_boundary_union = set()
    ingested_boundary_counts = {}
    for name, path in ingested_runs.items():
        f_csv = path / "decision_boundary_areas_per_run.csv"
        f_json = path / "decision_boundary_areas.json"
        rows_csv, n_csv = read_decision_boundary_csv(f_csv)
        rows_json, n_json = read_decision_boundary_json(f_json)
        rows = rows_csv or rows_json
        n = n_csv if n_csv > 0 else n_json
        ingested_boundary_counts[name] = n
        ingested_boundary_union.update(rows)
    report["ingested_db_runs"] = {k: v for k, v in ingested_boundary_counts.items()}
    report["ingested_db_union_count"] = len(ingested_boundary_union)

    # merged boundary JSON
    merged_db_json_file = AGG_COMBINED / "merged_decision_boundary_areas.json"
    merged_db_raw_json_file = AGG_RAW / "merged_decision_boundary_areas.json"
    merged_db_rows, merged_db_count = read_decision_boundary_json(merged_db_json_file)
    if merged_db_count == 0:
        merged_db_rows, merged_db_count = read_decision_boundary_json(merged_db_raw_json_file)
    report["merged_db_count"] = merged_db_count
    report["merged_db_rows_present"] = len(merged_db_rows)

    # For decision boundary comparison, compare dataset+activation ignoring seed indexes
    src_db_da = set((d, a) for (d, a, _s) in src_boundary_rows)
    ingested_db_da = set((d, a) for (d, a, _s) in ingested_boundary_union)
    merged_db_da = set((d, a) for (d, a, _s) in merged_db_rows)
    report["per_run_db_source_in_ingested_da"] = {
        "src_total": len(src_db_da),
        "ingested_total": len(ingested_db_da),
        "in_ingested": len(src_db_da.intersection(ingested_db_da)),
        "missing": len(src_db_da - ingested_db_da),
    }
    report["per_run_db_source_in_merged_da"] = {
        "src_total": len(src_db_da),
        "merged_total": len(merged_db_da),
        "in_merged": len(src_db_da.intersection(merged_db_da)),
        "missing": len(src_db_da - merged_db_da),
    }

    # 6) Datasets dir check
    src_datasets_dir = SRC_CONTROLS / "datasets"
    agg_datasets_dir = AGG_RAW / "datasets"
    report["datasets_presence"] = check_datasets(src_datasets_dir, agg_datasets_dir)

    # 7) run manifest: merged_run_manifest.json
    merged_manifest = AGG_RAW / "merged_run_manifest.json"
    if merged_manifest.exists():
        with open(merged_manifest, encoding="utf-8") as fh:
            mj = json.load(fh)
        report["merged_run_manifest_keys"] = list(mj.get("runs", {}).keys())
        report["merged_run_manifest_source_runs"] = mj.get("source_runs", [])
    else:
        report["merged_run_manifest_keys"] = []
        report["merged_run_manifest_source_runs"] = []

    # 8) present raw_data runs listing
    raw_subdirs = [d.name for d in AGG_RAW.iterdir() if d.is_dir()]
    report["raw_data_subdirs"] = raw_subdirs

    # 9) Sanity: ensure that all runs from the original control file are present under ingested runs (via manifest run name)
    # The original control folder is itself a run, and combined has run_ingested_*
    # Check that ingested run names are present in merged manifest source_runs
    report["sanity_ingested_in_manifest"] = {
        r: (any(r in s for s in report.get("merged_run_manifest_source_runs", [])) or r in report.get("merged_run_manifest_keys", []))
        for r in list(ingested_runs.keys())
    }

    # Summaries for human reading
    def print_summary():
        print("AGGREGATE INTEGRITY CHECK SUMMARY\n")
        print(f"Source control per_run_accuracies rows: {src_count}")
        print(f"Ingested union of per_run_accuracies rows (sum unique): {len(ingested_rows_union)}")
        print(f"Merged aggregate per_run_accuracies rows: {merged_count}\n")

        print("Compare source -> ingested:")
        print(json.dumps(report["per_run_acc_source_in_ingested"], indent=2))
        print("Compare source -> merged:")
        print(json.dumps(report["per_run_acc_source_in_merged"], indent=2))
        print("\nDecision boundary rows:")
        print(f"Source DB per-run rows: {src_boundary_count}")
        print(f"Ingested union DB rows: {len(ingested_boundary_union)}")
        print("Compare source DB dataset+activation -> ingested:")
        print(json.dumps(report["per_run_db_source_in_ingested_da"], indent=2))
        print("Compare source DB dataset+activation -> merged:")
        print(json.dumps(report["per_run_db_source_in_merged_da"], indent=2))
        print(f"Merged DB json count: {report.get('merged_db_count')}, present rows: {report.get('merged_db_rows_present')}")

        print("\nDataset file presence: ")
        print(json.dumps(report["datasets_presence"], indent=2))

        print("\nMerged run manifest keys: ")
        print(json.dumps(report["merged_run_manifest_keys"], indent=2))
        print("Source runs: ")
        print(json.dumps(report["merged_run_manifest_source_runs"], indent=2))

        print("\nIngested runs found on disk and listed in manifest? ")
        print(json.dumps(report["sanity_ingested_in_manifest"], indent=2))

    print_summary()

    # Save report to a file
    out = AGG_COMBINED / "verify_aggregate_counts_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to: {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
