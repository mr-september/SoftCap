"""
Aggregate results from scaling benchmarks.

Reads *_summary.json files from vision and/or NLP result directories,
computes mean ± std across seeds, performs pairwise t-tests, and
outputs Markdown comparison tables.

Usage (from repo root):
    python benchmarks/analysis/aggregate_results.py

    # Vision only:
    python benchmarks/analysis/aggregate_results.py --vision-dir benchmarks/vision/results

    # NLP only:
    python benchmarks/analysis/aggregate_results.py --nlp-dir benchmarks/nlp/results

    # Both + export LaTeX:
    python benchmarks/analysis/aggregate_results.py --latex
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.metrics import aggregate_seeds, pairwise_ttest, build_comparison_table
from benchmarks.shared.logging_utils import load_run_summaries


def aggregate_vision(result_dir: str) -> dict:
    """Aggregate vision benchmark results by activation."""
    summaries = load_run_summaries(result_dir)
    if not summaries:
        print(f"  No summaries found in {result_dir}")
        return {}

    by_activation = defaultdict(list)
    for s in summaries:
        act = s.get("activation", "unknown")
        by_activation[act].append(s)

    print(f"\n{'='*60}")
    print("VISION BENCHMARK: ResNet-18 / CIFAR-10")
    print(f"{'='*60}")

    table = build_comparison_table(
        {name: results for name, results in by_activation.items()},
        metric_key="best_test_acc",
        baseline_name="ReLU",
    )
    print(table)
    print()

    # Detailed per-activation stats
    agg = {}
    for name, results in sorted(by_activation.items()):
        stats = aggregate_seeds(results, "best_test_acc")
        agg[name] = stats
        seeds = [r.get("seed", "?") for r in results]
        accs = [r["best_test_acc"] for r in results]
        print(f"  {name:20s}  {stats['mean']:.2f} ± {stats['std']:.2f}%  "
              f"(seeds: {seeds}, accs: {[f'{a:.2f}' for a in accs]})")

    return agg


def aggregate_nlp(result_dir: str) -> dict:
    """Aggregate NLP benchmark results by activation × task."""
    summaries = load_run_summaries(result_dir)
    if not summaries:
        print(f"  No summaries found in {result_dir}")
        return {}

    # Group by (activation, task)
    by_act_task = defaultdict(list)
    for s in summaries:
        act = s.get("activation", "unknown")
        task = s.get("task", "unknown")
        by_act_task[(act, task)].append(s)

    print(f"\n{'='*60}")
    print("NLP BENCHMARK: DistilBERT / GLUE")
    print(f"{'='*60}")

    # Print per-task tables
    tasks_seen = sorted(set(t for _, t in by_act_task.keys()))
    agg = {}

    for task in tasks_seen:
        print(f"\n--- {task.upper()} ---")
        task_results = {act: results for (act, t), results in by_act_task.items() if t == task}

        table = build_comparison_table(
            task_results,
            metric_key="best_val_acc",
            baseline_name="GELU",
        )
        print(table)

        for act, results in sorted(task_results.items()):
            stats = aggregate_seeds(results, "best_val_acc")
            agg[f"{act}_{task}"] = stats

    return agg


def export_latex_table(vision_agg: dict, nlp_agg: dict, output_path: str):
    """Export results as a LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Scaling benchmark results: mean $\pm$ std across 3 seeds.}",
        r"\label{tab:scaling_benchmarks}",
    ]

    if vision_agg:
        lines.append(r"\begin{subtable}{\linewidth}")
        lines.append(r"\centering")
        lines.append(r"\caption{ResNet-18 / CIFAR-10 (Top-1 Accuracy \%)}")
        lines.append(r"\begin{tabular}{lc}")
        lines.append(r"\toprule")
        lines.append(r"Activation & Test Accuracy \\")
        lines.append(r"\midrule")
        for name, stats in sorted(vision_agg.items()):
            lines.append(f"{name} & ${stats['mean']:.2f} \\pm {stats['std']:.2f}$ \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{subtable}")

    if nlp_agg:
        lines.append(r"\begin{subtable}{\linewidth}")
        lines.append(r"\centering")
        lines.append(r"\caption{DistilBERT / GLUE (Validation Accuracy \%)}")
        lines.append(r"\begin{tabular}{lc}")
        lines.append(r"\toprule")
        lines.append(r"Activation\_Task & Val Accuracy \\")
        lines.append(r"\midrule")
        for name, stats in sorted(nlp_agg.items()):
            lines.append(f"{name} & ${stats['mean']:.2f} \\pm {stats['std']:.2f}$ \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{subtable}")

    lines.append(r"\end{table}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate scaling benchmark results")
    parser.add_argument("--vision-dir", type=str, default="benchmarks/vision/results")
    parser.add_argument("--nlp-dir", type=str, default="benchmarks/nlp/results")
    parser.add_argument("--latex", action="store_true", help="Export LaTeX table")
    parser.add_argument("--latex-output", type=str,
                        default="benchmarks/analysis/scaling_results.tex")
    args = parser.parse_args()

    vision_agg = {}
    nlp_agg = {}

    if Path(args.vision_dir).exists():
        vision_agg = aggregate_vision(args.vision_dir)
    else:
        print(f"Vision results dir not found: {args.vision_dir}")

    if Path(args.nlp_dir).exists():
        nlp_agg = aggregate_nlp(args.nlp_dir)
    else:
        print(f"NLP results dir not found: {args.nlp_dir}")

    if args.latex and (vision_agg or nlp_agg):
        export_latex_table(vision_agg, nlp_agg, args.latex_output)

    if not vision_agg and not nlp_agg:
        print("\nNo results found. Run benchmarks first.")


if __name__ == "__main__":
    main()
