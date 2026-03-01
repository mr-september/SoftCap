"""
Plot training curves and comparison charts from benchmark results.

Usage:
    python benchmarks/analysis/plot_results.py --vision-dir benchmarks/vision/results
    python benchmarks/analysis/plot_results.py --nlp-dir benchmarks/nlp/results
    python benchmarks/analysis/plot_results.py  # Both
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_training_curves(result_dir: str) -> dict:
    """Load .jsonl training logs grouped by activation."""
    curves = defaultdict(list)
    result_path = Path(result_dir)
    for p in sorted(result_path.glob("*.jsonl")):
        # Parse run name: e.g. "ReLU_seed42.jsonl"
        run_name = p.stem
        parts = run_name.rsplit("_seed", 1)
        if len(parts) == 2:
            act_name = parts[0]
        else:
            act_name = run_name

        epochs = []
        with open(p) as f:
            for line in f:
                epochs.append(json.loads(line))
        curves[act_name].append({"run_name": run_name, "epochs": epochs})
    return curves


def plot_vision_curves(result_dir: str, output_dir: str):
    """Plot training curves for vision benchmark."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    curves = load_training_curves(result_dir)
    if not curves:
        print(f"No training logs found in {result_dir}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot 1: Test accuracy curves (mean ± std across seeds)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for act_name, runs in sorted(curves.items()):
        # Align epochs across seeds
        max_epochs = max(len(r["epochs"]) for r in runs)
        acc_matrix = np.full((len(runs), max_epochs), np.nan)
        for i, r in enumerate(runs):
            for j, ep in enumerate(r["epochs"]):
                acc_matrix[i, j] = ep.get("test_acc", ep.get("val_acc", np.nan))

        mean_acc = np.nanmean(acc_matrix, axis=0)
        std_acc = np.nanstd(acc_matrix, axis=0)
        epochs = np.arange(1, max_epochs + 1)

        ax.plot(epochs, mean_acc, label=act_name)
        if len(runs) > 1:
            ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("ResNet-18 / CIFAR-10: Test Accuracy by Activation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "vision_test_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / 'vision_test_accuracy.png'}")

    # Plot 2: Training loss curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for act_name, runs in sorted(curves.items()):
        max_epochs = max(len(r["epochs"]) for r in runs)
        loss_matrix = np.full((len(runs), max_epochs), np.nan)
        for i, r in enumerate(runs):
            for j, ep in enumerate(r["epochs"]):
                loss_matrix[i, j] = ep.get("train_loss", np.nan)

        mean_loss = np.nanmean(loss_matrix, axis=0)
        epochs = np.arange(1, max_epochs + 1)
        ax.plot(epochs, mean_loss, label=act_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("ResNet-18 / CIFAR-10: Training Loss by Activation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "vision_train_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / 'vision_train_loss.png'}")


def plot_nlp_bar_chart(result_dir: str, output_dir: str):
    """Plot bar chart of NLP benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    from benchmarks.shared.logging_utils import load_run_summaries

    summaries = load_run_summaries(result_dir)
    if not summaries:
        print(f"No summaries found in {result_dir}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by (activation, task)
    by_act_task = defaultdict(list)
    for s in summaries:
        act = s.get("activation", "unknown")
        task = s.get("task", "unknown")
        by_act_task[(act, task)].append(s.get("best_val_acc", 0.0))

    tasks = sorted(set(t for _, t in by_act_task.keys()))
    activations = sorted(set(a for a, _ in by_act_task.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(tasks))
    width = 0.8 / len(activations)

    for i, act in enumerate(activations):
        means = []
        stds = []
        for task in tasks:
            vals = by_act_task.get((act, task), [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds, label=act, capsize=3)

    ax.set_xlabel("GLUE Task")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("DistilBERT / GLUE: Validation Accuracy by Activation")
    ax.set_xticks(x + width * (len(activations) - 1) / 2)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "nlp_glue_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / 'nlp_glue_accuracy.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--vision-dir", type=str, default="benchmarks/vision/results")
    parser.add_argument("--nlp-dir", type=str, default="benchmarks/nlp/results")
    parser.add_argument("--output-dir", type=str, default="benchmarks/analysis/figures")
    args = parser.parse_args()

    if Path(args.vision_dir).exists():
        print("Plotting vision results...")
        plot_vision_curves(args.vision_dir, args.output_dir)
    if Path(args.nlp_dir).exists():
        print("Plotting NLP results...")
        plot_nlp_bar_chart(args.nlp_dir, args.output_dir)


if __name__ == "__main__":
    main()
