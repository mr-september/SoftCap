"""
DistilBERT / GLUE Scaling Benchmark

Fine-tunes DistilBERT on GLUE tasks with swappable activation functions.
All hyperparameters are FIXED across runs — only the activation changes.

Dependencies (install in WSL venv before running):
    pip install transformers datasets accelerate evaluate

Usage (from repo root, in WSL venv):
    # Full benchmark (all activations × all tasks × 3 seeds):
    python benchmarks/nlp/train_distilbert_glue.py

    # Single task / activation / seed:
    python benchmarks/nlp/train_distilbert_glue.py --task sst2 --activation GELU --seed 42

    # Quick sanity check (SST-2 only, 1 epoch, 1 seed):
    python benchmarks/nlp/train_distilbert_glue.py --quick

    # Mini-sweep (SST-2, a-star variants, 1 seed):
    python benchmarks/nlp/train_distilbert_glue.py --minisweep

Expected baselines (dev set, GELU):
    SST-2: ~91-92%
    MNLI:  ~82-83%  (matched/mismatched)
    QNLI:  ~88-89%
    MRPC:  ~87-88%

Hardware requirements:
    ~4 GB VRAM per run (DistilBERT + AMP)
    SST-2: ~30-40 min per seed | MNLI: ~3-4h per seed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.seeding import seed_everything
from benchmarks.shared.logging_utils import BenchmarkLogger
from benchmarks.shared.activation_registry import (
    get_nlp_activations,
    get_nlp_minisweep_activations,
    clone_activation,
)
from benchmarks.shared.efficiency import get_amp_context

# ---------------------------------------------------------------------------
# Fixed hyperparameters (DO NOT CHANGE across activation runs)
# ---------------------------------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 32
MAX_LENGTH = 128
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
SEEDS = [42, 123, 456]

# GLUE task configuration
TASK_CONFIG = {
    "sst2":  {"num_labels": 2, "metric": "accuracy"},
    "mnli":  {"num_labels": 3, "metric": "accuracy"},
    "qnli":  {"num_labels": 2, "metric": "accuracy"},
    "mrpc":  {"num_labels": 2, "metric": "accuracy"},
}
# Extended tasks (optional)
EXTENDED_TASKS = {
    "qqp":   {"num_labels": 2, "metric": "accuracy"},
    "cola":  {"num_labels": 2, "metric": "matthews_correlation"},
    "stsb":  {"num_labels": 1, "metric": "pearson"},
    "rte":   {"num_labels": 2, "metric": "accuracy"},
}

PRIORITY_TASKS = ["sst2", "mnli", "qnli", "mrpc"]


def _check_hf_deps():
    """Check that HuggingFace dependencies are installed."""
    missing = []
    for pkg in ["transformers", "datasets", "accelerate", "evaluate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing HuggingFace dependencies: {missing}")
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def load_glue_dataset(task_name: str, tokenizer, max_length: int):
    """Load and tokenize a GLUE task dataset."""
    from datasets import load_dataset

    # Map task names to HuggingFace GLUE subset names
    hf_name = {"sst2": "sst2", "mnli": "mnli", "qnli": "qnli", "mrpc": "mrpc",
                "qqp": "qqp", "cola": "cola", "stsb": "stsb", "rte": "rte"}[task_name]

    dataset = load_dataset("glue", hf_name)

    # Tokenization mapping for each task
    key_map = {
        "sst2": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "mrpc": ("sentence1", "sentence2"),
        "qqp":  ("question1", "question2"),
        "cola": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "rte":  ("sentence1", "sentence2"),
    }
    key1, key2 = key_map[task_name]

    def tokenize_fn(examples):
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(
            *args,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names
                            if "label" not in dataset["train"].column_names
                            else [c for c in dataset["train"].column_names if c != "label"])

    tokenized.set_format("torch")
    return tokenized


def create_dataloader(dataset_split, batch_size, shuffle=False):
    """Create a DataLoader from a HuggingFace dataset split."""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp):
    """Train for one epoch. Returns (avg_loss, accuracy, epoch_time)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_amp_context(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    elapsed = time.time() - t0
    return total_loss / total, 100.0 * correct / total, elapsed


@torch.no_grad()
def evaluate(model, loader, device, use_amp, task_name="sst2"):
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with get_amp_context(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs.loss.item() * input_ids.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def run_single(
    act_name: str,
    act_fn,
    task_name: str,
    seed: int,
    epochs: int,
    device: torch.device,
    log_dir: str,
    use_amp: bool = True,
):
    """Run a single fine-tuning experiment (one activation × one task × one seed)."""
    _check_hf_deps()
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        get_linear_schedule_with_warmup,
    )
    from benchmarks.nlp.distilbert_utils import swap_distilbert_activations, get_activation_param_count

    seed_everything(seed)

    run_name = f"{act_name}_{task_name}_seed{seed}"
    logger = BenchmarkLogger(log_dir, run_name)

    task_cfg = TASK_CONFIG.get(task_name, EXTENDED_TASKS.get(task_name))
    if task_cfg is None:
        raise ValueError(f"Unknown GLUE task: {task_name}")

    logger.set_metadata(
        activation=act_name,
        task=task_name,
        seed=seed,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        use_amp=use_amp,
    )

    # Tokenizer + Data
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_glue_dataset(task_name, tokenizer, MAX_LENGTH)

    train_loader = create_dataloader(dataset["train"], BATCH_SIZE, shuffle=True)

    # Validation split varies by task
    val_key = "validation" if "validation" in dataset else "validation_matched"
    val_loader = create_dataloader(dataset[val_key], BATCH_SIZE, shuffle=False)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=task_cfg["num_labels"]
    )

    # Swap activation if not GELU (GELU is the default)
    if act_name != "GELU":
        swap_distilbert_activations(model, clone_activation(act_fn))
        act_params = get_activation_param_count(model)
        print(f"  Swapped activation to {act_name} ({act_params} extra params)")

    # Disable monitoring for speed
    for m in model.modules():
        if hasattr(m, "monitoring"):
            m.monitoring = False

    model = model.to(device)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp, task_name)

        logger.log_epoch(epoch, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        print(
            f"  [{run_name}] Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%  "
            f"best={best_acc:.2f}%  ({epoch_time:.1f}s)"
        )

    logger.save_summary({
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_acc": val_acc,
    })
    logger.close()
    print(f"  ✓ {run_name} done — best val acc: {best_acc:.2f}%")
    return {"best_val_acc": best_acc, "best_epoch": best_epoch, "task": task_name}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DistilBERT / GLUE activation benchmark"
    )
    parser.add_argument("--task", type=str, default=None,
                        help="Single GLUE task (sst2, mnli, qnli, mrpc)")
    parser.add_argument("--activation", type=str, default=None,
                        help="Single activation name")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--quick", action="store_true",
                        help="Quick: SST-2 only, 1 epoch, 1 seed")
    parser.add_argument("--minisweep", action="store_true",
                        help="Mini-sweep: SST-2, a-star variants, 1 seed")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--log-dir", type=str,
                        default="benchmarks/nlp/results",
                        help="Output log directory")
    args = parser.parse_args()

    _check_hf_deps()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = not args.no_amp and torch.cuda.is_available()

    # Determine configuration
    if args.minisweep:
        activations = get_nlp_minisweep_activations()
        tasks = ["sst2"]
        seeds = [42]
        epochs = args.epochs or EPOCHS
        print(f"\n=== Mini-sweep: {len(activations)} activations × SST-2 × 1 seed ===\n")
    elif args.quick:
        activations = get_nlp_activations()
        tasks = ["sst2"]
        seeds = [42]
        epochs = args.epochs or 1
        print(f"\n=== Quick mode: {len(activations)} activations × SST-2 × 1 epoch ===\n")
    else:
        activations = get_nlp_activations()
        tasks = PRIORITY_TASKS
        seeds = SEEDS
        epochs = args.epochs or EPOCHS
        print(f"\n=== Full benchmark: {len(activations)} activations × {len(tasks)} tasks × {len(seeds)} seeds ===\n")

    if args.activation:
        if args.activation not in activations:
            print(f"Unknown activation: {args.activation}")
            print(f"Available: {list(activations.keys())}")
            sys.exit(1)
        activations = {args.activation: activations[args.activation]}

    if args.task:
        tasks = [args.task]
    if args.seed is not None:
        seeds = [args.seed]

    # Run all combinations
    all_results = {}
    for act_name, act_fn in activations.items():
        for task_name in tasks:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Activation: {act_name}  |  Task: {task_name}  |  Seed: {seed}")
                print(f"{'='*60}")
                result = run_single(
                    act_name=act_name,
                    act_fn=act_fn,
                    task_name=task_name,
                    seed=seed,
                    epochs=epochs,
                    device=device,
                    log_dir=args.log_dir,
                    use_amp=use_amp,
                )
                key = f"{act_name}_{task_name}"
                all_results.setdefault(key, []).append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, results in all_results.items():
        accs = [r["best_val_acc"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
        print(f"  {key:35s}  {mean_acc:.2f} ± {std_acc:.2f}%")

    print("\nResults saved to:", args.log_dir)


if __name__ == "__main__":
    main()
