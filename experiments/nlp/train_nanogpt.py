"""NanoGPT bounded-activation protocol training (small causal LM).

This script trains a miniature GPT architecture to test bounded activations in
decoder-local insertion points:
- attention projections (`q`, `k`, `v`, or combinations)
- FFN hidden activation
- legacy output activation hook

Integrates `MechanisticTelemetry` for rich scaling diagnostics.
"""

import math
import os
import sys
import time
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nlp.nanogpt import NanoGPT, NanoGPTConfig
from softcap.logging import seed_everything, BenchmarkLogger
from softcap.control_activations import get_named_activation_suite
from softcap.telemetry import MechanisticTelemetry

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False


def _maybe_sync_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_dataloaders(batch_size: int, block_size: int, device: str = "auto"):
    """Loads WikiText-2 and chunks it into block_size pieces."""
    if not HAS_HF:
        print("Please install huggingface dependencies: pip install datasets transformers")
        sys.exit(1)
        
    print("Loading WikiText-2 dataset...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
        
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
        
    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, num_proc=4
    )
    
    # Create simple PyTorch iterators
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = str(device)

    def get_batch(split):
        data = lm_datasets[split]
        dataset_len = len(data)
        val_pos = 0

        while True:
            # Random sampling for training, sequential for validation
            if split == 'train':
                ix = torch.randint(dataset_len, (batch_size,))
            else:
                if dataset_len == 0:
                    raise ValueError("Validation dataset is empty")
                start = val_pos
                end = min(start + batch_size, dataset_len)
                ix = torch.arange(start, end)
                val_pos = end
                if val_pos >= dataset_len:
                    val_pos = 0
                
            x = torch.stack([torch.tensor(data[i.item()]['input_ids'][:-1], dtype=torch.long) for i in ix])
            y = torch.stack([torch.tensor(data[i.item()]['input_ids'][1:], dtype=torch.long) for i in ix])
            
            if target_device == 'cuda':
                x = x.pin_memory().to('cuda', non_blocking=True)
                y = y.pin_memory().to('cuda', non_blocking=True)
            else:
                x = x.to(target_device)
                y = y.to(target_device)
            yield x, y

    return get_batch('train'), get_batch('validation'), len(lm_datasets['train']), len(lm_datasets['validation'])


def _compute_ece(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error for token predictions."""
    if confidence.numel() == 0:
        return float("nan")

    bins = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
    ece = torch.tensor(0.0, device=confidence.device)
    total = confidence.numel()
    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        conf_bin = confidence[mask].mean()
        acc_bin = correct[mask].float().mean()
        frac = mask.float().mean()
        ece += frac * torch.abs(acc_bin - conf_bin)

    return float(ece.item())


def evaluate_language_model(
    model: nn.Module,
    val_iter,
    *,
    val_steps: int,
    ctx,
) -> dict:
    """Compute validation metrics beyond perplexity for protocol decisions."""
    model.eval()
    val_nll_sum = 0.0
    val_token_total = 0
    token_correct = 0
    token_total = 0
    top5_correct = 0
    confidence_chunks = []
    correct_chunks = []

    with torch.no_grad():
        for _ in range(val_steps):
            X, Y = next(val_iter)
            with ctx:
                logits, loss = model(X, targets=Y)

            logits_f = logits.float()
            probs = torch.softmax(logits_f, dim=-1)
            pred = probs.argmax(dim=-1)

            flat_y = Y.reshape(-1)
            valid_mask = flat_y != -1
            if valid_mask.any():
                flat_pred = pred.reshape(-1)
                flat_probs = probs.reshape(-1, probs.size(-1))
                flat_valid_probs = flat_probs[valid_mask]
                flat_valid_y = flat_y[valid_mask]
                flat_valid_pred = flat_pred[valid_mask]

                batch_tokens = int(flat_valid_y.numel())
                val_nll_sum += float(loss.item()) * batch_tokens
                val_token_total += batch_tokens

                token_correct += int((flat_valid_pred == flat_valid_y).sum().item())
                token_total += int(flat_valid_y.numel())

                topk_idx = torch.topk(flat_valid_probs, k=min(5, flat_valid_probs.size(-1)), dim=-1).indices
                top5_correct += int((topk_idx == flat_valid_y.unsqueeze(-1)).any(dim=-1).sum().item())

                conf, _ = flat_valid_probs.max(dim=-1)
                corr = (flat_valid_pred == flat_valid_y)
                confidence_chunks.append(conf)
                correct_chunks.append(corr)

    avg_val_loss = (val_nll_sum / val_token_total) if val_token_total > 0 else float("nan")
    val_ppl = math.exp(min(20.0, avg_val_loss)) if avg_val_loss == avg_val_loss else float("nan")
    token_acc = (token_correct / token_total) if token_total > 0 else float("nan")
    top5_acc = (top5_correct / token_total) if token_total > 0 else float("nan")

    if confidence_chunks:
        conf_all = torch.cat(confidence_chunks)
        corr_all = torch.cat(correct_chunks)
        ece = _compute_ece(conf_all, corr_all)
    else:
        ece = float("nan")

    return {
        "val_loss": avg_val_loss,
        "val_ppl": val_ppl,
        "val_token_acc": token_acc,
        "val_top5_acc": top5_acc,
        "val_ece": ece,
    }


def main():
    parser = argparse.ArgumentParser("NanoGPT bounded-activation scaler")
    parser.add_argument("--activation", type=str, default="None", help="Name of Q/K activation")
    parser.add_argument("--qkv-targets", type=str, default="q,k", help="Comma separated targets for Q/K/V activation")
    parser.add_argument("--mlp-activation", type=str, default="None", help="Activation for the FFN hidden layer")
    parser.add_argument("--logit-softcap", type=str, default="None", help="Legacy output activation hook")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--log-freq", type=int, default=100, help="Telemetry sampling freq")
    parser.add_argument("--hessian", action="store_true", help="Enable heavy Hessian HVPs")
    parser.add_argument("--log-dir", type=str, default="results/nlp")
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="If >0, cap validation batches per epoch for faster sweeps.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional explicit run name.")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="results/nlp/checkpoints")
    parser.add_argument("--throughput-warmup-steps", type=int, default=10)
    parser.add_argument("--early-stop-on-nonfinite", action="store_true")
    parser.add_argument("--early-stop-loss-threshold", type=float, default=0.0)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Prepare activation functions
    # Using the standard experimental set for the registry
    registry = get_named_activation_suite("full")
    
    def parse_act(name: str):
        normalized = str(name).strip()
        if normalized.lower() == "none":
            return None
        if normalized not in registry:
            print(f"Unknown activation: {normalized}. Available: {list(registry.keys())}")
            sys.exit(1)
        return torch.hub.copy.deepcopy(registry[normalized])
        
    qk_act_fn = parse_act(args.activation)
    mlp_act_fn = parse_act(args.mlp_activation)
    logit_cap_fn = parse_act(args.logit_softcap)
    qkv_targets = tuple(t.strip().lower() for t in args.qkv_targets.split(',') if t.strip())
        
    config = NanoGPTConfig(
        block_size=args.block_size,
        vocab_size=50304,
        n_layer=4, # Mini causal LM parameters for public release repo
        n_head=4,
        n_embd=256,
        qk_activation=qk_act_fn,
        qkv_activation_targets=qkv_targets,
        mlp_activation=mlp_act_fn,
        final_logit_softcap=logit_cap_fn
    )
    
    # 2. Setup Model & Optimizations
    model = NanoGPT(config)
    model.to(device)
    
    if args.compile and device == 'cuda':
        print("Compiling model...")
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-1)
    
    # 3. Setup Telemetry
    telemetry = MechanisticTelemetry(
        model, 
        logging_freq=args.log_freq, 
        enable_hessian=args.hessian
    )
    
    run_name = args.run_name or f"nanogpt_{args.activation}_seed{args.seed}"
    logger = BenchmarkLogger(args.log_dir, run_name)
    logger.set_metadata(
        activation=args.activation,
        qkv_targets=",".join(qkv_targets),
        mlp_activation=args.mlp_activation,
        seed=args.seed,
        model_size=sum(p.numel() for p in model.parameters()),
        batch_size=args.batch_size,
        lr=args.lr,
        block_size=args.block_size,
        compiled=args.compile,
    )
    
    # 4. Data
    train_iter, val_iter, n_train, n_val = get_dataloaders(
        args.batch_size,
        config.block_size,
        device=device,
    )
    iters_per_epoch = max(1, n_train // args.batch_size)
    grad_accum_steps = 4
    
    print(f"Starting {run_name} for {args.epochs} epochs ({iters_per_epoch} iters/epoch).")
    
    # 5. Training Loop
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
    if device == 'cuda':
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        ctx = nullcontext()

    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    total_train_tokens = 0
    timed_train_tokens = 0
    timed_train_time_s = 0.0
    total_eval_time_s = 0.0
    measured_train_steps = 0
    global_step = 0
    t0 = time.time()

    diverged = False
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for step in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch}", leave=False):
            telemetry.prepare_step()
            optimizer.zero_grad(set_to_none=True)
            _maybe_sync_cuda(device)
            train_step_t0 = time.perf_counter()
            
            # Gradient Accumulation
            batch_loss = 0.0
            for micro_step in range(grad_accum_steps):
                X, Y = next(train_iter)
                with ctx:
                    logits, raw_loss = model(X, targets=Y)
                    loss = raw_loss / grad_accum_steps

                if args.early_stop_on_nonfinite:
                    if not torch.isfinite(raw_loss):
                        diverged = True
                        break
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach().item())

            if diverged:
                break
                
            scaler.step(optimizer)
            scaler.update()
            _maybe_sync_cuda(device)
            train_step_time_s = time.perf_counter() - train_step_t0

            total_train_tokens += args.batch_size * args.block_size * grad_accum_steps
            global_step += 1
            if global_step > max(0, int(args.throughput_warmup_steps)):
                timed_train_tokens += args.batch_size * args.block_size * grad_accum_steps
                timed_train_time_s += train_step_time_s
                measured_train_steps += 1
            
            # Finalize Telemetry
            telem_metrics = telemetry.finalize_step(loss=loss)
            if telem_metrics:
                telem_metrics['step'] = step + (epoch - 1) * iters_per_epoch
                logger.log_epoch(epoch, telem_metrics)
            
            epoch_loss += batch_loss

        if diverged:
            break
            
        avg_train_loss = epoch_loss / iters_per_epoch
        
        # Validation
        eval_t0 = time.perf_counter()
        val_steps = max(1, math.ceil(n_val / args.batch_size))
        if args.eval_max_batches > 0:
            val_steps = min(val_steps, args.eval_max_batches)
        eval_metrics = evaluate_language_model(
            model,
            val_iter,
            val_steps=val_steps,
            ctx=ctx,
        )
        _maybe_sync_cuda(device)
        total_eval_time_s += time.perf_counter() - eval_t0
        val_loss = eval_metrics["val_loss"]
        val_ppl = eval_metrics["val_ppl"]
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        best_val_ppl = min(best_val_ppl, val_ppl)
            
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.3f}")
        logger.log_epoch(epoch, eval_metrics)

    total_time = time.time() - t0
    tokens_per_second = timed_train_tokens / max(1e-9, timed_train_time_s)

    if args.save_checkpoint:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "run_name": run_name,
        }, checkpoint_path)

    logger.save_summary({
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "total_time_s": total_time,
        "tokens_per_second": tokens_per_second,
    })
    logger.close()
    
if __name__ == "__main__":
    main()
