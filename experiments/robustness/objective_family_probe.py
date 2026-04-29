#!/usr/bin/env python3
"""Compact objective-family robustness probe.

This benchmark keeps the data, backbone, optimizer, and activation set fixed
while varying the training objective. The goal is not to rank activations
globally, but to test whether the repo's current mechanism claims survive small
objective changes in a matched setting.

Representative objective families:
- `bce`: likelihood-style binary classification via `BCEWithLogitsLoss`
- `cross_entropy`: 2-class softmax classification via `CrossEntropyLoss`
- `hinge`: margin loss on signed logits
- `mse`: regression-style supervision on sigmoid probabilities
- `infonce`: contrastive pretraining with a linear probe readout
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from softcap.control_activations import A_STAR_VALUES, get_standard_experimental_set
from softcap.logging import seed_everything


DEFAULT_DATASETS_DIR = PROJECT_ROOT / "data" / "toy_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "robustness" / "objective_family_probe"
DEFAULT_DATASETS = ("xor", "moons", "spiral")
DEFAULT_OBJECTIVES = ("bce", "cross_entropy", "hinge", "mse", "infonce")
DEFAULT_ACTIVATIONS = (
    "SoftCap",
    "SwishCap",
    "SparseCap",
    "ReLU",
    "Tanh",
    "GELU",
    "SiLU",
    "ReLU6",
    "HardTanh",
)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    family: str
    output_dim: int
    metric_name: str


OBJECTIVE_SPECS: Dict[str, ObjectiveSpec] = {
    "bce": ObjectiveSpec("bce", "likelihood", 1, "val_accuracy"),
    "cross_entropy": ObjectiveSpec("cross_entropy", "likelihood", 2, "val_accuracy"),
    "hinge": ObjectiveSpec("hinge", "margin", 1, "val_accuracy"),
    "mse": ObjectiveSpec("mse", "regression", 1, "val_accuracy"),
    "infonce": ObjectiveSpec("infonce", "contrastive", 0, "probe_val_accuracy"),
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_csv_list(raw: str | Sequence[str] | None, *, default: Sequence[str]) -> List[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",")]
    else:
        items = [str(item).strip() for item in raw]
    return [item for item in items if item] or list(default)


def _proportional_counts(total_target: int, class_counts: Sequence[int]) -> List[int]:
    if total_target <= 0:
        raise ValueError("total_target must be positive")
    if total_target > sum(class_counts):
        raise ValueError("total_target cannot exceed the population size")

    n_classes = len(class_counts)
    base = [0] * n_classes
    remaining = total_target

    if total_target >= n_classes:
        for i, count in enumerate(class_counts):
            if count > 0:
                base[i] = 1
                remaining -= 1

    if remaining < 0:
        base = [0] * n_classes
        remaining = total_target

    residual_capacity = [max(0, count - base_i) for count, base_i in zip(class_counts, base)]
    residual_total = sum(residual_capacity)
    if remaining == 0 or residual_total == 0:
        return base

    raw = [remaining * cap / residual_total for cap in residual_capacity]
    floor = [int(math.floor(x)) for x in raw]
    counts = [base_i + floor_i for base_i, floor_i in zip(base, floor)]
    assigned = sum(floor)
    leftovers = remaining - assigned
    order = sorted(
        range(n_classes),
        key=lambda i: (raw[i] - floor[i], residual_capacity[i]),
        reverse=True,
    )
    for idx in order:
        if leftovers <= 0:
            break
        if counts[idx] < class_counts[idx]:
            counts[idx] += 1
            leftovers -= 1
    return counts


def stratified_subsample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples >= len(y):
        return np.arange(len(y), dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_indices = [np.where(y == cls)[0] for cls in classes]
    class_counts = [len(idx) for idx in class_indices]
    target_counts = _proportional_counts(max_samples, class_counts)

    chosen: List[np.ndarray] = []
    for idx, target in zip(class_indices, target_counts):
        shuffled = np.array(idx, copy=True)
        rng.shuffle(shuffled)
        chosen.append(shuffled[:target])

    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out.astype(np.int64, copy=False)


def stratified_split_indices(y: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        shuffled = np.array(idx, copy=True)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_fraction))
        n_val = max(1, min(len(shuffled) - 1, n_val))
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def load_toy_dataset(
    name: str,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    max_samples: int | None = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path = datasets_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing toy dataset CSV: {path}")

    rows: List[Tuple[float, float, int]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["x1"]), float(row["x2"]), int(float(row["y"]))))

    arr = np.array(rows, dtype=np.float32)
    x = arr[:, :2]
    y = arr[:, 2].astype(np.int64)

    if max_samples is not None:
        indices = stratified_subsample_indices(y, max_samples=max_samples, seed=seed)
        x = x[indices]
        y = y[indices]

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def build_activation_suite(
    names: Sequence[str],
) -> Dict[str, nn.Module]:
    available = get_standard_experimental_set(include_bounded=True)
    suite: Dict[str, nn.Module] = {}
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown activation {name!r}. Available: {sorted(available)}")
        activation = copy.deepcopy(available[name])
        # Force a* fixed for probe
        if name in A_STAR_VALUES and hasattr(activation, "a"):
            with torch.no_grad():
                activation.a.fill_(float(A_STAR_VALUES[name]))
            activation.a.requires_grad_(False)
        if hasattr(activation, "monitoring"):
            activation.monitoring = False
        suite[name] = activation
    return suite


class ActivationTelemetry:
    """Average strict-zero and derivative-dead fractions across activation calls."""

    def __init__(self, model: nn.Module, *, near_zero_eps: float = 1e-3) -> None:
        self.near_zero_eps = float(near_zero_eps)
        self.calls = 0
        self.strict_zero_sum = 0.0
        self.near_zero_sum = 0.0
        self.deriv_zero_sum = 0.0
        self.deriv_near_sum = 0.0
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        for module in model.modules():
            if hasattr(module, "derivative"):
                self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not inputs:
            return
        pre_activation = inputs[0].detach()
        post_activation = output.detach()
        derivative = module.derivative(pre_activation)

        eps = self.near_zero_eps
        self.calls += 1
        self.strict_zero_sum += float((post_activation == 0).float().mean().cpu().item())
        self.near_zero_sum += float((post_activation.abs() < eps).float().mean().cpu().item())
        self.deriv_zero_sum += float((derivative == 0).float().mean().cpu().item())
        self.deriv_near_sum += float((derivative.abs() < eps).float().mean().cpu().item())

    def summary(self) -> Dict[str, float]:
        if self.calls == 0:
            return {
                "activation_calls": 0.0,
                "strict_zero_fraction": 0.0,
                "near_zero_fraction": 0.0,
                "derivative_zero_fraction": 0.0,
                "derivative_near_zero_fraction": 0.0,
            }
        denom = float(self.calls)
        return {
            "activation_calls": denom,
            "strict_zero_fraction": self.strict_zero_sum / denom,
            "near_zero_fraction": self.near_zero_sum / denom,
            "derivative_zero_fraction": self.deriv_zero_sum / denom,
            "derivative_near_zero_fraction": self.deriv_near_sum / denom,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ToyBackbone(nn.Module):
    """Small MLP backbone shared across objective families."""

    def __init__(
        self,
        activation_fn: nn.Module,
        *,
        input_dim: int = 2,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_activation = copy.deepcopy(activation_fn)
        self.hidden = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_activations.append(copy.deepcopy(activation_fn))
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_activation = copy.deepcopy(activation_fn)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_activation(self.input(x))
        for layer, activation in zip(self.hidden, self.hidden_activations):
            x = activation(layer(x))
        x = self.embedding_activation(self.embedding(x))
        return x


class ToySupervisedModel(nn.Module):
    def __init__(self, activation_fn: nn.Module, *, output_dim: int, hidden_dim: int, embedding_dim: int, depth: int) -> None:
        super().__init__()
        self.backbone = ToyBackbone(
            activation_fn,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            depth=depth,
        )
        self.head = nn.Linear(embedding_dim, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        logits = self.head(embedding)
        return logits, embedding


class ToyContrastiveModel(nn.Module):
    def __init__(self, activation_fn: nn.Module, *, hidden_dim: int, embedding_dim: int, projection_dim: int, depth: int) -> None:
        super().__init__()
        self.backbone = ToyBackbone(
            activation_fn,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            depth=depth,
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            copy.deepcopy(activation_fn),
            nn.Linear(embedding_dim, projection_dim),
        )
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        projection = self.projector(embedding)
        return embedding, projection


def _global_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().cpu().item())
        total += value * value
    return math.sqrt(total)


def _evaluate_predictions(logits: torch.Tensor, y: torch.Tensor, objective: str) -> Tuple[torch.Tensor, float]:
    if objective == "cross_entropy":
        pred = logits.argmax(dim=1)
        acc = float((pred == y).float().mean().cpu().item())
        return pred, acc

    if logits.ndim == 2:
        logits = logits.squeeze(-1)
    if objective in {"bce", "mse"}:
        pred = (torch.sigmoid(logits) >= 0.5).long()
    else:
        pred = (logits >= 0.0).long()
    acc = float((pred == y).float().mean().cpu().item())
    return pred, acc


def _compute_supervised_loss(logits: torch.Tensor, y: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "bce":
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
    if objective == "cross_entropy":
        return F.cross_entropy(logits, y)
    if objective == "hinge":
        signed_y = y.float() * 2.0 - 1.0
        return torch.relu(1.0 - signed_y * logits.squeeze(-1)).mean()
    if objective == "mse":
        probs = torch.sigmoid(logits.squeeze(-1))
        return F.mse_loss(probs, y.float())
    raise ValueError(f"Unsupported supervised objective: {objective}")


def _nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, *, temperature: float) -> torch.Tensor:
    batch_size = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    reps = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(reps, reps.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, -1e9)
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    denominator = torch.logsumexp(sim, dim=1)
    return -(positives - denominator).mean()


def _augment_points(x: torch.Tensor, *, noise_scale: float, scale_jitter: float) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_scale
    scales = 1.0 + torch.randn(x.size(0), 1, device=x.device) * scale_jitter
    return x * scales + noise


def train_supervised_objective(
    *,
    objective: str,
    activation: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    embedding_dim: int,
    depth: int,
    device: torch.device,
) -> Dict[str, object]:
    spec = OBJECTIVE_SPECS[objective]
    model = ToySupervisedModel(
        activation,
        output_dim=spec.output_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        depth=depth,
    ).to(device)
    telemetry = ActivationTelemetry(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    best_val_acc = 0.0
    final_val_acc = 0.0
    diverged = False

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        grad_norm_sum = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x)
            loss = _compute_supervised_loss(logits, batch_y, objective)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            grad_norm = _global_grad_norm(model.parameters())
            optimizer.step()

            _, batch_acc = _evaluate_predictions(logits.detach(), batch_y, objective)
            train_loss_sum += float(loss.detach().cpu().item())
            train_acc_sum += batch_acc
            grad_norm_sum += grad_norm
            batches += 1

        if diverged:
            break

        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits, _ = model(batch_x)
                loss = _compute_supervised_loss(logits, batch_y, objective)
                _, batch_acc = _evaluate_predictions(logits, batch_y, objective)
                val_loss_sum += float(loss.cpu().item())
                val_acc_sum += batch_acc
                val_batches += 1

        final_val_acc = val_acc_sum / max(1, val_batches)
        best_val_acc = max(best_val_acc, final_val_acc)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / max(1, batches),
                "train_accuracy": train_acc_sum / max(1, batches),
                "val_loss": val_loss_sum / max(1, val_batches),
                "val_accuracy": final_val_acc,
                "grad_norm": grad_norm_sum / max(1, batches),
            }
        )

    result = {
        "history": history,
        "diverged": diverged,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": final_val_acc,
        "telemetry": telemetry.summary(),
    }
    telemetry.close()
    return result


def train_contrastive_objective(
    *,
    activation: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    probe_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    embedding_dim: int,
    projection_dim: int,
    depth: int,
    temperature: float,
    jitter_std: float,
    scale_jitter: float,
    device: torch.device,
) -> Dict[str, object]:
    model = ToyContrastiveModel(
        activation,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        depth=depth,
    ).to(device)
    telemetry = ActivationTelemetry(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    pretrain_history: List[Dict[str, float]] = []
    diverged = False

    data_scale = float(train_x.std(dim=0).mean().item())
    effective_noise = jitter_std * max(data_scale, 1e-6)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        grad_norm_sum = 0.0
        batches = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            view_i = _augment_points(batch_x, noise_scale=effective_noise, scale_jitter=scale_jitter)
            view_j = _augment_points(batch_x, noise_scale=effective_noise, scale_jitter=scale_jitter)
            optimizer.zero_grad(set_to_none=True)
            _, z_i = model(view_i)
            _, z_j = model(view_j)
            loss = _nt_xent_loss(z_i, z_j, temperature=temperature)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            grad_norm_sum += _global_grad_norm(model.parameters())
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            batches += 1
        if diverged:
            break
        pretrain_history.append(
            {
                "epoch": float(epoch),
                "contrastive_loss": loss_sum / max(1, batches),
                "grad_norm": grad_norm_sum / max(1, batches),
            }
        )

    probe_history: List[Dict[str, float]] = []
    best_probe_val_acc = 0.0
    final_probe_val_acc = 0.0

    if not diverged:
        model.eval()
        with torch.no_grad():
            train_embeddings = model.backbone(train_x.to(device))
            val_embeddings = model.backbone(val_x.to(device))

        probe = nn.Linear(embedding_dim, 2).to(device)
        nn.init.xavier_uniform_(probe.head.weight if hasattr(probe, 'head') else probe.weight)
        nn.init.zeros_(probe.head.bias if hasattr(probe, 'head') else probe.bias)
        probe_optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        probe_train_loader = DataLoader(
            TensorDataset(train_embeddings.detach(), train_y.to(device)),
            batch_size=batch_size,
            shuffle=True,
        )
        probe_val_loader = DataLoader(
            TensorDataset(val_embeddings.detach(), val_y.to(device)),
            batch_size=batch_size,
            shuffle=False,
        )

        for epoch in range(probe_epochs):
            probe.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            train_batches = 0
            for batch_x, batch_y in probe_train_loader:
                probe_optimizer.zero_grad(set_to_none=True)
                logits = probe(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                probe_optimizer.step()
                train_loss_sum += float(loss.detach().cpu().item())
                train_acc_sum += float((logits.argmax(dim=1) == batch_y).float().mean().cpu().item())
                train_batches += 1

            probe.eval()
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_x, batch_y in probe_val_loader:
                    logits = probe(batch_x)
                    loss = F.cross_entropy(logits, batch_y)
                    val_loss_sum += float(loss.cpu().item())
                    val_acc_sum += float((logits.argmax(dim=1) == batch_y).float().mean().cpu().item())
                    val_batches += 1

            final_probe_val_acc = val_acc_sum / max(1, val_batches)
            best_probe_val_acc = max(best_probe_val_acc, final_probe_val_acc)
            probe_history.append(
                {
                    "epoch": float(epoch),
                    "probe_train_loss": train_loss_sum / max(1, train_batches),
                    "probe_train_accuracy": train_acc_sum / max(1, train_batches),
                    "probe_val_loss": val_loss_sum / max(1, val_batches),
                    "probe_val_accuracy": final_probe_val_acc,
                }
            )

    result = {
        "history": pretrain_history,
        "probe_history": probe_history,
        "diverged": diverged,
        "best_probe_val_accuracy": best_probe_val_acc,
        "final_probe_val_accuracy": final_probe_val_acc,
        "telemetry": telemetry.summary(),
    }
    telemetry.close()
    return result


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def aggregate_runs(seed_runs: Sequence[Dict[str, object]], *, objective: str) -> Dict[str, float]:
    if objective == "infonce":
        best_key = "best_probe_val_accuracy"
        final_key = "final_probe_val_accuracy"
    else:
        best_key = "best_val_accuracy"
        final_key = "final_val_accuracy"

    best_values = [float(run.get(best_key, 0.0)) for run in seed_runs]
    final_values = [float(run.get(final_key, 0.0)) for run in seed_runs]
    diverged = [1.0 if bool(run.get("diverged", False)) else 0.0 for run in seed_runs]

    strict_zero = [float(run["telemetry"]["strict_zero_fraction"]) for run in seed_runs]
    near_zero = [float(run["telemetry"]["near_zero_fraction"]) for run in seed_runs]
    deriv_zero = [float(run["telemetry"]["derivative_zero_fraction"]) for run in seed_runs]
    deriv_near = [float(run["telemetry"]["derivative_near_zero_fraction"]) for run in seed_runs]

    best_mean, best_std = _mean_std(best_values)
    final_mean, final_std = _mean_std(final_values)
    strict_mean, strict_std = _mean_std(strict_zero)
    near_mean, near_std = _mean_std(near_zero)
    deriv_mean, deriv_std = _mean_std(deriv_zero)
    deriv_near_mean, deriv_near_std = _mean_std(deriv_near)

    return {
        "n_seeds": float(len(seed_runs)),
        "diverged_runs": float(sum(diverged)),
        "best_metric_mean": best_mean,
        "best_metric_std": best_std,
        "final_metric_mean": final_mean,
        "final_metric_std": final_std,
        "strict_zero_fraction_mean": strict_mean,
        "strict_zero_fraction_std": strict_std,
        "near_zero_fraction_mean": near_mean,
        "near_zero_fraction_std": near_std,
        "derivative_zero_fraction_mean": deriv_mean,
        "derivative_zero_fraction_std": deriv_std,
        "derivative_near_zero_fraction_mean": deriv_near_mean,
        "derivative_near_zero_fraction_std": deriv_near_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Objective-family robustness probe.")
    parser.add_argument("--datasets", default=None, help="Comma separated datasets.")
    parser.add_argument("--objectives", default=None, help="Comma separated objectives.")
    parser.add_argument("--activations", default=None, help="Comma separated activations.")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    datasets = _parse_csv_list(args.datasets, default=DEFAULT_DATASETS)
    objectives = _parse_csv_list(args.objectives, default=DEFAULT_OBJECTIVES)
    activations = _parse_csv_list(args.activations, default=DEFAULT_ACTIVATIONS)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = build_activation_suite(activations)
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---")
        summary[dataset_name] = {}
        
        try:
            x, y = load_toy_dataset(dataset_name, max_samples=args.max_samples)
        except Exception as e:
            print(f"Skipping {dataset_name}: {e}")
            continue

        for objective in objectives:
            print(f"  Objective: {objective}")
            summary[dataset_name][objective] = {}
            
            for act_name, activation in suite.items():
                print(f"    Activation: {act_name}")
                seed_runs: List[Dict[str, object]] = []
                
                for seed in range(args.n_seeds):
                    seed_everything(seed)
                    train_idx, val_idx = stratified_split_indices(y.numpy(), val_fraction=0.2, seed=seed)
                    
                    if objective == "infonce":
                        run = train_contrastive_objective(
                            activation=activation,
                            train_x=x[train_idx],
                            train_y=y[train_idx],
                            val_x=x[val_idx],
                            val_y=y[val_idx],
                            epochs=args.epochs,
                            probe_epochs=args.probe_epochs,
                            batch_size=args.batch_size,
                            lr=1e-3,
                            weight_decay=0.0,
                            hidden_dim=64,
                            embedding_dim=64,
                            projection_dim=32,
                            depth=2,
                            temperature=0.1,
                            jitter_std=0.08,
                            scale_jitter=0.05,
                            device=device,
                        )
                    else:
                        run = train_supervised_objective(
                            objective=objective,
                            activation=activation,
                            train_x=x[train_idx],
                            train_y=y[train_idx],
                            val_x=x[val_idx],
                            val_y=y[val_idx],
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=1e-3,
                            weight_decay=0.0,
                            hidden_dim=64,
                            embedding_dim=64,
                            depth=2,
                            device=device,
                        )
                    seed_runs.append(run)
                
                summary[dataset_name][objective][act_name] = aggregate_runs(seed_runs, objective=objective)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nProbe complete. Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
