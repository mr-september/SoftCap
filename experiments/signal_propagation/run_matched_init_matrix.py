"""Matched-initialization matrix for Section 2 claim tightening.

Computes:
1) variance propagation metric (nu_30 / nu_0)
2) downstream validation accuracy

Common run family:
- Architecture: DeepMLP (30 hidden layers, no BN)
- Task: Fashion-MNIST classification
- Activations: SoftCap / SwishCap / SparseCap / ReLU / GELU / SiLU
- Inits: Xavier / Kaiming / Orthogonal
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

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Locate repo root (works at any depth in the tree)
_repo_p = Path(__file__).resolve().parent
while _repo_p != _repo_p.parent and not (_repo_p / 'softcap').is_dir():
    _repo_p = _repo_p.parent
project_root = _repo_p
sys.path.insert(0, str(project_root))

from softcap.activations import (
    SoftCap,
    SwishCap,
    SparseCap,
)
from softcap.models import DeepMLP


A_STAR = {
    "SoftCap": 2.890625,
    "SwishCap": 2.43359375,
    "SparseCap": 2.14,
}


@dataclass
class RunResult:
    activation: str
    init: str
    seed: int
    nu_0: float
    nu_30: float
    nu_ratio_30_0: float
    val_acc: float


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_activation(name: str) -> nn.Module:
    if name == "SoftCap":
        act = SoftCap(a_init=A_STAR["SoftCap"])
        act.a.requires_grad = False
        return act
    if name == "SwishCap":
        act = SwishCap(a_init=A_STAR["SwishCap"])
        act.a.requires_grad = False
        return act
    if name == "SparseCap":
        act = SparseCap(a_init=A_STAR["SparseCap"])
        act.a.requires_grad = False
        return act
    if name == "ReLU":
        return nn.ReLU()
    if name == "GELU":
        return nn.GELU()
    if name == "SiLU":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def apply_init(model: nn.Module, init_name: str, activation_name: str) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_name == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_name == "kaiming":
                nonlin = "relu" if activation_name == "ReLU" else "linear"
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
            elif init_name == "orthogonal":
                nn.init.orthogonal_(m.weight)
            else:
                raise ValueError(f"Unknown init: {init_name}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@torch.no_grad()
def compute_variance_ratio(model: DeepMLP, device: torch.device, batch_size: int = 2048) -> Dict[str, float]:
    x = torch.randn(batch_size, 784, device=device)
    nu0 = x.var(unbiased=False).item()

    for layer in model.layers[:-1]:
        x = layer(x)
        if hasattr(model.activation_fn, "activation_function"):
            x = model.activation_fn.activation_function(x)
        else:
            x = model.activation_fn(x)
    nu30 = x.var(unbiased=False).item()
    return {"nu_0": float(nu0), "nu_30": float(nu30), "nu_ratio_30_0": float(nu30 / max(nu0, 1e-12))}


def build_loaders(data_root: str, train_limit: int, val_limit: int, batch_size: int) -> Dict[str, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
    val = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
    train = Subset(train, range(train_limit))
    val = Subset(val, range(val_limit))
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def train_eval(model: DeepMLP, loaders: Dict[str, DataLoader], device: torch.device, epochs: int) -> float:
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in loaders["train"]:
            xb = xb.to(device).view(xb.size(0), -1)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loaders["val"]:
            xb = xb.to(device).view(xb.size(0), -1)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += pred.eq(yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / max(total, 1)


def summarize(results: List[RunResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in results:
        table.setdefault(r.activation, {})
        cell = table[r.activation].setdefault(r.init, {"nu_ratio_30_0": [], "val_acc": []})
        cell["nu_ratio_30_0"].append(r.nu_ratio_30_0)
        cell["val_acc"].append(r.val_acc)
    for act in table:
        for ini in table[act]:
            nu = np.array(table[act][ini]["nu_ratio_30_0"], dtype=float)
            va = np.array(table[act][ini]["val_acc"], dtype=float)
            table[act][ini] = {
                "nu_ratio_30_0_mean": float(nu.mean()),
                "nu_ratio_30_0_std": float(nu.std(ddof=0)),
                "val_acc_mean": float(va.mean()),
                "val_acc_std": float(va.std(ddof=0)),
                "n": int(len(nu)),
            }
    return table


def write_outputs(out_dir: Path, results: List[RunResult], summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / "matched_init_matrix_results.json"
    summary_json = out_dir / "matched_init_matrix_summary.json"
    csv_path = out_dir / "matched_init_matrix.csv"
    md_path = out_dir / "MATCHED_INIT_MATRIX.md"

    with raw_json.open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["activation,init,seed,nu_0,nu_30,nu_ratio_30_0,val_acc"]
    for r in results:
        lines.append(
            f"{r.activation},{r.init},{r.seed},{r.nu_0:.8f},{r.nu_30:.8f},{r.nu_ratio_30_0:.8f},{r.val_acc:.4f}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Matched Initialization Matrix",
        "",
        "| Activation | Init | nu_30/nu_0 (mean ± std) | Val acc % (mean ± std) | n |",
        "|---|---|---:|---:|---:|",
    ]
    for act in sorted(summary.keys()):
        for ini in ["xavier", "kaiming", "orthogonal"]:
            s = summary[act][ini]
            md_lines.append(
                f"| {act} | {ini} | {s['nu_ratio_30_0_mean']:.3f} ± {s['nu_ratio_30_0_std']:.3f} | "
                f"{s['val_acc_mean']:.2f} ± {s['val_acc_std']:.2f} | {s['n']} |"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="runs/signal_propagation/matched_init_matrix")
    ap.add_argument("--data-root", default="data/fashion_mnist")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = ["SoftCap", "SwishCap", "SparseCap", "ReLU", "GELU", "SiLU"]
    inits = ["xavier", "kaiming", "orthogonal"]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    loaders = build_loaders(args.data_root, args.train_limit, args.val_limit, args.batch_size)
    results: List[RunResult] = []

    for seed in seeds:
        seed_all(seed)
        for act_name in activations:
            for init_name in inits:
                act = make_activation(act_name)
                model = DeepMLP(
                    activation_fn=copy.deepcopy(act),
                    input_dim=784,
                    hidden_dim=128,
                    output_dim=10,
                    num_layers=30,
                    use_batch_norm=False,
                ).to(device)
                apply_init(model, init_name, act_name)
                vr = compute_variance_ratio(model, device)
                val_acc = train_eval(model, loaders, device, epochs=args.epochs)
                results.append(
                    RunResult(
                        activation=act_name,
                        init=init_name,
                        seed=seed,
                        nu_0=vr["nu_0"],
                        nu_30=vr["nu_30"],
                        nu_ratio_30_0=vr["nu_ratio_30_0"],
                        val_acc=val_acc,
                    )
                )
                print(
                    f"{act_name:9s} | {init_name:10s} | seed={seed} | "
                    f"nu30/nu0={vr['nu_ratio_30_0']:.3f} | val_acc={val_acc:.2f}"
                )

    summary = summarize(results)
    write_outputs(Path(args.output_dir), results, summary)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
