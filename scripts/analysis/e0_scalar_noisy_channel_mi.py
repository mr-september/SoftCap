#!/usr/bin/env python3
"""
E0: Scalar noisy-channel mutual information probe for SoftCap-family activations.

This is an exploratory micro-benchmark intended to support the discussion in
`paper/sections/08_discussion.md` (§8.6). It estimates mutual information for a
simple scalar channel:

    X ~ N(0, 1)
    Y = f(X) + ε,  ε ~ N(0, σ^2)

using a discretization (histogram) estimator for I(X; Y).

Important caveats:
  - MI estimates depend on binning and range choices; interpret *comparatively*.
  - For continuous X and deterministic Y=f(X), I(X;Y) can be ill-defined/infinite.
    Adding noise makes I(X; Y) finite and estimator-friendly.

Example:
  python3 scripts/analysis/information_theory/e0_scalar_noisy_channel_mi.py \\
    --n 500000 --bins 256 --sigmas 0.01,0.1,0.3,1.0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def identity(x: np.ndarray) -> np.ndarray:
    return x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softcap(x: np.ndarray, a: float) -> np.ndarray:
    return np.where(x <= 0.0, 0.0, a * np.tanh(x))


def swishcap(x: np.ndarray, a: float) -> np.ndarray:
    u = a * x
    neg = 2.0 * a * x * _sigmoid(u)
    pos = a * np.tanh(x)
    return np.where(x <= 0.0, neg, pos)


def sparsecap(x: np.ndarray, a: float) -> np.ndarray:
    out = np.zeros_like(x)

    pos_mask = x > 0.0
    out[pos_mask] = a * np.tanh(x[pos_mask])

    notch_mask = (x > -a) & (x <= 0.0)
    if np.any(notch_mask):
        x_q = x[notch_mask]
        out[notch_mask] = x_q * (x_q + a) ** 3 * (a - 3.0 * x_q) / (a**3)

    return out


def _entropy_1d_hist(x: np.ndarray, *, bins: int, x_range: Tuple[float, float]) -> float:
    x = np.clip(x, x_range[0], x_range[1])
    counts, _ = np.histogram(x, bins=bins, range=x_range)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mi_2d_hist(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> float:
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])

    counts, _, _ = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    total = counts.sum()
    if total <= 0:
        return 0.0

    pxy = counts.astype(np.float64) / float(total)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py

    mask = pxy > 0
    # If pxy>0 then px>0 and py>0, so denom>0 on mask.
    mi_nats = float((pxy[mask] * (np.log(pxy[mask]) - np.log(denom[mask]))).sum())
    return mi_nats / math.log(2.0)


@dataclass(frozen=True)
class Variant:
    name: str
    activation: str
    a_regime: str
    a_value: Optional[float]
    fn: Callable[[np.ndarray], np.ndarray]


def _parse_sigmas(arg: str) -> List[float]:
    items = [s.strip() for s in arg.split(",") if s.strip()]
    return [float(s) for s in items]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500_000, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--bins", type=int, default=256, help="Bins per axis for discretization")
    parser.add_argument(
        "--sigmas",
        type=_parse_sigmas,
        default=[0.01, 0.1, 0.3, 1.0],
        help="Comma-separated noise std list, e.g. 0.01,0.1,0.3,1.0",
    )
    parser.add_argument("--x_clip", type=float, default=6.0, help="Clipping range for X histogram")
    parser.add_argument("--y_clip", type=float, default=6.0, help="Clipping range for Y histogram")
    parser.add_argument(
        "--include_controls",
        action="store_true",
        help="Include simple controls (Identity, ReLU, Tanh) at native scale.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary (for paper artifacts).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(size=args.n).astype(np.float64)

    # Paper-reported a* values (Section 1.3).
    a_star_softcap = 2.891
    a_star_swishcap = 2.434
    a_star_sparsecap = 2.140

    variants: List[Variant] = [
        Variant("SoftCap (a=1)", "SoftCap", "a=1", 1.0, lambda t: softcap(t, 1.0)),
        Variant(f"SoftCap (a=a*={a_star_softcap})", "SoftCap", "a=a*", a_star_softcap, lambda t: softcap(t, a_star_softcap)),
        Variant("SwishCap (a=1)", "SwishCap", "a=1", 1.0, lambda t: swishcap(t, 1.0)),
        Variant(f"SwishCap (a=a*={a_star_swishcap})", "SwishCap", "a=a*", a_star_swishcap, lambda t: swishcap(t, a_star_swishcap)),
        Variant("SparseCap (a=1)", "SparseCap", "a=1", 1.0, lambda t: sparsecap(t, 1.0)),
        Variant(f"SparseCap (a=a*={a_star_sparsecap})", "SparseCap", "a=a*", a_star_sparsecap, lambda t: sparsecap(t, a_star_sparsecap)),
    ]

    if args.include_controls:
        variants.extend(
            [
                Variant("Identity", "Identity", "control", None, identity),
                Variant("ReLU", "ReLU", "control", None, relu),
                Variant("Tanh", "Tanh", "control", None, tanh),
            ]
        )

    x_range = (-args.x_clip, args.x_clip)
    y_range = (-args.y_clip, args.y_clip)

    print("E0: Scalar noisy-channel MI probe")
    print(f"n={args.n:,}  bins={args.bins}  seed={args.seed}  x_range={x_range}  y_range={y_range}")
    print()
    print("| sigma | variant | I(X;Y) (bits) | H(Y) (bits) |")
    print("|---:|---|---:|---:|")

    rows = []
    for sigma in args.sigmas:
        eps = rng.normal(loc=0.0, scale=float(sigma), size=args.n).astype(np.float64)
        for v in variants:
            y = v.fn(x) + eps
            mi = _mi_2d_hist(x, y, bins=args.bins, x_range=x_range, y_range=y_range)
            hy = _entropy_1d_hist(y, bins=args.bins, x_range=y_range)
            print(f"| {sigma:.4g} | {v.name} | {mi:.4f} | {hy:.4f} |")
            rows.append(
                {
                    "sigma": float(sigma),
                    "variant": v.name,
                    "activation": v.activation,
                    "a_regime": v.a_regime,
                    "a_value": (float(v.a_value) if v.a_value is not None else None),
                    "mi_bits": float(mi),
                    "hy_bits": float(hy),
                }
            )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment": "e0_scalar_noisy_channel_mi",
            "description": (
                "Scalar noisy-channel mutual information probe for SoftCap-family activations (activation-only). "
                "X~N(0,1), Y=f(X)+eps with eps~N(0,sigma^2). MI and marginal entropy estimated via "
                "2D/1D histograms (discretized, clipped)."
            ),
            "date": _dt.date.today().isoformat(),
            "command": " ".join(sys.argv),
            "params": {
                "n": int(args.n),
                "bins": int(args.bins),
                "seed": int(args.seed),
                "sigmas": [float(s) for s in args.sigmas],
                "x_range": [float(x_range[0]), float(x_range[1])],
                "y_range": [float(y_range[0]), float(y_range[1])],
                "estimator": "histogram",
            },
            "a_star": {
                "SoftCap": float(a_star_softcap),
                "SwishCap": float(a_star_swishcap),
                "SparseCap": float(a_star_sparsecap),
            },
            "results": rows,
        }

        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
