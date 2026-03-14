#!/usr/bin/env python3
"""Run the paper confound audit in a single, resumable order.

The suite intentionally stays narrow:
1. Muon longer-horizon explicit-clamp follow-up
2. Muon transformer normalization interaction probe
3. Optional Muon Cap-family a-regime sensitivity screen
4. ResNet-20 BatchNorm cross-check with checkpoint export for post-hoc telemetry

This script is the single entrypoint. It reuses the existing Muon and ResNet
experiment runners so checkpointing/resume behavior remains aligned with the
rest of the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "muon_clamp_epochs": 4,
        "muon_norm_epochs": 2,
        "muon_seeds": 1,
        "muon_norm_modes": ["layernorm", "identity"],
        "muon_norm_activations": ["SwishCap", "SparseCap", "ReLU", "GELU"],
        "muon_regime_norm_modes": ["layernorm", "identity"],
        "resnet_epochs": 10,
        "resnet_seeds": [0],
        "resnet_checkpoint_epochs": [10],
        "resnet_activations": [
            "ReLU",
            "GELU",
            "SwishCap_astar_fixed",
            "SparseCap_astar_fixed",
        ],
        "resnet_extra_regimes": [
            "SwishCap_1_learnable",
            "SparseCap_1_learnable",
        ],
    },
    "paper": {
        "muon_clamp_epochs": 30,
        "muon_norm_epochs": 10,
        "muon_seeds": 3,
        "muon_norm_modes": ["layernorm", "rmsnorm", "identity"],
        "muon_norm_activations": [
            "SwishCap",
            "SparseCap",
            "ReLU",
            "GELU",
            "ReLU6",
            "HardTanh",
        ],
        "muon_regime_norm_modes": ["layernorm", "identity"],
        "resnet_epochs": 30,
        "resnet_seeds": [0, 1, 2],
        "resnet_checkpoint_epochs": [10, 30],
        "resnet_activations": [
            "ReLU",
            "GELU",
            "ReLU6",
            "HardTanh",
            "SwishCap_astar_fixed",
            "SparseCap_astar_fixed",
        ],
        "resnet_extra_regimes": [
            "SwishCap_1_learnable",
            "SparseCap_1_learnable",
        ],
    },
}

VALID_STAGES = (
    "muon_clamp",
    "muon_norm",
    "muon_regimes",
    "resnet_bn",
)
VALID_RESNET_BN_MODES = ("with_bn", "without_bn")


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def run_command(command: Sequence[str], *, cwd: Path, dry_run: bool) -> float:
    rendered = " ".join(command)
    print(rendered)
    if dry_run:
        return 0.0
    start = time.time()
    subprocess.run(list(command), cwd=str(cwd), check=True)
    return time.time() - start


def build_muon_command(
    python_bin: str,
    *,
    repo_root: Path,
    output_dir: Path,
    epochs: int,
    seeds: int,
    activations: Sequence[str],
    norm_mode: str = "layernorm",
    lr: float | None = None,
    wd: float | None = None,
    clamp: float | None = None,
    verification: bool = True,
    only_inits: Sequence[str] | None = None,
    only_a_policies: Sequence[str] | None = None,
    resume: bool = True,
) -> List[str]:
    cmd = [
        python_bin,
        "scripts/experiments/run_muon.py",
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--seeds",
        str(seeds),
        "--norm-mode",
        norm_mode,
        "--only-activations",
        ",".join(activations),
    ]
    if verification:
        cmd.append("--verification")
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if wd is not None:
        cmd.extend(["--wd", str(wd)])
    if clamp is not None:
        cmd.extend(["--qk-score-clamp", str(clamp)])
    if only_inits:
        cmd.extend(["--only-inits", ",".join(only_inits)])
    if only_a_policies:
        cmd.extend(["--only-a-policies", ",".join(only_a_policies)])
    cmd.append("--resume" if resume else "--no-resume")
    return cmd


def build_resnet_command(
    python_bin: str,
    *,
    output_dir: Path,
    activation: str,
    seed: int,
    epochs: int,
    checkpoint_epochs: Sequence[int],
    use_batch_norm: bool,
    resume: bool,
    force: bool,
) -> List[str]:
    cmd = [
        python_bin,
        "scripts/experiments/cv/resnet20_experiment.py",
        "--activation",
        activation,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if checkpoint_epochs:
        cmd.append("--checkpoint-epoch")
        cmd.extend([str(epoch) for epoch in checkpoint_epochs])
    cmd.append("--resume" if resume else "--no-resume")
    if force:
        cmd.append("--force")
    if not use_batch_norm:
        cmd.append("--no-batch-norm")
    return cmd


def summarize_resnet_bn(output_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for results_path in sorted(output_dir.glob("*_bn/seed*/**/*_results.json")):
        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        bn_mode = results_path.parents[1].name
        rows.append(
            {
                "activation": payload["activation"],
                "bn_mode": bn_mode,
                "seed": int(payload["hyperparameters"]["seed"]),
                "best_val_accuracy": float(payload["best_val_accuracy"]),
                "test_accuracy": float(payload["test_accuracy"]),
                "final_train_accuracy": float(payload["final_train_accuracy"]),
                "use_batch_norm": bool(payload["hyperparameters"]["use_batch_norm"]),
                "checkpoint_epochs": payload["hyperparameters"].get("checkpoint_epochs", []),
                "results_file": str(results_path),
            }
        )

    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        activation_bucket = grouped.setdefault(row["activation"], {})
        mode_bucket = activation_bucket.setdefault(
            row["bn_mode"],
            {
                "use_batch_norm": row["use_batch_norm"],
                "seeds": [],
                "best_val_accuracy": [],
                "test_accuracy": [],
                "final_train_accuracy": [],
            },
        )
        mode_bucket["seeds"].append(row["seed"])
        mode_bucket["best_val_accuracy"].append(row["best_val_accuracy"])
        mode_bucket["test_accuracy"].append(row["test_accuracy"])
        mode_bucket["final_train_accuracy"].append(row["final_train_accuracy"])

    summary: Dict[str, Any] = {
        "num_runs": len(rows),
        "by_activation": {},
    }
    for activation, by_mode in grouped.items():
        summary["by_activation"][activation] = {}
        for bn_mode, bucket in by_mode.items():
            summary["by_activation"][activation][bn_mode] = {
                "use_batch_norm": bucket["use_batch_norm"],
                "seeds": sorted(bucket["seeds"]),
                "best_val_accuracy": mean_std(bucket["best_val_accuracy"]),
                "test_accuracy": mean_std(bucket["test_accuracy"]),
                "final_train_accuracy": mean_std(bucket["final_train_accuracy"]),
            }

    summary_path = output_dir / "aggregate_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# ResNet BatchNorm Cross-Check Summary",
        "",
        f"- Runs found: {len(rows)}",
        "",
        "| Activation | BN mode | Best val (%) | Test acc (%) | Final train (%) | Seeds |",
        "|---|---|---:|---:|---:|---|",
    ]
    for activation in sorted(summary["by_activation"]):
        for bn_mode in sorted(summary["by_activation"][activation]):
            bucket = summary["by_activation"][activation][bn_mode]
            best_val = bucket["best_val_accuracy"]
            test_acc = bucket["test_accuracy"]
            final_train = bucket["final_train_accuracy"]
            lines.append(
                "| "
                f"{activation} | {bn_mode} | "
                f"{best_val['mean']:.2f} ± {best_val['std']:.2f} | "
                f"{test_acc['mean']:.2f} ± {test_acc['std']:.2f} | "
                f"{final_train['mean']:.2f} ± {final_train['std']:.2f} | "
                f"{','.join(str(seed) for seed in bucket['seeds'])} |"
            )

    summary_md = output_dir / "aggregate_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_manifest(manifest_dir: Path, manifest: Dict[str, Any], repo_root: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / "suite_manifest.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    lines = [
        "# Paper Confound Audit",
        "",
        f"- Profile: `{manifest['profile']}`",
        f"- Suite tag: `{manifest['suite_tag']}`",
        f"- Resume: `{manifest['resume']}`",
        f"- Include a-sensitivity: `{manifest['include_a_sensitivity']}`",
        f"- Cap activation for clamp follow-up: `{manifest['cap_activation']}`",
        "",
        "## Stage Plan",
        "",
    ]
    for stage in manifest["stages"]:
        lines.append(f"### {stage['name']}")
        lines.append(f"- Description: {stage['description']}")
        lines.append(f"- Output dir: `{stage['output_dir']}`")
        lines.append(f"- Status: `{stage['status']}`")
        if stage.get("elapsed_sec") is not None:
            lines.append(f"- Elapsed seconds: `{stage['elapsed_sec']:.1f}`")
        lines.append("- Commands:")
        for command in stage["commands"]:
            lines.append(f"  - `{command['command']}`")
        if stage.get("notes"):
            lines.append("- Notes:")
            for note in stage["notes"]:
                lines.append(f"  - {note}")
        lines.append("")

    md_path = manifest_dir / "suite_manifest.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_stage_selection(raw: str, include_a_sensitivity: bool) -> List[str]:
    if raw.strip().lower() == "all":
        stages = ["muon_clamp", "muon_norm", "resnet_bn"]
        if include_a_sensitivity:
            stages.insert(2, "muon_regimes")
        return stages
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in VALID_STAGES]
    if invalid:
        raise ValueError(f"Unknown stage(s): {invalid}. Valid: {list(VALID_STAGES)} or all")
    if not include_a_sensitivity:
        requested = [item for item in requested if item != "muon_regimes"]
    return requested


def parse_csv_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def parse_int_csv_list(raw: str | None) -> List[int] | None:
    items = parse_csv_list(raw)
    if items is None:
        return None
    return [int(item) for item in items]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="paper")
    parser.add_argument("--suite-tag", default="paper_confound_audit")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated subset of stages to run. Choices: all, muon_clamp, muon_norm, muon_regimes, resnet_bn.",
    )
    parser.add_argument("--cap-activation", choices=["SoftCap", "SwishCap", "SparseCap"], default="SwishCap")
    parser.add_argument("--include-a-sensitivity", action="store_true", help="Add the optional Cap-family a-regime sensitivity stage.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for sub-runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force-resnet", action="store_true", help="Pass --force to ResNet sub-runs.")
    parser.add_argument(
        "--resnet-activations",
        default=None,
        help="Optional comma-separated filter for the resnet_bn stage (e.g. HardTanh,SwishCap_astar_fixed).",
    )
    parser.add_argument(
        "--resnet-seeds",
        default=None,
        help="Optional comma-separated filter for the resnet_bn stage seeds (e.g. 2 or 0,2).",
    )
    parser.add_argument(
        "--resnet-bn-modes",
        default=None,
        help="Optional comma-separated filter for the resnet_bn stage: with_bn,without_bn.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    profile = PROFILE_CONFIGS[args.profile]
    selected_stages = parse_stage_selection(args.stages, args.include_a_sensitivity)
    resnet_activation_filter = parse_csv_list(args.resnet_activations)
    resnet_seed_filter = parse_int_csv_list(args.resnet_seeds)
    resnet_bn_mode_filter = parse_csv_list(args.resnet_bn_modes)
    if resnet_bn_mode_filter is not None:
        invalid_modes = [mode for mode in resnet_bn_mode_filter if mode not in VALID_RESNET_BN_MODES]
        if invalid_modes:
            raise ValueError(
                f"Unknown resnet BN mode(s): {invalid_modes}. Valid: {list(VALID_RESNET_BN_MODES)}"
            )

    suite_root = repo_root / "runs" / args.suite_tag
    muon_root = suite_root / "muon"
    resnet_root = suite_root / "resnet_bn"
    manifest_root = suite_root / "manifest"

    manifest: Dict[str, Any] = {
        "profile": args.profile,
        "suite_tag": args.suite_tag,
        "resume": args.resume,
        "include_a_sensitivity": args.include_a_sensitivity,
        "cap_activation": args.cap_activation,
        "stages": [],
    }

    stage_specs: List[Dict[str, Any]] = []

    if "muon_clamp" in selected_stages:
        output_dir = muon_root / "clamp_longhorizon"
        activations = [args.cap_activation, "ReLU", "GELU"]
        commands = [
            build_muon_command(
                args.python,
                repo_root=repo_root,
                output_dir=output_dir,
                epochs=profile["muon_clamp_epochs"],
                seeds=profile["muon_seeds"],
                activations=activations,
                norm_mode="layernorm",
                lr=0.1,
                wd=0.01,
                clamp=None,
                verification=True,
                resume=args.resume,
            ),
            build_muon_command(
                args.python,
                repo_root=repo_root,
                output_dir=output_dir,
                epochs=profile["muon_clamp_epochs"],
                seeds=profile["muon_seeds"],
                activations=activations,
                norm_mode="layernorm",
                lr=0.1,
                wd=0.01,
                clamp=30.0,
                verification=True,
                resume=args.resume,
            ),
        ]
        stage_specs.append(
            {
                "name": "muon_clamp",
                "description": "Longer-horizon high-LR Muon clamp follow-up using the paper-default Cap activation against ReLU/GELU.",
                "output_dir": output_dir,
                "commands": commands,
                "notes": [
                    "Cap-family activations use the verification regime: softcap_optimal init + a=a* fixed.",
                    "ReLU/GELU use kaiming + a=1.0.",
                    "Clamp threshold is fixed at 30 based on the earlier 10-epoch diagnostics.",
                ],
            }
        )

    if "muon_norm" in selected_stages:
        output_dir = muon_root / "norm_probe"
        commands = []
        for norm_mode in profile["muon_norm_modes"]:
            commands.append(
                build_muon_command(
                    args.python,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    epochs=profile["muon_norm_epochs"],
                    seeds=profile["muon_seeds"],
                    activations=profile["muon_norm_activations"],
                    norm_mode=norm_mode,
                    lr=0.02,
                    wd=0.01,
                    verification=True,
                    resume=args.resume,
                )
            )
        stage_specs.append(
            {
                "name": "muon_norm",
                "description": "Transformer-first normalization interaction probe across a compact verification set.",
                "output_dir": output_dir,
                "commands": commands,
                "notes": [
                    "Norm modes are ordered from baseline to stronger perturbation: LayerNorm, RMSNorm, Identity.",
                    "This stage is a transformer-local mechanism probe, not a norm-elimination benchmark.",
                ],
            }
        )

    if "muon_regimes" in selected_stages:
        output_dir = muon_root / "norm_regime_sensitivity"
        commands = []
        for norm_mode in profile["muon_regime_norm_modes"]:
            commands.append(
                build_muon_command(
                    args.python,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    epochs=profile["muon_norm_epochs"],
                    seeds=profile["muon_seeds"],
                    activations=["SwishCap", "SparseCap"],
                    norm_mode=norm_mode,
                    lr=0.02,
                    wd=0.01,
                    verification=False,
                    only_inits=["softcap_optimal"],
                    only_a_policies=["a=1.0", "a=a*", "a=learnable"],
                    resume=args.resume,
                )
            )
        stage_specs.append(
            {
                "name": "muon_regimes",
                "description": "Optional Cap-family a-regime sensitivity screen for the transformer normalization probe.",
                "output_dir": output_dir,
                "commands": commands,
                "notes": [
                    "Restricted to SwishCap and SparseCap because those are the paper-facing Cap variants.",
                    "Uses softcap_optimal init only, to keep the regime comparison compact and interpretable.",
                ],
            }
        )

    if "resnet_bn" in selected_stages:
        activations = list(profile["resnet_activations"])
        if args.include_a_sensitivity:
            activations.extend(profile["resnet_extra_regimes"])
        if resnet_activation_filter is not None:
            invalid_activations = [act for act in resnet_activation_filter if act not in activations]
            if invalid_activations:
                raise ValueError(
                    f"Unknown resnet activation(s) for profile '{args.profile}': {invalid_activations}"
                )
            activations = [act for act in activations if act in set(resnet_activation_filter)]

        resnet_seeds = list(profile["resnet_seeds"])
        if resnet_seed_filter is not None:
            invalid_seeds = [seed for seed in resnet_seed_filter if seed not in resnet_seeds]
            if invalid_seeds:
                raise ValueError(
                    f"Unknown resnet seed(s) for profile '{args.profile}': {invalid_seeds}"
                )
            resnet_seeds = [seed for seed in resnet_seeds if seed in set(resnet_seed_filter)]

        bn_modes = list(VALID_RESNET_BN_MODES)
        if resnet_bn_mode_filter is not None:
            bn_modes = [mode for mode in bn_modes if mode in set(resnet_bn_mode_filter)]

        commands: List[List[str]] = []
        for bn_mode in bn_modes:
            use_batch_norm = (bn_mode == "with_bn")
            for seed in resnet_seeds:
                run_dir = resnet_root / bn_mode / f"seed{seed}"
                for activation in activations:
                    commands.append(
                        build_resnet_command(
                            args.python,
                            output_dir=run_dir,
                            activation=activation,
                            seed=seed,
                            epochs=profile["resnet_epochs"],
                            checkpoint_epochs=profile["resnet_checkpoint_epochs"],
                            use_batch_norm=use_batch_norm,
                            resume=args.resume,
                            force=args.force_resnet,
                        )
                    )
        stage_specs.append(
            {
                "name": "resnet_bn",
                "description": "Supportive ResNet-20 BatchNorm cross-check with explicit Cap-family regimes and checkpoint export.",
                "output_dir": resnet_root,
                "commands": commands,
                "notes": [
                    "ResNet defaults to the paper-facing Cap regimes: SwishCap_a* fixed and SparseCap_a* fixed.",
                    "Checkpoint epochs are chosen for later post-hoc telemetry rather than dense checkpoint retention.",
                    f"Activation filter: {','.join(activations) if activations else '(none)'}",
                    f"Seed filter: {','.join(str(seed) for seed in resnet_seeds) if resnet_seeds else '(none)'}",
                    f"BN mode filter: {','.join(bn_modes) if bn_modes else '(none)'}",
                ],
            }
        )

    write_manifest(manifest_root, manifest, repo_root)

    for stage_spec in stage_specs:
        print(f"\n{'=' * 88}")
        print(f"[Stage] {stage_spec['name']}")
        print(stage_spec["description"])
        print(f"Output dir: {relpath(stage_spec['output_dir'], repo_root)}")
        print(f"{'=' * 88}\n")

        stage_record: Dict[str, Any] = {
            "name": stage_spec["name"],
            "description": stage_spec["description"],
            "output_dir": relpath(stage_spec["output_dir"], repo_root),
            "status": "running",
            "commands": [],
            "notes": stage_spec.get("notes", []),
            "elapsed_sec": None,
        }
        manifest["stages"].append(stage_record)
        write_manifest(manifest_root, manifest, repo_root)

        stage_start = time.time()
        try:
            for command in stage_spec["commands"]:
                elapsed = run_command(command, cwd=repo_root, dry_run=args.dry_run)
                stage_record["commands"].append(
                    {
                        "command": " ".join(command),
                        "elapsed_sec": elapsed,
                    }
                )
                write_manifest(manifest_root, manifest, repo_root)

            if not args.dry_run and stage_spec["name"] == "resnet_bn":
                summarize_resnet_bn(stage_spec["output_dir"])

            stage_record["status"] = "completed"
        except subprocess.CalledProcessError as exc:
            stage_record["status"] = f"failed ({exc.returncode})"
            write_manifest(manifest_root, manifest, repo_root)
            raise
        finally:
            stage_record["elapsed_sec"] = time.time() - stage_start
            write_manifest(manifest_root, manifest, repo_root)

    print("\nPaper confound audit complete.")
    print(f"Manifest: {relpath(manifest_root, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
