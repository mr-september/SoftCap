#!/usr/bin/env python3
"""Single paper-facing entry point for the release experiment suite.

This script keeps the public reproduction surface small. Each subcommand maps
to one paper-facing experiment and delegates to the underlying benchmark
runners with the canonical activation set and profile defaults already chosen.

Examples:
    python scripts/run_paper_experiments.py grokking --profile paper
    python scripts/run_paper_experiments.py muon --profile quick
    python scripts/run_paper_experiments.py all-main --profile paper --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ORDER = (
    "grokking",
    "ood-heavy",
    "ood-angular",
    "muon",
)


def _python_bin(args: argparse.Namespace) -> str:
    return args.python or sys.executable


def _resume_flag(args: argparse.Namespace) -> str:
    return "--resume" if args.resume else "--no-resume"


def _append_resume(command: List[str], args: argparse.Namespace) -> List[str]:
    command.append(_resume_flag(args))
    return command


def _build_grokking_command(args: argparse.Namespace) -> List[str]:
    command = [
        _python_bin(args),
        "experiments/grokking/modular_arithmetic_benchmark.py",
        "--all-activations",
        "--mode",
        "standard",
    ]
    if args.profile == "quick":
        command.extend(
            [
                "--quick",
                "--output-dir",
                "runs/paper/grokking_quick",
            ]
        )
    else:
        command.extend(
            [
                "--epochs",
                "10000",
                "--n-seeds",
                "3",
                "--eval-interval",
                "100",
                "--output-dir",
                "runs/paper/grokking",
            ]
        )
    return _append_resume(command, args)


def _build_ood_heavy_command(args: argparse.Namespace) -> List[str]:
    command = [
        _python_bin(args),
        "experiments/ood/heavy_tailed_ood_benchmark.py",
        "--paper-set",
    ]
    if args.profile == "quick":
        command.extend(
            [
                "--epochs",
                "10",
                "--n-seeds",
                "1",
                "--output-dir",
                "runs/paper/ood_heavy_quick",
            ]
        )
    else:
        command.extend(
            [
                "--epochs",
                "100",
                "--n-seeds",
                "5",
                "--output-dir",
                "runs/paper/ood_heavy",
            ]
        )
    return _append_resume(command, args)


def _build_ood_angular_command(args: argparse.Namespace) -> List[str]:
    command = [
        _python_bin(args),
        "experiments/ood/radial_sector_benchmark.py",
        "--paper-set",
    ]
    if args.profile == "quick":
        command.extend(
            [
                "--epochs",
                "10",
                "--n-seeds",
                "1",
                "--output-dir",
                "runs/paper/ood_angular_quick",
            ]
        )
    else:
        command.extend(
            [
                "--epochs",
                "100",
                "--n-seeds",
                "5",
                "--output-dir",
                "runs/paper/ood_angular",
            ]
        )
    return _append_resume(command, args)


def _build_muon_command(args: argparse.Namespace) -> List[str]:
    command = [
        _python_bin(args),
        "scripts/experiments/run_muon.py",
        "--verification",
    ]
    if args.profile == "quick":
        command.extend(
            [
                "--quick",
                "--epochs",
                "3",
                "--seeds",
                "1",
                "--output-dir",
                "runs/paper/muon_quick",
            ]
        )
    else:
        command.extend(
            [
                "--epochs",
                "30",
                "--seeds",
                "3",
                "--output-dir",
                "runs/paper/muon",
            ]
        )
    return _append_resume(command, args)


def _build_confounds_command(args: argparse.Namespace) -> List[str]:
    command = [
        _python_bin(args),
        "scripts/experiments/run_paper_confound_audit.py",
        "--profile",
        "quick" if args.profile == "quick" else "paper",
        "--suite-tag",
        "paper_confound_audit",
    ]
    if not args.resume:
        command.append("--no-resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_commands(experiment: str, args: argparse.Namespace) -> List[List[str]]:
    builders = {
        "grokking": [_build_grokking_command],
        "ood-heavy": [_build_ood_heavy_command],
        "ood-angular": [_build_ood_angular_command],
        "muon": [_build_muon_command],
        "confounds": [_build_confounds_command],
        "all-main": [
            _build_grokking_command,
            _build_ood_heavy_command,
            _build_ood_angular_command,
            _build_muon_command,
        ],
        "all-release": [
            _build_grokking_command,
            _build_ood_heavy_command,
            _build_ood_angular_command,
            _build_muon_command,
            _build_confounds_command,
        ],
    }
    try:
        return [builder(args) for builder in builders[experiment]]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment group: {experiment}") from exc


def _render_command(command: Sequence[str]) -> str:
    return " ".join(command)


def _run_commands(commands: Iterable[Sequence[str]], *, dry_run: bool) -> int:
    for command in commands:
        print(_render_command(command))
        if dry_run:
            continue
        completed = subprocess.run(list(command), cwd=str(PROJECT_ROOT))
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=("grokking", "ood-heavy", "ood-angular", "muon", "confounds", "all-main", "all-release"),
        help="Paper-facing experiment or bundle to run.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "paper"),
        default="paper",
        help="Use the quick smoke-test profile or the paper reproduction profile.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for delegated runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume sub-runs when supported.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)

    commands = build_commands(args.experiment, args)
    return _run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
