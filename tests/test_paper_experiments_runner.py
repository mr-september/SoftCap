from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_paper_experiments.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_main_dry_run_prints_canonical_commands():
    completed = _run("all-main", "--profile", "quick", "--dry-run")
    assert completed.returncode == 0

    stdout = completed.stdout
    assert "modular_arithmetic_benchmark.py --all-activations --mode standard --quick" in stdout
    assert "heavy_tailed_ood_benchmark.py --paper-set --epochs 10 --n-seeds 1" in stdout
    assert "radial_sector_benchmark.py --paper-set --epochs 10 --n-seeds 1" in stdout
    assert "scripts/experiments/run_muon.py --verification --quick --epochs 3 --seeds 1" in stdout


def test_confounds_dry_run_uses_single_audit_entrypoint():
    completed = _run("confounds", "--profile", "quick", "--dry-run", "--no-resume")
    assert completed.returncode == 0

    stdout = completed.stdout
    assert "scripts/experiments/run_paper_confound_audit.py --profile quick --suite-tag paper_confound_audit --no-resume --dry-run" in stdout
