# Experiments

This directory contains the lower-level experiment implementations behind the
public release runners.

For paper-facing reproduction, start at the repo root with:

```bash
python scripts/run_paper_experiments.py --help
```

The release policy is simple:

- expose stable public commands through `scripts/run_paper_experiments.py`
- keep activation selection in suite helpers such as `softcap.control_activations`
- keep appendix-only controls opt-in rather than part of the default surface

## Structure

```text
experiments/
├── grokking/            # Modular arithmetic phase-transition benchmark
├── ood/                 # Heavy-tailed and radial-sector OOD benchmarks
├── representation/      # Representation and SAE benchmarks
├── signal_propagation/  # Matched-init matrix benchmark
└── base/                # Shared base experiment utilities
```

## Standardized Activation Suites

Use the suite helpers instead of manually assembling activation lists:

```python
from softcap.control_activations import (
    get_standard_experimental_set,
    get_bounded_controls,
)

paper_suite = get_standard_experimental_set()
appendix_controls = get_bounded_controls()
```

The default paper-facing suite is:

- `SoftCap`, `SwishCap`, `SparseCap`
- `ReLU`, `Tanh`, `GELU`, `SiLU`

Appendix-only bounded controls are currently:

- `ReLU6`
- `HardTanh`

## Guidance

If you extend the public release surface:

- prefer adding a new subcommand to `scripts/run_paper_experiments.py`
- keep output roots under `runs/`
- keep legacy aliases only for compatibility with old artifacts and summaries
