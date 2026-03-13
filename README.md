# Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SoftCap Family](paper/figures/fig1_softcap_family.png)

## Overview

`SoftCap` is the public sister repository for the v2 preprint. It exposes a
small, stable release surface:

- Core family: `SoftCap`, `SwishCap`, `SparseCap`
- Release controls: `ReLU`, `Tanh`, `GELU`, `SiLU`
- Appendix-only bounded controls: `ReLU6`, `HardTanh` via explicit opt-in helpers
- Single paper-facing runner: `python scripts/run_paper_experiments.py ...`

The current bundled paper PDF is the refreshed v2 preprint: [main.pdf](main.pdf).

## Canonical Activation Set

| Canonical name | Legacy alias | Role |
| :--- | :--- | :--- |
| **SoftCap** | `ParametricTanhSoftCap` | Bounded half-rectifier baseline |
| **SwishCap** | `ParametricSmoothNotchTanhSoftCapV2` | Smooth negative-branch Cap variant |
| **SparseCap** | `ParametricQuinticNotchTanhSoftCap` | Hard-zero sparse Cap variant |

## Install

```bash
git clone https://github.com/mr-september/SoftCap.git
cd SoftCap
pip install -r requirements.txt
```

## Quick Usage

```python
import torch
from softcap.activations import SparseCap

act = SparseCap(a_init=1.0)
x = torch.randn(32, 128)
y = act(x)
```

## Reproducibility

The public reproduction surface is the paper runner:

```bash
python scripts/run_paper_experiments.py --help
```

Run one paper-facing experiment:

```bash
python scripts/run_paper_experiments.py grokking --profile paper
python scripts/run_paper_experiments.py ood-heavy --profile paper
python scripts/run_paper_experiments.py ood-angular --profile paper
python scripts/run_paper_experiments.py muon --profile paper
python scripts/run_paper_experiments.py confounds --profile paper
```

Or run the main bundle:

```bash
python scripts/run_paper_experiments.py all-main --profile paper
```

`--profile quick` is the smoke-test mode for each experiment.

## Tests

```bash
pytest tests/
```

## Citation

If you use this work, please cite the preprint:

**Cai, L., & Tang, J. (2026). Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity. Zenodo. https://doi.org/10.5281/zenodo.18829083**

```bibtex
@article{cai2026beyond,
  title={Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity},
  author={Cai, Larry and Tang, Jie},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18829083},
  url={https://doi.org/10.5281/zenodo.18829083}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
