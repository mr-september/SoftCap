# Beyond ReLU and GELU: SoftCap Bounded Activations for Stability and Sparsity

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SoftCap Family](paper/figures/fig1_softcap_family.png)

## Abstract

We introduce the **SoftCap family**, bounded rectifying activations derived from explicit continuity and sparsity constraints rather than empirical search [@ramachandran2017searching]. The family comprises **SoftCap** ($C^0$), **SwishCap** ($C^1$), and **SparseCap** ($C^2$), all sharing a bounded positive branch $a\tanh(x)$ with analytically derived, variance-preserving scalar $a^*$ [@glorot2010understanding; @he2015delving; @klambauer2017selu].

In high-learning-rate grokking stress tests, SwishCap achieves 100% survival across all tested rates, whereas hard-zero variants exhibit sharp collapse boundaries, indicating that origin smoothness and negative-side gradient transport govern stability more strongly than boundedness alone [@power2022grokking; @balduzzi2017shattered]. Applied after Q/K projections in Muon-trained ViTs, bounded activations reduce peak pre-softmax attention scores by 3–4×, reducing reliance on explicit clamping [@vaswani2017attention; @dosovitskiy2021vit]. Under heavy-tailed contamination, they suppress outlier logit gaps by over two orders of magnitude, imposing an architectural confidence ceiling without explicit calibration [@ovadia2019can; @guo2017calibration].

While trailing ReLU/GELU by $\approx$4 pp in standard supervised regimes [@nair2010relu; @hendrycks2016gelu], these results establish a constrained design map in which continuity order and notch geometry determine predictable trade-offs across stability, sparsity, and dynamic-range control.

---

## Install

```bash
git clone https://github.com/mr-september/SoftCap.git
cd SoftCap
pip install -r requirements.txt
```

## PyTorch Implementation & Quick Usage

The canonical PyTorch implementations of the SoftCap family are provided below for immediate accessibility.

```python
import torch
import torch.nn as nn

class SoftCap(nn.Module):
    """SoftCap: f(x)=0 for x<=0 else a*tanh(x)."""
    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.a, min=1e-3)
        output = torch.zeros_like(x)
        pos = x > 0
        output[pos] = a * torch.tanh(x[pos])
        return output

class SwishCap(nn.Module):
    """SwishCap: C1 smooth notch + tanh positive branch scaled by a."""
    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.a, min=1e-3)
        s = torch.sigmoid(a * x)
        neg = 2 * a * x * s
        pos = a * torch.tanh(x)
        return torch.where(x <= 0, neg, pos)

class SparseCap(nn.Module):
    """SparseCap: minimum-degree C2 hard-zero quintic notch with parametric scale a."""
    def __init__(self, a_init: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.a, min=1e-3)
        notch = (x > -a) & (x <= 0.0)
        pos = x > 0.0

        output = torch.zeros_like(x)
        x_q = x[notch]
        x_plus_a = x_q + a
        output[notch] = x_q * (x_plus_a ** 3) * (a - 3.0 * x_q) / (a ** 3)
        output[pos] = a * torch.tanh(x[pos])
        return output
```

You can also import them from the module or copy them directly into your project:

```python
import torch
from softcap.activations import SoftCap, SwishCap, SparseCap

act = SparseCap(a_init=1.0)
x = torch.randn(32, 128)
y = act(x)
```

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

The full paper is also available as [main.pdf](main.pdf).

## License

Apache 2.0. See [LICENSE](LICENSE).
