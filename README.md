# Designing Bounded Activations: Geometric Control of Neural Network Stability, Sparsity, Quantization, and Energy-Based Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SoftCap Family](paper/figures/fig1_softcap_family.png)

## Abstract

While unbounded activations like ReLU and SiLU drive modern architectures, their lack of geometric constraints necessitates compensatory architectures to mitigate failure modes in stability, quantization, and implicit modeling. We address this with a constraints-first design framework yielding the **SoftCap family**, a $C^0$-$C^2$ progression of bounded rectifiers: **SoftCap** (exact-zero), **SwishCap** ($C^1$ derivative-matched), and **SparseCap** ($C^2$ quintic notch), ready as drop-in replacements. Derived in closed form and anchored by variance-preserving initialization, we replace benchmark-driven empirical search with principled forward constraints.

In grokking stress tests, SwishCap achieves 100% survivorship across all 16 aggressive configurations, supporting the mechanism that origin-adjacent recovery geometry and tight forward scale jointly expand the safe operating region. In transformers, the SoftCap family remains stable at learning-rate multipliers of **up to 80$\times$** where standard controls collapse, while providing perplexity gains over standard GELU baselines and reducing peak attention scores by 3--4$\times$, decreasing reliance on explicit clamping; concurrently, bounded FFNs suppress post-activation outliers by 15$\times$ and reduce INT8 quantization-induced perplexity degradation by **over 25%**. Across heavy-tailed OOD shifts, the same bounded geometry compresses outlier logit gaps by up to 85$\times$. Structurally, SparseCap natively generates 8.9% structural sparsity in NanoGPT query activations, establishing the mathematical foundation for sparse attention without post-hoc thresholding.

Finally, in energy-based models (EBMs), bounded geometry provides an architectural alternative to objective-level landscape regularization: the SoftCap family suppresses score-tail excursions, reduces spurious drift, and raises the high-fidelity sampling ceiling. Together, these findings support geometric activation bounds as a shared mechanism for regulating failure modes across explicit and implicit architectures, offering a unified framework for robust model design.

---

## Install

```bash
git clone https://github.com/mr-september/SoftCap.git
cd SoftCap
pip install -r requirements.txt
```

## PyTorch Implementation & Quick Usage

The canonical PyTorch implementations of the SoftCap family are provided in `softcap/activations.py`.

```python
import torch
from softcap.activations import SoftCap, SwishCap, SparseCap

# Recommended initialization (a*) is derived from variance mapping
from softcap.control_activations import A_STAR_VALUES

act = SparseCap(a_init=A_STAR_VALUES["SparseCap"])
x = torch.randn(32, 128)
y = act(x)
```

## Research Tracks

This repository is organized into several research tracks corresponding to the main experiments in the paper:

- **Grokking & Stability**: Stress-testing continuity and boundedness in high-LR modular arithmetic.
- **OOD Robustness**: Evaluating behavior under heavy-tailed and angular covariate shifts.
- **NLP (NanoGPT)**: Decoder-only transformer scaling with bounded attention projections (Q/K).
- **Robustness Probe**: Testing activation mechanism consistency across objective families (BCE, Hinge, MSE, InfoNCE).
- **Muon-trained ViTs**: Evaluation in high-performance vision transformers.

To run the experiments:

```bash
# Run a quick smoke test for all main experiments
python scripts/run_paper_experiments.py all-main --profile quick

# Run full paper reproduction (requires GPU)
python scripts/run_paper_experiments.py all-main --profile paper
```

## Tests

```bash
pytest tests/
```

## Citation

If you use this work, please cite the preprint:

```bibtex
This repository is anonymized for double-blind review. A full citation to the published paper and the corresponding reproducibility archives will be provided upon de-anonymization.
```

## License

Apache 2.0. See [LICENSE](LICENSE).
