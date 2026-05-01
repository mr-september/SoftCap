# Designing Bounded Activations: Geometric Control of Neural Network Stability, Sparsity, Quantization, and Energy-Based Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SoftCap Family](paper/figures/fig1_softcap_family.png)

## Abstract

While unbounded activations like ReLU and SiLU drive modern architectures, their lack of geometric constraints creates shared failure modes for stability, quantization, and implicit modeling. We address this with a constraints-first design framework yielding the **SoftCap family**, a $C^0$-$C^2$ progression of bounded rectifiers: **SoftCap** (exact-zero), **SwishCap** ($C^1$ derivative-matched), and **SparseCap** ($C^2$ quintic notch), ready as drop-in replacements. These functions are derived in closed form and anchored by variance-preserving initialization, replacing benchmark-driven empirical search with principled forward constraints.

In high-learning-rate grokking stress tests, SwishCap ($a=1$, fixed) achieves 100% survival across all 16 aggressive configurations, supporting the mechanism that origin-adjacent recovery geometry and tight forward scale jointly expand the safe operating region. In transformers, the SoftCap family provides raw perplexity gains over standard GELU baselines while reducing peak attention scores by 3--4$\times$, decreasing reliance on explicit clamping; concurrently, bounded FFNs suppress post-activation outliers and reduce dynamic INT8 quantization sensitivity. Across heavy-tailed OOD shifts, the same bounded geometry compresses outlier logit gaps by up to 85$\times$. Structurally, SparseCap natively generates 8.9% structural sparsity in NanoGPT query activations, establishing the mathematical foundation for sparse attention without post-hoc thresholding.

Finally, in diagnostic EBMs, bounded geometry reshapes the learned energy/score field: the SoftCap family suppresses score-tail excursions, reduces spurious drift across disconnected modes, and raises the high-fidelity sampling ceiling in controlled topologies. Together, these findings support geometric activation bounds as a shared mechanism for regulating failure modes across explicit and implicit architectures, offering a unified framework for robust model design.

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

**Cai, L., & Tang, J. (2026). Designing Bounded Activations: Geometric Control of Neural Network Stability, Sparsity, Quantization, and Energy-Based Models. Zenodo. https://doi.org/10.5281/zenodo.18829082**

```bibtex
@article{cai2026designing,
  title={Designing Bounded Activations: Geometric Control of Neural Network Stability, Sparsity, Quantization, and Energy-Based Models},
  author={Cai, Larry and Tang, Jie},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18829082},
  url={https://doi.org/10.5281/zenodo.18829082}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
