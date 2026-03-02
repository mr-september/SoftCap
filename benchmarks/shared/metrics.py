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
Metrics computation and statistical testing for scaling benchmarks.
"""

from typing import Dict, List
import numpy as np
from scipy import stats


def aggregate_seeds(results: List[Dict[str, float]], key: str) -> Dict[str, float]:
    """Compute mean ± std across seeds for a given metric.

    Args:
        results: List of per-seed result dicts.
        key: The metric name to aggregate.

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'n' for that metric.
    """
    values = [r[key] for r in results if key in r]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(arr),
    }


def pairwise_ttest(
    group_a: List[float],
    group_b: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Independent two-sample t-test between two activation conditions.

    Returns t-statistic, p-value, and Cohen's d effect size.
    """
    a, b = np.array(group_a), np.array(group_b)
    t_stat, p_val = stats.ttest_ind(a, b, alternative=alternative)

    # Cohen's d
    pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 +
                          (len(b) - 1) * b.std(ddof=1) ** 2) /
                         (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "n_a": len(a),
        "n_b": len(b),
    }


def build_comparison_table(
    all_results: Dict[str, List[Dict[str, float]]],
    metric_key: str,
    baseline_name: str = "ReLU",
) -> str:
    """Build a Markdown comparison table across activations.

    Args:
        all_results: {activation_name: [per_seed_results, ...]}
        metric_key: Which metric to compare (e.g. 'test_acc').
        baseline_name: Which activation to use as statistical reference.

    Returns:
        Markdown table string.
    """
    lines = [
        f"| Activation | {metric_key} (mean ± std) | vs {baseline_name} p-value |",
        "| :--- | ---: | ---: |",
    ]

    baseline_vals = [r[metric_key] for r in all_results.get(baseline_name, [])]

    for name, results in sorted(all_results.items()):
        agg = aggregate_seeds(results, metric_key)
        mean_std = f"{agg['mean']:.2f} ± {agg['std']:.2f}"

        if name == baseline_name or not baseline_vals:
            pval_str = "—"
        else:
            vals = [r[metric_key] for r in results]
            tt = pairwise_ttest(vals, baseline_vals)
            pval_str = f"{tt['p_value']:.4f}"

        lines.append(f"| {name} | {mean_std} | {pval_str} |")

    return "\n".join(lines)
