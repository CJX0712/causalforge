"""Benchmark aggregation metrics.

Given a list of :class:`EstimatorResult` (one per Monte-Carlo replication) and
the known ground-truth ATE, compute the summary statistics used in the
benchmark table: mean ATE, absolute bias, RMSE, 95% CI coverage and mean CI
width. ``NaN`` results (estimator errors) are excluded so a single failure
never poisons the aggregate.
"""

import numpy as np


def _finite(results):
    return [r for r in results if r.ate is not None and np.isfinite(r.ate)]


def aggregate(results, true_ate):
    """Return (mean_ate, bias, rmse, coverage, mean_ci_width).

    ``bias`` / ``rmse`` / ``coverage`` are ``None`` when ``true_ate`` is
    unknown or no finite result exists.
    """
    good = _finite(results)
    if not good:
        return float("nan"), None, None, None, None

    ates = np.array([r.ate for r in good], dtype=np.float64)
    mean_ate = float(np.mean(ates))

    if true_ate is None:
        return mean_ate, None, None, None, None

    bias = float(np.abs(mean_ate - true_ate))
    rmse = float(np.sqrt(np.mean((ates - true_ate) ** 2)))

    inside = [
        (r.ci_low is not None and r.ci_high is not None
         and r.ci_low <= true_ate <= r.ci_high)
        for r in good
    ]
    coverage = float(np.mean(inside)) if inside else None

    widths = [r.ci_high - r.ci_low for r in good
              if r.ci_low is not None and r.ci_high is not None]
    mean_ci_width = float(np.mean(widths)) if widths else None

    return mean_ate, bias, rmse, coverage, mean_ci_width


def relative_bias_reduction(baseline_bias, candidate_bias):
    """Fractional reduction of bias vs the baseline (clamped to [-inf, 1])."""
    if baseline_bias is None or candidate_bias is None:
        return None
    if baseline_bias <= 1e-9:
        return 0.0
    return float(max(-9.0, 1.0 - candidate_bias / baseline_bias))
