"""Naive difference-in-means estimator (strong baseline / control).

This is the textbook "compare averages" estimator. It ignores confounders
entirely, so on a confounded problem it is *biased*. We keep it as the
reference baseline the SOTA backend must beat by a quantified margin.
"""

import numpy as np

from ..core.interfaces import BaseCausalEstimator
from ..core.types import EstimatorResult

_Z95 = 1.959963984540054


class NaiveDifference(BaseCausalEstimator):
    name = "naive_difference"
    backend = "baseline"

    def fit(self, data) -> "NaiveDifference":
        self._fitted = True
        return self

    def estimate_ate(self, data) -> EstimatorResult:
        t1 = data.Y[data.T > 0.5]
        t0 = data.Y[data.T <= 0.5]
        if len(t1) == 0 or len(t0) == 0:
            return EstimatorResult(name=self.name, ate=float("nan"), backend=self.backend,
                                   available=True, note="no treated/control units")
        ate = float(np.mean(t1) - np.mean(t0))
        v1 = np.var(t1, ddof=1) / len(t1) if len(t1) > 1 else 0.0
        v0 = np.var(t0, ddof=1) / len(t0) if len(t0) > 1 else 0.0
        se = float(np.sqrt(v1 + v0))
        return EstimatorResult(
            name=self.name, ate=ate, ate_se=se,
            ci_low=ate - _Z95 * se, ci_high=ate + _Z95 * se,
            backend=self.backend,
        )
