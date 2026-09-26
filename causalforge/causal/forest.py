"""Optional Causal Forest backend (EconML).

Demonstrates the "skip missing heavy backend" discipline: if ``econml`` is not
installed in the environment the estimator reports ``available()=False`` and
the benchmark skips it (flagged in the report) instead of failing. The
gradient-boosted DML in :mod:`causalforge.causal.gdml` is the always-available
SOTA backend; this is a complementary forest-based SOTA method.
"""

import numpy as np

from ..core.interfaces import BaseCausalEstimator
from ..core.types import EstimatorResult

_Z95 = 1.959963984540054


def available_econml() -> bool:
    try:
        import econml  # noqa: F401
        return True
    except Exception:
        return False


class CausalForestEconML(BaseCausalEstimator):
    name = "causal_forest_econml"
    backend = "tier0"

    def __init__(self, seed: int = 42, n_estimators: int = 200):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self._ate = None
        self._se = None

    def available(self) -> bool:
        return available_econml()

    def fit(self, data) -> "CausalForestEconML":
        if not self.available():
            from ..core.errors import BackendUnavailableError

            raise BackendUnavailableError("econml")
        from econml.dml import CausalForestDML

        est = CausalForestDML(
            model_y="auto", model_t="auto", discrete_treatment=True,
            n_estimators=self.n_estimators, random_state=self.seed,
        )
        est.fit(data.Y, data.T, X=data.X)
        # econml 0.17: ``ate`` is a method (takes X); ``ate_stderr_`` is the
        # per-sample standard-error attribute. Mean over samples = aggregate SE.
        point = est.ate(data.X)
        self._ate = float(np.mean(point))
        self._se = float(np.mean(est.ate_stderr_))
        self._fitted = True
        return self

    def estimate_ate(self, data) -> EstimatorResult:
        if not self._fitted:
            self.fit(data)
        return EstimatorResult(
            name=self.name, ate=self._ate, ate_se=self._se,
            ci_low=self._ate - _Z95 * self._se, ci_high=self._ate + _Z95 * self._se,
            backend="econml",
        )
