"""OLS adjustment estimator (Tier-1 offline fallback, pure numpy).

Fits ``Y ~ T + X`` by ordinary least squares in closed form and reads the
treatment coefficient as the ATE. No external ML dependency: this is the
always-available fallback when gradient-boosting backends are missing.
"""

import numpy as np

from ..core.interfaces import BaseCausalEstimator
from ..core.types import EstimatorResult

_Z95 = 1.959963984540054


class OLSAdjustment(BaseCausalEstimator):
    name = "ols_adjustment"
    backend = "tier1"

    def __init__(self, seed: int = 42, add_intercept: bool = True):
        super().__init__(seed)
        self.add_intercept = add_intercept
        self._coef = None
        self._se = None

    def fit(self, data) -> "OLSAdjustment":
        X = data.X
        cols = [X, data.T.reshape(-1, 1)]
        if self.add_intercept:
            cols.insert(0, np.ones((X.shape[0], 1)))
        Xd = np.hstack(cols)
        y = data.Y
        xtx = Xd.T @ Xd
        xtx_inv = np.linalg.inv(xtx)
        beta = xtx_inv @ Xd.T @ y
        resid = y - Xd @ beta
        n, p = Xd.shape
        sigma2 = float(resid @ resid) / max(n - p, 1)
        cov = sigma2 * xtx_inv
        # coefficient of T is the last column
        self._coef = float(beta[-1])
        self._se = float(np.sqrt(max(cov[-1, -1], 0.0)))
        self._fitted = True
        return self

    def estimate_ate(self, data) -> EstimatorResult:
        if not self._fitted:
            self.fit(data)
        se = self._se if self._se is not None else 0.0
        return EstimatorResult(
            name=self.name, ate=self._coef, ate_se=se,
            ci_low=self._coef - _Z95 * se, ci_high=self._coef + _Z95 * se,
            backend=self.backend,
        )
