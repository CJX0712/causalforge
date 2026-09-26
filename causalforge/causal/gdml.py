"""Cross-fitted Double Machine Learning (R-learner) -- Tier-0 SOTA backend.

Implements the partially-linear DML estimator (Chernozhukov et al., 2018) with
sample-splitting (cross-fitting) for low bias and valid inference:

    Y = theta * T + g(X) + noise
    T = m(X) + error

Out-of-fold nuisance fits (``g`` for Y|X, ``m`` for T|X) produce residuals;
theta is the closed-form OLS coefficient of Y_res on T_res, and its standard
error follows the DML influence-function variance. Nuisance models are taken
from top open-source gradient boosters (LightGBM / XGBoost) with a
scikit-learn gradient-boosting fallback, so the method degrades gracefully.

The ablation variant disables cross-fitting (nuisances fit on the full sample)
to demonstrate the bias it introduces -- a citable property of DML.
"""

import numpy as np
from sklearn.model_selection import KFold

from ..core.interfaces import BaseCausalEstimator
from ..core.types import EstimatorResult

_Z95 = 1.959963984540054


def available_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def available_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_backend(pref: str) -> str:
    if pref in ("lightgbm", "xgboost"):
        ok = available_lightgbm() if pref == "lightgbm" else available_xgboost()
        return pref if ok else "sklearn"
    # auto: prefer the fastest C++ booster, fall back to sklearn
    if available_lightgbm():
        return "lightgbm"
    if available_xgboost():
        return "xgboost"
    return "sklearn"


def _make_regressor(backend: str, seed: int):
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, random_state=seed, n_jobs=1, verbose=-1,
        )
    if backend == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=4, random_state=seed,
    )


class GradientDML(BaseCausalEstimator):
    name = "gradient_dml"
    backend = "tier0"

    def __init__(self, seed: int = 42, n_splits: int = 4, cross_fit: bool = True,
                 backend: str = "auto"):
        super().__init__(seed)
        self.n_splits = int(n_splits)
        self.cross_fit = cross_fit
        self.backend_pref = backend
        self._backend = _resolve_backend(backend)
        self._ate = None
        self._se = None

    def available(self) -> bool:
        return True  # works with the sklearn fallback

    def _estimate(self, data):
        backend = self._backend
        X, T, Y = data.X, data.T, data.Y
        n = len(Y)

        if self.cross_fit and self.n_splits >= 2 and n >= self.n_splits * 10:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            y_res = np.empty(n)
            t_res = np.empty(n)
            for tr, te in kf.split(X):
                g = _make_regressor(backend, self.seed)
                m = _make_regressor(backend, self.seed)
                g.fit(X[tr], Y[tr])
                m.fit(X[tr], T[tr])
                y_res[te] = Y[te] - g.predict(X[te])
                t_res[te] = T[te] - m.predict(X[te])
        else:
            g = _make_regressor(backend, self.seed)
            m = _make_regressor(backend, self.seed)
            g.fit(X, Y)
            m.fit(X, T)
            y_res = Y - g.predict(X)
            t_res = T - m.predict(X)

        denom = float(np.sum(t_res ** 2))
        if denom <= 1e-12:
            theta = 0.0
        else:
            theta = float(np.sum(t_res * y_res) / denom)

        resid_orth = y_res - theta * t_res
        var = np.mean(resid_orth ** 2 * t_res ** 2) / (np.mean(t_res ** 2) ** 2 + 1e-12)
        se = float(np.sqrt(var / n))
        return theta, se, backend

    def fit(self, data) -> "GradientDML":
        self._ate, self._se, self._backend = self._estimate(data)
        self._fitted = True
        return self

    def estimate_ate(self, data) -> EstimatorResult:
        if not self._fitted:
            self.fit(data)
        label = self.name + ("" if self.cross_fit else "_nocv")
        note = "cross-fit (DML)" if self.cross_fit else "no cross-fit (ablation)"
        return EstimatorResult(
            name=label, ate=self._ate, ate_se=self._se,
            ci_low=self._ate - _Z95 * self._se, ci_high=self._ate + _Z95 * self._se,
            backend=self._backend, note=note,
        )
