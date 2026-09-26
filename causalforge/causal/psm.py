"""Propensity-score matching estimator (Tier-1 offline fallback, sklearn-free core).

Implements caliper-based nearest-neighbour matching with replacement on a
logistic-regression propensity score. Caliper matching (only pair units whose
propensity scores are within a tolerance) plus 1:k matching with replacement
is the standard, robust PSM recipe and avoids the bias that naive 1:1 matching
suffers in the no-common-support tail. A symmetric (treated- and
control-anchored) scheme yields an ATE estimate. This is the always-available
offline fallback and a mid-tier baseline.
"""

import numpy as np

from ..core.interfaces import BaseCausalEstimator
from ..core.types import EstimatorResult

_Z95 = 1.959963984540054


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _fit_propensity(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Propensity scores via logistic regression (scikit-learn, deterministic)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(x, y.astype(int))
    return model.predict_proba(x)[:, 1]


def _match(e, anchor_mask, other_mask, y, k, caliper):
    """Nearest-k caliper matching (with replacement).

    Returns list of ``y[anchor] - y[other]`` for each matched pair.
    """
    anchors = np.where(anchor_mask)[0]
    others = np.where(other_mask)[0]
    if len(others) == 0:
        return []
    diffs = []
    for a in anchors:
        d = np.abs(e[others] - e[a])
        mask = d <= caliper
        if not mask.any():
            continue
        cand = others[mask]
        dd = np.abs(e[cand] - e[a])
        pick = cand[np.argsort(dd)[:k]]
        for o in pick:
            diffs.append(y[a] - y[o])
    return diffs


class PropensityScoreMatching(BaseCausalEstimator):
    name = "propensity_score_matching"
    backend = "tier1"

    def __init__(self, seed: int = 42, match_ratio: int = 3, caliper: "float | None" = None):
        super().__init__(seed)
        self.match_ratio = int(match_ratio)
        self.caliper = caliper
        self._ate = None
        self._se = None

    def fit(self, data) -> "PropensityScoreMatching":
        xs = _standardize(data.X)
        e = _fit_propensity(xs, data.T, self.seed)
        caliper = self.caliper if self.caliper is not None else 0.2 * float(np.std(e))

        # treated-anchored -> Yt - Yc ; control-anchored -> Yc - Yt (negate)
        diffs_t = _match(e, data.T > 0.5, data.T <= 0.5, data.Y, self.match_ratio, caliper)
        diffs_c = _match(e, data.T <= 0.5, data.T > 0.5, data.Y, self.match_ratio, caliper)
        diffs = np.array(list(diffs_t) + [-x for x in diffs_c], dtype=np.float64)

        if len(diffs) == 0:
            self._ate = float("nan")
            self._se = None
        else:
            self._ate = float(np.mean(diffs))
            self._se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        self._fitted = True
        return self

    def estimate_ate(self, data) -> EstimatorResult:
        if not self._fitted:
            self.fit(data)
        se = self._se if self._se is not None else 0.0
        finite = np.isfinite(self._ate)
        return EstimatorResult(
            name=self.name, ate=self._ate, ate_se=se,
            ci_low=(self._ate - _Z95 * se) if finite else None,
            ci_high=(self._ate + _Z95 * se) if finite else None,
            backend=self.backend,
        )
