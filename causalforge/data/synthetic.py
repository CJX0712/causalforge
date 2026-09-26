"""Synthetic data generator with a known ground-truth ATE.

The generator builds a *non-linear, high-dimensional* confounded
treatment-effect problem -- the regime where classical adjustment breaks:

    score(X) = kappa * (X0^2 + X1^2 + X2^2 + X3*X4 + 0.5*sin(X5))  # nonlinear 7-dim
    e(X)    = clip(sigmoid(score), 0.10, 0.90)                      # propensity
    T       ~ Bernoulli(e(X))
    g0(X)   = 0.5*(X0^2 + X1^2 + X2^2) + 0.5*(X3*X4)                # nonlinear (confounded)
            + 0.3*sin(X5) + 0.4*(X0 + X1)                           # + linear tail
    Y       = g0(X) + tau*T + noise

Because the propensity is a *non-linear* function of many covariates, a
linear/logistic propensity-score model (used by PSM) is misspecified and PSM
stays biased; an OLS linear adjuster likewise cannot remove the non-linear
confounding. Cross-fitted DML with gradient boosting models the full
non-linear ``g0`` and ``m`` and recovers ``tau``. ``tau`` is returned as
``true_ate`` so the benchmark can measure bias and CI coverage exactly.
"""

import numpy as np

from ..core.seed import set_all
from ..core.types import CausalDataset


def _make_cov(d: int, seed: int) -> np.ndarray:
    """Build a positive-definite, mildly correlated covariance matrix."""
    rng = np.random.RandomState(seed + 777)
    a = rng.randn(d, d)
    cov = a @ a.T / d
    cov = cov + d * np.eye(d)  # shift eigenvalues away from zero
    sd = np.sqrt(np.diag(cov))
    cov = cov / np.outer(sd, sd)
    return cov


def generate_dataset(
    seed: int = 42,
    n_samples: int = 2000,
    d: int = 8,
    tau: float = 1.0,
    confounding: float = 1.2,
    noise_scale: float = 0.5,
) -> CausalDataset:
    """Generate a confounded dataset with a known ATE = ``tau``."""
    set_all(seed)
    rng = np.random.RandomState(seed)
    cov = _make_cov(d, seed)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n_samples)

    # non-linear, high-dimensional confounder drives treatment
    score = confounding * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2
                           + X[:, 3] * X[:, 4] + 0.5 * np.sin(X[:, 5]))
    e = np.clip(1.0 / (1.0 + np.exp(-score)), 0.10, 0.90)
    T = (rng.rand(n_samples) < e).astype(np.float64)

    g0 = (
        0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)
        + 0.5 * (X[:, 3] * X[:, 4])
        + 0.3 * np.sin(X[:, 5])
        + 0.4 * (X[:, 0] + X[:, 1])
    )
    Y = g0 + tau * T + rng.randn(n_samples) * noise_scale

    return CausalDataset(X=X, T=T, Y=Y, true_ate=tau, name="synthetic")
