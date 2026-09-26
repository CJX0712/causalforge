"""Core data types for CausalForge."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalDataset:
    """A causal estimation problem.

    Convention: ``Y = g(X) + tau * T + noise`` with binary or continuous
    treatment ``T``. When ``true_ate`` is provided the dataset can be used as
    a ground-truth benchmark.
    """

    X: np.ndarray  # covariates, shape (n, d)
    T: np.ndarray  # treatment, shape (n,)
    Y: np.ndarray  # outcome, shape (n,)
    true_ate: float | None = None
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.T = np.asarray(self.T, dtype=np.float64).ravel()
        self.Y = np.asarray(self.Y, dtype=np.float64).ravel()
        n = self.X.shape[0]
        if self.T.shape[0] != n or self.Y.shape[0] != n:
            raise ValueError("X, T and Y must share the same number of rows")


@dataclass
class EstimatorResult:
    """Result of a single ATE estimation."""

    name: str
    ate: float
    ate_se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    backend: str = "tier1"
    available: bool = True
    note: str = ""


@dataclass
class BenchmarkRow:
    """One row of the Monte-Carlo benchmark table."""

    dataset: str
    true_ate: float | None
    n_samples: int
    n_reps: int
    estimator: str
    backend: str
    available: bool
    mean_ate: float
    bias: float | None
    rmse: float | None
    coverage: float | None
    mean_ci_width: float | None
    ablation: str = ""
