"""Protocol / base class for causal estimators.

Every estimator (baseline, Tier-1 fallback, Tier-0 SOTA backend) conforms to
``BaseCausalEstimator``. This keeps the pipeline and the benchmark loop
agnostic to the concrete method and lets us swap backends by configuration.
"""

from abc import ABC, abstractmethod

from .types import CausalDataset, EstimatorResult


class BaseCausalEstimator(ABC):
    """Common contract for all ATE estimators."""

    name: str = "base"
    backend: str = "tier1"

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._fitted = False

    @abstractmethod
    def fit(self, data: CausalDataset) -> "BaseCausalEstimator":
        """Fit any required nuisance models on ``data``."""

    @abstractmethod
    def estimate_ate(self, data: CausalDataset) -> EstimatorResult:
        """Return an :class:`EstimatorResult` for the ATE of ``data``."""

    def available(self) -> bool:
        """Whether the estimator (and its backend) can run in this env."""
        return True
