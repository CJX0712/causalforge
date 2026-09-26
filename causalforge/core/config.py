"""Configuration with ENV_CAUSALFORGE_* overrides and schema validation.

Configuration is read once and validated. Every field has a sane default and
can be overridden through environment variables, which keeps the system
twelve-factor and CI-friendly.
"""

import os
from dataclasses import dataclass

from .errors import ConfigError


@dataclass
class Config:
    seed: int = 42
    n_samples: int = 2000
    n_reps: int = 14
    n_splits: int = 4
    test_size: float = 0.5
    ci_alpha: float = 0.05
    backend_pref: str = "auto"  # auto | lightgbm | xgboost | sklearn
    match_ratio: int = 3  # 1:k nearest-neighbour matching (with replacement) for PSM

    @classmethod
    def from_env(cls) -> "Config":
        def _int(key: str, default: int) -> int:
            raw = os.environ.get(f"CAUSALFORGE_{key}")
            return int(raw) if raw is not None and raw != "" else default

        def _float(key: str, default: float) -> float:
            raw = os.environ.get(f"CAUSALFORGE_{key}")
            return float(raw) if raw is not None and raw != "" else default

        def _str(key: str, default: str) -> str:
            raw = os.environ.get(f"CAUSALFORGE_{key}")
            return raw if raw is not None and raw != "" else default

        cfg = cls(
            seed=_int("SEED", 42),
            n_samples=_int("N_SAMPLES", 2000),
            n_reps=_int("N_REPS", 10),
            n_splits=_int("N_SPLITS", 4),
            test_size=_float("TEST_SIZE", 0.5),
            ci_alpha=_float("CI_ALPHA", 0.05),
            backend_pref=_str("BACKEND_PREF", "auto"),
            match_ratio=_int("MATCH_RATIO", 3),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("CAUSALFORGE_SEED must be >= 0")
        if self.n_samples < 50:
            raise ConfigError("CAUSALFORGE_N_SAMPLES must be >= 50")
        if self.n_reps < 1:
            raise ConfigError("CAUSALFORGE_N_REPS must be >= 1")
        if self.n_splits < 2:
            raise ConfigError("CAUSALFORGE_N_SPLITS must be >= 2")
        if not (0.0 < self.test_size < 1.0):
            raise ConfigError("CAUSALFORGE_TEST_SIZE must be in (0, 1)")
        if not (0.0 < self.ci_alpha < 1.0):
            raise ConfigError("CAUSALFORGE_CI_ALPHA must be in (0, 1)")
        if self.backend_pref not in ("auto", "lightgbm", "xgboost", "sklearn"):
            raise ConfigError("CAUSALFORGE_BACKEND_PREF must be auto|lightgbm|xgboost|sklearn")
        if self.match_ratio < 1:
            raise ConfigError("CAUSALFORGE_MATCH_RATIO must be >= 1")
