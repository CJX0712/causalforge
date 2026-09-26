"""Global deterministic seeding for CausalForge.

A single entry point sets every RNG we depend on. Determinism is a hard
requirement of the delivery spec: the demo must produce identical
benchmark.json on two consecutive runs with the same seed.
"""

import os
import random

import numpy as np

_SEEDED = False


def set_all(seed: int = 42) -> None:
    """Set all global RNG states for full reproducibility.

    Covers the Python ``random`` module, NumPy's legacy + default RNG, and
    (best-effort) PyTorch. Estimator-level libraries (LightGBM / XGBoost /
    scikit-learn) receive their own ``random_state`` argument so they do not
    rely on global state.
    """
    global _SEEDED
    seed = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    _SEEDED = True


def get_seed() -> int:
    return int(os.environ.get("PYTHONHASHSEED", "42"))
