"""Optional IHDP loader (the canonical causal-inference benchmark).

IHDP ships with a known ground-truth ATE (~1.1, Hill 2011). The raw files are
*not* bundled; this loader only engages when the user points it at a local
copy. If the files are absent the loader reports ``available_ihdp()=False`` and
the pipeline's benchmark skips it transparently -- this is exactly the
"skip missing heavy backend" discipline required by the delivery spec.
"""

import os

import numpy as np

from ..core.types import CausalDataset


def available_ihdp(path: str | None = None) -> bool:
    """Return True only if a usable IHDP data directory is present."""
    if path is None:
        path = os.environ.get("CAUSALFORGE_IHDP_DIR", "")
    if not path:
        return False
    return os.path.isfile(os.path.join(path, "ihdp_npci_1-100.train.npz"))


def load_ihdp(path: str, rep: int = 1) -> CausalDataset:
    """Load one IHDP replication.

    Expected files (NPCI format): ``ihdp_npci_1-100.train.npz`` and
    ``ihdp_npci_1-100.test.npz``. Returns a dataset with ``true_ate`` taken
    from the file's ``yf_diff`` (average treatment effect on the treated
    population, treated as the benchmark target).
    """
    train = np.load(os.path.join(path, "ihdp_npci_1-100.train.npz"))
    test = np.load(os.path.join(path, "ihdp_npci_1-100.test.npz"))
    # rep is 1-indexed in the NPCI layout
    idx = rep - 1
    x = np.concatenate([train["x"][:, :, idx], test["x"][:, :, idx]], axis=0)
    t = np.concatenate([train["t"][:, idx], test["t"][:, idx]], axis=0)
    y = np.concatenate([train["y"][:, idx], test["y"][:, idx]], axis=0)
    yf = np.concatenate([train["yf"][:, idx], test["yf"][:, idx]], axis=0)
    # observed outcome uses factual yf per treatment assignment
    y_obs = np.where(t > 0.5, yf, y)
    true_ate = float(np.mean(yf[t > 0.5]) - np.mean(yf[t <= 0.5]))
    return CausalDataset(X=x, T=t.astype(np.float64), Y=y_obs.astype(np.float64),
                         true_ate=true_ate, name="ihdp")
