import numpy as np

from causalforge.data.loaders import available_ihdp
from causalforge.data.synthetic import generate_dataset


def test_generate_shape_and_true_ate():
    data = generate_dataset(seed=1, n_samples=300, tau=1.0)
    assert data.X.shape == (300, 8)
    assert data.T.shape == (300,)
    assert data.Y.shape == (300,)
    assert data.true_ate == 1.0


def test_generate_deterministic():
    a = generate_dataset(seed=123, n_samples=200, tau=0.8)
    b = generate_dataset(seed=123, n_samples=200, tau=0.8)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.T, b.T)
    assert np.array_equal(a.Y, b.Y)


def test_ihdp_unavailable_by_default():
    # No IHDP dir bundled -> loader must report unavailable (skip discipline)
    assert available_ihdp("") is False
