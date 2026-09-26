import numpy as np
import pytest

from causalforge.core import errors
from causalforge.core.config import Config
from causalforge.core.seed import set_all
from causalforge.core.types import CausalDataset


def test_config_default_and_validate():
    cfg = Config.from_env()
    cfg.validate()  # must not raise
    assert cfg.seed == 42
    assert cfg.n_splits >= 2


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CAUSALFORGE_N_SAMPLES", "500")
    monkeypatch.setenv("CAUSALFORGE_BACKEND_PREF", "sklearn")
    cfg = Config.from_env()
    assert cfg.n_samples == 500
    assert cfg.backend_pref == "sklearn"


def test_config_invalid_backend(monkeypatch):
    monkeypatch.setenv("CAUSALFORGE_BACKEND_PREF", "torchxxxx")
    with pytest.raises(errors.ConfigError):
        Config.from_env()


def test_seed_determinism():
    set_all(42)
    a = np.random.rand(10)
    set_all(42)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_types_row_mismatch():
    with pytest.raises(ValueError):
        CausalDataset(X=np.zeros((5, 2)), T=np.zeros(4), Y=np.zeros(5))


def test_error_codes():
    assert errors.ConfigError().code == "C100"
    assert errors.DataError().code == "C200"
    assert errors.EstimatorError().code == "C300"
    assert errors.PipelineError().code == "C500"
