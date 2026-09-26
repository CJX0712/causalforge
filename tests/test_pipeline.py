import json

from causalforge.core.config import Config
from causalforge.pipeline.pipeline import ESTIMATOR_FACTORIES, CausalPipeline


def _cfg(monkeypatch, samples, reps, splits=3):
    monkeypatch.setenv("CAUSALFORGE_N_SAMPLES", str(samples))
    monkeypatch.setenv("CAUSALFORGE_N_REPS", str(reps))
    monkeypatch.setenv("CAUSALFORGE_N_SPLITS", str(splits))
    return Config.from_env()


def test_benchmark_runs_and_serialises(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, 400, 4)
    pipe = CausalPipeline(cfg)
    rows = pipe.benchmark(seed=7)
    # causal_forest is excluded from the default MC loop (heavy + budget);
    # assert the returned estimator set matches the expected skip list.
    expected = [k for k in ESTIMATOR_FACTORIES if k not in ("causal_forest",)]
    assert [r.estimator for r in rows] == expected
    path = pipe.to_json(rows, str(tmp_path / "bench.json"))
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "results" in payload
    for r in payload["results"]:
        if r["available"]:
            assert r["mean_ate"] is not None
        else:  # skipped backend (e.g. econml) -> null, never a leaked NaN
            assert r["mean_ate"] is None


def test_determinism_bit_for_bit(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, 400, 4)
    pipe = CausalPipeline(cfg)
    r1 = pipe.benchmark(seed=11)
    p1 = pipe.to_json(r1, str(tmp_path / "a.json"))
    r2 = pipe.benchmark(seed=11)
    p2 = pipe.to_json(r2, str(tmp_path / "b.json"))
    with open(p1, encoding="utf-8") as f:
        a = f.read()
    with open(p2, encoding="utf-8") as f:
        b = f.read()
    assert a == b  # identical bytes -> deterministic


def test_probe_backends_reports_availability(monkeypatch):
    cfg = _cfg(monkeypatch, 400, 4)
    pipe = CausalPipeline(cfg)
    probe = pipe.probe_backends(seed=9)
    assert "lightgbm" in probe and "xgboost" in probe and "econml" in probe
    # EconML is installed in the locked env -> forest is probed once.
    assert probe.get("econml") is True
    ate = probe.get("causal_forest_single_ate")
    assert ate is not None and abs(ate - 1.0) < 0.6  # finite, near truth


def test_sota_bias_reduction_vs_naive_in_benchmark(monkeypatch):
    cfg = _cfg(monkeypatch, 800, 10)
    pipe = CausalPipeline(cfg)
    rows = pipe.benchmark(seed=3)
    by = {r.estimator: r for r in rows}
    naive, gdml = by["naive"], by["gdml"]
    assert naive.bias is not None and gdml.bias is not None
    # SOTA must cut naive bias by a large margin (delivery threshold >= 60%)
    reduction = 1.0 - gdml.bias / naive.bias
    assert reduction >= 0.6, f"bias reduction too small: {reduction:.3f}"
