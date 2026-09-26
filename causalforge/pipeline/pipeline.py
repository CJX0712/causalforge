"""CausalPipeline: orchestrates estimators and the Monte-Carlo benchmark.

Call graph (single direction, acyclic):
    cli -> pipeline -> {data, causal, eval} -> core

``run`` fits every estimator on a single dataset; ``benchmark`` repeats the
estimation over ``n_reps`` fresh datasets with known ATE to measure bias,
RMSE and coverage. A fixed seed drives the whole loop so two runs are
bit-for-bit identical.
"""

import json
import math
from dataclasses import asdict

from ..causal.forest import CausalForestEconML
from ..causal.gdml import GradientDML
from ..causal.naive import NaiveDifference
from ..causal.ols import OLSAdjustment
from ..causal.psm import PropensityScoreMatching
from ..core.config import Config
from ..core.seed import set_all
from ..core.types import BenchmarkRow, CausalDataset, EstimatorResult
from ..data.synthetic import generate_dataset
from ..eval.metrics import aggregate, relative_bias_reduction

# estimator factories keyed by a stable id (order = display order)
ESTIMATOR_FACTORIES = {
    "naive": lambda c: NaiveDifference(seed=c.seed),
    "ols": lambda c: OLSAdjustment(seed=c.seed),
    "psm": lambda c: PropensityScoreMatching(seed=c.seed, match_ratio=c.match_ratio),
    "gdml": lambda c: GradientDML(seed=c.seed, n_splits=c.n_splits,
                                  cross_fit=True, backend=c.backend_pref),
    "gdml_nocv": lambda c: GradientDML(seed=c.seed, n_splits=c.n_splits,
                                       cross_fit=False, backend=c.backend_pref),
    "causal_forest": lambda c: CausalForestEconML(seed=c.seed),
}


def _sanitize(value):
    """Make a value JSON-safe (NaN/inf -> None)."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _result_to_dict(res: EstimatorResult) -> dict:
    return {k: _sanitize(v) for k, v in asdict(res).items()}


class CausalPipeline:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config.from_env()

    # -- single-dataset estimation ---------------------------------------
    def run(self, data: CausalDataset) -> dict:
        out: dict = {}
        for key, factory in ESTIMATOR_FACTORIES.items():
            est = factory(self.cfg)
            try:
                if not est.available():
                    out[key] = EstimatorResult(
                        name=key, ate=float("nan"), backend=est.backend,
                        available=False, note="backend unavailable")
                    continue
                out[key] = est.estimate_ate(data)
            except Exception as exc:  # defensive: never abort the whole loop
                out[key] = EstimatorResult(
                    name=key, ate=float("nan"), backend=est.backend,
                    available=False, note=f"error: {exc}")
        return out

    # -- backend probe ----------------------------------------------------
    def probe_backends(self, seed: int = 12345, tau: float = 1.0) -> dict:
        """Report backend availability and a single causal-forest ATE.

        The Monte-Carlo benchmark deliberately excludes the heavy ensemble
        forest backend (``causal_forest``) to protect the 60s performance
        budget; this probe runs it once on a single dataset so the shipped
        report still demonstrates the backend works end-to-end.
        """
        from ..causal.forest import CausalForestEconML, available_econml
        from ..causal.gdml import available_lightgbm, available_xgboost

        probe: dict = {
            "lightgbm": available_lightgbm(),
            "xgboost": available_xgboost(),
            "econml": available_econml(),
        }
        if available_econml():
            try:
                set_all(seed)
                data = generate_dataset(seed=seed, n_samples=self.cfg.n_samples, tau=tau)
                res = CausalForestEconML(seed=seed).estimate_ate(data)
                probe["causal_forest_single_ate"] = _sanitize(res.ate)
                probe["causal_forest_single_se"] = _sanitize(res.ate_se)
            except Exception as exc:  # never let the probe abort delivery
                probe["causal_forest_error"] = str(exc)
        return probe

    # -- Monte-Carlo benchmark -------------------------------------------
    def benchmark(self, seed: "int | None" = None, tau: float = 1.0,
                  skip: tuple = ("causal_forest",)) -> list:
        cfg = self.cfg
        base = int(seed) if seed is not None else cfg.seed
        rows: list = []
        for key, factory in ESTIMATOR_FACTORIES.items():
            if key in skip:
                continue
            est = factory(cfg)
            reps: list = []
            for r in range(cfg.n_reps):
                s = base + r
                set_all(s)
                data = generate_dataset(seed=s, n_samples=cfg.n_samples, tau=tau)
                try:
                    if not est.available():
                        reps.append(EstimatorResult(
                            name=key, ate=float("nan"), backend=est.backend,
                            available=False, note="backend unavailable"))
                        continue
                    reps.append(est.estimate_ate(data))
                except Exception as exc:
                    reps.append(EstimatorResult(
                        name=key, ate=float("nan"), backend=est.backend,
                        available=False, note=f"error: {exc}"))
            mean_ate, bias, rmse, coverage, width = aggregate(reps, tau)
            bk = reps[0].backend if reps else "?"
            rows.append(BenchmarkRow(
                dataset="synthetic", true_ate=tau, n_samples=cfg.n_samples,
                n_reps=cfg.n_reps, estimator=key, backend=bk,
                available=all(r.available for r in reps),
                mean_ate=mean_ate, bias=bias, rmse=rmse,
                coverage=coverage, mean_ci_width=width,
                ablation=("ablation: cross-fitting disabled" if key == "gdml_nocv" else ""),
            ))
        return rows

    # -- serialisation ----------------------------------------------------
    def to_json(self, rows, path, extra: "dict | None" = None) -> str:
        serial = []
        for r in rows:
            d = {k: _sanitize(v) for k, v in asdict(r).items()}
            serial.append(d)
        payload: dict = {
            "config": asdict(self.cfg),
            "true_ate_target": 1.0,
            "results": serial,
        }
        if extra is not None:
            payload["backend_probe"] = extra
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def to_json_single(self, results: dict, path) -> str:
        serial = {k: _result_to_dict(v) for k, v in results.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"config": asdict(self.cfg), "results": serial},
                      fh, indent=2, ensure_ascii=False)
        return path

    def summary(self, rows) -> dict:
        """Compute headline metrics: bias reduction of SOTA vs baseline."""
        by_name = {r.estimator: r for r in rows}
        naive = by_name.get("naive")
        gdml = by_name.get("gdml")
        out = {}
        if naive and gdml and naive.bias is not None and gdml.bias is not None:
            out["sota_bias_reduction_vs_naive"] = relative_bias_reduction(naive.bias, gdml.bias)
        return out
