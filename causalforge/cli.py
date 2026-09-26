"""Command-line entry point for CausalForge.

Usage:
    python -m causalforge.cli benchmark [--out benchmark.json] [--seed N]
    python -m causalforge.cli run [--seed N] [--n N_SAMPLES] [--out RESULT.json]
    python -m causalforge.cli version
"""

import argparse
import sys

from .core.config import Config
from .core.seed import set_all
from .data.synthetic import generate_dataset
from .pipeline.pipeline import CausalPipeline

_TABLE_HEADERS = ["estimator", "backend", "mean_ate", "bias", "rmse", "cov@95", "ci_w"]


def _fmt(value, width, kind="f"):
    if value is None:
        s = "-"
    elif kind == "f":
        s = f"{value:.4f}"
    elif kind == "p":
        s = f"{value:.3f}"
    else:
        s = str(value)
    return f"{s:>{width}}"


def print_table(rows) -> None:
    widths = [12, 10, 11, 9, 9, 9, 9]
    print("  ".join(h.ljust(w) for h, w in zip(_TABLE_HEADERS, widths)))
    for r in rows:
        print("  ".join([
            r.estimator.ljust(widths[0]),
            str(r.backend).ljust(widths[1]),
            _fmt(r.mean_ate, widths[2]),
            _fmt(r.bias, widths[3]),
            _fmt(r.rmse, widths[4]),
            _fmt(r.coverage, widths[5], "p"),
            _fmt(r.mean_ci_width, widths[6]),
        ]))


def _ensure_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(argv=None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="causalforge",
        description="CausalForge: world-class causal effect estimation")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("benchmark", help="Monte-Carlo benchmark vs baselines")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument("--seed", type=int, default=None)

    r = sub.add_parser("run", help="Estimate ATE on a single synthetic dataset")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--n", type=int, default=1500, dest="n_samples")
    r.add_argument("--out", default=None)

    sub.add_parser("version", help="Print version and exit")

    args = parser.parse_args(argv)
    cfg = Config.from_env()
    pipe = CausalPipeline(cfg)

    if args.cmd == "version":
        from . import __author__, __version__
        print(f"CausalForge {__version__}  (author: {__author__})")
        return 0

    if args.cmd == "benchmark":
        set_all(args.seed if args.seed is not None else cfg.seed)
        rows = pipe.benchmark(seed=args.seed)
        probe = pipe.probe_backends(seed=args.seed if args.seed is not None else cfg.seed)
        path = pipe.to_json(rows, args.out, extra=probe)
        print_table(rows)
        print(f"\nbackend probe: {probe}")
        print(f"\nbenchmark written -> {path}")
        return 0

    if args.cmd == "run":
        set_all(args.seed)
        data = generate_dataset(seed=args.seed, n_samples=args.n_samples)
        res = pipe.run(data)
        print(f"dataset: synthetic  true_ate={data.true_ate}")
        for k, v in res.items():
            ate = "-" if v.ate is None or v.ate != v.ate else f"{v.ate:.4f}"
            print(f"  {k:<16} ate={ate:<10} backend={v.backend:<9} "
                  f"available={v.available!s:<5} {v.note}")
        if args.out:
            pipe.to_json_single(res, args.out)
            print(f"\nresult written -> {args.out}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
