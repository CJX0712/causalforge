"""End-to-end demo for CausalForge.

Generates a Monte-Carlo benchmark, writes ``benchmark.json`` next to this
script, and prints a summary. Two consecutive runs with the same seed produce
bit-for-bit identical JSON (determinism check baked into the delivery spec).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalforge.core.config import Config
from causalforge.core.seed import set_all
from causalforge.pipeline.pipeline import CausalPipeline

_TRUE_ATE = 1.0


def main() -> int:
    cfg = Config.from_env()
    set_all(cfg.seed)
    pipe = CausalPipeline(cfg)
    rows = pipe.benchmark(seed=cfg.seed, tau=_TRUE_ATE)
    probe = pipe.probe_backends(seed=cfg.seed, tau=_TRUE_ATE)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json")
    pipe.to_json(rows, out_path, extra=probe)

    print("CausalForge demo complete.")
    print(f"  benchmark.json -> {out_path}")
    print(f"  target true_ate = {_TRUE_ATE}")
    for r in rows:
        bias = "-" if r.bias is None else f"{r.bias:.4f}"
        cov = "-" if r.coverage is None else f"{r.coverage:.3f}"
        flag = "" if r.available else "  [SKIPPED: backend unavailable]"
        print(f"  {r.estimator:<16} mean_ate={r.mean_ate:+.4f}  "
              f"bias={bias:<8} cov@95={cov}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
