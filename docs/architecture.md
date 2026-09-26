# CausalForge — Architecture

> Author: 晨星 · Version: 0.1.0 · Domain: Causal Inference (treatment-effect estimation)

CausalForge estimates the **Average Treatment Effect (ATE)** from observational
data under confounding, and proves its quality with a Monte-Carlo benchmark
against strong baselines. The headline method is **cross-fitted Double Machine
Learning (DML / R-Learner, Chernozhukov et al., 2018)** with gradient-boosted
nuisance models.

---

## 1. Layered skeleton (one-way, acyclic)

```
cli.py                 # entry point (argparse)
  │
  ▼
pipeline/pipeline.py   # CausalPipeline: orchestrates estimators + MC benchmark
  │
  ├──► data/           # synthetic.py (ground-truth ATE) · loaders.py (IHDP, optional)
  ├──► causal/         # estimators: naive · ols · psm · gdml · forest
  ├──► eval/           # metrics.py (aggregate bias/RMSE/coverage)
  │
  ▼
core/                  # seed · config · errors · types · interfaces (infra)
```

**Rule:** `cli → pipeline → {data, causal, eval} → core`. No backward edges, no
cycles. `core` depends on nothing inside the package; estimators depend only on
`core` + `data` + `eval`; the pipeline composes them.

---

## 2. Estimator tiers

| Tier | Estimator | Backend | Role | Always available? |
|------|-----------|--------|------|------------------|
| baseline | `NaiveDifference` | pure numpy | unadjusted Y\|T=1 − Y\|T=0 | ✅ |
| tier-1 | `OLSAdjustment` | closed-form OLS | linear covariate adjustment | ✅ |
| tier-1 | `PropensityScoreMatching` | sklearn LR + caliper k-NN | PSM offline fallback | ✅ |
| tier-0 (SOTA) | `GradientDML` | LightGBM / XGBoost / sklearn | cross-fitted DML | ✅ (sklearn fallback) |
| tier-0 (SOTA) | `CausalForestEconML` | EconML `CausalForestDML` | forest DML | ⚠️ optional (auto-skip) |

- **Offline fallback (tier-1):** PSM is built on `scikit-learn`
  `LogisticRegression` + a hand-written caliper nearest-neighbour matcher
  (with replacement, 1:k). It needs no heavy dependency and is the
  always-available mid-tier baseline.
- **SOTA backend probe:** each tier-0 backend is gated behind an
  `available_*()` function. Missing backends (e.g. `econml` absent) make the
  estimator report `available()=False` and the benchmark **skips it
  transparently** (flagged in the report) instead of failing.

---

## 3. Data generating process (synthetic benchmark)

A *non-linear, high-dimensional* confounded problem — the regime where classical
adjustment breaks:

```
score(X) = κ·(X0² + X1² + X2² + X3·X4 + 0.5·sin(X5))   # non-linear 7-dim confounder
e(X)    = clip(sigmoid(score), 0.10, 0.90)             # propensity (positivity clip)
T       ~ Bernoulli(e(X))
g0(X)   = 0.5·(X0²+X1²+X2²) + 0.5·(X3·X4) + 0.3·sin(X5) + 0.4·(X0+X1)
Y       = g0(X) + τ·T + N(0, σ²)
```

Because the propensity is a **non-linear** function of many covariates, the
linear/logistic propensity model used by PSM is misspecified → PSM stays biased;
OLS cannot remove the non-linear confounding either. Cross-fitted DML with
gradient boosting models the full non-linear `g0` and `m` and recovers `τ`.
`τ` is returned as `true_ate` so the benchmark measures bias and CI coverage
exactly.

Defaults: `n_samples=2000, d=8, τ=1.0, κ=1.2, σ=0.5`.

---

## 4. Cross-fitted DML (tier-0)

```
Y = θ·T + g(X) + ε
T = m(X) + η

for each fold:
    fit g, m on the OTHER folds
    residuals:  Y_res = Y − g(X),  T_res = T − m(X)   (out-of-fold)
θ = Σ(T_res·Y_res) / Σ(T_res²)            # closed-form orthogonal estimator
SE(θ) = sqrt( mean( (Y_res − θ·T_res)² · T_res² ) / mean(T_res²)² / n )
```

Cross-fitting (sample-splitting) removes the regularization bias of the
nuisance fits and yields valid inference. The ablation variant
(`gdml_nocv`) fits nuisances on the full sample to demonstrate the bias DML is
designed to avoid.

---

## 5. Determinism

- Single entry point `core.seed.set_all(seed)` seeds `PYTHONHASHSEED`, `random`,
  and `numpy` (torch best-effort).
- The Monte-Carlo loop advances `seed = base + rep` so every replication is
  independent and reproducible.
- Two consecutive runs with the same seed produce **bit-for-bit identical**
  `benchmark.json` (covered by `test_determinism_bit_for_bit`).

---

## 6. Configuration

`core/config.Config` is read once and validated; every field has a default and
can be overridden via `CAUSALFORGE_*` environment variables
(`CAUSALFORGE_N_SAMPLES`, `CAUSALFORGE_N_REPS`, `CAUSALFORGE_SEED`,
`CAUSALFORGE_BACKEND_PREF`, `CAUSALFORGE_MATCH_RATIO`, …). Twelve-factor and
CI-friendly.

---

## 7. Validation harness

`CausalPipeline.benchmark()` repeats estimation over `n_reps` fresh datasets
with known ATE and aggregates, per estimator:

- `mean_ate`, absolute `bias`, `rmse`
- `coverage` — fraction of 95% CIs containing the true ATE
- `mean_ci_width`

`NaN`/error results are excluded so one failure never poisons the aggregate.
`NaN`/`inf` are serialised to `null` (never `NaN`) in JSON.

---

## 8. Performance budget

The full demo (5-estimator MC benchmark **excluding** the heavy ensemble forest
+ a single forest probe) runs in **< 60s** on the locked environment
(measured 12s). The heavy `causal_forest` estimator is excluded from the 10-rep
MC loop to protect this budget and is instead exercised once via
`probe_backends()`.
