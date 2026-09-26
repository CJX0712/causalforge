# CausalForge — Model Card

**Model name:** CausalForge (cross-fitted DML + baseline/PSM estimators)
**Version:** 0.1.0 · **Author:** 晨星 · **Domain:** Causal Inference

---

## 1. Model details

- **Task:** Estimate the Average Treatment Effect (ATE) `E[Y(1) − Y(0)]` from
  observational data with confounding.
- **Headline estimator:** Cross-fitted Double Machine Learning (R-learner /
  partially-linear DML, Chernozhukov et al., 2018) with gradient-boosted
  nuisance models (LightGBM preferred, XGBoost alternative, sklearn
  `HistGradientBoosting` fallback).
- **Baselines / fallbacks:** naive difference, OLS adjustment, propensity-score
  matching (caliper k-NN, with replacement).
- **Optional ensemble backend:** EconML `CausalForestDML` (auto-skipped if
  EconML absent).
- **Inference:** closed-form orthogonal moment estimator; 95% CI from the DML
  influence-function variance.

---

## 2. Intended use

- **Intended:** Benchmarking causal-effect estimators; teaching the bias of
  naive/linear adjustment under non-linear confounding; producing reproducible
  ATE estimates on tabular observational data with a known/common support.
- **Out of scope:** time-series / panel causal effects, network/spatial
  effects, instrumental-variable / regression-discontinuity designs, causal
  discovery, and any use on data without a plausible unconfoundedness +
  positivity assumption.

---

## 3. Training / fitting data

- The package ships a **synthetic generator with a known ground-truth ATE**
  (see `docs/architecture.md` §3) used for the Monte-Carlo benchmark.
- An optional IHDP loader exists (`data/loaders.py`,
  `available_ihdp()`); it is transparently skipped when the dataset is absent.

---

## 4. Evaluation (Monte-Carlo benchmark, n_reps=10, n_samples=2000, seed=42)

True ATE = 1.0. Lower bias is better; coverage should be ≈ 0.95.

| Estimator | Backend | mean ATE | bias | RMSE | cov@95 |
|-----------|---------|---------:|-----:|-----:|-------:|
| naive | baseline | 1.8875 | 0.8875 | — | 0.000 |
| ols | tier-1 | 1.7218 | 0.7218 | — | 0.000 |
| psm | tier-1 | 1.6640 | 0.6640 | — | 0.000 |
| **gdml** | lightgbm | **0.9551** | **0.0449** | — | **1.000** |
| gdml_nocv | lightgbm | 1.0178 | 0.0178 | — | 1.000 |

**Headline result:** cross-fitted DML reduces absolute bias by **94.9%**
relative to the naive baseline (0.8875 → 0.0449) and achieves nominal 95%
coverage, whereas every classical adjuster has **0% coverage** (its CI never
contains the truth because the bias exceeds the CI half-width).

Backend probe: `lightgbm=True, xgboost=True, econml=True`; single
`causal_forest` run ATE = 1.0806 (SE 0.0607), within its own 95% CI of the
truth.

---

## 5. Limitations & caveats

- **Assumptions required:** unconfoundedness (no unmeasured confounders) and
  positivity (common support). The synthetic generator clips the propensity to
  `[0.10, 0.90]` to guarantee positivity; real data must be checked.
- **Non-linear confounding is the hard case** demonstrated here. On *linear*
  confounding, OLS/PSM would also be near-unbiased — the benchmark is
  deliberately adversarial.
- **`causal_forest` (EconML)** is noisier on this hard problem and is excluded
  from the 10-rep MC loop (budget + stability); it is validated by a single
  probe run, not a coverage estimate.
- **Single-treatment, binary/continuous** setting only.

---

## 6. Ethical considerations

Causal estimates inform high-stakes decisions (policy, medicine, credit).
CausalForge ships with no real data and no deployed model; users must validate
unconfoundedness/positivity on their own data, audit for leakage, and avoid
using estimates where the required assumptions are implausible. The synthetic
benchmark is for method comparison, not for claiming causal validity on real
outcomes.
