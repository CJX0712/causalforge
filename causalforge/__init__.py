"""CausalForge — world-class causal effect estimation.

CausalForge estimates conditional / average treatment effects (ATE) with a
reusable pipeline that compares a strong baseline (naive difference, OLS
adjustment, propensity-score matching) against a state-of-the-art backend
(cross-fitted Double Machine Learning / R-learner built on LightGBM /
XGBoost / scikit-learn gradient boosting). All randomness flows through a
single seed entry point so results are bit-for-bit reproducible.

Author: 晨星
"""

__version__ = "0.1.0"
__author__ = "晨星"
