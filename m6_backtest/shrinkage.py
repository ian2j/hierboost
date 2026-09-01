"""Adaptive, causally-valid shrinkage of the RPS quintile forecast toward the
naive uniform 20%-per-bucket baseline. Same expanding-window discipline as
momentum_signal.py's adaptive sign: period p's blend weight depends only on
strictly earlier periods' own already-realized RPS-vs-blend curves, blended with
a pre-registered prior -- never on period p's own outcome.
"""
import numpy as np
import pandas as pd

from scoring import compute_rps, RANK_COLS

UNIFORM = pd.Series([0.2] * 5, index=RANK_COLS)
BLEND_GRID = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100% toward uniform

# Pre-registered prior, before any M6-specific evidence: models are commonly
# overconfident by default (this project's own earlier finding, the
# momentum_uncertainty_scale fix, was exactly this lesson) -- a modest shrink is
# a reasonable starting point, not tuned to this dataset's realized RPS.
PRIOR_BLEND = 0.25
PRIOR_N = 3


def rps_curve(forecast_df, period_returns):
    """RPS at each blend level in BLEND_GRID, for one already-realized period."""
    out = []
    for b in BLEND_GRID:
        blended = forecast_df * (1 - b) + UNIFORM.values * b
        rps, _ = compute_rps(period_returns, blended)
        out.append(rps)
    return np.array(out)


def best_blend(curve):
    return float(BLEND_GRID[np.argmin(curve)])


def adaptive_blend_weight(prior_best_blends):
    """prior_best_blends: list of per-period best-blend values, one per strictly
    earlier period (each computed from that period's own already-known outcome).
    Shrinkage blend of the pre-registered prior and their expanding-window mean."""
    n = len(prior_best_blends)
    mean_realized = float(np.mean(prior_best_blends)) if n > 0 else PRIOR_BLEND
    return (PRIOR_N * PRIOR_BLEND + n * mean_realized) / (PRIOR_N + n)


def apply_blend(forecast_df, blend):
    return forecast_df * (1 - blend) + UNIFORM.values * blend
