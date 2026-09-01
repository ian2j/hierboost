"""Cheap, model-independent momentum diagnostic: used only to decide the sign/
magnitude of the momentum overlay in forecast.py, not as a forecast itself. Kept
separate from forecast_asset() deliberately -- it must be computable without
running the expensive Gibbs fit, so that an expanding-window sign estimate can be
built from many periods' worth of history cheaply.
"""
import numpy as np


def raw_momentum_zscore(target, period_end_date, prices, long_window=252, short_window=20):
    """Same quantity as forecast.py's per-factor momentum_std, but computed directly
    on the target asset's own daily returns instead of a factor block -- how far the
    recent short_window mean daily return sits from the long_window mean, in units
    of the long_window's own daily-return std. Returns None if there isn't enough
    history (mirrors forecast_asset's own MIN_HISTORY gate)."""
    hist = prices.loc[:period_end_date]
    if target not in hist.columns:
        return None
    target_hist = hist[target].dropna().tail(long_window)
    if target_hist.shape[0] < 60:
        return None
    rets = target_hist.pct_change().dropna()
    mu, sigma = rets.mean(), rets.std()
    if sigma == 0 or np.isnan(sigma):
        return None
    recent = rets.tail(short_window).mean()
    return float((recent - mu) / sigma)
