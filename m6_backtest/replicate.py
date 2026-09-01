"""Returns-based style analysis via the hierboost block-factor machinery: express
an external target's returns (e.g. a trend-following index like DBMF) as a sparse
combination of the 100 M6 assets, using the same correlation-block decomposition
and spike-and-slab selection as forecast.py's per-asset forecaster, then convert
the fitted block-level exposures back into per-asset decision weights.

This is Sharpe's returns-based style analysis with spike-and-slab doing the
constituent selection instead of a plain constrained regression -- an
investment-decision strategy distinct from (and independent of) the momentum/
reversal overlay in forecast.py. Unlike forecast.py's forecast_asset, there is no
momentum overlay and no posterior-predictive simulation: the goal here is a
single best-fit *replicating portfolio* per period, not a quintile-rank forecast,
so the RPS side of a submission is unaffected -- only the Decision weights change.
"""
import numpy as np
import pandas as pd

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian, gibbs_sampler_gaussian

MIN_HISTORY = 60


def zscore(df):
    return (df - df.mean()) / df.std()


def replicate_weights(target_prices, period_end_date, prices, long_window=252,
                       rho=0.9, kappa=100.0, xi1=4.0, rank=None, seed=0):
    """target_prices: Series of the external index's own price history (any date
    range; only data strictly before period_end_date is used here -- no
    lookahead). prices: the 100 M6 assets' price panel, same convention as
    forecast.py. Returns a raw (unnormalized) Series of per-asset weights, or
    None if there isn't enough overlapping history yet."""
    target_hist = target_prices.dropna().loc[:period_end_date].tail(long_window)
    if target_hist.shape[0] < MIN_HISTORY:
        return None

    window_start = target_hist.index[0]
    hist = prices.loc[:period_end_date]
    window = hist.loc[window_start:period_end_date]

    other_cols = list(prices.columns)
    coverage = window[other_cols].notna().mean()
    valid_others = coverage[coverage >= 0.9].index.tolist()
    if len(valid_others) < 5:
        return None

    combined = pd.concat([target_hist.rename("__target__"), window[valid_others]], axis=1).dropna()
    if combined.shape[0] < MIN_HISTORY:
        return None
    rets = combined.pct_change().dropna()
    y_raw = rets["__target__"].values
    Xp_ret = rets[valid_others]

    n = len(y_raw)
    mu_y, sigma_y = y_raw.mean(), y_raw.std()
    if sigma_y == 0 or np.isnan(sigma_y):
        return None
    y = (y_raw - mu_y) / sigma_y

    labels = blocks_from_correlation_threshold(Xp_ret.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = np.array(valid_others)

    factor_cols, loadings = [], []
    for b, idx in membership.items():
        members = cols[idx]
        Xb = zscore(Xp_ret[members]).values
        if len(members) == 1:
            factor_std = Xb[:, 0]
            w_load = np.array([1.0])
        else:
            factor_std, _ = gaussian_block_factor(Xb)
            w_load = Xb.std(axis=0)
            w_load = w_load / w_load.sum() if w_load.sum() > 0 else np.ones(len(members)) / len(members)
        factor_cols.append(factor_std)
        loadings.append((members, w_load))

    Z = np.column_stack(factor_cols)
    # No sector-relevance prior here, unlike forecast.py's per-asset forecaster:
    # DBMF is a multi-asset macro/trend strategy with no single natural GICS
    # sector, and classification.py's relevance table (built for equity/ETF
    # targets) would wrongly penalize the fixed-income and commodity blocks that
    # a trend-following index actually trades (it hard-codes those to a 0.05
    # "irrelevant" weight for ordinary equity targets). Uniform relevance lets
    # the spike-and-slab's own sparsity prior (xi0) do the selection instead.
    wr = np.ones(Z.shape[1])

    X_design = np.column_stack([np.ones(n), Z])
    K = Z.shape[1]
    xi0 = np.log(3 / max(K, 4)) - np.log(1 - 3 / max(K, 4))

    warm = fit_em_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0, rank=rank)
    gr = gibbs_sampler_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0,
                                 n_samples=1000, burn_in=400, rank=rank,
                                 beta_init=warm.beta, sigma_g2_init=warm.sigma_g2,
                                 sigma_y2_init=warm.sigma_y2, seed=seed)
    gamma_mean = gr.beta[:, 1:].mean(axis=0)  # (K,) posterior mean block exposure

    weights = {}
    for (members, w_load), g in zip(loadings, gamma_mean):
        for m, wl in zip(members, w_load):
            weights[m] = weights.get(m, 0.0) + g * wl

    return pd.Series(weights)
