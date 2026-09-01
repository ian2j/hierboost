"""Per-asset, per-period forecast: the core unit of the walk-forward backtest.

For one target asset, using only price history strictly before `period_end_date`:
  1. Decorrelate the *other* available assets into blocks (excluding the target,
     to avoid any self-referential leakage) via the same correlation-threshold +
     probabilistic-PCA factor extraction used in finance_cross_sectional.py.
  2. Fit the Gaussian spike-and-slab with a sector-relevance prior (the target's
     own sector vs each block's member sectors), then run the Gibbs sampler to get
     posterior draws of the factor exposures (gamma).
  3. Combine each exposure draw with a simple momentum estimate of that factor's
     own recent return (short trailing window vs the long fitting window's own
     mean) to get one simulated next-period return per draw.

Momentum is a deliberately simple, standard, non-hierboost overlay -- estimating a
factor's own future return is a different problem than deciding which factors
matter for a given asset, and isn't what this framework is built to solve. The
posterior draws are what carry genuine, asset-specific uncertainty forward.
"""
import numpy as np
import pandas as pd

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian, gibbs_sampler_gaussian

from classification import relevance_for_pair

MIN_HISTORY = 60


def zscore(df):
    return (df - df.mean()) / df.std()


def forecast_asset(target, period_end_date, prices, sectors, period_len_days=20,
                    long_window=252, short_window=20, rho=0.9, kappa=100.0, xi1=4.0,
                    n_samples=1000, burn_in=400, rank=None, seed=0, momentum_sign=1.0,
                    momentum_uncertainty_scale=1.0, include_residual_noise=True):
    """Returns dict(draws=array of simulated period returns, mean=posterior mean
    period return, n_blocks=int) or None if there isn't enough history to fit."""
    hist = prices.loc[:period_end_date]
    if target not in hist.columns:
        return None
    target_hist = hist[target].dropna().tail(long_window)
    if target_hist.shape[0] < MIN_HISTORY:
        return None

    window_start = target_hist.index[0]
    window = hist.loc[window_start:period_end_date]

    other_cols = [c for c in prices.columns if c != target]
    coverage = window[other_cols].notna().mean()
    valid_others = coverage[coverage >= 0.9].index.tolist()
    if len(valid_others) < 5:
        return None

    combined = pd.concat([window[target].rename("__target__"), window[valid_others]], axis=1).dropna()
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
    target_sector = sectors[target]

    factor_cols, wr_list, momentum_std_list, momentum_se_list = [], [], [], []
    for b, idx in membership.items():
        members = cols[idx]
        Xb_raw = Xp_ret[members].values
        Xb = zscore(Xp_ret[members]).values
        if len(members) == 1:
            factor_raw = Xb_raw[:, 0]
            factor_std = Xb[:, 0]
        else:
            factor_std, _ = gaussian_block_factor(Xb)
            w_load = Xb.std(axis=0)
            w_load = w_load / w_load.sum() if w_load.sum() > 0 else np.ones(len(members)) / len(members)
            factor_raw = Xb_raw @ w_load
        factor_cols.append(factor_std)

        rel = np.mean([relevance_for_pair(target_sector, sectors.get(m, "Broad Market")) for m in members])
        wr_list.append(rel)

        mu_f, sigma_f = factor_raw.mean(), factor_raw.std()
        recent_window = factor_raw[-short_window:]
        recent = recent_window.mean()
        momentum_std_list.append((recent - mu_f) / sigma_f if sigma_f > 0 else 0.0)
        # Standard error of that recent-window mean, in the same standardized units --
        # this is what was missing before: a fixed momentum point estimate gets treated
        # as certain regardless of how noisy the window it came from actually was. A
        # factor that's been choppy recently (high short-window realized vol relative
        # to its long-run vol) gets a wider SE here, which below turns into genuinely
        # wider -- not just smaller -- predictive uncertainty, instead of a fixed
        # dampening factor applied uniformly regardless of actual recent noise.
        std_short = recent_window.std()
        se_std = (std_short / np.sqrt(short_window)) / sigma_f if sigma_f > 0 else 0.0
        momentum_se_list.append(se_std)

    Z = np.column_stack(factor_cols)
    wr = np.array(wr_list)
    wr = wr / wr.max() if wr.max() > 0 else wr
    momentum_std = momentum_sign * np.array(momentum_std_list)
    momentum_se = momentum_uncertainty_scale * np.array(momentum_se_list)

    X_design = np.column_stack([np.ones(n), Z])
    K = Z.shape[1]
    xi0 = np.log(3 / max(K, 4)) - np.log(1 - 3 / max(K, 4))

    warm = fit_em_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0, rank=rank)
    gr = gibbs_sampler_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0,
                                 n_samples=n_samples, burn_in=burn_in, rank=rank,
                                 beta_init=warm.beta, sigma_g2_init=warm.sigma_g2,
                                 sigma_y2_init=warm.sigma_y2, seed=seed)

    gamma_draws = gr.beta[:, 1:]                      # (n_samples, K)
    beta0_draws = gr.beta[:, 0]                        # (n_samples,)
    sigma_y2_draws = gr.sigma_y2                        # (n_samples,) -- residual/idiosyncratic variance

    rng = np.random.default_rng(seed)
    momentum_noisy = momentum_std[None, :] + rng.standard_normal((n_samples, len(momentum_std))) * momentum_se[None, :]
    daily_std_pred = beta0_draws + (gamma_draws * momentum_noisy).sum(axis=1)   # (n_samples,), persistent daily rate
    period_std_mean = daily_std_pred * period_len_days   # persistent drift -> scales linearly over the period

    # Residual noise (gr.sigma_y2, sampled by the Gibbs sampler but previously
    # never used downstream here) is a *new*, independent shock each trading day
    # -- unlike the persistent drift/momentum terms above, it must aggregate as
    # sqrt(n_days), not n_days, when converting daily to period scale. Omitting
    # it entirely (as before) means the predictive draws only reflected parameter
    # and momentum-estimate uncertainty, never "even with the right model, a
    # day's return has unexplained noise on top" -- almost certainly the largest
    # missing source of predictive spread, and a likely reason RPS never beat
    # the naive benchmark: quintile probabilities were systematically too
    # concentrated for what a proper scoring rule like RPS rewards.
    if include_residual_noise:
        residual_period_std = rng.standard_normal(n_samples) * np.sqrt(sigma_y2_draws * period_len_days)
    else:
        residual_period_std = 0.0

    period_std_pred = period_std_mean + residual_period_std
    period_pred = mu_y * period_len_days + sigma_y * period_std_pred

    return dict(draws=period_pred, mean=float(period_pred.mean()), n_blocks=K, n_obs=n)
