"""Temporal-block variant of forecast.py: prototype for the spatiotemporal
extension discussed with Ian (2026-08-28) -- not yet wired into
run_backtest.py's walk-forward.

Revision history within this prototype (kept here rather than hidden, since both
revisions were real, informative smoke-test findings, not just bug fixes):

v1 (first smoke test): fed each spatial block's raw daily-return member(s)
directly into hierboost.state_space.fit_temporal_block_factor, as if the block's
own daily observations were the "temporal member" columns. For the ~94% of
blocks that are singletons at the production correlation threshold (rho=0.9;
confirmed empirically, see project notes), this degenerates to fitting AR(1) to
one asset's own daily returns, which came back essentially flat (median
rho ~ -0.01) -- correctly reflecting near-zero daily-return autocorrelation for
liquid single names, but NOT the same ~20-trading-day-horizon phenomenon
momentum_signal.py's reversal diagnostic targets. Wrong frequency, not a bug.

v2 (this version): two clearly separated stages, matching the "decorrelate
space, then decorrelate time" framing discussed with Ian.
  Stage A (space, unchanged from forecast.py): each spatial block's member(s)
  are combined into ONE daily return series via the same std-weighted linear
  combination forecast.py already uses for its momentum diagnostic
  (`factor_raw`) -- trivial for a singleton block, a weighted blend for a
  multi-member one.
  Stage B (time, new): that one daily series is turned into 4 DISJOINT
  (non-overlapping) window-return features -- not the nested/cumulative
  5/10/20/60-day windows temporal_demo.py used (which all share the same most
  recent days by construction, and were already flagged in that demo's own
  notes as producing an artificially inflated apparent R^2 for that reason).
  Disjoint chunks mean any within-day correlation across the 4 features
  reflects genuine shared trend, not arithmetic overlap. fit_temporal_block_factor
  is then fit on these 4 columns, giving a real AR(1) trend state per block.

Two further scale issues, both caught by working through the arithmetic before
running rather than after (no smoke test needed to find these -- they were
findable from the model's own definitions):

1. The one-step (1-day) AR(1) prediction is the wrong horizon for a ~20-
   trading-day-ahead forecast: with rho this close to 1 (slow-moving smoothed
   states are highly autocorrelated almost by construction, not because of
   genuine long-range predictability), a 1-step prediction is trivially close
   to today's value and says almost nothing about the actual forecast horizon.
2. Even the h-step-ahead ENDPOINT value (rho^h * z_last) is the wrong quantity
   to multiply by gamma: gamma was fit against DAILY (y, z) pairs, so
   gamma*z_level predicts a single day's rate, not a period-cumulative return.
   The right quantity is gamma times the AVERAGE predicted daily rate over the
   whole period -- see h_step_ar1_average_forecast -- matching forecast.py's
   own "daily rate * period_len_days" convention exactly, rather than
   introducing a new, inconsistent scale convention.

v3 (this version): the v2 smoke test ran correctly but 47.5x slower per
asset-period fit than the snapshot version -- disqualifying at 100 assets x 58
periods (would run ~24+ hours). Root cause: at the production spatial
threshold, ~94% of blocks are singletons -- just one other asset's own
state -- and that state does not depend on which target is being forecast,
yet v2 refit it from scratch, via 200-iteration EM, inside EVERY target's own
regression that referenced it (~100x redundant EM fits per period for the
same underlying series). build_asset_temporal_cache fits each universe
asset's own disjoint-window AR(1) state ONCE per period; forecast_asset_temporal
takes that cache and looks singleton blocks up instead of refitting them, only
still fitting fresh for the rare (~6%) multi-member blocks. Cached and
freshly-fit z-series are stored as pandas Series indexed by date (not raw
numpy positions) specifically because a cached asset's own valid-date range
need not exactly match a given target's own dropna'd window -- date-based
intersection across all of a target's blocks (see the common_idx computation
below) handles that correctly regardless of minor per-target coverage
differences, where the earlier position-based t_idx alignment would not have.
"""
import numpy as np
import pandas as pd

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.state_space import fit_temporal_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian, gibbs_sampler_gaussian

from classification import relevance_for_pair

MIN_HISTORY = 60
MIN_EFFECTIVE_ROWS = 40  # floor on usable rows after the window-feature warmup is trimmed
# Disjoint (offset, width) chunks in trading days, spanning a 60-day total lookback
# with zero overlap between chunks -- see module docstring for why disjoint rather
# than nested/cumulative windows.
CHUNKS = ((0, 5), (5, 5), (10, 10), (20, 40))


def zscore(df):
    return (df - df.mean()) / df.std()


def zscore_cols(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def disjoint_window_features(x, chunks=CHUNKS, min_effective_rows=MIN_EFFECTIVE_ROWS):
    """x: (n,) raw daily-return series (one per day, already time-ordered).
    Returns (feat, t_idx) where feat is (n - total_lookback, len(chunks)) and
    t_idx gives the original row positions each feature row corresponds to
    (so the regression target can be trimmed to the identical rows), or
    (None, None) if there isn't enough history left after the warmup. Uses a
    cumulative-sum trick for O(1) window sums instead of a per-day loop.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    total_lookback = max(o + w for o, w in chunks)
    if n - total_lookback < min_effective_rows:
        return None, None
    csum = np.concatenate([[0.0], np.cumsum(x)])
    t_idx = np.arange(total_lookback, n)
    cols = []
    for offset, width in chunks:
        hi = t_idx - offset               # inclusive end position
        lo = t_idx - offset - width + 1   # inclusive start position
        cols.append(csum[hi + 1] - csum[lo])
    return np.column_stack(cols), t_idx


def h_step_ar1_average_forecast(rho, z_last, p_last, state_var, h):
    """Mean and variance of the AVERAGE of z_{t+1},...,z_{t+h} given the filtered
    state at t, under z_{t+k} = rho*z_{t+k-1} + eta_k, eta_k ~ N(0, state_var) iid.

    This -- not the single h-step-ahead endpoint value -- is the right quantity
    to combine with gamma: gamma was fit against DAILY (y, z) pairs, so
    gamma*z_level predicts a DAILY rate, and the period's cumulative effect
    needs the AVERAGE daily rate over the period (matching forecast.py's own
    "daily rate * period_len_days" convention), not gamma times a single future
    day's state value. Variance computed by direct O(h^2) summation over the
    h x h covariance matrix (h ~ 20, trivial cost) rather than a hand-derived
    closed form, to avoid an off-by-one/sign error in the averaging formula.
    Sane check: each Var(z_{t+k}) individually reverts toward the model's own
    pinned stationary variance (1) as k grows, so the averaged uncertainty
    can't silently understate long-horizon uncertainty either.
    """
    ks = np.arange(1, h + 1)
    rho2k = rho ** (2 * ks)
    means = (rho ** ks) * z_last
    avg_mean = means.mean()

    if abs(rho) < 1 - 1e-9:
        var_z = rho2k * p_last + state_var * (1 - rho2k) / (1 - rho ** 2)
    else:
        var_z = rho2k * p_last + state_var * ks  # rho ~ 1 edge case, avoid 0/0

    lag = np.abs(ks[:, None] - ks[None, :])
    var_min = np.minimum(var_z[:, None], var_z[None, :])
    cov = (rho ** lag) * var_min  # Cov(z_{t+i}, z_{t+j}) = rho^|i-j| * Var(z_{t+min(i,j)})
    avg_var = cov.mean()
    return avg_mean, avg_var


def _fit_temporal_state(daily_series, chunks, em_iter, period_len_days):
    """daily_series: pd.Series of raw daily returns (DatetimeIndex). Returns
    (z: pd.Series, rho, next_pred, next_var) or None if there isn't enough
    history for the disjoint-window warmup. Shared helper between the cache
    builder (per-asset) and forecast_asset_temporal's fresh-fit fallback path
    (per multi-member block) so both go through identical logic.
    """
    feat, t_idx = disjoint_window_features(daily_series.values, chunks)
    if feat is None:
        return None
    feat_z = zscore_cols(feat)
    result = fit_temporal_block_factor(feat_z, n_iter=em_iter)
    z_series = pd.Series(result.z, index=daily_series.index[t_idx])
    avg_mean, avg_var = h_step_ar1_average_forecast(
        result.rho, result.z[-1], result.z_var[-1], result.state_var, period_len_days)
    return z_series, result.rho, avg_mean, avg_var


def build_asset_temporal_cache(symbols, prices, period_end_date, period_len_days,
                                long_window=252, chunks=CHUNKS, em_iter=200):
    """Precompute every universe asset's own disjoint-window AR(1) trend state
    ONCE per period -- see v3 note in the module docstring for why. Returns
    dict(symbol -> (z_series, rho, next_pred, next_var) or None). Independent
    of any specific target asset: uses each symbol's own price history only.
    """
    hist = prices.loc[:period_end_date]
    cache = {}
    for sym in symbols:
        if sym not in hist.columns:
            cache[sym] = None
            continue
        px = hist[sym].dropna().tail(long_window)
        if px.shape[0] < MIN_HISTORY:
            cache[sym] = None
            continue
        daily = px.pct_change().dropna()
        cache[sym] = _fit_temporal_state(daily, chunks, em_iter, period_len_days)
    return cache


def forecast_asset_temporal(target, period_end_date, prices, sectors, period_len_days=20,
                             long_window=252, short_window=20, rho=0.9, kappa=100.0, xi1=4.0,
                             n_samples=1000, burn_in=400, rank=None, seed=0,
                             include_residual_noise=True, em_iter=200, asset_cache=None):
    """Same contract as forecast.forecast_asset (dict(draws=..., mean=..., n_blocks=...,
    n_obs=...) or None). `rho` here is the correlation-threshold for SPATIAL block
    formation (forecast.py's existing parameter name) -- distinct from each block's
    own fitted AR(1) temporal rho reported in rho_per_block below. `asset_cache`,
    if given (from build_asset_temporal_cache, same period_end_date/period_len_days),
    is used for singleton blocks instead of refitting; multi-member blocks (rare)
    are always fit fresh since their membership can vary slightly by target.
    """
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
    y_series = pd.Series((rets["__target__"].values - rets["__target__"].values.mean())
                          / rets["__target__"].values.std(), index=rets.index)
    mu_y, sigma_y = rets["__target__"].values.mean(), rets["__target__"].values.std()
    if sigma_y == 0 or np.isnan(sigma_y):
        return None
    Xp_ret = rets[valid_others]

    labels = blocks_from_correlation_threshold(Xp_ret.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = np.array(valid_others)
    target_sector = sectors[target]

    block_z, wr_list = {}, []
    rho_per_block, next_pred_per_block, next_var_per_block = [], [], []
    for b, idx in membership.items():
        members = cols[idx]

        cached = asset_cache.get(members[0]) if (asset_cache is not None and len(members) == 1) else None
        if cached is not None:
            z_s, rho_b, next_mean, next_var = cached
        else:
            Xb_raw = Xp_ret[members].values
            if len(members) == 1:
                block_daily = pd.Series(Xb_raw[:, 0], index=Xp_ret.index)
            else:
                Xb = zscore(Xp_ret[members]).values
                w_load = Xb.std(axis=0)
                w_load = w_load / w_load.sum() if w_load.sum() > 0 else np.ones(len(members)) / len(members)
                block_daily = pd.Series(Xb_raw @ w_load, index=Xp_ret.index)
            fit = _fit_temporal_state(block_daily, CHUNKS, em_iter, period_len_days)
            if fit is None:
                return None  # not enough history for the window-feature warmup
            z_s, rho_b, next_mean, next_var = fit

        block_z[b] = z_s
        rho_per_block.append(rho_b)
        next_pred_per_block.append(next_mean)
        next_var_per_block.append(next_var)

        rel = np.mean([relevance_for_pair(target_sector, sectors.get(m, "Broad Market")) for m in members])
        wr_list.append(rel)

    # Date-based intersection, not positional: a cached asset's own valid range
    # need not exactly match this target's dropna'd window (see v3 docstring note).
    common_idx = rets.index
    for z_s in block_z.values():
        common_idx = common_idx.intersection(z_s.index)
    if len(common_idx) < MIN_EFFECTIVE_ROWS:
        return None

    Z = np.column_stack([block_z[b].loc[common_idx].values for b in membership.keys()])
    y_eff = y_series.loc[common_idx].values
    n_eff = len(common_idx)
    wr = np.array(wr_list)
    wr = wr / wr.max() if wr.max() > 0 else wr
    next_pred = np.array(next_pred_per_block)     # (K,) -- h-step-ahead block forecast
    next_sd = np.sqrt(np.array(next_var_per_block))  # (K,) -- h-step-ahead predictive SD

    X_design = np.column_stack([np.ones(n_eff), Z])
    K = Z.shape[1]
    xi0 = np.log(3 / max(K, 4)) - np.log(1 - 3 / max(K, 4))

    warm = fit_em_gaussian(X_design, y_eff, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0, rank=rank)
    gr = gibbs_sampler_gaussian(X_design, y_eff, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0,
                                 n_samples=n_samples, burn_in=burn_in, rank=rank,
                                 beta_init=warm.beta, sigma_g2_init=warm.sigma_g2,
                                 sigma_y2_init=warm.sigma_y2, seed=seed)

    gamma_draws = gr.beta[:, 1:]
    beta0_draws = gr.beta[:, 0]
    sigma_y2_draws = gr.sigma_y2

    rng = np.random.default_rng(seed)
    # next_pred/next_sd are the AVERAGE predicted daily block-state level (and its
    # SD) over the upcoming period -- same role as forecast.py's momentum_std/
    # momentum_se, so the same "daily rate * period_len_days" scaling applies here.
    next_noisy = next_pred[None, :] + rng.standard_normal((n_samples, len(next_pred))) * next_sd[None, :]
    daily_std_pred = beta0_draws + (gamma_draws * next_noisy).sum(axis=1)
    period_std_mean = daily_std_pred * period_len_days

    if include_residual_noise:
        residual_period_std = rng.standard_normal(n_samples) * np.sqrt(sigma_y2_draws * period_len_days)
    else:
        residual_period_std = 0.0

    period_std_pred = period_std_mean + residual_period_std
    period_pred = mu_y * period_len_days + sigma_y * period_std_pred

    return dict(draws=period_pred, mean=float(period_pred.mean()), n_blocks=K, n_obs=n_eff,
                rho_per_block=rho_per_block, next_pred_per_block=next_pred_per_block,
                next_var_per_block=next_var_per_block)
