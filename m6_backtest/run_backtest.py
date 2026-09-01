"""Full walk-forward backtest: 100 M6 assets x 12 monthly periods.

Period boundaries: the 12 real M6 evaluation windows, March 7 2022 - February 2
2023 (see CORRECT_PERIOD_STARTS below). An earlier version of this file guessed
"last trading day of each calendar month starting Jan 31, 2022" -- that guess was
wrong by 5+ weeks: the M6 companion paper (arXiv:2310.13357, Table 1) gives 12
submission deadlines (Sundays, 18:00 GMT) 28 days apart starting March 6, 2022;
each investment period runs from the following Monday for ~20 trading days. This
was confirmed empirically, not just read off the paper: team T's actual (real,
historical) submitted weights for "1st Submission" / "6th Submission" / "12th
Submission" in data/submissions.csv were scored against every plausible trading
window in data/summary_leaderboard.xlsx's official Prices sheet, and the window
[2022-03-07, 2022-04-01] reproduces the official Global-leaderboard M1 IR
(3.04816) to 6 decimal places (and M6's IR to 7) -- not a coincidence. The
Jan31-Mar6 span in assets_m6.csv is pre-competition trial/lookback data, not a
scored period (data/summary_leaderboard.xlsx's Submissions sheet literally labels
some early rows "Trial run").

For each period, for each asset: fit using only data strictly before the period
start (forecast.forecast_asset), or fall back to a neutral (zero-conviction, equal
sector-blind) forecast if there isn't enough history (DRE before its post-merger
exit is excluded entirely from period ~10 onward; DRE and AVB have no pre-2022
yfinance lookback at all -- see download_prices.py). Posterior draws are combined
across all 100 assets, independently per asset (each asset's own Gibbs chain), to
simulate the cross-sectional quintile ranking and to build the long-short weights.
"""
import os
# Must happen before numpy (or anything that imports numpy) is loaded: BLAS reads
# these once at library-load time. Without this, every one of the 100 per-asset
# Gibbs fits below independently fans out across all available cores via
# NumPy/BLAS's own internal threading -- that's *already* what makes a plain
# serial Python loop over assets show up as ~19 cores busy in `ps`, despite there
# being no explicit parallelism in this file. Adding process-level parallelism
# (below) on top of that unchanged would oversubscribe the machine (N worker
# processes x ~19 BLAS threads each, all fighting over the same cores) and make
# things slower, not faster. One thread per process, N processes instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# hierboost's factor-block fit (gaussian_block_factor -> latent.py) uses JAX, which
# has its own separate CPU thread pool (not controlled by the vars above) and its
# own JIT compilation step the first time each jax.jit-wrapped function is called
# in a given process. Cap its threads too, and force CPU (no point probing for a
# GPU that isn't there).
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The env vars above are necessary but turned out NOT sufficient, verified
# empirically: even with all of them set, a single forecast_asset() call still
# pegged ~13-18 cores at once (JAX's own thread pool doesn't fully respect
# XLA_FLAGS in practice on this build). The only fix that actually held up under
# testing is an OS-level hard cap: pin this process's CPU affinity so the kernel
# scheduler physically cannot run it on more than a few cores, regardless of what
# any library does internally. This measurably helped, not hurt -- capped to 2
# cores, a single asset fit ran *faster* (~0.65s vs ~0.7-1.5s uncapped), because
# removing the oversubscription also removes the thread-contention overhead that
# came with it. Capped to 4 here (not 2) to leave real headroom for whatever else
# is running.
try:
    os.sched_setaffinity(0, set(range(min(4, os.cpu_count() or 4))))
except AttributeError:
    pass  # not on Linux -- no equivalent hard guarantee available; env vars above are best-effort only

import json
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

from classification import classify
from forecast import forecast_asset
from scoring import compute_rps, compute_ir, RANK_COLS
from momentum_signal import raw_momentum_zscore
from shrinkage import rps_curve, best_blend, adaptive_blend_weight, apply_blend

DRE_LAST_DATE = pd.Timestamp("2022-11-28")
N_SIM = 1000

# Pre-registered prior for the adaptive momentum sign (see run_backtest_adaptive
# design note below) -- a mild reversal tilt, consistent with the magnitude of
# short-horizon (weekly/monthly) reversal effects documented in Jegadeesh (1990)
# and Lehmann (1990), decades before M6 (2022). Chosen before looking at any M6
# result; NOT tuned against this dataset's outcome.
ADAPTIVE_PRIOR_CORR = -0.10
ADAPTIVE_PRIOR_N = 3


def load_official_prices():
    df = pd.read_csv("data/assets_m6.csv")
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="symbol", values="price")
    return wide


# The Monday after each of the 12 official submission deadlines (arXiv:2310.13357
# Table 1: Sundays 2022-03-06 .. 2023-01-08, 28 days apart), plus the confirmed
# final boundary. Verified against team T's real submissions -- see module
# docstring. Snapped to the nearest actual trading date on/after each target below.
CORRECT_PERIOD_STARTS = [
    "2022-03-07", "2022-04-04", "2022-05-02", "2022-05-30", "2022-06-27",
    "2022-07-25", "2022-08-22", "2022-09-19", "2022-10-17", "2022-11-14",
    "2022-12-12", "2023-01-09", "2023-02-02",
]


def period_boundaries(official_wide):
    """Snap each target date to the nearest actual trading date on/after it that
    has real data for (almost) every asset -- assets_m6.csv includes calendar
    dates for closed-market US holidays (e.g. 2022-05-30, Memorial Day) as
    near-all-NaN rows (only the LSE-listed assets trade), which would otherwise
    get silently picked as a boundary and corrupt every asset's return for that
    period. 0.9 coverage matches the threshold already used in forecast.py."""
    idx = official_wide.index.sort_values()
    coverage = official_wide.notna().mean(axis=1)
    valid_idx = idx[coverage.loc[idx] >= 0.9]
    bounds = []
    for d in CORRECT_PERIOD_STARTS:
        target = pd.Timestamp(d)
        candidates = valid_idx[valid_idx >= target]
        bounds.append(candidates[0] if len(candidates) else valid_idx[-1])
    return bounds  # 13 dates -> 12 periods


def neutral_forecast(n_sim):
    return dict(draws=np.zeros(n_sim), mean=0.0, n_blocks=0, n_obs=0)


# Each of the 100 per-asset fits within a period is fully independent (own data
# slice, own Gibbs chain) -- embarrassingly parallel. `prices_yf`/`sectors` are
# loaded once per worker via the Pool initializer (not re-sent per asset): with
# the default 'fork' start method the worker inherits them via copy-on-write from
# the parent's memory at pool-creation time, so this is a one-time, not per-task,
# cost.
_worker_state = {}


def _init_worker(prices_yf, sectors):
    _worker_state["prices_yf"] = prices_yf
    _worker_state["sectors"] = sectors


def _fit_one_asset(args):
    sym, start, period_len, seed, momentum_sign = args
    if sym == "DRE" and start >= DRE_LAST_DATE:
        return sym, neutral_forecast(N_SIM)
    res = forecast_asset(sym, start, _worker_state["prices_yf"], _worker_state["sectors"],
                          period_len_days=max(period_len, 1), n_samples=N_SIM, burn_in=400,
                          seed=seed, momentum_sign=momentum_sign)
    return sym, res if res is not None else neutral_forecast(N_SIM)


def period_momentum_zscores(symbols, prices, period_start):
    """Cheap, per-asset momentum diagnostic at a single point in time -- computable
    with only data available *as of* period_start, i.e. exactly what a real-time
    trader would have had on hand."""
    z = {}
    for s in symbols:
        v = raw_momentum_zscore(s, period_start, prices)
        if v is not None:
            z[s] = v
    return pd.Series(z)


def adaptive_momentum_sign(realized_corrs):
    """Shrinkage blend of the pre-registered literature prior and the expanding-
    window average of *already-realized* per-period correlations (momentum z-score
    at that period's start vs. that period's actual return). Strictly causal: the
    sign used for period p depends only on periods 1..p-1, each of which is only
    knowable once period p starts (period p-1 has already ended by then)."""
    n = len(realized_corrs)
    mean_realized = float(np.mean(realized_corrs)) if n > 0 else 0.0
    return (ADAPTIVE_PRIOR_N * ADAPTIVE_PRIOR_CORR + n * mean_realized) / (ADAPTIVE_PRIOR_N + n)


def run_period(symbols, sectors, prices_yf, official_wide, start, end, seed, momentum_sign=1.0, pool=None):
    period_len = (official_wide.loc[start:end].index.shape[0]) - 1
    t0 = time.time()
    task_args = [(sym, start, period_len, seed + i, momentum_sign) for i, sym in enumerate(symbols)]

    if pool is not None:
        per_asset = dict(pool.map(_fit_one_asset, task_args))
        print(f"    all {len(symbols)} assets fit, {time.time()-t0:.0f}s elapsed", flush=True)
    else:
        _init_worker(prices_yf, sectors)  # reuse the same fit function in-process
        per_asset = {}
        for i, args in enumerate(task_args):
            sym, res = _fit_one_asset(args)
            per_asset[sym] = res
            if (i + 1) % 20 == 0:
                print(f"    ...{i+1}/{len(symbols)} assets fit, {time.time()-t0:.0f}s elapsed", flush=True)

    draws_matrix = np.column_stack([per_asset[s]["draws"] for s in symbols])  # (N_SIM, 100)
    mean_pred = pd.Series({s: per_asset[s]["mean"] for s in symbols})

    n = len(symbols)
    from scipy.stats import rankdata
    order = rankdata(draws_matrix, method="ordinal", axis=1)          # (N_SIM, n), 1..n
    bucket = np.clip(np.ceil(order / n * 5).astype(int), 1, 5)         # (N_SIM, n), 1..5
    counts = np.zeros((n, 5))
    for k in range(1, 6):
        counts[:, k - 1] = (bucket == k).sum(axis=0)
    forecast_df = pd.DataFrame(counts / N_SIM, index=symbols, columns=RANK_COLS)

    signal = mean_pred - mean_pred.median()
    denom = signal.abs().sum()
    weights = signal / denom if denom > 0 else signal * 0.0

    return forecast_df, weights, per_asset


if __name__ == "__main__":
    import sys
    adaptive = "--adaptive" in sys.argv
    shrink = "--shrink" in sys.argv
    if adaptive:
        momentum_sign = None  # decided per-period below, causally
        out_suffix = "_adaptive"
        print("adaptive mode: momentum_sign estimated per-period from prior periods only")
    else:
        momentum_sign = -1.0 if "--reversal" in sys.argv else 1.0
        out_suffix = "_reversal" if momentum_sign < 0 else ""
        print(f"momentum_sign={momentum_sign}  (pass --reversal to flip, --adaptive for causal per-period sign)")
    if shrink:
        out_suffix += "_shrink"
        print("shrink mode: RPS forecast blended toward uniform, blend weight estimated per-period from prior periods only")

    symbols = json.load(open("data/symbols.json"))
    asset_info = json.load(open("data/asset_info.json"))
    sectors = {s: classify(s, asset_info) for s in symbols}
    prices_yf = pd.read_pickle("data/yf_prices.pkl")
    official_wide = load_official_prices()

    bounds = period_boundaries(official_wide)
    print(f"{len(bounds)-1} periods: {bounds}")

    # Parallel across assets defaults OFF -- see run_backtest.py module docstring
    # note and project memory: on this machine, hierboost's JAX-based factor fit
    # pays its JIT-compilation cost independently in every forked worker process,
    # which made an early test *slower* than serial and pegged every core, visibly
    # slowing down the rest of the laptop. Serial mode alone (with BLAS/JAX capped
    # to 1 thread each, done above) already runs at the same wall-clock speed as
    # the old implicit ~19-core-via-BLAS-threading version, just using 1 core
    # instead of ~19 -- that's the safe win, kept unconditionally. True
    # multi-process parallelism needs the JIT-warmup/oversubscription issue
    # actually fixed and re-verified before it's worth turning on; pass
    # --parallel to opt in once that's done.
    use_pool = "--parallel" in sys.argv
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    for i, a in enumerate(sys.argv):
        if a == "--n-workers" and i + 1 < len(sys.argv):
            n_workers = int(sys.argv[i + 1])
    pool = mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(prices_yf, sectors)) if use_pool else None
    print(f"{f'{n_workers} worker processes' if use_pool else 'serial (single core, BLAS/JAX capped to 1 thread)'}, "
          f"{os.cpu_count()} logical cores available")

    results = []
    all_forecasts = {}
    realized_corrs = []  # filled in only *after* each period's outcome is known
    prior_best_blends = []  # same discipline, for the shrinkage blend
    try:
        for p in range(len(bounds) - 1):
            start, end = bounds[p], bounds[p + 1]

            if adaptive:
                sign_p = adaptive_momentum_sign(realized_corrs)
                print(f"\n=== Period {p+1}: {start.date()} -> {end.date()} "
                      f"(momentum_sign={sign_p:.3f}, from {len(realized_corrs)} prior period(s)) ===", flush=True)
            else:
                sign_p = momentum_sign
                print(f"\n=== Period {p+1}: {start.date()} -> {end.date()} ===", flush=True)

            t0 = time.time()
            forecast_df, weights, per_asset = run_period(symbols, sectors, prices_yf, official_wide,
                                                           start, end, seed=1000 * p, momentum_sign=sign_p,
                                                           pool=pool)
            print(f"  fit+simulate: {time.time()-t0:.0f}s")

            if shrink:
                blend_p = adaptive_blend_weight(prior_best_blends)
                scored_forecast_df = apply_blend(forecast_df, blend_p)
                print(f"  shrinkage blend={blend_p:.3f} (from {len(prior_best_blends)} prior period(s))")
            else:
                blend_p = None
                scored_forecast_df = forecast_df

            period_returns = (official_wide.loc[end] / official_wide.loc[start] - 1).dropna()
            rps, rps_detail = compute_rps(period_returns, scored_forecast_df)

            # pct_change on the *full* series before slicing, not after -- slicing
            # first would make the period's own start-day return NaN (no prior row
            # inside the slice) and silently drop it, shifting every counted return
            # one trading day late relative to the official calculation, which uses
            # the previous day's close (just before the period start) as the first
            # day's baseline. Confirmed against team T's real submissions: this
            # ordering reproduces the official Global-leaderboard M1/M6 IR values to
            # 6 decimal places; the slice-then-diff ordering did not.
            daily = official_wide.pct_change(fill_method=None).loc[start:end].dropna(how="all")
            ir, ir_detail = compute_ir(daily, weights)

            print(f"  RPS={rps:.4f}  IR={ir:.4f}")
            results.append(dict(period=p + 1, start=str(start.date()), end=str(end.date()), rps=rps, ir=ir,
                                 momentum_sign=sign_p, blend=blend_p))
            all_forecasts[p + 1] = dict(forecast=scored_forecast_df, raw_forecast=forecast_df, weights=weights)

            if adaptive:
                # Only now -- after this period's actual outcome exists -- fold its
                # realized momentum-vs-return correlation into future periods' priors.
                z_start = period_momentum_zscores(symbols, prices_yf, start)
                common = z_start.index.intersection(period_returns.index)
                if len(common) > 10:
                    corr_p = np.corrcoef(z_start[common].values, period_returns[common].values)[0, 1]
                    if not np.isnan(corr_p):
                        realized_corrs.append(corr_p)
                        print(f"  realized momentum-return corr this period: {corr_p:.3f}")

            if shrink:
                # Same discipline as above: only usable for *future* periods' blend
                # choice, since it needs this period's own now-known outcome.
                curve = rps_curve(forecast_df, period_returns)
                prior_best_blends.append(best_blend(curve))
                print(f"  this period's own best-blend-in-hindsight: {prior_best_blends[-1]:.2f} "
                      f"(RPS range {curve.min():.4f}-{curve.max():.4f})")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results{out_suffix}.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nMean RPS across 12 periods: {results_df.rps.mean():.4f}")
    print(f"Mean IR across 12 periods:  {results_df.ir.mean():.4f}")

    import pickle
    with open(f"all_forecasts{out_suffix}.pkl", "wb") as f:
        pickle.dump(all_forecasts, f)
    print(f"\nSaved results{out_suffix}.csv and all_forecasts{out_suffix}.pkl")
