"""First REAL scored (RPS/IR) comparison between the snapshot pipeline
(forecast.py, with its already-validated adaptive causal momentum sign -- the
v4 design, the fair baseline to compare against since the temporal model's
whole point is to replace that same adaptive-sign mechanism with something
endogenous) and the temporal pipeline (forecast_temporal.py, v3/cached), on
the FULL 100-asset universe -- not just the 7-asset point-prediction smoke
test, which had no RPS/IR signal at all. Still only N_PERIODS periods: not
remotely enough for a statistically meaningful comparison (the full project's
own headline results needed 48 periods to reach significance), but it's the
first point in this exercise where "does the temporal version look better,"
not just "does it run," is even askable. Same causal discipline as
run_backtest.py's --adaptive mode: period p's momentum sign depends only on
periods 1..p-1's own already-realized outcomes.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import numpy as np
import pandas as pd

from classification import classify
from forecast import forecast_asset
from forecast_temporal import forecast_asset_temporal, build_asset_temporal_cache
from run_backtest import (
    load_official_prices, period_boundaries, DRE_LAST_DATE, N_SIM,
    adaptive_momentum_sign, period_momentum_zscores,
)
from scoring import compute_rps, compute_ir, RANK_COLS

N_PERIODS = 3


def neutral_forecast(n_sim):
    return dict(draws=np.zeros(n_sim), mean=0.0, n_blocks=0, n_obs=0)


def build_forecast_df_and_weights(symbols, per_asset):
    draws_matrix = np.column_stack([per_asset[s]["draws"] for s in symbols])
    mean_pred = pd.Series({s: per_asset[s]["mean"] for s in symbols})
    n = len(symbols)
    from scipy.stats import rankdata
    order = rankdata(draws_matrix, method="ordinal", axis=1)
    bucket = np.clip(np.ceil(order / n * 5).astype(int), 1, 5)
    counts = np.zeros((n, 5))
    for k in range(1, 6):
        counts[:, k - 1] = (bucket == k).sum(axis=0)
    forecast_df = pd.DataFrame(counts / N_SIM, index=symbols, columns=RANK_COLS)
    signal = mean_pred - mean_pred.median()
    denom = signal.abs().sum()
    weights = signal / denom if denom > 0 else signal * 0.0
    return forecast_df, weights


def run_period_snapshot(symbols, sectors, prices_yf, start, period_len, momentum_sign, seed_base):
    per_asset = {}
    for i, sym in enumerate(symbols):
        if sym == "DRE" and start >= DRE_LAST_DATE:
            per_asset[sym] = neutral_forecast(N_SIM)
            continue
        res = forecast_asset(sym, start, prices_yf, sectors, period_len_days=max(period_len, 1),
                              n_samples=N_SIM, burn_in=400, seed=seed_base + i, momentum_sign=momentum_sign)
        per_asset[sym] = res if res is not None else neutral_forecast(N_SIM)
    return build_forecast_df_and_weights(symbols, per_asset)


def run_period_temporal(symbols, sectors, prices_yf, start, period_len, seed_base):
    t0 = time.time()
    asset_cache = build_asset_temporal_cache(symbols, prices_yf, start, max(period_len, 1))
    t_cache = time.time() - t0
    n_cached = sum(1 for v in asset_cache.values() if v is not None)

    per_asset = {}
    for i, sym in enumerate(symbols):
        if sym == "DRE" and start >= DRE_LAST_DATE:
            per_asset[sym] = neutral_forecast(N_SIM)
            continue
        res = forecast_asset_temporal(sym, start, prices_yf, sectors, period_len_days=max(period_len, 1),
                                       n_samples=N_SIM, burn_in=400, seed=seed_base + i,
                                       asset_cache=asset_cache)
        per_asset[sym] = res if res is not None else neutral_forecast(N_SIM)
    return build_forecast_df_and_weights(symbols, per_asset), t_cache, n_cached


def main():
    symbols = json.load(open("data/symbols.json"))
    asset_info = json.load(open("data/asset_info.json"))
    sectors = {s: classify(s, asset_info) for s in symbols}
    prices_yf = pd.read_pickle("data/yf_prices.pkl")
    official_wide = load_official_prices()
    bounds = period_boundaries(official_wide)

    print(f"Scored comparison: {len(symbols)} assets x {N_PERIODS} periods, "
          f"snapshot+adaptive-sign vs temporal+cache\n")

    results = []
    realized_corrs = []  # for the snapshot pipeline's adaptive momentum sign, same causal rule as run_backtest.py
    for p in range(N_PERIODS):
        start, end = bounds[p], bounds[p + 1]
        period_len = (official_wide.loc[start:end].index.shape[0]) - 1
        period_returns = (official_wide.loc[end] / official_wide.loc[start] - 1).dropna()
        daily = official_wide.pct_change(fill_method=None).loc[start:end].dropna(how="all")

        sign_p = adaptive_momentum_sign(realized_corrs)
        print(f"=== Period {p+1}: {start.date()} -> {end.date()} (snapshot momentum_sign={sign_p:.3f}) ===")

        t0 = time.time()
        snap_df, snap_w = run_period_snapshot(symbols, sectors, prices_yf, start, period_len, sign_p, seed_base=1000 * p)
        t_snap = time.time() - t0
        rps_snap, _ = compute_rps(period_returns, snap_df)
        ir_snap, _ = compute_ir(daily, snap_w)
        print(f"  snapshot:  RPS={rps_snap:.4f}  IR={ir_snap:+.4f}  ({t_snap:.0f}s)")

        t0 = time.time()
        (temp_df, temp_w), t_cache, n_cached = run_period_temporal(symbols, sectors, prices_yf, start, period_len, seed_base=1000 * p)
        t_temp = time.time() - t0
        rps_temp, _ = compute_rps(period_returns, temp_df)
        ir_temp, _ = compute_ir(daily, temp_w)
        print(f"  temporal:  RPS={rps_temp:.4f}  IR={ir_temp:+.4f}  ({t_temp:.0f}s, cache {n_cached}/{len(symbols)} in {t_cache:.0f}s)")

        results.append(dict(period=p + 1, start=str(start.date()), end=str(end.date()),
                             momentum_sign=sign_p, rps_snap=rps_snap, ir_snap=ir_snap,
                             rps_temp=rps_temp, ir_temp=ir_temp, t_snap=t_snap, t_temp=t_temp))

        z_start = period_momentum_zscores(symbols, prices_yf, start)
        common = z_start.index.intersection(period_returns.index)
        if len(common) > 10:
            corr_p = np.corrcoef(z_start[common].values, period_returns[common].values)[0, 1]
            if not np.isnan(corr_p):
                realized_corrs.append(corr_p)

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nMean RPS:  snapshot={df.rps_snap.mean():.4f}  temporal={df.rps_temp.mean():.4f}  "
          f"(naive benchmark ~0.160)")
    print(f"Mean IR:   snapshot={df.ir_snap.mean():+.4f}  temporal={df.ir_temp.mean():+.4f}")
    df.to_csv("smoke_test_temporal_scored_results.csv", index=False)
    print("\nSaved smoke_test_temporal_scored_results.csv")
    print(f"\n(n={N_PERIODS} periods -- far too small to draw a real conclusion from; "
          f"this is a plausibility check, not a result.)")


if __name__ == "__main__":
    main()
