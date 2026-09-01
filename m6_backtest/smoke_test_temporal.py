"""Smoke test for forecast_temporal.py, before committing to a full walk-forward
re-run. Same discipline as prior smoke tests in this project (adaptive momentum
sign was smoke-tested on a 2-period subset first) -- small asset subset, small
period count, checking for crashes/NaNs/timing/plausible rho values, and a
side-by-side comparison against the existing snapshot (forecast.py) pipeline on
the identical assets/periods. Not a real backtest result -- too small a sample to
mean anything score-wise, purely a correctness and plausibility check.
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
from run_backtest import load_official_prices, period_boundaries
from scoring import compute_rps, compute_ir, RANK_COLS

# A deliberately diverse small subset, all confirmed present in data/symbols.json
# (an earlier pass included AAPL, which is NOT part of the M6 100-asset universe --
# a wrong test-harness pick, not a pipeline bug; forecast_asset/forecast_asset_temporal
# both handled it gracefully by returning None rather than crashing): mega-cap
# tech, a sector SPDR, a bond ETF, a commodity ETF, an international ETF. Not
# cherry-picked for a good result -- picked for structural diversity before
# looking at any score.
SMOKE_ASSETS = ["AMZN", "XLK", "TLT", "GSG", "EWJ", "JPM", "META"]
N_SMOKE_PERIODS = 2


def main():
    symbols = json.load(open("data/symbols.json"))
    asset_info = json.load(open("data/asset_info.json"))
    sectors = {s: classify(s, asset_info) for s in symbols}
    prices_yf = pd.read_pickle("data/yf_prices.pkl")
    official_wide = load_official_prices()
    bounds = period_boundaries(official_wide)

    print(f"Smoke test: {len(SMOKE_ASSETS)} assets x {N_SMOKE_PERIODS} periods, "
          f"snapshot (forecast_asset) vs temporal (forecast_asset_temporal)\n")

    rows = []
    for p in range(N_SMOKE_PERIODS):
        start, end = bounds[p], bounds[p + 1]
        print(f"=== Period {p+1}: {start.date()} -> {end.date()} ===")
        period_len = (official_wide.loc[start:end].index.shape[0]) - 1
        period_returns = (official_wide.loc[end] / official_wide.loc[start] - 1)

        # Built ONCE per period across the FULL 100-symbol universe (not just the
        # 7 smoke-test targets) -- this is the real usage pattern: every target's
        # regression shares it, which is the whole point of the fix. Timed
        # separately from the per-target calls below since in a real walk-forward
        # this cost is paid once per period, not once per (period, target) pair.
        t0 = time.time()
        asset_cache = build_asset_temporal_cache(symbols, prices_yf, start, period_len)
        t_cache = time.time() - t0
        n_cached = sum(1 for v in asset_cache.values() if v is not None)
        print(f"  asset cache built: {n_cached}/{len(symbols)} assets, {t_cache:.1f}s")

        for sym in SMOKE_ASSETS:
            t0 = time.time()
            snap = forecast_asset(sym, start, prices_yf, sectors, period_len_days=period_len,
                                   n_samples=500, burn_in=300, seed=p * 100)
            t_snap = time.time() - t0

            t0 = time.time()
            temp = forecast_asset_temporal(sym, start, prices_yf, sectors, period_len_days=period_len,
                                            n_samples=500, burn_in=300, seed=p * 100,
                                            asset_cache=asset_cache)
            t_temp = time.time() - t0

            actual = period_returns.get(sym, np.nan)
            if snap is None or temp is None:
                print(f"  {sym}: SKIPPED (insufficient history) snap={snap is not None} temp={temp is not None}")
                continue

            snap_mean = snap["mean"]
            temp_mean = temp["mean"]
            n_nan_snap = np.isnan(snap["draws"]).sum()
            n_nan_temp = np.isnan(temp["draws"]).sum()
            rhos = temp["rho_per_block"]

            print(f"  {sym:5s} n_blocks(snap/temp)={snap['n_blocks']:2d}/{temp['n_blocks']:2d}  "
                  f"actual={actual:+.4f}  snap_pred={snap_mean:+.4f} ({t_snap:.2f}s)  "
                  f"temp_pred={temp_mean:+.4f} ({t_temp:.2f}s)  "
                  f"nan(snap/temp)={n_nan_snap}/{n_nan_temp}  "
                  f"rho[min/med/max]={np.min(rhos):+.2f}/{np.median(rhos):+.2f}/{np.max(rhos):+.2f}")

            rows.append(dict(period=p + 1, sym=sym, actual=actual, snap_pred=snap_mean,
                              temp_pred=temp_mean, t_snap=t_snap, t_temp=t_temp,
                              n_blocks=temp["n_blocks"], rho_median=np.median(rhos),
                              rho_min=np.min(rhos), rho_max=np.max(rhos)))

    df = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nMean fit time: snapshot={df.t_snap.mean():.2f}s  temporal={df.t_temp.mean():.2f}s "
          f"({df.t_temp.mean()/df.t_snap.mean():.1f}x)")
    print(f"rho distribution across all block fits: min={df.rho_min.min():.3f} "
          f"median={df.rho_median.median():.3f} max={df.rho_max.max():.3f}")
    print(f"corr(snap_pred, temp_pred) across the {len(df)} asset-periods tested: "
          f"{df.snap_pred.corr(df.temp_pred):.3f}")
    df.to_csv("smoke_test_temporal_results.csv", index=False)
    print("\nSaved smoke_test_temporal_results.csv")


if __name__ == "__main__":
    main()
