"""One-period diagnostic: why did the temporal pipeline score RPS=0.232 (worse
than the naive 0.16 benchmark) and IR=-2.72 on period 1, when the snapshot
pipeline scored RPS=0.160/IR=-1.20 on the identical period? Same diagnostic
the project's own earlier overconfidence bug (v1) was caught with: check (a)
directional correlation (predicted mean vs actual realized return -- does the
model have ANY signal, even weak/noisy) and (b) forecast confidence/spread
(is it overconfident given whatever signal it does have) -- these look
identical at the RPS/IR summary level but need different fixes.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import numpy as np
import pandas as pd

from classification import classify
from forecast import forecast_asset
from forecast_temporal import forecast_asset_temporal, build_asset_temporal_cache
from run_backtest import load_official_prices, period_boundaries, DRE_LAST_DATE, N_SIM

symbols = json.load(open("data/symbols.json"))
asset_info = json.load(open("data/asset_info.json"))
sectors = {s: classify(s, asset_info) for s in symbols}
prices_yf = pd.read_pickle("data/yf_prices.pkl")
official_wide = load_official_prices()
bounds = period_boundaries(official_wide)

start, end = bounds[0], bounds[1]
period_len = (official_wide.loc[start:end].index.shape[0]) - 1
period_returns = (official_wide.loc[end] / official_wide.loc[start] - 1).dropna()
print(f"Period 1: {start.date()} -> {end.date()}, momentum_sign=-0.100 (prior only)\n")

asset_cache = build_asset_temporal_cache(symbols, prices_yf, start, max(period_len, 1))

snap_mean, temp_mean, snap_std, temp_std = {}, {}, {}, {}
for i, sym in enumerate(symbols):
    if sym == "DRE" and start >= DRE_LAST_DATE:
        continue
    s = forecast_asset(sym, start, prices_yf, sectors, period_len_days=period_len,
                        n_samples=N_SIM, burn_in=400, seed=i, momentum_sign=-0.100)
    t = forecast_asset_temporal(sym, start, prices_yf, sectors, period_len_days=period_len,
                                 n_samples=N_SIM, burn_in=400, seed=i, asset_cache=asset_cache)
    if s is not None:
        snap_mean[sym], snap_std[sym] = s["mean"], s["draws"].std()
    if t is not None:
        temp_mean[sym], temp_std[sym] = t["mean"], t["draws"].std()

snap_mean, temp_mean = pd.Series(snap_mean), pd.Series(temp_mean)
snap_std, temp_std = pd.Series(snap_std), pd.Series(temp_std)

common = period_returns.index.intersection(snap_mean.index).intersection(temp_mean.index)
actual = period_returns.loc[common]

print(f"n_assets compared: {len(common)}")
print(f"\ncorr(predicted mean, actual return):")
print(f"  snapshot: {np.corrcoef(snap_mean.loc[common], actual)[0,1]:+.4f}")
print(f"  temporal: {np.corrcoef(temp_mean.loc[common], actual)[0,1]:+.4f}")

print(f"\nmean |predicted mean| (a confidence/magnitude proxy):")
print(f"  snapshot: {snap_mean.loc[common].abs().mean():.4f}")
print(f"  temporal: {temp_mean.loc[common].abs().mean():.4f}")
print(f"  actual:   {actual.abs().mean():.4f}")

print(f"\nmean predicted draws std (period-return units, calibration proxy -- compare to actual cross-sectional std):")
print(f"  snapshot: {snap_std.loc[common].mean():.4f}")
print(f"  temporal: {temp_std.loc[common].mean():.4f}")
print(f"  actual cross-sectional std this period: {actual.std():.4f}")

print(f"\nratio |predicted mean| / draws_std (a z-score-like 'how many SDs is my point guess from zero' -- "
      f"high values mean confidently far from 'I don't know'):")
print(f"  snapshot: {(snap_mean.loc[common].abs() / snap_std.loc[common]).mean():.4f}")
print(f"  temporal: {(temp_mean.loc[common].abs() / temp_std.loc[common]).mean():.4f}")

df = pd.DataFrame(dict(actual=actual, snap_mean=snap_mean.loc[common], temp_mean=temp_mean.loc[common],
                        snap_std=snap_std.loc[common], temp_std=temp_std.loc[common]))
df.to_csv("diagnose_temporal_p1_results.csv")
print("\nSaved diagnose_temporal_p1_results.csv")
