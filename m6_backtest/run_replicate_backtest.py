"""Walk-forward test of the DBMF-replication investment-decision strategy: each
period, refit replicate_weights() using only data through that period's start
(same causal discipline as run_backtest.py --adaptive), renormalize to
sum(|w|) <= 1, and score IR against the real M6 asset universe/dates. RPS is
unaffected by decision weights, so this only tests the IR/investment side, as a
direct alternative to the momentum/reversal overlay -- reuses v4's RPS forecast
unchanged if a combined submission is ever wanted.
"""
import json
import time
import numpy as np
import pandas as pd

from replicate import replicate_weights
from run_backtest import load_official_prices, period_boundaries
from scoring import compute_ir

if __name__ == "__main__":
    prices_yf = pd.read_pickle("data/yf_prices.pkl")
    dbmf = pd.read_pickle("data/dbmf_price.pkl")["DBMF"]
    official_wide = load_official_prices()
    bounds = period_boundaries(official_wide)

    print(f"DBMF history: {dbmf.index.min().date()} .. {dbmf.index.max().date()}, "
          f"{len(dbmf)} obs")

    results = []
    all_weights = {}
    all_log_rets = []
    for p in range(len(bounds) - 1):
        start, end = bounds[p], bounds[p + 1]
        t0 = time.time()
        raw_w = replicate_weights(dbmf, start, prices_yf, seed=1000 * p)
        if raw_w is None:
            print(f"Period {p+1}: not enough history, skipping (neutral weights)")
            weights = pd.Series(dtype=float)
        else:
            denom = raw_w.abs().sum()
            weights = raw_w / denom if denom > 0 else raw_w * 0.0

        daily = official_wide.pct_change(fill_method=None).loc[start:end].dropna(how="all")
        ir, log_ret = compute_ir(daily, weights) if len(weights) else (np.nan, pd.Series(dtype=float))
        all_log_rets.append(log_ret)

        n_active = (weights.abs() > 1e-6).sum()
        max_w = weights.abs().max() if len(weights) else np.nan
        print(f"Period {p+1}: {start.date()} -> {end.date()}  IR={ir:.4f}  "
              f"n_active={n_active}  max|w|={max_w:.3f}  ({time.time()-t0:.1f}s)")
        results.append(dict(period=p + 1, start=str(start.date()), end=str(end.date()),
                             ir=ir, n_active=int(n_active), max_w=max_w))
        all_weights[p + 1] = weights

    df = pd.DataFrame(results)
    df.to_csv("results_replicate.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nMean IR across periods (mean-of-monthly): {df.ir.mean():.4f}")

    pooled = pd.concat(all_log_rets)
    pooled_ir = pooled.sum() / pooled.std(ddof=1)
    print(f"Pooled Global IR (official aggregation): {pooled_ir:.4f}  (n_days={len(pooled)})")

    import pickle
    with open("all_weights_replicate.pkl", "wb") as f:
        pickle.dump(all_weights, f)
    print("Saved results_replicate.csv and all_weights_replicate.pkl")
