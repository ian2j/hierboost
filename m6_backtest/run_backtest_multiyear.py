"""Extends the validated 12-period M6 walk-forward (run_backtest.py --adaptive
--shrink) continuously through 3 more ~12-period "years" (36 more periods, same
28-day cadence as the real M6 deadlines), to test whether the v6 result
generalizes out-of-sample to periods nobody looked at while building any of
this -- 2022 was the only year any diagnosis/tuning ever touched.

Periods 1-12: identical to run_backtest.py --adaptive --shrink (same data
source, assets_m6.csv, same seeds) -- reproduced here rather than re-run
separately so the adaptive momentum-sign and shrinkage state carries forward
continuously into periods 13-48, exactly as a real trader's model would keep
learning rather than resetting at an arbitrary year boundary.

Periods 13-48: no official M6 data exists past Feb 2023 (competition ended) --
scored against yfinance's own price panel instead, disclosed as such. AVB is
excluded from this whole extended range (not just the lookback fit): confirmed
live that yfinance serves *no* AVB history at all before 2026-07, a data-source
quirk unrelated to any real corporate event (AvalonBay traded normally). DRE
stays excluded from ~period 10 onward as before (real 2022 delisting). Effective
universe for periods 13-48: 98 of the 100 assets.
"""
import json
import pickle
import numpy as np
import pandas as pd

from classification import classify
from run_backtest import (
    run_period, load_official_prices, CORRECT_PERIOD_STARTS,
    adaptive_momentum_sign, period_momentum_zscores,
)
from shrinkage import rps_curve, best_blend, adaptive_blend_weight, apply_blend
from scoring import compute_rps, compute_ir

N_EXTRA_YEARS = 3
PERIODS_PER_YEAR = 12


def extended_period_starts():
    """Continues seamlessly from period 12's actual final boundary (2023-02-02),
    adding clean 28-day increments from there. NOT the same as continuing the
    arithmetic "+28 days from period 12's start" pattern (which would land on
    2023-02-06): period 12's own end was determined empirically (closest match
    to team T's real IR), 24 days after its start rather than the usual 28 --
    the M6 competition's own final period was apparently a few days short of
    the clean cadence. Restarting from the theoretical +28 date instead of the
    real empirical one would leave a 4-day gap with no market data in it;
    continuing from the actual boundary keeps full, gapless daily coverage."""
    last_boundary = pd.Timestamp(CORRECT_PERIOD_STARTS[-1])  # 2023-02-02, period 12's real end
    starts = list(CORRECT_PERIOD_STARTS)  # 13 dates, periods 1-12
    n_new = N_EXTRA_YEARS * PERIODS_PER_YEAR
    for i in range(1, n_new + 1):
        starts.append((last_boundary + pd.Timedelta(days=28 * i)).strftime("%Y-%m-%d"))
    return starts


def extended_price_panel(official_wide):
    """Official M6 data (assets_m6.csv, via official_wide) for dates within its
    own range; yfinance's own panel beyond that (no official source exists past
    Feb 2023). AVB/DRE naturally end up NaN in the yfinance-only extension
    (AVB has no yfinance history at all pre-2026; DRE stopped trading in
    2022) -- downstream .dropna() calls already used for scoring handle this
    the same way they already handle any other missing asset."""
    yf_ext = pd.read_pickle("data/yf_prices_extended.pkl")
    cutoff = official_wide.index.max()
    yf_tail = yf_ext.loc[yf_ext.index > cutoff]
    combined = pd.concat([official_wide, yf_tail], axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def snap_boundaries(price_panel, target_dates):
    idx = price_panel.index.sort_values()
    coverage = price_panel.notna().mean(axis=1)
    valid_idx = idx[coverage.loc[idx] >= 0.85]  # slightly relaxed vs 0.9: fewer
    # assets in the yfinance-only tail (98 vs 100), so the same absolute count
    # of "missing today" assets is a bigger fraction -- 0.85 still safely
    # excludes genuine holiday-only rows (which have ~10-20% coverage, not ~85%+)
    bounds = []
    for d in target_dates:
        target = pd.Timestamp(d)
        candidates = valid_idx[valid_idx >= target]
        bounds.append(candidates[0] if len(candidates) else valid_idx[-1])
    return bounds


if __name__ == "__main__":
    import time

    symbols = json.load(open("data/symbols.json"))
    asset_info = json.load(open("data/asset_info.json"))
    sectors = {s: classify(s, asset_info) for s in symbols}
    prices_yf = pd.read_pickle("data/yf_prices_extended.pkl")  # same lookback source throughout
    official_wide_m6 = load_official_prices()
    price_panel = extended_price_panel(official_wide_m6)

    starts = extended_period_starts()
    bounds = snap_boundaries(price_panel, starts)
    n_periods = len(bounds) - 1
    print(f"{n_periods} periods total (12 original M6 + {n_periods - 12} extended): "
          f"{bounds[0].date()} .. {bounds[-1].date()}")

    results = []
    all_forecasts = {}
    realized_corrs = []
    prior_best_blends = []
    for p in range(n_periods):
        start, end = bounds[p], bounds[p + 1]
        era = "M6" if p < 12 else f"extended-yr{(p - 12) // 12 + 1}"

        sign_p = adaptive_momentum_sign(realized_corrs)
        print(f"\n=== Period {p+1}/{n_periods} [{era}]: {start.date()} -> {end.date()} "
              f"(momentum_sign={sign_p:.3f}, from {len(realized_corrs)} prior) ===", flush=True)

        t0 = time.time()
        forecast_df, weights, per_asset = run_period(symbols, sectors, prices_yf, price_panel,
                                                       start, end, seed=1000 * p, momentum_sign=sign_p, pool=None)
        print(f"  fit+simulate: {time.time()-t0:.0f}s")

        blend_p = adaptive_blend_weight(prior_best_blends)
        scored_forecast_df = apply_blend(forecast_df, blend_p)
        print(f"  shrinkage blend={blend_p:.3f} (from {len(prior_best_blends)} prior)")

        period_returns = (price_panel.loc[end] / price_panel.loc[start] - 1).dropna()
        rps, _ = compute_rps(period_returns, scored_forecast_df)

        daily = price_panel.pct_change(fill_method=None).loc[start:end].dropna(how="all")
        ir, _ = compute_ir(daily, weights)

        print(f"  RPS={rps:.4f}  IR={ir:.4f}  n_assets_scored={len(period_returns)}")
        results.append(dict(period=p + 1, era=era, start=str(start.date()), end=str(end.date()),
                             rps=rps, ir=ir, momentum_sign=sign_p, blend=blend_p,
                             n_assets_scored=len(period_returns)))
        all_forecasts[p + 1] = dict(forecast=scored_forecast_df, raw_forecast=forecast_df, weights=weights)

        z_start = period_momentum_zscores(symbols, prices_yf, start)
        common = z_start.index.intersection(period_returns.index)
        if len(common) > 10:
            corr_p = np.corrcoef(z_start[common].values, period_returns[common].values)[0, 1]
            if not np.isnan(corr_p):
                realized_corrs.append(corr_p)

        curve = rps_curve(forecast_df, period_returns)
        prior_best_blends.append(best_blend(curve))

    results_df = pd.DataFrame(results)
    results_df.to_csv("results_multiyear.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))
    print("\nBy era:")
    print(results_df.groupby("era")[["rps", "ir"]].mean())

    with open("all_forecasts_multiyear.pkl", "wb") as f:
        pickle.dump(all_forecasts, f)
    print("\nSaved results_multiyear.csv and all_forecasts_multiyear.pkl")
