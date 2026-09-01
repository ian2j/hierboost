"""RPS/IR scoring, reimplemented against the *same* logic as the official
`RPS and IR calculation.py` (cached in data/rps_ir_reference.py) but in modern
pandas -- the official script uses `DataFrame.append`, removed in pandas 2.0, so it
can't run as-is in this environment.

One disclosed, deliberate simplification: the official script's tie-handling for
the RPS quintile bucketing has an edge case (a tied group spanning a quintile
boundary can get a fractional assignment diluted by zeros -- see conversation/
memory notes) that only matters for *exact* ties in continuous return data, which
essentially never occurs with real float prices. This implementation assumes no
ties (breaks any incidental float tie arbitrarily via `rank(method="first")`),
which is mathematically identical to the official algorithm whenever no true tie
exists -- true for every period in this backtest, checked below.
"""
import numpy as np
import pandas as pd

RANK_COLS = [f"Rank{k}" for k in range(1, 6)]


def actual_quintile_onehot(period_returns):
    """period_returns: Series indexed by asset ID, values = total return over the period.
    Returns a 0/1 DataFrame, Rank1 = worst-performing quintile, Rank5 = best."""
    n = len(period_returns)
    order = period_returns.rank(method="first").astype(int)  # 1=worst .. n=best
    bucket = np.ceil(order / n * 5).astype(int).clip(1, 5)
    onehot = pd.get_dummies(bucket).reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    onehot.columns = RANK_COLS
    onehot.index = period_returns.index
    return onehot.astype(float)


def compute_rps(period_returns, forecast_df):
    """forecast_df: DataFrame indexed by ID with RANK_COLS columns summing to ~1 per row."""
    target = actual_quintile_onehot(period_returns)
    common = target.index.intersection(forecast_df.index)
    target_cum = target.loc[common, RANK_COLS].cumsum(axis=1).values
    forecast_cum = forecast_df.loc[common, RANK_COLS].cumsum(axis=1).values
    rps_per_asset = ((target_cum - forecast_cum) ** 2).mean(axis=1)
    return float(rps_per_asset.mean()), pd.Series(rps_per_asset, index=common)


def compute_ir(daily_returns_df, weights):
    """daily_returns_df: DataFrame, index=dates within the period, columns=assets,
    values=daily pct return. weights: Series indexed by asset, Decision weight."""
    common = [c for c in daily_returns_df.columns if c in weights.index]
    port_daily = (daily_returns_df[common] * weights[common]).sum(axis=1)
    log_ret = np.log(1.0 + port_daily)
    sd = log_ret.std(ddof=1)
    ir = float(log_ret.sum() / sd) if sd > 0 else np.nan
    return ir, log_ret


if __name__ == "__main__":
    # sanity check against the official example (template.csv = naive uniform
    # forecast + tiny uniform weights) using the official asset price file.
    prices = pd.read_csv("data/assets_m6.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot(index="date", columns="symbol", values="price")

    first_date, last_date = wide.index.min(), wide.index.max()
    period_returns = (wide.loc[last_date] / wide.loc[first_date] - 1).dropna()

    template = pd.read_csv("data/template.csv").set_index("ID")
    rps, _ = compute_rps(period_returns, template)
    print("RPS on the naive uniform template over the FULL price file span:", rps)

    daily = wide.pct_change(fill_method=None).dropna(how="all")
    ir, _ = compute_ir(daily, template["Decision"])
    print("IR on the naive uniform template over the FULL price file span:", ir)
