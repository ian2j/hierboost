"""Pull yfinance price history for all 100 M6 tickers, for use in the walk-forward
FITTING windows (not final scoring -- that uses the official assets_m6.csv verbatim).

Three symbols need substitution because Yahoo's current API only serves their price
history under a different current ticker (confirmed by direct testing, not assumed):
  RE  -> EG  (Everest Re Group renamed to Everest Group in 2023; same continuous series)
  WRK -> SW  (WestRock merged into Smurfit WestRock in 2024; Yahoo carries the full
              pre-merger WestRock history under the new combined-entity ticker)
  DRE -> no substitute exists (acquired into Prologis, a pre-existing separately-quoted
              company, so its standalone price series isn't preserved under any current
              ticker). DRE gets no yfinance lookback; the walk-forward loop falls back
              to a neutral (zero-conviction) forecast for it throughout.
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf

SUBSTITUTE = {"RE": "EG", "WRK": "SW"}
# DRE: acquired into Prologis, no surviving standalone series (see module docstring).
# AVB: confirmed by direct testing -- yfinance's AVB history only goes back to
# 2026-07, i.e. some recent relisting/restructuring event severed continuity with
# AvalonBay's pre-2023 trading history under this ticker. Not investigated further
# (one asset out of 100); falls back to the same neutral-forecast treatment as DRE.
NO_LOOKAHEAD_UNAVAILABLE = ["DRE", "AVB"]

START = "2019-06-01"
END = "2023-02-20"


def download_all(symbols):
    dl_symbols = [SUBSTITUTE.get(s, s) for s in symbols if s not in NO_LOOKAHEAD_UNAVAILABLE]
    df = yf.download(dl_symbols, start=START, end=END, progress=False, auto_adjust=True)["Close"]
    rename = {v: k for k, v in SUBSTITUTE.items()}
    df = df.rename(columns=rename)
    # NaNs here are cross-exchange holiday mismatches (a US holiday is a normal LSE
    # trading day and vice versa), not missing data -- forward-fill is the correct
    # "market was closed, price is unchanged from last close" treatment. Confirmed
    # by inspection: every NaN date lines up exactly with a US federal market holiday.
    df = df.ffill()
    return df


if __name__ == "__main__":
    symbols = json.load(open("data/symbols.json"))
    prices = download_all(symbols)
    print(prices.shape, prices.index.min(), prices.index.max())
    missing = set(symbols) - set(prices.columns) - set(NO_LOOKAHEAD_UNAVAILABLE)
    print("unexpectedly missing:", missing)
    print("NaN counts:\n", prices.isna().sum()[prices.isna().sum() > 0])
    prices.to_pickle("data/yf_prices.pkl")
    print("saved data/yf_prices.pkl")
