"""Real-data test of hierboost.copula's marginal transform: does the Gaussian-copula
generalization of the block-latent model actually help on real, heterogeneous-marginal
market data, not just the controlled synthetic check in copula_synthetic_validation.py?

Motivation (real, citable stylized fact, not a made-up test case): for a given name/ETF,
its own daily RETURN and daily dollar VOLUME are known to co-move -- the classic
"volume-volatility relation" / Mixture-of-Distributions-Hypothesis (Clark 1973, "A
Subordinated Stochastic Process Model with Finite Variance for Speculative Prices"):
both are driven by a shared, unobserved daily "information flow"/activity intensity.
That is exactly hierboost's block-latent premise -- a shared per-block latent driving
several correlated observed features -- but return and dollar volume have wildly
different marginal shapes (return ~ roughly symmetric, dollar volume ~ strongly
right-skewed/lognormal-like), exactly the case factor.py's raw PPCA was never built for
(see its own docstring: "ideally already standardized").

Design: block ETFs' own {return, dollar_volume} pairs (one block per ETF, block
structure supplied directly, not correlation-threshold-derived -- the pairing is
economically motivated, not data-mined), fit each block's shared "activity" latent
either the raw way (marginal=None) or through the copula transform first
(marginal="copula"), then use those per-ETF latents to explain AAPL's own return via
HierBoostRegressor's normal spike-and-slab step. Compared on a genuine chronological
train/test split (no shuffling -- same no-look-ahead discipline as earthquake_japan/
uk_weather), since in-sample correlation gains that don't survive an honest split are
exactly the earthquake demo's cautionary lesson.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from hierboost.estimator import HierBoostRegressor

TARGET = "AAPL"
PREDICTORS = ["SPY", "QQQ", "XLK", "VGT", "SMH", "IWM", "XLF", "XLE", "TLT", "GLD"]
TRAIN_FRAC = 0.75


def load_data():
    tickers = [TARGET] + PREDICTORS
    df = yf.download(tickers, period="3y", progress=False, auto_adjust=True)
    close = df["Close"]
    volume = df["Volume"]
    rets = np.log(close).diff()
    dollar_vol = (close * volume)
    data = pd.concat([rets.add_suffix("_ret"), dollar_vol.add_suffix("_dvol")], axis=1).dropna()
    return data


def build_design(data, target_kind="return"):
    """target_kind="return": AAPL's own next-day-style directional log return (hard --
    brushes against market efficiency, included as the naive first attempt).
    target_kind="volatility": AAPL's own absolute log return, a realized-volatility proxy.
    This is the theoretically matched target for the volume-volatility/MDH mechanism this
    demo is built on (Clark 1973's shared "activity" latent drives volatility/volume
    co-movement, not next-day directional return) -- cross-asset volatility spillover is
    well-documented and far more predictable than directional returns.
    """
    raw = data[f"{TARGET}_ret"].values
    y = np.abs(raw) if target_kind == "volatility" else raw
    y = (y - y.mean()) / y.std()

    cols, block_id = [], []
    for k, tic in enumerate(PREDICTORS):
        cols += [f"{tic}_ret", f"{tic}_dvol"]
        block_id += [k, k]
    X = data[cols].values
    # standardize returns; leave dollar volume RAW (its native right-skewed scale is
    # exactly the point -- pre-standardizing wouldn't remove the skew, but log-scaling it
    # here would silently do the copula's job by hand, defeating the comparison)
    for j, c in enumerate(cols):
        if c.endswith("_ret"):
            X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    return X, y, np.array(block_id, dtype=float), cols


def run(data, target_kind):
    X, y, block_id, cols = build_design(data, target_kind=target_kind)
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    print(f"\n{'=' * 78}\nTarget: AAPL {target_kind}  ({n} days, {n_train} train / {n - n_train} test, "
          f"chronological split)\n{'=' * 78}")

    if target_kind == "return":
        print("\nSkewness of each raw predictor column (dollar volume vs return):")
        for j, c in enumerate(cols):
            print(f"  {c:12s} skew={stats.skew(X[:n_train, j]):+.2f}")

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    coords = block_id  # nominal per-column block label; unused numerically when block_id is supplied directly

    def fit_and_eval(marginal):
        reg = HierBoostRegressor(decorrelate="sar", marginal=marginal, fit_method="em",
                                  xi0=-1.0, kappa=100.0)
        reg.fit(X_train, y_train, coords=coords, block_id=block_id,
                feature_names=[f"{tic}_activity" for tic in PREDICTORS])
        pred_train = reg.predict(X_train)
        pred_test = reg.predict(X_test)
        r2_train = 1 - np.sum((y_train - pred_train) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        r2_test = 1 - np.sum((y_test - pred_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        return reg, r2_train, r2_test

    reg_raw, r2_train_raw, r2_test_raw = fit_and_eval(None)
    reg_cop, r2_train_cop, r2_test_cop = fit_and_eval("copula")

    print(f"\nHeld-out R^2 (train -> test, chronological, no leakage):")
    print(f"  raw block-latent (marginal=None)     : train R^2={r2_train_raw:.4f}  test R^2={r2_test_raw:.4f}")
    print(f"  copula block-latent (marginal=copula): train R^2={r2_train_cop:.4f}  test R^2={r2_test_cop:.4f}")

    print(f"\nPosterior inclusion probability (theta_hat) per ETF's activity latent, "
          f"raw vs copula, sorted by copula:")
    order = np.argsort(-reg_cop.theta_hat_)
    for i in order:
        print(f"  {PREDICTORS[i]:6s}  raw theta={reg_raw.theta_hat_[i]:.4f}   "
              f"copula theta={reg_cop.theta_hat_[i]:.4f}")
    return reg_raw, reg_cop, (r2_train_raw, r2_test_raw, r2_train_cop, r2_test_cop)


if __name__ == "__main__":
    data = load_data()
    run(data, target_kind="return")
    reg_raw_v, reg_cop_v, _ = run(data, target_kind="volatility")
    print(f"\n{reg_cop_v.summary()}")
