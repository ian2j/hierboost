"""The temporal counterpart of finance_demo.py: instead of decorrelating several
cross-sectionally collinear ETFs at one point in time (Chapter 4's original axis), this
decorrelates several *rolling-window* return features of one predictor through time --
5/10/20/60-day momentum windows are exactly as collinear as LD-correlated SNPs, for the
same underlying reason: they're all noisy, overlapping views of one evolving momentum
state. hierboost.state_space.fit_temporal_block_factor replaces each predictor's four
window-return columns with one AR(1)-smoothed trend latent, closed-form (EM + Kalman/RTS
smoother, no autodiff) since returns are already continuous -- see
hierboost.latent's structure="ar1" for the discrete/binomial analogue.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierboost.state_space import fit_temporal_block_factor

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

TARGET = "AAPL"
PREDICTORS = ["SPY", "QQQ", "XLK"]
WINDOWS = [5, 10, 20, 60]
N_BOOT = 1000
SEED = 0


def load_log_returns():
    df = yf.download([TARGET] + PREDICTORS, period="5y", progress=False, auto_adjust=True)["Close"]
    return np.log(df).diff().dropna()


def rolling_window_features(ret, windows):
    """Backward-looking cumulative return over each window, at every day -- uses only
    information available as of that day, same causal discipline the M6 backtest work
    requires even though this demo isn't itself a forecasting exercise."""
    return pd.DataFrame({f"{w}d": ret.rolling(w).sum() for w in windows}).dropna()


def zscore(df):
    return (df - df.mean()) / df.std()


def ols_fit(y, X_df):
    return sm.OLS(y, sm.add_constant(X_df)).fit(cov_type="HC1")


def bootstrap_sd(y, X_df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    X = sm.add_constant(X_df).values
    coefs = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        coefs[b] = beta
    return coefs[:, 1:].std(axis=0)


if __name__ == "__main__":
    rets = load_log_returns()
    print(f"{len(rets)} trading days, target={TARGET}, predictors={PREDICTORS}, windows={WINDOWS}")

    raw_cols, latent_cols, fitted_rho, latent_var = {}, {}, {}, {}
    common_index = None
    for p in PREDICTORS:
        feats = rolling_window_features(rets[p], WINDOWS)
        common_index = feats.index if common_index is None else common_index.intersection(feats.index)

    for p in PREDICTORS:
        feats = rolling_window_features(rets[p], WINDOWS).loc[common_index]
        feats_z = zscore(feats)
        for w in WINDOWS:
            raw_cols[f"{p}_{w}d"] = feats_z[f"{w}d"]

        res = fit_temporal_block_factor(feats_z.values)
        latent_cols[f"{p}_trend"] = res.z
        latent_var[p] = res.z_var
        fitted_rho[p] = res.rho
        print(f"  {p}: fitted AR(1) trend persistence rho={res.rho:.3f}, "
              f"loadings(5/10/20/60d)={np.round(res.loadings, 2)}")

    y = rets[TARGET].loc[common_index].values
    raw_df = pd.DataFrame(raw_cols, index=common_index)
    latent_df = pd.DataFrame(latent_cols, index=common_index)

    corr = raw_df.corr()
    print("\nRaw rolling-window feature correlation matrix:\n", corr.round(2))

    model_raw = ols_fit(y, raw_df)
    model_latent = ols_fit(y, latent_df)
    sd_raw = bootstrap_sd(y, raw_df)
    sd_latent = bootstrap_sd(y, latent_df)

    print(f"\nR^2 ({raw_df.shape[1]} raw window-return features): "
          f"raw={model_raw.rsquared:.3f}  adjusted={model_raw.rsquared_adj:.3f}")
    print(f"R^2 ({latent_df.shape[1]} AR(1) trend latents):        "
          f"raw={model_latent.rsquared:.3f}  adjusted={model_latent.rsquared_adj:.3f}")

    print(f"\nRaw OLS coefficients on {TARGET} return, with bootstrap SD:")
    for name, coef, sd in zip(raw_df.columns, model_raw.params.values[1:], sd_raw):
        print(f"  {name:10s}  coef={coef:+.4f}  bootstrap_sd={sd:.4f}")

    print(f"\nAR(1) trend-latent OLS coefficients on {TARGET} return, with bootstrap SD:")
    for name, coef, sd in zip(latent_df.columns, model_latent.params.values[1:], sd_latent):
        print(f"  {name:10s}  coef={coef:+.4f}  bootstrap_sd={sd:.4f}")

    # --- Figure 1: within-predictor window-return correlation heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Rolling-window return correlations, by predictor\n(each 4x4 block: same predictor, different lookback windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "temporal_corr_heatmap.png"), dpi=140)
    plt.close(fig)

    # --- Figure 2: coefficient instability, raw window returns vs AR(1) trend latents ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    x = np.arange(len(raw_df.columns))
    ax.bar(x, model_raw.params.values[1:], yerr=sd_raw, capsize=4, color="tab:red", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(raw_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"coefficient on {TARGET} return")
    ax.set_title(f"Raw OLS on {raw_df.shape[1]} collinear window-return features\n(error bars = bootstrap SD)")

    ax2 = axes[1]
    xl = np.arange(len(latent_df.columns))
    ax2.bar(xl, model_latent.params.values[1:], yerr=sd_latent, capsize=4, color="teal", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(xl); ax2.set_xticklabels(latent_df.columns, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel(f"coefficient on {TARGET} return")
    ax2.set_title("OLS on AR(1) trend latents (temporal Ch4)\n(error bars = bootstrap SD)")
    fig.suptitle(f"{TARGET}: raw window-return regression vs temporal block-factor regression")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "temporal_coef_comparison.png"), dpi=140)
    plt.close(fig)

    # --- Figure 3: the smoothed AR(1) trend trajectories themselves ---
    fig, axes = plt.subplots(len(PREDICTORS), 1, figsize=(11, 3 * len(PREDICTORS)), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), PREDICTORS):
        z = latent_df[f"{p}_trend"].values
        sd = np.sqrt(latent_var[p])
        t = np.arange(len(z))
        ax.fill_between(t, z - 2 * sd, z + 2 * sd, color="teal", alpha=0.2, label="±2 SD")
        ax.plot(t, z, color="teal", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{p}: smoothed momentum trend state (fitted rho={fitted_rho[p]:.3f})")
        ax.set_ylabel("trend (z-units)")
    axes_arr = np.atleast_1d(axes)
    axes_arr[-1].set_xlabel("trading day")
    axes_arr[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"AR(1)-smoothed momentum states extracted from {WINDOWS}-day rolling returns")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "temporal_trend_states.png"), dpi=140)
    plt.close(fig)

    print(f"\nFigures written to {OUT}/")
