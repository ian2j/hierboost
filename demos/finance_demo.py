"""Chapter 4's block-latent idea on real market data: decomposes a stock's returns
against a set of collinear sector/index ETFs, comparing raw regression, ridge, and
block-factor regression."""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

TARGET = "AAPL"
PREDICTORS = ["SPY", "QQQ", "XLK", "VGT", "SMH", "XLF", "KRE", "XLE", "XLV", "XLY"]
RHO = 0.75
N_BOOT = 1000
SEED = 0


def load_returns():
    df = yf.download([TARGET] + PREDICTORS, period="3y", progress=False, auto_adjust=True)["Close"]
    rets = np.log(df).diff().dropna()
    return rets[TARGET].values, rets[PREDICTORS]


def zscore(df):
    return (df - df.mean()) / df.std()


def build_block_factors(Xp_df, rho):
    labels = blocks_from_correlation_threshold(Xp_df.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = Xp_df.columns.to_numpy()

    factors, names, loadings_report = {}, [], {}
    for b, idx in membership.items():
        members = cols[idx]
        Xb = zscore(Xp_df[members]).values
        if len(members) == 1:
            factors[members[0]] = Xb[:, 0]
            names.append(members[0])
            loadings_report[members[0]] = {members[0]: 1.0}
        else:
            score, loadings = gaussian_block_factor(Xb)
            fname = "+".join(members) + "_factor"
            factors[fname] = score
            names.append(fname)
            loadings_report[fname] = dict(zip(members, loadings))
    return pd.DataFrame(factors, index=Xp_df.index)[names], loadings_report


def ols_fit(y, X_df):
    X = sm.add_constant(X_df)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model


def bootstrap_sd(y, X_df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    X = sm.add_constant(X_df).values
    coefs = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        coefs[b] = beta
    return coefs[:, 1:].std(axis=0)  # drop intercept


def half_split_coef_shift(y, X_df):
    """How much do the fitted loadings themselves change between the first and second
    half of the sample? A direct, second measure of estimation instability alongside
    the bootstrap SD (contemporaneous factor loadings aren't a forecasting exercise,
    so a train/test predictive R^2 isn't the right stability check here)."""
    n = len(y)
    cut = n // 2
    X = sm.add_constant(X_df).values
    beta1, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
    beta2, *_ = np.linalg.lstsq(X[cut:], y[cut:], rcond=None)
    return beta1[1:], beta2[1:]


if __name__ == "__main__":
    y, Xp_raw = load_returns()
    print(f"{len(y)} trading days, target={TARGET}, predictors={PREDICTORS}")

    corr = Xp_raw.corr()
    print("\nPredictor correlation matrix:\n", corr.round(2))

    Xp_z = zscore(Xp_raw)
    factors_df, loadings_report = build_block_factors(Xp_raw, rho=RHO)
    print(f"\nBlocks at rho={RHO}:")
    for fname, load in loadings_report.items():
        print(f"  {fname}: {load}")

    model_raw = ols_fit(y, Xp_z)
    model_factor = ols_fit(y, factors_df)
    sd_raw = bootstrap_sd(y, Xp_z)
    sd_factor = bootstrap_sd(y, factors_df)

    r2_raw_in = model_raw.rsquared
    r2_factor_in = model_factor.rsquared
    r2_raw_adj = model_raw.rsquared_adj
    r2_factor_adj = model_factor.rsquared_adj

    print(f"\nR^2 (10 raw predictors):    raw={r2_raw_in:.3f}   adjusted={r2_raw_adj:.3f}")
    print(f"R^2 (4 block factors):      raw={r2_factor_in:.3f}   adjusted={r2_factor_adj:.3f}")

    beta1_raw, beta2_raw = half_split_coef_shift(y, Xp_z)
    beta1_fac, beta2_fac = half_split_coef_shift(y, factors_df)
    print("\nCoefficient shift, first half of sample -> second half (raw OLS):")
    for name, b1, b2 in zip(Xp_z.columns, beta1_raw, beta2_raw):
        print(f"  {name:6s}  {b1:+.4f} -> {b2:+.4f}")
    print("Coefficient shift, first half of sample -> second half (block-factor OLS):")
    for name, b1, b2 in zip(factors_df.columns, beta1_fac, beta2_fac):
        print(f"  {name:35s}  {b1:+.4f} -> {b2:+.4f}")

    print("\nRaw OLS coefficients (on standardized predictors), with bootstrap SD:")
    for name, coef, sd in zip(Xp_z.columns, model_raw.params.values[1:], sd_raw):
        print(f"  {name:6s}  coef={coef:+.4f}  bootstrap_sd={sd:.4f}")

    print("\nBlock-factor OLS coefficients, with bootstrap SD:")
    for name, coef, sd in zip(factors_df.columns, model_factor.params.values[1:], sd_factor):
        print(f"  {name:35s}  coef={coef:+.4f}  bootstrap_sd={sd:.4f}")

    # --- Figure 1: correlation heatmap ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PREDICTORS))); ax.set_xticklabels(PREDICTORS, rotation=45, ha="right")
    ax.set_yticks(range(len(PREDICTORS))); ax.set_yticklabels(PREDICTORS)
    for i in range(len(PREDICTORS)):
        for j in range(len(PREDICTORS)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Predictor ETF return correlations (3y daily)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_corr_heatmap.png"), dpi=140)
    plt.close(fig)

    # --- Figure 2: coefficient instability, raw vs block-factor ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    x = np.arange(len(Xp_z.columns))
    ax.bar(x, model_raw.params.values[1:], yerr=sd_raw, capsize=4, color="tab:red", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(Xp_z.columns, rotation=45, ha="right")
    ax.set_ylabel(f"coefficient on {TARGET} return (per 1 SD move)")
    ax.set_title("Raw OLS on all 10 correlated ETFs\n(error bars = bootstrap SD)")

    ax2 = axes[1]
    xf = np.arange(len(factors_df.columns))
    short_names = [n.replace("_factor", "") if "_factor" in n else n for n in factors_df.columns]
    ax2.bar(xf, model_factor.params.values[1:], yerr=sd_factor, capsize=4, color="teal", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(xf); ax2.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel(f"coefficient on {TARGET} return (per 1 SD move)")
    ax2.set_title("OLS on block factors (Chapter-4 style)\n(error bars = bootstrap SD)")
    fig.suptitle(f"{TARGET}: raw regression vs block-factor regression on the same information")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_coef_comparison.png"), dpi=140)
    plt.close(fig)

    # --- Figure 3: coefficient shift between first and second half of the sample ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    ax = axes[0]
    for i, name in enumerate(Xp_z.columns):
        ax.plot([0, 1], [beta1_raw[i], beta2_raw[i]], marker="o", color="tab:red", alpha=0.8)
        ax.annotate(name, (1.02, beta2_raw[i]), fontsize=8, va="center")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["first half\nof sample", "second half\nof sample"])
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylabel(f"coefficient on {TARGET} return")
    ax.set_title("Raw OLS: loadings estimated separately\non each half of the sample")

    ax2 = axes[1]
    for i, name in enumerate(factors_df.columns):
        short = name.replace("_factor", "")
        ax2.plot([0, 1], [beta1_fac[i], beta2_fac[i]], marker="o", color="teal", alpha=0.85)
        ax2.annotate(short, (1.02, beta2_fac[i]), fontsize=8, va="center")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["first half\nof sample", "second half\nof sample"])
    ax2.set_xlim(-0.15, 1.55)
    ax2.set_ylabel(f"coefficient on {TARGET} return")
    ax2.set_title("Block-factor OLS: loadings estimated\nseparately on each half")
    fig.suptitle(f"{TARGET}: how much do the fitted loadings move between two halves of the same sample?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_stability_over_time.png"), dpi=140)
    plt.close(fig)

    print(f"\nFigures written to {OUT}/")
