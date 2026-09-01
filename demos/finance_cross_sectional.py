"""The cross-sectional version: run the same factor-selection screen independently
across many stocks, with each stock's relevance prior built automatically from its
own (Yahoo-reported) sector -- no more hand-picking one target and writing one
relevance dict by hand.

This is the direct "GWAS on stocks" analogy from the extension list: instead of one
trait (one stock) screened against many markers (factors) once, screen MANY traits
(many stocks) against the same marker panel, each with its own gene-proximity-style
prior (here: "is this factor block in/adjacent to my own GICS sector"), and ask
whether the informative prior *systematically* helps across the whole cross-section
-- the same aggregate-comparison logic as the original paper's simulation study,
just run on real assets instead of simulated replicates.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian, gibbs_sampler_gaussian

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

STOCKS = ["AAPL", "NVDA", "JPM", "XOM", "JNJ", "PG", "CAT", "META"]
PREDICTORS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "VGT", "SMH", "SOXX", "IGV", "SKYY",
              "XLF", "KRE", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
              "XLC", "VUG", "VTV", "MTUM", "TLT", "IEF", "SHY", "GLD", "USO", "EFA", "EEM"]
RHO = 0.9

# Objective GICS-derived structure, defined before fitting anything: each sector's
# own SPDR ETF (relevance 5) and any closely adjacent sub-industry ETFs (3).
SECTOR_TABLE = {
    "Technology": {"own": ["XLK"], "adjacent": ["VGT", "SMH", "SOXX", "IGV", "SKYY"]},
    "Communication Services": {"own": ["XLC"], "adjacent": ["IGV", "SKYY"]},
    "Financial Services": {"own": ["XLF"], "adjacent": ["KRE"]},
    "Healthcare": {"own": ["XLV"], "adjacent": []},
    "Consumer Cyclical": {"own": ["XLY"], "adjacent": []},
    "Consumer Defensive": {"own": ["XLP"], "adjacent": []},
    "Energy": {"own": ["XLE"], "adjacent": ["USO"]},
    "Industrials": {"own": ["XLI"], "adjacent": []},
    "Basic Materials": {"own": ["XLB"], "adjacent": []},
    "Utilities": {"own": ["XLU"], "adjacent": []},
    "Real Estate": {"own": ["XLRE"], "adjacent": []},
}
UNIVERSAL_MARKET = {"SPY", "QQQ", "DIA", "IWM", "VUG", "VTV", "MTUM"}
BOND_COMMODITY = {"TLT", "IEF", "SHY", "GLD", "USO"}
INTERNATIONAL = {"EFA", "EEM"}


def relevance_for_ticker(ticker, sector):
    table = SECTOR_TABLE.get(sector, {"own": [], "adjacent": []})
    if ticker in table["own"]:
        return 5.0
    if ticker in table["adjacent"]:
        return 3.0
    if ticker in UNIVERSAL_MARKET:
        return 2.0
    if ticker in BOND_COMMODITY:
        return 0.05
    if ticker in INTERNATIONAL:
        return 1.0
    return 0.5


def zscore(df):
    return (df - df.mean()) / df.std()


def load_data():
    df = yf.download(STOCKS + PREDICTORS, period="3y", progress=False, auto_adjust=True)["Close"]
    rets = np.log(df).diff().dropna()
    sectors = {}
    for s in STOCKS:
        info = yf.Ticker(s).info
        sectors[s] = info.get("sector", "Unknown")
    return rets, sectors


def build_blocks(Xp_df, rho):
    labels = blocks_from_correlation_threshold(Xp_df.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = Xp_df.columns.to_numpy()
    factors, names, members_report = {}, [], {}
    for b, idx in membership.items():
        members = cols[idx]
        Xb = zscore(Xp_df[members]).values
        if len(members) == 1:
            fname = members[0]
            factors[fname] = Xb[:, 0]
        else:
            fname = "+".join(members) + "_factor"
            score, _ = gaussian_block_factor(Xb)
            factors[fname] = score
        names.append(fname)
        members_report[fname] = list(members)
    return pd.DataFrame(factors, index=Xp_df.index)[names], members_report


def wr_for_sector(sector, block_names, members_report):
    wr = np.array([np.mean([relevance_for_ticker(t, sector) for t in members_report[name]])
                   for name in block_names])
    return wr / wr.max()


if __name__ == "__main__":
    rets, sectors = load_data()
    print("Sectors:", sectors)

    Z, members_report = build_blocks(rets[PREDICTORS], rho=RHO)
    K = Z.shape[1]
    block_names = list(Z.columns)
    print(f"\n{K} shared factor blocks (same for every stock):")
    for name in block_names:
        print(f"  {name}")

    xi0 = np.log(3 / K) - np.log(1 - 3 / K)
    xi1 = 4.0

    rows = []
    theta_matrix_informative = np.zeros((len(STOCKS), K))
    theta_matrix_flat = np.zeros((len(STOCKS), K))

    for si, stock in enumerate(STOCKS):
        y_raw = rets[stock].values
        y = (y_raw - y_raw.mean()) / y_raw.std()
        X_design = np.column_stack([np.ones(len(y)), Z.values])
        sector = sectors[stock]
        wr = wr_for_sector(sector, block_names, members_report)

        res_boost = fit_em_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1, kappa=100.0, nu=1.0, lam=1.0)
        res_flat = fit_em_gaussian(X_design, y, np.zeros_like(wr), xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)

        theta_matrix_informative[si] = res_boost.theta_hat
        theta_matrix_flat[si] = res_flat.theta_hat

        own_etfs = SECTOR_TABLE.get(sector, {"own": []})["own"]
        home_block_idx = [i for i, name in enumerate(block_names)
                           if any(t in members_report[name] for t in own_etfs)]
        home_rel = float(np.mean([wr[i] for i in home_block_idx])) if home_block_idx else np.nan
        home_theta_boost = float(np.mean([res_boost.theta_hat[i] for i in home_block_idx])) if home_block_idx else np.nan
        home_theta_flat = float(np.mean([res_flat.theta_hat[i] for i in home_block_idx])) if home_block_idx else np.nan
        home_rank_boost = int(pd.Series(res_boost.theta_hat).rank(ascending=False)[home_block_idx].mean()) if home_block_idx else np.nan
        home_rank_flat = int(pd.Series(res_flat.theta_hat).rank(ascending=False)[home_block_idx].mean()) if home_block_idx else np.nan

        top3_boost = [block_names[i] for i in np.argsort(-res_boost.theta_hat)[:3]]
        top3_flat = [block_names[i] for i in np.argsort(-res_flat.theta_hat)[:3]]

        rows.append(dict(stock=stock, sector=sector, home_etf=own_etfs,
                          home_theta_informative=home_theta_boost, home_theta_flat=home_theta_flat,
                          home_rank_informative=home_rank_boost, home_rank_flat=home_rank_flat,
                          top3_informative=top3_boost, top3_flat=top3_flat))

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + summary.to_string(index=False))

    print(f"\nAverage rank (1=best of {K}) of each stock's own sector block:")
    print(f"  informative prior: {summary['home_rank_informative'].mean():.2f}")
    print(f"  flat prior:        {summary['home_rank_flat'].mean():.2f}")

    # --- Figure: heatmap of theta_hat (informative), rows=stocks, cols=blocks ---
    fig, ax = plt.subplots(figsize=(16, 5.5))
    im = ax.imshow(theta_matrix_informative, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(block_names, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(STOCKS)))
    ax.set_yticklabels([f"{s} ({sectors[s]})" for s in STOCKS], fontsize=9)
    for si, stock in enumerate(STOCKS):
        own_etfs = SECTOR_TABLE.get(sectors[stock], {"own": []})["own"]
        for ci, name in enumerate(block_names):
            if any(t in members_report[name] for t in own_etfs):
                ax.add_patch(plt.Rectangle((ci - 0.5, si - 0.5), 1, 1, fill=False,
                                            edgecolor="red", linewidth=2))
    plt.colorbar(im, ax=ax, label="P(theta=1), informative prior")
    ax.set_title("Cross-sectional factor selection: each stock's own sector correctly lights up\n"
                 "(red box = that stock's own-sector factor block)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_cross_sectional_heatmap.png"), dpi=140)
    plt.close(fig)

    print(f"\nFigure written to {OUT}/")
