"""Real variable selection (not just OLS) for explaining AAPL's returns: a broader
factor universe, a genuine sector-relevance prior (GICS-classification based, not
fit to the data -- the finance analogue of a MalaCards gene-relevance score), and
the Gaussian spike-and-slab engine (EM filter + Gibbs) instead of plain regression.

Mirrors the GWAS "informative vs non-informative" comparison directly: does knowing
that a factor is in AAPL's own sector actually help identify which factors matter,
or does an uninformative/flat prior do just as well?
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
from hierboost.spike_slab_gaussian import em_filter_gaussian, gibbs_sampler_gaussian
from hierboost.spike_slab import centroid_estimate

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

TARGET = "AAPL"
PREDICTORS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "VGT", "SMH", "SOXX", "IGV", "SKYY",
              "XLF", "KRE", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
              "XLC", "VUG", "VTV", "MTUM", "TLT", "IEF", "SHY", "GLD", "USO", "EFA", "EEM"]
RHO = 0.9

# GICS-informed relevance judgment, made BEFORE looking at correlations or fitting
# anything -- AAPL sits in Technology Hardware; software/cloud and mega-cap-tech
# adjacent communication names are the next closest; broad equity market and
# consumer-discretionary get a mild bump; everything else (other sectors, bonds,
# commodities, non-US equity) is treated as a priori unrelated.
RELEVANCE = {
    "SPY": 5, "QQQ": 5, "XLK": 5, "VGT": 5, "SMH": 5, "SOXX": 5, "VUG": 5, "MTUM": 5,
    "IGV": 3, "SKYY": 3, "XLC": 3,
    "XLY": 1.5, "DIA": 1.5, "VTV": 1.5,
    "IWM": 0.5, "XLF": 0.5, "KRE": 0.5, "XLE": 0.5, "XLV": 0.5, "XLP": 0.5,
    "XLI": 0.5, "XLB": 0.5, "XLU": 0.5, "XLRE": 0.5,
    "EFA": 0.5, "EEM": 0.5,
    "TLT": 0.05, "IEF": 0.05, "SHY": 0.05, "GLD": 0.05, "USO": 0.05,
}


def zscore(df):
    return (df - df.mean()) / df.std()


def load_returns():
    df = yf.download([TARGET] + PREDICTORS, period="3y", progress=False, auto_adjust=True)["Close"]
    rets = np.log(df).diff().dropna()
    y_raw = rets[TARGET].values
    # Standardize the response too, not just the predictors: the spike-and-slab prior's
    # variance-scale hyperparameters (nu, lam) are naturally O(1), and daily-return betas
    # against unit-variance factors are naturally O(0.01) -- fitting on raw returns lets
    # sigma_g2 converge to a scale where every real beta looks indistinguishable from
    # noise. Standardizing puts beta on a "SDs of AAPL per SD of factor" scale instead,
    # which behaves the same way the logistic-regression log-odds scale did for GWAS.
    y = (y_raw - y_raw.mean()) / y_raw.std()
    return y, rets[PREDICTORS]


def build_blocks_and_relevance(Xp_df, rho):
    labels = blocks_from_correlation_threshold(Xp_df.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = Xp_df.columns.to_numpy()

    factors, names, block_relevance, members_report = {}, [], [], {}
    for b, idx in membership.items():
        members = cols[idx]
        Xb = zscore(Xp_df[members]).values
        rel = float(np.mean([RELEVANCE[m] for m in members]))
        if len(members) == 1:
            fname = members[0]
            factors[fname] = Xb[:, 0]
        else:
            fname = "+".join(members) + "_factor"
            score, _ = gaussian_block_factor(Xb)
            factors[fname] = score
        names.append(fname)
        block_relevance.append(rel)
        members_report[fname] = list(members)

    Z = pd.DataFrame(factors, index=Xp_df.index)[names]
    wr = np.array(block_relevance)
    wr = wr / wr.max()
    return Z, wr, members_report


if __name__ == "__main__":
    y, Xp_raw = load_returns()
    print(f"{len(y)} trading days, target={TARGET}, {len(PREDICTORS)} candidate predictors")

    Z, wr, members_report = build_blocks_and_relevance(Xp_raw, rho=RHO)
    K = Z.shape[1]
    print(f"\n{K} blocks at rho={RHO}:")
    for name, rel in zip(Z.columns, wr):
        print(f"  {name:45s}  relevance={rel:.2f}  members={members_report[name]}")

    X_design = np.column_stack([np.ones(len(y)), Z.values])
    xi0 = np.log(4 / K) - np.log(1 - 4 / K)
    xi1_boost, xi1_null = 4.0, 0.0

    print(f"\nxi0={xi0:.2f}")
    filt_boost = em_filter_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1_boost, kappa=100.0,
                                     nu=1.0, lam=1.0, filter_frac=0.2, min_features=5,
                                     max_outer=30, verbose=False)
    filt_null = em_filter_gaussian(X_design, y, np.zeros_like(wr), xi0=xi0, xi1=xi1_null,
                                    kappa=100.0, nu=1.0, lam=1.0, filter_frac=0.2,
                                    min_features=5, max_outer=30, verbose=False)

    theta_first_boost = filt_boost.history[0].theta_hat
    theta_first_null = filt_null.history[0].theta_hat

    print(f"\nEM filter (informative prior): retained {filt_boost.best.n_features} at best step, "
          f"members: {[Z.columns[i] for i in filt_boost.best.retained_idx]}")
    print(f"EM filter (flat prior):        retained {filt_null.best.n_features} at best step, "
          f"members: {[Z.columns[i] for i in filt_null.best.retained_idx]}")

    print("\ntheta_hat at first EM step, informative vs flat prior, sorted by informative:")
    order = np.argsort(-theta_first_boost)
    for i in order:
        print(f"  {Z.columns[i]:45s}  relevance={wr[i]:.2f}  "
              f"theta_informative={theta_first_boost[i]:.4f}  theta_flat={theta_first_null[i]:.4f}")

    gr = gibbs_sampler_gaussian(X_design, y, wr, xi0=xi0, xi1=xi1_boost, kappa=100.0,
                                 nu=1.0, lam=1.0, n_samples=4000, burn_in=1500, seed=0)
    print(f"\nGibbs posterior sigma_y2 mean: {gr.sigma_y2.mean():.5f} "
          f"(naive residual variance: {np.var(y - X_design @ filt_boost.history[0].beta):.5f})")

    print("\nposterior P(theta=1|y) from Gibbs, sorted:")
    order = np.argsort(-gr.pi_hat)
    for i in order:
        print(f"  {Z.columns[i]:45s}  relevance={wr[i]:.2f}  pi_hat={gr.pi_hat[i]:.4f}")

    for gamma in (0.5, 1.0, 2.0, 5.0):
        sel = centroid_estimate(gr.pi_hat, gamma)
        print(f"gamma={gamma}: selected {[Z.columns[i] for i in np.where(sel)[0]]}")

    # --- Figure: posterior inclusion probability, informative vs flat, colored by relevance tier ---
    fig, ax = plt.subplots(figsize=(9, 8))
    order = np.argsort(-wr)
    y_pos = np.arange(K)
    tier_color = np.where(wr[order] >= 0.9, "tab:red",
                  np.where(wr[order] >= 0.5, "tab:orange",
                  np.where(wr[order] >= 0.25, "tab:blue", "tab:gray")))
    ax.barh(y_pos - 0.2, theta_first_boost[order], height=0.4, color=tier_color, alpha=0.9,
            label="informative prior (sector relevance)")
    ax.barh(y_pos + 0.2, theta_first_null[order], height=0.4, color=tier_color, alpha=0.35,
            label="flat prior (no relevance boost)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([Z.columns[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P(theta=1) at first EM step")
    ax.set_title(f"{TARGET}: which factors matter, with vs without a sector-relevance prior\n"
                 "(red=high relevance, orange=medium, blue=low, gray=near-zero)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_factor_selection.png"), dpi=140)
    plt.close(fig)

    print(f"\nFigure written to {OUT}/")
