"""Does a fitted block latent generalize, or is it glued to the one regression it was
estimated in?

Every finance/genomics/weather demo in this codebase so far fits a block latent Z on
some predictor panel and evaluates it ONLY against the single target it was built to
predict (finance_factor_selection.py: AAPL return only; finance_cross_sectional.py:
fits a fresh set of blocks -- reused across stocks, but never tests those blocks'
predictive latents out-of-sample on a NEW downstream task). Nobody has asked whether
gaussian_block_factor's closed-form PPCA compression of a predictor universe is a
reusable, general-purpose *representation* of that universe, or just a one-off
artifact tied to the regression it happened to be fit alongside.

This script tests that directly, with a real held-out split (no shuffling -- this is
a time series) and three transfer scenarios sharing one frozen block latent:

  (a) same target (AAPL return), held-out PERIOD          -- the ordinary OOS check
  (b) same stock, DIFFERENT target variable (AAPL realized volatility instead of
      return) -- reusing the compressed representation for a new downstream task
  (c) DIFFERENT target stock (JPM, a different GICS sector than AAPL) -- reusing the
      representation for an entirely different regression problem

For each, three arms:
  - frozen transfer   : project_block_factor'd version of the block latent fit once
                        on AAPL/FIT (blocks + loadings never re-estimated)
  - raw features      : skip the latent, regress directly on the 31 zscored raw ETFs
  - task-specific refit: what you'd get re-running gaussian_block_factor from scratch,
                        "as if" you'd never reused anything

A structural point falls out of running this (spoiler, see bottom of __main__):
blocks_from_correlation_threshold and gaussian_block_factor are both pure functions of
the PREDICTOR panel -- neither one ever looks at y. So "refit from scratch on the same
FIT-window ETF panel" is not an independent estimate at all: it is bit-identical to the
"frozen" latent already in hand, for every scenario, by construction. That's a real
finding about this pipeline's design, not a coding shortcut -- see the interpretation
in __main__ for what it implies about what a genuinely target-aware "refit ceiling"
would require (target-aware block MEMBERSHIP, not just re-estimated loadings).

House-style notes carried over from finance_cross_sectional.py / finance_factor_selection.py:
same 31-ETF predictor universe, same rho=0.9 correlation-threshold blocking, same
em_filter_gaussian spike-and-slab engine, same period="3y" yfinance window. Departures,
specific to this experiment: simple pct_change returns (not log returns) and a flat/
uninformative prior throughout (xi1=0) -- the sector-relevance prior is a different,
already-tested axis (see finance_factor_selection.py); mixing it in here would
confound "does the LATENT transfer" with "does the PRIOR transfer", which is a
separate question.
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
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

TARGET_A = "AAPL"   # Technology -- the stock the block latent is fit/validated on
TARGET_B = "JPM"    # Financial Services -- different-sector transfer target, scenario (c)
PREDICTORS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "VGT", "SMH", "SOXX", "IGV", "SKYY",
              "XLF", "KRE", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
              "XLC", "VUG", "VTV", "MTUM", "TLT", "IEF", "SHY", "GLD", "USO", "EFA", "EEM"]
RHO = 0.9          # same correlation-threshold cutoff as finance_cross_sectional.py
VOL_WINDOW = 20    # trailing 1-month (in trading days) rolling std of returns --
                   # the standard, simplest realized-volatility proxy; nothing fancier
                   # (no EWMA, no Garman-Klass) since the point here is transfer, not
                   # building a better vol estimator
FIT_FRAC = 0.7     # chronological 70/30 split, no shuffling
XI_EXPECTED_ACTIVE = 4.0  # weakly-informative "expect ~4 active predictors" prior,
                          # same style as finance_factor_selection.py's flat-prior arm
KAPPA, NU, LAM = 100.0, 1.0, 1.0
FILTER_KW = dict(filter_frac=0.2, min_features=5, max_outer=30)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    tickers = sorted(set([TARGET_A, TARGET_B] + PREDICTORS))
    prices = yf.download(tickers, period="3y", progress=False, auto_adjust=True)["Close"]
    prices = prices.dropna()
    rets = prices.pct_change().dropna()
    vol_A = rets[TARGET_A].rolling(VOL_WINDOW).std()
    # Trim to where both plain returns AND the rolling-vol feature are defined --
    # the rolling window eats VOL_WINDOW-1 extra rows at the start.
    common_idx = rets.index.intersection(vol_A.dropna().index)
    return rets.loc[common_idx], vol_A.loc[common_idx]


def chronological_split(index, frac=FIT_FRAC):
    cut = int(len(index) * frac)
    return index[:cut], index[cut:]


def zscore_apply(x, mean, std):
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Frozen block latent: fit ONLY on the FIT window, apply via project_block_factor
# ---------------------------------------------------------------------------

def fit_block_latent(Xp_raw_fit, rho):
    """Fit block membership (correlation threshold) + per-block standardization stats
    + per-block PPCA loadings, all from Xp_raw_fit alone. Returns (transform, Z_fit)
    where `transform` carries everything needed to reproduce this exact latent on new
    data via apply_block_latent -- nothing here or in project_block_factor ever looks
    at a target y.
    """
    fit_mean = Xp_raw_fit.mean()
    fit_std = Xp_raw_fit.std()
    Xp_z_fit = zscore_apply(Xp_raw_fit, fit_mean, fit_std)

    labels = blocks_from_correlation_threshold(Xp_z_fit.values, rho=rho)
    membership = block_membership_lists(labels)
    cols = Xp_raw_fit.columns.to_numpy()

    blocks, factor_cols, names = {}, {}, []
    for b, idx in membership.items():
        members = cols[idx]
        Xb = Xp_z_fit[members].values
        if len(members) == 1:
            fname = members[0]
            factor_cols[fname] = Xb[:, 0]
            blocks[fname] = dict(members=list(members), loadings=None, train_mean=None)
        else:
            fname = "+".join(members) + "_factor"
            score, loadings = gaussian_block_factor(Xb)
            blocks[fname] = dict(members=list(members), loadings=loadings,
                                  train_mean=Xb.mean(axis=0))
            factor_cols[fname] = score
        names.append(fname)

    Z_fit = pd.DataFrame(factor_cols, index=Xp_raw_fit.index)[names]
    transform = dict(fit_mean=fit_mean, fit_std=fit_std, blocks=blocks, names=names)
    return transform, Z_fit


def apply_block_latent(transform, Xp_raw_new):
    """Apply an ALREADY-FITTED transform (from fit_block_latent) to new raw predictor
    data. Every block's factor score comes from project_block_factor (or a plain
    zscore for singleton blocks) -- nothing is re-estimated here."""
    Xp_z_new = zscore_apply(Xp_raw_new, transform["fit_mean"], transform["fit_std"])
    cols = {}
    for fname in transform["names"]:
        info = transform["blocks"][fname]
        Xb = Xp_z_new[info["members"]].values
        if info["loadings"] is None:
            cols[fname] = Xb[:, 0]
        else:
            cols[fname] = project_block_factor(Xb, info["loadings"], info["train_mean"])
    return pd.DataFrame(cols, index=Xp_raw_new.index)[transform["names"]]


# ---------------------------------------------------------------------------
# Outcome regression (spike-and-slab, flat/uninformative prior) on top of whatever
# feature set (block latent OR raw ETFs) is handed in -- generic over both.
# ---------------------------------------------------------------------------

def fit_outcome(X_fit_df, y_fit_raw):
    """Standardize y on the FIT window only, fit em_filter_gaussian with a flat prior
    (wr=0, xi1=0 -- we're testing latent transfer, not the informative-prior mechanism
    already covered by finance_factor_selection.py / finance_cross_sectional.py)."""
    y_mean, y_std = float(y_fit_raw.mean()), float(y_fit_raw.std())
    y_z = ((y_fit_raw - y_mean) / y_std).values
    p = X_fit_df.shape[1]
    X_design = np.column_stack([np.ones(len(y_z)), X_fit_df.values])
    wr = np.zeros(p)
    k_expected = min(XI_EXPECTED_ACTIVE, p - 0.5)
    xi0 = np.log(k_expected / p) - np.log(1 - k_expected / p)
    filt = em_filter_gaussian(X_design, y_z, wr, xi0=xi0, xi1=0.0,
                               kappa=KAPPA, nu=NU, lam=LAM, **FILTER_KW)
    return filt, y_mean, y_std


def predict(filt, y_mean, y_std, X_df):
    idx = filt.best.retained_idx
    beta = filt.best.beta
    X = np.column_stack([np.ones(len(X_df)), X_df.values[:, idx]])
    return X @ beta * y_std + y_mean


def oos_r2(y_true_raw, y_pred_raw, fit_mean_raw):
    """Out-of-sample R^2 against a FIT-window mean baseline (Campbell-Thompson style:
    the "naive" benchmark is only allowed to know the FIT window, same as every model
    here) -- not the held-out window's own mean, which would leak held-out information
    into the benchmark itself."""
    ss_res = np.sum((y_true_raw - y_pred_raw) ** 2)
    ss_tot = np.sum((y_true_raw - fit_mean_raw) ** 2)
    return 1.0 - ss_res / ss_tot


def safe_corr(y_true, y_pred):
    if np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_scenario(label, y_fit_raw, y_held_raw, Z_fit, Z_held, Xp_z_fit, Xp_z_held):
    """Frozen-transfer and raw-feature arms for one (target, window) pair. The
    task-specific-refit arm is NOT recomputed here -- see __main__ for why it is
    provably identical to the frozen arm in this pipeline."""
    filt_f, ymf, ysf = fit_outcome(Z_fit, y_fit_raw)
    pred_f = predict(filt_f, ymf, ysf, Z_held)
    r2_f = oos_r2(y_held_raw.values, pred_f, y_fit_raw.mean())
    corr_f = safe_corr(y_held_raw.values, pred_f)

    filt_r, ymr, ysr = fit_outcome(Xp_z_fit, y_fit_raw)
    pred_r = predict(filt_r, ymr, ysr, Xp_z_held)
    r2_r = oos_r2(y_held_raw.values, pred_r, y_fit_raw.mean())
    corr_r = safe_corr(y_held_raw.values, pred_r)

    print(f"\n[{label}]")
    print(f"  frozen latent : held-out R^2={r2_f:+.4f}  corr={corr_f:+.4f}  "
          f"(retained {filt_f.best.n_features} of {Z_fit.shape[1]} blocks)")
    print(f"  raw features  : held-out R^2={r2_r:+.4f}  corr={corr_r:+.4f}  "
          f"(retained {filt_r.best.n_features} of {Xp_z_fit.shape[1]} ETFs)")
    return dict(frozen_r2=r2_f, frozen_corr=corr_f, raw_r2=r2_r, raw_corr=corr_r)


if __name__ == "__main__":
    rets, vol_A = load_data()
    fit_idx, held_idx = chronological_split(rets.index)
    print(f"{len(rets)} total trading days -> FIT={len(fit_idx)}  HELD-OUT={len(held_idx)}")
    print(f"FIT window:      {fit_idx[0].date()} -> {fit_idx[-1].date()}")
    print(f"HELD-OUT window: {held_idx[0].date()} -> {held_idx[-1].date()}")

    Xp_raw_fit = rets.loc[fit_idx, PREDICTORS]
    Xp_raw_held = rets.loc[held_idx, PREDICTORS]

    # --- Step 4: fit the frozen block latent on the FIT window only ---
    transform, Z_fit = fit_block_latent(Xp_raw_fit, RHO)
    Z_held = apply_block_latent(transform, Xp_raw_held)
    K = Z_fit.shape[1]
    print(f"\n{K} frozen factor blocks at rho={RHO} (fit on FIT window only):")
    for name in transform["names"]:
        print(f"  {name:35s} members={transform['blocks'][name]['members']}")

    # raw-feature baseline: same zscore stats (FIT-window mean/std), no compression
    Xp_z_fit = zscore_apply(Xp_raw_fit, transform["fit_mean"], transform["fit_std"])
    Xp_z_held = zscore_apply(Xp_raw_held, transform["fit_mean"], transform["fit_std"])

    # --- Fit Task A: AAPL return ~ frozen block latents, FIT window (smoke check) ---
    y_A_fit = rets.loc[fit_idx, TARGET_A]
    y_A_held = rets.loc[held_idx, TARGET_A]
    filt_A, yA_mean, yA_std = fit_outcome(Z_fit, y_A_fit)
    pred_A_insample = predict(filt_A, yA_mean, yA_std, Z_fit)
    r2_smoke = oos_r2(y_A_fit.values, pred_A_insample, y_A_fit.mean())
    print(f"\nSmoke check -- Fit Task A in-sample R^2 (AAPL return, FIT window): {r2_smoke:+.4f}")
    print(f"  retained {filt_A.best.n_features}/{K} blocks: "
          f"{[transform['names'][i] for i in filt_A.best.retained_idx]}")
    assert 0.0 < r2_smoke < 1.0, "Fit Task A looks degenerate -- stop before trusting transfer results"

    # --- Verify the structural point up front: does refitting the block-factor
    # pipeline from scratch on the SAME fit-window ETF panel actually change anything?
    transform_refit, Z_fit_refit = fit_block_latent(Xp_raw_fit, RHO)
    max_diff = float(np.max(np.abs(Z_fit.values - Z_fit_refit.values)))
    print(f"\nDiagnostic: max abs diff between 'frozen' and 'refit-from-scratch' block "
          f"latents on the identical FIT panel: {max_diff:.2e}")
    print("  (both blocks_from_correlation_threshold and gaussian_block_factor are pure "
          "functions of the predictor panel -- they never see y -- so this is expected "
          "to be ~0 regardless of which target we're about to regress on)")

    # --- Scenario (a): same target, held-out period ---
    res_a = run_scenario("(a) AAPL return, held-out period", y_A_fit, y_A_held,
                          Z_fit, Z_held, Xp_z_fit, Xp_z_held)
    res_a["refit_r2"], res_a["refit_corr"] = res_a["frozen_r2"], res_a["frozen_corr"]

    # --- Scenario (b): same stock, different target variable (AAPL realized vol) ---
    y_vol_fit = vol_A.loc[fit_idx]
    y_vol_held = vol_A.loc[held_idx]
    res_b = run_scenario("(b) AAPL realized volatility, held-out period", y_vol_fit, y_vol_held,
                          Z_fit, Z_held, Xp_z_fit, Xp_z_held)
    res_b["refit_r2"], res_b["refit_corr"] = res_b["frozen_r2"], res_b["frozen_corr"]

    # --- Scenario (c): different target stock (JPM), held-out period ---
    y_B_fit = rets.loc[fit_idx, TARGET_B]
    y_B_held = rets.loc[held_idx, TARGET_B]
    res_c = run_scenario("(c) JPM return, held-out period", y_B_fit, y_B_held,
                          Z_fit, Z_held, Xp_z_fit, Xp_z_held)
    res_c["refit_r2"], res_c["refit_corr"] = res_c["frozen_r2"], res_c["frozen_corr"]

    # --- Summary table ---
    scenarios = [("(a) same target, held-out period", res_a),
                 ("(b) same stock, new target (volatility)", res_b),
                 ("(c) different target stock (JPM)", res_c)]
    print("\n" + "=" * 78)
    print(f"{'scenario':42s} {'frozen R2':>10s} {'raw R2':>10s} {'refit R2':>10s}")
    print("-" * 78)
    for label, r in scenarios:
        print(f"{label:42s} {r['frozen_r2']:>+10.4f} {r['raw_r2']:>+10.4f} {r['refit_r2']:>+10.4f}")
    print("=" * 78)
    print("(refit R2 == frozen R2 exactly in every row -- see diagnostic above for why)")

    # --- Figure: grouped bar chart of held-out R^2, 3 scenarios x 3 arms ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(3)
    width = 0.26
    frozen_vals = [res_a["frozen_r2"], res_b["frozen_r2"], res_c["frozen_r2"]]
    raw_vals = [res_a["raw_r2"], res_b["raw_r2"], res_c["raw_r2"]]
    refit_vals = [res_a["refit_r2"], res_b["refit_r2"], res_c["refit_r2"]]
    ax.bar(x - width, frozen_vals, width, label="frozen transfer (block latent)", color="teal")
    ax.bar(x, raw_vals, width, label="raw features (31 ETFs)", color="tab:red", alpha=0.85)
    ax.bar(x + width, refit_vals, width, label="task-specific refit (== frozen, see note)",
           color="tab:gray", alpha=0.6, hatch="//")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["(a) AAPL return\nheld-out period", "(b) AAPL volatility\nnew target",
                         "(c) JPM return\nnew stock"])
    ax.set_ylabel("held-out R^2 (FIT-window-mean baseline)")
    ax.set_title("Does a block latent fit on AAPL/FIT transfer? Three scenarios, three arms\n"
                 "(refit bars sit exactly on frozen bars -- block construction never sees y)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_transfer_signal.png"), dpi=140)
    plt.close(fig)

    print(f"\nFigure written to {OUT}/finance_transfer_signal.png")
