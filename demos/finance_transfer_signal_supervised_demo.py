"""Closes a gap left open by finance_transfer_signal_demo.py: that script set out to
compare a FROZEN block latent against a "task-specific refit" arm, but discovered
(numerically, max abs diff = 0.0) that blocks_from_correlation_threshold and
gaussian_block_factor are both pure functions of the predictor panel alone -- neither
ever looks at y. So there was never a genuinely target-AWARE alternative in that
experiment to compare against; "refit from scratch" was bit-identical to "frozen"
by construction, for every scenario.

This script builds that missing target-aware alternative -- hierboost.factor's new
supervised_block_factor (1-component PLS: w = Xb.T @ y, normalized; the direction of
MAXIMUM COVARIANCE with y, instead of gaussian_block_factor's direction of maximum
variance within the block) -- and uses it to ask the sharper question the original
script's structural finding left unanswered: is a target-aware factor actually
target-SPECIFIC, in the sense that matters? Not "does knowing y help" (trivially yes,
any signal at all beats none) but: does a factor fit to maximize covariance with STOCK
A's return do better predicting A and WORSE predicting STOCK B than a factor that never
looked at either -- i.e. does supervision buy specificity, or does it just buy a
slightly different flavor of the same market-wide co-movement structure that the
unsupervised PPCA factor was already capturing?

Design: fit the supervised block factor TWICE on the FIT window -- once against AAPL's
own FIT-window return ("AAPL-supervised"), once against JPM's own FIT-window return
("JPM-supervised") -- freeze each via apply_supervised_block_factor, and cross-evaluate
both against both stocks' held-out returns (own-target and wrong-target transfer),
alongside the original unsupervised frozen factor as a target-blind reference point.
Block MEMBERSHIP itself is still computed by blocks_from_correlation_threshold, which
still never looks at y (that is unavoidable without also restructuring the blocking
step itself -- out of scope here, see finance_transfer_signal_demo.py's closing note on
what a genuinely target-aware "refit ceiling" would require of block membership, not
just loadings) -- what varies between the AAPL-supervised and JPM-supervised transforms
below is only the per-block factor DIRECTION w, never which raw ETFs land in which block.

Reuses finance_transfer_signal_demo.py's data loading/splitting/outcome-regression
helpers verbatim (load_data, chronological_split, zscore_apply, fit_outcome, predict,
oos_r2, safe_corr, fit_block_latent, apply_block_latent) -- nothing about that machinery
needed to change, only the factor-construction step. See that script for the full
house-style rationale (flat prior, 31-ETF universe, rho=0.9, 70/30 chronological split).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import supervised_block_factor, apply_supervised_block_factor
from finance_transfer_signal_demo import (
    OUT, TARGET_A, TARGET_B, PREDICTORS, RHO, FIT_FRAC,
    load_data, chronological_split, zscore_apply,
    fit_outcome, predict, oos_r2, safe_corr,
    fit_block_latent, apply_block_latent,
)


# ---------------------------------------------------------------------------
# Supervised (target-aware) block latent: same block MEMBERSHIP machinery as
# fit_block_latent, but each multi-member block's factor direction comes from
# supervised_block_factor(Xb, y) instead of gaussian_block_factor(Xb).
# ---------------------------------------------------------------------------

def fit_supervised_block_latent(Xp_raw_fit, y_fit_raw, rho):
    """Fit block membership (correlation threshold, target-blind, same as
    fit_block_latent) + per-block supervised (max-covariance-with-y) direction, using
    y_fit_raw ONLY on the FIT window. Returns (transform, Z_fit); apply the transform to
    new predictor data with apply_supervised_block_latent, which never re-estimates w
    and never looks at any new y.
    """
    fit_mean = Xp_raw_fit.mean()
    fit_std = Xp_raw_fit.std()
    Xp_z_fit = zscore_apply(Xp_raw_fit, fit_mean, fit_std)

    y_mean, y_std = float(y_fit_raw.mean()), float(y_fit_raw.std())
    y_z = ((y_fit_raw - y_mean) / y_std).values

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
            blocks[fname] = dict(members=list(members), w=None)
        else:
            fname = "+".join(members) + "_factor"
            score, w = supervised_block_factor(Xb, y_z)
            blocks[fname] = dict(members=list(members), w=w)
            factor_cols[fname] = score
        names.append(fname)

    Z_fit = pd.DataFrame(factor_cols, index=Xp_raw_fit.index)[names]
    transform = dict(fit_mean=fit_mean, fit_std=fit_std, blocks=blocks, names=names,
                      supervised_on=y_fit_raw.name)
    return transform, Z_fit


def apply_supervised_block_latent(transform, Xp_raw_new):
    """Apply an already-fitted supervised transform to new raw predictor data -- no
    re-estimation, no y involved at all (mirrors apply_block_latent's role)."""
    Xp_z_new = zscore_apply(Xp_raw_new, transform["fit_mean"], transform["fit_std"])
    cols = {}
    for fname in transform["names"]:
        info = transform["blocks"][fname]
        Xb = Xp_z_new[info["members"]].values
        if info["w"] is None:
            cols[fname] = Xb[:, 0]
        else:
            cols[fname] = apply_supervised_block_factor(Xb, info["w"])
    return pd.DataFrame(cols, index=Xp_raw_new.index)[transform["names"]]


def evaluate(Z_fit, Z_held, y_fit_raw, y_held_raw):
    """One frozen-factor -> one-target evaluation cell: fit the outcome regression on
    the FIT window (Z_fit against y_fit_raw), predict on Z_held, score against
    y_held_raw. Identical machinery to finance_transfer_signal_demo.py's frozen arm."""
    filt, ym, ys = fit_outcome(Z_fit, y_fit_raw)
    pred = predict(filt, ym, ys, Z_held)
    r2 = oos_r2(y_held_raw.values, pred, y_fit_raw.mean())
    corr = safe_corr(y_held_raw.values, pred)
    return dict(r2=r2, corr=corr, n_features=filt.best.n_features, K=Z_fit.shape[1])


if __name__ == "__main__":
    rets, vol_A = load_data()
    fit_idx, held_idx = chronological_split(rets.index)
    print(f"{len(rets)} total trading days -> FIT={len(fit_idx)}  HELD-OUT={len(held_idx)}")

    Xp_raw_fit = rets.loc[fit_idx, PREDICTORS]
    Xp_raw_held = rets.loc[held_idx, PREDICTORS]
    y_A_fit, y_A_held = rets.loc[fit_idx, TARGET_A], rets.loc[held_idx, TARGET_A]
    y_B_fit, y_B_held = rets.loc[fit_idx, TARGET_B], rets.loc[held_idx, TARGET_B]

    # --- Fit the two supervised transforms, each frozen after the FIT window ---
    transform_A, Z_A_fit = fit_supervised_block_latent(Xp_raw_fit, y_A_fit, RHO)
    Z_A_held = apply_supervised_block_latent(transform_A, Xp_raw_held)
    transform_B, Z_B_fit = fit_supervised_block_latent(Xp_raw_fit, y_B_fit, RHO)
    Z_B_held = apply_supervised_block_latent(transform_B, Xp_raw_held)
    K = Z_A_fit.shape[1]
    print(f"\n{K} blocks at rho={RHO} (membership is target-blind and identical for "
          f"both supervised fits -- only the per-block direction w differs)")

    # --- Reference: the original target-blind unsupervised frozen factor, recomputed
    # here for a clean side-by-side (bit-identical to finance_transfer_signal_demo.py's
    # own numbers, since it's the same deterministic pipeline on the same data) ---
    transform_U, Z_U_fit = fit_block_latent(Xp_raw_fit, RHO)
    Z_U_held = apply_block_latent(transform_U, Xp_raw_held)

    # --- Diagnostic: how different are the AAPL-supervised and JPM-supervised
    # directions, block by block, and how different are they from the unsupervised
    # (PPCA) direction? cos-similarity near +-1 means "same direction essentially,
    # supervision didn't buy anything new here"; near 0 means genuinely different bets.
    print(f"\n{'block':35s} {'cos(w_AAPL, w_JPM)':>20s} {'cos(w_AAPL, PPCA)':>20s} "
          f"{'cos(w_JPM, PPCA)':>18s}")
    for fname in transform_A["names"]:
        wA = transform_A["blocks"][fname]["w"]
        wB = transform_B["blocks"][fname]["w"]
        wU = transform_U["blocks"][fname]["loadings"]
        if wA is None or wU is None:
            continue  # singleton block, no direction to compare
        cos_AB = float(wA @ wB / (np.linalg.norm(wA) * np.linalg.norm(wB)))
        cos_AU = float(wA @ wU / (np.linalg.norm(wA) * np.linalg.norm(wU)))
        cos_BU = float(wB @ wU / (np.linalg.norm(wB) * np.linalg.norm(wU)))
        print(f"{fname:35s} {cos_AB:>20.3f} {cos_AU:>20.3f} {cos_BU:>18.3f}")

    # --- The six evaluation cells ---
    cell_A_own = evaluate(Z_A_fit, Z_A_held, y_A_fit, y_A_held)     # AAPL-sup -> AAPL
    cell_A_wrong = evaluate(Z_A_fit, Z_A_held, y_B_fit, y_B_held)   # AAPL-sup -> JPM
    cell_B_own = evaluate(Z_B_fit, Z_B_held, y_B_fit, y_B_held)     # JPM-sup  -> JPM
    cell_B_wrong = evaluate(Z_B_fit, Z_B_held, y_A_fit, y_A_held)   # JPM-sup  -> AAPL
    cell_U_A = evaluate(Z_U_fit, Z_U_held, y_A_fit, y_A_held)       # unsupervised -> AAPL
    cell_U_B = evaluate(Z_U_fit, Z_U_held, y_B_fit, y_B_held)       # unsupervised -> JPM

    rows = [
        ("AAPL-supervised -> AAPL (own target)",  cell_A_own),
        ("AAPL-supervised -> JPM  (wrong target)", cell_A_wrong),
        ("JPM-supervised  -> JPM  (own target)",  cell_B_own),
        ("JPM-supervised  -> AAPL (wrong target)", cell_B_wrong),
        ("unsupervised    -> AAPL (reference)",   cell_U_A),
        ("unsupervised    -> JPM  (reference)",   cell_U_B),
    ]
    print("\n" + "=" * 86)
    print(f"{'factor -> target':42s} {'held-out R2':>12s} {'corr':>8s} {'features':>10s}")
    print("-" * 86)
    for label, c in rows:
        print(f"{label:42s} {c['r2']:>+12.4f} {c['corr']:>+8.4f} "
              f"{c['n_features']:>4d}/{c['K']:<5d}")
    print("=" * 86)

    # --- The specificity question, stated numerically ---
    own_minus_wrong_A = cell_A_own["r2"] - cell_A_wrong["r2"]
    own_minus_wrong_B = cell_B_own["r2"] - cell_B_wrong["r2"]
    wrong_vs_ref_A = cell_B_wrong["r2"] - cell_U_A["r2"]   # JPM-sup on AAPL vs unsupervised on AAPL
    wrong_vs_ref_B = cell_A_wrong["r2"] - cell_U_B["r2"]   # AAPL-sup on JPM vs unsupervised on JPM
    def verdict(diff):
        if diff < 0:
            return "WORSE than unsupervised, as the specificity hypothesis predicts"
        return "NOT worse than unsupervised -- specificity hypothesis not supported here"

    print("\nSpecificity diagnostics:")
    print(f"  AAPL-supervised: own-target R2 - wrong-target R2 = {own_minus_wrong_A:+.4f}")
    print(f"  JPM-supervised:  own-target R2 - wrong-target R2 = {own_minus_wrong_B:+.4f}")
    print(f"  JPM-supervised on AAPL (wrong) vs unsupervised on AAPL (ref): "
          f"{wrong_vs_ref_A:+.4f}  ({verdict(wrong_vs_ref_A)})")
    print(f"  AAPL-supervised on JPM (wrong) vs unsupervised on JPM (ref): "
          f"{wrong_vs_ref_B:+.4f}  ({verdict(wrong_vs_ref_B)})")

    # --- Figure: grouped bar chart of held-out R^2 across all six cells ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels_short = ["AAPL-sup\n->AAPL (own)", "AAPL-sup\n->JPM (wrong)",
                     "JPM-sup\n->JPM (own)", "JPM-sup\n->AAPL (wrong)",
                     "unsup.\n->AAPL (ref)", "unsup.\n->JPM (ref)"]
    vals = [c["r2"] for _, c in rows]
    colors = ["teal", "tab:orange", "teal", "tab:orange", "tab:gray", "tab:gray"]
    ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels_short, fontsize=8)
    ax.set_ylabel("held-out R^2 (FIT-window-mean baseline)")
    ax.set_title("Supervised (max-covariance-with-y) block factor: own-target vs\n"
                 "wrong-target transfer, against the unsupervised PPCA factor as reference")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "finance_transfer_signal_supervised.png"), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/finance_transfer_signal_supervised.png")
