"""Robustness check + deeper dig on the zeta_pair_residual.py "+75 degree dominant
block" finding.

Two concerns raised and checked here:

1. WRAP-AROUND ARTIFACT (found real, fixed): a histogram of the raw `resid` values
   showed 25/150 rows sitting right at the +-pi boundary, with a clear bimodal gap in
   the middle of the range -- strong evidence the underlying quantity is a smoothly
   varying angle that wraps across the branch cut, which would corrupt an ordinary
   least-squares fit on the raw signed radian value (a wrap looks like a huge outlier
   to OLS even though it's a tiny true change). Fixed by regressing on cos(resid) and
   sin(resid) separately (the wrap-safe representation) instead of resid itself.

2. BINNING-ARTIFACT check: does the dominant sector survive changing bin count and
   phase offset? If it's a real concentration of "explaining power" at a specific
   angle, the winning region should track a consistent real angle as the arbitrary
   bin grid is moved/resized, not jump incoherently.

Then: for whichever region survives both checks, actually look at which (m,n) pairs
populate it, across a few different (t,u) rows, for a real arithmetic explanation
rather than trusting the theta_hat number alone.
"""
import numpy as np
import pandas as pd

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian
from sklearn.preprocessing import StandardScaler

from zeta_pair_residual import T_VALUES, U_VALUES, natural_M, chi, wrap_pi
import mpmath as mp


def row_terms(t, u):
    """Un-binned: every (m,n) pair's rotated (aligned-frame) contribution, plus m,n,k."""
    M = natural_M(t)
    m = np.arange(1, M + 1)
    n = np.arange(1, M + 1)
    mm, nn = np.meshgrid(m, n, indexing="ij")
    k = (mm * nn).astype(float)
    mag = (mm.astype(float) ** (-0.5 + u)) * (nn.astype(float) ** (-0.5 - u))
    phase = -t * np.log(k)
    target = float(mp.arg(chi(mp.mpc(0.5 - u, t))))
    rel_phase = wrap_pi(phase - target)
    terms = mag * np.exp(1j * rel_phase)
    return mm.ravel(), nn.ravel(), k.ravel(), rel_phase.ravel(), terms.ravel(), target


def build_binned_dataset(n_bins, offset_deg=0.0):
    offset = np.deg2rad(offset_deg)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1) + offset
    rows = []
    for t in T_VALUES:
        for u in U_VALUES:
            _, _, _, rel_phase, terms, _ = row_terms(t, u)
            # wrap rel_phase into the (possibly offset) bin grid
            shifted = wrap_pi(rel_phase - offset)
            idx = np.clip(np.digitize(shifted, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1, 0, n_bins - 1)
            block_re = np.zeros(n_bins)
            block_im = np.zeros(n_bins)
            for b in range(n_bins):
                sel = terms[idx == b]
                block_re[b] = sel.real.sum()
                block_im[b] = sel.imag.sum()
            total = terms.sum()
            resid = float(np.arctan2(total.imag, total.real))
            rows.append(dict(t=t, u=u, resid=resid, cos_r=np.cos(resid), sin_r=np.sin(resid),
                              **{f"re_{b}": block_re[b] for b in range(n_bins)},
                              **{f"im_{b}": block_im[b] for b in range(n_bins)}))
    df = pd.DataFrame(rows)
    bin_centers_deg = np.rad2deg((np.linspace(-np.pi, np.pi, n_bins + 1)[:-1] + np.pi / n_bins)) + offset_deg
    return df, bin_centers_deg


def fit_blocks(df, n_bins, target_col):
    Z = np.zeros((len(df), n_bins))
    for b in range(n_bins):
        block = df[[f"re_{b}", f"im_{b}"]].to_numpy()
        if np.allclose(block.std(axis=0), 0):
            continue
        s, _ = gaussian_block_factor(block)
        Z[:, b] = s
    y = df[target_col].to_numpy()
    Zs = StandardScaler().fit_transform(Z)
    Xd = np.column_stack([np.ones(len(y)), Zs])
    wr = np.ones(n_bins)
    filt = em_filter_gaussian(Xd, y, wr, xi0=np.log(0.3 / 0.7), xi1=0.0, kappa=100.0,
                               nu=1.0, lam=1.0, filter_frac=0.3, min_features=3, max_outer=50)
    return filt.best


def top_block_report(best, retained, bin_centers_deg, label):
    order = np.argsort(-best.theta_hat)
    top = order[0]
    b = retained[top]
    print(f"  [{label}] top block: {bin_centers_deg[b]:+7.1f} deg  theta_hat={best.theta_hat[top]:.4f}  "
          f"coef={best.beta[1+top]:+.4f}  (retained {len(retained)} blocks)")
    return bin_centers_deg[b], best.theta_hat[top]


def wrap_safe_check(n_bins=12):
    print(f"=== wrap-safe check (n_bins={n_bins}, cos/sin targets instead of raw resid) ===")
    df, centers = build_binned_dataset(n_bins)
    for target_col in ("resid", "cos_r", "sin_r"):
        best = fit_blocks(df, n_bins, target_col)
        retained = best.retained_idx if best.retained_idx is not None else np.arange(n_bins)
        top_block_report(best, retained, centers, target_col)


def binning_robustness():
    print("\n=== binning robustness (top block by explaining cos(resid) fluctuation) ===")
    for n_bins in (8, 12, 16, 24, 36):
        for offset_deg in (0.0, 7.5, 15.0):
            df, centers = build_binned_dataset(n_bins, offset_deg)
            best = fit_blocks(df, n_bins, "cos_r")
            retained = best.retained_idx if best.retained_idx is not None else np.arange(n_bins)
            top_block_report(best, retained, centers, f"n_bins={n_bins:2d} offset={offset_deg:5.1f}")


def inspect_hot_region(center_deg, half_width_deg=15.0):
    print(f"\n=== which (m,n) pairs populate the {center_deg:+.0f} deg region ===")
    lo, hi = np.deg2rad(center_deg - half_width_deg), np.deg2rad(center_deg + half_width_deg)
    for t in T_VALUES:
        u = 0.25  # representative mid-range u
        mm, nn, k, rel_phase, terms, target = row_terms(t, u)
        in_region = (rel_phase >= lo) & (rel_phase <= hi) if lo < hi else None
        sel = np.where(in_region)[0]
        # rank by magnitude within the region
        mags = np.abs(terms[sel])
        order = sel[np.argsort(-mags)][:6]
        pairs = [(int(mm[i]), int(nn[i]), int(k[i]), abs(terms[i])) for i in order]
        print(f"  t={t:7.1f} u={u:.2f}  top pairs in region (m,n,k=mn,|term|): "
              + ", ".join(f"({m},{n},{kk},{mag:.3f})" for m, n, kk, mag in pairs))


if __name__ == "__main__":
    wrap_safe_check()
    binning_robustness()
    inspect_hot_region(75.0)
