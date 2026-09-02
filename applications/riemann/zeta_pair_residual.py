"""Prototype: hierboost as a descriptive tool for which angular blocks of terms in the
truncated zeta product g(u) = P_M(1/2-u+it)*P_M(1/2+u+it) pull its argument toward the
known closed-form target angle (exact for the full zeta, via chi). A variance-
decomposition use of hierboost, not a predictive one."""
import numpy as np
import pandas as pd
import mpmath as mp
from sklearn.preprocessing import StandardScaler

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

mp.mp.dps = 25
N_BINS = 12
T_VALUES = [110.0, 250.0, 510.0, 750.0, 1050.0, 1200.0]  # spread across the earlier pilot's t range
U_VALUES = np.linspace(0.02, 0.48, 25)


def chi(s):
    return 2 ** s * mp.pi ** (s - 1) * mp.sin(mp.pi * s / 2) * mp.gamma(1 - s)


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def natural_M(t):
    return max(8, int(np.floor(np.sqrt(t / (2 * np.pi)))))


def row_blocks(t, u):
    """Returns (block_re[12], block_im[12], residual) for one (t,u)."""
    M = natural_M(t)
    m = np.arange(1, M + 1)
    n = np.arange(1, M + 1)
    mm, nn = np.meshgrid(m, n, indexing="ij")
    k = (mm * nn).astype(float)
    mag = (mm.astype(float) ** (-0.5 + u)) * (nn.astype(float) ** (-0.5 - u))
    phase = -t * np.log(k)
    target = float(mp.arg(chi(mp.mpc(0.5 - u, t))))
    rel_phase = wrap_pi(phase - target)  # 0 = perfectly aligned with target direction
    terms = mag * np.exp(1j * rel_phase)  # already rotated into the "aligned=real axis" frame

    bin_edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    bin_idx = np.digitize(rel_phase.ravel(), bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
    block_re = np.zeros(N_BINS)
    block_im = np.zeros(N_BINS)
    flat = terms.ravel()
    for b in range(N_BINS):
        sel = flat[bin_idx == b]
        block_re[b] = sel.real.sum()
        block_im[b] = sel.imag.sum()

    residual = float(np.arctan2(block_im.sum(), block_re.sum()))  # = arg(g_M(u)) - target, wrapped
    return block_re, block_im, residual, M


def build_dataset():
    rows = []
    for t in T_VALUES:
        for u in U_VALUES:
            block_re, block_im, resid, M = row_blocks(t, u)
            rows.append(dict(t=t, u=u, M=M, resid=resid,
                              **{f"re_{b}": block_re[b] for b in range(N_BINS)},
                              **{f"im_{b}": block_im[b] for b in range(N_BINS)}))
    return pd.DataFrame(rows)


def block_latent(df):
    """Collapse each angular bin's (Re,Im) pair into one shared latent block score."""
    n = len(df)
    Z = np.zeros((n, N_BINS))
    for b in range(N_BINS):
        block = df[[f"re_{b}", f"im_{b}"]].to_numpy()
        if np.allclose(block.std(axis=0), 0):
            Z[:, b] = 0.0
            continue
        s, _ = gaussian_block_factor(block)
        Z[:, b] = s
    return Z


def pooled_fit(df):
    Z = block_latent(df)
    y = df["resid"].to_numpy()
    Zs = StandardScaler().fit_transform(Z)
    Xd = np.column_stack([np.ones(len(y)), Zs])
    wr = np.ones(Z.shape[1])
    filt = em_filter_gaussian(Xd, y, wr, xi0=np.log(0.3 / 0.7), xi1=0.0, kappa=100.0,
                               nu=1.0, lam=1.0, filter_frac=0.3, min_features=3, max_outer=50)
    best = filt.best
    retained = best.retained_idx if best.retained_idx is not None else np.arange(N_BINS)
    bin_centers_deg = (np.linspace(-np.pi, np.pi, N_BINS + 1)[:-1] + np.pi / N_BINS) * 180 / np.pi
    print(f"\nPooled fit across all {len(df)} (t,u) rows -- {len(retained)}/{N_BINS} angular "
          f"blocks retained (residual R^2 on full data: "
          f"{1 - np.sum((y - Xd[:, [0]+list(1+retained)] @ best.beta[[0]+list(1+np.arange(len(retained)))])**2)/np.sum((y-y.mean())**2):.3f}):")
    order = np.argsort(-best.theta_hat)
    for i in order:
        b = retained[i]
        print(f"  bin centered {bin_centers_deg[b]:+7.1f} deg from target  theta_hat={best.theta_hat[i]:.4f}"
              f"  coef={best.beta[1+i]:+.4f}")
    return Z, retained, best


def magnitude_profile_by_u(df):
    """Direct descriptive check (no fitting): for each t, which angular bin holds the
    largest |block| as u sweeps -- does the dominant bin shift, and does the SAME t-slice
    picture hold across different t."""
    print("\nDominant angular bin (by |block| magnitude) vs u, per t:")
    for t in T_VALUES:
        sub = df[df["t"] == t].sort_values("u")
        mags = np.sqrt(sub[[f"re_{b}" for b in range(N_BINS)]].to_numpy() ** 2
                        + sub[[f"im_{b}" for b in range(N_BINS)]].to_numpy() ** 2)
        dom_bin = mags.argmax(axis=1)
        u_arr = sub["u"].to_numpy()
        # print at a handful of u checkpoints
        idxs = np.linspace(0, len(u_arr) - 1, 6).astype(int)
        row_str = "  ".join(f"u={u_arr[i]:.2f}->bin{dom_bin[i]:2d}" for i in idxs)
        print(f"  t={t:7.1f} M={sub['M'].iloc[0]:2d}:  {row_str}")


if __name__ == "__main__":
    df = build_dataset()
    print(f"Built {len(df)} rows ({len(T_VALUES)} t-values x {len(U_VALUES)} u-values), "
          f"M range {df['M'].min()}-{df['M'].max()}")
    print(f"Residual (rad) stats: mean={df['resid'].mean():.4f} std={df['resid'].std():.4f} "
          f"[{df['resid'].min():.4f}, {df['resid'].max():.4f}]")
    pooled_fit(df)
    magnitude_profile_by_u(df)
