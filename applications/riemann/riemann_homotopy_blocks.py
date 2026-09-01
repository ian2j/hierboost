"""Genuine BLOCK-level analysis of the homotopy-exception pilot (riemann_homotopy_pilot.py).

That first pilot ran HierBoostClassifier with decorrelate=None -- meaning every one of
the 21 raw features (Re/Im at 9 checkpoints + 3 final-value summaries) was its own
singleton unit in the spike-and-slab prior. That answers "does this feature set predict
exceptions" but NOT hierboost's actual namesake question -- which BLOCKS of related
features matter, and does that block importance vary systematically.

Natural block structure here: the 9 checkpoints are ORDERED (10%, 20%, ..., 90% of the
way through the partial sum), each checkpoint contributing a (Re, Im) pair -- collapse
each pair into one shared latent block-factor via hierboost.factor.gaussian_block_factor
(probabilistic PCA), same mechanism Haxby/20-Newsgroups used for continuous features with
a binary outcome (HierBoostClassifier(decorrelate=...) can't do this directly for a
binomial response -- it routes unconditionally through Chapter 4's discrete/JAX path,
which assumes raw features are themselves Binomial; a known API gap documented in
newsgroups_demo.py). The 3 final-value summary features (log|S_M|, cos/sin(arg S_M))
form one additional "final" block. 10 blocks total, fit with hierboost.spike_slab.em_filter
directly on the block scores.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab import em_filter

from riemann_homotopy_pilot import build_dataset, CHECKPOINTS

BLOCK_NAMES = [f"checkpoint_{int(f*100)}pct" for f in CHECKPOINTS] + ["final_value"]


def build_blocks(X):
    """X: (n,21) = 9 Re + 9 Im + 3 final-value columns, in that column order
    (see riemann_homotopy_pilot.row_features/build_dataset)."""
    n = X.shape[0]
    re_cols, im_cols, final_cols = X[:, :9], X[:, 9:18], X[:, 18:21]
    scores = np.zeros((n, 10))
    loadings_list = []
    for k in range(9):
        block = np.column_stack([re_cols[:, k], im_cols[:, k]])
        s, l = gaussian_block_factor(block)
        scores[:, k] = s
        loadings_list.append(l)
    s, l = gaussian_block_factor(final_cols)
    scores[:, 9] = s
    loadings_list.append(l)
    return scores, loadings_list


def main():
    X, trivial, gap, y, feature_names = build_dataset()
    t = trivial[:, 0]  # log(t) -- recover raw t separately
    data = json.load(open("/home/ian/Research/Riemann/writeup/homotopy_exception_gap_data.json"))
    t_raw = np.array([d["t"] for d in data])

    Z, loadings_list = build_blocks(X)
    print("\nBlock-factor loadings (how Re/Im, or the 3 final-value stats, combine into "
          "each block's single latent score):")
    for name, l in zip(BLOCK_NAMES, loadings_list):
        print(f"  {name:20s} {np.round(l, 3)}")

    wr = np.ones(Z.shape[1])
    xi0 = np.log(y.mean() / (1 - y.mean()))
    Xd = np.column_stack([np.ones(len(y)), Z])
    filt = em_filter(Xd, y, wr, xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                      filter_frac=0.3, min_features=3, max_outer=50)
    best = filt.best
    retained = best.retained_idx if best.retained_idx is not None else np.arange(Z.shape[1])
    retained_names = [BLOCK_NAMES[i] for i in retained]

    print(f"\nFull-data block-level em_filter fit -- {len(retained)}/10 blocks retained "
          f"(of {Z.shape[1]} candidate blocks):")
    order = np.argsort(-best.theta_hat)
    for i in order:
        print(f"  {retained_names[i]:20s} theta_hat={best.theta_hat[i]:.4f}  coef={best.beta[1+i]:+.4f}")

    # ---- does each retained block's relationship to exception risk change with t? ----
    print("\nBlock x t interaction test (logistic, top blocks by theta_hat):")
    t_c = (np.log(t_raw) - np.log(t_raw).mean())
    for i in order[:3]:
        block_name = retained_names[i]
        z = Z[:, retained[i]]
        z_c = (z - z.mean()) / z.std()
        Xint = sm.add_constant(np.column_stack([z_c, t_c, z_c * t_c]))
        try:
            res = sm.Logit(y, Xint).fit(disp=0)
            b, se, p = res.params[3], res.bse[3], res.pvalues[3]
            print(f"  {block_name:20s} block:log(t) interaction coef={b:+.3f}  se={se:.3f}  p={p:.3f}")
        except Exception as e:
            print(f"  {block_name:20s} fit failed: {e}")


if __name__ == "__main__":
    main()
