"""Confirms or kills the horseshoe-shrinkage hypothesis from calibration_check_joint.py's
finding 3: does joint.py's posterior mean for the causal block stay biased toward zero
even at LARGE n (where sampling noise is no longer a plausible explanation), and does
that bias shrink as the true signal gets stronger? Both would point specifically at the
regularized horseshoe prior over-shrinking weak effects, not at insufficient MCMC
sampling or small-n noise.

One large-n fit per signal strength (not a multi-replication study -- posterior mean at
large n is already low-variance, so a single fit per setting is informative).
"""
import numpy as np
from scipy.special import expit

from calibration_check_joint import (FITS, COORDS, BLOCK_ID, N_TRIALS, XI0, K_BLOCKS,
                                      CAUSAL_BLOCKS, simulate_one_block)
from hierboost.estimator import HierBoostClassifier
from hierboost.joint import run_joint_inference, posterior_association_summary


def simulate_at_gamma(rng, n, gamma_true):
    X_parts = []
    eta = np.zeros(n)
    for b in range(K_BLOCKS):
        Xb, Zt = simulate_one_block(rng, FITS[b], n)
        X_parts.append(Xb)
        if b == CAUSAL_BLOCKS[0]:
            eta += gamma_true * Zt
    X = np.column_stack(X_parts)
    y = (rng.random(n) < expit(eta)).astype(float)
    return X, y


if __name__ == "__main__":
    n = 3000
    for gamma_true in [0.15, 0.6]:
        rng = np.random.default_rng(7)
        X, y = simulate_at_gamma(rng, n, gamma_true)

        clf = HierBoostClassifier(decorrelate="sar", fit_method="em", xi0=XI0, xi1=0.0, kappa=100.0)
        clf.fit(X, y, coords=COORDS, block_id=BLOCK_ID, n_trials=N_TRIALS)
        beta_plugin = clf.beta_[1:]

        mcmc, block_ids = run_joint_inference(X, y, BLOCK_ID, clf.latent_fits_, n_trials=N_TRIALS,
                                               num_warmup=500, num_samples=1000, seed=1,
                                               progress_bar=False)
        summ = posterior_association_summary(mcmc, block_ids)

        print(f"\ngamma_true(Z-units)={gamma_true}  n={n}")
        print(f"  plug-in beta (asymptotic reference, causal block): {beta_plugin[CAUSAL_BLOCKS[0]]:.4f}")
        print(f"  joint posterior mean gamma, all blocks: {np.round(summ['gamma_mean'], 4)}")
        print(f"  joint causal block posterior mean: {summ['gamma_mean'][CAUSAL_BLOCKS[0]]:.4f}")
        ratio = summ['gamma_mean'][CAUSAL_BLOCKS[0]] / beta_plugin[CAUSAL_BLOCKS[0]]
        print(f"  ratio joint/plugin (1.0 = no shrinkage bias relative to plug-in's own scale): {ratio:.3f}")
