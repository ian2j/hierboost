"""Synthetic validation of the pMOM non-local-prior E/M-step (2026-09-19), BEFORE
touching the genomics ground-truth simulation -- this project's standing discipline
for new inference machinery.

Two rounds, kept both because the gap between them is itself the finding:
  round 1 (RAW_FEATURE_SIM=False path, see git history / project memory) -- applied
  theta_conditional directly to raw correlated features, no compression step. BOTH
  local and pMOM hit perfect precision (1.0) in both an i.i.d. and an LD-correlated
  variant -- informative, but it turned out not to reproduce the genomics failure at
  all, because it was missing something central to how hierboost actually processes
  an LD block: PPCA compression to one shared latent BEFORE the outcome model ever
  sees the data.
  round 2 (this file) adds that step back: raw features are grouped into blocks with
  real within-block correlation, each block is compressed to ONE score via
  gaussian_block_factor (unsupervised, X only, exactly hierboost's real pipeline),
  and spike-and-slab (local or pMOM) runs on the COMPRESSED scores -- the properly
  analogous test to what genomics_1kg_simulated_causal.py actually does.

n=1000, 30 blocks of 5 correlated raw features each (rho=0.85, matching real LD
magnitudes), K_CAUSAL=4 blocks each containing exactly 1 true causal raw feature
among its 5, target h2=0.4 -- same difficulty parameters as every ground-truth test
in this project's genomics catalog.
"""
import numpy as np
import pandas as pd

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import (fit_em_gaussian, fit_em_gaussian_pmom,
                                            em_filter_gaussian, em_filter_gaussian_pmom)

N, N_BLOCKS, GROUP_SIZE, K_CAUSAL, TARGET_H2 = 1000, 30, 5, 4, 0.4
WITHIN_GROUP_RHO = 0.85
XI0 = np.log(0.15 / 0.85)
KAPPA = 100.0
TAU = KAPPA / 3.0  # variance-matched, see theta_conditional_pmom's docstring
N_REPLICATES = 30
P = N_BLOCKS  # after compression, one column per block


def simulate(rng):
    """Raw block-correlated features (real within-block structure), one true causal
    SNP inside each of K_CAUSAL randomly chosen blocks."""
    X_raw = np.zeros((N, N_BLOCKS * GROUP_SIZE))
    block_of = np.repeat(np.arange(N_BLOCKS), GROUP_SIZE)
    for g in range(N_BLOCKS):
        latent = rng.normal(size=N)
        for j in range(GROUP_SIZE):
            noise = rng.normal(size=N)
            X_raw[:, g * GROUP_SIZE + j] = (np.sqrt(WITHIN_GROUP_RHO) * latent
                                             + np.sqrt(1 - WITHIN_GROUP_RHO) * noise)

    causal_blocks = rng.choice(N_BLOCKS, size=K_CAUSAL, replace=False)
    causal_raw_idx = np.array([g * GROUP_SIZE + rng.integers(GROUP_SIZE) for g in causal_blocks])
    true_beta = rng.normal(size=K_CAUSAL)
    Xc = X_raw[:, causal_raw_idx] - X_raw[:, causal_raw_idx].mean(axis=0)
    genetic = Xc @ true_beta
    noise_var = genetic.var() * (1 - TARGET_H2) / TARGET_H2
    y = genetic + rng.normal(scale=np.sqrt(noise_var), size=N)
    return X_raw, block_of, y, set(causal_blocks.tolist())


def compress(X_raw, block_of):
    """Unsupervised PPCA compression, one score per block -- exactly
    genomics_1kg_demo.py's fit_block_factors, X only, never y."""
    Z = np.zeros((N, N_BLOCKS))
    for g in range(N_BLOCKS):
        Xb = X_raw[:, block_of == g]
        score, _ = gaussian_block_factor(Xb)
        Z[:, g] = score
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (Z - mu) / sd


def power_precision(retained_set, causal_blocks):
    tp = len(retained_set & causal_blocks)
    power = tp / K_CAUSAL
    precision = tp / len(retained_set) if retained_set else np.nan
    return power, precision, len(retained_set)


def run():
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(N_REPLICATES):
        X_raw, block_of, y, causal_blocks = simulate(rng)
        Z = compress(X_raw, block_of)
        Zd = np.column_stack([np.ones(N), Z])
        wr = np.ones(P)

        # local, PPL-filter
        filt = em_filter_gaussian(Zd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0,
                                   filter_frac=0.2, min_features=4, max_outer=30)
        retained = set(filt.best.retained_idx.tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="local_PPLfilter", power=power, precision=precision, n_retained=n_ret))

        # local, direct threshold on one full (p=30) fit
        res_full = fit_em_gaussian(Zd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0)
        retained = set(np.where(res_full.theta_hat > 0.5)[0].tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="local_threshold", power=power, precision=precision, n_retained=n_ret))

        # pMOM, PPL-filter
        filt = em_filter_gaussian_pmom(Zd, y, wr, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0,
                                        filter_frac=0.2, min_features=4, max_outer=30)
        retained = set(filt.best.retained_idx.tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="pmom_PPLfilter", power=power, precision=precision, n_retained=n_ret))

        # pMOM, direct threshold on one full (p=30) fit
        res_full = fit_em_gaussian_pmom(Zd, y, wr, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0)
        retained = set(np.where(res_full.theta_hat > 0.5)[0].tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="pmom_threshold", power=power, precision=precision, n_retained=n_ret))

        print(f"rep {rep}: local_thr power={rows[-3]['power']:.2f} prec={rows[-3]['precision']:.2f} | "
              f"pmom_thr power={rows[-1]['power']:.2f} prec={rows[-1]['precision']:.2f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(f"{N_REPLICATES} replicates, PPCA-compressed blocks (the real pipeline shape): "
          f"N={N}, {N_BLOCKS} blocks x {GROUP_SIZE} raw features, K_CAUSAL={K_CAUSAL}, h2={TARGET_H2}")
    print("=" * 90)
    summary = df.groupby("config").agg(power=("power", "mean"), power_std=("power", "std"),
                                        precision=("precision", "mean"), precision_std=("precision", "std"),
                                        n_retained=("n_retained", "mean"))
    print(summary.to_string())
    return df


if __name__ == "__main__":
    run()
