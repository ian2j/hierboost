"""Complementary-pairs stability selection (hierboost.stability.
stability_select_complementary_pairs, Shah & Samworth 2013) vs the existing random-
subspace stability_select (Meinshausen & Buhlmann 2010 variant, see
tests/stability_selection_validation.py), same synthetic architecture and same pMOM
inner fitter as that file (N=1164, 167 candidate blocks of 5 correlated raw features
each, K_CAUSAL=4, target h2=0.4 -- matched to the real LCT problem's dimensionality).
Isolates what the complementary-pairs AGGREGATION mechanism buys over independent
random subsampling, holding everything else (prior, inner threshold grid, outer
threshold grid) fixed.
"""
import time
import numpy as np
import pandas as pd

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian_pmom
from hierboost.stability import stability_select, stability_select_complementary_pairs, admit

N, N_BLOCKS, GROUP_SIZE, K_CAUSAL, TARGET_H2 = 1164, 167, 5, 4, 0.4
WITHIN_GROUP_RHO = 0.85
XI0 = np.log(0.15 / 0.85)
KAPPA = 100.0
TAU = KAPPA / 3.0
P = N_BLOCKS


def simulate(rng):
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


def make_fit_selector(Zd_full, y, wr_full, inner_thresh):
    def fit_selector(subset_idx):
        cols = np.concatenate([[0], subset_idx + 1])
        Zsub = Zd_full[:, cols]
        wr_sub = wr_full[subset_idx]
        res = fit_em_gaussian_pmom(Zsub, y, wr_sub, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0)
        return np.where(res.theta_hat > inner_thresh)[0]
    return fit_selector


OUTER_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def run(n_replicates=15, n_subsets=50, subsample_frac=0.5, inner_thresh_grid=(0.02, 0.05, 0.1)):
    """n_subsets is used directly for stability_select and halved for
    stability_select_complementary_pairs's n_pairs, so BOTH methods run exactly
    n_subsets total selector fits per inner threshold -- equal compute budget."""
    rng_master = np.random.default_rng(0)
    rows = []
    t_start = time.time()
    for rep in range(n_replicates):
        X_raw, block_of, y, causal_blocks = simulate(rng_master)
        Z = compress(X_raw, block_of)
        Zd = np.column_stack([np.ones(N), Z])
        wr = np.ones(P)

        rng_stab = np.random.default_rng(1000 + rep)
        rng_cp = np.random.default_rng(3000 + rep)
        for inner_thresh in inner_thresh_grid:
            fit_selector = make_fit_selector(Zd, y, wr, inner_thresh)

            stab = stability_select(fit_selector, P, n_subsets=n_subsets,
                                     subsample_frac=subsample_frac, rng=rng_stab)
            cp = stability_select_complementary_pairs(fit_selector, P, n_pairs=n_subsets // 2,
                                                        rng=rng_cp)

            for outer_t in OUTER_GRID:
                admitted = admit(stab, outer_t)
                power, precision, n_ret = power_precision(set(admitted.tolist()), causal_blocks)
                rows.append(dict(rep=rep, config=f"random_inner{inner_thresh}",
                                  outer_thresh=outer_t, power=power, precision=precision, n_retained=n_ret))

                admitted_cp = admit(cp, outer_t)
                power, precision, n_ret = power_precision(set(admitted_cp.tolist()), causal_blocks)
                rows.append(dict(rep=rep, config=f"comp_pairs_inner{inner_thresh}",
                                  outer_thresh=outer_t, power=power, precision=precision, n_retained=n_ret))

        elapsed = time.time() - t_start
        print(f"rep {rep} done, elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby(["config", "outer_thresh"], dropna=False).agg(
        power=("power", "mean"), precision=("precision", "mean"),
        n_retained=("n_retained", "mean"), n_reps=("power", "count")).reset_index()
    print("\n" + "=" * 100)
    print(summary.to_string(index=False))
    return df, summary


if __name__ == "__main__":
    df, summary = run()
    df.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/b69659aa-1692-48e7-a136-db8f15be2200/scratchpad/comp_pairs_raw.csv", index=False)
    summary.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/b69659aa-1692-48e7-a136-db8f15be2200/scratchpad/comp_pairs_summary.csv", index=False)
