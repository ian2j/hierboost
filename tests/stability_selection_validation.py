"""Stability-selection prototype (2026-09-19), BEFORE touching real genomics data --
same standing discipline as synthetic_pmom_validation.py.

Unlike that file's easy dimensions (N=1000, 30 candidate blocks -- a 33:1 N:p ratio
under which even a naive theta_hat threshold already hit near-perfect precision, a
result that then failed to transfer to the real LCT problem), THIS simulation is
built at the real problem's actual, harder dimensionality: N=1164, 167 candidate
blocks of 5 correlated raw features each, K_CAUSAL=4, target h2=0.4 -- a ~7:1 N:p
ratio matching applications/genomics/genomics_1kg_simulated_causal.py exactly. The
point is to stress-test stability selection (Meinshausen & Buhlmann 2010, random-
subspace variant -- see hierboost/stability.py's docstring) somewhere it can
actually fail, so a synthetic "it works" here is informative before spending real-
data compute.

Baseline (no stability selection): direct theta_hat threshold and the PPL-filter's
own best step, pMOM prior -- reusing the same pattern as
genomics_1kg_pmom_threshold_sweep.py, just on synthetic data of matched shape.
"""
import time
import numpy as np
import pandas as pd

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab_gaussian import fit_em_gaussian_pmom, em_filter_gaussian_pmom
from hierboost.stability import stability_select, admit

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
    """Closure for stability_select: fits pMOM EM on the subset's columns, returns
    local indices whose theta_hat exceeds the (loose) inner threshold."""
    def fit_selector(subset_idx):
        cols = np.concatenate([[0], subset_idx + 1])
        Zsub = Zd_full[:, cols]
        wr_sub = wr_full[subset_idx]
        res = fit_em_gaussian_pmom(Zsub, y, wr_sub, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0)
        return np.where(res.theta_hat > inner_thresh)[0]
    return fit_selector


OUTER_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def run(n_replicates=15, n_subsets=50, subsample_frac=0.5, inner_thresh_grid=(0.02, 0.05, 0.1)):
    rng_master = np.random.default_rng(0)
    rows = []
    t_start = time.time()
    for rep in range(n_replicates):
        X_raw, block_of, y, causal_blocks = simulate(rng_master)
        Z = compress(X_raw, block_of)
        Zd = np.column_stack([np.ones(N), Z])
        wr = np.ones(P)

        # baselines: direct threshold + PPL-filter best step, single full fit
        res_full = fit_em_gaussian_pmom(Zd, y, wr, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0)
        for t in (0.1, 0.3, 0.5):
            retained = set(np.where(res_full.theta_hat > t)[0].tolist())
            power, precision, n_ret = power_precision(retained, causal_blocks)
            rows.append(dict(rep=rep, config=f"baseline_threshold_{t}", power=power,
                              precision=precision, n_retained=n_ret))
        filt = em_filter_gaussian_pmom(Zd, y, wr, xi0=XI0, xi1=0.0, tau=TAU, nu=1.0, lam=1.0,
                                        filter_frac=0.2, min_features=4, max_outer=30)
        retained = set(filt.best.retained_idx.tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="baseline_PPLfilter", power=power,
                          precision=precision, n_retained=n_ret))

        # stability selection, sweep inner threshold, fixed subsample_frac/n_subsets
        rng_stab = np.random.default_rng(1000 + rep)
        for inner_thresh in inner_thresh_grid:
            fit_selector = make_fit_selector(Zd, y, wr, inner_thresh)
            stab = stability_select(fit_selector, P, n_subsets=n_subsets,
                                     subsample_frac=subsample_frac, rng=rng_stab)
            for outer_t in OUTER_GRID:
                admitted = admit(stab, outer_t)
                # (a) admission itself as the final selection
                retained = set(admitted.tolist())
                power, precision, n_ret = power_precision(retained, causal_blocks)
                rows.append(dict(rep=rep, config=f"stability_inner{inner_thresh}_admitonly",
                                  outer_thresh=outer_t, power=power, precision=precision, n_retained=n_ret))
                # (b) refit pMOM on admitted set, threshold at 0.5 ("apply hierboost the usual way").
                # sigma_g2 is inherited FIXED from the full/null-rich fit rather than
                # re-estimated on the (positive-enriched) admitted set -- see
                # fit_em_gaussian_pmom's sigma_g2_fixed docstring for why re-estimating
                # it here collapses theta_hat to ~0 even for genuinely causal features.
                if len(admitted) >= 1:
                    cols = np.concatenate([[0], admitted + 1])
                    res_refit = fit_em_gaussian_pmom(Zd[:, cols], y, wr[admitted], xi0=XI0, xi1=0.0,
                                                       tau=TAU, nu=1.0, lam=1.0,
                                                       sigma_g2_init=res_full.sigma_g2, sigma_g2_fixed=True)
                    final_local = np.where(res_refit.theta_hat > 0.5)[0]
                    retained = set(admitted[final_local].tolist())
                else:
                    retained = set()
                power, precision, n_ret = power_precision(retained, causal_blocks)
                rows.append(dict(rep=rep, config=f"stability_inner{inner_thresh}_refit",
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
    df.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/26060d51-cdc3-4930-8ed2-25273509cc38/scratchpad/stability_synth_raw.csv", index=False)
    summary.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/26060d51-cdc3-4930-8ed2-25273509cc38/scratchpad/stability_synth_summary.csv", index=False)
