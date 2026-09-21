"""Local-slab Gibbs+centroid vs local-slab EM+stability-select, on the SAME synthetic
architecture as stability_selection_validation.py (N=1164, 167 candidate blocks of 5
correlated raw features, K_CAUSAL=4, target h2=0.4 -- matched to the real LCT problem's
dimensionality). That file used the pMOM prior throughout; this one holds the prior
fixed to the LOCAL Gaussian slab on both arms deliberately, so the comparison isolates
what full-posterior Gibbs sampling (hierboost.spike_slab_gaussian.gibbs_sampler_gaussian
+ hierboost.spike_slab.centroid_estimate) buys over the EM point-estimate path (direct
threshold, PPL-filter, stability selection) -- using pMOM on only one arm would conflate
the prior change with the sampling-vs-point-estimate change this is meant to isolate.

Gibbs runs ONCE per replicate on the full 167-candidate set (no subsampling/refitting
needed -- centroid_estimate is a direct decision rule on the marginal posterior
pi_hat = theta.mean(axis=0)), swept over the same outer-threshold grid stability
selection's `admit()` uses, via gamma = (1-t)/t so centroid_estimate(pi_hat, gamma)
selects exactly {j : pi_hat[j] >= t}. Directly comparable rows: config starting with
"gibbs_centroid" vs "local_stability_..._admitonly" at the same outer_thresh.
"""
import time
import numpy as np
import pandas as pd

from hierboost.factor import gaussian_block_factor
from hierboost.spike_slab import centroid_estimate
from hierboost.spike_slab_gaussian import fit_em_gaussian, em_filter_gaussian, gibbs_sampler_gaussian
from hierboost.stability import stability_select, admit

N, N_BLOCKS, GROUP_SIZE, K_CAUSAL, TARGET_H2 = 1164, 167, 5, 4, 0.4
WITHIN_GROUP_RHO = 0.85
XI0 = np.log(0.15 / 0.85)
KAPPA = 100.0
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
    """Local-slab counterpart of stability_selection_validation.make_fit_selector."""
    def fit_selector(subset_idx):
        cols = np.concatenate([[0], subset_idx + 1])
        Zsub = Zd_full[:, cols]
        wr_sub = wr_full[subset_idx]
        res = fit_em_gaussian(Zsub, y, wr_sub, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0)
        return np.where(res.theta_hat > inner_thresh)[0]
    return fit_selector


OUTER_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
GIBBS_GAMMA_GRID = (1.0 - OUTER_GRID) / OUTER_GRID  # centroid_estimate(pi_hat, gamma) <=> pi_hat >= t


def run(n_replicates=15, n_subsets=50, subsample_frac=0.5, inner_thresh_grid=(0.02, 0.05, 0.1),
        gibbs_n_samples=2000, gibbs_burn_in=500):
    rng_master = np.random.default_rng(0)
    rows = []
    t_start = time.time()
    for rep in range(n_replicates):
        X_raw, block_of, y, causal_blocks = simulate(rng_master)
        Z = compress(X_raw, block_of)
        Zd = np.column_stack([np.ones(N), Z])
        wr = np.ones(P)

        # local-slab EM baselines: direct threshold + PPL-filter best step, single full fit
        res_full = fit_em_gaussian(Zd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0)
        for t in (0.1, 0.3, 0.5):
            retained = set(np.where(res_full.theta_hat > t)[0].tolist())
            power, precision, n_ret = power_precision(retained, causal_blocks)
            rows.append(dict(rep=rep, config=f"local_baseline_threshold_{t}", power=power,
                              precision=precision, n_retained=n_ret))
        filt = em_filter_gaussian(Zd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0,
                                   filter_frac=0.2, min_features=4, max_outer=30)
        retained = set(filt.best.retained_idx.tolist())
        power, precision, n_ret = power_precision(retained, causal_blocks)
        rows.append(dict(rep=rep, config="local_baseline_PPLfilter", power=power,
                          precision=precision, n_retained=n_ret))

        # local-slab stability selection, sweep inner threshold, fixed subsample_frac/n_subsets
        rng_stab = np.random.default_rng(1000 + rep)
        for inner_thresh in inner_thresh_grid:
            fit_selector = make_fit_selector(Zd, y, wr, inner_thresh)
            stab = stability_select(fit_selector, P, n_subsets=n_subsets,
                                     subsample_frac=subsample_frac, rng=rng_stab)
            for outer_t in OUTER_GRID:
                admitted = admit(stab, outer_t)
                retained = set(admitted.tolist())
                power, precision, n_ret = power_precision(retained, causal_blocks)
                rows.append(dict(rep=rep, config=f"local_stability_inner{inner_thresh}_admitonly",
                                  outer_thresh=outer_t, power=power, precision=precision, n_retained=n_ret))

        # local-slab Gibbs: ONE full-posterior fit per replicate, then sweep the decision
        # threshold via centroid_estimate -- no per-threshold refitting needed.
        gr = gibbs_sampler_gaussian(Zd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=1.0, lam=1.0,
                                     n_samples=gibbs_n_samples, burn_in=gibbs_burn_in,
                                     seed=2000 + rep)
        for outer_t, gamma in zip(OUTER_GRID, GIBBS_GAMMA_GRID):
            selected = centroid_estimate(gr.pi_hat, gamma=gamma)
            retained = set(np.where(selected)[0].tolist())
            power, precision, n_ret = power_precision(retained, causal_blocks)
            rows.append(dict(rep=rep, config="gibbs_centroid", outer_thresh=outer_t,
                              power=power, precision=precision, n_retained=n_ret))

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
    df.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/b69659aa-1692-48e7-a136-db8f15be2200/scratchpad/gibbs_vs_stability_raw.csv", index=False)
    summary.to_csv("/tmp/claude-1000/-home-ian-Research-Math-prompts/b69659aa-1692-48e7-a136-db8f15be2200/scratchpad/gibbs_vs_stability_summary.csv", index=False)
