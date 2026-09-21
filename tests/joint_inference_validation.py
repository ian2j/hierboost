"""First calibration check of fit_method='joint' (hierboost.joint's NUTS/regularized-
horseshoe pipeline, wired into HierBoostClassifier 2026-09-21) against the plug-in EM
pipeline (fit_latent_block_model) it's meant to replace, for decorrelate="ar1" +
Binomial response.

NOTE on scope: the existing ~59%/63%/88% coverage numbers referenced throughout
estimator.py's docstrings (_fit_joint, _compute_approx_covariance, bootstrap_ci) are
for the GAUSSIAN decorrelate="sar"/"ar1" path (factor.py/state_space.py's PPCA/RTS-
smoother pipeline) -- a structurally different plug-in pipeline from the Binomial/JAX
one (hierboost.latent) that fit_method='joint' actually replaces. There is no existing
Binomial-specific coverage benchmark in this repo to reproduce; this script establishes
the first one, rather than replaying an old number.

Single-block AR(1) DGP (mirrors test_estimator.py's
test_ar1_zvar_correction_widens_se_and_improves_coverage, adapted from Gaussian y to
Binomial X *and* Bernoulli y): one shared AR(1) latent z_t drives both the block's raw
Binomial(n_trials) features (via fixed loadings) and a Bernoulli y (via a known
gamma_true) -- exactly the quasi-complete-separation setup newton_update_ztilde's
"KNOWN LIMITATION" docstring describes (resp_term makes Z-tilde fit y directly, so the
outer EM alternation can drive gamma to a large, physically implausible value).
"""
import time
import numpy as np

from hierboost.estimator import HierBoostClassifier

M, N, N_REPS = 4, 250, 20
TRUE_RHO = 0.7
LOADINGS_TRUE = np.array([1.0, 0.8, -0.6, 1.2])
GAMMA_TRUE = 1.3
N_TRIALS = 2
COORDS = np.arange(M, dtype=float)
BLOCK_ID = np.zeros(M, dtype=int)
Z_CRIT = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def simulate(rng):
    z = np.zeros(N)
    sv = 1 - TRUE_RHO ** 2
    for t in range(1, N):
        z[t] = TRUE_RHO * z[t - 1] + rng.normal(0, np.sqrt(sv))
    p = 1 / (1 + np.exp(-(np.outer(z, LOADINGS_TRUE) + rng.normal(0, 0.4, (N, M)))))
    X = rng.binomial(N_TRIALS, np.clip(p, 0.01, 0.99)).astype(float)
    y = rng.binomial(1, 1 / (1 + np.exp(-(GAMMA_TRUE * z))))
    return X, y


def fit_and_report(rng, fit_method, **fit_kwargs):
    X, y = simulate(rng)
    clf = HierBoostClassifier(decorrelate="ar1", fit_method=fit_method,
                               random_state=int(rng.integers(0, 2**31 - 1)), **fit_kwargs)
    clf.fit(X, y, coords=COORDS, block_id=BLOCK_ID, n_trials=N_TRIALS)
    beta, se = clf.beta_[1], clf.beta_se_[1]
    lo, hi = beta - Z_CRIT * se, beta + Z_CRIT * se
    return np.array([GAMMA_TRUE]), np.array([lo]), np.array([hi]), beta


def run():
    results = {"em": [], "joint": []}
    t0 = time.time()
    for rep in range(N_REPS):
        # same seed for both methods' rng -> identical simulated dataset this rep
        seed = 1000 + rep
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)

        beta_true, lo_em, hi_em, beta_em = fit_and_report(r1, "em")
        beta_true, lo_joint, hi_joint, beta_joint = fit_and_report(r2, "joint",
                                                                    n_samples=500, burn_in=200)
        results["em"].append((beta_em, lo_em[0], hi_em[0]))
        results["joint"].append((beta_joint, lo_joint[0], hi_joint[0]))
        print(f"rep {rep}: em beta={beta_em:.3f} [{lo_em[0]:.3f},{hi_em[0]:.3f}]  "
              f"joint beta={beta_joint:.3f} [{lo_joint[0]:.3f},{hi_joint[0]:.3f}]  "
              f"elapsed={time.time()-t0:.1f}s")

    print("\n" + "=" * 90)
    print(f"{N_REPS} replicates, single-block AR(1) Binomial DGP, true gamma={GAMMA_TRUE}, "
          f"95% nominal CI")
    print("=" * 90)
    print(f"{'method':<10}{'coverage':>10}{'mean_width':>12}{'mean_beta':>12}{'median_beta':>13}{'beta_std':>10}")
    summary = {}
    for method, vals in results.items():
        vals = np.array(vals)  # (n_reps, 3): beta_hat, lo, hi
        covered = (GAMMA_TRUE >= vals[:, 1]) & (GAMMA_TRUE <= vals[:, 2])
        width = vals[:, 2] - vals[:, 1]
        summary[method] = dict(coverage=covered.mean(), mean_width=width.mean(),
                                mean_beta=vals[:, 0].mean(), median_beta=np.median(vals[:, 0]),
                                beta_std=vals[:, 0].std())
        s = summary[method]
        print(f"{method:<10}{s['coverage']:>10.3f}{s['mean_width']:>12.3f}{s['mean_beta']:>12.3f}"
              f"{s['median_beta']:>13.3f}{s['beta_std']:>10.3f}")
    return results, summary


if __name__ == "__main__":
    run()
