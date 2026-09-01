"""Does cross-fitting (hierboost.latent.fit_latent_block_model_crossfit) restore
credible-interval coverage relative to the vanilla plug-in? Reuses the exact same
self-consistent generative DGP as calibration_check_joint.py (same FITS/COORDS/
BLOCK_ID/BETA_TRUE), so results are directly comparable to that script's numbers
(vanilla plug-in: causal coverage 55.0%, null 83.8%, n=20 reps; joint: 50.0%/100.0%).

The reference truth (0.430 for the causal block) does not need to be re-derived for
cross-fitting: it reflects the true data-generating process as related to Z-tilde
through BlockLatentFit's own CT link, common to any estimator targeting the same
quantity -- confirmed directly (see conversation) that vanilla plug-in's finite-sample
mean beta_hat (0.422, n=400) already closely tracks the n=8000 asymptotic value (0.430),
i.e. the vanilla plug-in's problem is centered on interval width/shape, not point-
estimate location, so the same reference is the right target for cross-fit too.

No NUTS here -- cross-fitting only touches the EM/Newton plug-in pipeline, ~3x the cost
of one vanilla fit_latent_block_model call (two half-sample fits + one full-sample fit
for the returned fits/deltas), so more replications are affordable than the joint.py
comparison.

Run inside .venv-jax:
    .venv-jax/bin/python3 calibration_check_crossfit.py [n_reps]
"""
import time
import sys
import numpy as np
from scipy.stats import norm as _norm

from calibration_check_joint import simulate_dataset, COORDS, BLOCK_ID, N_TRIALS, XI0, K_BLOCKS, CAUSAL_BLOCKS, BETA_TRUE
from hierboost.latent import fit_latent_block_model_crossfit, fit_latent_block_model
from hierboost.spike_slab import _precision_from_theta
from hierboost.calibration import check_ci_coverage

CI = 0.95
Z_CRIT = _norm.ppf((1 + CI) / 2)
KAPPA = 100.0
WR = np.zeros(K_BLOCKS)


def _beta_se_from_em(em_result, X_design):
    """Same Laplace-approximation SE estimator.py's _compute_approx_covariance uses,
    applied directly to a raw hierboost.latent em_result (not going through
    HierBoostClassifier, since fit_latent_block_model_crossfit is a new lower-level
    function not yet wired into the estimator API -- matching the project's existing
    convention of validating a new mechanism at the functional-API level first, e.g.
    newsgroups_demo.py)."""
    from scipy.special import expit
    mu = expit(X_design @ em_result.beta)
    precision = _precision_from_theta(em_result.theta_hat, KAPPA)  # length K+1, incl. intercept
    sigma_diag = em_result.sigma2 / precision
    Wt = np.clip(mu * (1.0 - mu), 1e-6, None)
    XtWX = X_design.T @ (Wt[:, None] * X_design)
    cov = np.linalg.inv(XtWX + np.diag(1.0 / sigma_diag))
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def one_replication(rng, n, n_outer=15, n_inner_newton=8, hyper_n_steps=150):
    X, y = simulate_dataset(rng, n)

    result_vanilla = fit_latent_block_model(X, y, COORDS, BLOCK_ID, WR, XI0, 0.0, KAPPA, 1.0, 1.0,
                                             n_trials=N_TRIALS, n_outer=n_outer,
                                             n_inner_newton=n_inner_newton, hyper_n_steps=hyper_n_steps,
                                             seed=int(rng.integers(1_000_000_000)))
    Xd_vanilla = np.column_stack([np.ones(n), result_vanilla["Z"]])
    beta_v = result_vanilla["em_result"].beta[1:]
    se_v = _beta_se_from_em(result_vanilla["em_result"], Xd_vanilla)[1:]

    result_cf = fit_latent_block_model_crossfit(X, y, COORDS, BLOCK_ID, WR, XI0, 0.0, KAPPA, 1.0, 1.0,
                                                  n_trials=N_TRIALS, n_outer=n_outer,
                                                  n_inner_newton=n_inner_newton, hyper_n_steps=hyper_n_steps,
                                                  seed=int(rng.integers(1_000_000_000)), n_folds=2)
    Xd_cf = np.column_stack([np.ones(n), result_cf["Z"]])
    beta_cf = result_cf["em_result"].beta[1:]
    se_cf = _beta_se_from_em(result_cf["em_result"], Xd_cf)[1:]

    return (beta_v, beta_v - Z_CRIT * se_v, beta_v + Z_CRIT * se_v), \
           (beta_cf, beta_cf - Z_CRIT * se_cf, beta_cf + Z_CRIT * se_cf)


if __name__ == "__main__":
    n = 400
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    print(f"K={K_BLOCKS} blocks, n={n}, causal block={CAUSAL_BLOCKS[0]}, "
          f"reference true gamma={BETA_TRUE[CAUSAL_BLOCKS[0]]}, {n_reps} paired reps "
          f"(vanilla plug-in vs 2-fold cross-fit, same data each rep)")

    rng = np.random.default_rng(0)
    v_betas, v_los, v_his, cf_betas, cf_los, cf_his = [], [], [], [], [], []
    t0 = time.time()
    for r in range(n_reps):
        (bv, lv, hv), (bcf, lcf, hcf) = one_replication(rng, n)
        v_betas.append(bv); v_los.append(lv); v_his.append(hv)
        cf_betas.append(bcf); cf_los.append(lcf); cf_his.append(hcf)
        print(f"  rep {r + 1}/{n_reps}  vanilla_causal={bv[CAUSAL_BLOCKS[0]]:+.3f}  "
              f"crossfit_causal={bcf[CAUSAL_BLOCKS[0]]:+.3f}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    def report_v(_rng, i=[0]):
        j = i[0]; i[0] += 1
        return BETA_TRUE, v_los[j], v_his[j]

    def report_cf(_rng, i=[0]):
        j = i[0]; i[0] += 1
        return BETA_TRUE, cf_los[j], cf_his[j]

    cov_v = check_ci_coverage(report_v, n_reps=n_reps, ci=CI, seed=0)
    cov_cf = check_ci_coverage(report_cf, n_reps=n_reps, ci=CI, seed=0)

    v_betas, cf_betas = np.array(v_betas), np.array(cf_betas)
    print(f"\n{'=' * 70}\nRESULTS ({n_reps} paired reps, nominal {CI:.0%} CI, "
          f"true causal gamma reference={BETA_TRUE[CAUSAL_BLOCKS[0]]})\n{'=' * 70}")
    print(f"Vanilla plug-in causal beta_hat: mean={v_betas[:, CAUSAL_BLOCKS[0]].mean():.3f} "
          f"std={v_betas[:, CAUSAL_BLOCKS[0]].std():.3f}")
    print(f"  causal coverage={cov_v.coverage_causal:.3f} (n={cov_v.n_causal}), "
          f"null coverage={cov_v.coverage_null:.3f} (n={cov_v.n_null}), "
          f"width causal={cov_v.mean_ci_width_causal:.3f}")
    print(f"Cross-fit (2-fold) causal beta_hat: mean={cf_betas[:, CAUSAL_BLOCKS[0]].mean():.3f} "
          f"std={cf_betas[:, CAUSAL_BLOCKS[0]].std():.3f}")
    print(f"  causal coverage={cov_cf.coverage_causal:.3f} (n={cov_cf.n_causal}), "
          f"null coverage={cov_cf.coverage_null:.3f} (n={cov_cf.n_null}), "
          f"width causal={cov_cf.mean_ci_width_causal:.3f}")
