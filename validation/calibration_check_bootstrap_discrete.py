"""Does bootstrap_ci also fix coverage on the discrete/Binomial branch
(hierboost.latent, decorrelate="sar" via HierBoostClassifier)? Reuses the exact
self-consistent generative DGP already validated in calibration_check_joint.py
(known Z_true -> BlockLatentFit's own CT link -> Binomial genotypes -> Bernoulli y),
and its already-established asymptotic reference (gamma_true=0.15 Z-units -> 0.430 on
the fitted scale) as ground truth -- both established and justified in that script's
own docstring/conversation record, not re-derived here.

Run inside .venv-jax:
    .venv-jax/bin/python3 calibration_check_bootstrap_discrete.py [n_reps] [n_boot]
"""
import sys
import time
import numpy as np

from calibration_check_joint import simulate_dataset, COORDS, BLOCK_ID, N_TRIALS, XI0, BETA_TRUE
from hierboost.estimator import HierBoostClassifier
from hierboost.calibration import check_ci_coverage

CI = 0.95


def fit_and_report(rng, n=400, n_boot=30):
    X, y = simulate_dataset(rng, n)
    clf = HierBoostClassifier(decorrelate="sar", fit_method="em", xi0=XI0, xi1=0.0, kappa=100.0)
    clf.fit(X, y, coords=COORDS, block_id=BLOCK_ID, n_trials=N_TRIALS)
    result = clf.bootstrap_ci(X, y, coords=COORDS, block_id=BLOCK_ID, n_trials=N_TRIALS,
                               n_boot=n_boot, ci=CI, random_state=int(rng.integers(1_000_000_000)))
    return BETA_TRUE, result["lo"], result["hi"]


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = np.random.default_rng(0)
    t0 = time.time()
    for r in range(n_reps):
        beta_true, lo, hi = fit_and_report(rng, n_boot=n_boot)
        print(f"rep {r+1}/{n_reps} done ({time.time()-t0:.0f}s elapsed) "
              f"causal_lo={lo[0]:.3f} causal_hi={hi[0]:.3f} true={beta_true[0]:.3f}", flush=True)
