"""Final validation of _HierBoostBase.bootstrap_ci on the SAME decorrelate="sar",
Gaussian, EM setup calibration_check.py's original "KEY TEST" used (causal coverage
~59-63% with the naive/closed-form-corrected Laplace SE). Dogfoods the actual wired-in
API (not a standalone hand-rolled loop) at a higher replication count for a trustworthy
number.
"""
import numpy as np
from hierboost.estimator import HierBoostRegressor
from hierboost.calibration import check_ci_coverage

M_PER_BLOCK, N_BLOCKS = 4, 3
COORDS = np.concatenate([np.arange(M_PER_BLOCK, dtype=float) + b * 100 for b in range(N_BLOCKS)])
BETA_TRUE = np.array([2.0, 0.0, -1.5])


def gen(n, rng):
    latents = rng.normal(size=(n, N_BLOCKS))
    X = np.zeros((n, M_PER_BLOCK * N_BLOCKS))
    for b in range(N_BLOCKS):
        loadings = rng.normal(1, 0.3, M_PER_BLOCK)
        X[:, b * M_PER_BLOCK:(b + 1) * M_PER_BLOCK] = (
            np.outer(latents[:, b], loadings) + rng.normal(0, 0.3, (n, M_PER_BLOCK)))
    y = latents @ BETA_TRUE + rng.normal(0, 0.5, n)
    return X, y


def fit_and_report(rng, n=400, n_boot=150):
    X, y = gen(n, rng)
    reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=COORDS)
    result = reg.bootstrap_ci(X, y, coords=COORDS, n_boot=n_boot,
                               random_state=int(rng.integers(1_000_000_000)))
    return BETA_TRUE, result["lo"], result["hi"]


if __name__ == "__main__":
    res = check_ci_coverage(fit_and_report, n_reps=60, ci=0.95, seed=0)
    print(f"bootstrap_ci coverage (60 reps, 150 boot draws each): "
          f"causal={res.coverage_causal:.3f} (n={res.n_causal}), "
          f"null={res.coverage_null:.3f} (n={res.n_null}), "
          f"width causal={res.mean_ci_width_causal:.3f}, width null={res.mean_ci_width_null:.3f}")
