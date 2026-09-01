"""Quick check: does a parametric-ish bootstrap (resample individuals, refit BOTH the
PPCA loadings AND gamma each time) close the remaining coverage gap that the closed-form
z_var correction only partially closed? If so, confirms the missing piece is loading/
obs_var estimation uncertainty (not captured by conditioning on them as fixed), not a
new, third mechanism.
"""
import numpy as np
from scipy.stats import norm as _norm
from hierboost.estimator import HierBoostRegressor
from hierboost.calibration import check_ci_coverage

CI = 0.95
Z_CRIT = _norm.ppf((1 + CI) / 2)


def gen(n, seed, m_per_block=4, n_blocks=3):
    r = np.random.default_rng(seed)
    latents = r.normal(size=(n, n_blocks))
    X = np.zeros((n, m_per_block * n_blocks))
    for b in range(n_blocks):
        loadings = r.normal(1, 0.3, m_per_block)
        X[:, b * m_per_block:(b + 1) * m_per_block] = (
            np.outer(latents[:, b], loadings) + r.normal(0, 0.3, (n, m_per_block)))
    y = latents @ np.array([2.0, 0.0, -1.5]) + r.normal(0, 0.5, n)
    return X, y


def fit_and_report_bootstrap(rng, n=400, n_boot=300):
    coords = np.concatenate([np.arange(4, dtype=float) + b * 100 for b in range(3)])
    seed = int(rng.integers(1_000_000_000))
    X, y = gen(n, seed)
    reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    beta_hat = reg.beta_[1:]

    boot_betas = np.zeros((n_boot, len(beta_hat)))
    brng = np.random.default_rng(seed + 999)
    for bi in range(n_boot):
        idx = brng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        try:
            regb = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
            regb.fit(Xb, yb, coords=coords)
            boot_betas[bi] = regb.beta_[1:]
        except Exception:
            boot_betas[bi] = beta_hat
    lo = np.percentile(boot_betas, 2.5, axis=0)
    hi = np.percentile(boot_betas, 97.5, axis=0)
    beta_true = np.array([2.0, 0.0, -1.5])
    return beta_true, lo, hi


if __name__ == "__main__":
    res = check_ci_coverage(fit_and_report_bootstrap, n_reps=30, ci=CI, seed=0)
    print(f"Bootstrap coverage: causal={res.coverage_causal:.3f} (n={res.n_causal}), "
          f"null={res.coverage_null:.3f} (n={res.n_null}), width causal={res.mean_ci_width_causal:.3f}")
