"""The empirical study hierboost.calibration's utilities exist to run: is hierboost's
reported uncertainty (credible intervals, theta_hat posterior inclusion probabilities)
actually honest? Never checked anywhere in this project before -- the practitioner API
(estimator.py's `.summary()`) reports these numbers as if they mean something, but
nothing has verified they do.

Two configurations, in order of how interesting the answer is:
1. **Baseline** (`decorrelate=None`, flat spike-and-slab): the simplest case, and the
   right place to confirm the fundamental CI/theta_hat machinery is sound before
   layering on block-latent complexity. Gaussian + Binomial, EM (Laplace-approximation
   covariance) + Gibbs (empirical posterior covariance) -- four combinations, since the
   two response families and two fit methods use genuinely different uncertainty
   mechanisms internally (_compute_approx_covariance's Laplace branch vs. np.cov on the
   Gibbs draws).
2. **The actual motivating question** (`decorrelate="sar"`): does the two-stage
   plug-in pipeline (fit the block latent, then treat it as fixed known data for the
   outcome regression) understate uncertainty, the way the dissertation itself already
   flags as a known limitation of the plug-in approach ("a fully Bayesian approach is
   impractical") -- the reason `joint.py`'s full NUTS fit exists? If real, this should
   show up as UNDER-coverage (empirical coverage below the 95% nominal rate) relative to
   the baseline, since `latent.py`/`factor.py`'s own estimation uncertainty from stage 1
   never propagates into stage 2's reported `beta_se_`.

Uses fit_method="em"/"gibbs" only (not "em_filter"): em_filter can drop features between
replications, which would misalign the fixed-length true-vs-reported comparison this
script relies on -- a real limitation of this check (see write-up), not worked around
here since em/gibbs already answer the calibration question directly.
"""
import time
import numpy as np
from scipy.stats import norm as _norm

from hierboost.estimator import HierBoostRegressor, HierBoostClassifier
from hierboost.calibration import check_ci_coverage, check_theta_calibration

CI = 0.95
Z = _norm.ppf((1 + CI) / 2)


def report(label, cov, rel):
    print(f"\n--- {label} ---")
    print(f"  CI coverage (nominal {CI:.0%}): causal={cov.coverage_causal:.3f} "
          f"(n={cov.n_causal}), null={cov.coverage_null:.3f} (n={cov.n_null}), "
          f"overall={cov.coverage_overall:.3f}")
    print(f"  mean CI width: causal={cov.mean_ci_width_causal:.3f}, null={cov.mean_ci_width_null:.3f}")
    print(f"  theta_hat calibration: ECE={rel.ece:.3f}, Brier={rel.brier:.3f}")
    print(f"  reliability table (bin -> mean theta_hat | empirical rate | count):")
    for lo, hi, mt, er, c in zip(rel.bin_edges[:-1], rel.bin_edges[1:],
                                  rel.bin_mean_theta, rel.bin_empirical_rate, rel.bin_count):
        if c > 0:
            print(f"    [{lo:.1f},{hi:.1f}) -> {mt:.3f} | {er:.3f} | n={c}")


# ---- Configuration 1: flat (no decorrelate) --------------------------------------

def make_flat_closures(response, fit_method, n=300, p=30,
                        causal_idx=(2, 7, 15, 21, 27), causal_beta=(1.8, -2.0, 1.5, -1.6, 2.2)):
    beta_true = np.zeros(p)
    for idx, b in zip(causal_idx, causal_beta):
        beta_true[idx] = b
    xi0 = np.log(len(causal_idx) / p / (1 - len(causal_idx) / p))
    cls = HierBoostRegressor if response == "gaussian" else HierBoostClassifier

    def one_fit(rng):
        X = rng.normal(size=(n, p))
        if response == "gaussian":
            y = X @ beta_true + rng.normal(scale=1.0, size=n)
        else:
            from scipy.special import expit
            y = (rng.random(n) < expit(X @ beta_true)).astype(float)
        reg = cls(fit_method=fit_method, xi0=xi0, kappa=100.0,
                   n_samples=1500, burn_in=500, random_state=int(rng.integers(1_000_000_000)))
        reg.fit(X, y)
        return reg

    def fit_and_report(rng):
        reg = one_fit(rng)
        beta_hat, se = reg.beta_[1:], reg.beta_se_[1:]
        return beta_true, beta_hat - Z * se, beta_hat + Z * se

    def fit_and_report_theta(rng):
        reg = one_fit(rng)
        return beta_true != 0, reg.theta_hat_

    return fit_and_report, fit_and_report_theta


# ---- Configuration 2: decorrelate="sar" (block-latent plug-in) -------------------

def make_sar_closures(fit_method, n=400, m_per_block=4, n_blocks=6,
                       causal_blocks=(0, 3), causal_beta=(2.0, -1.6)):
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])
    beta_true = np.zeros(n_blocks)
    for b, val in zip(causal_blocks, causal_beta):
        beta_true[b] = val
    xi0 = np.log(len(causal_blocks) / n_blocks / (1 - len(causal_blocks) / n_blocks))

    def one_fit(rng):
        latents = rng.normal(size=(n, n_blocks))
        X = np.zeros((n, m_per_block * n_blocks))
        for b in range(n_blocks):
            loadings = rng.normal(1, 0.3, m_per_block)
            X[:, b * m_per_block:(b + 1) * m_per_block] = (
                np.outer(latents[:, b], loadings) + rng.normal(0, 0.3, (n, m_per_block)))
        y = latents @ beta_true + rng.normal(0, 0.5, n)
        reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method=fit_method, xi0=xi0, kappa=100.0,
                                  n_samples=1500, burn_in=500, random_state=int(rng.integers(1_000_000_000)))
        reg.fit(X, y, coords=coords)
        return reg

    def fit_and_report(rng):
        reg = one_fit(rng)
        beta_hat, se = reg.beta_[1:], reg.beta_se_[1:]
        return beta_true, beta_hat - Z * se, beta_hat + Z * se

    def fit_and_report_theta(rng):
        reg = one_fit(rng)
        return beta_true != 0, reg.theta_hat_

    return fit_and_report, fit_and_report_theta


if __name__ == "__main__":
    configs = [
        ("Baseline: no-decorrelate, Gaussian, EM", *make_flat_closures("gaussian", "em"), 300),
        ("Baseline: no-decorrelate, Gaussian, Gibbs", *make_flat_closures("gaussian", "gibbs"), 60),
        ("Baseline: no-decorrelate, Binomial, EM", *make_flat_closures("binomial", "em"), 300),
        ("KEY TEST: decorrelate='sar', Gaussian, EM (plug-in)", *make_sar_closures("em"), 200),
        ("KEY TEST: decorrelate='sar', Gaussian, Gibbs (plug-in)", *make_sar_closures("gibbs"), 50),
    ]
    for label, fit_and_report, fit_and_report_theta, n_reps in configs:
        t0 = time.time()
        cov = check_ci_coverage(fit_and_report, n_reps=n_reps, ci=CI, seed=0)
        rel = check_theta_calibration(fit_and_report_theta, n_reps=n_reps, n_bins=10, seed=1)
        report(f"{label}  ({n_reps} reps, {time.time() - t0:.0f}s)", cov, rel)
