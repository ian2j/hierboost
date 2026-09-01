"""Simulation-based calibration/coverage checks for hierboost's reported uncertainty
(`theta_hat`, and the credible intervals `estimator.py`'s `.summary()`/`.beta_se_`/
`.beta_cov_` compute) -- validates a claim the practitioner API already makes rather than
adding a new modeling capability.

Motivated by a real, specific concern already implicit in this project's own structure:
`joint.py` exists because the dissertation itself flags the two-stage plug-in pipeline
(fit block latents by Newton/EM, then treat them as fixed known data for the outcome
regression) as a known limitation ("a fully Bayesian approach is impractical") -- that
concern has never actually been empirically tested here. If it's real, plug-in
`decorrelate="sar"/"ar1"` fits should UNDER-cover (report CIs narrower than their nominal
level) relative to the no-decorrelate baseline, because the latent factor's own
estimation uncertainty from the first stage never propagates into the second stage's
reported CIs.

Two distinct, standard properties for a Bayesian variable-selection method:
- Credible interval coverage: repeatedly simulate data from a KNOWN ground truth, fit,
  and check whether the true coefficient falls inside the reported (1-alpha) CI at
  close to the nominal rate.
- theta_hat calibration: among all (feature, replication) pairs with reported posterior
  inclusion probability near some value p, is the TRUE inclusion rate also near p (a
  reliability-diagram / Brier-score check, the same sense classifier probabilities are
  calibrated in).

Both functions are deliberately estimator-agnostic: the caller supplies a small closure
that generates one replication's data, fits whatever hierboost configuration is being
tested, and extracts the ground truth plus the reported summary -- this file only knows
how to aggregate and score the results, not how to run any particular fit.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CoverageResult:
    ci: float
    coverage_causal: float
    coverage_null: float
    coverage_overall: float
    n_causal: int
    n_null: int
    mean_ci_width_causal: float
    mean_ci_width_null: float


def check_ci_coverage(fit_and_report, n_reps=200, ci=0.95, seed=0):
    """fit_and_report(rng) -> (beta_true, lo, hi), each a length-p array (no intercept),
    where beta_true is 0 for a truly-null feature/block and nonzero for a truly-causal
    one, and (lo, hi) is the reported (1-ci)% credible interval. Runs n_reps independent
    replications and aggregates empirical coverage, reported separately for causal vs.
    null features -- a spike-and-slab prior can calibrate very differently for the two
    (shrinkage bias can make a real-but-weak effect's CI overconfidently exclude it,
    which coverage_causal alone would catch even if coverage_null looked fine).
    """
    rng = np.random.default_rng(seed)
    covered_causal, covered_null = [], []
    width_causal, width_null = [], []
    for _ in range(n_reps):
        beta_true, lo, hi = fit_and_report(rng)
        beta_true, lo, hi = np.asarray(beta_true), np.asarray(lo), np.asarray(hi)
        causal = beta_true != 0
        inside = (beta_true >= lo) & (beta_true <= hi)
        width = hi - lo
        covered_causal.append(inside[causal])
        covered_null.append(inside[~causal])
        width_causal.append(width[causal])
        width_null.append(width[~causal])
    covered_causal = np.concatenate(covered_causal)
    covered_null = np.concatenate(covered_null)
    width_causal = np.concatenate(width_causal)
    width_null = np.concatenate(width_null)
    all_covered = np.concatenate([covered_causal, covered_null])
    n_c, n_n = len(covered_causal), len(covered_null)
    return CoverageResult(
        ci=ci,
        coverage_causal=float(covered_causal.mean()) if n_c else float("nan"),
        coverage_null=float(covered_null.mean()) if n_n else float("nan"),
        coverage_overall=float(all_covered.mean()) if len(all_covered) else float("nan"),
        n_causal=n_c, n_null=n_n,
        mean_ci_width_causal=float(width_causal.mean()) if n_c else float("nan"),
        mean_ci_width_null=float(width_null.mean()) if n_n else float("nan"),
    )


@dataclass
class ReliabilityResult:
    bin_edges: np.ndarray
    bin_mean_theta: np.ndarray
    bin_empirical_rate: np.ndarray
    bin_count: np.ndarray
    ece: float   # expected calibration error: count-weighted mean |theta - empirical rate|
    brier: float  # mean squared error of theta_hat against the 0/1 true-inclusion label


def check_theta_calibration(fit_and_report_theta, n_reps=200, n_bins=10, seed=0):
    """fit_and_report_theta(rng) -> (is_causal, theta_hat), both length-p boolean/float
    arrays, across n_reps replications. Bins ALL (replication, feature) pairs by
    theta_hat and compares each bin's mean theta_hat to its empirical true-inclusion
    rate (a reliability diagram), plus overall ECE and Brier score summaries.
    """
    rng = np.random.default_rng(seed)
    all_causal, all_theta = [], []
    for _ in range(n_reps):
        is_causal, theta_hat = fit_and_report_theta(rng)
        all_causal.append(np.asarray(is_causal))
        all_theta.append(np.asarray(theta_hat))
    is_causal = np.concatenate(all_causal).astype(float)
    theta_hat = np.concatenate(all_theta)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta_hat, edges[1:-1]), 0, n_bins - 1)
    bin_mean_theta = np.full(n_bins, np.nan)
    bin_empirical_rate = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        bin_count[b] = mask.sum()
        if mask.any():
            bin_mean_theta[b] = theta_hat[mask].mean()
            bin_empirical_rate[b] = is_causal[mask].mean()

    valid = bin_count > 0
    ece = float(np.sum(bin_count[valid] * np.abs(bin_mean_theta[valid] - bin_empirical_rate[valid]))
                / bin_count[valid].sum())
    brier = float(np.mean((theta_hat - is_causal) ** 2))
    return ReliabilityResult(bin_edges=edges, bin_mean_theta=bin_mean_theta,
                              bin_empirical_rate=bin_empirical_rate, bin_count=bin_count,
                              ece=ece, brier=brier)
