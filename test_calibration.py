"""Sanity checks for hierboost.calibration's aggregation/scoring arithmetic itself --
using known-answer toy closures, not a real hierboost fit (that's the empirical study in
calibration_check.py, a statistical finding to read and interpret, not a pass/fail unit
test). Confirms the coverage/ECE/Brier machinery computes what it claims to before
trusting it on the real estimator.
"""
import numpy as np

from hierboost.calibration import check_ci_coverage, check_theta_calibration


def test_coverage_is_one_when_intervals_always_contain_truth():
    def fit_and_report(rng):
        beta_true = np.array([0.0, 1.5, 0.0, -2.0])
        return beta_true, beta_true - 1.0, beta_true + 1.0

    res = check_ci_coverage(fit_and_report, n_reps=20, ci=0.95, seed=0)
    assert res.coverage_causal == 1.0
    assert res.coverage_null == 1.0
    assert res.coverage_overall == 1.0
    assert res.n_causal == 40  # 2 causal features x 20 reps
    assert res.n_null == 40


def test_coverage_is_zero_when_intervals_never_contain_truth():
    def fit_and_report(rng):
        beta_true = np.array([0.0, 1.5, -2.0])
        lo = beta_true + 5.0  # interval entirely above the truth
        hi = beta_true + 6.0
        return beta_true, lo, hi

    res = check_ci_coverage(fit_and_report, n_reps=10, ci=0.95, seed=0)
    assert res.coverage_causal == 0.0
    assert res.coverage_null == 0.0


def test_coverage_matches_nominal_rate_for_a_genuinely_correct_gaussian_ci():
    p = 20
    beta_true = np.zeros(p)
    beta_true[:5] = 2.0  # 5 causal, 15 null

    def fit_and_report(rng):
        beta_hat = beta_true + rng.normal(size=p)  # unit-variance sampling noise
        z = 1.959963985  # exact 95% normal critical value
        return beta_true, beta_hat - z, beta_hat + z

    res = check_ci_coverage(fit_and_report, n_reps=2000, ci=0.95, seed=1)
    # a correctly-constructed 95% CI should cover close to 95% of the time -- generous
    # tolerance since this is itself a Monte Carlo estimate (n=2000*20 draws total)
    assert abs(res.coverage_overall - 0.95) < 0.02
    assert abs(res.coverage_causal - 0.95) < 0.03
    assert abs(res.coverage_null - 0.95) < 0.03


def test_ece_and_brier_are_zero_for_perfectly_calibrated_theta():
    def fit_and_report_theta(rng):
        is_causal = np.array([True, True, False, False, False])
        theta_hat = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        return is_causal, theta_hat

    res = check_theta_calibration(fit_and_report_theta, n_reps=10, n_bins=10, seed=0)
    assert res.ece < 1e-9
    assert res.brier < 1e-9
    assert res.bin_count.sum() == 50  # 5 features x 10 reps


def test_ece_detects_a_known_miscalibration():
    # theta_hat is always 0.9 regardless of truth, but the true inclusion rate is only 0.5
    # -- a textbook overconfident classifier, ECE should land near |0.9 - 0.5| = 0.4
    def fit_and_report_theta(rng):
        is_causal = rng.random(200) < 0.5
        theta_hat = np.full(200, 0.9)
        return is_causal, theta_hat

    res = check_theta_calibration(fit_and_report_theta, n_reps=20, n_bins=10, seed=2)
    assert abs(res.ece - 0.4) < 0.05
    # all mass should land in theta's own bin
    populated_bins = np.where(res.bin_count > 0)[0]
    assert len(populated_bins) == 1


def test_reliability_bin_mean_theta_tracks_a_uniform_theta_generator():
    def fit_and_report_theta(rng):
        theta_hat = rng.uniform(0, 1, size=500)
        is_causal = rng.random(500) < theta_hat  # theta_hat IS the true probability here
        return is_causal, theta_hat

    res = check_theta_calibration(fit_and_report_theta, n_reps=20, n_bins=10, seed=3)
    valid = res.bin_count > 0
    # a genuinely well-specified theta_hat should track the empirical rate within each bin
    assert np.nanmean(np.abs(res.bin_mean_theta[valid] - res.bin_empirical_rate[valid])) < 0.08
    assert res.ece < 0.08


if __name__ == "__main__":
    import sys
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
