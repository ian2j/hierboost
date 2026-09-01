"""Sanity checks for hierboost.copula (the Gaussian-copula marginal transform), main env
only -- pure numpy/scipy, no JAX.
"""
import numpy as np
from scipy import stats

from hierboost.copula import (fit_empirical_marginal, fit_parametric_marginal, to_gaussian,
                               from_gaussian, fit_block_transforms, apply_block_transforms,
                               to_gaussian_scale)


def test_empirical_transform_is_roughly_standard_normal_in_sample():
    rng = np.random.default_rng(0)
    x = rng.lognormal(mean=1.0, sigma=0.8, size=5000)  # deliberately skewed
    z, _ = to_gaussian_scale(x.reshape(-1, 1))
    z = z[:, 0]
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    # a skewed input should look Gaussian after the transform, not before
    assert abs(stats.skew(z)) < 0.15
    assert abs(stats.skew(x)) > 1.0


def test_empirical_transform_out_of_sample_consistency():
    rng = np.random.default_rng(1)
    x_train = rng.gamma(shape=2.0, scale=1.5, size=2000)
    tr = fit_empirical_marginal(x_train)
    x_test = rng.gamma(shape=2.0, scale=1.5, size=500)
    z_test = to_gaussian(x_test, tr)
    # same generating distribution -> out-of-sample z should also be ~standard normal,
    # not just the training data (confirms the fitted reference generalizes, isn't
    # overfit to the exact training sample)
    assert abs(z_test.mean()) < 0.15
    assert abs(z_test.std() - 1.0) < 0.15


def test_empirical_transform_out_of_range_new_point_is_clamped_not_infinite():
    x_train = np.arange(1.0, 101.0)
    tr = fit_empirical_marginal(x_train)
    z = to_gaussian(np.array([-1000.0, 1000.0]), tr)
    assert np.all(np.isfinite(z))
    assert z[0] < 0 and z[1] > 0


def test_empirical_round_trip_recovers_original_scale_for_distinct_values():
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 2.0, size=1000)
    tr = fit_empirical_marginal(x)
    z = to_gaussian(x, tr)
    x_back = from_gaussian(z, tr)
    assert np.allclose(x_back, x, atol=1e-6)


def test_parametric_poisson_jitter_is_uniform_before_ppf():
    rng = np.random.default_rng(3)
    mu = 8.0
    x = rng.poisson(mu, size=20000)
    tr = fit_parametric_marginal(x, "poisson")
    z = to_gaussian(x, tr, rng=rng)
    # jittered-CDF construction should make z approximately standard normal even though
    # x itself is a small-integer count with heavy ties
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05


def test_parametric_poisson_ties_get_different_z_via_jitter():
    rng = np.random.default_rng(4)
    tr = fit_parametric_marginal(rng.poisson(8.0, size=5000), "poisson")
    x_ties = np.full(200, 8.0)
    z = to_gaussian(x_ties, tr, rng=rng)
    # same raw count, jittered CDF should not collapse every tie to one identical z
    assert len(np.unique(np.round(z, 6))) > 50


def test_parametric_binomial_jitter_spreads_genotype_dosage_levels():
    # 0/1/2 genotype dosage -- the canonical Binomial(n=2, p) real-data use case
    rng = np.random.default_rng(6)
    x = rng.binomial(2, 0.15, size=20000).astype(float)
    tr = fit_parametric_marginal(x, "binom")
    z = to_gaussian(x, tr, rng=rng)
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    # each discrete level should map to a contiguous, non-overlapping z range, ordered
    # the same way as x (monotone), with real within-level spread from the jitter
    for level in (0.0, 1.0, 2.0):
        zl = z[x == level]
        assert zl.max() - zl.min() > 0.1
    assert z[x == 0].max() < z[x == 1].min()
    assert z[x == 1].max() < z[x == 2].min()


def test_fit_block_transforms_parametric_requires_matching_families_length():
    X = np.zeros((10, 3))
    try:
        fit_block_transforms(X, kind="parametric", families=["poisson", "poisson"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_block_transforms_matches_manual_columnwise_application():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.lognormal(size=300), rng.gamma(2.0, size=300)])
    transforms = fit_block_transforms(X, kind="empirical")
    Z = apply_block_transforms(X, transforms)
    Z_manual = np.column_stack([to_gaussian(X[:, 0], transforms[0]),
                                 to_gaussian(X[:, 1], transforms[1])])
    assert np.allclose(Z, Z_manual)


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
