"""Sanity checks for hierboost.spike_slab_glm (Poisson / Negative-Binomial spike-and-slab).
No JAX needed -- these families reuse the same main-env IRLS machinery as spike_slab.py
and spike_slab_gaussian.py, just with a different variance/link pair.
"""
import numpy as np
from scipy.special import expit

from hierboost.spike_slab_glm import (fit_em_poisson, em_filter_poisson, ppl_poisson,
                                       fit_em_nb, em_filter_nb, ppl_nb, _update_r,
                                       gibbs_sampler_nb)


def _make_poisson_data(n=2000, p=20, true_idx=(0, 1, 2), true_beta=(0.8, -0.6, 0.5),
                        intercept=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_full = np.zeros(p)
    for j, b in zip(true_idx, true_beta):
        beta_full[j] = b
    eta = intercept + X @ beta_full
    mu = np.exp(np.clip(eta, -10, 10))
    y = rng.poisson(mu)
    X_design = np.column_stack([np.ones(n), X])
    return X_design, y, beta_full, intercept


def test_fit_em_poisson_recovers_known_coefficients():
    X_design, y, beta_true, intercept = _make_poisson_data()
    p = X_design.shape[1] - 1
    wr = np.zeros(p)
    res = fit_em_poisson(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    assert res.beta.shape == (p + 1,)
    # the three true signal coordinates should end up with the highest inclusion probs
    top3 = np.argsort(res.theta_hat)[::-1][:3]
    assert set(top3) == {0, 1, 2}
    # recovered coefficient signs/magnitudes should be in the right ballpark
    np.testing.assert_allclose(res.beta[0], intercept, atol=0.3)
    np.testing.assert_allclose(res.beta[1:4], beta_true[:3], atol=0.3)


def test_em_filter_poisson_drops_noise_features():
    X_design, y, beta_true, _ = _make_poisson_data(p=40)
    p = X_design.shape[1] - 1
    wr = np.zeros(p)
    filt = em_filter_poisson(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                              filter_frac=0.3, min_features=5)
    best = filt.best
    assert best.n_features < p  # filtering actually removed something
    kept_original_idx = best.retained_idx
    assert 0 in kept_original_idx and 1 in kept_original_idx and 2 in kept_original_idx


def test_poisson_offset_shifts_only_intercept():
    n, p = 1500, 5
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, p))
    beta_full = np.array([0.6, -0.4, 0.0, 0.0, 0.0])
    log_pop = rng.uniform(2.0, 5.0, size=n)  # e.g. log(population)
    eta = 0.2 + X @ beta_full + log_pop
    mu = np.exp(np.clip(eta, -10, 10))
    y = rng.poisson(mu)
    X_design = np.column_stack([np.ones(n), X])
    wr = np.zeros(p)

    res_with_offset = fit_em_poisson(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0,
                                      nu=1.0, lam=1.0, offset=log_pop)
    # slope on the two real signal features should be recovered close to truth
    np.testing.assert_allclose(res_with_offset.beta[1:3], beta_full[:2], atol=0.25)
    np.testing.assert_allclose(res_with_offset.beta[0], 0.2, atol=0.3)


def test_ppl_poisson_improves_with_better_mean():
    y = np.array([1.0, 5.0, 0.0, 3.0, 8.0])
    good = ppl_poisson(y, y.astype(float))
    bad = ppl_poisson(y, np.full_like(y, y.mean()))
    assert good < bad


def _make_nb_data(n=2500, p=15, true_idx=(0, 1), true_beta=(0.7, -0.5),
                   intercept=0.8, r_true=5.0, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_full = np.zeros(p)
    for j, b in zip(true_idx, true_beta):
        beta_full[j] = b
    eta = intercept + X @ beta_full
    mu = np.exp(np.clip(eta, -10, 10))
    # NB via Gamma-Poisson mixture: y ~ Poisson(mu * g), g ~ Gamma(r, 1/r) (mean 1)
    g = rng.gamma(shape=r_true, scale=1.0 / r_true, size=n)
    y = rng.poisson(mu * g)
    X_design = np.column_stack([np.ones(n), X])
    return X_design, y, beta_full, intercept, r_true


def test_fit_em_nb_recovers_known_coefficients_and_dispersion():
    X_design, y, beta_true, intercept, r_true = _make_nb_data()
    p = X_design.shape[1] - 1
    wr = np.zeros(p)
    res = fit_em_nb(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    top2 = np.argsort(res.theta_hat)[::-1][:2]
    assert set(top2) == {0, 1}
    np.testing.assert_allclose(res.beta[1:3], beta_true[:2], atol=0.3)
    # dispersion r is much harder to pin down exactly than beta; just check it's in a
    # sane ballpark (same order of magnitude as the true value, not e.g. 1e-3 or 1e6)
    assert 1.0 < res.r < 50.0


def test_update_r_recovers_dispersion_given_true_mu():
    rng = np.random.default_rng(3)
    n = 5000
    mu = np.full(n, 4.0)
    r_true = 3.0
    g = rng.gamma(shape=r_true, scale=1.0 / r_true, size=n)
    y = rng.poisson(mu * g)
    r_hat = _update_r(y, mu)
    assert 1.5 < r_hat < 6.0


def test_nb_collapses_toward_poisson_as_r_grows():
    # with a huge true dispersion, NB variance mu + mu^2/r -> mu (Poisson); fitting
    # should recover a large r and near-identical beta to the Poisson fit on the
    # same (near-Poisson-variance) data.
    n, p = 2000, 6
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, p))
    beta_full = np.array([0.5, -0.3, 0.0, 0.0, 0.0, 0.0])
    eta = 0.3 + X @ beta_full
    mu = np.exp(np.clip(eta, -10, 10))
    y = rng.poisson(mu)  # exactly Poisson-distributed data
    X_design = np.column_stack([np.ones(n), X])
    wr = np.zeros(p)

    res_pois = fit_em_poisson(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    res_nb = fit_em_nb(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    np.testing.assert_allclose(res_nb.beta, res_pois.beta, atol=0.15)
    assert res_nb.r > 20.0  # should infer near-Poisson (large-r) dispersion


def test_gibbs_sampler_nb_recovers_known_coefficients_and_dispersion():
    X_design, y, beta_true, intercept, r_true = _make_nb_data(n=2000, p=15, seed=2)
    p = X_design.shape[1] - 1
    wr = np.zeros(p)
    gr = gibbs_sampler_nb(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                           n_samples=600, burn_in=300, seed=0)
    beta_mean = gr.beta.mean(axis=0)
    top2 = np.argsort(gr.pi_hat)[::-1][:2]
    assert set(top2) == {0, 1}
    np.testing.assert_allclose(beta_mean[0], intercept, atol=0.3)
    np.testing.assert_allclose(beta_mean[1:3], beta_true[:2], atol=0.3)
    assert 1.0 < gr.r.mean() < 50.0
    assert gr.beta.shape == (600, p + 1)
    assert gr.theta.shape == (600, p)


def test_gibbs_sampler_nb_offset_recovered():
    n, p = 1500, 5
    rng = np.random.default_rng(6)
    X = rng.normal(size=(n, p))
    beta_full = np.array([0.6, -0.4, 0.0, 0.0, 0.0])
    log_pop = rng.uniform(2.0, 4.0, size=n)
    eta = 0.2 + X @ beta_full + log_pop
    mu = np.exp(np.clip(eta, -10, 10))
    r_true = 8.0
    g = rng.gamma(shape=r_true, scale=1.0 / r_true, size=n)
    y = rng.poisson(mu * g)
    X_design = np.column_stack([np.ones(n), X])
    wr = np.zeros(p)

    gr = gibbs_sampler_nb(X_design, y, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                           offset=log_pop, n_samples=500, burn_in=250, seed=1)
    beta_mean = gr.beta.mean(axis=0)
    np.testing.assert_allclose(beta_mean[1:3], beta_full[:2], atol=0.3)
    np.testing.assert_allclose(beta_mean[0], 0.2, atol=0.35)


def test_ppl_nb_improves_with_better_mean():
    y = np.array([2.0, 6.0, 1.0, 4.0, 9.0])
    good = ppl_nb(y, y.astype(float), r=10.0)
    bad = ppl_nb(y, np.full_like(y, y.mean()), r=10.0)
    assert good < bad
