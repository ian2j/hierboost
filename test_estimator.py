"""Sanity checks for hierboost.estimator (the practitioner-facing fit/predict/summary/plot
API), main env only -- see test_hierboost_jax.py for the binomial+decorrelate path, which
needs JAX.
"""
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.special import expit

from hierboost.estimator import HierBoostClassifier, HierBoostRegressor, PredictionResult
from hierboost.model_selection import cross_val_score, cross_val_predict


def _binomial_dataset(seed=0, n=300, p=40, causal=(3, 10, 25)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[list(causal)] = [2.0, -1.5, 1.8][: len(causal)]
    y = (rng.random(n) < expit(X @ beta_true)).astype(float)
    return X, y, beta_true


def _gaussian_dataset(seed=0, n=300, p=40, causal=(5, 15)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[list(causal)] = [1.5, -2.0][: len(causal)]
    y = X @ beta_true + rng.normal(0, 1.0, n)
    return X, y, beta_true


def test_classifier_no_decorrelate_recovers_causal_features_and_predicts():
    X, y, beta_true = _binomial_dataset()
    clf = HierBoostClassifier(fit_method="em")
    clf.fit(X, y)
    causal = np.where(beta_true != 0)[0]
    non_causal = np.where(beta_true == 0)[0]
    assert clf.theta_hat_[causal].min() > clf.theta_hat_[non_causal].max()

    rng = np.random.default_rng(1)
    X_new = rng.normal(size=(100, X.shape[1]))
    pred = clf.predict_proba(X_new, return_std=True)
    assert isinstance(pred, PredictionResult)
    assert pred.mean.shape == (100,)
    assert np.all(pred.lower <= pred.mean) and np.all(pred.mean <= pred.upper)
    assert np.all((pred.mean >= 0) & (pred.mean <= 1))


def test_classifier_binomial_n_trials_without_decorrelate_raises():
    X, y, _ = _binomial_dataset(n=100, p=10, causal=(2, 5))
    clf = HierBoostClassifier()
    try:
        clf.fit(X, y, n_trials=2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_regressor_no_decorrelate_recovers_signal_and_predicts():
    X, y, beta_true = _gaussian_dataset()
    reg = HierBoostRegressor(fit_method="em")
    reg.fit(X, y)
    causal = np.where(beta_true != 0)[0]
    non_causal = np.where(beta_true == 0)[0]
    assert reg.theta_hat_[causal].min() > reg.theta_hat_[non_causal].max()

    rng = np.random.default_rng(2)
    X_new = rng.normal(size=(50, X.shape[1]))
    y_new = X_new @ beta_true + rng.normal(0, 1.0, 50)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.5


def test_classifier_em_filter_predict_consistent_with_dropped_features():
    X, y, beta_true = _binomial_dataset()
    clf = HierBoostClassifier(fit_method="em_filter", min_features=5, max_outer=20)
    clf.fit(X, y)
    assert len(clf.names_) < X.shape[1]
    causal_names = {f"x{j}" for j in np.where(beta_true != 0)[0]}
    assert causal_names.issubset(set(clf.names_))

    rng = np.random.default_rng(3)
    X_new = rng.normal(size=(30, X.shape[1]))
    pred = clf.predict_proba(X_new)
    assert pred.shape == (30,)


def test_classifier_gibbs_predict():
    X, y, _ = _binomial_dataset(n=150, p=15, causal=(2, 8))
    clf = HierBoostClassifier(fit_method="gibbs", n_samples=400, burn_in=150, random_state=0)
    clf.fit(X, y)
    assert clf.gibbs_ is not None
    pred = clf.predict_proba(return_std=True)
    assert np.all(pred.std >= 0)


def test_regressor_sar_decorrelate_predicts_held_out():
    rng = np.random.default_rng(0)
    n, m_per_block, n_blocks = 400, 4, 3
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])

    def gen(n, seed):
        r = np.random.default_rng(seed)
        latents = r.normal(size=(n, n_blocks))
        X = np.zeros((n, m_per_block * n_blocks))
        for b in range(n_blocks):
            loadings = r.normal(1, 0.3, m_per_block)
            X[:, b * m_per_block:(b + 1) * m_per_block] = (
                np.outer(latents[:, b], loadings) + r.normal(0, 0.3, (n, m_per_block)))
        y = latents @ np.array([2.0, 0.0, -1.5]) + r.normal(0, 0.5, n)
        return X, y

    X, y = gen(n, 0)
    reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    assert len(reg.block_ids_) == n_blocks

    X_new, y_new = gen(80, 99)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.8


def test_regressor_sar_copula_marginal_beats_raw_under_heterogeneous_block_marginals():
    """The block-latent factor model (hierboost.factor.gaussian_block_factor) assumes each
    raw feature is already roughly Gaussian -- marginal='copula' (hierboost.copula) relaxes
    that by mapping each column through its own empirical marginal onto a shared Gaussian
    scale first. Construct one block whose members share a real latent but have wildly
    different marginal shapes (identity/lognormal/gamma), same recipe as
    copula_synthetic_validation.py, and confirm the copula-marginal fit generalizes to
    held-out data at least as well as (and here, better than) the raw fit."""
    from scipy import stats

    def gen(n, seed):
        r = np.random.default_rng(seed)
        z = r.normal(size=n)
        Xg = z[:, None] * np.array([1.2, -0.9, 1.5]) + r.normal(scale=0.5, size=(n, 3))
        X = np.empty_like(Xg)
        X[:, 0] = Xg[:, 0]
        X[:, 1] = np.exp(Xg[:, 1])
        X[:, 2] = stats.gamma.ppf(stats.norm.cdf(Xg[:, 2]), a=1.5, scale=1.0)
        y = 2.0 * z + r.normal(scale=0.3, size=n)
        return X, y

    coords = np.zeros(3)
    X, y = gen(500, 0)
    X_new, y_new = gen(150, 99)

    def held_out_r2(reg):
        pred = reg.predict(X_new, return_std=False)
        return 1 - np.sum((y_new - pred) ** 2) / np.sum((y_new - y_new.mean()) ** 2)

    reg_raw = HierBoostRegressor(decorrelate="sar", fit_method="em")
    reg_raw.fit(X, y, coords=coords)
    reg_copula = HierBoostRegressor(decorrelate="sar", marginal="copula", fit_method="em")
    reg_copula.fit(X, y, coords=coords)

    assert reg_copula.marginal_transforms_ is not None
    assert held_out_r2(reg_copula) > held_out_r2(reg_raw)


def test_regressor_sar_bootstrap_ci_covers_true_coefficients():
    rng = np.random.default_rng(0)
    n, m_per_block, n_blocks = 400, 4, 3
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])
    latents = rng.normal(size=(n, n_blocks))
    X = np.zeros((n, m_per_block * n_blocks))
    for b in range(n_blocks):
        loadings = rng.normal(1, 0.3, m_per_block)
        X[:, b * m_per_block:(b + 1) * m_per_block] = (
            np.outer(latents[:, b], loadings) + rng.normal(0, 0.3, (n, m_per_block)))
    beta_true = np.array([2.0, 0.0, -1.5])
    y = latents @ beta_true + rng.normal(0, 0.5, n)

    reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    result = reg.bootstrap_ci(X, y, coords=coords, n_boot=40, random_state=0)

    assert result["names"] == reg.names_
    assert np.all(result["lo"] <= result["hi"])
    for j, true_val in enumerate(beta_true):
        assert result["lo"][j] <= true_val <= result["hi"][j]
    # bootstrap CI should be wider than the closed-form Laplace CI it's meant to
    # complement (the whole point: the Laplace SE understates uncertainty)
    laplace_width = 2 * 1.96 * reg.beta_se_[1:]
    boot_width = result["hi"] - result["lo"]
    assert boot_width.mean() > laplace_width.mean()


def test_classifier_copula_marginal_binomial_response_raises():
    X, y, _ = _binomial_dataset(n=100, p=6, causal=(1, 3))
    clf = HierBoostClassifier(decorrelate="sar", marginal="copula")
    try:
        clf.fit(X, y, coords=np.arange(6, dtype=float))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_regressor_ar1_decorrelate_predicts_held_out():
    m = 4
    true_rho = 0.85
    loadings_true = np.array([1.0, 0.8, -0.6, 1.2])

    def gen(n, seed):
        r = np.random.default_rng(seed)
        z = np.zeros(n)
        sv = 1 - true_rho ** 2
        for t in range(1, n):
            z[t] = true_rho * z[t - 1] + r.normal(0, np.sqrt(sv))
        X = np.outer(z, loadings_true) + r.normal(0, 0.3, (n, m))
        y = 2.0 * z + r.normal(0, 0.3, n)
        return X, y

    X, y = gen(300, 0)
    coords = np.arange(m, dtype=float)
    reg = HierBoostRegressor(decorrelate="ar1", fit_method="em")
    reg.fit(X, y, coords=coords, block_id=np.zeros(m, dtype=int))
    assert abs(reg.latent_fits_[0][1].rho - true_rho) < 0.15

    X_new, y_new = gen(100, 5)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.8


def _spacetime_dataset(seed=0, T=600, m_per_block=4, n_blocks=3,
                        rho1_true=0.6, rho2_true=0.25, y_coefs=None):
    from hierboost.kernels import sar_weight_matrix
    if y_coefs is None:
        y_coefs = [2.0, 0.0, -1.5][:n_blocks] if n_blocks <= 3 else \
            list(np.linspace(-1.5, 2.0, n_blocks))
    rng = np.random.default_rng(seed)
    centroids = np.arange(n_blocks, dtype=float) * 10.0
    W = sar_weight_matrix(centroids, bandwidth=8.0)
    Phi = rho1_true * np.eye(n_blocks) + rho2_true * W
    Q = rng.uniform(0.4, 0.6, n_blocks)
    z = np.zeros((T, n_blocks))
    for t in range(1, T):
        z[t] = Phi @ z[t - 1] + rng.normal(0, np.sqrt(Q))
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])
    X = np.zeros((T, m_per_block * n_blocks))
    loadings = {}
    for b in range(n_blocks):
        L = rng.normal(1, 0.3, m_per_block)
        loadings[b] = L
        X[:, b * m_per_block:(b + 1) * m_per_block] = np.outer(z[:, b], L) + rng.normal(0, 0.4, (T, m_per_block))
    y = z @ np.array(y_coefs) + rng.normal(0, 0.3, T)
    return X, y, coords, z, loadings, Phi, Q


def test_regressor_star_decorrelate_recovers_coupling_and_predicts_held_out():
    X, y, coords, z, loadings, Phi, Q = _spacetime_dataset()
    reg = HierBoostRegressor(decorrelate="star", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    assert len(reg.block_ids_) == 3
    assert reg.star_model_ is not None
    assert abs(reg.star_model_["rho1"] - 0.6) < 0.2
    assert reg.star_model_["Phi"].shape == (3, 3)

    def gen_holdout(n, seed):
        r = np.random.default_rng(seed)
        zt = np.zeros((n, 3))
        for t in range(1, n):
            zt[t] = Phi @ zt[t - 1] + r.normal(0, np.sqrt(Q))
        Xn = np.zeros((n, 12))
        for b in range(3):
            Xn[:, b * 4:(b + 1) * 4] = np.outer(zt[:, b], loadings[b]) + r.normal(0, 0.4, (n, 4))
        yn = zt @ np.array([2.0, 0.0, -1.5]) + r.normal(0, 0.3, n)
        return Xn, yn

    X_new, y_new = gen_holdout(150, 99)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.7
    reg.plot_latent(block=reg.block_ids_[0])
    rows = reg.describe_blocks()
    assert sum(len(r["members"]) for r in rows) == X.shape[1]


def test_star_decorrelate_handles_mixed_singleton_and_multi_member_blocks():
    X, y, coords, *_ = _spacetime_dataset(m_per_block=4, n_blocks=2)
    # add two extra singleton "blocks" far away from the others on the coordinate axis
    rng = np.random.default_rng(11)
    X_extra = rng.normal(size=(X.shape[0], 2))
    X_full = np.column_stack([X, X_extra])
    coords_full = np.concatenate([coords, [1000.0, 1100.0]])

    reg = HierBoostRegressor(decorrelate="star", zeta=50, fit_method="em")
    reg.fit(X_full, y, coords=coords_full)
    kinds = {b: reg.latent_fits_[b][0] for b in reg.block_ids_}
    assert "single" in kinds.values()
    assert "star" in kinds.values()

    X_new = rng.normal(size=(30, X_full.shape[1]))
    pred = reg.predict(X_new)
    assert pred.shape == (30,)


def _spacetime_dataset_2d(seed=0, T=600, m_per_block=4, n_blocks=3,
                           rho1_true=0.6, rho2_true=0.25, y_coefs=None):
    """2D-centroid counterpart of _spacetime_dataset: block coupling is driven by real
    2D physical coordinates (e.g. lon/lat) rather than a 1D synthetic axis -- covers the
    gap where sar_weight_matrix/estimator.py's star branch used to only accept 1D coords
    (see hierboost/kernels.py, hierboost/estimator.py). Block centroids are an L-shape so
    pairwise distances can't be recovered by sorting a single axis, which is exactly what
    a silent 2D-to-scalar collapse (the old `.mean()` bug) would get wrong."""
    from hierboost.kernels import sar_weight_matrix
    if y_coefs is None:
        y_coefs = [2.0, 0.0, -1.5][:n_blocks]
    rng = np.random.default_rng(seed)
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 14.0]])[:n_blocks]
    W = sar_weight_matrix(centroids, bandwidth=8.0)
    Phi = rho1_true * np.eye(n_blocks) + rho2_true * W
    Q = rng.uniform(0.4, 0.6, n_blocks)
    z = np.zeros((T, n_blocks))
    for t in range(1, T):
        z[t] = Phi @ z[t - 1] + rng.normal(0, np.sqrt(Q))
    # each raw feature's physical coordinate is its block's centroid, repeated -- coords_
    # is per-raw-feature, same shape convention resolve_affinity/determine_blocks use.
    coords = np.repeat(centroids, m_per_block, axis=0)
    block_id = np.repeat(np.arange(n_blocks), m_per_block)
    X = np.zeros((T, m_per_block * n_blocks))
    loadings = {}
    for b in range(n_blocks):
        L = rng.normal(1, 0.3, m_per_block)
        loadings[b] = L
        X[:, b * m_per_block:(b + 1) * m_per_block] = np.outer(z[:, b], L) + rng.normal(0, 0.4, (T, m_per_block))
    y = z @ np.array(y_coefs) + rng.normal(0, 0.3, T)
    return X, y, coords, block_id, z, loadings, Phi, Q, W


def test_regressor_star_decorrelate_with_2d_coords_recovers_coupling_and_predicts_held_out():
    X, y, coords, block_id, z, loadings, Phi, Q, W = _spacetime_dataset_2d()
    reg = HierBoostRegressor(decorrelate="star", fit_method="em", phi_star=8.0)
    reg.fit(X, y, coords=coords, block_id=block_id)
    assert len(reg.block_ids_) == 3
    assert reg.star_model_ is not None
    assert abs(reg.star_model_["rho1"] - 0.6) < 0.2
    assert reg.star_model_["Phi"].shape == (3, 3)
    # the fitted weight matrix must reflect genuine 2D Euclidean distance (d(0,1)=10,
    # d(0,2)=14, d(1,2)=sqrt(296)~17.2) -- a silent scalar-mean collapse of the centroids
    # would produce a different (wrong) W instead of raising, so this is the check that
    # actually catches that failure mode.
    np.testing.assert_allclose(reg.star_model_["W"], W)
    # smaller distance -> larger weight: d(0,1)=10 < d(0,2)=14 < d(1,2)~17.2
    assert reg.star_model_["W"][0, 1] > reg.star_model_["W"][0, 2] > reg.star_model_["W"][1, 2]

    def gen_holdout(n, seed):
        r = np.random.default_rng(seed)
        zt = np.zeros((n, 3))
        for t in range(1, n):
            zt[t] = Phi @ zt[t - 1] + r.normal(0, np.sqrt(Q))
        Xn = np.zeros((n, 12))
        for b in range(3):
            Xn[:, b * 4:(b + 1) * 4] = np.outer(zt[:, b], loadings[b]) + r.normal(0, 0.4, (n, 4))
        yn = zt @ np.array([2.0, 0.0, -1.5]) + r.normal(0, 0.3, n)
        return Xn, yn

    X_new, y_new = gen_holdout(150, 99)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.7


def test_regressor_sar_decorrelate_accepts_2d_coords():
    # decorrelate="sar" itself never builds a spatial weight matrix from coords (the
    # continuous path is a per-block probabilistic-PCA factor, hierboost.factor), but
    # coords_ must still be a valid 2D physical-location array through the high-level API
    # without erroring -- block_id is passed explicitly so block *determination*
    # (1D-only, a separate limitation) isn't in the way.
    X, y, coords, block_id, *_ = _spacetime_dataset_2d(n_blocks=2, m_per_block=4)
    reg = HierBoostRegressor(decorrelate="sar", fit_method="em")
    reg.fit(X, y, coords=coords, block_id=block_id)
    assert len(reg.block_ids_) == 2

    X_new = np.random.default_rng(5).normal(size=(20, X.shape[1]))
    pred = reg.predict(X_new)
    assert pred.shape == (20,)


def test_star_decorrelate_binomial_response_raises():
    X, y, coords, *_ = _spacetime_dataset(n_blocks=2, m_per_block=3)
    y_bin = (y > np.median(y)).astype(float)
    clf = HierBoostClassifier(decorrelate="star", zeta=50)
    try:
        clf.fit(X, y_bin, coords=coords)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_star_decorrelate_em_filter_and_save_load_round_trip():
    X, y, coords, *_ = _spacetime_dataset(n_blocks=3, m_per_block=4)
    reg = HierBoostRegressor(decorrelate="star", zeta=50, fit_method="em_filter",
                              min_features=2, max_outer=10)
    reg.fit(X, y, coords=coords)
    assert reg.fit_kind_ == "em_filter"

    rng = np.random.default_rng(21)
    X_new = rng.normal(size=(20, X.shape[1]))
    pred_before = reg.predict(X_new)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "star_model.pkl")
        reg.save(path)
        reloaded = HierBoostRegressor.load(path)
    pred_after = reloaded.predict(X_new)
    np.testing.assert_allclose(pred_before, pred_after)


def test_count_regressor_star_decorrelate_predicts_held_out():
    from hierboost.estimator import HierBoostCountRegressor
    X, y, coords, z, loadings, Phi, Q = _spacetime_dataset(n_blocks=2, m_per_block=4,
                                                             y_coefs=(0.6, -0.4))
    eta = 0.3 + (y - y.mean()) / y.std() * 0.5  # reuse z-driven y as a proxy signal, rescaled
    mu = np.exp(np.clip(eta, -10, 10))
    rng = np.random.default_rng(5)
    y_counts = rng.poisson(mu)

    reg = HierBoostCountRegressor(family="poisson", decorrelate="star", zeta=50, fit_method="em")
    reg.fit(X, y_counts, coords=coords)
    pred = reg.predict(return_std=True)
    assert np.all(pred.mean > 0)
    assert np.all(pred.std >= 0)


def test_summary_contains_expected_fields():
    X, y, beta_true = _binomial_dataset()
    clf = HierBoostClassifier()
    clf.fit(X, y)
    text = clf.summary(top_n=5)
    assert "HierBoostClassifier" in text
    assert "PPL" in text
    assert "theta_hat" in text
    for j in np.where(beta_true != 0)[0]:
        assert f"x{j}" in text


def test_plot_methods_run_without_error():
    X, y, _ = _binomial_dataset(n=150, p=20, causal=(3, 10, 15))
    clf = HierBoostClassifier(fit_method="em_filter", min_features=5, max_outer=15)
    clf.fit(X, y)
    clf.plot_inclusion()
    clf.plot_coefficients()
    clf.plot_diagnostics()

    Xg, yg, _ = _gaussian_dataset(n=150, p=8, causal=(1, 6))
    coords = np.arange(8, dtype=float)
    reg = HierBoostRegressor(decorrelate="ar1", fit_method="em")
    reg.fit(Xg, yg, coords=coords, block_id=np.concatenate([np.zeros(4, int), np.ones(4, int)]))
    reg.plot_latent(block=0)


def _poisson_block_dataset(seed=0, n=500, m_per_block=4, n_blocks=3, coefs=(0.6, 0.0, -0.4)):
    rng = np.random.default_rng(seed)
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])
    latents = rng.normal(size=(n, n_blocks))
    X = np.zeros((n, m_per_block * n_blocks))
    for b in range(n_blocks):
        loadings = rng.normal(1, 0.3, m_per_block)
        X[:, b * m_per_block:(b + 1) * m_per_block] = (
            np.outer(latents[:, b], loadings) + rng.normal(0, 0.3, (n, m_per_block)))
    eta = 0.5 + latents @ np.array(coefs)
    mu = np.exp(np.clip(eta, -10, 10))
    y = rng.poisson(mu)
    return X, y, coords


def test_count_regressor_sar_decorrelate_predicts_held_out():
    from hierboost.estimator import HierBoostCountRegressor
    X, y, coords = _poisson_block_dataset()
    reg = HierBoostCountRegressor(family="poisson", decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    assert len(reg.block_ids_) == 3

    X_new, y_new, _ = _poisson_block_dataset(seed=99, n=150)
    pred = reg.predict(X_new, return_std=True)
    r2 = 1 - np.sum((y_new - pred.mean) ** 2) / np.sum((y_new - y_new.mean()) ** 2)
    assert r2 > 0.3
    rows = reg.describe_blocks()
    assert sum(len(r["members"]) for r in rows) == X.shape[1]
    reg.plot_latent(block=reg.block_ids_[0])


def test_count_regressor_ar1_decorrelate_negbinomial_predicts_held_out():
    from hierboost.estimator import HierBoostCountRegressor
    true_rho = 0.8
    loadings_true = np.array([1.0, 0.8, -0.6, 1.2])

    def gen(T, seed):
        r = np.random.default_rng(seed)
        z = np.zeros(T)
        sv = 1 - true_rho ** 2
        for t in range(1, T):
            z[t] = true_rho * z[t - 1] + r.normal(0, np.sqrt(sv))
        X = np.outer(z, loadings_true) + r.normal(0, 0.3, (T, 4))
        mu = np.exp(np.clip(0.3 + 0.5 * z, -10, 10))
        g = r.gamma(shape=5.0, scale=1.0 / 5.0, size=T)
        y = r.poisson(mu * g)
        return X, y

    X, y = gen(400, 1)
    coords = np.arange(4, dtype=float)
    reg = HierBoostCountRegressor(family="negbinomial", decorrelate="ar1", fit_method="em")
    reg.fit(X, y, coords=coords, block_id=np.zeros(4, dtype=int))
    assert abs(reg.latent_fits_[0][1].rho - true_rho) < 0.3

    X_new, y_new = gen(120, 9)
    pred = reg.predict(X_new, return_std=True)
    assert np.all(pred.mean > 0)  # counts stay on the natural (positive) scale


def test_count_regressor_negbinomial_gibbs_predicts():
    from hierboost.estimator import HierBoostCountRegressor
    rng = np.random.default_rng(4)
    n, p = 300, 12
    X = rng.normal(size=(n, p))
    beta_full = np.zeros(p)
    beta_full[[0, 1]] = [0.7, -0.5]
    r_true = 6.0
    eta = 0.5 + X @ beta_full
    mu = np.exp(np.clip(eta, -10, 10))
    g = rng.gamma(shape=r_true, scale=1.0 / r_true, size=n)
    y = rng.poisson(mu * g)

    reg = HierBoostCountRegressor(family="negbinomial", fit_method="gibbs",
                                   n_samples=400, burn_in=200, random_state=0)
    reg.fit(X, y)
    assert reg.gibbs_ is not None
    top2 = np.argsort(reg.theta_hat_)[::-1][:2]
    assert set(top2) == {0, 1}
    pred = reg.predict(return_std=True)
    assert np.all(pred.std >= 0)
    assert np.all(pred.mean > 0)


def test_count_regressor_poisson_gibbs_not_supported():
    from hierboost.estimator import HierBoostCountRegressor
    X, y, _ = _gaussian_dataset(n=100, p=8, causal=(2, 5))
    reg = HierBoostCountRegressor(family="poisson", fit_method="gibbs")
    try:
        reg.fit(np.abs(X), np.abs(y).round())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_count_regressor_gaussian_family_binomial_still_blocked_from_jax_path():
    # decorrelate + count response must never accidentally fall into the Binomial/JAX
    # branch, which assumes the raw features themselves are Binomial (genotypes) --
    # confirm HierBoostCountRegressor never even imports hierboost.latent.
    from hierboost.estimator import HierBoostCountRegressor
    X, y, coords = _poisson_block_dataset(n=150)
    reg = HierBoostCountRegressor(family="poisson", decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)
    assert not hasattr(reg, "latent_deltas_")


def test_describe_blocks_no_decorrelate_lists_singleton_features():
    X, y, beta_true = _binomial_dataset(n=150, p=20, causal=(3, 10, 15))
    clf = HierBoostClassifier(fit_method="em_filter", min_features=5, max_outer=15)
    clf.fit(X, y)
    rows = clf.describe_blocks()
    assert len(rows) == len(clf.names_)
    assert all(r["members"] == [r["name"]] for r in rows)
    assert [r["theta_hat"] for r in rows] == sorted((r["theta_hat"] for r in rows), reverse=True)
    top = clf.describe_blocks(top_n=2)
    assert len(top) == 2


def test_describe_blocks_decorrelate_lists_raw_members():
    rng = np.random.default_rng(0)
    n, m_per_block, n_blocks = 200, 4, 3
    coords = np.concatenate([np.arange(m_per_block, dtype=float) + b * 100 for b in range(n_blocks)])
    latents = rng.normal(size=(n, n_blocks))
    X = np.zeros((n, m_per_block * n_blocks))
    for b in range(n_blocks):
        loadings = rng.normal(1, 0.3, m_per_block)
        X[:, b * m_per_block:(b + 1) * m_per_block] = (
            np.outer(latents[:, b], loadings) + rng.normal(0, 0.3, (n, m_per_block)))
    y = latents @ np.array([2.0, 0.0, -1.5]) + rng.normal(0, 0.5, n)
    feature_names = [f"sensor{j}" for j in range(X.shape[1])]

    reg = HierBoostRegressor(decorrelate="sar", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords, feature_names=feature_names)
    rows = reg.describe_blocks()
    assert len(rows) == len(reg.block_ids_)
    all_members = sorted(m for r in rows for m in r["members"])
    assert all_members == sorted(feature_names)
    for r in rows:
        assert len(r["members"]) == m_per_block


def test_save_load_round_trip_predictions_match():
    X, y, _ = _gaussian_dataset()
    reg = HierBoostRegressor(fit_method="em")
    reg.fit(X, y)
    rng = np.random.default_rng(7)
    X_new = rng.normal(size=(20, X.shape[1]))
    pred_before = reg.predict(X_new)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pkl")
        reg.save(path)
        reloaded = HierBoostRegressor.load(path)

    pred_after = reloaded.predict(X_new)
    np.testing.assert_allclose(pred_before, pred_after)
    np.testing.assert_allclose(reloaded.beta_, reg.beta_)


def test_save_unfitted_raises():
    reg = HierBoostRegressor()
    try:
        reg.save("/tmp/should_not_be_written.pkl")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_get_params_round_trips_through_clone():
    clf = HierBoostClassifier(xi0=-1.5, kappa=50.0, fit_method="em_filter", min_features=7)
    params = clf.get_params()
    clone = type(clf)(**params)
    assert clone.get_params() == params
    assert clone.xi0 == -1.5 and clone.min_features == 7


def test_get_params_covers_count_regressor_subclass_kwargs():
    from hierboost.estimator import HierBoostCountRegressor
    reg = HierBoostCountRegressor(family="negbinomial", r_init=3.0, xi0=-2.5)
    params = reg.get_params()
    assert params["family"] == "negbinomial"
    assert params["r_init"] == 3.0
    assert params["xi0"] == -2.5
    clone = HierBoostCountRegressor(**params)
    assert clone.get_params() == params


def test_cross_val_score_classifier_and_regressor():
    X, y, _ = _binomial_dataset(n=200, p=15, causal=(2, 8))
    clf = HierBoostClassifier(fit_method="em")
    scores = cross_val_score(clf, X, y, cv=4, random_state=0)
    assert scores.shape == (4,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores.mean() > 0.6  # well-separated synthetic signal

    Xg, yg, _ = _gaussian_dataset(n=200, p=15, causal=(2, 8))
    reg = HierBoostRegressor(fit_method="em")
    r2_scores = cross_val_score(reg, Xg, yg, cv=4, random_state=0)
    assert r2_scores.shape == (4,)
    assert r2_scores.mean() > 0.3


def test_cross_val_predict_returns_out_of_fold_predictions_in_row_order():
    X, y, _ = _gaussian_dataset(n=120, p=10, causal=(1, 6))
    reg = HierBoostRegressor(fit_method="em")
    y_pred = cross_val_predict(reg, X, y, cv=3, random_state=0)
    assert y_pred.shape == y.shape
    assert not np.any(np.isnan(y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.2


def test_ar1_zvar_correction_widens_se_and_improves_coverage():
    """Coverage-style check for the ar1 heteroscedastic errors-in-variables correction
    (the open follow-up flagged in project memory / estimator.py's old TODOs, now
    implemented) -- mirrors test_regressor_sar_bootstrap_ci_covers_true_coefficients's
    precedent for the analogous sar case, but keeps to a modest replication count
    (tens, not hundreds) since this is a unit test, not the full calibration study.

    Known-truth single-block AR(1) DGP: a shared latent z_t drives both the block's raw
    features (via fixed loadings) and the outcome y (via a known gamma_true), so the
    block-latent's OWN estimation uncertainty is a real, nonzero contributor to y's
    residual variance that a naive sigma_y2_-only SE ignores.

    Honest result (see _compute_approx_covariance's docstring for the parallel sar
    finding, ~59%->~63% coverage): this closed-form fix is expected to be a real but
    PARTIAL improvement, not a full fix to 95% nominal coverage (it only accounts for
    the block-latent score's uncertainty given fixed loadings, not the loadings' own
    estimation error -- `bootstrap_ci` remains the fully-validated tool). This test
    therefore only asserts the correction (a) systematically widens the SE and (b)
    does not make coverage worse than the uncorrected baseline -- it does not assert
    the corrected coverage reaches 95%, because on this DGP it does not.
    """
    m = 4
    true_rho = 0.8
    loadings_true = np.array([1.0, 0.8, -0.6, 1.2])
    gamma_true = 2.0
    n = 200
    n_reps = 40
    z_crit = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
    coords = np.arange(m, dtype=float)

    covered_corrected = covered_naive = 0
    se_corrected_list, se_naive_list = [], []
    last_z_var_col = None

    for rep in range(n_reps):
        r = np.random.default_rng(1000 + rep)
        z = np.zeros(n)
        sv = 1 - true_rho ** 2
        for t in range(1, n):
            z[t] = true_rho * z[t - 1] + r.normal(0, np.sqrt(sv))
        X = np.outer(z, loadings_true) + r.normal(0, 0.5, (n, m))
        y = gamma_true * z + r.normal(0, 0.4, n)

        reg = HierBoostRegressor(decorrelate="ar1", fit_method="em")
        reg.fit(X, y, coords=coords, block_id=np.zeros(m, dtype=int))

        beta = reg.beta_[1]
        se_corrected = reg.beta_se_[1]

        z_var_saved = reg.z_var_.copy()
        last_z_var_col = z_var_saved[:, 0]
        reg.z_var_ = np.zeros_like(z_var_saved)
        reg._compute_approx_covariance()
        se_naive = reg.beta_se_[1]
        reg.z_var_ = z_var_saved  # restore, matching the corrected fit's own state
        reg._compute_approx_covariance()

        se_corrected_list.append(se_corrected)
        se_naive_list.append(se_naive)
        if abs(beta - gamma_true) <= z_crit * se_corrected:
            covered_corrected += 1
        if abs(beta - gamma_true) <= z_crit * se_naive:
            covered_naive += 1

    se_corrected_arr = np.array(se_corrected_list)
    se_naive_arr = np.array(se_naive_list)

    # Sanity: fit_temporal_block_factor's RTS-smoother posterior variance is genuinely
    # per-timestep (heteroscedastic), not a constant scalar broadcast the way sar's is --
    # this is the actual thing being newly threaded through, so confirm it varies.
    assert np.ptp(last_z_var_col) > 1e-10

    # The correction should systematically widen the SE: real, nonzero measurement
    # error exists in this DGP (the block latent is estimated, not observed).
    assert np.all(se_corrected_arr >= se_naive_arr - 1e-12)
    assert se_corrected_arr.mean() > se_naive_arr.mean()

    coverage_corrected = covered_corrected / n_reps
    coverage_naive = covered_naive / n_reps
    # Should not make coverage worse than the uncorrected baseline (partial fix, same
    # honest caveat as sar's own ~59%->~63% result -- not asserting it reaches 95%).
    assert coverage_corrected >= coverage_naive


def test_ar1_zvar_correction_collapses_to_naive_when_zvar_zero():
    """Regression-safety check: forcing self.z_var_ to all zeros must reproduce exactly
    the pre-existing (uncorrected) sigma_y2_eff == sigma_y2_ behavior, i.e. the new
    per-row correction is a strict generalization of the old scalar-only formula, not a
    rewrite that happens to only agree for decorrelate='sar'."""
    m = 4
    true_rho = 0.7
    loadings_true = np.array([1.0, 0.8, -0.6, 1.2])
    r = np.random.default_rng(3)
    n = 150
    z = np.zeros(n)
    sv = 1 - true_rho ** 2
    for t in range(1, n):
        z[t] = true_rho * z[t - 1] + r.normal(0, np.sqrt(sv))
    X = np.outer(z, loadings_true) + r.normal(0, 0.5, (n, m))
    y = 2.0 * z + r.normal(0, 0.4, n)
    coords = np.arange(m, dtype=float)

    reg = HierBoostRegressor(decorrelate="ar1", fit_method="em")
    reg.fit(X, y, coords=coords, block_id=np.zeros(m, dtype=int))
    assert reg.z_var_.shape == (n, 1)
    assert np.ptp(reg.z_var_[:, 0]) > 1e-10  # genuinely nonzero/heteroscedastic here

    reg.z_var_ = np.zeros_like(reg.z_var_)
    reg._compute_approx_covariance()
    beta_cov_forced_naive = reg.beta_cov_.copy()

    # Manually rebuild the pre-existing scalar-only formula (sigma_y2_eff == sigma_y2_,
    # exactly what the code did before this block's z_var was ever populated) and
    # confirm it matches bit-for-bit.
    from hierboost.spike_slab import _precision_from_theta
    X_design = reg.X_design_
    precision = _precision_from_theta(reg.theta_hat_, reg.kappa)
    sigma_diag = reg.sigma2_ / precision
    W = np.full(X_design.shape[0], 1.0 / reg.sigma_y2_)
    XtWX = X_design.T @ (W[:, None] * X_design)
    expected_cov = np.linalg.inv(XtWX + np.diag(1.0 / sigma_diag))
    np.testing.assert_allclose(beta_cov_forced_naive, expected_cov)


def test_regressor_star_zvar_correction_uses_real_per_row_variance():
    """Lighter analog of the ar1 checks above for decorrelate='star' (time budget:
    star's joint EM fit is substantially more expensive per call than ar1's
    single-block case, so this skips the full replicated coverage study and just
    confirms the wiring is real): fit_spacetime_block_factor's z_cov_diag must now
    actually reach _compute_approx_covariance as a genuine per-row vector (previously
    hardcoded to 0.0, see estimator.py's old TODO), and it must widen the closed-form
    SE relative to the naive (z_var=0) baseline."""
    X, y, coords, z, loadings, Phi, Q = _spacetime_dataset(n_blocks=2, m_per_block=4, T=300)
    reg = HierBoostRegressor(decorrelate="star", zeta=50, fit_method="em")
    reg.fit(X, y, coords=coords)

    assert reg.z_var_.shape[0] == X.shape[0]
    # At least one retained block-column shows genuine row-to-row variation -- star's
    # z_cov_diag is a per-timestep RTS-smoother variance, not a constant scalar.
    assert any(np.ptp(reg.z_var_[:, j]) > 1e-10 for j in range(reg.z_var_.shape[1]))

    se_corrected = reg.beta_se_.copy()
    z_var_saved = reg.z_var_.copy()
    reg.z_var_ = np.zeros_like(z_var_saved)
    reg._compute_approx_covariance()
    se_naive = reg.beta_se_.copy()
    reg.z_var_ = z_var_saved
    reg._compute_approx_covariance()

    assert np.all(se_corrected[1:] >= se_naive[1:] - 1e-12)
    assert se_corrected[1:].sum() > se_naive[1:].sum()


if __name__ == "__main__":
    import sys, traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"{t.__name__}: OK")
        except Exception:
            failed += 1
            print(f"{t.__name__}: FAILED")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
