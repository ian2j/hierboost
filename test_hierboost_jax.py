"""Sanity checks for hierboost.latent and hierboost.joint (require JAX/NumPyro).

Run inside the isolated environment:
    source .venv-jax/bin/activate && python3 test_hierboost_jax.py
"""
import numpy as np
from scipy.special import expit

from spatial_boost.simulate import simulate_genotypes
from hierboost.blocks import threshold_blocks_1d
from hierboost.latent import fit_block_hyperparameters, fit_latent_block_model
from hierboost.joint import run_joint_inference, posterior_association_summary
from hierboost.estimator import HierBoostClassifier


def _make_causal_block_dataset(seed=1, n=100, p=24):
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.uniform(0, 6000, p))
    X = simulate_genotypes(n=n, positions=positions, maf=rng.uniform(0.15, 0.4, p),
                            ld_length=600, rng=rng)
    block_id = threshold_blocks_1d(positions, zeta=500)
    causal_block = 0
    idx0 = np.where(block_id == causal_block)[0]
    signal = X[:, idx0].mean(axis=1)
    signal = (signal - signal.mean()) / signal.std()
    eta = 1.5 * signal + rng.normal(0, 0.3, n)
    y = (rng.random(n) < expit(eta)).astype(float)
    return X, y, positions, block_id, causal_block, signal


def _make_causal_ar1_block_dataset(seed=1, n=150, m=6, n_trials=2, rho_true=0.7):
    """Temporal counterpart of _make_causal_block_dataset: no spatial simulator needed --
    each block member is a lag of the same underlying signal, decayed by rho_true**lag
    (further lags carry an attenuated echo of the true current state), exactly the
    structure structure="ar1" is meant to recover. A second, unrelated block checks the
    model still ranks the genuinely temporal block highest."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    signal = (signal - signal.mean()) / signal.std()
    y = (rng.random(n) < expit(1.5 * signal)).astype(float)

    lag = np.arange(m, dtype=float)
    decay = rho_true ** lag
    X_causal = rng.binomial(n_trials, expit(np.outer(signal, decay) * 2.0))

    z_null = rng.normal(size=n)
    X_null = rng.binomial(n_trials, expit(np.outer(z_null, decay) * 0.05))

    X = np.column_stack([X_causal, X_null])
    coords = np.concatenate([lag, lag])
    block_id = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    return X, y, coords, block_id, 0, signal


def test_fit_block_hyperparameters_ar1_reproduces_target_corr():
    rng = np.random.default_rng(0)
    m = 6
    coords = np.arange(m, dtype=float)  # lag order, not physical distance
    true_maf = rng.uniform(0.1, 0.4, m)
    d = np.abs(coords[:, None] - coords[None, :])
    target_corr = 0.7 ** d  # achievable by the directed AR(1) kernel's implied decay
    mu, phi, loss = fit_block_hyperparameters(coords, true_maf * 2, target_corr,
                                               n_steps=250, structure="ar1")
    assert mu.shape == (m,) and phi.shape == (m,)
    assert np.all(phi > 0)
    assert loss < 0.1


def test_latent_block_model_ar1_identifies_causal_temporal_block():
    X, y, coords, block_id, causal_block, signal = _make_causal_ar1_block_dataset()
    K = len(np.unique(block_id))
    wr = np.zeros(K)
    result = fit_latent_block_model(X, y, coords, block_id, wr, xi0=-1.0, xi1=0.0,
                                     kappa=100.0, nu=1.0, lam=1.0, n_outer=8,
                                     n_inner_newton=6, hyper_n_steps=150, seed=0,
                                     structure="ar1")
    theta = result["em_result"].theta_hat
    causal_k = result["block_ids"].index(causal_block)
    assert theta[causal_k] == theta.max()
    assert theta[causal_k] > 0.9
    corr = np.corrcoef(result["Z"][:, causal_k], signal)[0, 1]
    assert abs(corr) > 0.3


def test_fit_block_hyperparameters_reproduces_maf():
    from scipy.special import ndtr
    rng = np.random.default_rng(0)
    m = 8
    coords = np.sort(rng.uniform(0, 3000, m))
    true_maf = rng.uniform(0.1, 0.4, m)
    # a target correlation pattern the Gaussian-kernel family can actually represent
    # (exponential-ish decay with distance), so the fit is checked against an achievable
    # target rather than an arbitrary one the model has no way to reproduce exactly.
    d = np.abs(coords[:, None] - coords[None, :])
    target_corr = 2 * ndtr(-d / 800.0)
    mu, phi, loss = fit_block_hyperparameters(coords, true_maf * 2, target_corr, n_steps=250)
    assert mu.shape == (m,) and phi.shape == (m,)
    assert np.all(phi > 0)
    assert loss < 0.05


def test_fit_block_hyperparameters_sar_accepts_2d_coords():
    """hierboost.latent's _sar_B_jax had its own inline copy of hierboost.kernels.
    sar_weight_matrix's pre-fix formula (jnp.abs pairwise difference), so it carried the
    identical 1D-only limitation independently of kernels.py. Same fix (Euclidean
    distance for multi-D coords), same check as test_fit_block_hyperparameters_reproduces_maf
    above but with genuine 2D (e.g. lon/lat-style) coords instead of a 1D axis."""
    from scipy.special import ndtr
    rng = np.random.default_rng(0)
    m = 8
    coords = rng.uniform(0, 3000, size=(m, 2))
    true_maf = rng.uniform(0.1, 0.4, m)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    target_corr = 2 * ndtr(-d / 800.0)
    mu, phi, loss = fit_block_hyperparameters(coords, true_maf * 2, target_corr, n_steps=250)
    assert mu.shape == (m,) and phi.shape == (m,)
    assert np.all(phi > 0)
    assert loss < 0.05


def test_latent_block_model_identifies_causal_block():
    X, y, positions, block_id, causal_block, signal = _make_causal_block_dataset()
    K = len(np.unique(block_id))
    wr = np.zeros(K)
    result = fit_latent_block_model(X, y, positions, block_id, wr, xi0=-1.0, xi1=0.0,
                                     kappa=100.0, nu=1.0, lam=1.0, n_outer=8,
                                     n_inner_newton=6, hyper_n_steps=150, seed=0)
    theta = result["em_result"].theta_hat
    causal_k = result["block_ids"].index(causal_block)
    assert theta[causal_k] == theta.max()
    assert theta[causal_k] > 0.9
    # the fitted block-wise latent should track the true (unobserved) signal
    corr = np.corrcoef(result["Z"][:, causal_k], signal)[0, 1]
    assert abs(corr) > 0.3


def test_joint_inference_ranks_causal_block_highest():
    X, y, positions, block_id, causal_block, signal = _make_causal_block_dataset()
    K = len(np.unique(block_id))
    wr = np.zeros(K)
    result = fit_latent_block_model(X, y, positions, block_id, wr, xi0=-1.0, xi1=0.0,
                                     kappa=100.0, nu=1.0, lam=1.0, n_outer=3,
                                     n_inner_newton=4, hyper_n_steps=120, seed=0)
    mcmc, block_ids = run_joint_inference(X, y, block_id, result["fits"],
                                           num_warmup=300, num_samples=500, seed=0,
                                           progress_bar=False)
    summary = posterior_association_summary(mcmc, block_ids, practical_threshold=0.1)
    causal_k = block_ids.index(causal_block)
    assert summary["prob_association"][causal_k] == summary["prob_association"].max()
    assert summary["prob_association"][causal_k] > 0.95


def _make_temporal_binomial_dataset(seed=1, n=200, m=6, n_trials=2, rho_true=0.7):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    signal = (signal - signal.mean()) / signal.std()
    y = (rng.random(n) < expit(1.5 * signal)).astype(float)
    lag = np.arange(m, dtype=float)
    decay = rho_true ** lag
    X = rng.binomial(n_trials, expit(np.outer(signal, decay) * 2.0))
    return X, y, lag, signal


def test_estimator_classifier_ar1_decorrelate_predicts_held_out():
    X, y, lag, signal = _make_temporal_binomial_dataset(seed=1)
    clf = HierBoostClassifier(decorrelate="ar1", xi0=-1.0, xi1=0.0, kappa=100.0)
    clf.fit(X, y, coords=lag, block_id=np.zeros(len(lag), dtype=int), n_trials=2)
    assert clf.theta_hat_[0] > 0.5

    X_new, y_new, _, signal_new = _make_temporal_binomial_dataset(seed=2, n=80)
    pred = clf.predict_proba(X_new, return_std=True)
    assert pred.mean.shape == (80,)
    assert abs(np.corrcoef(pred.mean, signal_new)[0, 1]) > 0.4


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
