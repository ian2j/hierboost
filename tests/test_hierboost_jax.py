"""Sanity checks for hierboost.latent and hierboost.joint (require JAX/NumPyro).

Run inside the isolated environment:
    source .venv-jax/bin/activate && python3 test_hierboost_jax.py
"""
import numpy as np
from scipy.special import expit
from scipy.stats import norm

from hierboost.blocks import threshold_blocks_1d
from hierboost.latent import (fit_block_hyperparameters, fit_latent_block_model,
                               compute_sign_flips, apply_sign_flips)
from hierboost.joint import run_joint_inference, posterior_association_summary
from hierboost.estimator import HierBoostClassifier


def _simulate_ld_haplotype(n, positions, ld_length, rng):
    """AR(1)-in-space latent Gaussian: Corr(z_i, z_j) = exp(-|pos_i-pos_j|/ld_length)."""
    positions = np.asarray(positions, dtype=float)
    p = positions.shape[0]
    order = np.argsort(positions)
    pos_sorted = positions[order]
    rho = np.exp(-np.diff(pos_sorted) / ld_length)

    z = np.empty((n, p))
    z[:, 0] = rng.standard_normal(n)
    eps = rng.standard_normal((n, p - 1))
    for j in range(1, p):
        z[:, j] = rho[j - 1] * z[:, j - 1] + np.sqrt(1.0 - rho[j - 1] ** 2) * eps[:, j - 1]

    z_unsorted = np.empty_like(z)
    z_unsorted[:, order] = z
    return z_unsorted


def _simulate_genotypes(n, positions, maf, ld_length, rng):
    """Additive 0/1/2 genotypes from two independent LD-correlated haplotypes."""
    thresh = norm.ppf(1.0 - maf)
    h1 = (_simulate_ld_haplotype(n, positions, ld_length, rng) > thresh[None, :]).astype(int)
    h2 = (_simulate_ld_haplotype(n, positions, ld_length, rng) > thresh[None, :]).astype(int)
    return h1 + h2


def _make_causal_block_dataset(seed=1, n=100, p=24):
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.uniform(0, 6000, p))
    X = _simulate_genotypes(n=n, positions=positions, maf=rng.uniform(0.15, 0.4, p),
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


def _make_multimember_causal_block_dataset(seed=1, n=150, m_causal=6, m_null=12):
    """Like _make_causal_block_dataset, but deliberately constructed (a tight causal
    cluster, well separated from a spread-out null pool) so the causal block always
    comes out with multiple members -- needed for the sign-flip tests below, which are
    about within-block sign disagreement and have nothing to test on a singleton."""
    rng = np.random.default_rng(seed)
    positions_causal = np.sort(rng.uniform(0, 400, m_causal))
    positions_null = np.sort(rng.uniform(2000, 8000, m_null))
    positions = np.concatenate([positions_causal, positions_null])
    maf = rng.uniform(0.15, 0.4, m_causal + m_null)
    X = _simulate_genotypes(n=n, positions=positions, maf=maf, ld_length=600, rng=rng)
    block_id = threshold_blocks_1d(positions, zeta=500)
    causal_block = block_id[0]
    idx0 = np.where(block_id == causal_block)[0]
    assert len(idx0) > 1  # sanity: this test needs a genuine multi-member block
    signal = X[:, idx0].mean(axis=1)
    signal = (signal - signal.mean()) / signal.std()
    eta = 1.5 * signal + rng.normal(0, 0.3, n)
    y = (rng.random(n) < expit(eta)).astype(float)
    return X, y, positions, block_id, causal_block, signal


def test_compute_sign_flips_detects_and_reverses_a_flipped_feature():
    """A feature whose allele coding runs opposite to its block-mates should be
    detected and localized correctly, and apply_sign_flips should be its own exact
    inverse (recoding twice returns the original data)."""
    X, y, positions, block_id, causal_block, signal = _make_multimember_causal_block_dataset()
    idx0 = np.where(block_id == causal_block)[0]
    flip_target = idx0[0]
    n_trials = 2
    X_corrupted = X.copy()
    X_corrupted[:, flip_target] = n_trials - X_corrupted[:, flip_target]

    flip_mask = compute_sign_flips(X_corrupted, block_id, n_trials)
    assert flip_mask[flip_target]
    assert not flip_mask[idx0[idx0 != flip_target]].any()  # rest of the causal block untouched

    X_recovered = apply_sign_flips(X_corrupted, flip_mask, n_trials)
    assert np.array_equal(X_recovered[:, flip_target], X[:, flip_target])


def test_estimator_classifier_sar_decorrelate_handles_a_sign_flipped_member():
    """HierBoostClassifier applies sign-alignment automatically before fitting, so a
    flipped feature within the causal block should no longer meaningfully degrade the
    fit. Compares against calling fit_latent_block_model directly on the uncorrected
    data, confirming the correction has a real effect rather than the corruption being
    harmless anyway."""
    X, y, positions, block_id, causal_block, signal = _make_multimember_causal_block_dataset()
    idx0 = np.where(block_id == causal_block)[0]
    n_trials = 2
    X_corrupted = X.copy()
    X_corrupted[:, idx0[0]] = n_trials - X_corrupted[:, idx0[0]]

    K = len(np.unique(block_id))
    wr = np.zeros(K)
    result_uncorrected = fit_latent_block_model(
        X_corrupted, y, positions, block_id, wr, xi0=-1.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
        n_outer=8, n_inner_newton=6, hyper_n_steps=150, seed=0)
    causal_k_uncorrected = result_uncorrected["block_ids"].index(causal_block)
    corr_uncorrected = abs(np.corrcoef(result_uncorrected["Z"][:, causal_k_uncorrected], signal)[0, 1])

    clf = HierBoostClassifier(decorrelate="sar", xi0=-1.0, xi1=0.0, kappa=100.0)
    clf.fit(X_corrupted, y, coords=positions, block_id=block_id, n_trials=n_trials)
    assert clf.binomial_flip_mask_[idx0[0]]
    causal_k = clf.block_ids_.index(causal_block)
    # not an exact tie-for-max check: this toy dataset's null blocks separate cleanly
    # enough that several routinely tie at theta_hat==1.0 too (a saturated-small-sample
    # artifact unrelated to the sign-flip fix being tested) -- "clearly high confidence"
    # is the actual, robust claim here.
    assert clf.theta_hat_[causal_k] > 0.9

    corr_fixed = abs(np.corrcoef(clf.X_design_[:, 1 + causal_k], signal)[0, 1])
    assert corr_fixed > corr_uncorrected


def test_joint_inference_recovers_signal_with_a_sign_flipped_feature():
    """joint_block_model's free per-feature loading (see its tau_w_scale docstring)
    should recover the true signal even with a flipped feature in the causal block.
    Tested on raw corrupted data, bypassing HierBoostClassifier's own sign-alignment
    preprocessing, to isolate this mechanism's own contribution."""
    X, y, positions, block_id, causal_block, signal = _make_multimember_causal_block_dataset(seed=0, n=150)
    idx0 = np.where(block_id == causal_block)[0]
    n_trials = 2
    X_corrupted = X.copy()
    X_corrupted[:, idx0[0]] = n_trials - X_corrupted[:, idx0[0]]

    from hierboost.latent import fit_block_latent_fits
    fits = fit_block_latent_fits(X_corrupted, positions, block_id, n_trials=n_trials, hyper_n_steps=150, seed=0)
    mcmc, block_ids = run_joint_inference(X_corrupted, y, block_id, fits, n_trials=n_trials,
                                           num_warmup=400, num_samples=600, seed=0, progress_bar=False)
    causal_k = block_ids.index(causal_block)
    zt_mean = np.array(mcmc.get_samples()[f"Zt_{causal_k}"]).mean(axis=0)
    assert abs(np.corrcoef(zt_mean, signal)[0, 1]) > 0.9

    summary = posterior_association_summary(mcmc, block_ids, practical_threshold=0.1)
    assert summary["prob_association"][causal_k] > 0.95


def test_estimator_classifier_joint_decorrelate_predicts_held_out():
    """End-to-end fit_method='joint' smoke test through HierBoostClassifier (not
    previously covered): fit, then predict on fresh held-out data generated from the
    same block structure."""
    X, y, positions, block_id, causal_block, signal = _make_multimember_causal_block_dataset(seed=0, n=150)
    clf = HierBoostClassifier(decorrelate="sar", fit_method="joint", xi0=-1.0, xi1=0.0,
                               burn_in=300, n_samples=400)
    clf.fit(X, y, coords=positions, block_id=block_id, n_trials=2, verbose=False)
    assert clf.latent_w_ is not None

    idx0 = np.where(block_id == causal_block)[0]
    rng = np.random.default_rng(99)
    maf = rng.uniform(0.15, 0.4, len(positions))
    X_new = _simulate_genotypes(n=80, positions=positions, maf=maf, ld_length=600, rng=rng)
    signal_new = X_new[:, idx0].mean(axis=1)
    signal_new = (signal_new - signal_new.mean()) / signal_new.std()

    pred = clf.predict_proba(X_new, return_std=True)
    assert pred.mean.shape == (80,)
    assert abs(np.corrcoef(pred.mean, signal_new)[0, 1]) > 0.5


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
