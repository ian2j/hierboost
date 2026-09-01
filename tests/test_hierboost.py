"""Sanity checks for hierboost's generic core (no JAX needed -- run in the main env).
See test_hierboost_jax.py for the latent-block and joint-inference modules, which
require the isolated .venv-jax environment.
"""
import numpy as np
from hierboost.kernels import (gaussian_affinity_1d, gaussian_affinity_points, graph_affinity,
                                combine_affinity_with_relevance, sar_weight_matrix, fit_gaussian_bandwidth,
                                causal_affinity_1d, ar1_weight_matrix)
from hierboost.blocks import (merge_overlapping_extents, threshold_blocks_1d,
                               threshold_blocks_graph, block_membership_lists)
from hierboost.spike_slab import fit_em, centroid_estimate, embfdr
from hierboost.spike_slab_gaussian import fit_em_gaussian, em_filter_gaussian, gibbs_sampler_gaussian
from hierboost.structure import blocks_from_correlation_threshold, blocks_from_graphical_lasso
from hierboost.relevance import TextEmbeddingRelevance
from hierboost.state_space import fit_temporal_block_factor
from hierboost.spacetime import fit_spacetime_block_factor, filter_spacetime_block_factor
from hierboost.factor import supervised_block_factor, apply_supervised_block_factor


def test_gaussian_affinity_1d_matches_manual_gene_weight_example():
    # Same numbers as the paper's Fig 1 worked example: SNP at 1000, genes (980,995)
    # and (1020,1030), phi=10 -> w_a=0.29, w_b=0.02.
    bl = np.array([980.0, 1020.0])
    br = np.array([995.0, 1030.0])
    affinity = gaussian_affinity_1d(np.array([1000.0]), bl, br, bandwidth=10.0)
    assert np.allclose(affinity[0], [0.29, 0.02], atol=0.01), affinity


def test_gaussian_affinity_points_2d():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 2))
    groups = np.array([[0.0, 0.0], [5.0, 5.0]])
    A = gaussian_affinity_points(features, groups, bandwidth=1.0)
    assert A.shape == (10, 2)
    assert np.all((A >= 0) & (A <= 1))
    # a feature at the origin should have near-1 affinity to group 0, near-0 to group 1
    A0 = gaussian_affinity_points(np.array([[0.0, 0.0]]), groups, bandwidth=1.0)
    assert A0[0, 0] > 0.99 and A0[0, 1] < 1e-4


def test_graph_affinity_decays_with_distance():
    D = np.array([[0, 1, 5], [1, 0, 4], [5, 4, 0]], dtype=float)
    A = graph_affinity(D, bandwidth=2.0)
    assert A[0, 1] > A[0, 2]  # closer node has higher affinity


def test_sar_weight_matrix_symmetric_zero_diag_and_block_respecting():
    coords = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    block_id = np.array([0, 0, 0, 1, 1])
    B = sar_weight_matrix(coords, bandwidth=5.0, block_id=block_id)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)
    assert B[0, 3] == 0.0 and B[0, 4] == 0.0  # cross-block entries zeroed


def test_sar_weight_matrix_2d_matches_1d_reshaped_and_uses_euclidean_distance():
    # 1D coords reshaped to (p, 1) must give byte-identical results to the plain 1D call
    # (Euclidean distance in 1D is just |a-b|) -- the multi-D generalization must not
    # perturb the pre-existing 1D behavior.
    coords_1d = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    B_1d = sar_weight_matrix(coords_1d, bandwidth=5.0)
    B_reshaped = sar_weight_matrix(coords_1d[:, None], bandwidth=5.0)
    assert np.allclose(B_1d, B_reshaped)

    # genuine 2D (lon/lat-style) coords: a 3-4-5 right triangle, so Euclidean distance is
    # unambiguous and easy to check by hand.
    coords_2d = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    B = sar_weight_matrix(coords_2d, bandwidth=2.0)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)
    # points 0 and 1 are 3 apart, points 0 and 2 are 5 apart (not 3 or 4, so this would
    # fail if the old code silently collapsed 2D coords into a bogus 1D distance) -- closer
    # pair must have strictly higher affinity than the farther pair.
    assert B[0, 1] > B[0, 2]


def test_sar_weight_matrix_rejects_higher_rank_coords():
    try:
        sar_weight_matrix(np.zeros((5, 2, 2)), bandwidth=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_causal_affinity_1d_zeros_future_and_ongoing_groups():
    # feature observed at t=100: a group that resolved at t=90 should get positive
    # affinity; a group still running (ends at 110) or fully in the future (120-130)
    # must get exactly zero -- no look-ahead leakage.
    feature_time = np.array([100.0])
    group_l = np.array([80.0, 95.0, 120.0])
    group_r = np.array([90.0, 110.0, 130.0])
    A = causal_affinity_1d(feature_time, group_l, group_r, bandwidth=10.0)
    assert A[0, 0] > 0.0
    assert A[0, 1] == 0.0
    assert A[0, 2] == 0.0


def test_causal_affinity_1d_decays_with_elapsed_time():
    feature_time = np.array([100.0])
    group_l = np.array([50.0, 50.0])
    group_r = np.array([60.0, 95.0])  # resolved long ago vs recently
    A = causal_affinity_1d(feature_time, group_l, group_r, bandwidth=10.0)
    assert A[0, 1] > A[0, 0]  # more recently resolved -> higher affinity


def test_ar1_weight_matrix_directed_decay_and_block_respecting():
    t = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    block_id = np.array([0, 0, 0, 1, 1])
    B = ar1_weight_matrix(t, correlation_length=5.0, block_id=block_id)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)  # symmetric in |t_i - t_j|, as OU correlation should be
    assert B[0, 3] == 0.0 and B[0, 4] == 0.0  # cross-block entries zeroed
    assert B[0, 1] > B[0, 2]  # closer in time -> higher weight


def test_merge_overlapping_extents_matches_manual_case():
    starts = np.array([100.0, 500.0])
    ends = np.array([200.0, 700.0])
    rel = np.array([1.0, 2.0])
    bl, br, brel = merge_overlapping_extents(starts, ends, rel)
    assert bl.shape == br.shape == brel.shape
    assert np.isclose(brel[np.argmin(np.abs(bl - 100))], 1.0)


def test_threshold_blocks_1d_and_graph_agree_on_a_chain():
    coords = np.array([0, 1, 2, 10, 11, 30])
    bid_1d = threshold_blocks_1d(coords, zeta=3)
    D = np.abs(coords[:, None] - coords[None, :]).astype(float)
    bid_graph = threshold_blocks_graph(D, zeta=3)
    # both should produce the same partition {0,1,2}, {3,4}, {5}, up to label permutation
    def canon(labels):
        _, inv = np.unique(labels, return_inverse=True)
        return tuple(inv)
    assert canon(bid_1d) == canon(bid_graph)


def test_block_membership_lists():
    bid = np.array([0, 1, 0, 2])
    m = block_membership_lists(bid)
    assert set(m.keys()) == {0, 1, 2}
    assert list(m[0]) == [0, 2]


def test_fit_gaussian_bandwidth_recovers_known_phi():
    rng = np.random.default_rng(0)
    coords = np.sort(rng.uniform(0, 10000, 60))
    true_phi = 800.0
    from scipy.special import ndtr
    iu = np.triu_indices(60, k=1)
    d = np.abs(coords[iu[0]] - coords[iu[1]])
    corr = np.zeros((60, 60))
    corr[iu] = 2 * ndtr(-d / true_phi)
    corr = corr + corr.T
    phi_hat = fit_gaussian_bandwidth(coords, np.abs(corr))
    assert abs(phi_hat - true_phi) / true_phi < 0.15


def test_combine_affinity_with_relevance_normalizes():
    affinity = np.array([[1.0, 0.0], [0.5, 0.5]])
    relevance = np.array([2.0, 4.0])
    wr = combine_affinity_with_relevance(affinity, relevance)
    assert wr.max() == 1.0


def test_fit_em_runs_and_ranks_causal_feature_higher():
    rng = np.random.default_rng(0)
    n, p = 200, 50
    X_features = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[[3, 10]] = 2.0
    X = np.column_stack([np.ones(n), X_features])
    from scipy.special import expit
    y = (rng.random(n) < expit(0.0 + X_features @ beta_true)).astype(float)
    wr = np.zeros(p)
    res = fit_em(X, y, wr, xi0=-2.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    assert res.theta_hat[3] > np.median(res.theta_hat)
    assert res.theta_hat[10] > np.median(res.theta_hat)


def test_centroid_estimate_and_embfdr_consistency():
    pi_hat = np.array([0.9, 0.8, 0.1, 0.05])
    sel = centroid_estimate(pi_hat, gamma=1.0)
    assert list(sel) == [True, True, False, False]
    bfdr = embfdr(pi_hat, gamma=1.0)
    assert 0.0 <= bfdr <= 1.0


def test_structure_learning_recovers_correlated_groups():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 2))
    X = np.column_stack([z[:, 0] + 0.05 * rng.normal(size=300),
                          z[:, 0] + 0.05 * rng.normal(size=300),
                          z[:, 1] + 0.05 * rng.normal(size=300)])
    labels = blocks_from_correlation_threshold(X, rho=0.5)
    assert labels[0] == labels[1] != labels[2]
    labels_gl = blocks_from_graphical_lasso(X, alpha=0.05)
    assert labels_gl[0] == labels_gl[1] != labels_gl[2]


def test_gaussian_spike_slab_recovers_sparse_signal():
    rng = np.random.default_rng(0)
    n, p = 300, 60
    X_feat = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    causal = [3, 10, 25]
    beta_true[causal] = [2.0, -1.5, 1.0]
    X = np.column_stack([np.ones(n), X_feat])
    y = 0.5 + X_feat @ beta_true + rng.normal(0, 1.0, n)
    wr = np.zeros(p)

    res = fit_em_gaussian(X, y, wr, xi0=-2.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0)
    assert res.theta_hat[causal].min() > 0.9
    non_causal = [j for j in range(p) if j not in causal]
    assert res.theta_hat[causal].min() > res.theta_hat[non_causal].max()
    assert abs(res.sigma_y2 - 1.0) < 0.5

    filt = em_filter_gaussian(X, y, wr, xi0=-2.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                               filter_frac=0.25, min_features=5, max_outer=20)
    assert set(causal).issubset(set(filt.best.retained_idx.tolist()))

    gr = gibbs_sampler_gaussian(X, y, wr, xi0=-2.0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                                 n_samples=1500, burn_in=500, seed=0)
    assert gr.pi_hat[causal].min() > 0.9
    assert gr.pi_hat[causal].min() > gr.pi_hat[non_causal].max()


def test_temporal_block_factor_recovers_ar1_state_and_loadings():
    rng = np.random.default_rng(0)
    T, m = 500, 4
    true_rho = 0.85
    true_loadings = np.array([1.0, 0.8, -0.6, 1.2])
    z = np.zeros(T)
    state_sd = np.sqrt(1.0 - true_rho ** 2)
    for t in range(1, T):
        z[t] = true_rho * z[t - 1] + rng.normal(0, state_sd)
    X = np.outer(z, true_loadings) + rng.normal(0, 0.3, (T, m))

    res = fit_temporal_block_factor(X)
    assert abs(res.rho - true_rho) < 0.05
    assert np.corrcoef(z, res.z)[0, 1] > 0.95
    assert np.all(np.sign(res.loadings) == np.sign(true_loadings))
    assert np.all(np.diff(res.loglik) > -1e-6)  # EM log-likelihood is monotone non-decreasing


def test_spacetime_block_factor_degenerates_to_scalar_ar1_at_k_equals_1():
    # W is 1x1 and zero-diagonal by construction, so rho2 has nothing to act on --
    # a single-block STAR fit should exactly reduce to the plain scalar AR(1) case.
    rng = np.random.default_rng(0)
    T, m = 400, 4
    true_rho = 0.85
    true_loadings = np.array([1.0, 0.8, -0.6, 1.2])
    z = np.zeros(T)
    state_sd = np.sqrt(1.0 - true_rho ** 2)
    for t in range(1, T):
        z[t] = true_rho * z[t - 1] + rng.normal(0, state_sd)
    X = np.outer(z, true_loadings) + rng.normal(0, 0.3, (T, m))

    res_ar1 = fit_temporal_block_factor(X)
    res_star = fit_spacetime_block_factor({0: X}, [0], np.zeros((1, 1)))
    assert abs(res_star.rho1 - res_ar1.rho) < 1e-6
    assert res_star.rho2 == 0.0
    assert np.corrcoef(res_ar1.z, res_star.z[:, 0])[0, 1] > 0.9999


def test_spacetime_block_factor_recovers_spatial_coupling():
    rng = np.random.default_rng(1)
    centroids = np.array([0.0, 10.0, 20.0])
    W = sar_weight_matrix(centroids, bandwidth=8.0)
    rho1_true, rho2_true = 0.6, 0.25
    Phi_true = rho1_true * np.eye(3) + rho2_true * W
    assert np.max(np.abs(np.linalg.eigvalsh(Phi_true))) < 1.0  # stable by construction

    T = 800
    Q_true = np.array([0.5, 0.4, 0.6])
    z = np.zeros((T, 3))
    for t in range(1, T):
        z[t] = Phi_true @ z[t - 1] + rng.normal(0, np.sqrt(Q_true))

    m_per_block = 5
    X_by_block, loadings_true = {}, {}
    for b in range(3):
        L = rng.normal(1, 0.3, m_per_block)
        loadings_true[b] = L
        X_by_block[b] = np.outer(z[:, b], L) + rng.normal(0, 0.4, (T, m_per_block))

    res = fit_spacetime_block_factor(X_by_block, [0, 1, 2], W)
    assert abs(res.rho1 - rho1_true) < 0.15
    assert abs(res.rho2 - rho2_true) < 0.2
    assert np.sign(res.rho2) == np.sign(rho2_true)
    for b in range(3):
        assert np.corrcoef(z[:, b], res.z[:, b])[0, 1] > 0.9
    assert np.all(np.diff(res.loglik) > -1e-3)  # monotone non-decreasing, up to numerical slack


def test_filter_spacetime_block_factor_projects_new_data():
    rng = np.random.default_rng(2)
    centroids = np.array([0.0, 10.0])
    W = sar_weight_matrix(centroids, bandwidth=8.0)
    rho1_true, rho2_true = 0.7, 0.15
    Phi_true = rho1_true * np.eye(2) + rho2_true * W

    def gen(T, seed):
        r = np.random.default_rng(seed)
        z = np.zeros((T, 2))
        for t in range(1, T):
            z[t] = Phi_true @ z[t - 1] + r.normal(0, 0.6, 2)
        X_by_block = {}
        for b in range(2):
            L = np.array([1.0, 0.8, -0.5])
            X_by_block[b] = np.outer(z[:, b], L) + r.normal(0, 0.3, (T, 3))
        return X_by_block, z

    X_train, _ = gen(500, 0)
    fit = fit_spacetime_block_factor(X_train, [0, 1], W)

    X_new, z_new = gen(150, 5)
    z_hat, _, _ = filter_spacetime_block_factor(
        X_new, [0, 1], fit.loadings, fit.obs_var, fit.Phi, fit.Q, fit.train_mean)
    for b in range(2):
        assert np.corrcoef(z_new[:, b], z_hat[:, b])[0, 1] > 0.8


def test_supervised_block_factor_recovers_known_max_covariance_direction():
    # Construct X_block with NO noise, as an exact scalar multiple of y along a known
    # loading direction c: X_block = outer(y, c). Then w = X.T @ y = (y @ y) * c, so the
    # normalized max-covariance direction is exactly c / ||c|| -- an analytic answer,
    # not an approximate recovery target.
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    c = np.array([2.0, -1.0, 0.5])
    X_block = np.outer(y, c)

    score, w = supervised_block_factor(X_block, y)
    assert np.allclose(w, c / np.linalg.norm(c), atol=1e-8)
    # score = X_block @ w = y * (c @ c) / ||c|| = y * ||c||, i.e. exactly proportional to y
    assert np.allclose(score, y * np.linalg.norm(c), atol=1e-8)


def test_apply_supervised_block_factor_matches_fit_on_same_data():
    rng = np.random.default_rng(1)
    n, m = 150, 4
    y = rng.normal(size=n)
    X_block = rng.normal(size=(n, m)) + 0.5 * np.outer(y, rng.normal(size=m))

    score_fit, w = supervised_block_factor(X_block, y)
    score_applied = apply_supervised_block_factor(X_block, w)
    assert np.allclose(score_fit, score_applied)


def test_text_embedding_relevance_ranks_related_group_higher():
    rel = TextEmbeddingRelevance()
    groups = ["insulin resistance and glucose regulation",
              "photosynthesis and light reactions in plants"]
    scores = rel.score(groups, query="insulin regulation of blood glucose")
    assert scores[0] > scores[1]


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
