"""Sanity checks for the spatial_boost package: run before trusting the demo."""
import numpy as np
from spatial_boost.rank_utils import TruncatedDesign, woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample
from spatial_boost.weights import build_gene_blocks, gene_weights, select_phi_by_region
from spatial_boost.model import fit_em, theta_conditional
from spatial_boost.simulate import simulate_dataset


def test_woodbury_solve_matches_dense():
    # full rank (rank = min(n, p1)) => the SVD surrogate reconstructs X exactly,
    # so the Woodbury solve must match the dense solve to numerical precision.
    rng = np.random.default_rng(0)
    n, p1 = 60, 40
    X = rng.standard_normal((n, p1))
    w = rng.uniform(0.1, 1.0, n)
    sigma_diag = rng.uniform(0.5, 2.0, p1)
    v = rng.standard_normal(p1)

    design = TruncatedDesign(X, rank=p1)
    S = design.weighted_factor(w)
    got = woodbury_solve(S, sigma_diag, v)

    XtWX_full = X.T @ (w[:, None] * X)
    want = np.linalg.solve(XtWX_full + np.diag(1.0 / sigma_diag), v)
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < 1e-6, f"woodbury_solve mismatch: rel err {err}"
    print("test_woodbury_solve_matches_dense (full rank): OK, rel err", err)

    # truncated rank => approximation error should shrink as rank grows towards p1.
    errs = []
    for rank in (5, 15, 30, 40):
        design = TruncatedDesign(X, rank=rank)
        S = design.weighted_factor(w)
        got = woodbury_solve(S, sigma_diag, v)
        errs.append(np.linalg.norm(got - want) / np.linalg.norm(want))
    print("test_woodbury_solve_matches_dense (truncated): errs by rank ->", errs)
    assert errs[-1] < 1e-6
    assert errs[0] >= errs[1] >= errs[2]  # error should be monotone non-increasing in rank


def test_woodbury_sampler_covariance():
    rng = np.random.default_rng(1)
    n, p1, rank = 50, 30, 30  # full rank so exact == truncated
    X = rng.standard_normal((n, p1))
    w = rng.uniform(0.2, 1.0, n)
    sigma_diag = rng.uniform(0.5, 1.5, p1)
    c = rng.standard_normal(p1)

    design = TruncatedDesign(X, rank)
    S = design.weighted_factor(w)

    n_draws = 20000
    draws = np.empty((n_draws, p1))
    for i in range(n_draws):
        draws[i], mean = woodbury_mean_and_sample(S, sigma_diag, c, rng)

    XtWX = X.T @ (w[:, None] * X)
    want_cov = np.linalg.inv(XtWX + np.diag(1.0 / sigma_diag))
    want_mean = want_cov @ c

    emp_cov = np.cov(draws.T)
    emp_mean = draws.mean(axis=0)

    mean_err = np.linalg.norm(emp_mean - want_mean) / np.linalg.norm(want_mean)
    cov_err = np.linalg.norm(emp_cov - want_cov) / np.linalg.norm(want_cov)
    print("test_woodbury_sampler_covariance: mean rel err", mean_err, "cov rel err", cov_err)
    assert mean_err < 0.05 and cov_err < 0.1


def test_gene_weights_shape_and_range():
    rng = np.random.default_rng(2)
    gene_starts = np.array([100.0, 500.0, 1500.0])
    gene_ends = np.array([200.0, 700.0, 1600.0])
    gene_rel = np.array([1.0, 2.0, 0.5])
    positions = np.array([150.0, 600.0, 1000.0, 1550.0])

    bl, br, br_rel = build_gene_blocks(gene_starts, gene_ends, gene_rel)
    wr = gene_weights(positions, bl, br, br_rel, phi=50.0)
    assert wr.shape == (4,)
    assert wr.max() <= 1.0 + 1e-9
    assert wr[2] < wr[0]  # marker 1000 is far from any gene, should get near-zero weight
    print("test_gene_weights_shape_and_range: OK, wr =", wr)


def test_select_phi_reasonable():
    rng = np.random.default_rng(3)
    n, p = 80, 40
    positions = np.sort(rng.uniform(0, 100_000, p))
    from spatial_boost.simulate import simulate_genotypes
    X = simulate_genotypes(n, positions, np.full(p, 0.3), ld_length=5_000, rng=rng)
    phi = select_phi_by_region(positions, X, min_gap=30_000)
    assert phi.shape == (p,)
    assert np.all(phi > 0)
    print("test_select_phi_reasonable: OK, phi range", phi.min(), phi.max())


def test_em_recovers_signal_direction():
    data = simulate_dataset(n=300, p=800, n_genes=60, scenario="informative", m_causal=8, seed=42)
    res = fit_em(data.X, data.y, data.wr, xi0=-6.0, xi1=2.0, kappa=1000.0, nu=1.0, lam=1.0)
    causal_theta = res.theta_hat[data.theta_true]
    noncausal_theta = res.theta_hat[~data.theta_true]
    print("test_em_recovers_signal_direction: mean theta causal =", causal_theta.mean(),
          " mean theta non-causal =", noncausal_theta.mean())
    assert causal_theta.mean() > noncausal_theta.mean()


if __name__ == "__main__":
    test_woodbury_solve_matches_dense()
    test_woodbury_sampler_covariance()
    test_gene_weights_shape_and_range()
    test_select_phi_reasonable()
    test_em_recovers_signal_direction()
    print("\nAll sanity checks passed.")
