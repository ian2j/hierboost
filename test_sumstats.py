"""Validation for hierboost.sumstats: the summary-statistics-only (R, bhat, n) spike-
and-slab fit, checked against hierboost.spike_slab_gaussian's individual-level (X, y)
fit it is meant to reproduce exactly (see hierboost/sumstats.py's module docstring for
the "why this is exact, not approximate" derivation: given X/y standardized so that
X'X/n == R, X'y/n == bhat, y'y/n == y_var hold EXACTLY rather than approximately, the
two EM procedures solve the identical normal equations at every iteration).

Uses real LD structure from ~/genomics_1kg/lct_region.npz (real 1000 Genomes phase 3
genotypes, already fetched by this project's genomics_1kg_fetch.py) when available, to
validate against a realistic block-correlated design rather than an idealized one; falls
back to a synthetic block-correlated design otherwise so the suite still runs on a
machine without that cache.
"""
import os
import numpy as np
import pytest

from hierboost.spike_slab_gaussian import fit_em_gaussian, em_filter_gaussian, ppl_gaussian
from hierboost.sumstats import (
    fit_em_sumstats, em_filter_sumstats, ppl_sumstats, ssr_sumstats,
    summarize_by_block, _augment_R_bhat, gibbs_sampler_sumstats, effective_rank,
)
from hierboost.blocks import threshold_blocks_1d

LCT_PATH = os.path.expanduser("~/genomics_1kg/lct_region.npz")


def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _synthetic_block_design(n=3000, n_blocks=8, block_size=6, rho=0.85, seed=0):
    """Fallback block-correlated design (AR(1)-within-block latent factor model) used
    only if the real 1000G cache isn't present."""
    rng = np.random.default_rng(seed)
    p = n_blocks * block_size
    X = np.empty((n, p))
    for b in range(n_blocks):
        z = rng.normal(size=n)
        e = rng.normal(size=(n, block_size))
        X[:, b * block_size:(b + 1) * block_size] = (
            np.sqrt(rho) * z[:, None] + np.sqrt(1 - rho) * e)
    block_id = np.repeat(np.arange(n_blocks), block_size)
    return X, block_id


def _make_problem(seed=0, p_max=150, n_true=3, snr_beta=0.35):
    """Build a standardized (Xs, ys), the true sparse coefficient vector used to
    generate it, sufficient statistics (R, bhat, n, y_var), and a block_id -- all from
    either the real LCT region cache or the synthetic fallback."""
    rng = np.random.default_rng(seed)
    if os.path.exists(LCT_PATH):
        d = np.load(LCT_PATH)
        X_raw = d["X"][:, :p_max].astype(float)
        positions = d["positions"][:p_max]
        block_id = threshold_blocks_1d(positions, zeta=float(np.percentile(
            np.diff(np.sort(positions)), 75)))
        source = "real 1000G LCT region (subset)"
    else:
        X_raw, block_id = _synthetic_block_design(seed=seed)
        source = "synthetic block-correlated fallback"

    # drop any zero-variance columns (can happen with a small real subset) before
    # standardizing, so R has no degenerate rows/cols
    keep = X_raw.std(axis=0) > 1e-8
    X_raw = X_raw[:, keep]
    block_id = block_id[keep]
    Xs = _standardize(X_raw)
    n, p = Xs.shape

    true_idx = rng.choice(p, size=n_true, replace=False)
    beta_true = np.zeros(p)
    beta_true[true_idx] = rng.choice([-1.0, 1.0], size=n_true) * snr_beta
    y = Xs @ beta_true + rng.normal(scale=1.0, size=n)
    ys = _standardize(y[:, None])[:, 0]

    R = (Xs.T @ Xs) / n
    bhat = (Xs.T @ ys) / n
    y_var = float(np.mean(ys ** 2))  # == 1.0 up to floating point, by construction

    return dict(Xs=Xs, ys=ys, R=R, bhat=bhat, n=n, y_var=y_var, block_id=block_id,
                true_idx=true_idx, source=source)


COMMON_HYPERPARAMS = dict(xi0=-1.0, xi1=0.0, kappa=50.0, nu=1.0, lam=1.0)


def _localized(true_idx, selected_idx, R, corr_thresh=0.8):
    """True if some selected SNP is the true SNP itself or in strong LD (|R| >=
    corr_thresh) with it -- real LD blocks routinely contain several SNPs that tag a
    causal variant almost perfectly (see the real-data diagnostic in this module's
    docstring-adjacent comment above test_fit_em_sumstats_recovers_true_signal), so
    "exact same index" is not the right bar for a real-genotype fine-mapping check;
    "the model lands in the same LD neighborhood" is -- this is the same "distance from
    the known causal SNP" standard genomics_1kg_locus.py already applies project-wide."""
    selected_idx = np.asarray(list(selected_idx))
    return bool(np.any(np.abs(R[true_idx, selected_idx]) >= corr_thresh))


def test_ssr_sumstats_matches_raw_ssr_exactly():
    """ssr_sumstats(R, bhat, beta, n) must equal sum((y - X@beta)**2) computed from the
    raw standardized data, to floating-point precision, for an ARBITRARY beta -- this is
    the algebraic identity the whole module rests on (see module docstring)."""
    prob = _make_problem(seed=1)
    Xs, ys, R, bhat, n = prob["Xs"], prob["ys"], prob["R"], prob["bhat"], prob["n"]
    p = Xs.shape[1]
    X_design = np.column_stack([np.ones(n), Xs])

    rng = np.random.default_rng(2)
    for _ in range(5):
        beta = rng.normal(scale=0.3, size=p + 1)
        raw_ssr = float(np.sum((ys - X_design @ beta) ** 2))
        R_aug, bhat_aug = _augment_R_bhat(R, bhat)
        ss_ssr = ssr_sumstats(R_aug, bhat_aug, beta, n, y_var=prob["y_var"])
        np.testing.assert_allclose(ss_ssr, raw_ssr, rtol=1e-8, atol=1e-6)


def test_fit_em_sumstats_matches_fit_em_gaussian():
    """The must-pass sanity gate: fit_em_sumstats(R, bhat, n, ...) must recover
    essentially the same beta/theta_hat as fit_em_gaussian(X, y, ...) run directly on
    the same (standardized) individual-level data."""
    prob = _make_problem(seed=3)
    Xs, ys, R, bhat, n = prob["Xs"], prob["ys"], prob["R"], prob["bhat"], prob["n"]
    wr = np.zeros(Xs.shape[1])
    X_design = np.column_stack([np.ones(n), Xs])

    raw = fit_em_gaussian(X_design, ys, wr, **COMMON_HYPERPARAMS)
    ss = fit_em_sumstats(R, bhat, n, wr, y_var=prob["y_var"], **COMMON_HYPERPARAMS)

    np.testing.assert_allclose(ss.beta, raw.beta, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ss.theta_hat, raw.theta_hat, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ss.sigma_g2, raw.sigma_g2, rtol=1e-4)
    np.testing.assert_allclose(ss.sigma_y2, raw.sigma_y2, rtol=1e-4)
    print(f"\n[{prob['source']}] max |beta diff| = "
          f"{np.max(np.abs(ss.beta - raw.beta)):.3e}")


def test_fit_em_sumstats_recovers_true_signal():
    """Independent of parity with the raw-data fit: does the sumstats-only fit actually
    localize the true sparse support at all (the applied, not just algebraic, check).
    Checked at the LD-neighborhood level, not exact SNP identity: real LCT-region LD is
    strong enough that several SNPs tag a causal one at |r| > 0.9 (confirmed by
    inspection -- e.g. seed=4's true SNPs each have 5-10 near-perfect proxies within
    this 150-SNP window), so a multiple-regression spike-and-slab correctly distributing
    inclusion probability across an LD-linked proxy instead of the exact simulated SNP
    is expected fine-mapping behavior, not a bug (the same reason real fine-mapping
    reports credible SETS, not point estimates)."""
    prob = _make_problem(seed=4, n_true=3, snr_beta=0.5)
    R, bhat, n = prob["R"], prob["bhat"], prob["n"]
    wr = np.zeros(R.shape[0])
    ss = fit_em_sumstats(R, bhat, n, wr, y_var=prob["y_var"], **COMMON_HYPERPARAMS)
    top10 = np.argsort(ss.theta_hat)[::-1][:10]
    for t in prob["true_idx"]:
        assert _localized(t, top10, R), f"true SNP {t} has no LD-proxy in top10 {top10}"


def test_em_filter_sumstats_matches_em_filter_gaussian():
    """Same must-pass parity gate, but for the full filtering pipeline (repeated fit +
    drop-lowest-theta rounds) rather than a single fit -- checks that the two loops make
    the same drop decisions and land on the same retained set."""
    prob = _make_problem(seed=5, p_max=80)
    Xs, ys, R, bhat, n = prob["Xs"], prob["ys"], prob["R"], prob["bhat"], prob["n"]
    wr = np.zeros(Xs.shape[1])
    X_design = np.column_stack([np.ones(n), Xs])

    raw_filt = em_filter_gaussian(X_design, ys, wr, **COMMON_HYPERPARAMS,
                                   filter_frac=0.3, min_features=5, patience=3)
    ss_filt = em_filter_sumstats(R, bhat, n, wr, block_id=prob["block_id"],
                                  y_var=prob["y_var"], **COMMON_HYPERPARAMS,
                                  filter_frac=0.3, min_features=5, patience=3)

    raw_best, ss_best = raw_filt.best, ss_filt.best
    assert set(ss_best.retained_idx.tolist()) == set(raw_best.retained_idx.tolist())
    np.testing.assert_allclose(ss_best.beta, raw_best.beta, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ss_best.ppl, raw_best.ppl, rtol=1e-4)
    # true signal (or an LD proxy of it -- see test_fit_em_sumstats_recovers_true_signal)
    # should have survived filtering down to the retained set
    for t in prob["true_idx"]:
        assert _localized(t, ss_best.retained_idx, R), \
            f"true SNP {t} has no LD-proxy retained in {ss_best.retained_idx}"


def test_rank_truncated_path_matches_exact_at_full_rank():
    """TruncatedDesignFromR/woodbury_solve at rank == p1 (no truncation) should agree
    with the exact dense solve, the same way rank_utils.TruncatedDesign does for raw X."""
    prob = _make_problem(seed=6, p_max=60)
    R, bhat, n = prob["R"], prob["bhat"], prob["n"]
    wr = np.zeros(R.shape[0])
    exact = fit_em_sumstats(R, bhat, n, wr, y_var=prob["y_var"], **COMMON_HYPERPARAMS)
    full_rank = fit_em_sumstats(R, bhat, n, wr, y_var=prob["y_var"], rank=R.shape[0] + 1,
                                 **COMMON_HYPERPARAMS)
    np.testing.assert_allclose(full_rank.beta, exact.beta, rtol=1e-4, atol=1e-4)


def test_summarize_by_block_reports_true_block():
    prob = _make_problem(seed=7, n_true=1, snr_beta=0.6)
    R, bhat, n, block_id = prob["R"], prob["bhat"], prob["n"], prob["block_id"]
    wr = np.zeros(R.shape[0])
    filt = em_filter_sumstats(R, bhat, n, wr, block_id=block_id, y_var=prob["y_var"],
                               **COMMON_HYPERPARAMS, filter_frac=0.3, min_features=5)
    summary = summarize_by_block(filt.best, block_id)
    true_block = int(block_id[prob["true_idx"][0]])
    assert true_block in summary
    assert summary[true_block]["top_snp"] in filt.best.retained_idx.tolist()


def test_effective_rank_detects_real_ld_collinearity():
    """Real, densely-sampled LD is routinely much more rank-deficient than p<=n_ref
    alone would suggest (haplotype-block redundancy) -- confirmed on the real LCT
    region during development (254/473 eigenvalues below 1e-3 x the top eigenvalue in a
    p=473 window from 503 reference individuals). effective_rank should catch this on
    the same real data."""
    prob = _make_problem(seed=9, p_max=300)
    r_eff = effective_rank(prob["R"])
    assert r_eff < prob["R"].shape[0], (
        "expected real LD to be numerically rank-deficient, as observed during "
        "development -- if this now fails, either the cached region changed or "
        "effective_rank regressed")


def test_effective_rank_exact_on_synthetic_low_rank_matrix():
    """Sanity check on a matrix with an EXACTLY known rank (independent of any real-data
    quirks): a 3-factor Gaussian model gives a p x p correlation-like matrix with rank
    exactly 3 (plus the trivial full-rank identity component), so effective_rank at a
    strict tolerance should recover that exactly."""
    rng = np.random.default_rng(0)
    p, k = 20, 3
    L = rng.normal(size=(p, k))
    R_low_rank = L @ L.T
    R_low_rank /= np.sqrt(np.outer(np.diag(R_low_rank), np.diag(R_low_rank)))
    assert effective_rank(R_low_rank, rel_tol=1e-9) == k


def test_fit_em_sumstats_stays_bounded_with_huge_n_and_rank_deficient_R():
    """Regression test for a real divergence found and fixed during development: with a
    severely rank-deficient real-LD R (see test_effective_rank_detects_real_ld_
    collinearity) and a GWAS-scale n (~10^6, as in real GLGC summary stats -- far larger
    than any n seen in the rest of this test module, which uses actual sample sizes of a
    few thousand), the EXACT dense solve either raises LinAlgError or, when it doesn't,
    the M-step can still diverge (ssr collapses to 0, sigma_y2 collapses toward 0, beta
    blows up past 1e15 within a few iterations) if bhat has any component -- even
    floating-point-noise-sized -- outside R's numerically-supported subspace, since a
    huge n amplifies that component's effect on the RHS with no matching curvature to
    restrain it. rank="auto" (TruncatedDesignFromR + consistently projecting bhat onto
    the same truncated eigenbasis, not just truncating R's curvature) must keep the fit
    finite and bounded."""
    prob = _make_problem(seed=10, p_max=300)
    R, bhat = prob["R"], prob["bhat"]
    wr = np.zeros(R.shape[0])
    huge_n = 1_000_000

    res = fit_em_sumstats(R, bhat, huge_n, wr, y_var=1.0, rank="auto", **COMMON_HYPERPARAMS)
    assert np.all(np.isfinite(res.beta)), "beta diverged to non-finite with rank='auto'"
    assert np.max(np.abs(res.beta)) < 100.0, (
        f"beta magnitude {np.max(np.abs(res.beta)):.3e} suggests the divergence this "
        f"test guards against has come back")
    assert np.isfinite(res.sigma_y2) and res.sigma_y2 > 1e-6


def test_gibbs_sampler_sumstats_runs_and_is_sane():
    """Smoke test for the Gibbs sampler analog: not compared iteration-by-iteration to
    gibbs_sampler_gaussian (different RNG paths), but should produce a pi_hat that
    ranks the true signal on top, same standard as the raw-data Gibbs sampler."""
    prob = _make_problem(seed=8, p_max=40, n_true=2, snr_beta=0.6)
    R, bhat, n = prob["R"], prob["bhat"], prob["n"]
    wr = np.zeros(R.shape[0])
    gr = gibbs_sampler_sumstats(R, bhat, n, wr, y_var=prob["y_var"], n_samples=300,
                                 burn_in=100, seed=0, **COMMON_HYPERPARAMS)
    top2 = np.argsort(gr.pi_hat)[::-1][:2]
    assert set(top2) == set(prob["true_idx"].tolist())
