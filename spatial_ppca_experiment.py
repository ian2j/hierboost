"""Out-of-curiosity experiment: does forcing a block's PPCA loading direction to be
the SAR-implied smoothing vector ell(phi) = (I - B(phi))^-1 @ 1 (instead of letting it
be the empirical leading eigenvector, as hierboost.factor.gaussian_block_factor does)
ever help? This is the "spatially-informed PPCA" variant discussed but not built in
hierboost -- unlike hierboost.latent's discrete/Binomial Ch4 model, this stays fully
closed-form/conjugate because Gaussian-on-Gaussian is still Gaussian; the only thing
that needs fitting is the scalar bandwidth phi (plus a scalar factor-variance tau and
per-feature idiosyncratic variances sigma_j^2), via EM with the loading DIRECTION held
fixed at ell(phi)/||ell(phi)|| each iteration.

Two things tested:
1. Synthetic, correctly-specified (ground truth really is SAR-shaped): does the
   1-parameter (phi) constraint reduce variance vs. plain PPCA's m-parameter free
   loading, especially at small n?
2. Synthetic, misspecified (ground truth loading is an arbitrary direction unrelated
   to coords): does the constraint introduce a bias floor that free PPCA doesn't have?
3. Real data (one real 1000 Genomes LD block, one real UK-weather station cluster):
   does either regime actually describe real data, via held-out log-likelihood CV?
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

from hierboost.kernels import sar_weight_matrix
from hierboost.factor import gaussian_block_factor, project_block_factor


# ---------------------------------------------------------------------------
# The spatially-informed / SAR-constrained one-factor Gaussian model
# ---------------------------------------------------------------------------

def sar_loading_direction(coords, phi):
    """ell(phi) = (I - B(phi))^-1 @ 1, normalized to a unit vector -- the SAR-implied
    "effective loading" of a shared per-individual scalar factor onto each block member,
    after being smoothed through the same B matrix hierboost.latent's discrete model uses.
    """
    coords = np.asarray(coords, dtype=float)
    B = sar_weight_matrix(coords, phi)
    m = B.shape[0]
    C = np.linalg.inv(np.eye(m) - B)
    ell = C @ np.ones(m)
    norm = np.linalg.norm(ell)
    return ell / norm if norm > 0 else ell


def fit_phi_by_correlation_match(coords, corr_abs, phi_bounds=(1e-2, 1e6)):
    """Moment-match phi so the rank-1 correlation pattern implied by ell(phi) --
    |ell_j * ell_k| for a unit-norm ell -- best matches the observed |correlation|
    pattern, by least squares. Direct analogue of hierboost.kernels.fit_gaussian_
    bandwidth, but matching the actual SAR-implied loading shape instead of a raw
    kernel form.
    """
    iu = np.triu_indices(len(coords), k=1)
    target = np.abs(corr_abs)[iu]

    def loss(log_phi):
        ell = sar_loading_direction(coords, np.exp(log_phi))
        implied = np.abs(np.outer(ell, ell))[iu]
        return np.mean((implied - target) ** 2)

    lo, hi = np.log(phi_bounds[0]), np.log(phi_bounds[1])
    res = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
    return float(np.exp(res.x))


def _em_fixed_direction(Xc, ell, n_em=50, tol=1e-8, min_var_frac=0.02):
    """EM for the scalar tau and per-feature sigma2, with the unit loading DIRECTION
    ell held fixed -- the exact conditional MLE of (tau, sigma2) given that direction.
    Shared by sar_constrained_factor's final fit and fit_phi_by_profile_likelihood's
    inner loop (run once per candidate phi during the outer scalar search).

    Heywood-case guard: with the direction FIXED (not free, as in ordinary ML factor
    analysis), the sigma2 M-step can still drive one feature's residual variance to
    ~0 whenever ell happens to align unusually well with that one feature -- confirmed
    empirically on real LCT genotype data (sigma2 collapsed to the numerical floor for
    one SNP, producing a spuriously huge held-out log-likelihood). Floor sigma2 at a
    small fraction of the feature's own raw variance, the same role hierboost.
    spike_slab's Inv-Gamma prior on sigma2 plays elsewhere in this codebase -- a weak
    regularizer, not a hard constraint.
    """
    n, m = Xc.shape
    var_floor = min_var_frac * (Xc.var(axis=0) + 1e-12)
    sigma2 = Xc.var(axis=0) + 1e-6
    tau = 1.0
    prev_tau = tau
    for _ in range(n_em):
        loading = tau * ell
        z_var = 1.0 / (1.0 + np.sum(loading ** 2 / sigma2))
        z_mean = z_var * (Xc @ (loading / sigma2))
        Ez2_sum = np.sum(z_mean ** 2) + n * z_var

        Sjz = Xc.T @ z_mean  # (m,), sum_i x_ij * E[z_i]
        num = np.sum((ell / sigma2) * Sjz)
        denom = np.sum(ell ** 2 / sigma2) * Ez2_sum
        tau = num / denom if denom > 0 else 0.0

        loading = tau * ell
        sigma2 = (np.sum(Xc ** 2, axis=0) - 2.0 * loading * Sjz
                  + loading ** 2 * Ez2_sum) / n
        sigma2 = np.maximum(sigma2, var_floor)

        if abs(tau - prev_tau) < tol * (abs(prev_tau) + 1e-12):
            break
        prev_tau = tau
    return tau, sigma2


def _rank1_loglik_total(Xc, ell, tau, sigma2):
    """Total Gaussian log-likelihood of Xc under Sigma = tau^2 * ell @ ell.T +
    diag(sigma2), via the matrix-determinant lemma / Sherman-Morrison rank-1 update --
    avoids ever forming or inverting the m x m covariance matrix. Same low-rank-plus-
    diagonal trick hierboost.rank_utils uses throughout the rest of this codebase,
    applied here to the profile-likelihood search below (called once per candidate phi,
    so keeping it O(m) instead of O(m^3) matters if this were ever scaled to larger
    blocks)."""
    n, m = Xc.shape
    inv_sigma2 = 1.0 / sigma2
    S = np.sum(ell ** 2 * inv_sigma2)
    denom = 1.0 + tau ** 2 * S
    proj = Xc @ (ell * inv_sigma2)
    quad = np.sum(Xc ** 2 * inv_sigma2, axis=1) - (tau ** 2 / denom) * proj ** 2
    logdet = np.sum(np.log(sigma2)) + np.log(denom)
    ll_per_row = -0.5 * (m * np.log(2.0 * np.pi) + logdet + quad)
    return float(ll_per_row.sum())


def fit_phi_by_profile_likelihood(Xc, coords, phi_bounds=(1e-2, 1e6), n_em=50, min_var_frac=0.02):
    """The proper replacement for fit_phi_by_correlation_match: a genuine joint MLE of
    phi via profile likelihood, not a two-step correlation-pattern moment-match.

    For each candidate phi: ell(phi) is deterministic (just the SAR mechanism), so
    run the fixed-direction EM (_em_fixed_direction) to get the EXACT conditional MLE
    of (tau, sigma2) given that direction, then evaluate the exact marginal Gaussian
    log-likelihood at that optimum (_rank1_loglik_total). That defines a genuine
    profile log-likelihood L(phi) = max_{tau,sigma2} log p(X | phi, tau, sigma2), and
    a 1-D bounded scalar search over log(phi) finds its maximizer -- the actual joint
    MLE of all three parameters, since profiling out (tau, sigma2) exactly at their
    own conditional MLE for every phi tried is what makes this correct (not an
    approximation), unlike matching a separately-computed empirical correlation
    matrix, which throws away the exact likelihood in favor of a cruder pairwise-
    moment surrogate.
    """
    def neg_profile_ll(log_phi):
        phi = np.exp(log_phi)
        ell = sar_loading_direction(coords, phi)
        tau, sigma2 = _em_fixed_direction(Xc, ell, n_em=n_em, min_var_frac=min_var_frac)
        return -_rank1_loglik_total(Xc, ell, tau, sigma2)

    lo, hi = np.log(phi_bounds[0]), np.log(phi_bounds[1])
    res = minimize_scalar(neg_profile_ll, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-3})
    return float(np.exp(res.x))


def sar_constrained_factor(X_block, coords, phi=None, phi_method="profile",
                            n_em=50, tol=1e-8, min_var_frac=0.02):
    """One-factor Gaussian model x_i = tau * ell(phi) * z_i + eps_i, z_i ~ N(0,1),
    eps_i ~ N(0, diag(sigma2)), with the loading DIRECTION ell(phi)/||ell(phi)|| fixed
    by the SAR mechanism (physical coordinates), and (tau, sigma2_j) fit by EM. This
    is literally hierboost.factor.gaussian_block_factor's model with one extra
    constraint (loading direction is parametric, not free) -- everything else
    (Gaussian obs, one shared factor per individual) is unchanged.

    `phi_method="profile"` (default, new): fit_phi_by_profile_likelihood, the proper
    joint MLE. `phi_method="moment_match"`: the original, cruder two-step correlation-
    pattern match (fit_phi_by_correlation_match) -- kept only so the two can be
    compared directly; not recommended otherwise.

    Returns (factor_scores (n,), loading (m,) = tau*ell, sigma2 (m,), phi, z_var).
    """
    X = np.asarray(X_block, dtype=float)
    Xc = X - X.mean(axis=0)

    if phi is None:
        if phi_method == "profile":
            phi = fit_phi_by_profile_likelihood(Xc, coords, n_em=n_em, min_var_frac=min_var_frac)
        elif phi_method == "moment_match":
            corr = np.abs(np.nan_to_num(np.corrcoef(Xc.T), nan=0.0))
            phi = fit_phi_by_correlation_match(coords, corr)
        else:
            raise ValueError(f"unknown phi_method {phi_method!r}")
    ell = sar_loading_direction(coords, phi)
    tau, sigma2 = _em_fixed_direction(Xc, ell, n_em=n_em, tol=tol, min_var_frac=min_var_frac)

    loading = tau * ell
    z_var = 1.0 / (1.0 + np.sum(loading ** 2 / sigma2))
    z_mean = z_var * (Xc @ (loading / sigma2))
    return z_mean, loading, sigma2, phi, z_var


# ---------------------------------------------------------------------------
# Fix 1: cheap, retroactive sign-alignment preprocessing (per project memory
# "spatial-ppca-sign-flip" -- ell(phi) is entrywise non-negative by construction,
# so it cannot represent a feature anti-correlated with its block's shared factor,
# e.g. an arbitrary ref/alt allele-coding artifact). Detect the sign via one cheap
# plain-PPCA pass, flip disagreeing raw features, then fit the existing
# SAR-constrained model unchanged on the now-consistently-signed data.
# ---------------------------------------------------------------------------

def sign_align_flips(X_block):
    """+/-1 per feature: flip any feature whose plain-PPCA loading disagrees in sign
    with the block's consensus (majority) direction. One cheap PPCA pass, no coords
    needed -- this is pure sign detection, not a claim about the SAR shape."""
    _, loadings = gaussian_block_factor(X_block)
    return np.where(loadings < 0, -1.0, 1.0)


def sar_constrained_factor_signcorrected(X_block, coords, phi_method="profile",
                                          n_em=50, tol=1e-8, min_var_frac=0.02):
    """sar_constrained_factor preceded by sign_align_flips. Loading is reported back
    in the ORIGINAL (unflipped) feature space, so it composes with block_loglik on
    raw held-out data exactly like every other fit here -- flipping a feature negates
    off-diagonal covariance entries correctly and leaves sigma2 untouched (since
    flips_j^2 == 1), so no separate transform of held-out data is needed.
    Returns the same tuple as sar_constrained_factor plus `flips`.
    """
    X = np.asarray(X_block, dtype=float)
    flips = sign_align_flips(X)
    X_flipped = X * flips[None, :]
    z_mean, loading_f, sigma2, phi, z_var = sar_constrained_factor(
        X_flipped, coords, phi_method=phi_method, n_em=n_em, tol=tol, min_var_frac=min_var_frac)
    loading = loading_f * flips
    return z_mean, loading, sigma2, phi, z_var, flips


# ---------------------------------------------------------------------------
# Fix 2: the actual generalization -- a free per-feature loading w_j with a soft
# empirical-Bayes prior pulling it toward the SAR-implied direction, instead of a
# HARD constraint pinning it there. tau_c2 -> 0 recovers today's rigid SAR model;
# tau_c2 -> infinity recovers ordinary unconstrained PPCA; a real block lands
# wherever the data supports, and can represent a sign flip (w_j can go negative)
# because w_j is free, not literally tau*ell_j. mu0 (how strongly, and tau_c2 (how
# loosely) the data should be pulled toward ell are both estimated from the data by
# their own closed-form M-steps -- no hand-picked hyperparameter needed.
# ---------------------------------------------------------------------------

def _rank1_loglik_total_w(Xc, w, sigma2):
    """Same rank-1 (matrix-determinant-lemma/Sherman-Morrison) log-likelihood as
    _rank1_loglik_total, generalized to a free loading vector w instead of tau*ell
    (_rank1_loglik_total(Xc, ell, tau, sigma2) is the special case w = tau*ell)."""
    n, m = Xc.shape
    inv_sigma2 = 1.0 / sigma2
    S = np.sum(w ** 2 * inv_sigma2)
    denom = 1.0 + S
    proj = Xc @ (w * inv_sigma2)
    quad = np.sum(Xc ** 2 * inv_sigma2, axis=1) - (1.0 / denom) * proj ** 2
    logdet = np.sum(np.log(sigma2)) + np.log(denom)
    ll_per_row = -0.5 * (m * np.log(2.0 * np.pi) + logdet + quad)
    return float(ll_per_row.sum())


def _em_shrunk_direction(Xc, ell, n_em=100, tol=1e-8, min_var_frac=0.02):
    """EM for a FREE loading w with prior w_j ~ N(mu0*ell_j, tau_c2), instead of
    _em_fixed_direction's hard w = tau*ell. Every M-step is closed form:

      w_j    = (Sjz_j/sigma2_j + mu0*ell_j/tau_c2) / (Ez2_sum/sigma2_j + 1/tau_c2)
               -- ridge regression of x_j on z, shrunk toward mu0*ell_j instead of 0.
               tau_c2 -> 0 forces w -> mu0*ell (today's rigid model); tau_c2 -> infinity
               drops the 1/tau_c2 terms entirely, recovering PLAIN PPCA's own
               (sigma2-independent) free M-step exactly.
      sigma2 = standard per-feature residual-variance M-step, unchanged in form.
      mu0    = dot(ell, w) -- least-squares fit of w against the unit vector ell.
      tau_c2 = mean((w - mu0*ell)^2) -- empirical-Bayes: how much the data-fitted w
               actually deviates from the SAR shape, re-estimated every iteration so a
               genuinely sign-flipped or off-shape feature automatically loosens the
               prior instead of needing a hand-picked shrinkage strength.

    Returns (w, sigma2, mu0, tau_c2).
    """
    n, m = Xc.shape
    var_floor = min_var_frac * (Xc.var(axis=0) + 1e-12)
    sigma2 = Xc.var(axis=0) + 1e-6
    w = ell.copy()
    mu0 = 1.0
    tau_c2 = 0.1
    prev_w = w.copy()
    for _ in range(n_em):
        z_var = 1.0 / (1.0 + np.sum(w ** 2 / sigma2))
        z_mean = z_var * (Xc @ (w / sigma2))
        Ez2_sum = np.sum(z_mean ** 2) + n * z_var

        Sjz = Xc.T @ z_mean
        w = (Sjz / sigma2 + mu0 * ell / tau_c2) / (Ez2_sum / sigma2 + 1.0 / tau_c2)

        sigma2 = (np.sum(Xc ** 2, axis=0) - 2.0 * w * Sjz + w ** 2 * Ez2_sum) / n
        sigma2 = np.maximum(sigma2, var_floor)

        mu0 = float(np.dot(ell, w))
        tau_c2 = float(np.mean((w - mu0 * ell) ** 2)) + 1e-8

        if np.linalg.norm(w - prev_w) < tol * (np.linalg.norm(prev_w) + 1e-12):
            break
        prev_w = w.copy()
    return w, sigma2, mu0, tau_c2


def fit_phi_shrunk(Xc, coords, phi_bounds=(1e-2, 1e6), n_em=100, min_var_frac=0.02):
    """Profile-likelihood fit of phi for the shrinkage model -- identical outer 1-D
    search to fit_phi_by_profile_likelihood, just with the inner fit swapped from
    _em_fixed_direction to _em_shrunk_direction and the log-lik evaluated on the
    resulting free w via _rank1_loglik_total_w."""
    def neg_ll(log_phi):
        phi = np.exp(log_phi)
        ell = sar_loading_direction(coords, phi)
        w, sigma2, mu0, tau_c2 = _em_shrunk_direction(Xc, ell, n_em=n_em, min_var_frac=min_var_frac)
        return -_rank1_loglik_total_w(Xc, w, sigma2)

    lo, hi = np.log(phi_bounds[0]), np.log(phi_bounds[1])
    res = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    return float(np.exp(res.x))


def sar_shrinkage_factor(X_block, coords, phi=None, sign_correct=False,
                          n_em=100, min_var_frac=0.02):
    """Top-level fit for Fix 2: empirical-Bayes shrinkage toward the SAR direction.
    `sign_correct=True` additionally sign-aligns raw features first (Fix 1) -- worth
    trying combined, since Fix 1 gets the EM off to a better-signed start even though
    Fix 2's free w can in principle recover a flipped sign on its own.

    Returns (factor_scores, loading (m,), sigma2 (m,), phi, z_var, mu0, tau_c2).
    """
    X = np.asarray(X_block, dtype=float)
    flips = sign_align_flips(X) if sign_correct else np.ones(X.shape[1])
    X_flipped = X * flips[None, :]
    Xc = X_flipped - X_flipped.mean(axis=0)

    if phi is None:
        phi = fit_phi_shrunk(Xc, coords, n_em=n_em, min_var_frac=min_var_frac)
    ell = sar_loading_direction(coords, phi)
    w, sigma2, mu0, tau_c2 = _em_shrunk_direction(Xc, ell, n_em=n_em, min_var_frac=min_var_frac)

    loading = w * flips  # back to original feature-sign space; sigma2 unaffected by +/-1 flip
    z_var = 1.0 / (1.0 + np.sum(w ** 2 / sigma2))
    z_mean = z_var * (Xc @ (w / sigma2))
    return z_mean, loading, sigma2, phi, z_var, mu0, tau_c2


def block_loglik(X_block, loading, sigma2, train_mean):
    """Held-out marginal log-likelihood under a fitted one-factor model, per row,
    Sigma = outer(loading, loading) + diag(sigma2) -- same evaluation for both the
    free-PPCA and SAR-constrained fits, so this is the fair common yardstick."""
    Xc = np.asarray(X_block, dtype=float) - train_mean
    Sigma = np.outer(loading, loading) + np.diag(sigma2)
    return multivariate_normal.logpdf(Xc, mean=np.zeros(Xc.shape[1]), cov=Sigma, allow_singular=True)


def plain_ppca_fit(X_block, min_var_frac=0.02):
    """gaussian_block_factor plus the obs_var it computes internally but doesn't
    return -- needed here to score held-out log-likelihood on the same footing as
    the SAR-constrained variant. Same relative Heywood-case floor as
    sar_constrained_factor, so neither model gets an unfair numerical-degeneracy
    advantage/penalty in the comparison."""
    scores, loadings = gaussian_block_factor(X_block)
    X = np.asarray(X_block, dtype=float)
    Xc = X - X.mean(axis=0)
    resid = Xc - np.outer(scores, loadings)
    var_floor = min_var_frac * (Xc.var(axis=0) + 1e-12)
    obs_var = np.maximum(resid.var(axis=0), var_floor)
    return scores, loadings, obs_var, X.mean(axis=0)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def run_synthetic(seed, m=10, n_list=(10, 20, 50, 100, 300, 1000), n_reps=30, misspecified=False):
    """Four-way comparison per replicate: plain PPCA, SAR with the old moment-matched
    phi, SAR with the new profile-likelihood phi, and SAR with the ORACLE true phi
    (an upper bound on what any phi-estimator could achieve, not a real estimator)."""
    rng = np.random.default_rng(seed)
    coords = np.linspace(0, 9, m)
    true_phi = 2.0
    ell_sar = sar_loading_direction(coords, true_phi)

    if misspecified:
        ell_true = rng.normal(size=m)
        ell_true /= np.linalg.norm(ell_true)
    else:
        ell_true = ell_sar

    tau_true = 1.5
    sigma2_true = np.full(m, 1.0)

    keys = ["ppca_corr", "mm_corr", "prof_corr", "oracle_corr",
            "ppca_angle", "mm_angle", "prof_angle", "oracle_angle",
            "phi_mm", "phi_prof"]
    results = {n: {k: [] for k in keys} for n in n_list}

    for n in n_list:
        for rep in range(n_reps):
            rs = np.random.default_rng(seed * 100000 + n * 100 + rep)
            z_true = rs.normal(size=n)
            noise = rs.normal(size=(n, m)) * np.sqrt(sigma2_true)
            X = tau_true * np.outer(z_true, ell_true) + noise

            scores_p, loadings_p, _, _ = plain_ppca_fit(X)
            scores_mm, loadings_mm, _, phi_mm, _ = sar_constrained_factor(X, coords, phi_method="moment_match")
            scores_pr, loadings_pr, _, phi_pr, _ = sar_constrained_factor(X, coords, phi_method="profile")
            scores_or, loadings_or, _, _, _ = sar_constrained_factor(X, coords, phi=true_phi)

            def corr(s):
                return abs(np.corrcoef(s, z_true)[0, 1])

            def angle(l):
                return abs(np.dot(l / np.linalg.norm(l), ell_true))

            results[n]["ppca_corr"].append(corr(scores_p))
            results[n]["mm_corr"].append(corr(scores_mm))
            results[n]["prof_corr"].append(corr(scores_pr))
            results[n]["oracle_corr"].append(corr(scores_or))
            results[n]["ppca_angle"].append(angle(loadings_p))
            results[n]["mm_angle"].append(angle(loadings_mm))
            results[n]["prof_angle"].append(angle(loadings_pr))
            results[n]["oracle_angle"].append(angle(loadings_or))
            results[n]["phi_mm"].append(phi_mm)
            results[n]["phi_prof"].append(phi_pr)

    return results, ell_true, ell_sar


def print_synthetic_table(results, title, true_phi=2.0):
    print(f"\n=== {title} (true phi={true_phi}) ===")
    print(f"{'n':>6} | {'|corr(score, true z)|':^40} | {'phi_hat (median)':^16}")
    print(f"{'':>6} | {'PPCA':>9} {'SAR-mmatch':>11} {'SAR-profile':>12} {'SAR-oracle':>11} | "
          f"{'mmatch':>8} {'profile':>8}")
    for n, r in results.items():
        print(f"{n:>6} | {np.mean(r['ppca_corr']):>9.3f} {np.mean(r['mm_corr']):>11.3f} "
              f"{np.mean(r['prof_corr']):>12.3f} {np.mean(r['oracle_corr']):>11.3f} | "
              f"{np.median(r['phi_mm']):>8.2f} {np.median(r['phi_prof']):>8.2f}")


# ---------------------------------------------------------------------------
# Real-data tests
# ---------------------------------------------------------------------------

def real_data_cv(X, coords, n_folds=5, seed=0, label="", methods=None):
    """Real-data CV across all variants built so far. `methods` (default: all) picks
    a subset by key from: ppca, sar, sar_sc, shrink, shrink_sc."""
    all_methods = ["ppca", "sar", "sar_sc", "shrink", "shrink_sc"]
    methods = methods or all_methods
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fold_id = rng.permutation(n) % n_folds

    ll = {m: [] for m in methods}
    phis = {m: [] for m in methods if m != "ppca"}

    for k in range(n_folds):
        test = fold_id == k
        train = ~test
        X_train, X_test = X[train], X[test]
        mean_s = X_train.mean(axis=0)

        if "ppca" in methods:
            _, loadings_p, obs_var_p, mean_p = plain_ppca_fit(X_train)
            ll["ppca"].append(block_loglik(X_test, loadings_p, obs_var_p, mean_p).mean())

        if "sar" in methods:
            _, loadings, sigma2, phi, _ = sar_constrained_factor(X_train, coords, phi_method="profile")
            ll["sar"].append(block_loglik(X_test, loadings, sigma2, mean_s).mean())
            phis["sar"].append(phi)

        if "sar_sc" in methods:
            _, loadings, sigma2, phi, _, _ = sar_constrained_factor_signcorrected(X_train, coords)
            ll["sar_sc"].append(block_loglik(X_test, loadings, sigma2, mean_s).mean())
            phis["sar_sc"].append(phi)

        if "shrink" in methods:
            _, loadings, sigma2, phi, _, mu0, tau_c2 = sar_shrinkage_factor(X_train, coords, sign_correct=False)
            ll["shrink"].append(block_loglik(X_test, loadings, sigma2, mean_s).mean())
            phis["shrink"].append(phi)

        if "shrink_sc" in methods:
            _, loadings, sigma2, phi, _, mu0, tau_c2 = sar_shrinkage_factor(X_train, coords, sign_correct=True)
            ll["shrink_sc"].append(block_loglik(X_test, loadings, sigma2, mean_s).mean())
            phis["shrink_sc"].append(phi)

    labels = {"ppca": "PPCA (free, unconstrained)",
              "sar": "SAR-constrained (profile phi)",
              "sar_sc": "SAR-constrained + sign-align",
              "shrink": "Shrinkage (free w, EB prior)",
              "shrink_sc": "Shrinkage + sign-align"}
    print(f"\n=== Real data: {label} (m={X.shape[1]} features, n={n}) ===")
    print(f"Held-out mean log-lik per row:")
    for m in methods:
        phi_str = f"   (phi: {[f'{p:.2f}' for p in phis[m]]})" if m != "ppca" else ""
        print(f"  {labels[m]:32s}: {np.mean(ll[m]):>8.3f} +/- {np.std(ll[m]):.3f}{phi_str}")
    return ll, phis


def load_region_block(npz_path, m_target=18):
    """Densest-window LD block from any of the cached 1000-Genomes region files
    (same X/positions layout as lct_region.npz) -- slides a window of m_target
    consecutive (position-sorted) SNPs and keeps the one with the smallest total
    physical span, i.e. the tightest real LD block available in that region."""
    d = np.load(npz_path)
    X, pos = d["X"].astype(float), d["positions"].astype(float)
    order = np.argsort(pos)
    X, pos = X[:, order], pos[order]
    best = None
    for start in range(len(pos) - m_target):
        span = pos[start + m_target - 1] - pos[start]
        if best is None or span < best[0]:
            best = (span, start)
    _, start = best
    idx = np.arange(start, start + m_target)
    return X[:, idx], pos[idx]


def load_lct_block(m_target=18):
    return load_region_block("/home/ian/genomics_1kg/lct_region.npz", m_target)


def load_weather_cluster(target_idx=0, n_neighbors=9):
    d = np.load("/home/ian/Research/Math/prompts/uk_weather/data/precip_raw.npz")
    precip, centroids = d["precip"], d["centroids"]  # (51, T), (51, 2)
    dists = np.linalg.norm(centroids - centroids[target_idx], axis=1)
    idx = np.argsort(dists)[: n_neighbors + 1]
    X = np.log1p(precip[idx]).T  # (T, m)
    return X, centroids[idx]


if __name__ == "__main__":
    print("############################################################")
    print("# Synthetic test 1: ground truth genuinely SAR-shaped")
    print("############################################################")
    res1, ell_true1, ell_sar1 = run_synthetic(seed=0, misspecified=False)
    print_synthetic_table(res1, "Correctly-specified (true loading = SAR ell(phi=2.0))")

    print("\n############################################################")
    print("# Synthetic test 2: ground truth is an arbitrary (non-SAR) direction")
    print("############################################################")
    res2, ell_true2, ell_sar2 = run_synthetic(seed=1, misspecified=True)
    print_synthetic_table(res2, "Misspecified (true loading = random direction, unrelated to coords)")
    print(f"\n(sanity: cos(random true direction, SAR-implied direction) = "
          f"{abs(np.dot(ell_true2, ell_sar2)):.3f} -- should be small/moderate, not near 1)")

    print("\n############################################################")
    print("# Real data test 1: 1000 Genomes LCT region, one real LD block")
    print("############################################################")
    X_lct, pos_lct = load_lct_block(m_target=18)
    print(f"block span: {pos_lct[-1]-pos_lct[0]:.0f} bp, mean |corr| = "
          f"{np.mean(np.abs(np.corrcoef(X_lct.T))[np.triu_indices(18,1)]):.3f}")
    real_data_cv(X_lct, pos_lct, label="LCT LD block (genotype dosage treated as continuous)")

    print("\n############################################################")
    print("# Real data test 2: UK weather, one real station cluster")
    print("############################################################")
    X_wx, coords_wx = load_weather_cluster(target_idx=0, n_neighbors=9)
    print(f"cluster size: {X_wx.shape[1]} stations, mean |corr| = "
          f"{np.mean(np.abs(np.corrcoef(X_wx.T))[np.triu_indices(10,1)]):.3f}")
    real_data_cv(X_wx, coords_wx, label="UK weather station cluster (log1p precip)")
