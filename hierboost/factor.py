"""The continuous-observation counterpart of hierboost.latent's block-wise latent model.

Chapter 4's SAR + Binomial machinery exists because genotypes are discrete, noisy
readings (0/1/2 allele counts) of an underlying continuous trait -- that discreteness
is what forces the iterative Newton/autodiff fitting in latent.py. When the raw
features are already continuous (asset returns, sensor voltages, expression levels),
the analogous "shared block latent + per-feature loading + idiosyncratic noise" model
is a one-factor Gaussian model, whose maximum-likelihood fit is closed-form: the
leading eigenvector of the block's covariance matrix (probabilistic PCA, Tipping &
Bishop 1999). No iteration needed -- this is the same idea, just simpler because the
observation model is already conjugate.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

from .kernels import sar_weight_matrix


def gaussian_block_factor(X_block, return_variance=False):
    """One shared latent factor per block via probabilistic PCA's closed-form MLE.

    X_block: (n, m) array, ideally already standardized per column.
    Returns (factor_scores (n,), loadings (m,)) such that
    X_block ~= outer(factor_scores, loadings) + idiosyncratic noise,
    with the sign convention that the average loading is positive.

    `return_variance=True` additionally returns `z_var`: the closed-form POSTERIOR
    variance of the factor score under the model's own Gaussian generative assumptions
    (X_i,: | Z_i ~ N(Z_i * loadings, diag(obs_var)), Z_i ~ N(0, 1) prior) --
    Var(Z_i | X_i,:) = 1 / (1 + sum_j loadings_j^2 / obs_var_j), a standard conjugate
    Bayesian linear-Gaussian update. Because `loadings`/`obs_var` are population-level
    (shared across every individual in the block), this posterior variance is the SAME
    scalar for every individual -- there is no need for a per-individual value the way
    state_space.py's Kalman-filter-based z_var necessarily varies over time.

    This exists to fix a specific, confirmed problem (see project memory: calibration
    testing found ~55-60% credible-interval coverage against a 95% nominal target for
    `decorrelate="sar"` fits, the "generated regressors"/two-step-estimation problem,
    Pagan 1984 -- treating an ESTIMATED factor score as if it were the true, known Z when
    computing the downstream regression's standard error understates gamma's true
    uncertainty, regardless of how well Z itself is estimated). `z_var` is exactly the
    missing ingredient: estimator.py inflates the outcome regression's residual variance
    by `gamma_b^2 * z_var_b` per block to account for it (see
    `_HierBoostBase._compute_approx_covariance`).
    """
    X = np.asarray(X_block, dtype=float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]

    loadings = Vt[0] * S[0] / np.sqrt(n)
    factor_scores = U[:, 0] * np.sqrt(n)
    if loadings.mean() < 0:
        loadings, factor_scores = -loadings, -factor_scores
    if not return_variance:
        return factor_scores, loadings

    residual = Xc - np.outer(factor_scores, loadings)
    obs_var = np.clip(residual.var(axis=0), 1e-8, None)
    z_var = float(1.0 / (1.0 + np.sum(loadings ** 2 / obs_var)))
    return factor_scores, loadings, z_var


def project_block_factor(X_new_block, loadings, train_mean):
    """Out-of-sample counterpart of gaussian_block_factor: project new raw block
    observations onto an already-fitted set of loadings (least-squares factor score,
    holding loadings fixed) instead of jointly re-fitting both from scratch. `train_mean`
    must be the per-column mean gaussian_block_factor's caller centered on during
    training -- new data has to be centered the same way, not re-centered on its own mean.
    """
    Xc = np.asarray(X_new_block, dtype=float) - train_mean
    denom = loadings @ loadings
    return (Xc @ loadings) / denom if denom > 0 else np.zeros(Xc.shape[0])


def supervised_block_factor(X_block, y):
    """The supervised counterpart of gaussian_block_factor: instead of the direction of
    MAXIMUM VARIANCE within the block (PPCA, ignores y entirely), find the direction of
    MAXIMUM COVARIANCE with a target y. This is the textbook one-component partial least
    squares (PLS1) direction -- not a novel derivation, just the standard closed-form
    "first PLS component" (see e.g. Hastie, Tibshirani & Friedman, ESL 2nd ed. sec 3.5.2):

        w = X_block.T @ y,   w <- w / ||w||,   score = X_block @ w

    w_j is proportional to Cov(X_block[:, j], y), so w points in the direction that
    maximizes Cov(X_block @ w, y) subject to ||w||=1 -- the direction of steepest simple
    (univariate) association with y, aggregated across the block's members. It differs
    from PPCA's leading eigenvector precisely when the block's dominant internal
    covariation is NOT the part most associated with y (e.g. a block with a big common
    mode that is roughly orthogonal to y, plus a smaller mode that tracks y closely).

    X_block, y: (n, m) and (n,) arrays, both ideally already standardized (same "ideally
    already standardized" convention as gaussian_block_factor -- this function does not
    zscore for you). Returns (score (n,), w (m,)) to match gaussian_block_factor's
    (factor_scores, loadings) return shape, so callers can swap one for the other with
    minimal changes.

    Like gaussian_block_factor (and unlike it precisely BECAUSE it looks at y),
    supervised_block_factor makes block construction target-AWARE: fit it once on one
    target and the resulting w is specific to that target's covariance structure with
    the block, not a general-purpose compression of the block alone. Use
    `apply_supervised_block_factor` to score new data (e.g. a held-out period, or a
    different downstream target) against an already-fitted w, without re-estimating it
    and without ever letting the new data's own y leak into the direction.
    """
    X = np.asarray(X_block, dtype=float)
    yv = np.asarray(y, dtype=float)
    w = X.T @ yv
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    score = X @ w
    return score, w


def apply_supervised_block_factor(X_new_block, w):
    """Out-of-sample counterpart of supervised_block_factor: score new raw block data
    against an already-fitted direction `w`, with no re-estimation (mirrors
    project_block_factor's role for gaussian_block_factor). Unlike
    project_block_factor, there is no `train_mean` to re-center on -- supervised_block_factor
    never centers X_block itself (it assumes X_block, like gaussian_block_factor's input,
    is already on a standardized scale from the caller), so applying `w` to new data that
    was standardized the same way is a plain linear projection.
    """
    X = np.asarray(X_new_block, dtype=float)
    return X @ w


# ---------------------------------------------------------------------------
# sar_shrinkage_block_factor: a SAR-STRUCTURED counterpart of gaussian_block_factor,
# for when physical coordinates are available for a block's raw members and you want
# the block-latent's loading direction to actually USE that structure (unlike
# gaussian_block_factor itself, which is purely empirical and ignores coords even when
# they're supplied -- coords only ever decide block MEMBERSHIP upstream in blocks.py/
# structure.py, never the within-block loading shape).
#
# Motivation and validation are recorded in project memory ("spatial-ppca-sign-flip"):
# a first attempt (a HARD constraint pinning the loading to ell(phi) = (I -
# B(phi))^-1 @ 1, dissertation Ch4's own SAR mixing mechanism applied to a continuous/
# Gaussian observation instead of Ch4's discrete/Binomial one) won cleanly on one real
# 1000-Genomes LD block (LCT) but lost badly on three others (DARC/ACKR1, SLC24A5,
# EDAR) -- diagnosed to a real, previously-undocumented structural gap: whenever the
# SAR fit is in its stable/convergent regime (spectral radius of B below 1 -- true for
# any phi a reasonable moment-match or profile-likelihood fit would pick, since B's
# probit-kernel entries are in [0,1] and (I-B)^-1 = I + B + B^2 + ... converges there),
# ell(phi) is entrywise NON-NEGATIVE, so it cannot represent a feature that's
# anti-correlated with its block's shared factor -- an ordinary artifact of arbitrary
# reference-allele coding in genomics, and plausibly common in
# other domains too (sensor polarity, short-vs-long instruments, ...). This also
# affects hierboost.latent's actual discrete Ch4 model, which uses the identical
# (I-B)^-1 @ 1 quantity as the coefficient multiplying its own per-individual latent.
#
# The fix implemented here: replace the hard constraint with a soft, empirical-Bayes
# shrinkage prior, w_j ~ N(mu0 * ell_j(phi), tau_c2), pulling the loading toward the
# SAR shape rather than pinning it there. tau_c2 -> 0 recovers the hard-constrained
# model; tau_c2 -> infinity recovers gaussian_block_factor's own free-loading PPCA
# exactly (every M-step below reduces to the ordinary FA/PPCA update in that limit,
# see _em_shrunk_direction's docstring). mu0 and tau_c2 are both estimated from the
# data by their own closed-form M-steps (empirical Bayes), not hand-picked -- a
# feature (or a whole block) that genuinely disagrees with the SAR shape automatically
# loosens the prior instead of needing a manual override.
#
# Validated (see project memory for the full numbers): ties or beats plain
# gaussian_block_factor on every real block tried so far (4 independent 1000-Genomes
# LD blocks plus one real UK-weather station cluster), including improving on the one
# case where the hard-constrained model had already won outright. Synthetic tests
# confirm the mechanism directly: matches the hard-constrained model when the SAR
# shape is exactly correct, and degrades gracefully toward plain PPCA (rather than
# collapsing, as the hard-constrained model does) when 2 of 10 features are
# deliberately sign-flipped relative to the true shape.
# ---------------------------------------------------------------------------

@dataclass
class ShrinkageFactorResult:
    z: np.ndarray            # (n,) factor scores (posterior mean)
    loading: np.ndarray      # (m,) FREE per-feature loading -- unlike ell(phi) alone,
                             # this can be negative
    sigma2: np.ndarray       # (m,) idiosyncratic variances
    phi: float               # fitted SAR bandwidth (correlation length)
    mu0: float                # fitted prior-mean scale along ell(phi)
    tau_c2: float             # fitted empirical-Bayes prior variance -- how much the
                             # data was allowed to deviate from the SAR shape
    z_var: float              # posterior variance of z (population-level scalar, same
                             # role as gaussian_block_factor's own z_var)
    train_mean: np.ndarray    # per-column means used to center training data


def sar_loading_direction(coords, phi):
    """ell(phi) = (I - B(phi))^-1 @ 1, normalized to a unit vector -- the SAR-implied
    "effective loading" of a shared per-individual scalar factor onto each block
    member (dissertation Ch4's own mixing matrix, hierboost.kernels.sar_weight_matrix,
    applied to a vector of ones instead of a discrete/Binomial observation model).
    Entrywise non-negative whenever the SAR fit is in its stable/convergent regime
    (spectral radius of B below 1 -- true for any phi a reasonable fit would pick
    relative to the block's physical span; an implausibly large phi can push B's
    spectral radius above 1 and break this, but that phi would also be a degenerate
    fit for other reasons) -- see this module's SAR-shrinkage section docstring for
    why the non-negativity matters."""
    coords = np.asarray(coords, dtype=float)
    B = sar_weight_matrix(coords, phi)
    m = B.shape[0]
    C = np.linalg.inv(np.eye(m) - B)
    ell = C @ np.ones(m)
    norm = np.linalg.norm(ell)
    return ell / norm if norm > 0 else ell


def _rank1_loglik_total(Xc, w, sigma2):
    """Total Gaussian log-likelihood of centered data Xc under Sigma = outer(w, w) +
    diag(sigma2), via the matrix-determinant-lemma/Sherman-Morrison rank-1 update --
    avoids ever forming or inverting the m x m covariance matrix, the same low-rank-
    plus-diagonal trick hierboost.rank_utils uses throughout this codebase. Used by
    fit_phi_shrinkage's profile-likelihood search below (evaluated once per candidate
    phi)."""
    inv_sigma2 = 1.0 / sigma2
    S = np.sum(w ** 2 * inv_sigma2)
    denom = 1.0 + S
    proj = Xc @ (w * inv_sigma2)
    quad = np.sum(Xc ** 2 * inv_sigma2, axis=1) - (1.0 / denom) * proj ** 2
    logdet = np.sum(np.log(sigma2)) + np.log(denom)
    m = Xc.shape[1]
    ll_per_row = -0.5 * (m * np.log(2.0 * np.pi) + logdet + quad)
    return float(ll_per_row.sum())


def _em_shrunk_direction(Xc, ell, n_em=100, tol=1e-8, min_var_frac=0.02):
    """EM for a FREE loading w with empirical-Bayes prior w_j ~ N(mu0*ell_j, tau_c2),
    given a fixed SAR shape `ell`. Every M-step is closed form:

      w_j    = (Sjz_j/sigma2_j + mu0*ell_j/tau_c2) / (Ez2_sum/sigma2_j + 1/tau_c2)
               -- ridge regression of feature j on the factor, shrunk toward
               mu0*ell_j instead of toward 0. tau_c2 -> 0 forces w -> mu0*ell (the
               hard-constrained model); tau_c2 -> infinity drops the 1/tau_c2 terms,
               recovering ordinary PPCA/factor-analysis's own (sigma2-independent)
               free M-step exactly: w_j -> sum_i(x_ij*E[z_i]) / sum_i(E[z_i^2]).
      sigma2 = standard per-feature residual-variance M-step (unchanged in form from
               gaussian_block_factor's implicit one), floored at a small fraction of
               the feature's own raw variance -- guards a real Heywood-case failure
               mode confirmed on 1000-Genomes data (with the loading direction
               partially fixed by the prior, the sigma2 M-step can still drive one
               feature's residual variance toward 0 if `ell` happens to align
               unusually well with it), the same role hierboost.spike_slab's
               Inv-Gamma prior on sigma2 plays elsewhere in this codebase.
      mu0    = dot(ell, w) -- least-squares fit of w against the unit vector ell.
      tau_c2 = mean((w - mu0*ell)^2) -- how much the data-fitted w actually deviates
               from the SAR shape, re-estimated every iteration so a genuinely
               sign-flipped or off-shape feature automatically loosens the prior
               instead of needing a hand-picked shrinkage strength.

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


def fit_phi_shrinkage(Xc, coords, phi_bounds=(1e-2, 1e6), n_em=100, min_var_frac=0.02):
    """Profile-likelihood fit of phi: for each candidate phi, ell(phi) is
    deterministic, so run _em_shrunk_direction to its exact conditional MLE of
    (w, sigma2, mu0, tau_c2) given that shape, then evaluate the exact marginal
    log-likelihood there (_rank1_loglik_total) -- a genuine profile likelihood, not a
    cruder correlation-pattern moment-match. 1-D bounded scalar search over log(phi).
    """
    def neg_ll(log_phi):
        phi = np.exp(log_phi)
        ell = sar_loading_direction(coords, phi)
        w, sigma2, mu0, tau_c2 = _em_shrunk_direction(Xc, ell, n_em=n_em, min_var_frac=min_var_frac)
        return -_rank1_loglik_total(Xc, w, sigma2)

    lo, hi = np.log(phi_bounds[0]), np.log(phi_bounds[1])
    res = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    return float(np.exp(res.x))


def sar_shrinkage_block_factor(X_block, coords, phi=None, n_em=100, min_var_frac=0.02):
    """SAR-structured counterpart of gaussian_block_factor: one shared latent factor
    per block, with the loading direction softly shrunk toward the SAR mechanism's
    ell(phi) instead of estimated purely empirically. Requires physical (or temporal/
    any metric-space) coordinates for the block's raw members, unlike
    gaussian_block_factor. See this module's SAR-shrinkage section docstring above for
    the mechanism, motivation, and validation.

    `phi=None` (default): fit via profile likelihood (fit_phi_shrinkage). Pass a fixed
    value to skip that search (e.g. reusing a value already fit on a training fold).

    Returns a ShrinkageFactorResult.
    """
    X = np.asarray(X_block, dtype=float)
    train_mean = X.mean(axis=0)
    Xc = X - train_mean

    if phi is None:
        phi = fit_phi_shrinkage(Xc, coords, n_em=n_em, min_var_frac=min_var_frac)
    ell = sar_loading_direction(coords, phi)
    w, sigma2, mu0, tau_c2 = _em_shrunk_direction(Xc, ell, n_em=n_em, min_var_frac=min_var_frac)

    z_var = 1.0 / (1.0 + np.sum(w ** 2 / sigma2))
    z_mean = z_var * (Xc @ (w / sigma2))
    return ShrinkageFactorResult(z=z_mean, loading=w, sigma2=sigma2, phi=phi, mu0=mu0,
                                  tau_c2=tau_c2, z_var=z_var, train_mean=train_mean)


def project_shrinkage_block_factor(X_new_block, loading, sigma2, train_mean):
    """Out-of-sample counterpart of sar_shrinkage_block_factor: project new raw block
    observations onto an already-fitted (loading, sigma2), holding both fixed. Uses
    the model's own posterior-mean formula (weighted by sigma2), NOT
    project_block_factor's unweighted least-squares projection -- the two coincide
    when sigma2 is constant across features, but sar_shrinkage_block_factor's sigma2
    is typically more heterogeneous (a feature the prior mostly overrode keeps a
    larger sigma2), so weighting matters more here, and using the same formula the
    training fit itself uses keeps train- and test-time scoring consistent.
    """
    Xc = np.asarray(X_new_block, dtype=float) - train_mean
    z_var = 1.0 / (1.0 + np.sum(loading ** 2 / sigma2))
    return z_var * (Xc @ (loading / sigma2))
