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
import numpy as np


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
