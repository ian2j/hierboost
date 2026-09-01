"""Gaussian-copula marginal transform: map each feature onto a common Gaussian scale via
its own (empirical or parametric) marginal distribution, so that hierboost's closed-form
Gaussian block-latent machinery (factor.py's gaussian_block_factor, state_space.py's
fit_temporal_block_factor, spacetime.py's fit_spacetime_block_factor -- all built assuming
each raw feature is already roughly Gaussian, see factor.py's "ideally already standardized"
docstring note) can be applied unchanged to features whose marginals differ wildly (skewed
returns next to lognormal-ish trading volume, count data, bounded ratios) while still
enforcing one shared Gaussian correlation/latent structure between them.

This is the standard Gaussian-copula construction -- the "nonparanormal" of Liu, Lafferty &
Wasserman (2009) for the empirical/nonparametric case, or an ordinary parametric copula when
a feature's family is known (e.g. Poisson/NegBinomial counts, matching spike_slab_glm.py's
count families). Only the marginal-to-latent step changes: gaussian_block_factor,
fit_temporal_block_factor, and fit_spacetime_block_factor themselves are untouched -- they
simply receive Z (copula/Gaussian scale) in place of raw X. That is deliberate: the
correlation/factor engine already handles the "shared latent, per-feature loading,
idiosyncratic noise" part correctly (it's conjugate); the only thing that was ever wrong for
non-Gaussian features was treating raw X as if each column already lived on that scale.

Discrete marginals (counts) need one extra step: a naive empirical-CDF transform of ties
produces a non-uniform, biased copula scale (many observations share the exact same integer
value). The standard fix (Denuit & Lambert 2005) is the "continuized"/jittered CDF:
u = F(x - 1) + Unif(0, 1) * P(X = x), which IS uniform under the assumed discrete family
and only needs that family's CDF/PMF -- used here for kind="parametric" with a discrete
scipy.stats family (poisson, nbinom). The empirical/nonparametric route does not attempt
this correction (documented, not silently wrong) -- prefer kind="parametric" with the known
family when a block's raw features are counts.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats

# Winsorizing floor on the empirical CDF so norm.ppf never sees exactly 0 or 1 (which would
# map to +-inf) -- new/held-out points beyond the training range are clamped to this instead
# of extrapolated, the same "coarse but always defined" fallback style as the rest of the
# package (e.g. resolve_affinity's bandwidth fallback).
_EPS = 1e-6

_DISCRETE_FAMILIES = {"poisson", "nbinom", "binom", "geom"}


@dataclass
class MarginalTransform:
    """One column's fitted marginal, either the empirical (rank-based) CDF or a named
    scipy.stats family with its parameters -- everything `to_gaussian`/`from_gaussian`
    need to map that one column between its native scale and the shared Gaussian scale.
    """
    kind: str                          # "empirical" or a scipy.stats distribution name
    sorted_train: np.ndarray = None    # kind="empirical": sorted training values
    grid_u: np.ndarray = None          # kind="empirical": matching Hazen plotting positions
    dist: object = None                # kind=parametric: frozen scipy.stats distribution
    is_discrete: bool = False


def fit_empirical_marginal(x):
    """Rank-based marginal via the Hazen plotting-position ECDF: F_hat(x_(i)) = (i-0.5)/n
    for the i-th order statistic. Interpolated at arbitrary query points by `to_gaussian`,
    so it stays well-defined (monotone, no ties-driven flats beyond what the data itself
    has) for both training and new data queried against this same fitted reference.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    sorted_train = np.sort(x)
    grid_u = (np.arange(1, n + 1) - 0.5) / n
    return MarginalTransform(kind="empirical", sorted_train=sorted_train, grid_u=grid_u)


def fit_parametric_marginal(x, family):
    """Fit a named scipy.stats family (e.g. "gamma", "lognorm", "poisson", "nbinom") to one
    column by MLE (`.fit` for continuous families, a simple moment-matching plug-in for the
    two discrete count families used here since scipy has no generic discrete `.fit`).
    """
    x = np.asarray(x, dtype=float)
    is_discrete = family in _DISCRETE_FAMILIES
    if family == "poisson":
        dist = stats.poisson(mu=max(x.mean(), 1e-6))
    elif family == "nbinom":
        mean, var = x.mean(), x.var()
        if var <= mean:  # under-dispersed relative to Poisson -- fall back to near-Poisson NB
            var = mean * 1.01 + 1e-6
        r = max(mean ** 2 / (var - mean), 1e-3)
        p = r / (r + mean)
        dist = stats.nbinom(n=r, p=p)
    elif family == "binom":
        # n_trials fixed at the observed max (e.g. 2 for a biallelic genotype dosage
        # 0/1/2) -- moment-matching p from the mean, the standard MLE for Binomial p
        # with n known.
        n_trials = int(round(x.max()))
        p = np.clip(x.mean() / max(n_trials, 1), 1e-6, 1 - 1e-6)
        dist = stats.binom(n=n_trials, p=p)
    else:
        distn = getattr(stats, family)
        params = distn.fit(x)
        dist = distn(*params)
    return MarginalTransform(kind=family, dist=dist, is_discrete=is_discrete)


def to_gaussian(x_new, transform, rng=None):
    """Map new/held-out values through an already-fitted MarginalTransform onto the shared
    Gaussian (copula) scale. Applying the SAME fitted transform to both training and
    out-of-sample data (rather than refitting on each) is what keeps the latent scale
    consistent between them -- exactly analogous to factor.py's project_block_factor
    reusing training loadings instead of refitting PCA on new data.
    """
    x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
    if transform.kind == "empirical":
        u = np.interp(x_new, transform.sorted_train, transform.grid_u,
                      left=transform.grid_u[0], right=transform.grid_u[-1])
    elif transform.is_discrete:
        rng = np.random.default_rng() if rng is None else rng
        cdf_below = transform.dist.cdf(x_new - 1)
        pmf_here = np.clip(transform.dist.pmf(x_new), 1e-12, None)
        u = cdf_below + rng.uniform(size=x_new.shape) * pmf_here
    else:
        u = transform.dist.cdf(x_new)
    u = np.clip(u, _EPS, 1.0 - _EPS)
    return stats.norm.ppf(u)


def from_gaussian(z, transform):
    """Inverse of `to_gaussian`: map a Gaussian-scale value back to the column's native
    scale. Not needed by the block-latent fitting/prediction pipeline itself (the factor
    score lives on an arbitrary latent scale and is only ever fed into the downstream
    spike-and-slab regression, never inverted back to raw units) -- provided for
    diagnostics, e.g. reconstructing a denoised version of one original feature. For
    kind="empirical", inverts by interpolating the OTHER way (u -> x via the same stored
    order statistics); ties/repeated values in the training column collapse to their
    shared quantile, so this is not exact for heavily-tied discrete data, use the fitted
    parametric family's `.ppf` there instead.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    u = stats.norm.cdf(z)
    if transform.kind == "empirical":
        return np.interp(u, transform.grid_u, transform.sorted_train,
                          left=transform.sorted_train[0], right=transform.sorted_train[-1])
    return transform.dist.ppf(u)


def fit_block_transforms(X_block, kind="empirical", families=None):
    """Fit one MarginalTransform per column of a (n, m) block.

    kind="empirical" (default): rank-based, family-free, always available.
    kind="parametric": `families` must supply a scipy.stats family name per column
    (list/tuple of length m) -- use when you know a column's distribution (e.g. counts),
    since the parametric route handles ties/discreteness correctly via the jittered CDF
    (see module docstring), which the empirical route does not attempt.
    """
    X = np.asarray(X_block, dtype=float)
    m = X.shape[1]
    if kind == "empirical":
        return [fit_empirical_marginal(X[:, j]) for j in range(m)]
    if kind == "parametric":
        if families is None or len(families) != m:
            raise ValueError("kind='parametric' requires `families`, one scipy.stats name per column")
        return [fit_parametric_marginal(X[:, j], families[j]) for j in range(m)]
    raise ValueError(f"unknown kind={kind!r}; choose 'empirical' or 'parametric'")


def apply_block_transforms(X_block, transforms, rng=None):
    """Apply an already-fitted list of per-column MarginalTransforms (from
    `fit_block_transforms`) to a (n, m) block -- the same fitted reference for both the
    training call and every later out-of-sample call, so train/test stay on one scale.
    """
    X = np.asarray(X_block, dtype=float)
    Z = np.empty_like(X)
    for j, tr in enumerate(transforms):
        Z[:, j] = to_gaussian(X[:, j], tr, rng=rng)
    return Z


def to_gaussian_scale(X_block, kind="empirical", families=None):
    """Convenience for training data: fit + apply in one call. Returns (Z, transforms) --
    keep `transforms` and reuse `apply_block_transforms(X_new_block, transforms)` for any
    later held-out data, do not refit on new data.
    """
    transforms = fit_block_transforms(X_block, kind=kind, families=families)
    Z = apply_block_transforms(X_block, transforms)
    return Z, transforms
