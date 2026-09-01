"""Summary-statistics-only counterpart of hierboost.spike_slab_gaussian: fits the same
hierarchical spike-and-slab model, but from GWAS summary statistics (a per-SNP marginal
effect estimate `bhat` and sample size `n`) plus an LD reference panel (a SNP-by-SNP
correlation matrix `R`) -- no individual-level (X, y) required at all. This is the
sufficient-statistics regression idea behind Zhu & Stephens (2017) and SuSiE-RSS
(Zou, Carbonetto, Wang & Stephens 2022; `zou2022susierss` in paper/refs.bib), adapted
to this project's own EM spike-and-slab rather than SuSiE's IBSS.

The identity this relies on
----------------------------
For y = X*b + e with X's p columns standardized (mean 0, X'X/n ~ R, a correlation
matrix) and y standardized (mean 0, Var(y) ~= y_var, 1.0 if y is itself standardized),
the normal-equations sufficient statistics any EM/Gibbs step needs are exactly X'X and
X'y -- never the raw n x p data matrix. From GWAS summary stats alone:

    X'X / n ~= R      (from a REFERENCE panel -- see caveat below)
    X'y / n ~= bhat   (bhat_j = z_j * se_j, the per-SNP marginal/univariate effect;
                       or the file's own reported beta directly, if it is already on a
                       standardized per-allele scale -- check the file's documented
                       units, do not assume)

so R*b = bhat replaces X'X*b = X'y as the normal equations. This module reconstructs
X'X ~= n*R and X'y ~= n*bhat and plugs them into a parallel implementation of
spike_slab_gaussian.fit_em_gaussian's exact M-step math (see `_augment_R_bhat` for how
the intercept is handled, and the module-level docstring section "Why a parallel
implementation, not a thin adapter" below for why this isn't literally a call-through).

Known limitation, stated plainly rather than hidden: `R` here comes from a REFERENCE
panel (e.g. 1000 Genomes), not from the original GWAS's own individual-level genotypes,
which are never released. Reference-panel-vs-original-sample LD mismatch is a real,
general limitation of every RSS-style method (SuSiE-RSS included), not specific to this
implementation -- see `fit_em_sumstats`'s docstring and this project's real-data
validation script for where it can visibly distort results (e.g. spurious "signal" at a
SNP that is well-tagged in the reference panel but was differently tagged in the
original GWAS sample).

Why a parallel implementation, not a thin adapter
--------------------------------------------------
spike_slab_gaussian.fit_em_gaussian does not cache X'X/X'y as persistent sufficient
statistics internally -- it recomputes `X.T @ (w[:, None] * X)` and `X.T @ (y / sigma_y2)`
from the raw arrays every EM iteration. Because the Gaussian spike-and-slab's
per-observation weight `w = 1/sigma_y2` is homoskedastic (identical for every
individual, not data-dependent the way a GLM's IRLS weight would be), those raw-data
recomputations are ALGEBRAICALLY IDENTICAL, iteration by iteration, to
`(n/sigma_y2)*R_aug` and `(n/sigma_y2)*bhat_aug` computed once from sufficient
statistics that never change -- there is no `X`/`y`-shaped intermediate anywhere in the
true math that a thin wrapper could intercept. So this module reimplements the same
M-step in sufficient-statistics form (functions below mirror fit_em_gaussian's control
flow line-for-line) rather than bolting an adapter onto the raw-data function.

Design choice on hierboost.factor.gaussian_block_factor and blocking
----------------------------------------------------------------------
estimator.py's raw-data pipeline runs gaussian_block_factor to collapse each block of
*correlated raw X columns* into a single per-INDIVIDUAL latent score before regression
-- a decorrelation/dimensionality-reduction step that only makes sense when you have
per-individual observations to project onto. A pure summary-stats pipeline has no such
observations: only a p x p correlation structure and p marginal effect estimates. There
is also no need to invent a substitute -- R already IS the full correlation structure,
and the M-step below (via `TruncatedDesignFromR`/exact solve) already conditions on the
WHOLE active R exactly, so there's no separate decorrelation step to perform in the
first place; projecting blocks of R onto a leading eigenvector first, the way
gaussian_block_factor projects blocks of X, would only throw away information the exact
solve already uses, for no compensating benefit. So block_id here plays a different,
narrower role than in the raw-data pipeline: (1) `hierboost.structure.determine_blocks`
still supplies a sensible default LD-block partition when the caller doesn't have one,
using `threshold_blocks_graph` on the LD distance 1 - |R| (exactly the "group by actual
observed statistical relationship" mode structure.py already documents for
non-coordinate graphs); (2) `summarize_by_block` aggregates the per-SNP EM output
(theta_hat, beta) into a per-block report -- "which LD blocks are implicated" -- which
is the natural "block-level effect aggregation" fine-mapping methods report (a credible
block/region), rather than a single collapsed block score.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import invgamma

from .rank_utils import woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample
from .spike_slab import theta_conditional, _precision_from_theta
from .blocks import block_membership_lists
from .structure import determine_blocks


def _augment_R_bhat(R, bhat):
    """Build the (p+1) x (p+1) augmented LD matrix and (p+1) effect vector that make the
    sufficient-statistics M-step below produce the SAME updates fit_em_gaussian's raw-data
    M-step would, under fit_em_gaussian's own convention: X has an intercept column of
    ones prepended to p already-centered, unit-variance predictor columns, and y is
    itself centered to mean 0. Under that convention (exactly): X'X[0,0] = n (sum of
    1^2 over n rows), X'X[0, 1:] = 0 (an intercept column has zero sample correlation
    with any mean-centered column, by construction), and X'y[0] = n*mean(y) = 0. So the
    augmented R/bhat just pad in a unit variance / zero-correlation / zero-effect row and
    column for the intercept -- not an approximation, an exact consequence of the same
    standardization fit_em_gaussian already assumes of its own X, y.
    """
    R = np.asarray(R, dtype=float)
    bhat = np.asarray(bhat, dtype=float)
    p = R.shape[0]
    if R.shape != (p, p):
        raise ValueError(f"R must be square, got {R.shape}")
    if bhat.shape != (p,):
        raise ValueError(f"bhat must have shape ({p},) to match R, got {bhat.shape}")
    R_aug = np.zeros((p + 1, p + 1))
    R_aug[0, 0] = 1.0
    R_aug[1:, 1:] = R
    bhat_aug = np.concatenate([[0.0], bhat])
    return R_aug, bhat_aug


class TruncatedDesignFromR:
    """Rank-l eigendecomposition surrogate of the augmented LD matrix R_aug, playing the
    same role rank_utils.TruncatedDesign plays for a raw design matrix: produces an S
    (l x p1) with S^T S ~= (weight) * n * R_aug, that can then be fed UNCHANGED into
    rank_utils.woodbury_solve / woodbury_mean_and_sample -- genuine reuse of the existing
    Woodbury linear algebra, not a reimplementation of it.

    Unlike TruncatedDesign.weighted_factor (which takes a length-n per-observation weight
    vector, since a GLM's IRLS weight can vary by row), the Gaussian spike-and-slab's
    weight `w = 1/sigma_y2` is a single homoskedastic scalar -- there is no "row" axis at
    all in the summary-stats world, so `weighted_factor` here takes a plain scalar.

    IMPORTANT consistency requirement (found empirically fitting a real, severely
    rank-deficient LD matrix -- see genomics_ldlr_sumstats.py's investigation of an
    initial LinAlgError/divergence): whenever a caller truncates R_aug's rank, it MUST
    also project bhat_aug onto this same truncated eigenbasis (`project`) before using
    it anywhere else in the M-step (the RHS `c` AND the ssr/PPL diagnostic), not just in
    the curvature. If bhat_aug is left untouched while R_aug's curvature is truncated,
    any component of bhat_aug lying in the discarded near-null eigenspace of R_aug (real
    GWAS bhat and a real reference panel's R never lie in perfectly the same subspace --
    that IS the reference-panel-vs-original-sample mismatch this module documents
    elsewhere) gets "fit" by the M-step with essentially zero curvature to restrain it,
    driving beta, and then sigma_g2/sigma_y2 through the usual EM feedback, to blow up --
    confirmed exactly this failure mode on the real LDLR-locus data (ssr collapsed to
    the floor at 0.0, sigma_y2 collapsed toward 0, beta diverged past 1e15 within three
    EM iterations) before `reconstruct_R`/`project` were added and used consistently by
    fit_em_sumstats/em_filter_sumstats below.
    """

    def __init__(self, R_aug, n, rank):
        eigval, eigvec = np.linalg.eigh(R_aug)  # R_aug is symmetric PSD (up to float noise)
        order = np.argsort(eigval)[::-1]
        eigval = np.clip(eigval[order], 0.0, None)
        eigvec = eigvec[:, order]
        rank = min(rank, eigval.shape[0])
        self.V = eigvec[:, :rank]          # p1 x l
        self.eigval = eigval[:rank]        # l
        self.n = n
        self.rank = rank

    def relative_frobenius_error(self, R_aug):
        approx = (self.V * self.eigval) @ self.V.T
        return np.linalg.norm(R_aug - approx) / np.linalg.norm(R_aug)

    def reconstruct_R(self):
        """The rank-l reconstruction V @ diag(eigval) @ V^T -- the EFFECTIVE R_aug this
        truncated design actually corresponds to, for use as ssr_sumstats' quadratic-form
        matrix so the M-step's solve and its own diagnostics agree (see class docstring)."""
        return (self.V * self.eigval) @ self.V.T

    def project(self, vec):
        """Project `vec` onto the kept eigenbasis -- MUST be applied to bhat_aug wherever
        it's used alongside a truncated R_aug (see class docstring's consistency
        requirement)."""
        return self.V @ (self.V.T @ vec)

    def weighted_factor(self, w):
        scale = np.sqrt(max(w, 0.0) * self.n * self.eigval)
        return scale[:, None] * self.V.T   # l x p1


def effective_rank(R, rel_tol=1e-6):
    """Number of eigenvalues of R at least `rel_tol` times its largest eigenvalue --
    a practical numerical rank for a REAL sample LD matrix, which is routinely
    near-singular (not just theoretically singular when p > n_reference, but often
    numerically so well before that: dense real genotype data packs many SNPs into a
    handful of haplotype blocks, so several columns of R can be near-exact linear
    combinations of others). Intended as the `rank` argument to fit_em_sumstats /
    em_filter_sumstats whenever the exact dense solve raises a LinAlgError (or would be
    poorly conditioned even if it technically succeeds) -- the near-zero eigenvalue
    directions this excludes carry no independent information anyway (they correspond
    to exact/near-exact allelic redundancy, not a real signal being discarded), so
    truncating to them is the correct fix, not an approximation of convenience.
    """
    eigval = np.linalg.eigvalsh(np.asarray(R, dtype=float))
    thresh = rel_tol * max(eigval.max(), 0.0)
    return int(np.sum(eigval > thresh))


def ssr_sumstats(R_aug, bhat_aug, beta, n, y_var=1.0):
    """Sufficient-statistics reconstruction of sum((y - X @ beta)**2), the exact quantity
    fit_em_gaussian computes from raw (X, y): expand y'y - 2*beta'X'y + beta'X'X*beta and
    substitute X'X ~= n*R_aug, X'y ~= n*bhat_aug, y'y ~= n*y_var (the standardized-y
    convention documented at module level; pass the file's/simulation's own sample
    variance of y as y_var if it is not exactly 1). Clipped at 0 to absorb floating-point
    noise when beta is close to the OLS solution.
    """
    ssr = n * (y_var - 2.0 * beta @ bhat_aug + beta @ (R_aug @ beta))
    return max(float(ssr), 0.0)


@dataclass
class EMResultSumstats:
    beta: np.ndarray
    sigma_g2: float
    sigma_y2: float
    theta_hat: np.ndarray
    n_iter: int


def fit_em_sumstats(R, bhat, n, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0,
                     y_var=1.0, rank=None, beta_init=None, sigma_g2_init=1.0,
                     sigma_y2_init=1.0, max_iter=200, tol=1e-8):
    """Sufficient-statistics-only EM for the Gaussian spike-and-slab: takes (R, bhat, n)
    instead of (X, y). Mirrors spike_slab_gaussian.fit_em_gaussian's control flow and
    M-step formulas exactly, term-for-term substituting X'X -> n*R_aug, X'y -> n*bhat_aug,
    sum((y-Xb)^2) -> ssr_sumstats(...) (see module docstring for why this is exact, not
    approximate, given standardized X/y and a known y_var).

    R: (p, p) LD correlation matrix from a REFERENCE panel (see module docstring's
       reference-panel-vs-original-sample caveat -- this is a real limitation of every
       RSS-style method, not specific to this implementation).
    bhat: (p,) per-SNP standardized marginal effect estimate (z_j * se_j, or the file's
       own beta if already on that scale -- check the file's documented units).
    n: GWAS sample size (scalar).
    wr: (p,) per-SNP relevance/affinity weight, same role as spike_slab_gaussian's wr.
    rank: if given, solve through TruncatedDesignFromR's rank-l Woodbury path instead of
       an exact dense solve -- REQUIRED whenever R is severely rank-deficient (routine
       for real, densely-sampled LD: see hierboost.sumstats.effective_rank), since the
       exact solve can raise LinAlgError there, or silently be badly conditioned even
       when it doesn't. Pass the string "auto" to have this function call
       effective_rank(R) itself -- important inside em_filter_sumstats's outer loop,
       where R shrinks every round: a rank fixed once from the ORIGINAL p can become
       >= the current (shrunk) p1 after enough filtering, silently turning back into an
       unregularized exact solve for a submatrix that is often just as collinear as the
       original (real LD blocks don't get better-conditioned just because you removed a
       few unrelated SNPs elsewhere) -- confirmed causing exactly this renewed
       divergence on real LDLR-locus data before "auto" was added. bhat_aug is
       projected onto the same truncated eigenbasis as R_aug (see TruncatedDesignFromR's
       docstring) so the M-step and its own ssr/PPL diagnostics stay consistent --
       skipping that projection is what caused a real, confirmed EM divergence on real
       LDLR-locus GWAS data during development (ssr collapsed to 0, sigma_y2 collapsed
       toward 0, beta diverged past 1e15 in 3 iterations) before this was fixed.
    """
    R_aug, bhat_aug = _augment_R_bhat(R, bhat)
    p1 = R_aug.shape[0]
    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma_g2 = sigma_g2_init
    sigma_y2 = sigma_y2_init
    if rank == "auto":
        rank = effective_rank(R) + 1
    design = TruncatedDesignFromR(R_aug, n, rank) if rank is not None else None
    if design is not None:
        R_eff, bhat_eff = design.reconstruct_R(), design.project(bhat_aug)
    else:
        R_eff, bhat_eff = R_aug, bhat_aug

    it = 0
    for it in range(1, max_iter + 1):
        th = theta_conditional(beta[1:], sigma_g2, kappa, xi0, xi1, wr)
        precision = _precision_from_theta(th, kappa)
        sigma_g2_new = (0.5 * np.sum(beta ** 2 * precision) + lam) / (p1 / 2.0 + nu + 1.0)
        sigma_diag = sigma_g2_new / precision

        ssr = ssr_sumstats(R_eff, bhat_eff, beta, n, y_var)
        sigma_y2_new = (0.5 * ssr + lam_y) / (n / 2.0 + nu_y + 1.0)

        w = 1.0 / sigma_y2_new
        c = (n * w) * bhat_eff
        if design is not None:
            S = design.weighted_factor(w)
            beta_new = woodbury_solve(S, sigma_diag, c)
        else:
            XtWX = (n * w) * R_aug
            beta_new = np.linalg.solve(XtWX + np.diag(1.0 / sigma_diag), c)

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma_g2, sigma_y2 = beta_new, sigma_g2_new, sigma_y2_new
        if delta < tol:
            break

    th = theta_conditional(beta[1:], sigma_g2, kappa, xi0, xi1, wr)
    return EMResultSumstats(beta=beta, sigma_g2=sigma_g2, sigma_y2=sigma_y2,
                             theta_hat=th, n_iter=it)


def ppl_sumstats(R, bhat, beta, n, sigma_y2, y_var=1.0):
    """Sufficient-statistics counterpart of spike_slab_gaussian.ppl_gaussian: posterior
    predictive loss under squared error, reconstructed from (R, bhat, n) instead of
    (y, mu). Matches ppl_gaussian's value exactly at the population level: ppl_gaussian
    computes sum((y-mu)**2) + n*sigma_y2, and sum((y-mu)**2) here is exactly
    ssr_sumstats(...) once R/bhat/n substitute for the raw data's X'X/X'y/y'y."""
    R_aug, bhat_aug = _augment_R_bhat(R, bhat)
    ssr = ssr_sumstats(R_aug, bhat_aug, beta, n, y_var)
    return ssr + n * sigma_y2


@dataclass
class FilterStepSumstats:
    step: int
    n_features: int
    ppl: float
    rppl: float
    retained_idx: np.ndarray
    retained_blocks: np.ndarray
    beta: np.ndarray
    sigma_g2: float
    sigma_y2: float
    theta_hat: np.ndarray


@dataclass
class EMFilterResultSumstats:
    history: list = field(default_factory=list)
    best: "FilterStepSumstats" = None


def em_filter_sumstats(R, bhat, n, wr, xi0, xi1, kappa, nu, lam, block_id=None,
                        nu_y=1.0, lam_y=1.0, y_var=1.0, filter_frac=0.25, min_features=10,
                        max_outer=200, rank=None, patience=3, verbose=False):
    """EM-filtering pipeline over summary statistics -- the sufficient-statistics
    counterpart of spike_slab_gaussian.em_filter_gaussian: same distilled-sensing outer
    loop (fit, then drop the lowest-theta_hat fraction of the still-active SNPs, repeat
    until a PPL-based patience criterion trips), just calling fit_em_sumstats and
    ssr_sumstats/ppl_sumstats in place of fit_em_gaussian/ppl_gaussian.

    block_id: (p,) integer LD-block label per SNP, aligned to R/bhat's original SNP
    order. If None, a default partition is derived from R itself via
    hierboost.structure.determine_blocks(coords=1-|R|, method="threshold_graph") -- LD
    block structure is exactly the kind of "graph distance" that function already
    supports (see module docstring for why block_id is used only for reporting here,
    not for a block-latent-factor decorrelation step the way the raw-data pipeline uses
    it). Each FilterStepSumstats additionally reports `retained_blocks`: the sorted set
    of LD blocks that still have >=1 retained SNP at that step -- the natural
    "block-level effect aggregation" fine-mapping report (see `summarize_by_block` for
    a fuller per-block breakdown of the best step).
    """
    p = bhat.shape[0]
    if block_id is None:
        R_arr = np.asarray(R, dtype=float)
        dist = 1.0 - np.abs(R_arr)
        np.fill_diagonal(dist, 0.0)
        block_id = determine_blocks(coords=dist, method="threshold_graph")
    block_id = np.asarray(block_id)

    idx = np.arange(p)
    R_cur = np.asarray(R, dtype=float)
    bhat_cur = np.asarray(bhat, dtype=float)
    wr_cur = wr
    beta = None
    sigma_g2 = 1.0
    sigma_y2 = float(y_var)

    ppl_null = ppl_sumstats(R_cur, bhat_cur, np.zeros(p + 1), n, sigma_y2, y_var)

    history, best, bad_streak = [], None, 0
    for step in range(max_outer):
        res = fit_em_sumstats(R_cur, bhat_cur, n, wr_cur, xi0, xi1, kappa, nu, lam,
                               nu_y, lam_y, y_var=y_var, rank=rank, beta_init=beta,
                               sigma_g2_init=sigma_g2, sigma_y2_init=sigma_y2)
        ppl_val = ppl_sumstats(R_cur, bhat_cur, res.beta, n, res.sigma_y2, y_var)
        retained_blocks = np.array(sorted(set(block_id[idx].tolist())))
        record = FilterStepSumstats(step=step, n_features=idx.shape[0], ppl=ppl_val,
                                     rppl=ppl_val / ppl_null, retained_idx=idx.copy(),
                                     retained_blocks=retained_blocks, beta=res.beta,
                                     sigma_g2=res.sigma_g2, sigma_y2=res.sigma_y2,
                                     theta_hat=res.theta_hat)
        history.append(record)
        if verbose:
            print(f"step {step:3d}  p={idx.shape[0]:6d}  PPL={ppl_val:.5f}  rPPL={record.rppl:.4f}  "
                  f"n_blocks={retained_blocks.shape[0]}")

        if best is None or ppl_val < best.ppl:
            best, bad_streak = record, 0
        else:
            bad_streak += 1
        if bad_streak >= patience or idx.shape[0] <= min_features:
            break

        n_remove = max(1, int(np.ceil(filter_frac * idx.shape[0])))
        if idx.shape[0] - n_remove < 1:
            break
        order = np.argsort(res.theta_hat)
        keep_mask = np.ones(idx.shape[0], dtype=bool)
        keep_mask[order[:n_remove]] = False

        idx = idx[keep_mask]
        R_cur = np.asarray(R)[np.ix_(idx, idx)]
        bhat_cur = np.asarray(bhat)[idx]
        wr_cur = wr[idx]
        beta = np.concatenate([[res.beta[0]], res.beta[1:][keep_mask]])
        sigma_g2, sigma_y2 = res.sigma_g2, res.sigma_y2

    return EMFilterResultSumstats(history=history, best=best)


def summarize_by_block(best, block_id):
    """Aggregate a FilterStepSumstats' per-SNP output into a per-LD-block report: for
    each retained block, the SNP (in the block's ORIGINAL/full-p indexing) with the
    highest theta_hat, that SNP's theta_hat and beta, and the block's mean |beta| across
    its retained members. This is the "block-level effect aggregation" this module's
    docstring lands on in place of a block-latent-factor score: with no per-individual
    data to project a block onto, the meaningful block-level summary is which SNPs in
    the block the model implicates and how strongly, not a single collapsed value.

    best: a FilterStepSumstats (typically `.best` from an EMFilterResultSumstats).
    block_id: (p,) block label per SNP in the ORIGINAL (pre-filtering) SNP order --
    the same array passed to (or returned as a default by) em_filter_sumstats.
    Returns a dict {block_label: {"top_snp": original_idx, "top_theta": float,
    "top_beta": float, "mean_abs_beta": float, "n_retained_snps": int}}.
    """
    block_id = np.asarray(block_id)
    retained_idx = best.retained_idx           # original-SNP indices, aligned to best.theta_hat
    theta_hat = best.theta_hat
    beta = best.beta[1:]                        # drop the intercept slot
    out = {}
    for b in sorted(set(block_id[retained_idx].tolist())):
        local = np.where(block_id[retained_idx] == b)[0]
        top_local = local[np.argmax(theta_hat[local])]
        out[int(b)] = dict(
            top_snp=int(retained_idx[top_local]),
            top_theta=float(theta_hat[top_local]),
            top_beta=float(beta[top_local]),
            mean_abs_beta=float(np.mean(np.abs(beta[local]))),
            n_retained_snps=int(local.shape[0]),
        )
    return out


@dataclass
class GibbsResultSumstats:
    beta: np.ndarray
    theta: np.ndarray
    sigma_g2: np.ndarray
    sigma_y2: np.ndarray
    pi_hat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pi_hat = self.theta.mean(axis=0)


def gibbs_sampler_sumstats(R, bhat, n, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0,
                            y_var=1.0, n_samples=2000, burn_in=500, thin=1, rank=None,
                            beta_init=None, sigma_g2_init=1.0, sigma_y2_init=1.0,
                            theta_init=None, seed=None):
    """Sufficient-statistics counterpart of spike_slab_gaussian.gibbs_sampler_gaussian --
    same fully-conjugate draws, X'X/X'y/y'y substituted by n*R_aug/n*bhat_aug/n*y_var
    exactly as in fit_em_sumstats. `rank="auto"` is supported the same way (see
    fit_em_sumstats's docstring)."""
    rng = np.random.default_rng(seed)
    R_aug, bhat_aug = _augment_R_bhat(R, bhat)
    p1 = R_aug.shape[0]
    p = p1 - 1
    if rank == "auto":
        rank = effective_rank(R) + 1

    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma_g2 = sigma_g2_init
    sigma_y2 = sigma_y2_init
    theta = np.ones(p, dtype=bool) if theta_init is None else theta_init.copy()

    design = TruncatedDesignFromR(R_aug, n, rank) if rank is not None else None
    if design is not None:
        R_eff, bhat_eff = design.reconstruct_R(), design.project(bhat_aug)
    else:
        R_eff, bhat_eff = R_aug, bhat_aug

    beta_samples = np.empty((n_samples, p1))
    theta_samples = np.empty((n_samples, p), dtype=bool)
    sigma_g2_samples = np.empty(n_samples)
    sigma_y2_samples = np.empty(n_samples)

    total = burn_in + n_samples * thin
    kept = 0
    for it in range(total):
        precision = _precision_from_theta(theta.astype(float), kappa)
        shape = nu + p1 / 2.0
        scale = lam + 0.5 * np.sum(beta ** 2 * precision)
        sigma_g2 = invgamma.rvs(shape, scale=scale, random_state=rng)

        th_prob = theta_conditional(beta[1:], sigma_g2, kappa, xi0, xi1, wr)
        theta = rng.random(p) < th_prob

        ssr = ssr_sumstats(R_eff, bhat_eff, beta, n, y_var)
        sigma_y2 = invgamma.rvs(nu_y + n / 2.0, scale=lam_y + 0.5 * ssr, random_state=rng)

        sigma_diag = np.empty(p1)
        sigma_diag[0] = sigma_g2 * kappa
        sigma_diag[1:] = sigma_g2 * np.where(theta, kappa, 1.0)

        w = 1.0 / sigma_y2
        c = (n * w) * bhat_eff
        if design is not None:
            S = design.weighted_factor(w)
            beta, _ = woodbury_mean_and_sample(S, sigma_diag, c, rng)
        else:
            XtWX = (n * w) * R_aug
            beta, _ = exact_mean_and_sample(XtWX, sigma_diag, c, rng)

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_samples[kept] = beta
            theta_samples[kept] = theta
            sigma_g2_samples[kept] = sigma_g2
            sigma_y2_samples[kept] = sigma_y2
            kept += 1

    return GibbsResultSumstats(beta=beta_samples, theta=theta_samples,
                                sigma_g2=sigma_g2_samples, sigma_y2=sigma_y2_samples)
