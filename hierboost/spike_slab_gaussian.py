"""Gaussian-response counterpart of hierboost.spike_slab, for continuous outcomes
(e.g. asset returns) instead of the Bernoulli/logistic response spike_slab.py assumes.

Conjugate throughout: no Polya-Gamma augmentation, no IRLS linearization. Given the
current variance parameters, beta's conditional posterior mode/mean has a direct
closed form (ordinary Bayesian ridge regression), so the EM and Gibbs steps below
are simpler and faster than the logistic case -- and they reuse rank_utils.py
unchanged, since it was already written in terms of a generic per-observation
weight and a generic response-transformed target, not anything logistic-specific.
They also reuse spike_slab.py's theta_conditional/centroid_estimate/embfdr/
select_kappa_by_embfdr as-is, since those only ever operate on beta/theta_hat, never
on X or y directly.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import invgamma

from .rank_utils import TruncatedDesign, woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample
from .spike_slab import theta_conditional, theta_conditional_pmom, _precision_from_theta


def ppl_gaussian(y, mu, sigma_y2):
    """Posterior predictive loss under squared error for a Gaussian response."""
    n = y.shape[0]
    return float(np.sum((y - mu) ** 2) + n * sigma_y2)


@dataclass
class EMResultGaussian:
    beta: np.ndarray
    sigma_g2: float
    sigma_y2: float
    theta_hat: np.ndarray
    mu: np.ndarray
    n_iter: int


def fit_em_gaussian(X, y, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0, rank=None,
                     beta_init=None, sigma_g2_init=1.0, sigma_y2_init=1.0,
                     max_iter=200, tol=1e-8):
    """EM algorithm for the Gaussian-response spike-and-slab. Because the response is
    already linear-Gaussian, the beta M-step is an exact conditional-maximum, not an
    IRLS linearization: beta_new = (X^T X / sigma_y2 + Sigma_prior^-1)^-1 X^T y / sigma_y2.
    """
    n, p1 = X.shape
    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma_g2 = sigma_g2_init
    sigma_y2 = sigma_y2_init
    design = TruncatedDesign(X, rank) if rank is not None else None

    for it in range(1, max_iter + 1):
        th = theta_conditional(beta[1:], sigma_g2, kappa, xi0, xi1, wr)
        precision = _precision_from_theta(th, kappa)
        sigma_g2_new = (0.5 * np.sum(beta ** 2 * precision) + lam) / (p1 / 2.0 + nu + 1.0)
        sigma_diag = sigma_g2_new / precision

        ssr = np.sum((y - X @ beta) ** 2)
        sigma_y2_new = (0.5 * ssr + lam_y) / (n / 2.0 + nu_y + 1.0)

        w = np.full(n, 1.0 / sigma_y2_new)
        c = X.T @ (y / sigma_y2_new)
        if design is not None:
            S = design.weighted_factor(w)
            beta_new = woodbury_solve(S, sigma_diag, c)
        else:
            XtWX = X.T @ (w[:, None] * X)
            beta_new = np.linalg.solve(XtWX + np.diag(1.0 / sigma_diag), c)

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma_g2, sigma_y2 = beta_new, sigma_g2_new, sigma_y2_new
        if delta < tol:
            break

    th = theta_conditional(beta[1:], sigma_g2, kappa, xi0, xi1, wr)
    mu = X @ beta
    return EMResultGaussian(beta=beta, sigma_g2=sigma_g2, sigma_y2=sigma_y2,
                             theta_hat=th, mu=mu, n_iter=it)


def fit_em_gaussian_pmom(X, y, wr, xi0, xi1, tau, nu, lam, nu_y=1.0, lam_y=1.0, rank=None,
                          beta_init=None, sigma_g2_init=1.0, sigma_y2_init=1.0,
                          max_iter=200, tol=1e-8, max_step=5.0, eps=1e-6,
                          sigma_g2_fixed=False):
    """EM for the Gaussian-response spike-and-slab with a non-local (product-MOM) slab
    prior in place of the local Gaussian slab -- see theta_conditional_pmom's
    docstring for the E-step derivation, and project memory 2026-09-18/19 for the
    full derivation this implements.

    Unlike fit_em_gaussian's exact conjugate M-step, the pMOM slab's log(beta_j^2)
    term is not quadratic in beta, so it has no closed-form M-step the way the local
    prior does. This uses a damped Newton/MM step instead: a local quadratic (second-
    order Taylor) expansion of log(beta_j^2) around the current iterate a_j = beta_j,

        log(beta_j^2) ~= log(a_j^2) + (4/a_j)*beta_j - beta_j^2/a_j^2   (+ const)

    which turns the M-step back into a closed-form linear solve each iteration --
    same "linearize the non-quadratic term, then exact-solve" pattern as fit_em's
    logistic IRLS branch, generalizing fit_em_gaussian's exact line to:

        beta_new = (X^T X / sigma_y2 + Lambda)^-1 (X^T y / sigma_y2 + b)
        Lambda = diag(precision_j / sigma_g2 + 2*theta_hat_j / a_j^2)   (features only)
        b_j = 4 * theta_hat_j / a_j                                     (features only)

    where `precision_j / sigma_g2` is the same local-slab-style ridge term as before
    (tau standing in for kappa), and the two new terms are the linearized pMOM
    contribution. This step is NOT guaranteed to monotonically increase the
    log-posterior every iteration (a known, accepted limitation in the non-local-
    prior computational literature -- e.g. Shin/Bhattacharya/Johnson resort to
    stochastic search rather than an exact EM for the same underlying reason);
    convergence is only checked via the usual relative-beta tolerance.

    Two safeguards this needs that fit_em_gaussian does not, both because the update
    divides by the CURRENT iterate a_j:
    - `beta_init` defaults to a quick ridge solve, NOT all-zeros -- beta=0 exactly
      makes 1/a_j degenerate on the very first iteration.
    - `max_step` damps each raw step (same mechanism/rationale as the max_step
      guards in hierboost.latent's Newton updates, added there after an undamped
      Newton step was found to diverge -- see that module's docstring); `eps` floors
      |a_j| away from exactly 0 for the same reason.

    Derived fix (2026-09-21): the sigma_g2 update below now accounts for that missing
    term. Re-deriving the complete conditional posterior of sigma_g2 given (beta,
    theta) the way fit_em_gaussian's formula was derived, but using the pMOM slab's
    actual density for included features -- pi_1(beta) ~ sg2^(-3/2) *
    exp(-beta^2/(2*tau*sg2)), vs the local slab's sg2^(-1/2) for the SAME included
    features (see theta_conditional_pmom's docstring) -- each included feature
    (theta_j=1) contributes one extra unit of -log(sigma_g2) beyond what the local-
    slab-derived formula assumes. Taking the E-step's expectation over the soft
    theta_hat, the inverse-gamma shape parameter picks up an extra sum(theta_hat)
    term:

        sigma_g2_new = (0.5*sum(beta^2*precision) + lam) / (p1/2 + nu + 1 + sum(theta_hat))

    vs. fit_em_gaussian's (p1/2 + nu + 1) denominator. The intercept and excluded
    features are untouched (still local-slab, sg2^(-1/2)); only the extra
    sum(theta_hat) term is new, and it appears ONLY in the denominator -- the
    numerator, 0.5*sum(beta^2*precision), comes from the exponent (already correct,
    unchanged from fit_em_gaussian's derivation). In principle this should push back
    against the inflation mechanism the sigma_g2_fixed safeguard below was built to
    work around (confidently-included features pulling MORE denominator mass, not
    less) -- validated 2026-09-21 against the 30-replicate PPCA-compressed-block
    synthetic (tests/synthetic_pmom_validation.py): a real but SMALL effect,
    pmom_threshold power 0.300 -> 0.317 at unchanged perfect precision, no regression.

    **Tested, does NOT rescue the small-enriched-pool collapse case the sigma_g2_fixed
    safeguard exists for** (2026-09-21): on a targeted stress test matching that
    scenario (8 candidates, 5 truly causal, N=1000) the fix moved sigma_g2 by <1%
    (0.9049 -> 0.8998) and theta_hat stayed collapsed near 0 for every true positive,
    same as without the fix. Diagnosis: the correction term is sum(theta_hat) itself
    -- so in exactly the failure state this was meant to counter (theta_hat already
    collapsing toward 0), the correction vanishes right along with it. Self-
    reinforcing, not self-correcting: the fix only has teeth once theta_hat is
    already away from 0, which is precisely what fails in this scenario. Keep
    sigma_g2_fixed for small, pre-enriched refits; this fix is a general correctness
    improvement to the M-step, not a substitute for it.

    `sigma_g2_fixed=False` (default): sigma_g2 re-estimated each iteration using the
    corrected formula above. `sigma_g2_fixed=True`: sigma_g2 is held at
    `sigma_g2_init` for the whole fit, never re-estimated. Originally added because
    the (uncorrected) M-step's empirical-Bayes sigma_g2 had no anchor when the
    candidate set is pre-enriched for true positives (e.g. refitting on a stability-
    selection-admitted set, see hierboost/stability.py and the 2026-09-19
    investigation that found it): with few or no null features left to shrink sigma_g2
    down, it inflated to match whatever beta magnitudes were present, which then made
    those SAME betas look unremarkable to theta_conditional_pmom and collapsed
    theta_hat toward 0. As confirmed above, the corrected formula does NOT rescue
    this case (the correction term collapses along with theta_hat) -- keep
    sigma_g2_fixed=True for small, pre-enriched refits (post-stability-selection);
    `sigma_g2_init` should come from a fit over a larger, null-rich candidate pool
    (e.g. the pre-admission full fit's `res.sigma_g2`) in that case.
    """
    n, p1 = X.shape
    if beta_init is None:
        ridge_lam = 1.0
        beta = np.linalg.solve(X.T @ X + ridge_lam * np.eye(p1), X.T @ y)
    else:
        beta = beta_init.copy()
    sigma_g2 = sigma_g2_init
    sigma_y2 = sigma_y2_init
    design = TruncatedDesign(X, rank) if rank is not None else None

    for it in range(1, max_iter + 1):
        th = theta_conditional_pmom(beta[1:], sigma_g2, tau, xi0, xi1, wr, eps=eps)
        precision = _precision_from_theta(th, tau)  # same functional form as the local case, tau for kappa
        dof = p1 / 2.0 + nu + 1.0 + np.sum(th)  # +sum(th): pMOM's sg2^(-3/2) vs local
                                                 # slab's sg2^(-1/2) normalizing power
                                                 # for included features -- see docstring
        sigma_g2_new = sigma_g2 if sigma_g2_fixed else (
            0.5 * np.sum(beta ** 2 * precision) + lam) / dof

        a_safe = np.sign(beta[1:]) * np.maximum(np.abs(beta[1:]), eps)
        a_safe = np.where(a_safe == 0, eps, a_safe)  # sign(0)=0 edge case
        curvature_extra = 2.0 * th / a_safe ** 2
        linear_extra = 4.0 * th / a_safe

        lambda_diag = np.empty(p1)
        lambda_diag[0] = precision[0] / sigma_g2_new
        lambda_diag[1:] = precision[1:] / sigma_g2_new + curvature_extra
        b_vec = np.zeros(p1)
        b_vec[1:] = linear_extra

        ssr = np.sum((y - X @ beta) ** 2)
        sigma_y2_new = (0.5 * ssr + lam_y) / (n / 2.0 + nu_y + 1.0)

        w = np.full(n, 1.0 / sigma_y2_new)
        c = X.T @ (y / sigma_y2_new) + b_vec
        if design is not None:
            S = design.weighted_factor(w)
            beta_new = woodbury_solve(S, 1.0 / lambda_diag, c)
        else:
            XtWX = X.T @ (w[:, None] * X)
            beta_new = np.linalg.solve(XtWX + np.diag(lambda_diag), c)

        step = np.clip(beta_new - beta, -max_step, max_step)
        beta_new = beta + step

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma_g2, sigma_y2 = beta_new, sigma_g2_new, sigma_y2_new
        if delta < tol:
            break

    th = theta_conditional_pmom(beta[1:], sigma_g2, tau, xi0, xi1, wr, eps=eps)
    mu = X @ beta
    return EMResultGaussian(beta=beta, sigma_g2=sigma_g2, sigma_y2=sigma_y2,
                             theta_hat=th, mu=mu, n_iter=it)


@dataclass
class FilterStepGaussian:
    step: int
    n_features: int
    ppl: float
    rppl: float
    retained_idx: np.ndarray
    beta: np.ndarray
    sigma_g2: float
    sigma_y2: float
    theta_hat: np.ndarray
    cv_ppl: float = None  # held-out PPL used for step selection when cv_folds is set; None otherwise


@dataclass
class EMFilterResultGaussian:
    history: list = field(default_factory=list)
    best: "FilterStepGaussian" = None


def _cv_ppl_gaussian(X, y, wr, xi0, xi1, kappa, nu, lam, nu_y, lam_y, rank,
                      beta_init, sigma_g2_init, sigma_y2_init, cv_folds, cv_seed):
    """K-fold held-out PPL for a FIXED feature set -- Gaussian-response counterpart of
    spike_slab._cv_ppl_bernoulli. See that function's docstring and the 2026-09-17/18
    project-memory entry for the diagnosis this is fixing (in-sample PPL over-selects
    under moderate SNR + a sparse-relative-to-candidate-pool true architecture)."""
    n = X.shape[0]
    fold_id = np.random.default_rng(cv_seed).permutation(n) % cv_folds
    total = 0.0
    for k in range(cv_folds):
        test = fold_id == k
        train = ~test
        res_fold = fit_em_gaussian(X[train], y[train], wr, xi0, xi1, kappa, nu, lam, nu_y, lam_y,
                                    rank=rank, beta_init=beta_init, sigma_g2_init=sigma_g2_init,
                                    sigma_y2_init=sigma_y2_init)
        mu_test = X[test] @ res_fold.beta
        total += ppl_gaussian(y[test], mu_test, res_fold.sigma_y2)
    return total


def em_filter_gaussian(X, y, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0,
                        filter_frac=0.25, min_features=10, max_outer=200,
                        rank=None, patience=3, verbose=False, cv_folds=None, cv_seed=0):
    """EM-filtering pipeline for continuous responses -- same distilled-sensing loop
    as spike_slab.em_filter, just calling fit_em_gaussian and tracking sigma_y2 too.

    `cv_folds=None` (default): UNCHANGED in-sample-PPL step selection, same as every
    prior use of this function. `cv_folds=K`: opt-in K-fold held-out PPL for step
    selection -- see spike_slab.em_filter's docstring for the full rationale, which
    applies identically here (both response families showed the same over-selection
    failure mode when this was diagnosed)."""
    p = X.shape[1] - 1
    idx = np.arange(p)
    X_cur = X
    wr_cur = wr
    beta = None
    sigma_g2 = 1.0
    sigma_y2 = float(np.var(y))

    null_mu = np.full(y.shape[0], y.mean())
    ppl_null = ppl_gaussian(y, null_mu, float(np.var(y)))

    history, best, bad_streak = [], None, 0
    for step in range(max_outer):
        res = fit_em_gaussian(X_cur, y, wr_cur, xi0, xi1, kappa, nu, lam, nu_y, lam_y,
                               rank=rank, beta_init=beta, sigma_g2_init=sigma_g2,
                               sigma_y2_init=sigma_y2)
        ppl_val = ppl_gaussian(y, res.mu, res.sigma_y2)
        cv_ppl_val = None
        if cv_folds is not None:
            cv_ppl_val = _cv_ppl_gaussian(X_cur, y, wr_cur, xi0, xi1, kappa, nu, lam, nu_y, lam_y,
                                           rank, beta, sigma_g2, sigma_y2, cv_folds, cv_seed)
        select_val = cv_ppl_val if cv_folds is not None else ppl_val
        record = FilterStepGaussian(step=step, n_features=idx.shape[0], ppl=ppl_val,
                                     rppl=ppl_val / ppl_null, retained_idx=idx.copy(),
                                     beta=res.beta, sigma_g2=res.sigma_g2,
                                     sigma_y2=res.sigma_y2, theta_hat=res.theta_hat,
                                     cv_ppl=cv_ppl_val)
        history.append(record)
        if verbose:
            msg = f"step {step:3d}  p={idx.shape[0]:6d}  PPL={ppl_val:.5f}  rPPL={record.rppl:.4f}"
            if cv_ppl_val is not None:
                msg += f"  CV-PPL={cv_ppl_val:.5f}"
            print(msg)

        best_select_val = (best.cv_ppl if cv_folds is not None else best.ppl) if best is not None else None
        if best is None or select_val < best_select_val:
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
        X_cur = X[:, np.concatenate([[0], idx + 1])]
        wr_cur = wr[idx]
        beta = np.concatenate([[res.beta[0]], res.beta[1:][keep_mask]])
        sigma_g2, sigma_y2 = res.sigma_g2, res.sigma_y2

    return EMFilterResultGaussian(history=history, best=best)


def em_filter_gaussian_pmom(X, y, wr, xi0, xi1, tau, nu, lam, nu_y=1.0, lam_y=1.0,
                             filter_frac=0.25, min_features=10, max_outer=200,
                             rank=None, patience=3, verbose=False, max_step=5.0, eps=1e-6):
    """Non-local-prior (pMOM) counterpart of em_filter_gaussian -- same distilled-
    sensing filtering loop, calling fit_em_gaussian_pmom instead of fit_em_gaussian.
    See that function's docstring for the derivation and known simplifications.

    No cv_folds option here yet (unlike em_filter_gaussian) -- kept out of this first
    version deliberately to isolate what the non-local prior itself changes before
    layering another opt-in mechanism on top; yesterday's finding was that cv_folds
    only marginally helped the local prior anyway, so it's a lower priority than
    validating the core E/M-step change first.

    Elimination order (which features get dropped each round) is unchanged in
    spirit: still driven by the current step's theta_hat ranking, now computed under
    the pMOM E-step instead of the local one.
    """
    p = X.shape[1] - 1
    idx = np.arange(p)
    X_cur = X
    wr_cur = wr
    beta = None
    sigma_g2 = 1.0
    sigma_y2 = float(np.var(y))

    null_mu = np.full(y.shape[0], y.mean())
    ppl_null = ppl_gaussian(y, null_mu, float(np.var(y)))

    history, best, bad_streak = [], None, 0
    for step in range(max_outer):
        res = fit_em_gaussian_pmom(X_cur, y, wr_cur, xi0, xi1, tau, nu, lam, nu_y, lam_y,
                                    rank=rank, beta_init=beta, sigma_g2_init=sigma_g2,
                                    sigma_y2_init=sigma_y2, max_step=max_step, eps=eps)
        ppl_val = ppl_gaussian(y, res.mu, res.sigma_y2)
        record = FilterStepGaussian(step=step, n_features=idx.shape[0], ppl=ppl_val,
                                     rppl=ppl_val / ppl_null, retained_idx=idx.copy(),
                                     beta=res.beta, sigma_g2=res.sigma_g2,
                                     sigma_y2=res.sigma_y2, theta_hat=res.theta_hat)
        history.append(record)
        if verbose:
            print(f"step {step:3d}  p={idx.shape[0]:6d}  PPL={ppl_val:.5f}  rPPL={record.rppl:.4f}")

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
        X_cur = X[:, np.concatenate([[0], idx + 1])]
        wr_cur = wr[idx]
        beta = np.concatenate([[res.beta[0]], res.beta[1:][keep_mask]])
        sigma_g2, sigma_y2 = res.sigma_g2, res.sigma_y2

    return EMFilterResultGaussian(history=history, best=best)


@dataclass
class GibbsResultGaussian:
    beta: np.ndarray
    theta: np.ndarray
    sigma_g2: np.ndarray
    sigma_y2: np.ndarray
    pi_hat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pi_hat = self.theta.mean(axis=0)


def gibbs_sampler_gaussian(X, y, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0,
                            n_samples=2000, burn_in=500, thin=1, rank=None,
                            beta_init=None, sigma_g2_init=1.0, sigma_y2_init=1.0,
                            theta_init=None, seed=None):
    """Gibbs sampler for the Gaussian-response spike-and-slab. Fully conjugate: every
    conditional (sigma_g2, theta, sigma_y2, beta) has a closed-form draw, no data
    augmentation needed."""
    rng = np.random.default_rng(seed)
    n, p1 = X.shape
    p = p1 - 1

    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma_g2 = sigma_g2_init
    sigma_y2 = sigma_y2_init
    theta = np.ones(p, dtype=bool) if theta_init is None else theta_init.copy()

    design = TruncatedDesign(X, rank) if rank is not None else None

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

        ssr = np.sum((y - X @ beta) ** 2)
        sigma_y2 = invgamma.rvs(nu_y + n / 2.0, scale=lam_y + 0.5 * ssr, random_state=rng)

        sigma_diag = np.empty(p1)
        sigma_diag[0] = sigma_g2 * kappa
        sigma_diag[1:] = sigma_g2 * np.where(theta, kappa, 1.0)

        w = np.full(n, 1.0 / sigma_y2)
        c = X.T @ (y / sigma_y2)
        if design is not None:
            S = design.weighted_factor(w)
            beta, _ = woodbury_mean_and_sample(S, sigma_diag, c, rng)
        else:
            XtWX = X.T @ (w[:, None] * X)
            beta, _ = exact_mean_and_sample(XtWX, sigma_diag, c, rng)

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_samples[kept] = beta
            theta_samples[kept] = theta
            sigma_g2_samples[kept] = sigma_g2
            sigma_y2_samples[kept] = sigma_y2
            kept += 1

    return GibbsResultGaussian(beta=beta_samples, theta=theta_samples,
                                sigma_g2=sigma_g2_samples, sigma_y2=sigma_y2_samples)
