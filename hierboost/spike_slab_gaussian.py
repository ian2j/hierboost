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
from .spike_slab import theta_conditional, _precision_from_theta


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


@dataclass
class EMFilterResultGaussian:
    history: list = field(default_factory=list)
    best: "FilterStepGaussian" = None


def em_filter_gaussian(X, y, wr, xi0, xi1, kappa, nu, lam, nu_y=1.0, lam_y=1.0,
                        filter_frac=0.25, min_features=10, max_outer=200,
                        rank=None, patience=3, verbose=False):
    """EM-filtering pipeline for continuous responses -- same distilled-sensing loop
    as spike_slab.em_filter, just calling fit_em_gaussian and tracking sigma_y2 too."""
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
