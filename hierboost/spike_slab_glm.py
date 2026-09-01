"""Count-response (Poisson / Negative-Binomial) counterpart of hierboost.spike_slab
and hierboost.spike_slab_gaussian -- for outcomes like event counts (page views,
disease cases, species observations) instead of Bernoulli or continuous responses.

Neither Poisson nor NB is conjugate the way the Gaussian response is, and neither
has a Polya-Gamma-style exact augmentation the way logistic regression does (PG
augmentation covers Binomial and Negative-Binomial with *known* dispersion, but not
Poisson). What both families DO share with the existing two response models is
Fisher-scoring IRLS: at each outer iteration, holding the current beta fixed, the
M-step is again "build a weighted design matrix and solve a regularized normal
equation" -- literally the same TruncatedDesign/woodbury_solve machinery spike_slab.py
and spike_slab_gaussian.py already use, just with the weight and working-response
formulas swapped for a log-link, count-variance version:

    eta = X @ beta + offset            (offset: e.g. log(population) for rate models)
    mu  = exp(eta)
    W   = V(mu)                        Poisson: V(mu) = mu (canonical link, W = V(mu)
                                        exactly); NB: V(mu) = mu + mu^2/r, log link is
                                        NOT canonical for NB so W = mu / (1 + mu/r)
                                        (general Fisher-scoring weight), which -> mu
                                        as r -> infinity, recovering the Poisson case
    z   = (eta - offset) + (y - mu)/mu     (IRLS working response, g'(mu) = 1/mu for
                                             the log link in both families)
    v   = X^T (W * z)

Everything about the spike-and-slab prior itself (theta_conditional, the
precision-from-theta mixture, the sigma2 M-step) is response-agnostic and reused
unchanged from spike_slab.py, exactly as spike_slab_gaussian.py already does.

NB's dispersion r is estimated by 1-D profile MLE each outer iteration (bounded scalar
optimization on log(r), holding mu fixed) -- the same alternating-M-step idea as
sigma2/sigma_y2 in the other two files, just via numerical optimization instead of a
closed form (no conjugate closed form exists for r).

No Gibbs sampler here yet -- see README/project notes for the natural next step
(Polya-Gamma augmentation for NB with r held fixed, unlike Poisson which has no known
exact augmentation).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.stats import invgamma

from .rank_utils import TruncatedDesign, woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample
from .spike_slab import theta_conditional, _precision_from_theta

try:
    from polyagamma import random_polyagamma
except ImportError:  # pragma: no cover
    random_polyagamma = None

_ETA_CLIP = 30.0  # exp(30) ~ 1e13, comfortably beyond any real count-data mu


def ppl_poisson(y, mu):
    """Posterior predictive loss (Gelfand-Ghosh style) for a Poisson response:
    squared error plus the Poisson predictive variance itself, mu."""
    return float(np.sum((y - mu) ** 2 + mu))


def ppl_nb(y, mu, r):
    """Same construction for Negative-Binomial: predictive variance is mu + mu^2/r."""
    return float(np.sum((y - mu) ** 2 + mu + mu ** 2 / r))


def _nb_loglik_r(r, y, mu):
    return np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
                   + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def _update_r(y, mu, r_bounds=(1e-3, 1e6)):
    """1-D profile MLE for NB dispersion r, holding mu fixed -- no closed form exists
    (unlike sigma2/sigma_y2's conjugate inverse-gamma updates), so this is a bounded
    scalar optimization instead, same role as those closed-form M-steps."""
    lo, hi = np.log(r_bounds[0]), np.log(r_bounds[1])

    def neg_ll(log_r):
        return -_nb_loglik_r(np.exp(log_r), y, mu)

    res = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    return float(np.exp(res.x))


@dataclass
class EMResultPoisson:
    beta: np.ndarray
    sigma2: float
    theta_hat: np.ndarray
    mu: np.ndarray
    n_iter: int


def fit_em_poisson(X, y, wr, xi0, xi1, kappa, nu, lam, offset=None, rank=None,
                    beta_init=None, sigma2_init=1.0, max_iter=200, tol=1e-7):
    """EM algorithm for the Poisson-response spike-and-slab (log link, canonical)."""
    n, p1 = X.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma2 = sigma2_init
    design = TruncatedDesign(X, rank) if rank is not None else None

    for it in range(1, max_iter + 1):
        th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
        precision = _precision_from_theta(th, kappa)
        sigma2_new = (0.5 * np.sum(beta ** 2 * precision) + lam) / (p1 / 2.0 + nu + 1.0)
        sigma_diag = sigma2_new / precision

        eta_lin = X @ beta
        eta = np.clip(eta_lin + offset, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        W = np.clip(mu, 1e-6, None)
        z = eta_lin + (y - mu) / W
        v = X.T @ (W * z)

        if design is not None:
            S = design.weighted_factor(W)
            beta_new = woodbury_solve(S, sigma_diag, v)
        else:
            XtWX = X.T @ (W[:, None] * X)
            beta_new = np.linalg.solve(XtWX + np.diag(1.0 / sigma_diag), v)

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma2 = beta_new, sigma2_new
        if delta < tol:
            break

    th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
    mu = np.exp(np.clip(X @ beta + offset, -_ETA_CLIP, _ETA_CLIP))
    return EMResultPoisson(beta=beta, sigma2=sigma2, theta_hat=th, mu=mu, n_iter=it)


@dataclass
class FilterStepPoisson:
    step: int
    n_features: int
    ppl: float
    rppl: float
    retained_idx: np.ndarray
    beta: np.ndarray
    sigma2: float
    theta_hat: np.ndarray


@dataclass
class EMFilterResultPoisson:
    history: list = field(default_factory=list)
    best: "FilterStepPoisson" = None


def em_filter_poisson(X, y, wr, xi0, xi1, kappa, nu, lam, offset=None,
                       filter_frac=0.25, min_features=10, max_outer=200,
                       rank=None, patience=3, verbose=False):
    """Distilled-sensing EM-filtering loop for Poisson, same shape as
    spike_slab.em_filter / spike_slab_gaussian.em_filter_gaussian."""
    p = X.shape[1] - 1
    idx = np.arange(p)
    X_cur = X
    wr_cur = wr
    beta = None
    sigma2 = 1.0

    null_mu = np.full(y.shape[0], y.mean())
    ppl_null = ppl_poisson(y, null_mu)

    history, best, bad_streak = [], None, 0
    for step in range(max_outer):
        res = fit_em_poisson(X_cur, y, wr_cur, xi0, xi1, kappa, nu, lam, offset=offset,
                              rank=rank, beta_init=beta, sigma2_init=sigma2)
        ppl_val = ppl_poisson(y, res.mu)
        record = FilterStepPoisson(step=step, n_features=idx.shape[0], ppl=ppl_val,
                                    rppl=ppl_val / ppl_null, retained_idx=idx.copy(),
                                    beta=res.beta, sigma2=res.sigma2, theta_hat=res.theta_hat)
        history.append(record)
        if verbose:
            print(f"step {step:3d}  p={idx.shape[0]:6d}  PPL={ppl_val:.2f}  rPPL={record.rppl:.4f}")

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
        sigma2 = res.sigma2

    return EMFilterResultPoisson(history=history, best=best)


@dataclass
class EMResultNB:
    beta: np.ndarray
    sigma2: float
    r: float
    theta_hat: np.ndarray
    mu: np.ndarray
    n_iter: int


def fit_em_nb(X, y, wr, xi0, xi1, kappa, nu, lam, offset=None, r_init=10.0, rank=None,
              beta_init=None, sigma2_init=1.0, max_iter=200, tol=1e-7):
    """EM algorithm for the Negative-Binomial-response spike-and-slab, log link
    (non-canonical for NB, so W != V(mu) the way it does for Poisson/canonical links --
    see module docstring for the general Fisher-scoring weight used instead). r is
    re-estimated by profile MLE every outer iteration, alternating with the beta/theta
    updates exactly like sigma2/sigma_y2 do in the other two response modules."""
    n, p1 = X.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma2 = sigma2_init
    r = r_init
    design = TruncatedDesign(X, rank) if rank is not None else None

    for it in range(1, max_iter + 1):
        th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
        precision = _precision_from_theta(th, kappa)
        sigma2_new = (0.5 * np.sum(beta ** 2 * precision) + lam) / (p1 / 2.0 + nu + 1.0)
        sigma_diag = sigma2_new / precision

        eta_lin = X @ beta
        eta = np.clip(eta_lin + offset, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)

        r_new = _update_r(y, mu, r_bounds=(1e-3, 1e6))

        W = np.clip(mu * r_new / (r_new + mu), 1e-6, None)
        z = eta_lin + (y - mu) / np.clip(mu, 1e-6, None)
        v = X.T @ (W * z)

        if design is not None:
            S = design.weighted_factor(W)
            beta_new = woodbury_solve(S, sigma_diag, v)
        else:
            XtWX = X.T @ (W[:, None] * X)
            beta_new = np.linalg.solve(XtWX + np.diag(1.0 / sigma_diag), v)

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma2, r = beta_new, sigma2_new, r_new
        if delta < tol:
            break

    th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
    mu = np.exp(np.clip(X @ beta + offset, -_ETA_CLIP, _ETA_CLIP))
    return EMResultNB(beta=beta, sigma2=sigma2, r=r, theta_hat=th, mu=mu, n_iter=it)


@dataclass
class FilterStepNB:
    step: int
    n_features: int
    ppl: float
    rppl: float
    retained_idx: np.ndarray
    beta: np.ndarray
    sigma2: float
    r: float
    theta_hat: np.ndarray


@dataclass
class EMFilterResultNB:
    history: list = field(default_factory=list)
    best: "FilterStepNB" = None


def em_filter_nb(X, y, wr, xi0, xi1, kappa, nu, lam, offset=None, r_init=10.0,
                  filter_frac=0.25, min_features=10, max_outer=200,
                  rank=None, patience=3, verbose=False):
    """Distilled-sensing EM-filtering loop for Negative-Binomial, same shape as the
    other three em_filter_* functions across the package."""
    p = X.shape[1] - 1
    idx = np.arange(p)
    X_cur = X
    wr_cur = wr
    beta = None
    sigma2 = 1.0
    r = r_init

    null_mu = np.full(y.shape[0], y.mean())
    ppl_null = ppl_nb(y, null_mu, r_init)

    history, best, bad_streak = [], None, 0
    for step in range(max_outer):
        res = fit_em_nb(X_cur, y, wr_cur, xi0, xi1, kappa, nu, lam, offset=offset,
                         r_init=r, rank=rank, beta_init=beta, sigma2_init=sigma2)
        ppl_val = ppl_nb(y, res.mu, res.r)
        record = FilterStepNB(step=step, n_features=idx.shape[0], ppl=ppl_val,
                               rppl=ppl_val / ppl_null, retained_idx=idx.copy(),
                               beta=res.beta, sigma2=res.sigma2, r=res.r, theta_hat=res.theta_hat)
        history.append(record)
        if verbose:
            print(f"step {step:3d}  p={idx.shape[0]:6d}  PPL={ppl_val:.2f}  "
                  f"rPPL={record.rppl:.4f}  r={res.r:.2f}")

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
        sigma2, r = res.sigma2, res.r

    return EMFilterResultNB(history=history, best=best)


@dataclass
class GibbsResultNB:
    beta: np.ndarray          # (n_samples, p1)
    theta: np.ndarray         # (n_samples, p), bool
    sigma2: np.ndarray        # (n_samples,)
    r: np.ndarray             # (n_samples,) -- profile-MLE plug-in each sweep, see docstring
    pi_hat: np.ndarray = field(init=False)  # marginal P(theta_j=1|y)

    def __post_init__(self):
        self.pi_hat = self.theta.mean(axis=0)


def gibbs_sampler_nb(X, y, wr, xi0, xi1, kappa, nu, lam, offset=None, r_init=10.0,
                      n_samples=2000, burn_in=500, thin=1, rank=None,
                      beta_init=None, sigma2_init=1.0, theta_init=None, seed=None,
                      r_bounds=(1e-3, 1e6), update_r_every=1):
    """Gibbs sampler for the Negative-Binomial spike-and-slab via Polya-Gamma data
    augmentation -- the extension flagged at the top of this module ("the natural next
    step via Polya-Gamma with r held fixed").

    NB regression admits the same PG augmentation logistic regression does (Polson,
    Scott & Windle 2013): writing psi = logit(p) = log(mu/r) = eta - log(r) (eta =
    X @ beta + offset), the NB likelihood as a function of psi has the same
    (e^psi)^a / (1+e^psi)^b form PG augmentation linearizes, but with a=y_i and
    b=y_i+r instead of logistic regression's fixed a=y_i, b=1 -- so kappa_i=(y_i-r)/2
    (vs Bernoulli's fixed y_i-0.5) and the PG draw has per-observation shape y_i+r (vs
    Bernoulli's fixed shape 1). The offset and -log(r) term both shift psi away from the
    raw linear predictor eta_lin=X@beta by a known-per-sweep constant c=offset-log(r);
    completing the square in beta absorbs that shift into an adjusted target
    kappa_i - omega_i*c_i, everything else (the prior/theta/sigma2 machinery) is
    unchanged from spike_slab.gibbs_sampler, which this generalizes.

    UNLIKE beta/theta/sigma2, r has no simple conjugate full conditional here, so it is
    NOT drawn from a posterior each sweep -- every `update_r_every` iterations it is
    re-estimated by the same profile MLE fit_em_nb's M-step already uses (`_update_r`),
    conditional on the current beta. This is an empirical/plug-in treatment of r (a point
    estimate fed back in, not integrated over) -- an honest simplification, not a claim
    of exact joint posterior sampling over (beta, theta, sigma2, r); `r` samples are
    still returned (as the `r` array on the result) so downstream code can see how much
    it moved, but treat its posterior spread as understated.
    """
    if random_polyagamma is None:
        raise ImportError("pip install polyagamma")

    rng = np.random.default_rng(seed)
    n, p1 = X.shape
    p = p1 - 1
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)

    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma2 = sigma2_init
    r = r_init
    theta = np.ones(p, dtype=bool) if theta_init is None else theta_init.copy()

    design = TruncatedDesign(X, rank) if rank is not None else None

    beta_samples = np.empty((n_samples, p1))
    theta_samples = np.empty((n_samples, p), dtype=bool)
    sigma2_samples = np.empty(n_samples)
    r_samples = np.empty(n_samples)

    total = burn_in + n_samples * thin
    kept = 0
    for it in range(total):
        precision = _precision_from_theta(theta.astype(float), kappa)
        shape = nu + p1 / 2.0
        scale = lam + 0.5 * np.sum(beta ** 2 * precision)
        sigma2 = invgamma.rvs(shape, scale=scale, random_state=rng)

        th_prob = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
        theta = rng.random(p) < th_prob

        sigma_diag = np.empty(p1)
        sigma_diag[0] = sigma2 * kappa
        sigma_diag[1:] = sigma2 * np.where(theta, kappa, 1.0)

        eta_lin = X @ beta
        c_shift = offset - np.log(r)
        psi = eta_lin + c_shift
        omega = random_polyagamma(y + r, psi, random_state=rng)
        v = X.T @ ((y - r) / 2.0 - omega * c_shift)

        if design is not None:
            S = design.weighted_factor(omega)
            beta, _ = woodbury_mean_and_sample(S, sigma_diag, v, rng)
        else:
            XtOX = X.T @ (omega[:, None] * X)
            beta, _ = exact_mean_and_sample(XtOX, sigma_diag, v, rng)

        if it % update_r_every == 0:
            mu = np.exp(np.clip(X @ beta + offset, -_ETA_CLIP, _ETA_CLIP))
            r = float(np.clip(_update_r(y, mu, r_bounds=r_bounds), *r_bounds))

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_samples[kept] = beta
            theta_samples[kept] = theta
            sigma2_samples[kept] = sigma2
            r_samples[kept] = r
            kept += 1

    return GibbsResultNB(beta=beta_samples, theta=theta_samples, sigma2=sigma2_samples, r=r_samples)
