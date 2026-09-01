"""Domain-agnostic hierarchical spike-and-slab GLM: EM filtering + Polya-Gamma Gibbs.

This is the generic inference engine behind Johnston et al. (2016)'s Spatial Boost
model (Sec 4): a spike-and-slab logistic regression whose inclusion prior for each
feature is boosted by its affinity to "relevant" higher-level groups (genes, in the
original GWAS setting; anything with a notion of proximity/affinity here). Nothing
in this module refers to genotypes or genomic distance -- `wr` is just a per-feature
score in [0, 1] summarizing group affinity x group relevance (see hierboost.kernels).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit
from scipy.stats import invgamma

from .rank_utils import TruncatedDesign, woodbury_solve, woodbury_mean_and_sample, exact_mean_and_sample

try:
    from polyagamma import random_polyagamma
except ImportError:  # pragma: no cover
    random_polyagamma = None


def theta_conditional(beta_features, sigma2, kappa, xi0, xi1, wr):
    """P(theta_j = 1 | beta, sigma2, y). Shared by the EM E-step and the Gibbs theta-update."""
    logit_theta = (-0.5 * np.log(kappa)
                   - (beta_features ** 2 / (2.0 * sigma2)) * (1.0 / kappa - 1.0)
                   + xi0 + xi1 * wr)
    return expit(logit_theta)


def ppl(y, mu):
    """Posterior predictive loss under squared error."""
    return float(np.sum((y - mu) ** 2 + mu * (1.0 - mu)))


def _precision_from_theta(theta_hat_features, kappa):
    p1 = theta_hat_features.shape[0] + 1
    precision = np.empty(p1)
    precision[0] = 1.0 / kappa          # intercept: theta_0 = 1 always
    precision[1:] = theta_hat_features / kappa + (1.0 - theta_hat_features)
    return precision


@dataclass
class EMResult:
    beta: np.ndarray
    sigma2: float
    theta_hat: np.ndarray
    mu: np.ndarray
    n_iter: int


def fit_em(X, y, wr, xi0, xi1, kappa, nu, lam, rank=None,
           beta_init=None, sigma2_init=1.0, max_iter=200, tol=1e-7, firth=False):
    """Run the EM algorithm to convergence for a fixed feature set.

    `firth=True` (OFF by default -- see the negative result below; dense path only)
    applies Firth's (1993) "Bias reduction of maximum likelihood estimates" penalized-
    score correction to each M-step's logistic update. Motivation (see project memory
    for the full diagnosis):
    `hierboost.latent.fit_latent_block_model`'s outer loop calls this SAME function
    repeatedly on a block-latent design Z that is itself re-estimated each outer
    iteration using the CURRENT beta -- confirmed empirically that Z stays bounded and
    stable throughout, while beta can still run to extreme values (e.g. -36/+40 for a
    true effect of 0.8/-0.64) as Z drifts to fit y increasingly well. This is classical
    quasi-complete separation: an unpenalized (or, as here, only weakly ridge-penalized
    via kappa) logistic MLE has no finite optimum once the design nearly separates the
    outcome, and confirmed NOT fixable by tightening kappa (see the docstring in
    hierboost/latent.py's newton_update_ztilde). Firth's correction adds a term to the
    score equation equivalent to a Jeffreys-prior penalty, `|I(beta)|^(1/2)`, on the
    likelihood -- the standard, decades-old remedy specifically for this failure mode
    (routinely used in biostatistics for rare-event/near-separated logistic regression),
    which provably keeps beta finite even under exact separation. Implemented via the
    standard working-response form: augment each iteration's residual `y - mu` with
    `h_i * (0.5 - mu_i)`, `h_i` the penalized-design leverage
    `W_i * x_i' (X'WX + prior_precision)^{-1} x_i` -- computed from the SAME matrix
    already being solved for beta, one extra `n x p1` linear solve per iteration, not
    `n` separate ones. Only implemented for the dense (`rank=None`) path.

    **Tested, kept OFF by default -- a real negative result, not a working fix**
    (2026-08-30, see project memory): applied inside `fit_latent_block_model`'s outer
    alternation, Firth's correction shrank one diverging 2-causal-block scenario's
    blowup somewhat (-35.6/+40.0 -> -22.1/+26.1 at the standard settings) but did NOT
    stop it from continuing to grow with more outer iterations, and on a DIFFERENT,
    single-causal-block scenario it made the existing bias WORSE (8.6 -> 12.0 against a
    true value of 1.5, vs no correction). Worse still, turning it on UNCONDITIONALLY
    also degraded the plain no-decorrelate Binomial EM path's ALREADY-validated
    calibration (causal coverage 94.7% -> 89.1% over 300 replications) -- a real
    regression in previously-working, calibration-tested behavior. Diagnosis: Firth's
    correction fixes divergence WITHIN one fixed-design logistic fit, but the actual
    mechanism here is the OUTER alternation letting Z's correlation with y creep up
    across iterations (confirmed: even n_outer=3 already shows meaningful inflation,
    n_outer=5 is most of the way to the full blowup -- this is a fast, few-iteration
    effect, not a slow multi-iteration drift, so reducing n_outer doesn't reliably help
    either) -- a single-fit bias correction doesn't address a multi-iteration feedback
    process. Left available as an opt-in for further experimentation, not recommended.
    """
    n, p1 = X.shape
    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma2 = sigma2_init
    design = TruncatedDesign(X, rank) if rank is not None else None

    for it in range(1, max_iter + 1):
        th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
        precision = _precision_from_theta(th, kappa)

        sigma2_new = (0.5 * np.sum(beta ** 2 * precision) + lam) / (p1 / 2.0 + nu + 1.0)
        sigma_diag = sigma2_new / precision

        eta = X @ beta
        mu = expit(eta)
        W = np.clip(mu * (1.0 - mu), 1e-6, None)
        resid = y - mu

        if design is not None:
            S = design.weighted_factor(W)
            v = S.T @ (S @ beta) + X.T @ resid
            beta_new = woodbury_solve(S, sigma_diag, v)
        else:
            XtWX = X.T @ (W[:, None] * X)
            M = XtWX + np.diag(1.0 / sigma_diag)
            if firth:
                V = np.linalg.solve(M, X.T)              # (p1, n)
                h = np.clip(W * np.einsum("ij,ji->i", X, V), 0.0, 1.0)
                resid = resid + h * (0.5 - mu)
            v = XtWX @ beta + X.T @ resid
            beta_new = np.linalg.solve(M, v)

        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta, sigma2 = beta_new, sigma2_new
        if delta < tol:
            break

    th = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
    mu = expit(X @ beta)
    return EMResult(beta=beta, sigma2=sigma2, theta_hat=th, mu=mu, n_iter=it)


@dataclass
class FilterStep:
    step: int
    n_features: int
    ppl: float
    rppl: float
    retained_idx: np.ndarray
    beta: np.ndarray
    sigma2: float
    theta_hat: np.ndarray


@dataclass
class EMFilterResult:
    history: list = field(default_factory=list)
    best: "FilterStep" = None


def em_filter(X, y, wr, xi0, xi1, kappa, nu, lam,
              filter_frac=0.25, min_features=10, max_outer=200,
              rank=None, patience=3, verbose=False):
    """EM filtering pipeline (distilled-sensing style): fit, drop lowest-theta
    features, refit warm-started, track PPL, stop when it stops improving or too
    few features remain. Returns the full trace plus the best (min-PPL) step.
    """
    p = X.shape[1] - 1
    idx = np.arange(p)
    X_cur = X
    wr_cur = wr
    beta = None
    sigma2 = 1.0

    null_mu = np.full(y.shape[0], y.mean())
    ppl_null = ppl(y, null_mu)

    history, best, bad_streak = [], None, 0

    for step in range(max_outer):
        res = fit_em(X_cur, y, wr_cur, xi0, xi1, kappa, nu, lam, rank=rank,
                     beta_init=beta, sigma2_init=sigma2)
        ppl_val = ppl(y, res.mu)
        record = FilterStep(step=step, n_features=idx.shape[0], ppl=ppl_val,
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

    return EMFilterResult(history=history, best=best)


@dataclass
class GibbsResult:
    beta: np.ndarray          # (n_samples, p1)
    theta: np.ndarray         # (n_samples, p), bool
    sigma2: np.ndarray        # (n_samples,)
    pi_hat: np.ndarray = field(init=False)  # marginal P(theta_j=1|y)

    def __post_init__(self):
        self.pi_hat = self.theta.mean(axis=0)


def gibbs_sampler(X, y, wr, xi0, xi1, kappa, nu, lam,
                   n_samples=2000, burn_in=500, thin=1, rank=None,
                   beta_init=None, sigma2_init=1.0, theta_init=None, seed=None):
    """Gibbs sampler with Polya-Gamma data augmentation."""
    if random_polyagamma is None:
        raise ImportError("pip install polyagamma")

    rng = np.random.default_rng(seed)
    n, p1 = X.shape
    p = p1 - 1

    beta = np.zeros(p1) if beta_init is None else beta_init.copy()
    sigma2 = sigma2_init
    theta = np.ones(p, dtype=bool) if theta_init is None else theta_init.copy()

    design = TruncatedDesign(X, rank) if rank is not None else None
    c = X.T @ (y - 0.5)

    beta_samples = np.empty((n_samples, p1))
    theta_samples = np.empty((n_samples, p), dtype=bool)
    sigma2_samples = np.empty(n_samples)

    total = burn_in + n_samples * thin
    kept = 0
    for it in range(total):
        precision = _precision_from_theta(theta.astype(float), kappa)
        # NOTE: theta is 0/1 here so precision[1:] reduces exactly to theta/kappa+(1-theta).
        shape = nu + p1 / 2.0
        scale = lam + 0.5 * np.sum(beta ** 2 * precision)
        sigma2 = invgamma.rvs(shape, scale=scale, random_state=rng)

        th_prob = theta_conditional(beta[1:], sigma2, kappa, xi0, xi1, wr)
        theta = rng.random(p) < th_prob

        sigma_diag = np.empty(p1)
        sigma_diag[0] = sigma2 * kappa
        sigma_diag[1:] = sigma2 * np.where(theta, kappa, 1.0)

        eta = X @ beta
        omega = random_polyagamma(1.0, eta, random_state=rng)

        if design is not None:
            S = design.weighted_factor(omega)
            beta, _ = woodbury_mean_and_sample(S, sigma_diag, c, rng)
        else:
            XtOX = X.T @ (omega[:, None] * X)
            beta, _ = exact_mean_and_sample(XtOX, sigma_diag, c, rng)

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_samples[kept] = beta
            theta_samples[kept] = theta
            sigma2_samples[kept] = sigma2
            kept += 1

    return GibbsResult(beta=beta_samples, theta=theta_samples, sigma2=sigma2_samples)


def centroid_estimate(pi_hat, gamma=1.0):
    """Bayes-risk-optimal hard selection under a sensitivity/specificity trade-off gamma:
    select j iff pi_j >= 1/(1+gamma) (Carvalho & Lawrence 2008 centroid estimator)."""
    return pi_hat >= 1.0 / (1.0 + gamma)


def embfdr(theta_hat_conditional, gamma=1.0):
    """EM-based Bayesian false discovery rate at threshold 1/(1+gamma), using conditional
    posteriors from an EM fit as a fast proxy for full posterior inclusion probabilities."""
    selected = theta_hat_conditional >= 1.0 / (1.0 + gamma)
    if not np.any(selected):
        return 0.0
    return float(np.sum(1.0 - theta_hat_conditional[selected]) / np.sum(selected))


def select_kappa_by_embfdr(theta_hat_by_kappa, gammas, target_bfdr=0.05, gamma_fixed=None):
    """At a fixed operating threshold (1+gamma_fixed)^-1, pick the LARGEST kappa (sharpest
    spike/slab separation) whose EMBFDR still meets the target. Falls back to the kappa
    with the smallest EMBFDR at that threshold if none qualify. `gammas` is only used to
    build the full curves returned in `table` for plotting; selection uses `gamma_fixed`.
    """
    table = {kappa: [embfdr(th, g) for g in gammas] for kappa, th in theta_hat_by_kappa.items()}
    if gamma_fixed is None:
        gamma_fixed = np.median(gammas)

    qualifying = {kappa: embfdr(th, gamma_fixed) for kappa, th in theta_hat_by_kappa.items()
                  if embfdr(th, gamma_fixed) <= target_bfdr}
    if qualifying:
        best_kappa = max(qualifying, key=lambda k: k)
        return best_kappa, gamma_fixed, table

    all_bfdr = {kappa: embfdr(th, gamma_fixed) for kappa, th in theta_hat_by_kappa.items()}
    best_kappa = min(all_bfdr, key=all_bfdr.get)
    return best_kappa, gamma_fixed, table


def xi_bounds(kappa, s, gamma=1.0):
    """Upper bound on xi1 given kappa, min-sigma s, gamma (mirrors the paper's Sec 5.2)."""
    xi1_upper_of_xi0 = lambda xi0: (0.5 * np.log(kappa) - xi0 - np.log(gamma)
                                     - 0.5 * s ** 2 * (1.0 - 1.0 / kappa))
    xi0_upper = 0.5 * np.log(kappa) - 0.5 * s ** 2 * (1.0 - 1.0 / kappa) - np.log(gamma)
    return xi1_upper_of_xi0, xi0_upper
