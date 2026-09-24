"""Joint Bayesian inference for the block-latent model, replacing the two-stage
plug-in pipeline in hierboost.latent (fit Z-tilde/delta by Newton's method, THEN
treat them as fixed data for the gamma/theta/sigma2 EM) with a single joint
posterior sampled by NUTS. Estimation uncertainty in the block latents now
propagates into the posterior over block-outcome associations, which a point
estimate can never do -- exactly the gap the dissertation flags and leaves open
("a fully Bayesian approach is impractical... [we instead] focus on... the EM
filtering pipeline").

The one modeling change this requires: discrete spike-and-slab indicators (theta)
don't mix under Hamiltonian Monte Carlo. We swap in a regularized horseshoe prior
on the block effects (Piironen & Vehtari 2017) -- the standard, well-studied
continuous analogue of spike-and-slab used specifically because it plays well with
gradient-based samplers. The discrete spike-and-slab + EM/Gibbs path in
hierboost.spike_slab is unaffected; this is an alternative fitting route, not a
replacement.

**tau0 calibration (fixed 2026-08-29, see project memory for the diagnosis)**: the
original implementation used an uncalibrated `tau ~ HalfCauchy(1.0)` -- a generic
default with no relationship to K (how many blocks are competing for a fixed prior
"budget") or n (how much data is available to overrule the prior). Calibration
testing found this produced a striking, consistent ~2x shrinkage of the causal
block's posterior mean toward zero relative to the plug-in's own asymptotic value --
STABLE across a 4x range of true signal strength (0.15 and 0.6 in Z-units both gave
ratio~0.48-0.50), i.e. not "correctly shrinking a weak effect harder than a strong
one" (which would show a DECREASING ratio as signal strengthens) but a roughly
constant multiplicative miscalibration -- the signature of an under-scaled tau prior,
not a fundamental property of horseshoe shrinkage. Fixed by using Piironen &
Vehtari's own published calibration rule for tau0 (Sec 3, the GLM extension used for
logistic regression): `tau0 = (p0 / (K - p0)) * (sigma_approx / sqrt(n))`, p0 = the
expected number of truly relevant blocks (a prior sparsity guess, the same role
`xi0` plays for the EM path -- NOT re-derived from y), sigma_approx=2 (the paper's
own rule-of-thumb pseudo-residual-SD for a binary logistic response, also used by
the brms/rstanarm horseshoe implementations). Also added the "regularized" slab cap
(`c^2 lambda^2 / (c^2 + tau^2 lambda^2)`) the module docstring already claimed but
the original code never actually implemented -- bounds how large an included
block's effective local scale can grow, the other half of Piironen & Vehtari's
construction (without it, this was a plain, unregularized horseshoe).

Each block's shared-latent loading is a free `w` with a shrinkage prior toward the
SAR mechanism's own ell(phi) (see joint_block_model's `tau_w_scale` docstring), rather
than ell(phi) itself -- ell(phi) is entrywise non-negative in the SAR fit's stable
regime, so it can't represent a raw feature anti-correlated with its block (e.g. an
arbitrarily-coded reference allele); the free `w` can.

Per-block latent-prior hyperparameters (mu, phi -> Sigma_v) are taken as given
(fit once via hierboost.latent.fit_block_hyperparameters, same as the plug-in
path) rather than treated as top-level unknowns -- an intentional simplification,
not an oversight: those hyperparameters summarize known population genetics
(MAF/LD) that doesn't depend on this particular outcome y, so eliciting them
once via method-of-moments is standard practice even in a fully Bayesian model.

Requires jax and numpyro (`pip install numpyro`).
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def joint_block_model(X_blocks, y, CT_matrices, mu_v_list, Sigma_v_list, n_trials=2,
                       tau0=1.0, slab_scale=2.0, tau_w_scale=0.3):
    """`tau_w_scale` controls each feature's coefficient on its block's shared latent:
    instead of the SAR mechanism's fixed ell_j = CT[:, 0] (entrywise non-negative, so
    it can't represent a feature anti-correlated with its block, e.g. an arbitrarily-
    coded reference allele), the coefficient is a free w_j ~ Normal(mu0*ell_j, tau_w),
    with mu0/tau_w given their own priors and inferred jointly by NUTS. tau_w's
    HalfCauchy prior shrinks w toward the SAR shape by default but lets the posterior
    widen when the data disagrees."""
    n = y.shape[0]
    K = len(X_blocks)
    Z_cols = []
    for b in range(K):
        m = X_blocks[b].shape[1]
        mu_v, Sigma_v = mu_v_list[b], Sigma_v_list[b]
        ell_b = CT_matrices[b][:, 0]

        mu0_w = numpyro.sample(f"mu0_w_{b}", dist.Normal(1.0, 1.0))
        tau_w = numpyro.sample(f"tau_w_{b}", dist.HalfCauchy(tau_w_scale))
        w_raw = numpyro.sample(f"w_raw_{b}", dist.Normal(0.0, 1.0).expand([m]).to_event(1))
        w_b = numpyro.deterministic(f"w_{b}", mu0_w * ell_b + tau_w * w_raw)

        zt = numpyro.sample(f"Zt_{b}", dist.Normal(mu_v[0], jnp.sqrt(Sigma_v[0, 0])).expand([n]).to_event(1))
        if m > 1:
            delta = numpyro.sample(f"delta_{b}",
                                    dist.MultivariateNormal(mu_v[1:], Sigma_v[1:, 1:] + 1e-6 * jnp.eye(m - 1)))
            delta_contrib = CT_matrices[b][:, 1:] @ delta
        else:
            delta_contrib = jnp.zeros(m)

        U = zt[:, None] * w_b[None, :] + delta_contrib[None, :]
        numpyro.sample(f"X_{b}", dist.Binomial(total_count=n_trials, logits=U).to_event(1), obs=X_blocks[b])
        Z_cols.append(zt)

    Z = jnp.stack(Z_cols, axis=1)

    # Non-centered parameterization of the horseshoe (Betancourt & Girolami 2015): sampling
    # gamma directly as Normal(0, tau*lambda_tilde) creates a funnel that NUTS mixes poorly
    # in; sampling a unit-scale raw variable and rescaling deterministically fixes this.
    #
    # tau0 (passed in, calibrated by run_joint_inference -- see joint.py's module
    # docstring for the diagnosis this fixes): HalfCauchy(1.0) was an uncalibrated
    # default unrelated to K or n, empirically found to shrink the causal block's
    # posterior mean to ~50% of its true value, consistently across a 4x range of
    # signal strength -- the signature of a global scale miscalibration, not correct
    # signal-adaptive shrinkage. Piironen & Vehtari (2017) calibrate tau's own prior
    # scale to the problem's effective dimensionality instead of leaving it generic.
    #
    # slab_scale (the "regularized" part of "regularized horseshoe", also missing from
    # the original code despite the docstring already claiming it): caps how large an
    # included block's effective local scale can grow, `c^2 lambda^2 / (c^2 + tau^2
    # lambda^2)` -> lambda_tilde -> c as tau*lambda -> infinity, keeping genuinely
    # large effects on a weakly-informative Normal(0, slab_scale) slab instead of
    # letting the horseshoe's heavy tail grow unboundedly.
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    local_scale = numpyro.sample("local_scale", dist.HalfCauchy(1.0).expand([K]).to_event(1))
    lambda_tilde = (slab_scale * local_scale) / jnp.sqrt(slab_scale ** 2 + (tau * local_scale) ** 2)
    gamma_raw = numpyro.sample("gamma_raw", dist.Normal(0.0, 1.0).expand([K]).to_event(1))
    gamma = numpyro.deterministic("gamma", gamma_raw * tau * lambda_tilde)
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 5.0))

    eta = beta0 + Z @ gamma
    numpyro.sample("y", dist.Bernoulli(logits=eta).to_event(1), obs=y)


def run_joint_inference(X, y, block_id, block_fits, n_trials=2, p0=None, slab_scale=2.0,
                         tau_w_scale=0.3, num_warmup=500, num_samples=1000, seed=0, progress_bar=True):
    """block_fits: dict block_id -> hierboost.latent.BlockLatentFit (already fit via
    fit_block_hyperparameters), as produced inside fit_latent_block_model. Returns the
    MCMC object plus a posterior summary of block "association strength" comparable to
    the EM theta_hat (here: P(|gamma_b| exceeds a practical-significance threshold)).

    `p0`: prior guess at the number of truly relevant blocks (mirrors the role
    hierboost.spike_slab's `xi0` plays for the EM path -- a prior sparsity belief, not
    something derived from y). Sets tau's prior scale via Piironen & Vehtari (2017)'s
    own calibration rule, `tau0 = (p0/(K-p0)) * (sigma_approx/sqrt(n))` with
    sigma_approx=2 (their published rule-of-thumb pseudo-residual-SD for a binary
    logistic response). Defaults to max(1, K/10) -- a mildly-sparse prior guess,
    matching xi0's own EM-path default reasoning -- when not supplied.

    `p0` must satisfy 0 < p0 < K (the formula's denominator is K - p0; a user-supplied
    p0 outside that range raises ValueError rather than silently producing a non-finite
    tau0). BUG FIXED (2026-09-21, found wiring fit_method='joint' into
    HierBoostClassifier -- the very first caller to exercise K<=10): the auto-default
    `max(1, K/10)` equals K exactly for every K<=10 -- e.g. any single-block fit
    (K=1) -- making tau0's denominator exactly 0 and crashing with ZeroDivisionError
    before a single NUTS sample was ever drawn. Every decorrelate="ar1"/"sar" call
    with 10 or fewer blocks (a common case, e.g. one block per stability-selection-
    admitted region) hit this. Fixed by capping the auto-default strictly below K.
    """
    block_ids = sorted(block_fits.keys())
    from .blocks import block_membership_lists
    membership = block_membership_lists(block_id)

    X_blocks = [jnp.asarray(X[:, membership[b]], dtype=jnp.float32) for b in block_ids]
    CT_matrices = [jnp.asarray(block_fits[b].CT) for b in block_ids]
    mu_v_list = [jnp.asarray(block_fits[b].mu_v) for b in block_ids]
    Sigma_v_list = [jnp.asarray(jnp.linalg.inv(block_fits[b].Sigma_v_inv)) for b in block_ids]
    y_j = jnp.asarray(y, dtype=jnp.float32)

    n, K = y.shape[0], len(block_ids)
    if p0 is None:
        p0 = min(max(1.0, K / 10.0), K - 0.5)
    elif not (0 < p0 < K):
        raise ValueError(f"p0={p0} must satisfy 0 < p0 < K={K} (Piironen & Vehtari's "
                          f"tau0 formula divides by K - p0)")
    tau0 = (p0 / (K - p0)) * (2.0 / np.sqrt(n))

    kernel = NUTS(joint_block_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), X_blocks, y_j, CT_matrices, mu_v_list, Sigma_v_list, n_trials,
              tau0, slab_scale, tau_w_scale)
    return mcmc, block_ids


def posterior_association_summary(mcmc, block_ids, practical_threshold=0.1):
    """P(|gamma_b| > practical_threshold | y) per block -- the joint-inference analogue
    of the EM's theta_hat, but reflecting genuine posterior uncertainty in Z-tilde/delta
    rather than a point estimate plugged in beforehand."""
    samples = mcmc.get_samples()
    gamma_samples = np.array(samples["gamma"])  # (n_samples, K)
    prob_assoc = (np.abs(gamma_samples) > practical_threshold).mean(axis=0)
    ci_lo = np.percentile(gamma_samples, 2.5, axis=0)
    ci_hi = np.percentile(gamma_samples, 97.5, axis=0)
    return {
        "block_ids": block_ids,
        "prob_association": prob_assoc,
        "gamma_mean": gamma_samples.mean(axis=0),
        "gamma_ci_lo": ci_lo,
        "gamma_ci_hi": ci_hi,
    }


def posterior_beta_samples(mcmc):
    """(n_samples, K+1) array of [beta0, gamma_1..gamma_K] posterior draws, intercept
    first -- the same (n_samples, p1) shape/column order hierboost.spike_slab.
    GibbsResult.beta already uses, so downstream consumers (estimator.py's
    _compute_approx_covariance, which computes np.cov/.std over exactly this shape
    without caring how the samples were generated) work unchanged on a joint-inference
    fit. Added (2026-09-21) for wiring run_joint_inference into HierBoostClassifier --
    posterior_association_summary alone only ever surfaced gamma, not beta0 or the
    samples needed for a full posterior covariance."""
    samples = mcmc.get_samples()
    return np.column_stack([np.array(samples["beta0"]), np.array(samples["gamma"])])
