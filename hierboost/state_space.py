"""The temporal counterpart of hierboost.factor: a block of time-adjacent, correlated
continuous features sharing one evolving latent state instead of one static one.

factor.py's block-latent model assumes a block's members are i.i.d. draws around a
shared latent per row (probabilistic PCA) -- appropriate when the block axis is *space*
or *identity* (SNPs, ETFs) and the row axis (individuals, trading days) carries no
ordering the model needs to respect. A temporal block is the opposite: the block members
are, say, several lagged or rolling-window readings of the same underlying signal, and
it is the *row* axis (time) whose ordering matters -- the shared latent is one smoothly
evolving trajectory z_t, not one number per row. That is a dynamic factor model:

    x_t = loadings * z_t + eps_t,   eps_t ~ N(0, diag(obs_var))
    z_t = rho * z_{t-1} + eta_t,    eta_t ~ N(0, state_var)

Like factor.py this stays fully conjugate (linear-Gaussian throughout, no Polya-Gamma or
autodiff needed) -- but unlike factor.py's one-shot SVD, the temporal dependence itself
(rho) has to be estimated, so fitting is an EM loop over exact closed-form E/M steps
(Shumway & Stoffer 1982): a scalar Kalman filter/RTS smoother for the E-step, closed-form
conditional-MLE updates for the M-step. This is the continuous, no-look-ahead-needed
analogue of hierboost.latent's discrete/binomial temporal option -- see
`hierboost.latent`'s `structure="ar1"` for the version that reuses Chapter 4's JAX Newton
machinery when the observation model isn't conjugate.
"""
from dataclasses import dataclass, field
import numpy as np


def _kalman_filter(X, loadings, obs_var, rho, state_var, z0_mean=0.0, z0_var=None):
    """Forward filter, processing each block member as a separate scalar measurement of
    the same 1D state (exact for diagonal observation noise, and avoids ever inverting an
    m x m matrix). Returns per-step filtered/predicted mean & variance plus the marginal
    log-likelihood, all needed by the RTS smoother and the EM M-step. Defaults to the
    process's stationary prior at t=0; pass z0_mean/z0_var to warm-start from a specific
    state instead (e.g. continuing an already-fitted trajectory onto new data).
    """
    T, m = X.shape
    z_filt = np.empty(T)
    P_filt = np.empty(T)
    z_pred = np.empty(T)
    P_pred = np.empty(T)
    loglik = 0.0

    if z0_var is None:
        z0_var = state_var / max(1.0 - rho ** 2, 1e-6)
    z, P = z0_mean, z0_var
    for t in range(T):
        if t > 0:
            z = rho * z_filt[t - 1]
            P = rho ** 2 * P_filt[t - 1] + state_var
        z_pred[t], P_pred[t] = z, P
        for j in range(m):
            innov = X[t, j] - loadings[j] * z
            S = loadings[j] ** 2 * P + obs_var[j]
            K = P * loadings[j] / S
            z = z + K * innov
            P = P - K * loadings[j] * P
            loglik += -0.5 * (np.log(2.0 * np.pi * S) + innov ** 2 / S)
        z_filt[t], P_filt[t] = z, P
    return z_filt, P_filt, z_pred, P_pred, loglik


def _rts_smoother(z_filt, P_filt, z_pred, P_pred, rho):
    """Backward Rauch-Tung-Striebel pass, plus the lag-one smoothed covariance
    Cov(z_t, z_{t+1} | all data) the rho/state_var M-step needs."""
    T = z_filt.shape[0]
    z_smooth = np.empty(T)
    P_smooth = np.empty(T)
    P_lag1 = np.empty(T - 1)

    z_smooth[-1], P_smooth[-1] = z_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        J = rho * P_filt[t] / P_pred[t + 1]
        z_smooth[t] = z_filt[t] + J * (z_smooth[t + 1] - z_pred[t + 1])
        P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred[t + 1])
        P_lag1[t] = J * P_smooth[t + 1]
    return z_smooth, P_smooth, P_lag1


@dataclass
class TemporalFactorResult:
    z: np.ndarray            # smoothed shared latent trajectory (T,)
    z_var: np.ndarray        # smoothed posterior variance of z_t (T,)
    loadings: np.ndarray     # per-member loadings (m,)
    obs_var: np.ndarray      # per-member idiosyncratic variance (m,)
    rho: float               # fitted AR(1) coefficient of the shared state
    state_var: float         # innovation variance (== 1 - rho^2 under the unit-stationary-variance normalization)
    train_mean: np.ndarray = None   # per-member column mean used to center training data -- reuse for new data
    loglik: list = field(default_factory=list)
    n_iter: int = 0


def fit_temporal_block_factor(X_block, n_iter=200, tol=1e-6, rho_init=0.5):
    """Fit the one-factor dynamic-state model to a (T, m) block of time-adjacent series
    via EM. Analogous to factor.py's gaussian_block_factor(X_block) -> (scores, loadings),
    but the shared latent is now a smoothed AR(1) trajectory rather than one number per row,
    and the return also carries the fitted temporal correlation rho itself -- often the more
    interesting quantity (how persistent is the block's common state?).

    Identification: the state's stationary variance is pinned to 1 after every M-step (the
    loadings absorb all scale), the standard dynamic-factor-model normalization -- without it
    the (loadings, state_var) split is only identified up to an arbitrary rescaling.
    """
    X = np.asarray(X_block, dtype=float)
    T, m = X.shape
    Xc = X - X.mean(axis=0)

    z0 = Xc.mean(axis=1)
    z0 = (z0 - z0.mean()) / (z0.std() + 1e-8)
    loadings = np.array([np.cov(Xc[:, j], z0)[0, 1] for j in range(m)])
    obs_var = (Xc - np.outer(z0, loadings)).var(axis=0) + 1e-6
    rho = rho_init
    state_var = 1.0 - rho ** 2

    loglik_trace = []
    for it in range(n_iter):
        z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter(Xc, loadings, obs_var, rho, state_var)
        z_smooth, P_smooth, P_lag1 = _rts_smoother(z_filt, P_filt, z_pred, P_pred, rho)
        loglik_trace.append(loglik)

        second_moment = z_smooth ** 2 + P_smooth
        loadings_new = (Xc * z_smooth[:, None]).sum(axis=0) / second_moment.sum()
        resid = Xc - np.outer(z_smooth, loadings_new)
        obs_var_new = np.mean(resid ** 2, axis=0) + (loadings_new ** 2) * np.mean(P_smooth)

        cross = P_lag1 + z_smooth[:-1] * z_smooth[1:]
        auto_t, auto_t1 = second_moment[:-1], second_moment[1:]
        rho_new = np.clip(cross.sum() / auto_t.sum(), -0.98, 0.98)
        state_var_new = max(np.mean(auto_t1 - 2.0 * rho_new * cross + rho_new ** 2 * auto_t), 1e-6)

        scale = np.sqrt(state_var_new / (1.0 - rho_new ** 2))
        loadings, rho, state_var = loadings_new * scale, rho_new, 1.0 - rho_new ** 2
        obs_var = obs_var_new

        if it > 0 and abs(loglik - loglik_trace[-2]) < tol * (abs(loglik_trace[-2]) + 1e-8):
            break

    z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter(Xc, loadings, obs_var, rho, state_var)
    z_smooth, P_smooth, _ = _rts_smoother(z_filt, P_filt, z_pred, P_pred, rho)
    loglik_trace.append(loglik)

    if loadings.mean() < 0:
        loadings, z_smooth = -loadings, -z_smooth

    return TemporalFactorResult(z=z_smooth, z_var=P_smooth, loadings=loadings, obs_var=obs_var,
                                 rho=rho, state_var=state_var, train_mean=X.mean(axis=0),
                                 loglik=loglik_trace, n_iter=it + 1)


def filter_temporal_block_factor(X_new_block, loadings, obs_var, rho, state_var, train_mean,
                                  smooth=True, z0_mean=0.0, z0_var=None):
    """Out-of-sample counterpart of fit_temporal_block_factor: apply an already-fitted
    model to new data via a fixed-parameter Kalman filter (optionally RTS-smoothed)
    instead of re-running EM. Starts from the model's stationary prior by default,
    treating X_new_block as an unrelated new series; pass z0_mean/z0_var (e.g. the last
    training z/z_var) to instead continue an already-fitted trajectory forward in time.
    """
    Xc = np.asarray(X_new_block, dtype=float) - train_mean
    z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter(Xc, loadings, obs_var, rho, state_var,
                                                             z0_mean=z0_mean, z0_var=z0_var)
    if not smooth:
        return z_filt, P_filt, loglik
    z_smooth, P_smooth, _ = _rts_smoother(z_filt, P_filt, z_pred, P_pred, rho)
    return z_smooth, P_smooth, loglik
