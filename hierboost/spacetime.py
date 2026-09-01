"""Kronecker-structured space x time block-latent model -- hierboost.factor's spatial
block-factor idea (one shared latent per block) and hierboost.state_space's temporal one
(a smoothly evolving AR(1) trajectory) combined into a single joint model, per Ian's own
"who knows, eventually" aside about wanting this (see the project README's deferred-
next-steps note, and hierboost.estimator's `decorrelate` string-flag design, which was
built with a future "star" option in mind from the start).

Instead of K independent AR(1) trajectories (one per block -- decorrelate="ar1" applied
block-by-block, state_space.py), the K blocks' latent states z_t in R^K evolve together
under one shared VAR(1) transition

    z_t = Phi @ z_{t-1} + eta_t,   eta_t ~ N(0, diag(Q))

with Phi structurally constrained to

    Phi = rho1 * I + rho2 * W

where W is Chapter 4's SAR spatial weight matrix (hierboost.kernels.sar_weight_matrix)
evaluated on the blocks' spatial centroids (not their raw member coordinates) -- the
"Kronecker space x time" idea made tractable: rho1 is each block's own one-step
persistence (the scalar AR(1) case's rho, generalized), rho2 is how strongly a block's
spatial neighbors' PAST state leaks into its OWN present state. Two scalars, not a full
K x K unconstrained VAR(1) matrix, so the M-step for (rho1, rho2) reduces to a 2x2
linear system (see the derivation in fit_spacetime_block_factor) instead of a
K^2-parameter fit -- tractable and directly interpretable, unlike an unconstrained
VAR(1) Phi would be. With only one block (K=1), W is a 1x1 zero matrix, rho2 drops out
entirely, and this degenerates exactly to the ordinary scalar AR(1) case
(state_space.fit_temporal_block_factor) -- a useful consistency check.

Per-block observation model, per-block loadings/obs_var M-step, and the smoothed-
moments EM/RTS-smoother scaffolding are exactly hierboost.state_space's, generalized
from a scalar state to a K-dimensional one. Every raw measurement is still processed one
at a time (sequential scalar Kalman update, generalizing state_space.py's `_kalman_
filter` inner loop from a scalar-state rank-1 update to a K-dim-state rank-1 update)
since every measurement only ever loads on its own block's single state coordinate --
O(T*M*K^2) total cost instead of O(T*M^3) forming a dense M x M innovation covariance
every step, decisive once M (total raw features across all blocks) is much bigger than
K (block count).

Continuous/Gaussian observations only -- no JAX needed, same as state_space.py.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def _kalman_filter_star(Xc_by_block, block_ids, loadings, obs_var, Phi, Q,
                         z0_mean=None, z0_cov=None):
    """Forward filter, processing each raw member as a separate scalar measurement of
    its own block's coordinate of the K-dim state (exact for block-diagonal loadings and
    diagonal observation noise, and avoids ever forming an M x M innovation covariance).
    Mirrors state_space.py's `_kalman_filter`, generalized from a scalar to a K-vector
    state -- see that function's docstring for the per-measurement update this reduces to
    when K=1."""
    K = len(block_ids)
    T = Xc_by_block[block_ids[0]].shape[0]
    z_filt = np.empty((T, K))
    P_filt = np.empty((T, K, K))
    z_pred = np.empty((T, K))
    P_pred = np.empty((T, K, K))
    loglik = 0.0

    if z0_cov is None:
        z0_cov = solve_discrete_lyapunov(Phi, np.diag(Q))
    if z0_mean is None:
        z0_mean = np.zeros(K)

    z, P = z0_mean.copy(), z0_cov.copy()
    for t in range(T):
        if t > 0:
            z = Phi @ z_filt[t - 1]
            P = Phi @ P_filt[t - 1] @ Phi.T + np.diag(Q)
        z_pred[t], P_pred[t] = z, P
        for k, b in enumerate(block_ids):
            Xb = Xc_by_block[b]
            for j in range(Xb.shape[1]):
                Hval = loadings[b][j]
                innov = Xb[t, j] - Hval * z[k]
                S = Hval ** 2 * P[k, k] + obs_var[b][j]
                Kgain = Hval * P[:, k] / S
                z = z + Kgain * innov
                P = P - np.outer(Kgain, Hval * P[k, :])
                loglik += -0.5 * (np.log(2.0 * np.pi * S) + innov ** 2 / S)
        z_filt[t], P_filt[t] = z, P
    return z_filt, P_filt, z_pred, P_pred, loglik


def _rts_smoother_star(z_filt, P_filt, z_pred, P_pred, Phi):
    """Backward RTS pass, generalizing state_space.py's `_rts_smoother` from a scalar
    gain to the standard K x K smoother gain matrix J_t = P_filt[t] Phi^T P_pred[t+1]^-1;
    P_lag1[t] is Cov(z_t, z_{t+1} | data), same convention as the scalar version."""
    T, K = z_filt.shape
    z_smooth = np.empty((T, K))
    P_smooth = np.empty((T, K, K))
    P_lag1 = np.empty((T - 1, K, K))

    z_smooth[-1], P_smooth[-1] = z_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        J = P_filt[t] @ Phi.T @ np.linalg.inv(P_pred[t + 1])
        z_smooth[t] = z_filt[t] + J @ (z_smooth[t + 1] - z_pred[t + 1])
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T
        P_lag1[t] = J @ P_smooth[t + 1]
    return z_smooth, P_smooth, P_lag1


def _rho_star_mstep(z_smooth, P_smooth, P_lag1, W):
    """Closed-form M-step for (rho1, rho2) in Phi = rho1*I + rho2*W: minimizing
    sum_t E||z_t - rho1*z_{t-1} - rho2*W*z_{t-1}||^2 under the smoothed posterior reduces
    to a 2x2 linear system in the two unknown scalars (a weighted linear regression of
    the K-dim "response" z_t on the two K-dim "regressors" z_{t-1} and W*z_{t-1}, stacked
    over both time and the K state coordinates) -- see the module docstring's overview
    and this function's use of trace identities (a^T M a = trace(M a a^T) for a vector a
    and symmetric M) to reduce each K x K smoothed second moment to the scalar inner
    products the normal equations need. W is assumed symmetric (true for
    hierboost.kernels.sar_weight_matrix), which is what lets a^T W a and a^T W^2 a be
    written as plain traces without transposing W.
    """
    T = z_smooth.shape[0]
    W2 = W @ W
    S_11 = S_AW = S_WW = S_zz_lag = S_zWz_lag = 0.0
    for t in range(1, T):
        M_lag = np.outer(z_smooth[t - 1], z_smooth[t - 1]) + P_smooth[t - 1]
        C = P_lag1[t - 1].T + np.outer(z_smooth[t], z_smooth[t - 1])  # E[z_t z_{t-1}^T]
        S_11 += np.trace(M_lag)
        S_AW += np.trace(W @ M_lag)
        S_WW += np.trace(W2 @ M_lag)
        S_zz_lag += np.trace(C)
        S_zWz_lag += np.trace(W @ C)

    A = np.array([[S_11, S_AW], [S_AW, S_WW]])
    rhs = np.array([S_zz_lag, S_zWz_lag])
    rho1, rho2 = np.linalg.solve(A + 1e-10 * np.eye(2), rhs)

    if W.shape[0] > 1:
        w_eig = np.linalg.eigvalsh(W)
        max_abs = np.max(np.abs(rho1 + rho2 * w_eig))
    else:
        max_abs = abs(rho1)
    if max_abs >= 0.98:
        shrink = 0.98 / max_abs
        rho1, rho2 = rho1 * shrink, rho2 * shrink
    return float(rho1), float(rho2)


def _q_mstep(z_smooth, P_smooth, P_lag1, Phi):
    """Closed-form M-step for the (diagonal) innovation covariance, the standard
    Shumway-Stoffer state-noise update generalized to a vector state -- see
    state_space.py's scalar `state_var_new` line for the K=1 special case this reduces
    to (with Phi=rho, W absent)."""
    T, K = z_smooth.shape
    Q_sum = np.zeros((K, K))
    for t in range(1, T):
        M_t = np.outer(z_smooth[t], z_smooth[t]) + P_smooth[t]
        M_lag = np.outer(z_smooth[t - 1], z_smooth[t - 1]) + P_smooth[t - 1]
        C = P_lag1[t - 1].T + np.outer(z_smooth[t], z_smooth[t - 1])
        Q_sum += M_t - Phi @ C.T - C @ Phi.T + Phi @ M_lag @ Phi.T
    return np.clip(np.diag(Q_sum) / (T - 1), 1e-6, None)


@dataclass
class SpaceTimeFactorResult:
    z: np.ndarray                    # (T, K) smoothed shared latent trajectories
    z_cov_diag: np.ndarray            # (T, K) smoothed posterior variance per block
    loadings: dict                    # block_id -> (m_b,) per-member loadings
    obs_var: dict                     # block_id -> (m_b,) idiosyncratic variances
    train_mean: dict                  # block_id -> (m_b,) column means used to center
    block_ids: list                   # order matching z's columns
    W: np.ndarray                     # (K, K) spatial weight matrix used for coupling
    Phi: np.ndarray = field(init=False)   # (K, K) = rho1*I + rho2*W
    rho1: float = 0.0
    rho2: float = 0.0
    Q: np.ndarray = None              # (K,) diagonal innovation variances
    loglik: list = field(default_factory=list)
    n_iter: int = 0

    def __post_init__(self):
        K = len(self.block_ids)
        self.Phi = self.rho1 * np.eye(K) + self.rho2 * self.W


def fit_spacetime_block_factor(X_by_block, block_ids, W, n_iter=200, tol=1e-6,
                                rho1_init=0.5, rho2_init=0.0):
    """Fit the joint space x time block-factor model to K blocks' raw (T, m_b) member
    matrices via EM. Analogous to state_space.fit_temporal_block_factor(X_block) but
    fitting all K blocks' shared latents *jointly*, coupled through one structured
    transition Phi = rho1*I + rho2*W instead of K independent AR(1)s.

    `X_by_block`: dict block_id -> (T, m_b) raw array, all sharing the same T (time axis
    must already be aligned/ordered the same way across blocks). `block_ids`: the order
    z's columns follow. `W`: (K, K) symmetric spatial weight matrix over the blocks'
    centroids (hierboost.kernels.sar_weight_matrix).

    Identification: (loadings, z) has the usual factor-model scale/sign ambiguity, but
    ambiguity must be resolved with a SINGLE GLOBAL scalar/sign (applied identically to
    every block), not a per-block one -- a per-block rescaling z_b -> c_b*z_b would
    require Phi -> diag(c)^-1 Phi diag(c) to keep the fit unchanged, which is no longer
    of the form rho1*I + rho2*W unless every c_b is equal. A global rescale commutes
    through Phi exactly (rho1, rho2 are already scale-invariant), so this pins the
    output to a fixed overall scale/sign purely for readability, without touching
    rho1/rho2 at all.
    """
    K = len(block_ids)
    T = X_by_block[block_ids[0]].shape[0]
    train_mean = {b: X_by_block[b].mean(axis=0) for b in block_ids}
    Xc = {b: np.asarray(X_by_block[b], dtype=float) - train_mean[b] for b in block_ids}

    z = np.empty((T, K))
    loadings, obs_var = {}, {}
    for k, b in enumerate(block_ids):
        z0 = Xc[b].mean(axis=1)
        z0 = (z0 - z0.mean()) / (z0.std() + 1e-8)
        z[:, k] = z0
        loadings[b] = np.array([np.cov(Xc[b][:, j], z0)[0, 1] for j in range(Xc[b].shape[1])])
        obs_var[b] = (Xc[b] - np.outer(z0, loadings[b])).var(axis=0) + 1e-6

    rho1, rho2 = rho1_init, rho2_init
    Q = np.full(K, max(1.0 - rho1 ** 2, 1e-3))

    loglik_trace = []
    for it in range(n_iter):
        Phi = rho1 * np.eye(K) + rho2 * W
        z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter_star(
            Xc, block_ids, loadings, obs_var, Phi, Q)
        z_smooth, P_smooth, P_lag1 = _rts_smoother_star(z_filt, P_filt, z_pred, P_pred, Phi)
        loglik_trace.append(loglik)

        for k, b in enumerate(block_ids):
            zt = z_smooth[:, k]
            Pkk = P_smooth[:, k, k]
            second_moment_sum = float(np.sum(zt ** 2 + Pkk))
            loadings[b] = (Xc[b] * zt[:, None]).sum(axis=0) / second_moment_sum
            resid = Xc[b] - np.outer(zt, loadings[b])
            obs_var[b] = np.mean(resid ** 2, axis=0) + (loadings[b] ** 2) * np.mean(Pkk)

        rho1_new, rho2_new = _rho_star_mstep(z_smooth, P_smooth, P_lag1, W)
        Phi_new = rho1_new * np.eye(K) + rho2_new * W
        Q_new = _q_mstep(z_smooth, P_smooth, P_lag1, Phi_new)

        rho1, rho2, Q = rho1_new, rho2_new, Q_new
        z = z_smooth
        if it > 0 and abs(loglik - loglik_trace[-2]) < tol * (abs(loglik_trace[-2]) + 1e-8):
            break

    Phi = rho1 * np.eye(K) + rho2 * W
    z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter_star(Xc, block_ids, loadings, obs_var, Phi, Q)
    z_smooth, P_smooth, _ = _rts_smoother_star(z_filt, P_filt, z_pred, P_pred, Phi)
    loglik_trace.append(loglik)

    scale = float(np.sqrt(np.mean(z_smooth.var(axis=0))))
    if scale > 1e-12:
        z_smooth = z_smooth / scale
        P_smooth = P_smooth / scale ** 2
        for b in block_ids:
            loadings[b] = loadings[b] * scale
    mean_loading = np.mean([loadings[b].mean() for b in block_ids])
    if mean_loading < 0:
        z_smooth = -z_smooth
        for b in block_ids:
            loadings[b] = -loadings[b]

    z_cov_diag = np.diagonal(P_smooth, axis1=1, axis2=2).copy()
    return SpaceTimeFactorResult(z=z_smooth, z_cov_diag=z_cov_diag, loadings=loadings,
                                  obs_var=obs_var, train_mean=train_mean, block_ids=list(block_ids),
                                  W=W, rho1=rho1, rho2=rho2, Q=Q, loglik=loglik_trace, n_iter=it + 1)


def filter_spacetime_block_factor(X_new_by_block, block_ids, loadings, obs_var, Phi, Q,
                                   train_mean, smooth=True, z0_mean=None, z0_cov=None):
    """Out-of-sample counterpart of fit_spacetime_block_factor: apply an already-fitted
    model to new data via a fixed-parameter joint Kalman filter (optionally RTS-smoothed)
    instead of re-running EM -- the K-dim generalization of
    state_space.filter_temporal_block_factor."""
    Xc = {b: np.asarray(X_new_by_block[b], dtype=float) - train_mean[b] for b in block_ids}
    z_filt, P_filt, z_pred, P_pred, loglik = _kalman_filter_star(
        Xc, block_ids, loadings, obs_var, Phi, Q, z0_mean=z0_mean, z0_cov=z0_cov)
    if not smooth:
        return z_filt, P_filt, loglik
    z_smooth, P_smooth, _ = _rts_smoother_star(z_filt, P_filt, z_pred, P_pred, Phi)
    return z_smooth, P_smooth, loglik
