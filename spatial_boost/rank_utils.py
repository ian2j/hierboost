"""Rank-truncated linear algebra for the Spatial Boost model (Johnston et al., 2016, Sec 4.1).

The design matrix X is n x (p+1) with p in the tens of thousands, so forming or
inverting (p+1) x (p+1) matrices is infeasible. Following the paper's Eq. (8), we
keep a rank-l SVD surrogate of X and push every weighted normal-equations solve
through an l x l system via the Woodbury/Kailath identity.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


class TruncatedDesign:
    """Rank-l SVD surrogate X ~= U_l D_l V_l^T, reused across EM/Gibbs iterations."""

    def __init__(self, X, rank):
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        rank = min(rank, d.shape[0])
        self.U = U[:, :rank]
        self.d = d[:rank]
        self.Vt = Vt[:rank, :]
        self.rank = rank

    def relative_frobenius_error(self, X):
        approx = (self.U * self.d) @ self.Vt
        return np.linalg.norm(X - approx) / np.linalg.norm(X)

    def weighted_factor(self, w):
        """Return S (l x p1) such that S^T S approximates X^T diag(w) X."""
        UD = self.U * self.d                      # n x l
        Ws = np.sqrt(w)[:, None] * UD              # n x l
        A = Ws.T @ Ws                              # l x l = D U^T W U D
        C = np.linalg.cholesky(A + 1e-12 * np.eye(A.shape[0]))
        return C.T @ self.Vt                       # l x p1


def woodbury_solve(S, sigma_diag, v):
    """Solve (S^T S + diag(1/sigma_diag)) x = v without forming a (p1 x p1) matrix."""
    SD = S * sigma_diag[None, :]
    K = np.eye(S.shape[0]) + SD @ S.T
    rhs = SD @ v
    w = np.linalg.solve(K, rhs)
    return sigma_diag * v - sigma_diag * (S.T @ w)


def woodbury_mean_and_sample(S, sigma_diag, c, rng):
    """Mean and one draw from N(mean, (S^T S + diag(1/sigma_diag))^{-1}), mean = solve(.., c).

    Uses the fast structured-precision sampler: if u ~ N(0, diag(sigma_diag)) and
    delta ~ N(0, I_l) independently, then u - Sigma S^T (I + S Sigma S^T)^{-1}(S u + delta)
    has exactly the target zero-mean covariance (Bhattacharya et al. 2016-style identity).
    """
    l = S.shape[0]
    SD = S * sigma_diag[None, :]
    K = np.eye(l) + SD @ S.T
    Kf = cho_factor(K)

    w_mean = cho_solve(Kf, SD @ c)
    mean = sigma_diag * c - sigma_diag * (S.T @ w_mean)

    u = rng.standard_normal(S.shape[1]) * np.sqrt(sigma_diag)
    delta = rng.standard_normal(l)
    w_zero = cho_solve(Kf, S @ u + delta)
    zero_draw = u - sigma_diag * (S.T @ w_zero)

    return mean + zero_draw, mean


def exact_mean_and_sample(XtWX, sigma_diag, c, rng):
    """Dense counterpart of woodbury_mean_and_sample for small/moderate p."""
    Vb = XtWX + np.diag(1.0 / sigma_diag)
    L = np.linalg.cholesky(Vb)
    mean = cho_solve((L, True), c)
    z = rng.standard_normal(c.shape[0])
    draw = mean + solve_triangular(L.T, z, lower=False)
    return draw, mean
