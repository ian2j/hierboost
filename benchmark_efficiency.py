"""Demonstrates the point of Sec 4.1's rank-truncation: at genome-scale p, dense
(p+1)x(p+1) linear algebra is not just slower but nearly unusable, while the
Woodbury/SVD surrogate gives an identical answer (when rank = n, the SVD
reconstructs X exactly) in a fraction of the time.

This is a separate script from demo.py because the dense comparison run takes
several minutes; demo.py itself always uses the fast path.
"""
import time
import numpy as np
from spatial_boost.simulate import simulate_dataset
from spatial_boost.model import fit_em


def logit(p):
    return np.log(p) - np.log(1.0 - p)


if __name__ == "__main__":
    n, p, m_causal = 200, 20_000, 10
    data = simulate_dataset(n=n, p=p, n_genes=800, scenario="informative", m_causal=m_causal, seed=5)
    xi0 = logit(m_causal / p)

    t0 = time.time()
    res_trunc = fit_em(data.X, data.y, data.wr, xi0=xi0, xi1=3.0, kappa=100.0, nu=1.0, lam=1.0, rank=n)
    t_trunc = time.time() - t0
    print(f"rank-truncated (rank={n}) EM fit, p={p}: {t_trunc:.2f}s ({res_trunc.n_iter} iterations)")

    t0 = time.time()
    res_dense = fit_em(data.X, data.y, data.wr, xi0=xi0, xi1=3.0, kappa=100.0, nu=1.0, lam=1.0, rank=None)
    t_dense = time.time() - t0
    print(f"dense (p+1)x(p+1) EM fit, p={p}: {t_dense:.2f}s ({res_dense.n_iter} iterations)")

    rel_err = np.linalg.norm(res_trunc.beta - res_dense.beta) / np.linalg.norm(res_dense.beta)
    print(f"relative difference in fitted beta: {rel_err:.2e}  (rank=n reconstructs X exactly, "
          f"so this is numerical noise, not approximation error)")
    print(f"speedup: {t_dense / t_trunc:.1f}x")
