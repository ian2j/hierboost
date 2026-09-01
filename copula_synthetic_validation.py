"""Controlled synthetic check that the Gaussian-copula marginal transform (hierboost.copula)
actually buys something over feeding raw, heterogeneous-marginal features straight into
factor.py's gaussian_block_factor (probabilistic PCA), before trusting it on real data --
same discipline as the project's other validated fixes (e.g. the float32/Newton-damping
bugs in genomics_1kg_binomial_latent.py: sanity-test the mechanism on a case where the
right answer is known, first).

Construction: one shared latent factor z (n,) drives m=5 features on a common Gaussian
scale (x_j = loading_j * z + idiosyncratic noise_j, exactly gaussian_block_factor's own
generative assumption). Each feature is then pushed through its OWN monotone marginal
distortion -- identity, exp (lognormal, e.g. trading-volume-like), cube (heavy-tailed,
sign-preserving), a Gamma quantile-map, and a logistic squash (bounded, Beta-like) --
before being handed to the two competing pipelines. Monotone transforms preserve rank
correlation and the copula exactly, so the TRUE shared-factor structure is identical in
both cases; only each column's own marginal shape has been scrambled. This isolates
exactly the failure mode the copula extension targets.
"""
import numpy as np
from scipy import stats

from hierboost.factor import gaussian_block_factor
from hierboost.copula import to_gaussian_scale


def make_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    loadings_true = np.array([1.2, -0.9, 1.5, 0.8, -1.1])
    noise_sd = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    X_gauss = z[:, None] * loadings_true[None, :] + rng.normal(size=(n, 5)) * noise_sd[None, :]

    X_raw = np.empty_like(X_gauss)
    X_raw[:, 0] = X_gauss[:, 0]                                   # identity (already Gaussian)
    X_raw[:, 1] = np.exp(X_gauss[:, 1])                            # lognormal, e.g. volume-like
    X_raw[:, 2] = X_gauss[:, 2] ** 3                                # heavy-tailed, sign-preserving
    u3 = stats.norm.cdf(X_gauss[:, 3])
    X_raw[:, 3] = stats.gamma.ppf(u3, a=2.0, scale=1.5)             # Gamma marginal
    X_raw[:, 4] = 1.0 / (1.0 + np.exp(-X_gauss[:, 4]))              # bounded, Beta-like

    names = ["identity", "lognormal", "cubic-heavy-tail", "gamma", "logistic-bounded"]
    return z, X_raw, names


def corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    z_true, X_raw, names = make_data()

    print("Marginal shapes fed to both pipelines (skewness of each raw column):")
    for j, name in enumerate(names):
        print(f"  {name:20s} skew={stats.skew(X_raw[:, j]):+.2f}")

    scores_raw, loadings_raw = gaussian_block_factor(X_raw)
    Z_copula, transforms = to_gaussian_scale(X_raw, kind="empirical")
    scores_copula, loadings_copula = gaussian_block_factor(Z_copula)

    r_raw = corr(scores_raw, z_true)
    r_copula = corr(scores_copula, z_true)

    print(f"\ncorr(recovered factor, TRUE shared latent z):")
    print(f"  raw gaussian_block_factor(X)        : {r_raw:.4f}")
    print(f"  copula-transform + gaussian_block_factor(Z) : {r_copula:.4f}")
    print(f"  improvement: {r_copula - r_raw:+.4f}")

    # A second, harder regime: same construction but with EXTREME distortions (the
    # gamma/lognormal features get a heavier tail), where raw PPCA's linear/Euclidean
    # assumption should break down more severely.
    def make_extreme(n=2000, seed=1):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n)
        loadings_true = np.array([1.2, -0.9, 1.5, 0.8, -1.1])
        X_gauss = z[:, None] * loadings_true[None, :] + rng.normal(size=(n, 5)) * 0.6
        X_raw = np.empty_like(X_gauss)
        X_raw[:, 0] = X_gauss[:, 0]
        X_raw[:, 1] = np.exp(2.0 * X_gauss[:, 1])                   # more extreme lognormal
        X_raw[:, 2] = X_gauss[:, 2] ** 5                             # more extreme heavy tail
        u3 = stats.norm.cdf(X_gauss[:, 3])
        X_raw[:, 3] = stats.gamma.ppf(u3, a=0.5, scale=1.5)          # heavier-tailed gamma
        X_raw[:, 4] = 1.0 / (1.0 + np.exp(-3.0 * X_gauss[:, 4]))     # sharper squash
        return z, X_raw

    z_true2, X_raw2 = make_extreme()
    scores_raw2, _ = gaussian_block_factor(X_raw2)
    Z_copula2, _ = to_gaussian_scale(X_raw2, kind="empirical")
    scores_copula2, _ = gaussian_block_factor(Z_copula2)
    r_raw2 = corr(scores_raw2, z_true2)
    r_copula2 = corr(scores_copula2, z_true2)
    print(f"\nExtreme-distortion regime:")
    print(f"  raw gaussian_block_factor(X)        : {r_raw2:.4f}")
    print(f"  copula-transform + gaussian_block_factor(Z) : {r_copula2:.4f}")
    print(f"  improvement: {r_copula2 - r_raw2:+.4f}")
