"""Data simulator matching the Spatial Boost model's generative story (Sec 6.1):
real-ish LD structure, gene annotations, gene-proximity-weighted causal markers,
small non-causal + larger causal effect sizes, and a logistic-regression phenotype.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.stats import norm

from .weights import build_gene_blocks, gene_weights


def simulate_ld_haplotype(n, positions, ld_length, rng):
    """AR(1)-in-space latent Gaussian: Corr(z_i, z_j) = exp(-|pos_i-pos_j|/ld_length).

    A sequential (Markov) construction lets LD decay be genomewide-realistic while
    costing O(n*p) instead of the O(p^3) a dense multivariate-normal draw would need.
    """
    positions = np.asarray(positions, dtype=float)
    p = positions.shape[0]
    order = np.argsort(positions)
    pos_sorted = positions[order]
    rho = np.exp(-np.diff(pos_sorted) / ld_length)

    z = np.empty((n, p))
    z[:, 0] = rng.standard_normal(n)
    eps = rng.standard_normal((n, p - 1))
    for j in range(1, p):
        z[:, j] = rho[j - 1] * z[:, j - 1] + np.sqrt(1.0 - rho[j - 1] ** 2) * eps[:, j - 1]

    z_unsorted = np.empty_like(z)
    z_unsorted[:, order] = z
    return z_unsorted


def simulate_genotypes(n, positions, maf, ld_length, rng):
    """Additive 0/1/2 genotypes from two independent LD-correlated haplotypes."""
    thresh = norm.ppf(1.0 - maf)
    h1 = (simulate_ld_haplotype(n, positions, ld_length, rng) > thresh[None, :]).astype(int)
    h2 = (simulate_ld_haplotype(n, positions, ld_length, rng) > thresh[None, :]).astype(int)
    return h1 + h2


@dataclass
class SimulatedDataset:
    X: np.ndarray                  # n x (p+1), column 0 is the intercept
    y: np.ndarray
    positions: np.ndarray
    gene_starts: np.ndarray
    gene_ends: np.ndarray
    gene_relevance: np.ndarray
    wr: np.ndarray
    phi: float
    theta_true: np.ndarray
    beta_true: np.ndarray
    maf: np.ndarray


def simulate_dataset(n=200, p=3000, n_genes=150, chrom_length=5_000_000,
                      scenario="informative", m_causal=10, ld_length=20_000,
                      xi0=None, xi1=None, phi=None, causal_var=0.25, noise_var=0.01,
                      seed=0):
    """scenario in {"informative", "non_informative"}, mirroring Sec 6.1's two setups."""
    rng = np.random.default_rng(seed)

    positions = np.sort(rng.uniform(0, chrom_length, p))
    gene_len = np.clip(rng.exponential(30_000, n_genes), 1_000, 200_000)
    gene_starts = rng.uniform(0, chrom_length, n_genes)
    gene_ends = gene_starts + gene_len

    if scenario == "informative":
        gene_relevance = rng.lognormal(mean=0.0, sigma=1.5, size=n_genes)
        phi = 10_000.0 if phi is None else phi
        xi1 = 5.0 if xi1 is None else xi1
    elif scenario == "non_informative":
        gene_relevance = np.ones(n_genes)
        phi = chrom_length if phi is None else phi
        xi1 = 1.0 if xi1 is None else xi1
    else:
        raise ValueError("scenario must be 'informative' or 'non_informative'")

    xi0 = float(np.floor(np.log(m_causal / p) - np.log(1 - m_causal / p))) if xi0 is None else xi0

    block_l, block_r, block_rel = build_gene_blocks(gene_starts, gene_ends, gene_relevance)
    wr = gene_weights(positions, block_l, block_r, block_rel, phi)

    maf = rng.uniform(0.05, 0.5, p)
    X_markers = simulate_genotypes(n, positions, maf, ld_length, rng)

    prior_prob = np.clip(expit(xi0 + xi1 * wr), 1e-8, 1 - 1e-8)
    causal_idx = rng.choice(p, size=m_causal, replace=False, p=prior_prob / prior_prob.sum())
    theta_true = np.zeros(p, dtype=bool)
    theta_true[causal_idx] = True

    beta_markers = np.where(theta_true,
                             rng.normal(0.0, np.sqrt(causal_var), p),
                             rng.normal(0.0, np.sqrt(noise_var), p))
    beta_true = np.concatenate([[0.0], beta_markers])

    X = np.column_stack([np.ones(n), X_markers]).astype(float)
    prob = expit(X @ beta_true)
    y = (rng.random(n) < prob).astype(float)

    return SimulatedDataset(X=X, y=y, positions=positions, gene_starts=gene_starts,
                             gene_ends=gene_ends, gene_relevance=gene_relevance, wr=wr,
                             phi=phi, theta_true=theta_true, beta_true=beta_true, maf=maf)
