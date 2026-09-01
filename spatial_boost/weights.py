"""GWAS-flavored gene-proximity weights: a thin wrapper around hierboost's generic
feature<->group affinity kernels (see hierboost/kernels.py, hierboost/blocks.py).
"""
import numpy as np

from hierboost.blocks import merge_overlapping_extents, threshold_blocks_1d
from hierboost.kernels import gaussian_affinity_1d, combine_affinity_with_relevance, fit_gaussian_bandwidth


def build_gene_blocks(gene_starts, gene_ends, gene_relevance):
    """Partition overlapping genes into non-overlapping blocks (Sec 3.1, precaution 1)."""
    return merge_overlapping_extents(gene_starts, gene_ends, gene_relevance)


def gene_weights(marker_pos, block_l, block_r, block_relevance, phi, window_sigma=8.0):
    """Normalized w_j^T r for every marker (Sec 3.1, normalized per precaution 2).

    phi may be a scalar (genome-wide) or an array of length p (per-marker/per-region,
    as produced by select_phi_by_region below).
    """
    affinity = gaussian_affinity_1d(marker_pos, block_l, block_r, phi, window_sigma=window_sigma)
    return combine_affinity_with_relevance(affinity, block_relevance, normalize=True)


def select_phi_by_region(positions, genotypes, min_gap=30_000):
    """Replicates Sec 5.1: segment the chromosome into LD-adaptive regions, fit one
    phi per region from the region's observed genotype correlation decay.

    Returns an array of length p (one phi value per marker, constant within a region).
    """
    positions = np.asarray(positions, dtype=float)
    p = positions.shape[0]
    region_id = threshold_blocks_1d(positions, zeta=min_gap)

    phi = np.empty(p)
    for r in np.unique(region_id):
        members = np.where(region_id == r)[0]
        if members.shape[0] < 3:
            phi[members] = min_gap
            continue
        corr = np.corrcoef(genotypes[:, members].T)
        corr = np.nan_to_num(corr, nan=0.0)
        phi[members] = fit_gaussian_bandwidth(positions[members], np.abs(corr))
    return phi
