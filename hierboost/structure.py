"""Data-driven block/graph structure, as an alternative to the fixed distance
threshold zeta used in hierboost.blocks.threshold_blocks_1d (dissertation Ch4
Sec 4.1.1: "in practice I propose selecting a value of zeta... that strikes a
balance"). A fixed distance cutoff only ever groups features that happen to sit
close together on one coordinate; these functions instead group features by
their actual observed statistical relationship, which need not respect that
coordinate at all (e.g. two features far apart on a chromosome but in strong LD
through a shared regulatory mechanism, or two document features that co-occur
without being adjacent in the text).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .blocks import threshold_blocks_1d, threshold_blocks_graph


def blocks_from_kmeans(coords, voxels_per_block=8, n_blocks=None, seed=0):
    """K-means partition of multi-dimensional point coordinates into compact spatial
    groups -- the right tool when features form one dense, contiguous point cloud with
    no natural gaps to threshold at (pixel grids, voxel masks, geographic centroids),
    unlike threshold_blocks_1d/_graph, which assume gaps exist (genes strung along a
    chromosome, sensors scattered over a field) and silently collapse a solid blob into
    one giant connected component instead (confirmed empirically fitting Haxby fMRI
    voxels: threshold_graph merged all 464 VT-cortex voxels into a single block, since
    every voxel touches a neighbor at any radius >= one voxel spacing). K-means has no
    such failure mode because it partitions by nearest-centroid, not graph connectivity.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    if n_blocks is None:
        n_blocks = max(2, coords.shape[0] // voxels_per_block)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(coords)


def determine_blocks(X=None, coords=None, method="threshold", zeta=None, rho=0.3,
                      alpha="cv", max_features=200, gap_percentile=75,
                      voxels_per_block=8, n_blocks=None, seed=0):
    """Single entry point for turning raw features into a block_id array, covering both
    "I have a rule" (a fixed zeta/rho/alpha) and "just pick something sensible" (leave the
    threshold unset) -- the latter is what a practitioner without a genomic zeta or a
    pre-built graph in hand actually needs.

    method="threshold": 1D distance-cutoff blocking (threshold_blocks_1d, Ch4 Sec 4.1.1).
      If zeta is None, it defaults to the `gap_percentile`-th percentile of the sorted
      coordinate's consecutive gaps -- a "natural breaks" default that blocks at
      unusually large gaps instead of requiring a hand-picked cutoff.
    method="threshold_graph": the graph generalization (threshold_blocks_graph) -- pass a
      precomputed pairwise distance matrix as `coords`; zeta defaults the same way, over
      the matrix's off-diagonal entries.
    method="correlation": blocks_from_correlation_threshold(X, rho) -- purely data-driven,
      no coordinate needed at all.
    method="graphical_lasso": blocks_from_graphical_lasso(X, alpha, max_features).
    method="kmeans": blocks_from_kmeans(coords, voxels_per_block, n_blocks) -- for a dense,
      contiguous multi-D point cloud with no natural gaps (pixel/voxel grids, geographic
      centroids), where threshold/threshold_graph would collapse into one giant block.
      Pass raw multi-D `coords` directly (not a distance matrix); `n_blocks` overrides the
      `voxels_per_block`-derived default.
    """
    if method == "threshold":
        if coords is None:
            raise ValueError("method='threshold' requires coords")
        coords = np.asarray(coords, dtype=float)
        if zeta is None:
            gaps = np.diff(np.sort(coords))
            zeta = float(np.percentile(gaps, gap_percentile)) if gaps.size else 1.0
        return threshold_blocks_1d(coords, zeta)
    if method == "threshold_graph":
        if coords is None:
            raise ValueError("method='threshold_graph' requires a precomputed distance matrix as coords")
        D = np.asarray(coords, dtype=float)
        if zeta is None:
            iu = np.triu_indices_from(D, k=1)
            zeta = float(np.percentile(D[iu], gap_percentile)) if iu[0].size else 1.0
        return threshold_blocks_graph(D, zeta)
    if method == "correlation":
        if X is None:
            raise ValueError("method='correlation' requires X")
        return blocks_from_correlation_threshold(X, rho=rho)
    if method == "graphical_lasso":
        if X is None:
            raise ValueError("method='graphical_lasso' requires X")
        return blocks_from_graphical_lasso(X, alpha=alpha, max_features=max_features)
    if method == "kmeans":
        if coords is None:
            raise ValueError("method='kmeans' requires coords (raw multi-D point coordinates)")
        return blocks_from_kmeans(coords, voxels_per_block=voxels_per_block, n_blocks=n_blocks, seed=seed)
    raise ValueError(f"unknown method {method!r}; choose from "
                      f"'threshold', 'threshold_graph', 'correlation', 'graphical_lasso', 'kmeans'")


def blocks_from_correlation_threshold(X, rho=0.3):
    """Connected components of the feature-correlation graph thresholded at |corr| >= rho.
    The simplest data-driven alternative to a fixed distance cutoff."""
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0.0)
    adj = corr >= rho
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def blocks_from_graphical_lasso(X, alpha="cv", max_features=200):
    """Learn a sparse conditional-independence graph via graphical lasso and take its
    connected components as blocks. More principled than raw correlation thresholding:
    edges reflect *partial* correlation, so two features that are only related through a
    third (transitively correlated, not directly) don't spuriously merge into one block.
    Solvable only for a moderate number of features at once (O(p^3) per fit) -- for
    genome-scale p, first coarsen with threshold_blocks_1d and refine each resulting
    region separately.
    """
    from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

    if X.shape[1] > max_features:
        raise ValueError(f"graphical lasso is O(p^3); pass <= {max_features} features at a "
                          f"time (e.g. pre-split with hierboost.blocks.threshold_blocks_1d "
                          f"and refine each region separately)")
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    model = (GraphicalLassoCV(max_iter=200) if alpha == "cv"
              else GraphicalLasso(alpha=alpha, max_iter=200)).fit(Xs)

    adj = np.abs(model.precision_) > 1e-8
    np.fill_diagonal(adj, False)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels
