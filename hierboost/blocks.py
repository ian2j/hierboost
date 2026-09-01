"""Grouping raw features into blocks.

Two distinct operations that both show up in Johnston et al.'s work under the name
"blocks": (1) merging overlapping annotated groups (genes) into a non-overlapping
partition so a feature isn't double-boosted (Sec 3.1 precaution 1), and (2)
clustering raw *unannotated* features by proximity into contiguous blocks for
decorrelation (dissertation Ch4 Sec 4.1.1). Both generalize past 1D genomic
coordinates: (1) works for any extent-like annotation, (2) also has a graph form.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def merge_overlapping_extents(starts, ends, relevance):
    """Partition possibly-overlapping (start, end) extents into a non-overlapping set
    of blocks, each carrying the mean relevance of the extents that cover it. This is
    what prevents a feature from getting boosted twice by two overlapping groups.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    relevance = np.asarray(relevance, dtype=float)

    breakpoints = np.union1d(starts, ends)
    block_l = breakpoints[:-1]
    block_r = breakpoints[1:]
    mid = 0.5 * (block_l + block_r)

    covers = (mid[:, None] >= starts[None, :]) & (mid[:, None] < ends[None, :])
    n_cover = covers.sum(axis=1)
    block_relevance = np.zeros(block_l.shape[0])
    has_cover = n_cover > 0
    block_relevance[has_cover] = (covers[has_cover] @ relevance) / n_cover[has_cover]
    return block_l, block_r, block_relevance


def threshold_blocks_1d(coords, zeta):
    """Cluster features along a 1D coordinate into contiguous blocks: adjacent features
    within `zeta` units of each other share a block (dissertation Ch4 Sec 4.1.1). Returns
    an integer block-id array aligned to `coords`' original order.
    """
    coords = np.asarray(coords, dtype=float)
    order = np.argsort(coords)
    gaps = np.diff(coords[order])
    new_block = np.concatenate([[True], gaps >= zeta])
    block_id_sorted = np.cumsum(new_block) - 1
    block_id = np.empty_like(block_id_sorted)
    block_id[order] = block_id_sorted
    return block_id


def threshold_blocks_graph(distance_matrix, zeta):
    """Generalizes threshold_blocks_1d to arbitrary graphs: connected components of the
    graph obtained by keeping feature-feature edges with distance <= zeta. Use this when
    "adjacency" isn't a single ordered coordinate (e.g. a knowledge graph, a correlation
    graph, spatial neighbors in 2D)."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    adj = (distance_matrix <= zeta) & (distance_matrix > 0)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def block_membership_lists(block_id):
    """block_id (p,) -> dict {block: sorted array of feature indices in that block}."""
    blocks = {}
    for j, b in enumerate(block_id):
        blocks.setdefault(int(b), []).append(j)
    return {b: np.array(idx) for b, idx in blocks.items()}
