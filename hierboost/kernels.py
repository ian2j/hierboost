"""Feature <-> group affinity kernels.

Generalizes the gene-proximity weights of Johnston et al. (2016, Sec 3.1-3.2) from
"SNPs near genes on a chromosome" to "features near groups in any coordinate space,
graph, or precomputed affinity structure." Stripped of genomics, the idea is just:
each feature gets a prior-inclusion boost proportional to its affinity to groups,
weighted by each group's relevance to the outcome. hierboost.spike_slab only ever
consumes the combined score `wr = normalize(affinity @ relevance)`.
"""
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize_scalar


def combine_affinity_with_relevance(affinity, relevance, normalize=True):
    """wr_j = sum_g affinity[j,g] * relevance[g], optionally rescaled so max(wr) = 1
    (Sec 3.1 precaution 2: keeps xi1 estimates comparable across affinity/relevance schemes)."""
    wr = affinity @ np.asarray(relevance, dtype=float)
    if normalize:
        m = wr.max()
        if m > 0:
            wr = wr / m
    return wr


def _as_1d_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1:
        raise ValueError("gaussian_affinity_1d requires 1D coordinates; use gaussian_affinity_points "
                          "for multi-dimensional coordinates (embeddings, lat/lon, ...)")
    return coords


def gaussian_affinity_1d(feature_coords, group_l, group_r, bandwidth, window_sigma=8.0):
    """P(N(feature_coord, bandwidth^2) in [group_l, group_r]) for every (feature, group) pair.

    This is the exact generalization of the paper's Gaussian gene-weight kernel to any 1D
    coordinate (genomic position, sequence index, timestamp, document offset, ...). `bandwidth`
    may be a scalar (shared) or a per-feature array (e.g. from a region-adaptive fit, see
    fit_gaussian_bandwidth below).

    Only feature/group pairs within window_sigma*bandwidth of each other are evaluated --
    a Gaussian's mass beyond that is negligible -- keeping cost near O(p * local_group_count)
    even with many thousands of groups.
    """
    feature_coords = _as_1d_coords(feature_coords)
    group_l = np.asarray(group_l, dtype=float)
    group_r = np.asarray(group_r, dtype=float)
    p, G = feature_coords.shape[0], group_l.shape[0]

    bw = np.asarray(bandwidth, dtype=float)
    bw = np.full(p, bw) if bw.ndim == 0 else bw

    group_mid = 0.5 * (group_l + group_r)
    order = np.argsort(group_mid)
    mid_s, l_s, r_s = group_mid[order], group_l[order], group_r[order]

    affinity = np.zeros((p, G))
    for j in range(p):
        bwj = bw[j]
        if bwj <= 0:
            continue
        lo = np.searchsorted(mid_s, feature_coords[j] - window_sigma * bwj, side="left")
        hi = np.searchsorted(mid_s, feature_coords[j] + window_sigma * bwj, side="right")
        if hi <= lo:
            continue
        a = (l_s[lo:hi] - feature_coords[j]) / bwj
        b = (r_s[lo:hi] - feature_coords[j]) / bwj
        affinity[j, order[lo:hi]] = ndtr(b) - ndtr(a)
    return affinity


def gaussian_affinity_points(feature_coords, group_coords, bandwidth):
    """Point-to-point Gaussian kernel affinity for groups represented by a centroid rather
    than an extent -- works in any dimension (text/graph embeddings, lat/lon, PCA space, ...),
    unlike gaussian_affinity_1d which needs a 1D interval to integrate over.
    """
    feature_coords = np.asarray(feature_coords, dtype=float)
    if feature_coords.ndim == 1:
        feature_coords = feature_coords[:, None]
    group_coords = np.asarray(group_coords, dtype=float)
    if group_coords.ndim == 1:
        group_coords = group_coords[:, None]

    sq_dist = ((feature_coords[:, None, :] - group_coords[None, :, :]) ** 2).sum(axis=-1)
    bw = np.asarray(bandwidth, dtype=float)
    if bw.ndim == 0:
        return np.exp(-0.5 * sq_dist / bw ** 2)
    return np.exp(-0.5 * sq_dist / bw[:, None] ** 2)


def causal_affinity_1d(feature_time, group_l, group_r, bandwidth):
    """One-sided exponential-decay analogue of gaussian_affinity_1d, for coordinates that
    are actually a timeline rather than a genomic position. gaussian_affinity_1d is
    deliberately symmetric -- a SNP can sit on either side of a gene and still be boosted,
    since genomic proximity carries no direction. Time does carry a direction: a feature
    observed as of `feature_time` may only be boosted by a group (an event window, a known
    regime) that has *already resolved* by then (group_r <= feature_time). Using a group
    that starts or ends after feature_time would leak future information into today's prior
    -- exactly the look-ahead-bias failure mode a backtest cannot tolerate. Affinity decays
    like exp(-elapsed/bandwidth) in the time since the group resolved; groups still in
    progress or in the future get exactly zero affinity, no partial credit.
    """
    feature_time = _as_1d_coords(feature_time)
    group_l = np.asarray(group_l, dtype=float)
    group_r = np.asarray(group_r, dtype=float)

    bw = np.asarray(bandwidth, dtype=float)
    bw = np.full(feature_time.shape[0], bw) if bw.ndim == 0 else bw

    elapsed = feature_time[:, None] - group_r[None, :]
    resolved = elapsed >= 0.0
    affinity = np.where(resolved, np.exp(-elapsed / np.where(bw[:, None] > 0, bw[:, None], 1.0)), 0.0)
    affinity[bw <= 0] = 0.0
    return affinity


def graph_affinity(distance_matrix, bandwidth):
    """Kernel(graph distance) affinity for graph-structured domains (ontologies, knowledge
    graphs, citation/social networks, ...), where `distance_matrix` is a precomputed p x G
    shortest-path or dissimilarity matrix (e.g. scipy.sparse.csgraph.shortest_path)."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    bw = np.asarray(bandwidth, dtype=float)
    if bw.ndim == 0:
        return np.exp(-distance_matrix / bw)
    return np.exp(-distance_matrix / bw[:, None])


def sar_weight_matrix(coords, bandwidth, block_id=None):
    """Simultaneous-autoregressive spatial weight matrix (dissertation Ch4 Sec 4.1.2,
    Eq. 4.1-4.2): B_jk = Phi(-|s_j-s_k|/phi_j) + Phi(-|s_j-s_k|/phi_k) for j != k, 0 on
    the diagonal. `bandwidth` (phi) may be scalar or per-feature. If `block_id` is given,
    cross-block entries are zeroed so B (and hence C=(I-B)^-1) is block-diagonal -- the
    condition the dissertation recommends (phi_j <= zeta/3) for a tractable, block-wise fit.

    `coords` may be 1D (genomic position, a scalar synthetic axis, ...) or 2D (p, d) for a
    real multi-dimensional physical location (lon/lat, embeddings, ...) -- |s_j-s_k| is then
    Euclidean distance, the same generalization gaussian_affinity_points already applies to
    the boosting-prior kernel. 1D coords are reshaped to (p, 1) internally, so the formula
    and its output are identical to the pre-generalization scalar-|s_j-s_k| version.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    elif coords.ndim != 2:
        raise ValueError("sar_weight_matrix requires 1D or 2D (p, d) coordinates")
    p = coords.shape[0]
    bw = np.asarray(bandwidth, dtype=float)
    bw = np.full(p, bw) if bw.ndim == 0 else bw

    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    B = ndtr(-d / bw[:, None]) + ndtr(-d / bw[None, :])
    np.fill_diagonal(B, 0.0)
    if block_id is not None:
        block_id = np.asarray(block_id)
        same_block = block_id[:, None] == block_id[None, :]
        B = B * same_block
    return B


def ar1_weight_matrix(time_index, correlation_length, block_id=None):
    """Temporal counterpart of sar_weight_matrix: an Ornstein-Uhlenbeck/AR(1) weight
    matrix B_jk = rho(|t_j-t_k|) = exp(-|t_j-t_k|/correlation_length), 0 on the diagonal.

    sar_weight_matrix's B_jk = Phi(-d/phi_j)+Phi(-d/phi_k) is a symmetric weight decaying
    with *undirected* physical distance -- appropriate for a block of SNPs in LD, where
    neither member is "before" the other. Within a temporal block (e.g. several lagged or
    rolling-window versions of the same signal -- lag-1, lag-5, lag-20 returns -- which are
    exactly as collinear as LD-correlated SNPs and for the same structural reason: they are
    noisy views of one underlying autocorrelated process), the natural weight is instead an
    AR(1)/OU correlation, which this reduces to at unit spacing. Feeding this B into
    (I - B)^-1 in place of the SAR B is what turns Chapter 4's spatial block-latent model
    into a temporal one -- see hierboost.latent's `structure="ar1"` option. `time_index` may
    be any 1D coordinate (integer lag order, or actual irregularly-spaced timestamps -- the
    exponential form handles gaps from weekends/holidays gracefully, unlike a fixed-lag AR(1)).
    `correlation_length` may be scalar or per-feature, in the same units as time_index.
    """
    time_index = _as_1d_coords(time_index)
    p = time_index.shape[0]
    cl = np.asarray(correlation_length, dtype=float)
    cl = np.full(p, cl) if cl.ndim == 0 else cl

    d = np.abs(time_index[:, None] - time_index[None, :])
    denom = np.sqrt(cl[:, None] * cl[None, :])
    B = np.where(denom > 0, np.exp(-d / np.where(denom > 0, denom, 1.0)), 0.0)
    np.fill_diagonal(B, 0.0)
    if block_id is not None:
        block_id = np.asarray(block_id)
        same_block = block_id[:, None] == block_id[None, :]
        B = B * same_block
    return B


def fit_gaussian_bandwidth(coords, affinity_abs, bandwidth_bounds=(1.0, 1e8)):
    """Fit a single bandwidth phi so that 2*Phi(-|s_i-s_j|/phi) best matches (by MSE) an
    observed pairwise affinity magnitude `affinity_abs` (e.g. |correlation|) as a function
    of 1D coordinate distance. Generalizes the paper's Sec 5.1 LD-decay fitting procedure to
    any domain where "nearby features are more related" -- co-occurring words, adjacent
    time steps, neighboring sensors, etc.
    """
    coords = _as_1d_coords(coords)
    iu = np.triu_indices(len(coords), k=1)
    d = np.abs(coords[iu[0]] - coords[iu[1]])
    target = np.asarray(affinity_abs)[iu]

    def mse(log_phi):
        phi = np.exp(log_phi)
        pred = 2.0 * ndtr(-d / phi)
        return np.mean((pred - target) ** 2)

    lo, hi = bandwidth_bounds
    res = minimize_scalar(mse, bounds=(np.log(lo), np.log(hi)), method="bounded")
    return float(np.exp(res.x))


def resolve_affinity(feature_coords, kind="gaussian", group_l=None, group_r=None,
                      group_coords=None, distance_matrix=None, relevance=None,
                      bandwidth=None, X=None, normalize=True):
    """Single entry point turning coordinates + relevance into the per-feature boost
    score `wr` hierboost.spike_slab consumes -- covering both "I know the bandwidth" and
    "auto-fit it from data" (this project's fit_gaussian_bandwidth, Sec 5.1's LD-decay
    fit generalized) so a practitioner without a hand-picked phi in hand isn't stuck.

    kind="gaussian": (feature_coords, group_l, group_r) -> gaussian_affinity_1d, or
      (feature_coords, group_coords) -> gaussian_affinity_points, whichever pair of group
      args is given.
    kind="causal": (feature_coords, group_l, group_r) -> causal_affinity_1d -- temporal,
      one-sided, no look-ahead (see its docstring).
    kind="graph": distance_matrix -> graph_affinity.

    If bandwidth is None and kind is "gaussian"/"causal" with 1D feature_coords, it is
    auto-fit via fit_gaussian_bandwidth against |corr(X)| when a raw feature matrix `X`
    is supplied, falling back to the median feature spacing otherwise (a coarse but
    always-available default). If relevance is None, every group is treated as equally
    relevant (uniform boost, driven by affinity alone).
    """
    coords_arr = np.asarray(feature_coords, dtype=float)
    is_1d = coords_arr.ndim == 1

    if kind in ("gaussian", "causal") and bandwidth is None:
        if is_1d and X is not None:
            corr_abs = np.abs(np.nan_to_num(np.corrcoef(np.asarray(X).T), nan=0.0))
            bandwidth = fit_gaussian_bandwidth(coords_arr, corr_abs)
        elif is_1d and coords_arr.size > 1:
            gaps = np.diff(np.sort(coords_arr))
            bandwidth = float(np.median(gaps)) if np.any(gaps > 0) else 1.0
        else:
            bandwidth = 1.0

    if kind == "gaussian":
        if group_l is not None and group_r is not None:
            affinity = gaussian_affinity_1d(feature_coords, group_l, group_r, bandwidth)
        elif group_coords is not None:
            affinity = gaussian_affinity_points(feature_coords, group_coords, bandwidth)
        else:
            raise ValueError("kind='gaussian' requires (group_l, group_r) or group_coords")
    elif kind == "causal":
        if group_l is None or group_r is None:
            raise ValueError("kind='causal' requires group_l and group_r (group time extents)")
        affinity = causal_affinity_1d(feature_coords, group_l, group_r, bandwidth)
    elif kind == "graph":
        if distance_matrix is None:
            raise ValueError("kind='graph' requires distance_matrix")
        if bandwidth is None:
            off_diag = distance_matrix[np.asarray(distance_matrix) > 0]
            bandwidth = float(np.median(off_diag)) if off_diag.size else 1.0
        affinity = graph_affinity(distance_matrix, bandwidth)
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from 'gaussian', 'causal', 'graph'")

    relevance = np.ones(affinity.shape[1]) if relevance is None else np.asarray(relevance, dtype=float)
    return combine_affinity_with_relevance(affinity, relevance, normalize=normalize)
