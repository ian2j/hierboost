"""Tests the original Spatial Boost gene-proximity prior on top of block-latent
decorrelation: gives LD blocks near a candidate region a boosted prior inclusion
probability (gaussian_affinity_1d), varying region width and boost strength. Uses the
true causal SNP as the region center -- see genomics_1kg_boost_offset_sensitivity.py
for robustness to a mis-centered guess."""
import argparse
import glob
import json
import os
import time

import numpy as np

from genomics_1kg_demo import (load_data, build_blocks, cross_validate_hierboost_only,
                                full_data_fit)
from hierboost.blocks import block_membership_lists
from hierboost.kernels import gaussian_affinity_1d, combine_affinity_with_relevance

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_boost_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_KAPPA = 100.0
DEFAULT_GAP_PCT = 75
DEFAULT_PRIOR_P = 0.15

REGION_HALFWIDTHS = [2_000, 15_000, 75_000]   # narrow / moderate / wide "dilution"
XI1_GRID = [2.0, 4.0, 8.0]                     # weak / moderate / strong boost


def load_loci():
    loci = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        label = d["label"]
        data_path = os.path.expanduser(f"~/genomics_1kg/{label.lower()}_region.npz")
        loci.append(dict(label=label, gene=d.get("gene") or label,
                          causal_pos=d["causal_pos"], causal_rsid=d["causal_rsid"],
                          data_path=data_path))
    return loci


def block_positions_in_order(positions, block_id):
    """block_ids order matching fit_block_factors's Ztr column order exactly."""
    blocks = block_membership_lists(block_id)
    block_ids = sorted(blocks.keys())
    pos = np.array([positions[blocks[b]].mean() for b in block_ids])
    return pos, block_ids


def region_wr(block_pos, center, halfwidth, bandwidth):
    """Soft-edged boost centered on [center-halfwidth, center+halfwidth]; bandwidth
    controls how blurry the region's edges are (gaussian_affinity_1d treats each
    block's own position as having Gaussian uncertainty ~bandwidth, so a bigger
    bandwidth lets the boost reach further past the region's nominal edges)."""
    group_l = np.array([center - halfwidth])
    group_r = np.array([center + halfwidth])
    affinity = gaussian_affinity_1d(block_pos, group_l, group_r, bandwidth)
    return combine_affinity_with_relevance(affinity, relevance=np.array([1.0]))


def top_block_distance(best, block_ids, block_id, positions, causal_pos):
    blocks = block_membership_lists(block_id)
    block_pos_map = {b: float(positions[blocks[b]].mean()) for b in blocks}
    top_block, top_theta = None, -1.0
    for local_k, global_k in enumerate(best.retained_idx):
        theta = float(best.theta_hat[local_k])
        if theta > top_theta:
            top_theta = theta
            top_block = block_ids[global_k]
    top_pos = block_pos_map[top_block] if top_block is not None else None
    dist = abs(top_pos - causal_pos) if top_pos is not None else None
    return dist, top_theta


def run_setting(locus, halfwidth, xi1, seed=0):
    d = np.load(locus["data_path"])
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    block_id = build_blocks(positions, gap_percentile=DEFAULT_GAP_PCT)
    block_pos, block_ids = block_positions_in_order(positions, block_id)
    bandwidth = max(halfwidth * 0.3, 500.0)
    wr = region_wr(block_pos, locus["causal_pos"], halfwidth, bandwidth)
    xi0 = np.log(DEFAULT_PRIOR_P / (1 - DEFAULT_PRIOR_P))

    t0 = time.time()
    cv = cross_validate_hierboost_only(X, y, block_id, seed=seed, kappa=DEFAULT_KAPPA,
                                        xi0=xi0, wr=wr, xi1=xi1)
    best, full_block_ids = full_data_fit(X, y, block_id, kappa=DEFAULT_KAPPA, xi0=xi0,
                                          wr=wr, xi1=xi1)
    dist, top_theta = top_block_distance(best, full_block_ids, block_id, positions,
                                          locus["causal_pos"])
    elapsed = time.time() - t0

    row = dict(locus=locus["label"], gene=locus["gene"], halfwidth=halfwidth, xi1=xi1,
               bandwidth=bandwidth, acc_mean=cv["acc_mean"], acc_std=cv["acc_std"],
               n_features_cv=cv["n_features_mean"], dist_bp=dist, top_theta=top_theta,
               n_features_full=best.n_features, elapsed_sec=elapsed)
    print(f"{locus['label']:12s} halfwidth={halfwidth:<7d} xi1={xi1:<5g} "
          f"acc={cv['acc_mean']:.4f} n_feat={cv['n_features_mean']:.1f} "
          f"dist={dist:.0f}bp theta={top_theta:.3f} ({elapsed:.1f}s)")
    return row


def run_all(loci, seed=0):
    rows = []
    for locus in loci:
        for halfwidth in REGION_HALFWIDTHS:
            for xi1 in XI1_GRID:
                rows.append(run_setting(locus, halfwidth, xi1, seed=seed))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(OUT_DIR, "boost_sensitivity_results.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    loci = load_loci()
    print(f"{len(loci)} loci loaded from {RESULTS_DIR}")
    rows = run_all(loci, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} settings x locus results written to {args.out}")
