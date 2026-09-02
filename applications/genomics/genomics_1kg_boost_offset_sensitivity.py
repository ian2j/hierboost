"""Follow-up to genomics_1kg_boost_sensitivity.py: how far off can the boosted region's
center be from the true causal SNP and still help? Offsets the boost region by up to
+-60kb while measuring distance to the true SNP, across the same 3 region widths."""
import argparse
import json
import os
import time

import numpy as np

from genomics_1kg_demo import cross_validate_hierboost_only, full_data_fit, build_blocks
from genomics_1kg_boost_sensitivity import (load_loci, block_positions_in_order, region_wr,
                                             top_block_distance, DEFAULT_KAPPA,
                                             DEFAULT_GAP_PCT, DEFAULT_PRIOR_P,
                                             REGION_HALFWIDTHS, RESULTS_DIR)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_boost_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

XI1_FIXED = 4.0
OFFSETS = [-60_000, -30_000, -10_000, 10_000, 30_000, 60_000]  # signed, bp


def run_setting(locus, halfwidth, offset, xi1=XI1_FIXED, seed=0):
    d = np.load(locus["data_path"])
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    block_id = build_blocks(positions, gap_percentile=DEFAULT_GAP_PCT)
    block_pos, block_ids = block_positions_in_order(positions, block_id)
    bandwidth = max(halfwidth * 0.3, 500.0)
    guessed_center = locus["causal_pos"] + offset
    wr = region_wr(block_pos, guessed_center, halfwidth, bandwidth)
    xi0 = np.log(DEFAULT_PRIOR_P / (1 - DEFAULT_PRIOR_P))

    t0 = time.time()
    cv = cross_validate_hierboost_only(X, y, block_id, seed=seed, kappa=DEFAULT_KAPPA,
                                        xi0=xi0, wr=wr, xi1=xi1)
    best, full_block_ids = full_data_fit(X, y, block_id, kappa=DEFAULT_KAPPA, xi0=xi0,
                                          wr=wr, xi1=xi1)
    dist, top_theta = top_block_distance(best, full_block_ids, block_id, positions,
                                          locus["causal_pos"])  # true position, not guess
    elapsed = time.time() - t0

    row = dict(locus=locus["label"], gene=locus["gene"], halfwidth=halfwidth, xi1=xi1,
               offset=offset, guessed_center=guessed_center, acc_mean=cv["acc_mean"],
               acc_std=cv["acc_std"], n_features_cv=cv["n_features_mean"], dist_bp=dist,
               top_theta=top_theta, n_features_full=best.n_features, elapsed_sec=elapsed)
    print(f"{locus['label']:12s} halfwidth={halfwidth:<7d} offset={offset:<8d} "
          f"acc={cv['acc_mean']:.4f} dist={dist:.0f}bp theta={top_theta:.3f} ({elapsed:.1f}s)")
    return row


def run_all(loci, seed=0):
    rows = []
    for locus in loci:
        for halfwidth in REGION_HALFWIDTHS:
            for offset in OFFSETS:
                rows.append(run_setting(locus, halfwidth, offset, seed=seed))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(OUT_DIR, "boost_offset_results.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    loci = load_loci()
    print(f"{len(loci)} loci loaded from {RESULTS_DIR}")
    rows = run_all(loci, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} settings x locus results written to {args.out}")
