"""Hyperparameter sensitivity sweep (kappa, block granularity, prior inclusion
probability) across the same 10 loci used in the paper, one-at-a-time around the default
operating point -- does tuning actually change performance/localization on real data?"""
import argparse
import glob
import json
import os
import time

import numpy as np

from genomics_1kg_demo import (load_data, build_blocks, cross_validate_hierboost_only,
                                full_data_fit)
from hierboost.blocks import block_membership_lists

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
SENS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_sensitivity")
os.makedirs(SENS_DIR, exist_ok=True)

DEFAULT_KAPPA = 100.0
DEFAULT_GAP_PCT = 75
DEFAULT_PRIOR_P = 0.15

KAPPA_GRID = [10.0, 100.0, 1000.0, 10000.0]
GAP_PCT_GRID = [50, 75, 90]
PRIOR_P_GRID = [0.05, 0.15, 0.30]


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


def block_purity(X, idx):
    """SuSiE-style purity of one block: min |pairwise correlation| among its member
    SNPs. Singleton blocks are trivially pure (SuSiE's own convention). Ground-truth
    free -- uses only the genotype matrix and the block's own membership, never y or
    the causal SNP position."""
    if len(idx) <= 1:
        return 1.0
    sub = X[:, idx]
    with np.errstate(invalid="ignore"):
        R = np.corrcoef(sub, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    n = R.shape[0]
    off_diag = np.abs(R)[~np.eye(n, dtype=bool)]
    return float(off_diag.min())


def top_block_distance(X, y, block_id, positions, causal_pos, kappa, xi0, gap_percentile):
    best, block_ids = full_data_fit(X, y, block_id, kappa=kappa, xi0=xi0)
    blocks = block_membership_lists(block_id)
    block_pos = {b: float(positions[blocks[b]].mean()) for b in blocks}
    top_block, top_theta = None, -1.0
    theta_sum, purity_weighted_sum = 0.0, 0.0
    for local_k, global_k in enumerate(best.retained_idx):
        theta = float(best.theta_hat[local_k])
        b_id = block_ids[global_k]
        purity = block_purity(X, blocks[b_id])
        theta_sum += theta
        purity_weighted_sum += theta * purity
        if theta > top_theta:
            top_theta = theta
            top_block = b_id
    top_pos = block_pos[top_block] if top_block is not None else None
    dist = abs(top_pos - causal_pos) if top_pos is not None else None
    top_purity = block_purity(X, blocks[top_block]) if top_block is not None else None
    top_block_size = len(blocks[top_block]) if top_block is not None else None
    weighted_purity = (purity_weighted_sum / theta_sum) if theta_sum > 0 else None
    return dist, top_theta, best.n_features, top_purity, weighted_purity, top_block_size


def run_setting(locus, kappa, gap_percentile, prior_p, tag, seed=0):
    xi0 = np.log(prior_p / (1 - prior_p))
    d = np.load(locus["data_path"])
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    block_id = build_blocks(positions, gap_percentile=gap_percentile)
    n_blocks = len(np.unique(block_id))

    t0 = time.time()
    cv = cross_validate_hierboost_only(X, y, block_id, seed=seed, kappa=kappa, xi0=xi0)
    dist, top_theta, n_full, top_purity, weighted_purity, top_block_size = top_block_distance(
        X, y, block_id, positions, locus["causal_pos"], kappa, xi0, gap_percentile)
    elapsed = time.time() - t0

    row = dict(locus=locus["label"], gene=locus["gene"], sweep=tag,
               kappa=kappa, gap_percentile=gap_percentile, prior_p=prior_p,
               n_blocks=n_blocks, acc_mean=cv["acc_mean"], acc_std=cv["acc_std"],
               n_features_cv=cv["n_features_mean"], dist_bp=dist, top_theta=top_theta,
               n_features_full=n_full, top_purity=top_purity,
               weighted_purity=weighted_purity, top_block_size=top_block_size,
               elapsed_sec=elapsed)
    print(f"{locus['label']:12s} {tag:10s} kappa={kappa:<8g} gap_pct={gap_percentile:<4d} "
          f"prior_p={prior_p:<5g} acc={cv['acc_mean']:.4f} n_feat={cv['n_features_mean']:.1f} "
          f"dist={dist:.0f}bp theta={top_theta:.3f} purity={top_purity:.3f} ({elapsed:.1f}s)")
    return row


def run_all(loci, seed=0):
    rows = []
    for locus in loci:
        # default point (shared baseline across all three sweeps, fit once)
        rows.append(run_setting(locus, DEFAULT_KAPPA, DEFAULT_GAP_PCT, DEFAULT_PRIOR_P,
                                 "default", seed=seed))
        for kappa in KAPPA_GRID:
            if kappa == DEFAULT_KAPPA:
                continue
            rows.append(run_setting(locus, kappa, DEFAULT_GAP_PCT, DEFAULT_PRIOR_P,
                                     "kappa", seed=seed))
        for gap_pct in GAP_PCT_GRID:
            if gap_pct == DEFAULT_GAP_PCT:
                continue
            rows.append(run_setting(locus, DEFAULT_KAPPA, gap_pct, DEFAULT_PRIOR_P,
                                     "gap_percentile", seed=seed))
        for prior_p in PRIOR_P_GRID:
            if prior_p == DEFAULT_PRIOR_P:
                continue
            rows.append(run_setting(locus, DEFAULT_KAPPA, DEFAULT_GAP_PCT, prior_p,
                                     "prior_p", seed=seed))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(SENS_DIR, "sensitivity_results.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    loci = load_loci()
    print(f"{len(loci)} loci loaded from {RESULTS_DIR}")
    rows = run_all(loci, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} settings x locus results written to {args.out}")
