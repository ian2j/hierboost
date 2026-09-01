"""Unified per-locus driver for the 1000 Genomes case-study suite (added 2026-08-28,
building toward ~10 case studies for a paper launching the block-latent model).

For a given locus (chrom/start/end/pop_a/pop_b/known causal SNP position), this script:
  1. fetches the region from 1000 Genomes phase 3 (genomics_1kg_fetch.fetch_region),
     reusing a cached .npz if one already exists at --data
  2. builds LD blocks and runs the same 10-fold CV baseline comparison as
     genomics_1kg_demo.py (L1-logistic, PCA+logistic, linear SVM, Random Forest,
     hierboost continuous block-latent + spike-slab), now with SuSiE (Wang et al. 2020)
     added as a modern fine-mapping baseline (genomics_1kg_demo.susie_fold)
  3. runs a full-data fit for both hierboost (em_filter) and SuSiE (susie_full_fit),
     and computes each method's distance from its top-ranked SNP/block to the locus's
     known causal variant -- the same "does it land in the right neighborhood" metric
     used for LCT/SLC24A5/DARC
  4. writes a machine-readable JSON summary (results/<label>.json) and the existing
     two-panel figure, so the eventual paper's results table can be built by reading
     JSON files rather than re-transcribing printed output by hand

Does NOT run the discrete/Binomial JAX branch (hierboost.latent) -- that needs
.venv-jax and is run separately per locus via genomics_1kg_binomial_latent.py, same
two-environment split as the rest of the project.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from genomics_1kg_demo import (load_data, build_blocks, cross_validate, full_data_fit,
                                susie_full_fit, make_figure)
from hierboost.blocks import block_membership_lists

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
GENOMICS_VENV_PY = os.path.join(ROOT, ".venv-genomics", "bin", "python")


def run_locus(label, chrom, start, end, pop_a, pop_b, causal_pos, causal_rsid,
              data_path=None, gene=None, n_folds=10, seed=0):
    data_path = data_path or os.path.expanduser(f"~/genomics_1kg/{label.lower()}_region.npz")
    if not os.path.exists(data_path):
        # fetch_region needs pysam, which only lives in .venv-genomics (kept separate
        # from the main analysis env's numpy<2 pin) -- shell out rather than import it here
        print(f"=== fetching chr{chrom}:{start}-{end} ({pop_a} vs {pop_b}) ===")
        subprocess.run([GENOMICS_VENV_PY, os.path.join(HERE, "genomics_1kg_fetch.py"),
                         "--chrom", str(chrom), "--start", str(start), "--end", str(end),
                         "--pop_a", pop_a, "--pop_b", pop_b, "--out", data_path], check=True)
    else:
        print(f"=== reusing cached data at {data_path} ===")

    X, y, positions, pop_a, pop_b = load_data(data_path)
    block_id = build_blocks(positions)

    t0 = time.time()
    cv_df = cross_validate(X, y, block_id, n_folds=n_folds, seed=seed)
    cv_time = time.time() - t0
    cv_summary = cv_df.groupby("method").agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        n_features=("n_features", "mean")).reset_index()
    print("\n10-fold CV results:")
    print(cv_summary.to_string(index=False))

    best, block_ids = full_data_fit(X, y, block_id)
    blocks = block_membership_lists(block_id)
    block_pos = {b: float(positions[blocks[b]].mean()) for b in blocks}
    hb_top_block = None
    hb_top_theta = -1.0
    for local_k, global_k in enumerate(best.retained_idx):
        theta = float(best.theta_hat[local_k])
        if theta > hb_top_theta:
            hb_top_theta = theta
            hb_top_block = block_ids[global_k]
    hb_top_pos = block_pos[hb_top_block] if hb_top_block is not None else None
    hb_dist = abs(hb_top_pos - causal_pos) if hb_top_pos is not None else None
    print(f"\nhierboost full-data top block: pos~{hb_top_pos:.0f}, theta={hb_top_theta:.4f}, "
          f"{hb_dist:.0f}bp from {causal_rsid}")

    susie_res = susie_full_fit(X, y, positions)
    susie_dist = abs(susie_res["top_pos"] - causal_pos)
    n_cs = len(susie_res["cs"]) if susie_res["cs"] else 0
    print(f"SuSiE full-data top PIP SNP: pos={susie_res['top_pos']}, pip={susie_res['top_pip']:.4f}, "
          f"{susie_dist:.0f}bp from {causal_rsid}, {n_cs} credible set(s)")

    out_name = f"genomics_1kg_{label.lower()}_demo.png"
    make_figure(cv_df, best, block_ids, block_id, positions, chrom, label, out_name,
                pop_a=pop_a, pop_b=pop_b)

    summary = dict(
        label=label, gene=gene, chrom=str(chrom), start=int(start), end=int(end),
        pop_a=pop_a, pop_b=pop_b, causal_pos=int(causal_pos), causal_rsid=causal_rsid,
        n_individuals=int(X.shape[0]), n_snps=int(X.shape[1]), n_blocks=len(block_ids),
        cv_time_sec=cv_time,
        cv_results=cv_summary.to_dict(orient="records"),
        hierboost_full=dict(top_block=int(hb_top_block) if hb_top_block is not None else None,
                             top_pos=hb_top_pos, top_theta=hb_top_theta, dist_bp=hb_dist,
                             n_retained=int(best.n_features)),
        susie_full=dict(top_pos=susie_res["top_pos"], top_pip=susie_res["top_pip"],
                         dist_bp=float(susie_dist), n_credible_sets=n_cs),
        figure=out_name,
    )
    json_path = os.path.join(RESULTS_DIR, f"{label.lower()}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {json_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--gene", default=None)
    p.add_argument("--chrom", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pop_a", default="EUR")
    p.add_argument("--pop_b", default="AFR")
    p.add_argument("--causal_pos", type=int, required=True)
    p.add_argument("--causal_rsid", required=True)
    p.add_argument("--data", default=None)
    args = p.parse_args()
    run_locus(args.label, args.chrom, args.start, args.end, args.pop_a, args.pop_b,
               args.causal_pos, args.causal_rsid, data_path=args.data, gene=args.gene)
