"""Looks at hierboost's full retained block set per locus (not just the top block used
for the localization metric elsewhere): are the secondary retained blocks picking up
real signal, or just noise?"""
import json
import os

import numpy as np

from genomics_1kg_demo import build_blocks, full_data_fit
from hierboost.blocks import block_membership_lists

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_selection")
os.makedirs(OUT_DIR, exist_ok=True)

NARRATIVE_ORDER = ["lct", "slc24a5", "darc", "edar", "herc2_oca2", "abcc11",
                    "adh1b", "apol1", "fut2", "slc45a2"]


def load_loci():
    loci = []
    for name in NARRATIVE_ORDER:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        data_path = os.path.expanduser(f"~/genomics_1kg/{name}_region.npz")
        loci.append(dict(label=d["label"], gene=d.get("gene") or d["label"],
                          causal_pos=d["causal_pos"], causal_rsid=d["causal_rsid"],
                          chrom=d["chrom"], start=d["start"], end=d["end"],
                          data_path=data_path))
    return loci


def inspect(locus):
    d = np.load(locus["data_path"])
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    block_id = build_blocks(positions, gap_percentile=75)
    xi0 = np.log(0.15 / 0.85)
    best, block_ids = full_data_fit(X, y, block_id, kappa=100.0, xi0=xi0)

    blocks = block_membership_lists(block_id)
    block_pos = {b: float(positions[blocks[b]].mean()) for b in blocks}
    block_span = {b: (float(positions[blocks[b]].min()), float(positions[blocks[b]].max()))
                  for b in blocks}
    block_size = {b: len(blocks[b]) for b in blocks}

    retained = []
    for local_k, global_k in enumerate(best.retained_idx):
        b = block_ids[global_k]
        retained.append(dict(block=int(b), pos=block_pos[b], span=block_span[b],
                              n_snps=block_size[b], theta=float(best.theta_hat[local_k]),
                              dist_to_causal=abs(block_pos[b] - locus["causal_pos"])))
    retained.sort(key=lambda r: -r["theta"])

    print(f"\n=== {locus['label']} (chr{locus['chrom']}:{locus['start']}-{locus['end']}, "
          f"causal={locus['causal_rsid']}@{locus['causal_pos']}) ===")
    print(f"{len(retained)} blocks retained of {len(block_ids)} total. Full retained set, ranked by theta:")
    for r in retained:
        flag = " <-- causal SNP's own block" if r["span"][0] <= locus["causal_pos"] <= r["span"][1] else ""
        print(f"  block{r['block']:4d}  pos={r['pos']:>12.0f}  n_snps={r['n_snps']:3d}  "
              f"theta={r['theta']:.4f}  dist_to_causal={r['dist_to_causal']:>9.0f}bp{flag}")

    return dict(label=locus["label"], gene=locus["gene"], chrom=locus["chrom"],
                start=locus["start"], end=locus["end"], causal_pos=locus["causal_pos"],
                causal_rsid=locus["causal_rsid"], n_blocks_total=len(block_ids),
                n_retained=len(retained), retained=retained)


if __name__ == "__main__":
    loci = load_loci()
    print(f"{len(loci)} loci loaded")
    all_results = [inspect(locus) for locus in loci]
    out_path = os.path.join(OUT_DIR, "selection_inspection.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwritten to {out_path}")
