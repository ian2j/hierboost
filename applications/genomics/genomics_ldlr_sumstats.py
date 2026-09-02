"""Fits hierboost.sumstats (summary-stats-only spike-and-slab) to real GLGC LDL
cholesterol GWAS summary statistics at LDLR, using 1000 Genomes EUR genotypes as the LD
reference panel. Known limitation: the 503-person reference panel is far smaller than
GLGC's actual GWAS sample -- the split-half robustness check below is a direct check for
whether that mismatch is visibly distorting the result."""
import argparse
import json
import os
import subprocess

import numpy as np

from hierboost.blocks import threshold_blocks_1d
from hierboost.sumstats import em_filter_sumstats, summarize_by_block, effective_rank

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GENOMICS_VENV_PY = os.path.join(ROOT, ".venv-genomics", "bin", "python")
RESULTS_DIR = os.path.join(ROOT, "results")
GLGC_PATH = os.path.expanduser("~/genomics_1kg/glgc_ldl_eur_ldlr.npz")
KG_PATH = os.path.expanduser("~/genomics_1kg/ldlr_eur_region.npz")

CHROM, START, END = "19", 11_100_000, 11_300_000
# rs6511720, chr19:11,202,306 (GRCh37) -- verified live via Ensembl GRCh37 REST API
# (grch37.rest.ensembl.org/variation/human/rs6511720), the LDLR-intron-1 lead SNP for
# LDL-C identified in GWAS (see e.g. PMC5156384) and used as an LDLR-locus example in
# the SuSiE-RSS paper itself.
CAUSAL_POS, CAUSAL_RSID = 11_202_306, "rs6511720"

AMBIGUOUS_PAIRS = ({"A", "T"}, {"C", "G"})
COMMON_HYPERPARAMS = dict(xi0=-1.5, xi1=0.0, kappa=80.0, nu=1.0, lam=1.0)


def _ensure_fetched():
    if not os.path.exists(GLGC_PATH):
        print(f"=== fetching GLGC LDL EUR sumstats chr{CHROM}:{START}-{END} ===")
        subprocess.run([GENOMICS_VENV_PY, os.path.join(HERE, "genomics_glgc_fetch.py"),
                         "--trait", "LDL", "--ancestry", "EUR", "--chrom", CHROM,
                         "--start", str(START), "--end", str(END), "--out", GLGC_PATH],
                        check=True)
    else:
        print(f"=== reusing cached GLGC sumstats at {GLGC_PATH} ===")
    if not os.path.exists(KG_PATH):
        print(f"=== fetching 1000G EUR genotypes chr{CHROM}:{START}-{END} ===")
        subprocess.run([GENOMICS_VENV_PY, os.path.join(HERE, "genomics_1kg_fetch.py"),
                         "--chrom", CHROM, "--start", str(START), "--end", str(END),
                         "--pop_a", "EUR", "--pop_b", "EUR", "--out", KG_PATH], check=True)
    else:
        print(f"=== reusing cached 1000G genotypes at {KG_PATH} ===")


def harmonize(kg, glgc):
    pos_1kg = kg["positions"]
    ref_1kg = np.array([ra.split(">")[0] for ra in kg["ref_alt"]])
    alt_1kg = np.array([ra.split(">")[1] for ra in kg["ref_alt"]])

    glgc_by_pos = {}
    for i in range(glgc["pos"].shape[0]):
        glgc_by_pos.setdefault(int(glgc["pos"][i]), []).append(i)

    keep_idx, bhat_raw, se_arr, n_arr, flipped, n_ambig, n_mismatch, n_indel, n_unmatched = (
        [], [], [], [], 0, 0, 0, 0, 0)
    for j in range(pos_1kg.shape[0]):
        r1, a1 = ref_1kg[j], alt_1kg[j]
        cands = glgc_by_pos.get(int(pos_1kg[j]))
        if not cands:
            n_unmatched += 1
            continue
        i = cands[0]  # 1000G already restricted to biallelic; take the first GLGC row at this pos
        r2, a2 = glgc["ref"][i], glgc["alt"][i]
        if len(r1) != 1 or len(a1) != 1 or len(r2) != 1 or len(a2) != 1:
            n_indel += 1
            continue
        if {r1, a1} in AMBIGUOUS_PAIRS:
            n_ambig += 1
            continue
        if r1 == r2 and a1 == a2:
            sign = 1.0
        elif r1 == a2 and a1 == r2:
            sign = -1.0
            flipped += 1
        else:
            n_mismatch += 1
            continue
        keep_idx.append(j)
        bhat_raw.append(sign * float(glgc["effect_size"][i]))
        se_arr.append(float(glgc["se"][i]))
        n_arr.append(int(glgc["n"][i]))

    report = dict(n_1kg_snps=int(pos_1kg.shape[0]), n_glgc_snps=int(glgc["pos"].shape[0]),
                  n_kept=len(keep_idx), n_unmatched=n_unmatched, n_indel=n_indel,
                  n_ambiguous_dropped=n_ambig, n_allele_mismatch_dropped=n_mismatch,
                  n_sign_flipped=flipped)
    return (np.array(keep_idx), np.array(bhat_raw), np.array(se_arr), np.array(n_arr), report)


def build_sumstats(X_snps, bhat_raw, se_arr, n_arr):
    """z-score reconstruction: bhat_std_j = z_j / sqrt(n_j), z_j = EFFECT_SIZE_j / SE_j
    -- scale-invariant, so it never has to assume what units EFFECT_SIZE is reported in
    (see genomics_glgc_fetch.py's docstring). R is the sample LD correlation matrix from
    the (harmonized, same-order) 1000G EUR genotype columns."""
    mu = X_snps.mean(axis=0)
    sd = X_snps.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X_snps - mu) / sd
    R = (Xs.T @ Xs) / Xs.shape[0]
    z = bhat_raw / se_arr
    bhat_std = z / np.sqrt(n_arr)
    return R, bhat_std, z


def run(seed=0, filter_frac=0.2, min_features=20):
    _ensure_fetched()
    kg = np.load(KG_PATH, allow_pickle=True)
    glgc = np.load(GLGC_PATH, allow_pickle=True)

    keep_idx, bhat_raw, se_arr, n_arr, harm_report = harmonize(kg, glgc)
    print("harmonization:", json.dumps(harm_report, indent=2))

    positions = kg["positions"][keep_idx]
    X_snps = kg["X"][:, keep_idx].astype(float)
    R, bhat_std, z = build_sumstats(X_snps, bhat_raw, se_arr, n_arr)
    n_rep = int(np.median(n_arr))
    p = R.shape[0]
    print(f"\nharmonized to {p} SNPs, chr{CHROM}:{positions.min()}-{positions.max()}, "
          f"representative GWAS N (median across SNPs) = {n_rep}, "
          f"LD reference panel = {X_snps.shape[0]} 1000G EUR individuals")

    # naive marginal baseline: top |z| SNP, no LD/multiple-regression modeling at all
    top_z_local = int(np.argmax(np.abs(z)))
    top_z_pos = int(positions[top_z_local])
    top_z_dist = abs(top_z_pos - CAUSAL_POS)
    print(f"\n[baseline: top |z| SNP] pos={top_z_pos}, |z|={abs(z[top_z_local]):.2f}, "
          f"{top_z_dist} bp from {CAUSAL_RSID} (pos {CAUSAL_POS})")

    block_id = threshold_blocks_1d(positions, zeta=float(np.percentile(
        np.diff(np.sort(positions)), 75)))
    wr = np.zeros(p)
    # real LD from only 503 reference individuals over 473 tightly-linked SNPs is
    # severely rank-deficient (haplotype-block redundancy, not just the p>n bound) --
    # confirmed here (not assumed): eigh(R) showed >250/473 eigenvalues below 1e-3 x the
    # top eigenvalue, which made the exact dense solve raise LinAlgError("Singular
    # matrix") on the first attempt. rank="auto" routes every internal fit through the
    # rank-truncated Woodbury path at R's OWN numerical rank (hierboost.sumstats.
    # effective_rank), recomputed fresh every em_filter_sumstats round as R shrinks --
    # a rank fixed once from the original 473x473 matrix silently stopped truncating
    # anything once filtering dropped p below that fixed number, on an equally
    # collinear smaller submatrix, and reintroduced the same divergence; "auto" fixes
    # that by recomputing per round.
    print(f"R's numerical rank at rel_tol=1e-6: {effective_rank(R)}/{p} -- "
          f"using rank='auto' Woodbury path throughout the filter loop")
    filt = em_filter_sumstats(R, bhat_std, n_rep, wr, block_id=block_id,
                               y_var=1.0, rank="auto", **COMMON_HYPERPARAMS,
                               filter_frac=filter_frac, min_features=min_features)
    best = filt.best
    top_local = int(best.retained_idx[np.argmax(best.theta_hat)])
    top_pos = int(positions[top_local])
    top_dist = abs(top_pos - CAUSAL_POS)
    print(f"\n[hierboost sumstats fit] retained {best.n_features}/{p} SNPs, "
          f"top theta_hat={best.theta_hat.max():.4f} at pos={top_pos}, "
          f"{top_dist} bp from {CAUSAL_RSID}")

    block_summary = summarize_by_block(best, block_id)
    top5_blocks = sorted(block_summary.items(), key=lambda kv: -kv[1]["top_theta"])[:5]
    print(f"\ntop 5 retained LD blocks by theta_hat:")
    for b, info in top5_blocks:
        dist = abs(positions[info["top_snp"]] - CAUSAL_POS)
        print(f"  block {b}: top_snp pos={positions[info['top_snp']]}, "
              f"theta={info['top_theta']:.4f}, beta={info['top_beta']:.4f}, "
              f"{dist} bp from {CAUSAL_RSID}")
    top5_block_report = [dict(block=int(b), pos=int(positions[info["top_snp"]]),
                               theta=info["top_theta"], beta=info["top_beta"],
                               dist_bp=int(abs(positions[info["top_snp"]] - CAUSAL_POS)))
                          for b, info in top5_blocks]

    # robustness check: split the 503-person 1000G EUR reference panel in half, rebuild
    # R from each half independently (bhat/n untouched -- those come from GLGC, not the
    # reference panel), and see whether the top hit moves. Directly probes whether the
    # reference-panel-vs-original-GWAS-sample LD mismatch this module's docstring flags
    # is visibly distorting this particular locus's result, or whether the localization
    # is stable to which 1000G EUR individuals happen to be in the panel.
    rng = np.random.default_rng(seed)
    n_indiv = X_snps.shape[0]
    order = rng.permutation(n_indiv)
    half_a, half_b = order[:n_indiv // 2], order[n_indiv // 2:]
    split_results = {}
    for label, idx in [("half_a", half_a), ("half_b", half_b)]:
        Xh = X_snps[idx]
        muh, sdh = Xh.mean(axis=0), Xh.std(axis=0, ddof=0)
        sdh = np.where(sdh > 0, sdh, 1.0)
        Xhs = (Xh - muh) / sdh
        Rh = (Xhs.T @ Xhs) / Xhs.shape[0]
        filt_h = em_filter_sumstats(Rh, bhat_std, n_rep, wr, block_id=block_id, y_var=1.0,
                                     rank="auto", **COMMON_HYPERPARAMS,
                                     filter_frac=filter_frac, min_features=min_features)
        best_h = filt_h.best
        top_h_local = int(best_h.retained_idx[np.argmax(best_h.theta_hat)])
        top_h_pos = int(positions[top_h_local])
        split_results[label] = dict(n_individuals=int(idx.shape[0]),
                                     top_pos=top_h_pos,
                                     dist_bp=int(abs(top_h_pos - CAUSAL_POS)),
                                     n_retained=int(best_h.n_features))
        print(f"  [{label}, n={idx.shape[0]} indiv] top hit pos={top_h_pos}, "
              f"{abs(top_h_pos - CAUSAL_POS)} bp from {CAUSAL_RSID}")

    summary = dict(
        locus="LDLR", chrom=CHROM, start=START, end=END, causal_pos=CAUSAL_POS,
        causal_rsid=CAUSAL_RSID, harmonization=harm_report, n_snps_harmonized=p,
        gwas_n_median=n_rep, ld_reference_n=int(X_snps.shape[0]),
        baseline_top_z=dict(pos=top_z_pos, abs_z=float(abs(z[top_z_local])),
                             dist_bp=int(top_z_dist)),
        hierboost_sumstats=dict(n_retained=int(best.n_features), top_pos=top_pos,
                                 top_theta=float(best.theta_hat.max()),
                                 dist_bp=int(top_dist)),
        top5_blocks=top5_block_report,
        robustness_split=split_results,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sumstats_ldlr_real.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(seed=args.seed)
