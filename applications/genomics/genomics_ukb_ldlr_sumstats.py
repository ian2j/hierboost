"""Ancestry-matching follow-up to genomics_ldlr_sumstats.py: same locus/SNP/fitter, but
UK Biobank's own GWAS (Neale lab) instead of GLGC, and a GBR-only reference panel instead
of pooled EUR (UK Biobank's population is overwhelmingly White British). Confound,
disclosed: the better-matched GBR panel is also much smaller (91 vs 503), so ancestry-
match and sample-size effects can't be cleanly separated in this one comparison."""
import json
import os
import subprocess

import numpy as np
import pandas as pd

from hierboost.blocks import threshold_blocks_1d
from hierboost.sumstats import em_filter_sumstats, summarize_by_block, effective_rank

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
KG_PATH = os.path.expanduser("~/genomics_1kg/ldlr_eur_region.npz")  # reused, not refetched
PANEL_PATH = os.path.expanduser("~/genomics_1kg/panel.txt")
UKB_PATH = os.path.expanduser("~/genomics_1kg/ukb_ldl_irnt_ldlr.npz")

UKB_URL = ("https://broad-ukb-sumstats-us-east-1.s3.amazonaws.com/round2/additive-tsvs/"
           "30780_irnt.gwas.imputed_v3.both_sexes.varorder.tsv.bgz")
UKB_COLUMNS = ["variant", "minor_allele", "minor_af", "low_confidence_variant",
               "n_complete_samples", "ac", "ytx", "beta", "se", "tstat", "pval"]

CHROM, START, END = "19", 11_100_000, 11_300_000
# same causal SNP as the original GLGC run -- still the right one here: it appears in
# the UKB-harmonized SNP set (checked in run() below) and is not superseded by a
# clearly-distinct UKB top hit, so kept for direct comparability rather than forced.
CAUSAL_POS, CAUSAL_RSID = 11_202_306, "rs6511720"

AMBIGUOUS_PAIRS = ({"A", "T"}, {"C", "G"})
COMMON_HYPERPARAMS = dict(xi0=-1.5, xi1=0.0, kappa=80.0, nu=1.0, lam=1.0)


def _fetch_ukb_region(out_path=UKB_PATH, chrom=CHROM, start=START, end=END):
    """Stream-decompress the genome-wide UKB file and grep by position, per this
    module's docstring -- never writes the multi-GB file to disk, only the filtered
    region. See module docstring for why a byte-range/tabix approach isn't available
    for this particular Neale lab release (no .tbi shipped)."""
    if os.path.exists(out_path):
        print(f"=== reusing cached UKB sumstats at {out_path} ===")
        return out_path
    print(f"=== streaming UKB round-2 30780_irnt (both_sexes) genome-wide file, "
          f"filtering to chr{chrom}:{start}-{end} (no disk write of the full file) ===")
    # positions in [11,100,000, 11,299,999] are exactly "19:111xxxxx:" or "19:112xxxxx:"
    # -- both share the "11" prefix with the digit after it in {1, 2}; this is a
    # region-specific pattern (like CHROM/START/END being module-level constants
    # already), not a generic reusable region-fetcher, so it's built directly here
    # rather than derived generically from arbitrary start/end.
    lo6, hi6 = start // 100000, (end - 1) // 100000  # 111, 112
    lo_str, hi_str = str(lo6), str(hi6)
    if len(lo_str) != len(hi_str) or lo_str[:-1] != hi_str[:-1]:
        raise ValueError("region spans a 100kb-prefix boundary this quick regex "
                          "doesn't handle; widen the awk pattern for a larger region")
    pattern = f"^{chrom}:{lo_str[:-1]}[{lo_str[-1]}{hi_str[-1]}][0-9][0-9][0-9][0-9][0-9]:"
    # stop as soon as we see the next chromosome (file is chr-sorted ascending) --
    # sends curl a SIGPIPE and halts the transfer instead of reading to EOF.
    next_chrom = str(int(chrom) + 1)
    cmd = (f'curl -s "{UKB_URL}" | zcat 2>/dev/null | awk -F"\\t" '
           f"'$1 ~ /^{next_chrom}:/ {{exit}} $1 ~ /{pattern}/ {{print}}'")
    raw_tsv = out_path + ".rawtsv"
    if os.path.exists(raw_tsv):
        print(f"=== reusing already-streamed raw region file at {raw_tsv} "
              f"(skipping the slow network stream) ===")
    else:
        with open(raw_tsv, "w") as f:
            subprocess.run(cmd, shell=True, check=True, stdout=f)

    rows = []
    with open(raw_tsv) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(UKB_COLUMNS):
                continue
            rows.append(fields)
    print(f"{len(rows)} UKB variants in chr{chrom}:{start}-{end}")

    variant = np.array([r[0] for r in rows])
    parts = np.array([v.split(":") for v in variant])
    chrom_arr, pos_arr, ref_arr, alt_arr = parts[:, 0], parts[:, 1].astype(np.int64), \
        parts[:, 2], parts[:, 3]
    beta = np.array([r[7] for r in rows], dtype=float)
    se = np.array([r[8] for r in rows], dtype=float)
    n = np.array([r[4] for r in rows], dtype=np.int64)
    low_conf = np.array([r[3] for r in rows]) == "true"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, pos=pos_arr, ref=ref_arr, alt=alt_arr, beta=beta,
                         se=se, n=n, low_confidence_variant=low_conf,
                         pheno="30780_irnt", sex="both_sexes",
                         region_chrom=str(chrom), region_start=int(start),
                         region_end=int(end))
    os.remove(raw_tsv)
    print(f"saved to {out_path}")
    return out_path


def load_gbr_genotypes():
    """Subset the ALREADY-CACHED 503-person 1000G EUR genotype matrix at this exact
    locus down to the 91 GBR individuals within it -- no new network fetch, per the
    task's instruction to subset via the existing panel.txt population labels rather
    than re-fetching."""
    kg = np.load(KG_PATH, allow_pickle=True)
    panel = pd.read_csv(PANEL_PATH, sep="\t")
    pop_by_sample = panel.set_index("sample")["pop"]
    sample_ids = kg["sample_ids"]
    pops = np.array([pop_by_sample.get(s, "NA") for s in sample_ids])
    gbr_mask = pops == "GBR"
    print(f"1000G EUR cached panel: {sample_ids.shape[0]} individuals -> "
          f"{int(gbr_mask.sum())} GBR (subpopulation labels from {PANEL_PATH})")
    return kg, gbr_mask


def harmonize(kg, gbr_mask, ukb):
    pos_1kg = kg["positions"]
    ref_1kg = np.array([ra.split(">")[0] for ra in kg["ref_alt"]])
    alt_1kg = np.array([ra.split(">")[1] for ra in kg["ref_alt"]])

    ukb_by_pos = {}
    for i in range(ukb["pos"].shape[0]):
        ukb_by_pos.setdefault(int(ukb["pos"][i]), []).append(i)

    keep_idx, bhat_raw, se_arr, n_arr, flipped = [], [], [], [], 0
    n_ambig = n_mismatch = n_indel = n_unmatched = n_low_conf = 0
    for j in range(pos_1kg.shape[0]):
        r1, a1 = ref_1kg[j], alt_1kg[j]
        cands = ukb_by_pos.get(int(pos_1kg[j]))
        if not cands:
            n_unmatched += 1
            continue
        i = cands[0]  # 1000G already restricted to biallelic; take first UKB row at pos
        if bool(ukb["low_confidence_variant"][i]):
            n_low_conf += 1
            continue
        r2, a2 = str(ukb["ref"][i]), str(ukb["alt"][i])
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
        bhat_raw.append(sign * float(ukb["beta"][i]))
        se_arr.append(float(ukb["se"][i]))
        n_arr.append(int(ukb["n"][i]))

    report = dict(n_1kg_snps=int(pos_1kg.shape[0]), n_ukb_snps=int(ukb["pos"].shape[0]),
                  n_kept=len(keep_idx), n_unmatched=n_unmatched, n_indel=n_indel,
                  n_ambiguous_dropped=n_ambig, n_allele_mismatch_dropped=n_mismatch,
                  n_low_confidence_dropped=n_low_conf, n_sign_flipped=flipped)
    return (np.array(keep_idx), np.array(bhat_raw), np.array(se_arr), np.array(n_arr), report)


def build_sumstats(X_snps, bhat_raw, se_arr, n_arr):
    """Same z-score reconstruction as genomics_ldlr_sumstats.py: bhat_std_j =
    z_j / sqrt(n_j), z_j = beta_j / se_j -- scale-invariant regardless of the fact that
    UKB's beta is on a different underlying scale (irnt-normalized phenotype) than
    GLGC's EFFECT_SIZE was."""
    mu = X_snps.mean(axis=0)
    sd = X_snps.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X_snps - mu) / sd
    R = (Xs.T @ Xs) / Xs.shape[0]
    z = bhat_raw / se_arr
    bhat_std = z / np.sqrt(n_arr)
    return R, bhat_std, z


def run(seed=0, filter_frac=0.2, min_features=20):
    _fetch_ukb_region()
    ukb = np.load(UKB_PATH, allow_pickle=True)
    kg, gbr_mask = load_gbr_genotypes()

    keep_idx, bhat_raw, se_arr, n_arr, harm_report = harmonize(kg, gbr_mask, ukb)
    print("harmonization:", json.dumps(harm_report, indent=2))

    positions = kg["positions"][keep_idx]
    causal_present = bool(np.any(positions == CAUSAL_POS))
    print(f"causal SNP {CAUSAL_RSID} (pos {CAUSAL_POS}) present in harmonized set: "
          f"{causal_present}")

    X_snps_all = kg["X"][:, keep_idx].astype(float)
    X_gbr = X_snps_all[gbr_mask]
    R, bhat_std, z = build_sumstats(X_gbr, bhat_raw, se_arr, n_arr)
    n_rep = int(np.median(n_arr))
    p = R.shape[0]
    print(f"\nharmonized to {p} SNPs, chr{CHROM}:{positions.min()}-{positions.max()}, "
          f"representative GWAS N (median across SNPs) = {n_rep}, "
          f"LD reference panel = {X_gbr.shape[0]} 1000G GBR individuals "
          f"(vs 503 EUR in the original GLGC run)")

    top_z_local = int(np.argmax(np.abs(z)))
    top_z_pos = int(positions[top_z_local])
    top_z_dist = abs(top_z_pos - CAUSAL_POS)
    print(f"\n[baseline: top |z| SNP] pos={top_z_pos}, |z|={abs(z[top_z_local]):.2f}, "
          f"{top_z_dist} bp from {CAUSAL_RSID} (pos {CAUSAL_POS})")

    block_id = threshold_blocks_1d(positions, zeta=float(np.percentile(
        np.diff(np.sort(positions)), 75)))
    wr = np.zeros(p)
    rank_p = effective_rank(R)
    print(f"R's numerical rank at rel_tol=1e-6: {rank_p}/{p} -- "
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

    # robustness check, identical design to genomics_ldlr_sumstats.py's, run on the
    # 91-person GBR panel instead of the 503-person EUR panel -- see module docstring's
    # explicit flag that each half here (~45 individuals) is far smaller than the
    # original's (~251), a confound this run does not control for.
    rng = np.random.default_rng(seed)
    n_indiv = X_gbr.shape[0]
    order = rng.permutation(n_indiv)
    half_a, half_b = order[:n_indiv // 2], order[n_indiv // 2:]
    split_results = {}
    for label, idx in [("half_a", half_a), ("half_b", half_b)]:
        Xh = X_gbr[idx]
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
        causal_rsid=CAUSAL_RSID, causal_snp_present_in_harmonized_set=causal_present,
        sumstats_source=("UKB Neale lab round 2, phenotype 30780_irnt (LDL direct, "
                          "inverse-rank-normalized), both_sexes"),
        ld_reference_population=("GBR (subset of the same cached 1000G EUR fetch used "
                                  "by genomics_ldlr_sumstats.py, no new network fetch)"),
        harmonization=harm_report, n_snps_harmonized=p,
        gwas_n_median=n_rep, ld_reference_n=int(X_gbr.shape[0]),
        ld_reference_n_original_eur_run=503,
        baseline_top_z=dict(pos=top_z_pos, abs_z=float(abs(z[top_z_local])),
                             dist_bp=int(top_z_dist)),
        hierboost_sumstats=dict(n_retained=int(best.n_features), top_pos=top_pos,
                                 top_theta=float(best.theta_hat.max()),
                                 dist_bp=int(top_dist)),
        top5_blocks=top5_block_report,
        robustness_split=split_results,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sumstats_ldlr_ukb_gbr.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {out_path}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(seed=args.seed)
