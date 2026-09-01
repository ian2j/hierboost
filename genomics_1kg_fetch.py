"""Fetch a bounded genomic region from 1000 Genomes phase 3 via remote region-restricted
VCF access (pysam/htslib range requests against the .tbi index -- no full-chromosome
download, keeps memory/disk small by construction). Restricts samples to two
superpopulations (any pair of EUR/AFR/EAS/SAS/AMR, not just EUR vs AFR) and SNPs to
biallelic, MAF>=0.05. Saves a compact .npz (dosage matrix int8, positions, labels,
sample ids) for reuse.

Parameterized (generalized 2026-08-28 from the original LCT-only script, and again
2026-08-28 to accept --pop_a/--pop_b) so the same pipeline can replicate the "does the
Binomial model localize known biology better than the continuous approximation" finding
on other well-characterized selection loci -- not just LCT, and not just EUR-vs-AFR
sweeps: several textbook loci (EDAR, ABCC11, ADH1B) are EAS-specific, so the population
contrast that actually shows the selection signal has to be chosen per locus, not fixed.
Call with --chrom/--start/--end/--pop_a/--pop_b/--out, or import fetch_region() directly.
"""
import argparse
import os
import numpy as np
import pandas as pd
import pysam

PANEL_PATH = os.path.expanduser("~/genomics_1kg/panel.txt")
MAF_MIN = 0.05
VCF_URL_TEMPLATE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
                     "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")


def load_panel(pop_a="EUR", pop_b="AFR"):
    panel = pd.read_csv(PANEL_PATH, sep="\t")
    panel = panel[panel["super_pop"].isin([pop_a, pop_b])]
    return panel.set_index("sample")["super_pop"]


def fetch_region(chrom, start, end, out_path, maf_min=MAF_MIN, pop_a="EUR", pop_b="AFR"):
    panel = load_panel(pop_a, pop_b)
    vcf_url = VCF_URL_TEMPLATE.format(chrom=chrom)
    vcf = pysam.VariantFile(vcf_url)
    all_samples = list(vcf.header.samples)
    keep_mask = np.array([s in panel.index for s in all_samples])
    keep_samples = [s for s, k in zip(all_samples, keep_mask) if k]
    y = np.array([1.0 if panel[s] == pop_a else 0.0 for s in keep_samples])
    print(f"{len(all_samples)} total samples -> {len(keep_samples)} {pop_a}/{pop_b} "
          f"({int(y.sum())} {pop_a}, {int((1 - y).sum())} {pop_b})")

    dosages, positions, ref_alt = [], [], []
    n_seen, n_kept = 0, 0
    for rec in vcf.fetch(str(chrom), start, end):
        n_seen += 1
        if rec.alts is None or len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue  # biallelic SNPs only, no indels/multiallelic
        gt = np.array([sum(s) if None not in s else -1
                        for s in (rec.samples[samp]["GT"] for samp in keep_samples)])
        if (gt < 0).any():
            continue  # skip any missing calls in our sample subset (simplifies downstream)
        af = gt.mean() / 2.0
        maf = min(af, 1 - af)
        if maf < maf_min:
            continue
        dosages.append(gt.astype(np.int8))
        positions.append(rec.pos)
        ref_alt.append(f"{rec.ref}>{rec.alts[0]}")
        n_kept += 1

    X = np.column_stack(dosages)  # (n_samples, p_snps)
    positions = np.array(positions)
    print(f"{n_seen} variants scanned in chr{chrom}:{start}-{end}, {n_kept} biallelic SNPs "
          f"kept (MAF>={maf_min}), matrix shape {X.shape}")
    print(f"approx memory: {X.nbytes / 1e6:.2f} MB")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, positions=positions,
                         sample_ids=np.array(keep_samples), ref_alt=np.array(ref_alt),
                         pop_a=pop_a, pop_b=pop_b)
    print(f"saved to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", default="2")
    p.add_argument("--start", type=int, default=136_400_000)
    p.add_argument("--end", type=int, default=136_700_000)
    p.add_argument("--pop_a", default="EUR")
    p.add_argument("--pop_b", default="AFR")
    p.add_argument("--out", default=os.path.expanduser("~/genomics_1kg/lct_region.npz"))
    args = p.parse_args()
    fetch_region(args.chrom, args.start, args.end, args.out, pop_a=args.pop_a, pop_b=args.pop_b)
