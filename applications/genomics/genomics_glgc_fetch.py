"""Fetch a bounded genomic region's worth of rows from a GLGC 2021 (Graham et al.,
Nature 2021; Kanoni/Willer group at U Michigan) ancestry-specific lipid GWAS summary
statistics file, via remote range-restricted tabix access -- same "no full-file
download" pattern genomics_1kg_fetch.py already uses for 1000 Genomes VCFs, just
against a bgzip+tabix-indexed plain-text results file instead of a VCF. Needs pysam,
so lives in .venv-genomics like genomics_1kg_fetch.py (kept out of the main numpy<2 env
on purpose -- see README/pyproject.toml's `genomics` extra).

Files, access, and units all verified LIVE (2026-08-30), not assumed:
  - Download root: https://csg.sph.umich.edu/willer/public/glgc-lipids2021/ (freely
    browsable directory listing, no registration/login anywhere in the flow -- linked
    from the GLGC's own site, www.lipidgenetics.org).
  - Ancestry-specific single-variant results:
    .../results/ancestry_specific/{TRAIT}_INV_{ANCESTRY}_..._ALL.meta.singlevar.results.gz
    (+ a .tbi index sitting right next to it -- confirmed bgzip via the file's magic
    bytes, not just its .gz name, before assuming tabix random access would even work).
  - README.txt in that directory documents the columns explicitly (reproduced below);
    POS_b37 = GRCh37/hg19, matching 1000 Genomes phase 3's own coordinate system, so no
    liftover is needed to use both together:
        rsID, CHROM, POS_b37, REF (non-effect allele), ALT (effect allele), N,
        N_studies, POOLED_ALT_AF, EFFECT_SIZE (per ALT allele), SE,
        pvalue_neg_log10, pvalue, pvalue_neg_log10_GC, pvalue_GC
  - EFFECT_SIZE's underlying scale (raw lipid units vs already-standardized) is NOT
    resolved here, deliberately: hierboost.sumstats's z-score reconstruction
    (bhat_std = z/sqrt(n), z = EFFECT_SIZE/SE, computed downstream of this fetch, in
    genomics_ldlr_sumstats.py) only ever uses the dimensionless ratio EFFECT_SIZE/SE,
    which is invariant to whatever scale EFFECT_SIZE itself is reported in -- exactly to
    avoid having to silently guess a phenotype-scale convention this fetch script can't
    itself verify from the README alone.
  - The server serves both http and https; https hit a local libcurl/SSL-CA-bundle
    problem in the .venv-genomics environment (curl itself and https work fine outside
    that venv) -- http is used here since the server supports it and this is public,
    non-sensitive summary-level data with no credentials in the request.
"""
import argparse
import os
import numpy as np
import pysam

GLGC_URL_TEMPLATE = ("http://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/"
                      "ancestry_specific/{trait}_INV_{ancestry}_HRC_1KGP3_others_ALL."
                      "meta.singlevar.results.gz")
# column order per results/ancestry_specific/README.txt, fetched live 2026-08-30
COLUMNS = ["rsid", "chrom", "pos", "ref", "alt", "n", "n_studies", "alt_af",
           "effect_size", "se", "neglog10p", "pvalue", "neglog10p_gc", "pvalue_gc"]


def fetch_region(trait, ancestry, chrom, start, end, out_path):
    url = GLGC_URL_TEMPLATE.format(trait=trait, ancestry=ancestry)
    tbx = pysam.TabixFile(url)
    rows = list(tbx.fetch(str(chrom), start, end))
    print(f"{len(rows)} variants in {trait} {ancestry} chr{chrom}:{start}-{end}")

    data = {c: [] for c in COLUMNS}
    for line in rows:
        fields = line.split("\t")
        for c, v in zip(COLUMNS, fields):
            data[c].append(v.strip())

    out = dict(
        rsid=np.array(data["rsid"]),
        chrom=np.array(data["chrom"]),
        pos=np.array(data["pos"], dtype=np.int64),
        ref=np.array([r.upper() for r in data["ref"]]),
        alt=np.array([a.upper() for a in data["alt"]]),
        n=np.array(data["n"], dtype=np.int64),
        alt_af=np.array(data["alt_af"], dtype=float),
        effect_size=np.array(data["effect_size"], dtype=float),
        se=np.array(data["se"], dtype=float),
        pvalue=np.array(data["pvalue"], dtype=float),
        trait=trait, ancestry=ancestry, region_chrom=str(chrom),
        region_start=int(start), region_end=int(end),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"saved to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trait", default="LDL")
    p.add_argument("--ancestry", default="EUR")
    p.add_argument("--chrom", default="19")
    p.add_argument("--start", type=int, default=11_100_000)
    p.add_argument("--end", type=int, default=11_300_000)
    p.add_argument("--out", default=os.path.expanduser("~/genomics_1kg/glgc_ldl_eur_ldlr.npz"))
    args = p.parse_args()
    fetch_region(args.trait, args.ancestry, args.chrom, args.start, args.end, args.out)
