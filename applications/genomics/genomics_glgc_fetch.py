"""Fetches a bounded region's rows from a GLGC 2021 ancestry-specific lipid GWAS summary
statistics file via remote tabix range access, same no-full-download pattern as
genomics_1kg_fetch.py. Needs pysam (.venv-genomics). Only ever uses the dimensionless
EFFECT_SIZE/SE ratio downstream, so the phenotype's raw scale doesn't matter."""
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
