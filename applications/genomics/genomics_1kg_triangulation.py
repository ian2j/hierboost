"""Triangulation study (2026-09-01, Ian's own framing): instead of treating hierboost,
SuSiE, and classical single-SNP association tests as a horse race for which one alone
gets closest to the known causal SNP, test whether COMBINING all three -- an ensemble
selection rule -- beats every one of them individually. Reuses the same 10 loci already
fetched/cached for the paper (results/*.json has each locus's metadata + individual
method distances, sec:synthesis in paper/main.tex documents the horse-race baseline this
is trying to beat: hierboost wins localization on 4/10, SuSiE on 4/10, both miss on 2/10).

Adds exactly one new baseline that didn't exist in genomics_1kg_demo.py -- a classical
single-SNP association test -- then combines all three methods' per-SNP evidence via
three different ensemble rules (rank-average, floored geometric mean, pairwise-agreement
consensus). Reuses load_data/build_blocks/full_data_fit/susie_full_fit/hierboost_fold/
susie_fold from genomics_1kg_demo.py unchanged.

Single-SNP test convention: per-SNP Pearson correlation between raw dosage and the 0/1
ancestry label, vectorized across all SNPs at once, converted to a t-statistic p-value
(df = n-2) -- the linear-probability-model analogue of a per-SNP GWAS test, chosen to
match SuSiE's own already-documented linear-probability convention in this project
(genomics_1kg_demo.py's susie_fold docstring) rather than mixing a logistic single-SNP
test with a linear-probability multi-SNP one.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import pysusie as ps

from genomics_1kg_demo import (load_data, build_blocks, full_data_fit, susie_full_fit,
                                hierboost_fold, susie_fold, fit_block_factors)
from hierboost.blocks import block_membership_lists
from hierboost.spike_slab import em_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_triangulation")
os.makedirs(RESULTS_DIR, exist_ok=True)
MAIN_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


# ---------------------------------------------------------------------------
# New baseline: classical single-SNP association test
# ---------------------------------------------------------------------------

def single_snp_test(X, y):
    """Vectorized per-SNP Pearson correlation test (linear-probability-model analogue
    of a GWAS single-marker test). Returns (pvals, neglog10p) per SNP."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    pvals = 2 * stats.t.sf(np.abs(t), df=n - 2)
    pvals = np.clip(pvals, 1e-300, 1.0)
    return pvals, -np.log10(pvals)


def single_snp_full_fit(X, y, positions):
    pvals, neglogp = single_snp_test(X, y)
    top_idx = int(np.argmin(pvals))
    return dict(pvals=pvals, neglogp=neglogp, top_idx=top_idx,
                top_pos=int(positions[top_idx]), top_p=float(pvals[top_idx]))


def single_snp_fold(X_train, y_train, X_test, y_test, alpha=0.05):
    pvals, _ = single_snp_test(X_train, y_train)
    bonf = alpha / len(pvals)
    sel = np.where(pvals <= bonf)[0]
    if len(sel) == 0:
        sel = np.array([int(np.argmin(pvals))])
    clf = LogisticRegression(max_iter=2000).fit(X_train[:, sel], y_train)
    acc = clf.score(X_test[:, sel], y_test)
    return acc, len(sel)


# ---------------------------------------------------------------------------
# Per-SNP score alignment + ensemble rules
# ---------------------------------------------------------------------------

def hierboost_persnp_theta(best, block_ids, block_id):
    """Broadcast each retained block's theta_hat (0 for dropped blocks) onto every
    member SNP, so hierboost's block-level evidence lives on the same per-SNP axis as
    SuSiE's PIPs and the single-SNP test's scores."""
    blocks = block_membership_lists(block_id)
    theta_per_block = {b: 0.0 for b in block_ids}
    for local_k, global_k in enumerate(best.retained_idx):
        theta_per_block[block_ids[global_k]] = float(best.theta_hat[local_k])
    theta_snp = np.zeros(len(block_id))
    for b, idx in blocks.items():
        theta_snp[idx] = theta_per_block[b]
    return theta_snp


def pct_rank(x):
    return stats.rankdata(x) / len(x)


def triangulate(hb_theta, susie_pip, snp_neglogp, positions, eps=1e-3):
    """Three ensemble rules over the same three per-SNP evidence arrays. Returns a dict
    of rule_name -> (top_pos, extra_info)."""
    out = {}

    # 1) rank-average (Borda count): average percentile rank across the three methods
    rank_avg = (pct_rank(hb_theta) + pct_rank(susie_pip) + pct_rank(snp_neglogp)) / 3.0
    idx = int(np.argmax(rank_avg))
    out["rank_avg"] = dict(top_pos=int(positions[idx]), score=float(rank_avg[idx]))

    # 2) floored geometric mean: near-zero unless all three give some credit (an
    # "AND-like" combination), floored so one method's exact zero doesn't force every
    # SNP outside its evidence to tie at zero
    snp_pct = pct_rank(snp_neglogp)
    geo = (np.maximum(hb_theta, eps) * np.maximum(susie_pip, eps) *
           np.maximum(snp_pct, eps)) ** (1.0 / 3.0)
    idx = int(np.argmax(geo))
    out["geo_mean"] = dict(top_pos=int(positions[idx]), score=float(geo[idx]))

    # 3) pairwise-agreement consensus: each method's own top pick; if any two land
    # within `window` bp of each other, use their midpoint; else fall back to rank_avg's
    # pick and flag no-consensus
    window = 10000
    hb_pos = int(positions[np.argmax(hb_theta)])
    susie_pos = int(positions[np.argmax(susie_pip)])
    snp_pos = int(positions[np.argmax(snp_neglogp)])
    pairs = [("hb", "susie", hb_pos, susie_pos), ("hb", "snp", hb_pos, snp_pos),
             ("susie", "snp", susie_pos, snp_pos)]
    agreeing = [(a, b, pa, pb) for a, b, pa, pb in pairs if abs(pa - pb) <= window]
    if agreeing:
        a, b, pa, pb = min(agreeing, key=lambda t: abs(t[2] - t[3]))
        out["consensus"] = dict(top_pos=(pa + pb) // 2, consensus=True, agreeing_pair=f"{a}+{b}")
    else:
        out["consensus"] = dict(top_pos=out["rank_avg"]["top_pos"], consensus=False,
                                  agreeing_pair=None)
    return out


def triangulated_fold(X_train, y_train, X_test, y_test, block_id, xi0, positions,
                       kappa=100.0, filter_frac=0.2, min_features=10, top_k=15):
    """Fold-level ensemble classifier: refit all three methods on TRAIN only, rank-average
    their per-SNP scores, take the top_k SNPs, fit a joint logistic on that selection."""
    Ztr, _, block_ids = fit_block_factors(X_train, None, block_id)
    wr = np.ones(Ztr.shape[1])
    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr])
    filt = em_filter(Xd_train, y_train, wr, xi0=xi0, xi1=0.0, kappa=kappa, nu=1.0, lam=1.0,
                      filter_frac=filter_frac, min_features=min_features, max_outer=30)
    hb_theta = hierboost_persnp_theta(filt.best, block_ids, block_id)

    susie_fit = ps.susie(X_train, y_train, L=10)
    susie_pip = np.asarray(susie_fit.pip)

    _, snp_neglogp = single_snp_test(X_train, y_train)

    rank_avg = (pct_rank(hb_theta) + pct_rank(susie_pip) + pct_rank(snp_neglogp)) / 3.0
    sel = np.argsort(-rank_avg)[:top_k]

    clf = LogisticRegression(max_iter=2000).fit(X_train[:, sel], y_train)
    acc = clf.score(X_test[:, sel], y_test)
    return acc, len(sel)


# ---------------------------------------------------------------------------
# Per-locus driver
# ---------------------------------------------------------------------------

def run_locus_triangulation(label, chrom, causal_pos, causal_rsid, data_path,
                             n_folds=10, seed=0, run_cv=True):
    X, y, positions, pop_a, pop_b = load_data(data_path)
    block_id = build_blocks(positions)
    xi0 = np.log(0.15 / 0.85)

    best, block_ids = full_data_fit(X, y, block_id)
    hb_theta = hierboost_persnp_theta(best, block_ids, block_id)
    hb_top_idx = int(np.argmax(hb_theta))
    hb_dist = abs(int(positions[hb_top_idx]) - causal_pos)

    susie_res = susie_full_fit(X, y, positions)
    susie_pip = susie_res["pip"]
    susie_dist = abs(susie_res["top_pos"] - causal_pos)

    snp_res = single_snp_full_fit(X, y, positions)
    snp_dist = abs(snp_res["top_pos"] - causal_pos)

    ens = triangulate(hb_theta, susie_pip, snp_res["neglogp"], positions)
    ens_dist = {rule: abs(info["top_pos"] - causal_pos) for rule, info in ens.items()}

    print(f"\n=== {label} (chr{chrom}, causal {causal_rsid} @ {causal_pos}) ===")
    print(f"  hierboost   top={positions[hb_top_idx]:.0f}  dist={hb_dist:.0f}bp")
    print(f"  SuSiE       top={susie_res['top_pos']}  dist={susie_dist:.0f}bp")
    print(f"  single-SNP  top={snp_res['top_pos']}  dist={snp_dist:.0f}bp  p={snp_res['top_p']:.2e}")
    for rule, info in ens.items():
        extra = f" ({info.get('agreeing_pair')} agree)" if rule == "consensus" and info.get("consensus") else ""
        print(f"  {rule:10s} top={info['top_pos']}  dist={ens_dist[rule]:.0f}bp{extra}")

    cv_summary = None
    if run_cv:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rows = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            acc_hb, n_hb = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0)
            rows.append(dict(method="hierboost", acc=acc_hb, n_features=n_hb))

            acc_susie, n_susie = susie_fold(Xtr_s, ytr, Xte_s, yte)
            rows.append(dict(method="SuSiE", acc=acc_susie, n_features=n_susie))

            acc_snp, n_snp = single_snp_fold(Xtr_s, ytr, Xte_s, yte)
            rows.append(dict(method="single-SNP", acc=acc_snp, n_features=n_snp))

            acc_ens, n_ens = triangulated_fold(Xtr, ytr, Xte, yte, block_id, xi0, positions)
            rows.append(dict(method="triangulated (rank-avg top-15)", acc=acc_ens, n_features=n_ens))

        cv_df = pd.DataFrame(rows)
        cv_summary = cv_df.groupby("method").agg(
            acc_mean=("acc", "mean"), acc_std=("acc", "std"),
            n_features=("n_features", "mean")).reset_index().to_dict(orient="records")
        print("\n  10-fold CV:")
        for r in cv_summary:
            print(f"    {r['method']:35s} acc={r['acc_mean']:.4f}+-{r['acc_std']:.4f}  "
                  f"n_features~{r['n_features']:.1f}")

    summary = dict(
        label=label, chrom=str(chrom), causal_pos=int(causal_pos), causal_rsid=causal_rsid,
        n_snps=int(X.shape[1]), n_blocks=len(block_ids),
        individual=dict(
            hierboost=dict(top_pos=int(positions[hb_top_idx]), dist_bp=float(hb_dist)),
            susie=dict(top_pos=susie_res["top_pos"], dist_bp=float(susie_dist)),
            single_snp=dict(top_pos=snp_res["top_pos"], dist_bp=float(snp_dist),
                             top_p=snp_res["top_p"]),
        ),
        ensemble={rule: dict(top_pos=info["top_pos"], dist_bp=float(ens_dist[rule]),
                              **{k: v for k, v in info.items() if k not in ("top_pos", "score")})
                  for rule, info in ens.items()},
        cv_results=cv_summary,
    )
    json_path = os.path.join(RESULTS_DIR, f"{label.lower()}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary written to {json_path}")
    return summary


LOCI = [
    ("LCT", "2", "lct_region.npz"),
    ("SLC24A5", "15", "slc24a5_region.npz"),
    ("DARC", "1", "darc_region.npz"),
    ("EDAR", None, "edar_region.npz"),
    ("HERC2_OCA2", None, "herc2_oca2_region.npz"),
    ("ABCC11", None, "abcc11_region.npz"),
    ("ADH1B", None, "adh1b_region.npz"),
    ("APOL1", None, "apol1_region.npz"),
    ("FUT2", None, "fut2_region.npz"),
    ("SLC45A2", None, "slc45a2_region.npz"),
]


def load_locus_metadata(label):
    """Pull chrom/causal_pos/causal_rsid from the existing paper results/*.json rather
    than re-typing them (already verified live against Ensembl when the paper was built,
    per [[project-1kg-paper]])."""
    with open(os.path.join(MAIN_RESULTS_DIR, f"{label.lower()}.json")) as f:
        d = json.load(f)
    return d["chrom"], d["causal_pos"], d["causal_rsid"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default=None, help="run a single locus; omit to run all 10")
    p.add_argument("--no-cv", action="store_true", help="skip the 10-fold CV comparison")
    args = p.parse_args()

    home = os.path.expanduser("~/genomics_1kg")
    targets = [l for l in LOCI if args.label is None or l[0] == args.label.upper()]
    all_summaries = []
    for label, _chrom_hint, npz_name in targets:
        chrom, causal_pos, causal_rsid = load_locus_metadata(label)
        data_path = os.path.join(home, npz_name)
        s = run_locus_triangulation(label, chrom, causal_pos, causal_rsid, data_path,
                                     run_cv=not args.no_cv)
        all_summaries.append(s)
