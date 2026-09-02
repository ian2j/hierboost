"""Generalized real-phenotype (Geuvadis expression) eQTL validation, extending
genomics_1kg_lct_eqtl.py to any locus. LCT/MCM6 came back null (gut-specific gene, wrong
tissue); APOL1 is a better candidate since it's robustly expressed in this LCL dataset."""
import argparse
import gzip
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats

from hierboost.structure import determine_blocks
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

GEUV_EXPR_GZ = "/tmp/geuv_expr.txt.gz"


def fetch_expression_row(ensembl_id):
    with gzip.open(GEUV_EXPR_GZ, "rt") as f:
        header = f.readline().strip().split("\t")
        samples = header[4:]
        for line in f:
            if line.startswith(ensembl_id):
                vals = np.array([float(v) for v in line.strip().split("\t")[4:]])
                return dict(zip(samples, vals))
    raise ValueError(f"{ensembl_id} not found in {GEUV_EXPR_GZ}")


def load_aligned_data(npz_path, expr):
    d = np.load(npz_path, allow_pickle=True)
    X_all, positions, sample_ids = d["X"].astype(float), d["positions"], d["sample_ids"]
    keep = np.array([s in expr for s in sample_ids])
    X = X_all[keep]
    y = np.array([expr[s] for s in sample_ids[keep]])
    print(f"{keep.sum()} individuals with genotype + expression data, {X.shape[1]} SNPs")
    return X, y, positions


def build_blocks(positions):
    block_id = determine_blocks(coords=positions, method="threshold")
    sizes = np.bincount(block_id)
    print(f"{len(sizes)} LD blocks, size range [{sizes.min()}, {sizes.max()}]")
    return block_id


def fit_block_factors(X_train, X_test, block_id):
    blocks = block_membership_lists(block_id)
    block_ids = sorted(blocks.keys())
    Ztr = np.zeros((X_train.shape[0], len(block_ids)))
    Zte = np.zeros((X_test.shape[0], len(block_ids))) if X_test is not None else None
    for k, b in enumerate(block_ids):
        idx = blocks[b]
        Xb_tr = X_train[:, idx]
        if len(idx) == 1:
            Ztr[:, k] = Xb_tr[:, 0]
            if Zte is not None:
                Zte[:, k] = X_test[:, idx][:, 0]
            continue
        score, loadings = gaussian_block_factor(Xb_tr)
        Ztr[:, k] = score
        if Zte is not None:
            Zte[:, k] = project_block_factor(X_test[:, idx], loadings, Xb_tr.mean(axis=0))
    return Ztr, Zte, block_ids


def hierboost_fold(X_train, y_train, X_test, y_test, block_id):
    Ztr, Zte, block_ids = fit_block_factors(X_train, X_test, block_id)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr_s, Zte_s = (Ztr - mu) / sd, (Zte - mu) / sd
    wr = np.ones(Ztr.shape[1])
    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr_s])
    filt = em_filter_gaussian(Xd_train, y_train, wr, xi0=np.log(0.15 / 0.85), xi1=0.0,
                               kappa=100.0, nu=1.0, lam=1.0, nu_y=1.0, lam_y=1.0,
                               filter_frac=0.2, min_features=10, max_outer=30)
    best = filt.best
    Xd_test = np.column_stack([np.ones(len(y_test)), Zte_s[:, best.retained_idx]])
    pred = Xd_test @ best.beta
    return r2_score(y_test, pred), best.n_features


def cross_validate(X, y, block_id, n_folds=10, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    K = len(np.unique(block_id))
    rows = []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        r2_hb, n_hb = hierboost_fold(Xtr, ytr, Xte, yte, block_id)
        rows.append(dict(method="hierboost (LD block-latent + spike-slab)", r2=r2_hb, n_features=n_hb))

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        lasso = Lasso(alpha=0.05, max_iter=5000).fit(Xtr_s, ytr)
        rows.append(dict(method="Lasso (raw SNPs)", r2=r2_score(yte, lasso.predict(Xte_s)),
                          n_features=int((lasso.coef_ != 0).sum())))

        pca = PCA(n_components=min(K, Xtr.shape[0]-1)).fit(Xtr_s)
        lr_pca = LinearRegression().fit(pca.transform(Xtr_s), ytr)
        rows.append(dict(method=f"PCA({K})+linear", r2=r2_score(yte, lr_pca.predict(pca.transform(Xte_s))),
                          n_features=K))

        rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed).fit(Xtr, ytr)
        rows.append(dict(method="Random Forest (raw SNPs)", r2=r2_score(yte, rf.predict(Xte)),
                          n_features=X.shape[1]))
    return pd.DataFrame(rows)


def full_data_fit(X, y, block_id):
    Z, _, block_ids = fit_block_factors(X, None, block_id)
    Z_s = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    wr = np.ones(Z.shape[1])
    Xd = np.column_stack([np.ones(len(y)), Z_s])
    filt = em_filter_gaussian(Xd, y, wr, xi0=np.log(0.15 / 0.85), xi1=0.0, kappa=100.0,
                               nu=1.0, lam=1.0, nu_y=1.0, lam_y=1.0,
                               filter_frac=0.2, min_features=10, max_outer=30)
    return filt.best, block_ids


def run(npz_path, ensembl_id, gene_label, causal_pos, causal_rsid, gene_span=None):
    print(f"\n{'='*20} {gene_label} expression as outcome {'='*20}")
    expr = fetch_expression_row(ensembl_id)
    X, y, positions = load_aligned_data(npz_path, expr)

    # sanity check: raw univariate correlation at the SNP nearest the causal position
    idx0 = np.argmin(np.abs(positions - causal_pos))
    r0, p0 = stats.pearsonr(X[:, idx0], y)
    print(f"sanity check: corr(genotype @ {positions[idx0]} [{abs(positions[idx0]-causal_pos)}bp "
          f"from {causal_rsid}], {gene_label} expr) = r={r0:.4f}, p={p0:.4g}")

    block_id = build_blocks(positions)
    cv_df = cross_validate(X, y, block_id)
    summary = cv_df.groupby("method").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                           n_features=("n_features", "mean"))
    print(f"\n10-fold CV R^2 predicting {gene_label} expression:")
    print(summary.sort_values("r2_mean", ascending=False).to_string())

    best, block_ids = full_data_fit(X, y, block_id)
    blocks = block_membership_lists(block_id)
    block_pos = {b: float(positions[blocks[b]].mean()) for b in blocks}
    print(f"\nfull-data fit: {best.n_features} blocks retained of {len(block_ids)}")
    ranked = sorted(zip(best.retained_idx, best.theta_hat), key=lambda t: -t[1])
    for local_idx, theta in ranked[:15]:
        b = block_ids[local_idx]
        pos = block_pos[b]
        tag = ""
        if gene_span and gene_span[0] <= pos <= gene_span[1]:
            tag = f" <- {gene_label}"
        print(f"  block{b:4d}  pos={pos:>12.0f}  theta={theta:.4f}  "
              f"dist={abs(pos-causal_pos):>9.0f}bp{tag}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--ensembl_id", required=True)
    p.add_argument("--gene", required=True)
    p.add_argument("--causal_pos", type=int, required=True)
    p.add_argument("--causal_rsid", required=True)
    p.add_argument("--gene_start", type=int, default=None)
    p.add_argument("--gene_end", type=int, default=None)
    args = p.parse_args()
    span = (args.gene_start, args.gene_end) if args.gene_start else None
    run(args.npz, args.ensembl_id, args.gene, args.causal_pos, args.causal_rsid, span)
