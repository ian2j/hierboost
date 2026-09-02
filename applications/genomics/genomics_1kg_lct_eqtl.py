"""Real-phenotype validation for LCT, replacing the EUR/AFR ancestry-label proxy with
actual Geuvadis gene expression. Tests both LCT (gut-specific, expected null in this LCL
tissue) and MCM6 (the causal SNP's actual host gene, ubiquitously expressed -- expected
to show the real eQTL signal)."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from hierboost.structure import determine_blocks
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab_gaussian import em_filter_gaussian

LCT_NPZ = os.path.expanduser("~/genomics_1kg/lct_region.npz")
GEUV_HEADER = "/tmp/geuv_header.txt"
GEUV_ROWS = {"LCT": "/tmp/geuv_lct_row.txt", "MCM6": "/tmp/geuv_mcm6_row.txt"}
MCM6_SPAN = (136597196, 136633996)
LCT_SPAN = (136545410, 136594750)
CAUSAL_POS = 136608646


def load_expression(gene):
    samples = open(GEUV_HEADER).read().strip().split("\t")[4:]
    vals = np.array([float(v) for v in open(GEUV_ROWS[gene]).read().strip().split("\t")[4:]])
    return dict(zip(samples, vals))


def load_aligned_data(gene):
    d = np.load(LCT_NPZ, allow_pickle=True)
    X_all, positions, sample_ids = d["X"].astype(float), d["positions"], d["sample_ids"]
    expr = load_expression(gene)
    keep = np.array([s in expr for s in sample_ids])
    X = X_all[keep]
    y = np.array([expr[s] for s in sample_ids[keep]])
    print(f"{gene}: {keep.sum()} individuals with both genotype + expression data, "
          f"{X.shape[1]} SNPs")
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
    Ztr_s = (Ztr - Ztr.mean(0)) / (Ztr.std(0) + 1e-8)
    Zte_s = (Zte - Ztr.mean(0)) / (Ztr.std(0) + 1e-8)
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
        rows.append(dict(method="hierboost (LD block-latent + spike-slab)",
                          r2=r2_hb, n_features=n_hb))

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        lasso = Lasso(alpha=0.05, max_iter=5000).fit(Xtr_s, ytr)
        rows.append(dict(method="Lasso (raw SNPs)", r2=r2_score(yte, lasso.predict(Xte_s)),
                          n_features=int((lasso.coef_ != 0).sum())))

        pca = PCA(n_components=K).fit(Xtr_s)
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


def run(gene):
    print(f"\n{'='*20} {gene} expression as outcome {'='*20}")
    X, y, positions = load_aligned_data(gene)
    block_id = build_blocks(positions)

    cv_df = cross_validate(X, y, block_id)
    summary = cv_df.groupby("method").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                           n_features=("n_features", "mean"))
    print(f"\n10-fold CV R^2 predicting {gene} expression:")
    print(summary.sort_values("r2_mean", ascending=False))

    best, block_ids = full_data_fit(X, y, block_id)
    blocks = block_membership_lists(block_id)
    block_pos = {b: float(positions[blocks[b]].mean()) for b in blocks}
    print(f"\nfull-data fit: {best.n_features} blocks retained of {len(block_ids)}, "
          f"ranked by theta:")
    ranked = sorted(zip(best.retained_idx, best.theta_hat), key=lambda t: -t[1])
    for local_idx, theta in ranked:
        b = block_ids[local_idx]
        pos = block_pos[b]
        in_mcm6 = MCM6_SPAN[0] <= pos <= MCM6_SPAN[1]
        in_lct = LCT_SPAN[0] <= pos <= LCT_SPAN[1]
        tag = " <- MCM6" if in_mcm6 else (" <- LCT" if in_lct else "")
        print(f"  block{b:4d}  pos={pos:>12.0f}  theta={theta:.4f}  "
              f"dist_to_rs4988235={abs(pos-CAUSAL_POS):>9.0f}bp{tag}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gene", choices=["LCT", "MCM6", "both"], default="both")
    args = p.parse_args()
    genes = ["LCT", "MCM6"] if args.gene == "both" else [args.gene]
    for g in genes:
        run(g)
