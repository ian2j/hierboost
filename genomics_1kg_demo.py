"""Fifth cross-domain application, and a return to hierboost's origin domain -- but with
REAL genomic data for the first time (every prior GWAS demo in this project was
synthetic simulation). Real LD-block structure from 1000 Genomes phase 3, LCT gene
region (chr2:136.4-136.7Mb, GRCh37) -- the textbook population-genetics example of
strong recent positive selection (European lactase persistence), predicting EUR vs AFR
superpopulation ancestry from 659 biallelic SNPs (MAF>=0.05, 1164 individuals).

Uses threshold_blocks_1d directly on real chromosomal position -- the ORIGINAL Chapter 4
design (dissertation Sec 4.1.1), not the kmeans method built for Haxby's dense voxel
mask, since genomic position along a chromosome is exactly the "has real gaps" 1D
coordinate that blocking was designed for.

Same continuous factor.py + functional-API pattern as haxby_demo.py/newsgroups_demo.py
(not HierBoostClassifier, given the binomial+decorrelate API gap already documented) --
genotype dosage (0/1/2) is treated as a continuous feature for the decorrelation step,
consistent with how the model has been used successfully all session; the untested
literal Ch4 discrete/Binomial latent.py path (which would treat dosage as genuinely
Binomial n_trials=2) is a natural next step in .venv-jax, not attempted in this script.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import pysusie as ps

from hierboost.structure import determine_blocks
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab import em_filter

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def load_data(data_path):
    d = np.load(data_path)
    X, y, positions = d["X"].astype(float), d["y"], d["positions"]
    pop_a = str(d["pop_a"]) if "pop_a" in d else "EUR"
    pop_b = str(d["pop_b"]) if "pop_b" in d else "AFR"
    print(f"{X.shape[0]} individuals, {X.shape[1]} SNPs, "
          f"{int(y.sum())} {pop_a} / {int((1 - y).sum())} {pop_b}")
    return X, y, positions, pop_a, pop_b


def build_blocks(positions, gap_percentile=75):
    block_id = determine_blocks(coords=positions, method="threshold",
                                 gap_percentile=gap_percentile)  # auto zeta (Nth pct gap)
    sizes = np.bincount(block_id)
    print(f"{len(sizes)} LD blocks, size range [{sizes.min()}, {sizes.max()}], "
          f"mean {sizes.mean():.1f}")
    return block_id


def fit_block_factors(X_train, X_test, block_id):
    """Same discipline as haxby_demo.py/newsgroups_demo.py: fit on TRAIN only, project
    TEST via project_block_factor."""
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


def hierboost_fold(X_train, y_train, X_test, y_test, block_id, xi0, kappa=100.0,
                    filter_frac=0.2, min_features=10, wr=None, xi1=0.0):
    Ztr, Zte, block_ids = fit_block_factors(X_train, X_test, block_id)
    if wr is None:
        wr = np.ones(Ztr.shape[1])
    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr])
    filt = em_filter(Xd_train, y_train, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0,
                      filter_frac=filter_frac, min_features=min_features, max_outer=30)
    best = filt.best
    Xd_test = np.column_stack([np.ones(len(y_test)), Zte[:, best.retained_idx]])
    pred = (Xd_test @ best.beta) > 0
    acc = float(np.mean(pred == (y_test > 0.5)))
    return acc, best.n_features


def susie_fold(X_train, y_train, X_test, y_test, L=10):
    """SuSiE (Wang et al. 2020) fit directly on raw SNP dosages, as a modern
    fine-mapping baseline distinct from hierboost's LD-block-latent approach: SuSiE
    handles within-locus LD by allowing correlated variables to share a credible set
    rather than by pre-averaging them into one block latent. Linear-regression IBSS on
    the 0/1 ancestry label (a linear probability model) -- the standard way SuSiE is
    applied to binary phenotypes when a logistic variant isn't used, same convention as
    most applied fine-mapping papers running SuSiE on case/control GWAS.
    """
    fit = ps.susie(X_train, y_train, L=L)
    pip = np.asarray(fit.pip)
    cs = fit.sets.get("cs") if fit.sets else None
    if cs:
        sel = sorted(set(idx for s in cs for idx in np.atleast_1d(s)))
    else:
        sel = list(np.argsort(-pip)[:10])
    if len(sel) == 0:
        sel = list(np.argsort(-pip)[:1])
    clf = LogisticRegression(max_iter=2000).fit(X_train[:, sel], y_train)
    acc = clf.score(X_test[:, sel], y_test)
    return acc, len(sel)


def susie_full_fit(X, y, positions, L=10):
    """Full-data SuSiE fit for reporting PIPs/credible sets against the known causal SNP,
    the same role genomics_1kg_demo.py's em_filter full-data fit plays for hierboost."""
    fit = ps.susie(X, y, L=L)
    pip = np.asarray(fit.pip)
    cs = fit.sets.get("cs") if fit.sets else None
    top_idx = int(np.argmax(pip))
    return dict(pip=pip, cs=cs, top_idx=top_idx, top_pos=int(positions[top_idx]),
                top_pip=float(pip[top_idx]))


def cross_validate_hierboost_only(X, y, block_id, n_folds=10, seed=0, kappa=100.0, xi0=None,
                                   filter_frac=0.2, min_features=10, wr=None, xi1=0.0):
    """Same fold structure as cross_validate, but skips the baseline classifiers and SuSiE
    entirely -- for hyperparameter/prior sensitivity sweeps (varying kappa/xi0/block
    granularity/boosting-prior wr,xi1) where only hierboost's own fit changes across
    settings and re-fitting RF/SVM/PCA/SuSiE on every grid point would be pure waste."""
    if xi0 is None:
        xi0 = np.log(0.15 / 0.85)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, n_feats = [], []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        acc, n_hb = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0, kappa=kappa,
                                    filter_frac=filter_frac, min_features=min_features,
                                    wr=wr, xi1=xi1)
        accs.append(acc)
        n_feats.append(n_hb)
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                n_features_mean=float(np.mean(n_feats)))


def cross_validate(X, y, block_id, n_folds=10, seed=0):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    K = len(np.unique(block_id))
    xi0 = np.log(0.15 / 0.85)
    rows = []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        acc_hb, n_hb = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0)
        rows.append(dict(method="hierboost (LD block-latent + spike-slab)",
                          acc=acc_hb, n_features=n_hb))

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000)
        lasso.fit(Xtr_s, ytr)
        rows.append(dict(method="L1-logistic (raw SNPs)", acc=lasso.score(Xte_s, yte),
                          n_features=int((lasso.coef_ != 0).sum())))

        pca = PCA(n_components=K).fit(Xtr_s)
        log_pca = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr_s), ytr)
        rows.append(dict(method=f"PCA({K})+logistic", acc=log_pca.score(pca.transform(Xte_s), yte),
                          n_features=K))

        svm = SVC(kernel="linear", C=1.0).fit(Xtr_s, ytr)
        rows.append(dict(method="Linear SVM (raw SNPs)", acc=svm.score(Xte_s, yte),
                          n_features=X.shape[1]))

        rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed).fit(Xtr, ytr)
        rows.append(dict(method="Random Forest (raw SNPs)", acc=rf.score(Xte, yte),
                          n_features=X.shape[1]))

        acc_susie, n_susie = susie_fold(Xtr_s, ytr, Xte_s, yte)
        rows.append(dict(method="SuSiE credible-set SNPs + logistic", acc=acc_susie,
                          n_features=n_susie))

    return pd.DataFrame(rows)


def full_data_fit(X, y, block_id, kappa=100.0, xi0=None, filter_frac=0.2, min_features=10,
                   wr=None, xi1=0.0):
    if xi0 is None:
        xi0 = np.log(0.15 / 0.85)
    Z, _, block_ids = fit_block_factors(X, None, block_id)
    if wr is None:
        wr = np.ones(Z.shape[1])
    Xd = np.column_stack([np.ones(len(y)), Z])
    filt = em_filter(Xd, y, wr, xi0=xi0, xi1=xi1, kappa=kappa, nu=1.0, lam=1.0,
                      filter_frac=filter_frac, min_features=min_features, max_outer=30)
    return filt.best, block_ids


def make_figure(cv_df, best, block_ids, block_id, positions, chrom, label, out_name,
                 pop_a="EUR", pop_b="AFR"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    summary = cv_df.groupby("method").agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                           n_mean=("n_features", "mean")).reset_index()
    summary = summary.sort_values("acc_mean")
    y_pos = np.arange(len(summary))
    ax.barh(y_pos, summary["acc_mean"], xerr=summary["acc_std"], color="tab:blue", alpha=0.8)
    for i, (acc, n) in enumerate(zip(summary["acc_mean"], summary["n_mean"])):
        ax.text(acc + 0.01, i, f"~{n:.0f} features", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["method"], fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel(f"10-fold CV accuracy: {pop_a} vs {pop_b} ancestry ({label} region)")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    blocks = block_membership_lists(block_id)
    block_pos = {b: positions[blocks[b]].mean() for b in blocks}
    theta_full = np.zeros(len(block_ids))
    for local_k, global_k in enumerate(best.retained_idx):
        theta_full[global_k] = best.theta_hat[local_k]
    xs = [block_pos[b] for b in block_ids]
    ax2.scatter(xs, theta_full, s=18, color="tab:red", alpha=0.8)
    ax2.set_xlabel(f"chr{chrom} position (bp)")
    ax2.set_ylabel("P(LD block included | y), full-data em_filter fit")
    ax2.set_title(f"LD blocks distinguishing {pop_a} vs {pop_b} ancestry\n({label} region)")

    fig.suptitle("1000 Genomes: hierboost real LD-block-latent + spike-slab vs. standard baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/{out_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/genomics_1kg/lct_region.npz"))
    p.add_argument("--chrom", default="2")
    p.add_argument("--label", default="LCT")
    p.add_argument("--out", default="genomics_1kg_demo.png")
    args = p.parse_args()

    X, y, positions, pop_a, pop_b = load_data(args.data)
    block_id = build_blocks(positions)

    cv_df = cross_validate(X, y, block_id)
    print("\n10-fold CV results:")
    print(cv_df.groupby("method").agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                       n_features=("n_features", "mean")))

    best, block_ids = full_data_fit(X, y, block_id)
    print(f"\nFull-data em_filter: retained {best.n_features} LD blocks (of {len(block_ids)} total)")

    make_figure(cv_df, best, block_ids, block_id, positions, args.chrom, args.label, args.out,
                pop_a=pop_a, pop_b=pop_b)
