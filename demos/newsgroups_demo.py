"""Binary text classification (comp.sys.mac.hardware vs ibm.pc.hardware, a hard pair)
using TF-IDF word features grouped by co-occurrence correlation -- a semantic notion of
"proximity" rather than physical space or metadata. Same template as haxby_demo.py."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab import em_filter, fit_em

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

CATEGORIES = ["comp.sys.mac.hardware", "comp.sys.ibm.pc.hardware"]
N_VOCAB = 300
CORR_RHO = 0.15


def load_data():
    d = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                            remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=N_VOCAB, stop_words="english", min_df=5)
    X = vec.fit_transform(d.data).toarray()
    y = d.target.astype(float)  # 0/1 for the two categories
    vocab = np.array(vec.get_feature_names_out())
    print(f"{X.shape[0]} posts ({CATEGORIES[0]!r} vs {CATEGORIES[1]!r}), "
          f"{X.shape[1]} TF-IDF word features, class balance {np.bincount(d.target)}")
    return X, y, vocab


def fit_block_factors(X_train, X_test, block_id):
    """Same discipline as haxby_demo.py: fit gaussian_block_factor per block on TRAIN
    only, project TEST via project_block_factor -- avoids leaking test-fold structure
    into the unsupervised decorrelation step. Uses the lower-level functional API
    directly (not HierBoostClassifier) because TF-IDF weights are continuous, and
    HierBoostClassifier(decorrelate="sar") with a binomial response routes uncondi-
    tionally through Chapter 4's discrete/JAX Binomial pipeline (which assumes the RAW
    features themselves are binomial/discrete, e.g. genotypes) -- a real gap surfaced
    while building this demo, not a fit for continuous word weights + binary outcome."""
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


def hierboost_fold(X_train, y_train, X_test, y_test, block_id, xi0):
    Ztr, Zte, block_ids = fit_block_factors(X_train, X_test, block_id)
    wr = np.ones(Ztr.shape[1])
    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr])
    filt = em_filter(Xd_train, y_train, wr, xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                      filter_frac=0.2, min_features=10, max_outer=30)
    best = filt.best
    Xd_test = np.column_stack([np.ones(len(y_test)), Zte[:, best.retained_idx]])
    pred = (Xd_test @ best.beta) > 0
    acc = float(np.mean(pred == (y_test > 0.5)))
    return acc, best.n_features


def cross_validate(X, y, n_folds=10, seed=0):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    xi0 = np.log(0.15 / 0.85)  # prior belief ~15% of word blocks matter
    # fixed K for the PCA baseline, computed once on full data -- block_id is refit per
    # fold (correct discipline, no leakage) but its count wobbles fold to fold, which
    # would otherwise fragment "PCA(K)+logistic" into several barely-comparable rows
    K_PCA = len(np.unique(blocks_from_correlation_threshold(X, rho=CORR_RHO)))
    rows = []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        block_id = blocks_from_correlation_threshold(Xtr, rho=CORR_RHO)
        acc_hb, n_hb = hierboost_fold(Xtr, ytr, Xte, yte, block_id, xi0)
        rows.append(dict(method="hierboost (word block-latent + spike-slab)",
                          acc=acc_hb, n_features=n_hb))

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000)
        lasso.fit(Xtr_s, ytr)
        rows.append(dict(method="L1-logistic (raw words)", acc=lasso.score(Xte_s, yte),
                          n_features=int((lasso.coef_ != 0).sum())))

        pca = PCA(n_components=K_PCA).fit(Xtr_s)
        log_pca = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr_s), ytr)
        rows.append(dict(method=f"PCA({K_PCA})+logistic", acc=log_pca.score(pca.transform(Xte_s), yte),
                          n_features=K_PCA))

        svm = SVC(kernel="linear", C=1.0).fit(Xtr_s, ytr)
        rows.append(dict(method="Linear SVM (raw words)", acc=svm.score(Xte_s, yte),
                          n_features=X.shape[1]))

        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=seed).fit(Xtr, ytr)
        rows.append(dict(method="Random Forest (raw words)", acc=rf.score(Xte, yte),
                          n_features=X.shape[1]))

    return pd.DataFrame(rows)


def full_data_fit(X, y, vocab):
    """Fit once on ALL data (no held-out split) for the interpretability view -- same
    role as haxby_demo.py's full_data_inclusion_map. Uses em_filter directly (the same
    engine hierboost_fold uses per CV fold) so the reported inclusion probabilities
    reflect the same filter-and-refit selection the CV accuracy numbers are based on."""
    block_id = blocks_from_correlation_threshold(X, rho=CORR_RHO)
    xi0 = np.log(0.15 / 0.85)
    Z, _, block_ids = fit_block_factors(X, None, block_id)
    wr = np.ones(Z.shape[1])
    Xd = np.column_stack([np.ones(len(y)), Z])
    filt = em_filter(Xd, y, wr, xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                      filter_frac=0.2, min_features=10, max_outer=30)
    best = filt.best

    blocks = block_membership_lists(block_id)
    block_words = {b: list(vocab[idx]) for b, idx in blocks.items()}
    retained_block_ids = [block_ids[i] for i in best.retained_idx]
    return best, retained_block_ids, block_words


def make_figure(cv_df, best, retained_block_ids, block_words):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

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
    ax.set_xlabel("10-fold CV accuracy: Mac vs PC hardware newsgroup")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    order = np.argsort(-best.theta_hat)[:20]
    labels = []
    for i in order:
        b = retained_block_ids[i]
        words = block_words.get(b, [])
        labels.append(", ".join(words[:4]) + ("..." if len(words) > 4 else ""))
    ax2.barh(np.arange(len(order)), best.theta_hat[order], color="tab:purple", alpha=0.85)
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Posterior P(word block included | y), full-data em_filter fit")
    ax2.set_title("Top word blocks distinguishing Mac vs PC hardware posts")

    fig.suptitle("20 Newsgroups: hierboost word-correlation block-latent + spike-slab vs. standard baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "newsgroups_demo.png"), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/newsgroups_demo.png")


if __name__ == "__main__":
    X, y, vocab = load_data()

    cv_df = cross_validate(X, y)
    print("\n10-fold CV results:")
    print(cv_df.groupby("method").agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                       n_features=("n_features", "mean")))

    best, retained_block_ids, block_words = full_data_fit(X, y, vocab)
    print(f"\nFull-data em_filter: retained {best.n_features} word blocks (of "
          f"{len(np.unique(blocks_from_correlation_threshold(X, rho=CORR_RHO)))} total)")
    print("\nTop blocks' words:")
    order = np.argsort(-best.theta_hat)[:10]
    for i in order:
        b = retained_block_ids[i]
        print(f"  block{b} (theta={best.theta_hat[i]:.4f}): {block_words.get(b, [])}")

    make_figure(cv_df, best, retained_block_ids, block_words)
