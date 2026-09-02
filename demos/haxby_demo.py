"""fMRI face-vs-house decoding (Haxby 2001): spatially-adjacent, correlated voxels in
ventral temporal cortex play the role of an LD block. Collapses each spatial block into
one latent (gaussian_block_factor) and runs the same spike-and-slab engine used for
GWAS/finance. No informative relevance prior -- isolates the value of block-latent
decorrelation + sparsity alone against raw-voxel baselines."""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from nilearn import datasets
from nilearn.maskers import NiftiMasker

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from hierboost.blocks import block_membership_lists
from hierboost.factor import gaussian_block_factor, project_block_factor
from hierboost.spike_slab import em_filter

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

XI0 = np.log(0.10 / 0.90)   # prior belief ~10% of spatial blocks are truly discriminative
KAPPA, NU, LAM = 100.0, 1.0, 1.0


def load_face_house_data():
    haxby = datasets.fetch_haxby()
    labels = pd.read_csv(haxby.session_target[0], sep=" ")

    masker = NiftiMasker(mask_img=haxby.mask_vt[0], standardize=True, detrend=True)
    X_full = masker.fit_transform(haxby.func[0])  # (n_volumes, n_voxels), z-scored per voxel

    cond = labels["labels"].isin(["face", "house"]).values
    X = X_full[cond]
    y = (labels["labels"].values[cond] == "face").astype(float)
    chunks = labels["chunks"].values[cond]

    mask_img = nib.load(haxby.mask_vt[0])
    mask_data = mask_img.get_fdata().astype(bool)
    ijk = np.array(np.where(mask_data)).T  # matches boolean-index voxel order used above
    xyz = nib.affines.apply_affine(mask_img.affine, ijk)  # mm world coordinates

    print(f"{X.shape[0]} face/house volumes, {X.shape[1]} VT voxels, "
          f"{len(np.unique(chunks))} runs")
    return X, y, chunks, xyz


def build_blocks(xyz, voxels_per_block=8, seed=0):
    """hierboost.blocks' distance-threshold/graph blocking (threshold_blocks_1d/_graph)
    assumes features have natural GAPS to threshold at -- true for genes strung along a
    chromosome or sensors scattered over a field, but VT cortex is one solid, contiguous
    blob of touching voxels with no gaps at any radius >= one voxel spacing, so a
    connected-components threshold collapses the whole mask into a single block (verified:
    it did, degenerating the block-latent factor into a trivial single global PCA
    component). K-means on voxel coordinates is the right tool for partitioning a dense
    volumetric region into compact local patches instead -- the direct 3D analogue of
    tiling an image into spatial patches.
    """
    n_blocks = max(2, xyz.shape[0] // voxels_per_block)
    block_id = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(xyz)
    sizes = np.bincount(block_id)
    print(f"{len(sizes)} spatial blocks (K-means, ~{voxels_per_block} voxels/block target), "
          f"size range [{sizes.min()}, {sizes.max()}], mean {sizes.mean():.1f}")
    return block_id


def fit_block_factors(X_train, X_test, block_id):
    """Fit gaussian_block_factor per block on TRAIN only, project TEST via
    project_block_factor -- avoids leaking test-fold info into the unsupervised
    decorrelation step, the same discipline em_filter already applies to beta/theta."""
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
    K = Ztr.shape[1]
    wr = np.ones(K)

    Xd_train = np.column_stack([np.ones(len(y_train)), Ztr])
    filt = em_filter(Xd_train, y_train, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=NU, lam=LAM,
                      filter_frac=0.2, min_features=5, max_outer=30)
    best = filt.best

    Xd_test = np.column_stack([np.ones(len(y_test)), Zte[:, best.retained_idx]])
    pred = (Xd_test @ best.beta) > 0
    acc = float(np.mean(pred == (y_test > 0.5)))
    return acc, best.n_features


def baseline_folds(X_train, y_train, X_test, y_test, n_pca):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    results = {}

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000)
    lasso.fit(Xtr, y_train)
    results["L1-logistic (raw voxels)"] = (lasso.score(Xte, y_test), int((lasso.coef_ != 0).sum()))

    pca = PCA(n_components=min(n_pca, Xtr.shape[0] - 1)).fit(Xtr)
    log_pca = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr), y_train)
    results[f"PCA({n_pca})+logistic"] = (log_pca.score(pca.transform(Xte), y_test), n_pca)

    svm = SVC(kernel="linear", C=1.0).fit(Xtr, y_train)
    results["Linear SVM (raw voxels)"] = (svm.score(Xte, y_test), Xtr.shape[1])

    return results


def cross_validate(X, y, chunks, block_id, n_blocks):
    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups=chunks):
        acc_hb, n_hb = hierboost_fold(X[train_idx], y[train_idx], X[test_idx], y[test_idx], block_id)
        rows.append(dict(method="hierboost (spatial block-latent + spike-slab)",
                          acc=acc_hb, n_features=n_hb))

        base = baseline_folds(X[train_idx], y[train_idx], X[test_idx], y[test_idx], n_pca=n_blocks)
        for name, (acc, nf) in base.items():
            rows.append(dict(method=name, acc=acc, n_features=nf))
    return pd.DataFrame(rows)


def full_data_inclusion_map(X, y, block_id, xyz):
    """Fit once on ALL face/house volumes (no held-out split) to produce the
    interpretability map -- describing the full-data fit, same role as demo.py's
    single detailed walkthrough vs. its separate Monte Carlo accuracy comparison.

    Uses em_filter (not a single fit_em) so the map reflects the same iterative
    filter-and-refit selection the CV loop actually scores -- a raw one-shot fit_em's
    theta_hat spreads posterior mass thinly across all 58 blocks and every one lands
    below 0.5, which looked like "nothing selected" even though em_filter (used in every
    CV fold) does confidently retain ~12 of them."""
    Z, _, block_ids = fit_block_factors(X, None, block_id)
    wr = np.ones(Z.shape[1])
    Xd = np.column_stack([np.ones(len(y)), Z])
    filt = em_filter(Xd, y, wr, xi0=XI0, xi1=0.0, kappa=KAPPA, nu=NU, lam=LAM,
                      filter_frac=0.2, min_features=5, max_outer=30)
    best = filt.best

    voxel_pi = np.zeros(xyz.shape[0])
    blocks = block_membership_lists(block_id)
    for local_k, global_k in enumerate(best.retained_idx):
        voxel_pi[blocks[block_ids[global_k]]] = best.theta_hat[local_k]
    return voxel_pi, best


def make_figure(voxel_pi, xyz, cv_df):
    fig = plt.figure(figsize=(13, 6))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    retained = voxel_pi > 0
    ax.scatter(xyz[~retained, 0], xyz[~retained, 1], xyz[~retained, 2],
               c="lightgray", s=12, alpha=0.35, label="not retained")
    vmax = voxel_pi[retained].max() if retained.any() else 1.0
    sc = ax.scatter(xyz[retained, 0], xyz[retained, 1], xyz[retained, 2],
                     c=voxel_pi[retained], cmap="viridis", s=35, vmin=0, vmax=vmax,
                     edgecolors="k", linewidths=0.3, label="retained by em_filter")
    ax.set_title(f"em_filter-selected blocks ({retained.sum()}/{len(voxel_pi)} voxels)",
                 fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="P(theta=1), rescaled to selected set")
    ax.legend(fontsize=7, loc="upper left")
    fig.text(0.02, 0.02,
             f"Color = relative posterior weight within the selected set (max P(theta=1)={vmax:.2f});\n"
             "no single block dominates -- consistent with VT-cortex face/house signal being\n"
             "distributed rather than localized, so the *set* of blocks carries the signal.",
             fontsize=8, va="bottom")

    ax2 = fig.add_subplot(1, 2, 2)
    summary = cv_df.groupby("method").agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                           n_mean=("n_features", "mean")).reset_index()
    summary = summary.sort_values("acc_mean")
    y_pos = np.arange(len(summary))
    ax2.barh(y_pos, summary["acc_mean"], xerr=summary["acc_std"], color="tab:blue", alpha=0.8)
    for i, (acc, n) in enumerate(zip(summary["acc_mean"], summary["n_mean"])):
        ax2.text(acc + 0.02, i, f"~{n:.0f} features", va="center", fontsize=8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(summary["method"], fontsize=9)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax2.set_xlabel("Leave-one-run-out CV accuracy (face vs house)")
    ax2.set_xlim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Haxby face-vs-house decoding: hierboost spatial block-latent + spike-slab "
                 "vs. standard baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "haxby_demo.png"), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/haxby_demo.png")


if __name__ == "__main__":
    X, y, chunks, xyz = load_face_house_data()
    block_id = build_blocks(xyz)
    n_blocks = len(np.unique(block_id))

    cv_df = cross_validate(X, y, chunks, block_id, n_blocks)
    print("\nLeave-one-run-out CV results:")
    print(cv_df.groupby("method").agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                       n_features=("n_features", "mean")))

    voxel_pi, full_res = full_data_inclusion_map(X, y, block_id, xyz)
    n_selected = int((voxel_pi > 0).sum())
    max_theta = float(voxel_pi.max()) if n_selected else 0.0
    print(f"\nFull-data em_filter: {n_selected}/{len(voxel_pi)} voxels retained across "
          f"{full_res.n_features} spatial blocks (max individual block P(theta=1)={max_theta:.3f} "
          f"-- confidence stays diffuse across blocks, consistent with the distributed/overlapping "
          f"VT-cortex response pattern the Haxby dataset itself is documented for; the *set* is "
          f"jointly predictive even though no single block is individually confident)")

    make_figure(voxel_pi, xyz, cv_df)
