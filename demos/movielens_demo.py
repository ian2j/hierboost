"""First real dogfooding of HierBoostRegressor: predicts one movie's rating from other
movies' ratings, blocked by genre -- the finance-factor-selection idea with movies
standing in for assets. Target is "Star Wars (1977)", the most-rated movie. Also tries
correlation-threshold blocking as a comparison to genre blocking."""
import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from hierboost.estimator import HierBoostRegressor
from hierboost.blocks import block_membership_lists
from hierboost.structure import blocks_from_correlation_threshold

CORR_RHO = 0.2

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.expanduser("~/movielens_data")
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

TARGET_TITLE = "Star Wars (1977)"
N_CORE_MOVIES = 200
GENRE_NAMES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def fetch_ml100k():
    root = os.path.join(DATA_DIR, "ml-100k")
    if not os.path.exists(os.path.join(root, "u.data")):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Downloading MovieLens 100K from {ML100K_URL} ...")
        with urllib.request.urlopen(ML100K_URL, timeout=60) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                z.extractall(DATA_DIR)
    ratings = pd.read_csv(os.path.join(root, "u.data"), sep="\t",
                           names=["user_id", "item_id", "rating", "ts"])
    items = pd.read_csv(os.path.join(root, "u.item"), sep="|", encoding="latin-1", header=None,
                         names=["item_id", "title", "release_date", "video_release_date", "imdb_url"]
                               + GENRE_NAMES)
    return ratings, items


def build_dense_core(ratings, items, n_core=N_CORE_MOVIES):
    counts = ratings.groupby("item_id").size().sort_values(ascending=False)
    core_ids = counts.head(n_core).index
    target_id = items.loc[items["title"] == TARGET_TITLE, "item_id"].iloc[0]
    if target_id not in core_ids:
        core_ids = core_ids.union([target_id])

    pivot = ratings[ratings["item_id"].isin(core_ids)].pivot_table(
        index="user_id", columns="item_id", values="rating")
    pivot = pivot.dropna(subset=[target_id])  # only users who actually rated the target

    user_mean = pivot.mean(axis=1, skipna=True)
    centered = pivot.sub(user_mean, axis=0).fillna(0.0)  # 0 = "this user's own average"

    y = centered[target_id].values
    X_ids = [c for c in centered.columns if c != target_id]
    X = centered[X_ids].values

    genre_flags = items.set_index("item_id").loc[X_ids, GENRE_NAMES].values
    primary_genre = genre_flags.argmax(axis=1)  # first flagged genre -> non-overlapping block id
    genre_label = [GENRE_NAMES[g] for g in primary_genre]

    titles = items.set_index("item_id").loc[X_ids, "title"].tolist()
    target_title = items.loc[items["item_id"] == target_id, "title"].iloc[0]
    print(f"{len(y)} users rated {target_title!r}; {X.shape[1]} other movies in the dense core "
          f"({len(set(primary_genre))} distinct primary genres)")
    return X, y, np.array(primary_genre), titles, target_title


def cross_validate(X, y, block_id, n_folds=10, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    K = len(np.unique(block_id))
    xi0 = np.log(0.3 / 0.7)  # prior belief ~30% of genre blocks matter for one target movie
    rows = []

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        hb = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                                 fit_method="em_filter", filter_frac=0.2, min_features=5, max_outer=30)
        hb.fit(Xtr, ytr, coords=np.arange(X.shape[1]), block_id=block_id)
        pred_hb = np.asarray(hb.predict(Xte))
        rows.append(dict(method="hierboost (genre block-latent + spike-slab)",
                          r2=r2_score(yte, pred_hb), n_features=len(hb.names_)))

        # correlation-block variant: blocks fit on TRAIN only, same discipline as the
        # block-factor loadings themselves, to avoid leaking test-fold structure
        corr_block_id = blocks_from_correlation_threshold(Xtr, rho=CORR_RHO)
        hb_corr = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                                      fit_method="em_filter", filter_frac=0.2, min_features=5, max_outer=30)
        hb_corr.fit(Xtr, ytr, coords=np.arange(X.shape[1]), block_id=corr_block_id)
        pred_hb_corr = np.asarray(hb_corr.predict(Xte))
        rows.append(dict(method="hierboost (correlation block-latent + spike-slab)",
                          r2=r2_score(yte, pred_hb_corr), n_features=len(hb_corr.names_)))

        ridge = Ridge(alpha=5.0).fit(Xtr, ytr)
        rows.append(dict(method="Ridge (raw movies)", r2=r2_score(yte, ridge.predict(Xte)),
                          n_features=X.shape[1]))

        pca = PCA(n_components=K).fit(Xtr)
        ridge_pca = Ridge(alpha=1.0).fit(pca.transform(Xtr), ytr)
        rows.append(dict(method=f"PCA({K})+Ridge", r2=r2_score(yte, ridge_pca.predict(pca.transform(Xte))),
                          n_features=K))

        rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=seed).fit(Xtr, ytr)
        rows.append(dict(method="Random Forest (raw movies)", r2=r2_score(yte, rf.predict(Xte)),
                          n_features=X.shape[1]))

    return pd.DataFrame(rows)


def full_data_fit(X, y, block_id):
    xi0 = np.log(0.3 / 0.7)
    hb = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                             fit_method="em_filter", filter_frac=0.2, min_features=5, max_outer=30)
    hb.fit(X, y, coords=np.arange(X.shape[1]), block_id=block_id)
    return hb


def make_figure(cv_df, hb_full, target_title):
    # hb_full.names_ are generic "block{genre_index}" labels from the estimator API --
    # relabel with the actual genre name since readability is the whole point here
    block_to_genre = {f"block{g}": name for g, name in enumerate(GENRE_NAMES)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    summary = cv_df.groupby("method").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                           n_mean=("n_features", "mean")).reset_index()
    summary = summary.sort_values("r2_mean")
    y_pos = np.arange(len(summary))
    ax.barh(y_pos, summary["r2_mean"], xerr=summary["r2_std"], color="tab:blue", alpha=0.8)
    for i, (r2, n) in enumerate(zip(summary["r2_mean"], summary["n_mean"])):
        ax.text(r2 + 0.01, i, f"~{n:.0f} features", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["method"], fontsize=9)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"10-fold CV R² predicting {target_title!r} rating (user-centered)")

    ax2 = axes[1]
    names = hb_full.names_
    theta = hb_full.theta_hat_
    order = np.argsort(-theta)
    ax2.barh(np.arange(len(order)), theta[order], color="tab:purple", alpha=0.85)
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels([block_to_genre.get(names[i], names[i]) for i in order], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Posterior P(genre block included | y), full-data em_filter fit")
    ax2.set_title(f"Which genre blocks predict {target_title!r} ratings?")

    fig.suptitle("MovieLens 100K: hierboost genre block-latent + spike-slab vs. standard baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "movielens_demo.png"), dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {OUT}/movielens_demo.png")


if __name__ == "__main__":
    ratings, items = fetch_ml100k()
    X, y, block_id, titles, target_title = build_dense_core(ratings, items)

    cv_df = cross_validate(X, y, block_id)
    print("\n10-fold CV results:")
    print(cv_df.groupby("method").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                       n_features=("n_features", "mean")))

    hb_full = full_data_fit(X, y, block_id)
    print("\n" + hb_full.summary(top_n=len(np.unique(block_id))))

    make_figure(cv_df, hb_full, target_title)
