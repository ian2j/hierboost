"""New domain (regional macroeconomics) and first real-data test of
hierboost.kernels.graph_affinity / resolve_affinity(kind="graph") -- every prior spatial
demo in this project (earthquake_japan, uk_weather, Haxby) used a CONTINUOUS coordinate
kernel (gaussian_affinity_points on lon/lat or a k-means voxel blocking), never a literal
graph adjacency. State-border adjacency (derived from the Census Bureau's authoritative
county_adjacency.txt) is a genuinely different structure: unweighted hop-count on a real
network, not physical distance.

Design choice, made explicit rather than glossed over: raw state unemployment rates all
share a huge common national business-cycle factor (every state's rate rises together in
a national recession), so a naive lag-1 cross-state correlation would mostly just detect
"both states are in the same recession," not genuine REGIONAL spillover. To isolate the
latter -- the actually interesting question for a state-adjacency graph -- every state's
rate is expressed as a DEVIATION from the simple 51-state average at that month before
any correlation or fit is computed. This is the same "don't trust an unexamined
correlation" discipline the earthquake_japan/uk_weather generalization check embodies,
applied to a different confound (shared macro trend, not non-stationarity).

Task: predict a target state's deviation-from-national-average unemployment rate at
month t from every OTHER state's deviation at month t-1 (strict lag, no leakage).
Target state (Colorado) was pre-registered before any model was fit: 3rd-highest
border-degree state in the adjacency graph (see build_dataset.py), the same
"well-connected but not the single most extreme" logic as earthquake_japan's
"3rd-most-active" pick.
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_gaussian import em_filter_gaussian

TRAIN_FRAC = 0.8
DATA = "state_econ/data/dataset.npz"


def load():
    d = np.load(DATA, allow_pickle=True)
    values, states, target_k = d["values"], d["states"], int(d["target_k"])
    national_mean = values.mean(axis=1, keepdims=True)
    dev = values - national_mean  # (T, 51) deviation from national average
    return dev, d["graph_dist"], d["centroids"], states, target_k


def check_generalization(dev, target_k, train_frac=TRAIN_FRAC):
    """Learned from earthquake_japan/uk_weather: check whether the single most
    train-correlated predictor's relationship actually survives a held-out split
    BEFORE trusting anything else built on top of it."""
    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_k])
    y = dev[1:, target_k]
    X = dev[:-1][:, other_idx]  # lag-1
    n = len(y)
    n_train = int(n * train_frac)
    train_corr = np.array([np.corrcoef(X[:n_train, j], y[:n_train])[0, 1] for j in range(X.shape[1])])
    best_j = np.nanargmax(np.abs(train_corr))
    test_corr = np.corrcoef(X[n_train:, best_j], y[n_train:])[0, 1]
    print(f"generalization check: best train-correlated neighbor deviation "
          f"(r={train_corr[best_j]:+.3f}) -> held-out test r={test_corr:+.3f}")
    return train_corr[best_j], test_corr


def run(dev, D_graph, centroids, target_k, graph_bandwidth, kappa, verbose_header=True):
    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_k])
    y_all = dev[1:, target_k]
    X_all = dev[:-1][:, other_idx]
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    other_graph_dist = D_graph[target_k][other_idx]
    other_centroids = centroids[other_idx]
    target_centroid = centroids[target_k]

    results = {}

    naive_mu = np.zeros_like(y_test)  # deviation series is mean~0 by construction
    results["naive (predict zero deviation)"] = (naive_mu, 0)

    Xc_train = sm.add_constant(X_train)
    Xc_test = sm.add_constant(X_test)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X_train, y_train)
    results[f"RidgeCV ({X_train.shape[1]} states)"] = (ridge.predict(X_test), X_train.shape[1])

    rf = RandomForestRegressor(n_estimators=300, random_state=0, max_depth=6)
    rf.fit(X_train, y_train)
    results[f"Random Forest ({X_train.shape[1]} states)"] = (rf.predict(X_test), X_train.shape[1])

    Xd_train = np.column_stack([np.ones(n_train), X_train])
    Xd_test = np.column_stack([np.ones(len(y_test)), X_test])

    flat_wr = np.ones(X_train.shape[1])
    filt_flat = em_filter_gaussian(Xd_train, y_train, flat_wr, xi0=-1.0, xi1=1.0, kappa=kappa,
                                    nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
    b = filt_flat.best
    mu_flat = Xd_test[:, np.concatenate([[0], b.retained_idx + 1])] @ b.beta
    results[f"hierboost flat prior (kappa={kappa:g})"] = (mu_flat, len(b.retained_idx))

    wr_graph = resolve_affinity(other_graph_dist, kind="graph",
                                 distance_matrix=other_graph_dist[:, None],
                                 bandwidth=graph_bandwidth)
    filt_graph = em_filter_gaussian(Xd_train, y_train, wr_graph, xi0=-1.0, xi1=1.0, kappa=kappa,
                                     nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
    b = filt_graph.best
    mu_graph = Xd_test[:, np.concatenate([[0], b.retained_idx + 1])] @ b.beta
    results[f"hierboost graph-adjacency boost (bw={graph_bandwidth:g} hops, kappa={kappa:g})"] = \
        (mu_graph, len(b.retained_idx))

    wr_centroid = resolve_affinity(other_centroids, kind="gaussian", group_coords=target_centroid,
                                    bandwidth=None, X=X_train)
    filt_cent = em_filter_gaussian(Xd_train, y_train, wr_centroid, xi0=-1.0, xi1=1.0, kappa=kappa,
                                    nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
    b = filt_cent.best
    mu_cent = Xd_test[:, np.concatenate([[0], b.retained_idx + 1])] @ b.beta
    results[f"hierboost centroid-distance boost (kappa={kappa:g})"] = (mu_cent, len(b.retained_idx))

    if verbose_header:
        print(f"\n{n} months, {X_train.shape[1]} predictor states, target=k{target_k}, "
              f"{n_train} train / {n - n_train} test (chronological)")
        print("-" * 92)
        print(f"{'model':<52}{'test R2':>10}{'corr':>10}{'n_feat':>8}")
        print("-" * 92)
    for name, (mu, nf) in results.items():
        r2 = r2_score(y_test, mu)
        corr = np.corrcoef(mu, y_test)[0, 1] if np.std(mu) > 1e-10 else float("nan")
        print(f"{name:<52}{r2:>10.4f}{corr:>10.4f}{nf:>8d}")

    # Mechanism check (same discipline as uk_weather's theta_hat-vs-distance check):
    # does the boosting prior at least correctly re-rank POSTERIOR CONFIDENCE toward
    # literal graph neighbors, even when em_filter's greedy removal never actually
    # drops a feature (see project memory -- dense/diffuse signal here, same failure
    # mode em_filter showed on earthquake data)?
    finite = np.isfinite(other_graph_dist)
    rho_flat, _ = spearmanr(filt_flat.best.theta_hat[finite], other_graph_dist[finite])
    rho_graph, _ = spearmanr(filt_graph.best.theta_hat[finite], other_graph_dist[finite])
    top5 = states[other_idx[np.argsort(-filt_graph.best.theta_hat)[:5]]]
    print(f"mechanism check: theta_hat vs hop-distance spearman rho -- flat prior={rho_flat:+.3f}, "
          f"graph boost={rho_graph:+.3f}; graph-boost top-5 by theta_hat: {list(top5)}")
    return results, filt_graph, other_idx


if __name__ == "__main__":
    dev, D_graph, centroids, states, target_k = load()
    print(f"target state: {states[target_k]}")
    check_generalization(dev, target_k)

    print("\n=== main comparison (bandwidth=2 hops, kappa=100) ===")
    run(dev, D_graph, centroids, target_k, graph_bandwidth=2.0, kappa=100.0)

    print("\n=== hyperparameter sweep: graph bandwidth (hops) ===")
    for bw in [1.0, 2.0, 4.0, 8.0]:
        print(f"\n-- bandwidth={bw} hops --")
        run(dev, D_graph, centroids, target_k, graph_bandwidth=bw, kappa=100.0, verbose_header=False)

    print("\n=== hyperparameter sweep: kappa (sparsity prior) ===")
    for kap in [10.0, 100.0, 1000.0, 10000.0]:
        print(f"\n-- kappa={kap:g} --")
        run(dev, D_graph, centroids, target_k, graph_bandwidth=2.0, kappa=kap, verbose_header=False)
