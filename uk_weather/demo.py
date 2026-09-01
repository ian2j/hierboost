"""Counter-test to earthquake_japan/demo.py: same spatial-count-forecasting design
(HierBoostCountRegressor, Poisson/NB spike-and-slab, boosting-prior via real
lon/lat through resolve_affinity), same six-way baseline comparison, same
chronological train/test discipline -- but a domain chosen specifically because its
correlation-generating mechanism (synoptic-scale weather systems moving across a
region over multiple days) is physically stable and slow, unlike earthquake
triggering's fast, erratic point process. If the earthquake result's lesson (an
apparent correlation that doesn't survive an honest train/test split) was really
about *domain* stationarity rather than a flaw in the extension, this is where a
real, generalizing result should show up.

Task: predict a target grid point's UK/Ireland wet-day count in bin t from every
OTHER point's wet-day count over the preceding `window` bins (strictly before t --
same no-leakage discipline as before).
"""
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance

from hierboost.estimator import HierBoostCountRegressor
from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_glm import em_filter_nb

TRAIN_FRAC = 0.8
AFFINITY_BANDWIDTH = 3.0  # degrees -- identical choice to earthquake_japan/demo.py,
                          # deliberately not re-tuned per domain, for a fair comparison


def check_generalization(data_path, label):
    """Learned from earthquake_japan: a raw full-sample or train-only correlation can
    look real and still fail to generalize. Check the single most train-correlated
    predictor's out-of-sample behavior BEFORE trusting anything else."""
    d = np.load(data_path)
    counts, centroids, target_k = d["counts"], d["centroids"], int(d["target_k"])
    other_idx = np.array([k for k in range(counts.shape[1]) if k != target_k])
    y = counts[:, target_k].astype(float)
    X = counts[:, other_idx].astype(float)
    n = len(y)
    n_train = int(n * TRAIN_FRAC)

    train_corr = np.array([np.corrcoef(X[:n_train, j], y[:n_train])[0, 1] for j in range(X.shape[1])])
    best_j = np.nanargmax(train_corr)
    test_corr = np.corrcoef(X[n_train:, best_j], y[n_train:])[0, 1]
    print(f"[{label}] generalization check: best train-correlated point (r={train_corr[best_j]:.3f}) "
          f"-> held-out test r={test_corr:.3f}")
    return train_corr[best_j], test_corr


def run(data_path, window, label):
    d = np.load(data_path)
    counts, centroids, target_k = d["counts"], d["centroids"], int(d["target_k"])
    n_bins, K = counts.shape
    other_idx = np.array([k for k in range(K) if k != target_k])

    y_all = counts[window:, target_k].astype(float)
    X_all = np.zeros((n_bins - window, len(other_idx)))
    for w in range(1, window + 1):
        X_all += counts[window - w: n_bins - w][:, other_idx]
    offset_all = np.log1p(X_all.sum(axis=1))
    n = y_all.shape[0]
    p = X_all.shape[1]

    n_train = int(n * TRAIN_FRAC)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    off_train, off_test = offset_all[:n_train], offset_all[n_train:]
    print(f"\n{'#'*78}\n[{label}] window={window} bin(s), n={n} total "
          f"({n_train} train / {n - n_train} test, chronological split), p={p} predictor points")
    print(f"[{label}] target point: ({centroids[target_k,0]:.1f}E, {centroids[target_k,1]:.1f}N), "
          f"train mean={y_train.mean():.3f}, test mean={y_test.mean():.3f}")

    other_centroids = centroids[other_idx]
    target_centroid = centroids[target_k:target_k + 1]

    results = {}

    train_rate = y_train.sum() / np.expm1(off_train).sum()
    mu_naive = train_rate * np.expm1(off_test)
    results["naive (constant rate)"] = (mu_naive, p)

    Xc_train = sm.add_constant(X_train)
    Xc_test = sm.add_constant(X_test, has_constant="add")
    pois = sm.GLM(y_train, Xc_train, family=sm.families.Poisson(), offset=off_train).fit()
    mu_pois = pois.predict(Xc_test, offset=off_test)
    results[f"Poisson GLM (dense, {p} features)"] = (mu_pois, p)

    nb = sm.NegativeBinomial(y_train, Xc_train, offset=off_train).fit(disp=0)
    mu_nb = nb.predict(Xc_test, offset=off_test)
    results[f"NB GLM (dense, {p} features)"] = (mu_nb, p)

    rf = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=2)
    rf.fit(np.column_stack([X_train, off_train]), y_train)
    mu_rf = np.clip(rf.predict(np.column_stack([X_test, off_test])), 1e-6, None)
    results[f"Random Forest ({p} features + lag-total)"] = (mu_rf, p)

    m_flat = HierBoostCountRegressor(family="negbinomial", fit_method="em_filter",
                                      min_features=5, filter_frac=0.2)
    m_flat.fit(X_train, y_train, offset=off_train)
    mu_flat = m_flat.predict(X_test, offset_new=off_test)
    results["hierboost NB, flat prior"] = (np.asarray(mu_flat), len(m_flat.names_))

    wr_raw = resolve_affinity(other_centroids, kind="gaussian", group_coords=target_centroid,
                               bandwidth=AFFINITY_BANDWIDTH)
    X_design_train = np.column_stack([np.ones(n_train), X_train])
    filt = em_filter_nb(X_design_train, y_train, wr_raw, xi0=-2.0, xi1=2.0, kappa=100.0,
                         nu=1.0, lam=1.0, offset=off_train, filter_frac=0.2, min_features=5)
    best = filt.best
    retained = best.retained_idx
    X_design_test = np.column_stack([np.ones(X_test.shape[0]), X_test[:, retained]])
    mu_boost = np.exp(np.clip(X_design_test @ best.beta + off_test, -30, 30))
    results["hierboost NB, spatial-proximity boost"] = (mu_boost, len(retained))

    print("-" * 78)
    print(f"{'model':<40}{'Poisson deviance':>16}{'corr(pred,actual)':>18}{'n_feat':>6}")
    print("-" * 78)
    for name, (mu, n_feat) in results.items():
        mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
        dev = mean_poisson_deviance(y_test, mu)
        corr = np.corrcoef(mu, y_test)[0, 1]
        print(f"{name:<40}{dev:>16.4f}{corr:>18.4f}{n_feat:>6d}")
    print("-" * 78)

    dists = np.sqrt(((other_centroids[retained] - target_centroid) ** 2).sum(axis=1))
    all_dists = np.sqrt(((other_centroids - target_centroid) ** 2).sum(axis=1))
    print(f"[{label}] spatial-boost model retained {len(retained)} of {p} points; "
          f"distance to target (deg): retained mean={dists.mean():.2f}, "
          f"all-point mean={all_dists.mean():.2f}")
    return results


if __name__ == "__main__":
    check_generalization("uk_weather/data/weekly_wetdays.npz", "weekly")
    check_generalization("uk_weather/data/daily_wetdays.npz", "daily")
    run("uk_weather/data/weekly_wetdays.npz", window=1, label="weekly, lag-1")
    run("uk_weather/data/daily_wetdays.npz", window=3, label="daily, trailing 3-day window")
    run("uk_weather/data/daily_wetdays.npz", window=1, label="daily, lag-1")
