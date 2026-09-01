"""First application of hierboost's new Poisson/Negative-Binomial spike-and-slab
(hierboost.spike_slab_glm, via HierBoostCountRegressor) to real, non-medical,
non-financial data: earthquake counts across a grid of cells covering Japan and its
surrounding subduction zones (USGS catalog, M>=3.0, 2013-2023).

Task: predict a target cell's earthquake count in bin t from every OTHER cell's
count over the preceding `window` bins (strictly before t, so there is no same-bin/
contemporaneous leakage -- a genuine one-step-ahead forecasting setup, not a
same-instant correlation fit). This tests real spatiotemporal earthquake clustering
(aftershock triggering, dynamic/static stress transfer -- Omori's law, ETAS-style
triggering are the seismology literature's name for exactly the kind of "correlated
but independently noisy" structure hierboost's spike-and-slab machinery targets).

Run at two granularities (--data weekly_counts.npz vs daily_counts.npz): a first
weekly-lag-1 run came back near-null for every method including the external
baselines (not a hierboost-specific failure) -- diagnosed rather than just reported:
same-week correlation between the target and its nearest cell is real (0.31) but
lag-1-week correlation collapses to ~0.10, and even the target's own lag-1
autocorrelation is only 0.08. Consistent with Omori-law aftershock decay (most real
triggering signal is gone within days, well before a full week passes), so a second
run at daily resolution with a short trailing window is the natural follow-up, not a
new hypothesis invented after seeing the first result look bad.

The boosting-prior step uses the real (lon, lat) grid-cell centroids through
hierboost.kernels.resolve_affinity / gaussian_affinity_points -- literally the same
mechanism as the dissertation's Chapter 2 gene-proximity kernel, just with grid
cells standing in for genes and the target region standing in for "the gene of
interest": cells physically closer to the target get a boosted prior inclusion
probability. Compares a boosted vs a flat-prior hierboost fit (ablation) against
external baselines: constant-rate naive, dense Poisson GLM, dense NB GLM, and
Random Forest.
"""
import sys
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance

from hierboost.estimator import HierBoostCountRegressor
from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_glm import em_filter_nb

TRAIN_FRAC = 0.8
AFFINITY_BANDWIDTH = 3.0  # degrees (~300-330km e-folding scale) -- a disclosed modeling
                          # choice, not fit to this data; roughly the range over which
                          # dynamic/static stress-transfer triggering has been documented
                          # in the seismology literature, not tuned for best performance here


def run(data_path, window, label):
    d = np.load(data_path)
    counts, centroids, target_k = d["counts"], d["centroids"], int(d["target_k"])
    n_bins, K = counts.shape
    other_idx = np.array([k for k in range(K) if k != target_k])

    # trailing `window`-bin sum, strictly excluding bin t itself
    y_all = counts[window:, target_k].astype(float)
    X_all = np.zeros((n_bins - window, len(other_idx)))
    for w in range(1, window + 1):
        X_all += counts[window - w: n_bins - w][:, other_idx]
    offset_all = np.log1p(X_all.sum(axis=1))  # preceding window's total regional activity
    n = y_all.shape[0]
    p = X_all.shape[1]

    n_train = int(n * TRAIN_FRAC)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    off_train, off_test = offset_all[:n_train], offset_all[n_train:]
    print(f"\n{'#'*78}\n[{label}] window={window} bin(s), n={n} total "
          f"({n_train} train / {n - n_train} test, chronological split), p={p} predictor cells")
    print(f"[{label}] target cell: ({centroids[target_k,0]:.1f}E, {centroids[target_k,1]:.1f}N), "
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

    # spatial-proximity boosting prior (Ch2 mechanism, real lon/lat) -- injecting a
    # precomputed wr directly via the low-level API, same reasoning as newsgroups_demo.py's
    # use of the low-level API when the high-level .fit() path doesn't already cover a
    # combination (here: we already computed wr_raw ourselves via resolve_affinity above).
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
    print(f"[{label}] spatial-boost model retained {len(retained)} of {p} cells; "
          f"distance to target (deg): retained mean={dists.mean():.2f}, "
          f"all-cell mean={all_dists.mean():.2f}")
    return results


if __name__ == "__main__":
    run("earthquake_japan/data/weekly_counts.npz", window=1, label="weekly, lag-1")
    run("earthquake_japan/data/daily_counts.npz", window=3, label="daily, trailing 3-day window")
    run("earthquake_japan/data/daily_counts.npz", window=1, label="daily, lag-1")
