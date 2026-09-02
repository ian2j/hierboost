"""Prospective v2 re-test of the block-latent decorrelation criterion: real SNOTEL
snow-water-equivalent network in Colorado's San Juan Mountains (18 clean stations,
1990-2024). Predicted WORK, but checked against two new pre-fit conditions (max
cross-station correlation, target's own autocorrelation) before trusting it -- computed
and reported FIRST, per the whole point of this re-test."""
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_gaussian import em_filter_gaussian

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "predom_snowpack", "data")
RAW_JSON = os.path.join(DATA_DIR, "san_juan_wteq_raw.json")
PROCESSED_NPZ = os.path.join(DATA_DIR, "daily_swe.npz")
RESULTS_JSON = os.path.join(ROOT, "results", "predom_snowpack.json")
USER_AGENT = "hierboost-research/0.1 (research use; contact: ian2johnston@gmail.com)"

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
BEGIN_DATE = "1990-01-01"
END_DATE = "2024-12-31"
MISSING_FRAC_THRESHOLD = 0.02  # predom_solar's "clean <2% missing" precedent
TRAIN_FRAC = 0.8
SATURATION_THRESHOLD = 0.9   # condition 2, v2
PERSISTENCE_THRESHOLD = 0.3  # condition 5, v2

# San Juan Mountains (SW Colorado) SNTL stations -- verified live 2026-08-30 via
# GET .../services/v1/stations?stationTriplets=*:CO:SNTL&activeOnly=true (118 active
# CO stations returned; these 19 form a real, geographically contiguous cluster
# within the San Juans, all with records beginning 1978-1990).
# name -> (stationTriplet, lon, lat, elevation_ft)
CANDIDATES = {
    "LilyPond":       ("580:CO:SNTL", -106.54823, 37.38028, 11070),
    "WolfCreekSmt":   ("874:CO:SNTL", -106.80234, 37.47903, 10930),
    "UpperSanJuan":   ("840:CO:SNTL", -106.83528, 37.48563, 10140),
    "StumpLakes":     ("797:CO:SNTL", -107.63348, 37.47647, 11230),
    "Vallecito":      ("843:CO:SNTL", -107.50748, 37.48524, 10740),
    "MiddleCreek":    ("624:CO:SNTL", -107.03932, 37.61779, 11260),
    "ScotchCreek":    ("739:CO:SNTL", -108.00833, 37.64562, 9160),
    "Cascade2":       ("387:CO:SNTL", -107.80287, 37.65751, 8990),  # dropped, >2% missing
    "SpudMountain":   ("780:CO:SNTL", -107.77841, 37.69883, 10660),
    "Beartown":       ("327:CO:SNTL", -107.51240, 37.71433, 11580),
    "UpperRioGrande": ("839:CO:SNTL", -107.25971, 37.72172, 9370),
    "MolasLake":      ("632:CO:SNTL", -107.68933, 37.74929, 10610),
    "ElDientePeak":   ("465:CO:SNTL", -108.02235, 37.78607, 10210),
    "LizardHeadPass": ("586:CO:SNTL", -107.92475, 37.79895, 10190),
    "MineralCreek":   ("629:CO:SNTL", -107.72657, 37.84737, 10030),
    "RedMtnPass":     ("713:CO:SNTL", -107.71389, 37.89168, 11060),
    "LoneCone":       ("589:CO:SNTL", -108.19636, 37.89169, 9730),
    "Idarado":        ("538:CO:SNTL", -107.67620, 37.93389, 9780),
    "Slumgullion":    ("762:CO:SNTL", -107.20392, 37.99076, 11560),
}
TARGET_RANK = 2  # 0-indexed: 3rd-highest of the cleaned pool by mean WTEQ


def fetch_raw():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_JSON):
        print(f"[fetch] {RAW_JSON} already present, skipping download")
        return
    triplets = ",".join(v[0] for v in CANDIDATES.values())
    params = {
        "stationTriplets": triplets,
        "elements": "WTEQ",
        "duration": "DAILY",
        "beginDate": BEGIN_DATE,
        "endDate": END_DATE,
    }
    url = AWDB_BASE + "?" + urllib.parse.urlencode(params)
    print(f"[fetch] downloading {len(CANDIDATES)} stations' daily WTEQ, "
          f"{BEGIN_DATE}..{END_DATE}, from {AWDB_BASE}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    with open(RAW_JSON, "wb") as f:
        f.write(data)
    print(f"[fetch] saved {os.path.getsize(RAW_JSON)/1e6:.2f} MB -> {RAW_JSON}")


def build_daily_matrix():
    """Raw AWDB JSON -> rectangular (day x station) SWE matrix, dropping any station
    over the MISSING_FRAC_THRESHOLD and restricting to the common fully-covered date
    range across the kept stations -- same "no imputation, use the common
    well-covered window/set" discipline as uk_weather/predom_solar/predom_airquality."""
    if os.path.exists(PROCESSED_NPZ):
        print(f"[build] {PROCESSED_NPZ} already present, skipping rebuild")
        return
    with open(RAW_JSON) as f:
        raw = json.load(f)

    name_by_triplet = {v[0]: k for k, v in CANDIDATES.items()}
    series = {}
    for s in raw:
        trip = s["stationTriplet"]
        name = name_by_triplet[trip]
        if not s["data"]:
            print(f"[build] {name} ({trip}): NO DATA returned, dropping")
            continue
        vals = s["data"][0]["values"]
        series[name] = pd.Series({v["date"]: v.get("value", np.nan) for v in vals})

    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    missing_frac = panel.isna().sum() / len(panel)
    print(f"[build] {panel.shape[1]} stations fetched, {panel.shape[0]} calendar days "
          f"{panel.index.min().date()}..{panel.index.max().date()}")
    print(f"[build] missing fraction per station:\n{missing_frac.sort_values()}")

    keep = missing_frac[missing_frac < MISSING_FRAC_THRESHOLD].index.tolist()
    dropped = [c for c in panel.columns if c not in keep]
    print(f"[build] dropping {dropped} (>= {MISSING_FRAC_THRESHOLD:.0%} missing), "
          f"keeping {len(keep)} stations")

    complete = panel[keep].dropna(how="any").sort_index()
    print(f"[build] common complete-day panel: {complete.shape[0]} days, "
          f"{complete.index.min().date()}..{complete.index.max().date()}, "
          f"{len(keep)} stations, zero imputation")

    stations = keep
    values = complete[stations].to_numpy()
    days = np.array([str(d.date()) for d in complete.index])
    centroids = np.array([[CANDIDATES[s][1], CANDIDATES[s][2]] for s in stations])
    elevations = np.array([CANDIDATES[s][3] for s in stations])

    # pre-registered target pick: 3rd-highest of the cleaned pool by mean WTEQ across
    # the full sample -- decided before any model is fit, same rule family as
    # earthquake_japan/uk_weather/predom_solar/predom_airquality's "3rd-highest/most
    # active of the pool" convention.
    means = values.mean(axis=0)
    order = np.argsort(means)[::-1]
    target_k = int(order[TARGET_RANK])
    print(f"[build] mean WTEQ by station (top 6): "
          f"{sorted(zip(stations, means.round(2)), key=lambda x: -x[1])[:6]}")
    print(f"[build] target station = {stations[target_k]} "
          f"({CANDIDATES[stations[target_k]][0]}), rank {TARGET_RANK+1} of {len(stations)} "
          f"pool stations by mean WTEQ")

    np.savez(PROCESSED_NPZ, values=values, centroids=centroids, elevations=elevations,
             stations=np.array(stations), days=days, target_k=target_k)
    print(f"[build] saved -> {PROCESSED_NPZ}")


def make_lagged(values, target_k, window):
    """X[t] = trailing `window`-day MEAN of every OTHER station's daily SWE ending at
    day t-1 (strictly before t); y[t] = target station's SWE at day t. Averaged, not
    summed -- SWE is a level/state variable (like PM2.5's concentration), not an
    additive flow quantity (unlike solar generation totals or precipitation counts)."""
    n_days, K = values.shape
    other_idx = np.array([k for k in range(K) if k != target_k])
    y_all = values[window:, target_k]
    X_all = np.zeros((n_days - window, len(other_idx)))
    for w in range(1, window + 1):
        X_all += values[window - w: n_days - w][:, other_idx]
    X_all /= window
    return X_all, y_all, other_idx


def condition2_precheck(values, stations, label):
    """v2 CONDITION 2 PRE-CHECK (saturation), run BEFORE any fitting: same-day
    (lag-0) pairwise correlation matrix among the raw candidate stations' SWE series,
    computed directly from the raw fetched panel. Reports the maximum off-diagonal
    value. Per the document's rule of thumb, a value above ~0.9 is a real warning
    sign of near-saturation (the streamflow miss's failure mode) -- flagged plainly
    here regardless of which way it points, not used to justify skipping the rest of
    the pipeline."""
    corr = np.corrcoef(values.T)
    n = corr.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    offdiag_vals = corr[offdiag_mask]
    max_offdiag = float(np.max(offdiag_vals))
    median_offdiag = float(np.median(offdiag_vals))
    frac_above_09 = float((offdiag_vals > SATURATION_THRESHOLD).mean())
    i, j = np.unravel_index(np.argmax(np.where(offdiag_mask, corr, -np.inf)), corr.shape)

    flagged = max_offdiag > SATURATION_THRESHOLD
    print(f"\n[{label}] CONDITION-2 PRE-CHECK (saturation, computed on raw data, "
          f"no fitting): max same-day pairwise corr = {max_offdiag:.4f} "
          f"(between {stations[i]} and {stations[j]}), median = {median_offdiag:.4f}, "
          f"{frac_above_09:.1%} of all pairs exceed {SATURATION_THRESHOLD:g}.")
    if flagged:
        print(f"[{label}] >>> WARNING: max off-diagonal ({max_offdiag:.4f}) exceeds the "
              f"~{SATURATION_THRESHOLD:g} rule-of-thumb threshold -- this is a real "
              f"saturation warning sign (the same failure mode that sank the streamflow "
              f"domain, max gauge-to-gauge corr 0.995), regardless of how 'physically "
              f"appropriate' snowpack looks as a domain class. Proceeding to the full "
              f"pipeline anyway per protocol (rule of thumb, not an absolute gate), but "
              f"this materially downgrades confidence in the WORK prediction before any "
              f"model has been fit.")
    else:
        print(f"[{label}] max off-diagonal is below the {SATURATION_THRESHOLD:g} "
              f"threshold -- no saturation warning from this check.")
    return dict(max_offdiag_corr=max_offdiag, median_offdiag_corr=median_offdiag,
                frac_pairs_above_0_9=frac_above_09,
                most_correlated_pair=[str(stations[i]), str(stations[j])],
                flagged_saturated=bool(flagged), threshold=SATURATION_THRESHOLD)


def condition5_precheck(values, target_k, stations, label):
    """v2 CONDITION 5 PRE-CHECK (persistence vs. forecast lag), run BEFORE any
    fitting: target station's own lag-1 autocorrelation, computed directly from the
    raw fetched series, compared against the same-day cross-station correlation from
    condition 2. Per the document's rule of thumb, a SMALL lag-1 autocorrelation
    (<~0.3) alongside LARGE same-day cross-station correlation is the exact signature
    that sank the air-quality domain (real spatial correlation that doesn't survive a
    1-day-ahead forecast) -- flagged plainly here regardless of which way it points."""
    y = values[:, target_k]
    lag1_autocorr = float(np.corrcoef(y[1:], y[:-1])[0, 1])

    other_idx = np.array([k for k in range(values.shape[1]) if k != target_k])
    same_day_corr = np.array([np.corrcoef(values[:, target_k], values[:, k])[0, 1]
                               for k in other_idx])
    max_cross_corr = float(np.max(same_day_corr))
    mean_cross_corr = float(np.mean(same_day_corr))

    signature_flagged = lag1_autocorr < PERSISTENCE_THRESHOLD and max_cross_corr > SATURATION_THRESHOLD
    print(f"\n[{label}] CONDITION-5 PRE-CHECK (persistence vs. forecast lag, computed "
          f"on raw data, no fitting): target ({stations[target_k]}) own lag-1 "
          f"autocorrelation = {lag1_autocorr:.4f}; target's same-day cross-station "
          f"correlation: max = {max_cross_corr:.4f}, mean = {mean_cross_corr:.4f}.")
    if signature_flagged:
        print(f"[{label}] >>> WARNING: lag-1 autocorrelation ({lag1_autocorr:.4f}) is "
              f"small relative to the strong same-day cross-station correlation "
              f"({max_cross_corr:.4f}) -- this is the exact air-quality failure "
              f"signature: real spatial structure that will not survive into a "
              f"1-day-ahead forecast.")
    else:
        print(f"[{label}] lag-1 autocorrelation ({lag1_autocorr:.4f}) is "
              f"{'comparable to or larger than' if lag1_autocorr >= max_cross_corr else 'not small relative to'} "
              f"the same-day cross-station correlation -- no air-quality-type "
              f"persistence warning from this check; SWE's own day-to-day persistence "
              f"looks strong, as the domain's mechanism predicted.")
    return dict(target_lag1_autocorr=lag1_autocorr, target_max_same_day_cross_corr=max_cross_corr,
                target_mean_same_day_cross_corr=mean_cross_corr,
                flagged_air_quality_signature=bool(signature_flagged),
                persistence_threshold=PERSISTENCE_THRESHOLD)


def check_generalization(values, target_k, stations, label):
    """PRE-REGISTERED GATE (condition 4), run after the two v2 pre-checks above but
    before any downstream model fitting: does the single best train-correlated raw
    lag-1 predictor's relationship survive a held-out, chronological split? Same-sign
    and held-out |corr| >= 50% of train |corr| required to pass."""
    X_all, y_all, other_idx = make_lagged(values, target_k, window=1)
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    train_corr = np.array([np.corrcoef(X_all[:n_train, j], y_all[:n_train])[0, 1]
                            for j in range(X_all.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = np.corrcoef(X_all[n_train:, best_j], y_all[n_train:])[0, 1]
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    passes = bool(same_sign and ratio >= 0.5)
    print(f"[{label}] generalization gate: best train-correlated station = "
          f"{stations[other_idx[best_j]]} (train r={train_corr[best_j]:+.4f}) -> "
          f"held-out test r={test_corr:+.4f}  (ratio={ratio:.3f}, same_sign={same_sign}) "
          f"-> {'PASS' if passes else 'FAIL'}")
    return dict(best_predictor_station=str(stations[other_idx[best_j]]),
                train_corr=float(train_corr[best_j]), test_corr=float(test_corr),
                ratio=float(ratio), same_sign=bool(same_sign), passes=passes)


def run_models(values, centroids, target_k, stations, window, kappa=100.0, bandwidth=None,
                verbose_header=True):
    X_all, y_all, other_idx = make_lagged(values, target_k, window)
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    p = X_train.shape[1]

    other_centroids = centroids[other_idx]
    target_centroid = centroids[target_k:target_k + 1]
    other_dist = np.sqrt(((other_centroids - target_centroid) ** 2).sum(axis=1))

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    results = {}

    naive_mu = np.full_like(y_test, y_train.mean())
    results["naive (train mean)"] = (naive_mu, 0)

    lasso = LassoCV(cv=5, max_iter=20000, random_state=0).fit(Xtr_s, y_train)
    results[f"Lasso/L1 (raw, {p} stations)"] = (lasso.predict(Xte_s), int((lasso.coef_ != 0).sum()))

    K_pca = min(8, p - 1)
    pca = PCA(n_components=K_pca, random_state=0).fit(Xtr_s)
    lr = sm.OLS(y_train, sm.add_constant(pca.transform(Xtr_s))).fit()
    pred_pca = lr.predict(sm.add_constant(pca.transform(Xte_s), has_constant="add"))
    results[f"PCA({K_pca})+linear"] = (pred_pca, K_pca)

    rf = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=2).fit(X_train, y_train)
    results[f"Random Forest (raw, {p} stations)"] = (rf.predict(X_test), p)

    Xd_train = np.column_stack([np.ones(n_train), Xtr_s])
    Xd_test = np.column_stack([np.ones(len(y_test)), Xte_s])

    # hierboost's spike-slab priors are calibrated for roughly unit-scale responses
    # (same lesson as predom_solar/predom_airquality): standardize y for the fit,
    # rescale predictions back.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std

    wr_flat = np.ones(p)
    filt_flat = em_filter_gaussian(Xd_train, y_train_s, wr_flat, xi0=-2.0, xi1=0.0, kappa=kappa,
                                    nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
    bf = filt_flat.best
    mu_flat = (Xd_test[:, np.concatenate([[0], bf.retained_idx + 1])] @ bf.beta) * y_std + y_mean
    results[f"hierboost flat prior (kappa={kappa:g})"] = (mu_flat, len(bf.retained_idx))

    wr_spatial = resolve_affinity(other_centroids, kind="gaussian", group_coords=target_centroid,
                                   bandwidth=bandwidth, X=X_train)
    filt_sp = em_filter_gaussian(Xd_train, y_train_s, wr_spatial, xi0=-2.0, xi1=2.0, kappa=kappa,
                                  nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
    bs = filt_sp.best
    mu_sp = (Xd_test[:, np.concatenate([[0], bs.retained_idx + 1])] @ bs.beta) * y_std + y_mean
    results[f"hierboost spatial-proximity boost (kappa={kappa:g})"] = (mu_sp, len(bs.retained_idx))

    if verbose_header:
        print(f"\n{'#'*88}\n[window={window}d] n={n} days ({n_train} train / {n - n_train} test, "
              f"chronological), p={p} predictor stations, target={stations[target_k]}")
        print(f"[window={window}d] target train mean={y_train.mean():.2f} in SWE, "
              f"test mean={y_test.mean():.2f} in SWE")
        print("-" * 88)
        print(f"{'model':<42}{'test R2':>10}{'corr':>10}{'n_feat':>8}")
        print("-" * 88)
    row_results = {}
    for name, (mu, nf) in results.items():
        r2 = r2_score(y_test, mu)
        corr = np.corrcoef(mu, y_test)[0, 1] if np.std(mu) > 1e-10 else float("nan")
        print(f"{name:<42}{r2:>10.4f}{corr:>10.4f}{nf:>8d}")
        row_results[name] = dict(r2=float(r2), corr=float(corr) if np.isfinite(corr) else None, n_features=int(nf))

    rho_flat, _ = spearmanr(bf.theta_hat, other_dist)
    rho_sp, _ = spearmanr(bs.theta_hat, other_dist)
    top_n = min(5, len(bs.theta_hat))
    top5 = stations[other_idx[np.argsort(-bs.theta_hat)[:top_n]]]
    print(f"[window={window}d] mechanism check: theta_hat vs distance-to-target spearman rho -- "
          f"flat prior={rho_flat:+.3f}, spatial boost={rho_sp:+.3f}; "
          f"spatial-boost top-{top_n} by theta_hat: {list(top5)}")

    return dict(window=window, n=n, n_train=n_train, n_test=n - n_train, p=p,
                results=row_results,
                mechanism_check=dict(rho_theta_vs_distance_flat=float(rho_flat),
                                      rho_theta_vs_distance_spatial=float(rho_sp),
                                      spatial_top_by_theta=[str(z) for z in top5]),
                hierboost_flat_n_retained=len(bf.retained_idx),
                hierboost_spatial_n_retained=len(bs.retained_idx))


def sparsity_sweep(values, centroids, target_k, stations, window=1):
    """FAIL-rule check #3: does em_filter's sparsity ever meaningfully engage across a
    reasonable kappa sweep, or does it retain >90% of candidate stations regardless?"""
    X_all, y_all, other_idx = make_lagged(values, target_k, window)
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    p = X_train.shape[1]
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xd_train = np.column_stack([np.ones(n_train), Xtr_s])
    other_centroids = centroids[other_idx]
    target_centroid = centroids[target_k:target_k + 1]
    wr_spatial = resolve_affinity(other_centroids, kind="gaussian", group_coords=target_centroid,
                                   bandwidth=None, X=X_train)
    y_train_s = (y_train - y_train.mean()) / y_train.std()

    print(f"\n[sweep] em_filter feature retention vs kappa (p={p} candidate stations):")
    sweep = []
    for kap in [10.0, 100.0, 1000.0, 10000.0]:
        filt = em_filter_gaussian(Xd_train, y_train_s, wr_spatial, xi0=-2.0, xi1=2.0, kappa=kap,
                                   nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)
        n_ret = len(filt.best.retained_idx)
        frac = n_ret / p
        print(f"  kappa={kap:<10g} retained {n_ret}/{p} ({frac:.1%})")
        sweep.append(dict(kappa=kap, n_retained=n_ret, frac_retained=float(frac)))
    max_frac = max(s["frac_retained"] for s in sweep)
    return sweep, max_frac


def apply_outcome_rule(gate, main_run, sweep_max_frac):
    """Applies preregistration_10domain_test.md's WORK/FAIL/MIXED rule EXACTLY as
    written to this run's actual numbers -- no discretion."""
    reasons = []

    gate_pass = gate["passes"]
    reasons.append(f"gate: {'PASS' if gate_pass else 'FAIL'} (same_sign={gate['same_sign']}, "
                    f"ratio={gate['ratio']:.3f} {'>=' if gate['ratio']>=0.5 else '<'} 0.5)")

    r = main_run["results"]
    best_conv = max(max(v["r2"] for k, v in r.items() if k.startswith("Lasso")),
                     max(v["r2"] for k, v in r.items() if k.startswith("PCA")),
                     max(v["r2"] for k, v in r.items() if k.startswith("Random Forest")))
    hb_flat_r2 = max(v["r2"] for k, v in r.items() if k.startswith("hierboost flat"))
    hb_sp_r2 = max(v["r2"] for k, v in r.items() if k.startswith("hierboost spatial"))
    hb_best_r2 = max(hb_flat_r2, hb_sp_r2)
    naive_r2 = r["naive (train mean)"]["r2"]
    p = main_run["p"]
    n_feat_hb = min(main_run["hierboost_flat_n_retained"], main_run["hierboost_spatial_n_retained"])
    retention_frac = n_feat_hb / p

    close_to_best = (best_conv - hb_best_r2) <= 0.05
    sparse_enough = retention_frac <= 0.5
    rule2a = close_to_best and sparse_enough

    beats_naive_by_margin = (hb_best_r2 - naive_r2) >= 0.05
    mech = main_run["mechanism_check"]
    mechanism_validated = abs(mech["rho_theta_vs_distance_spatial"]) > abs(mech["rho_theta_vs_distance_flat"]) \
        and mech["rho_theta_vs_distance_spatial"] < -0.1  # closer stations (smaller distance) -> higher theta
    rule2b = beats_naive_by_margin and mechanism_validated

    reasons.append(f"2a: close_to_best_conventional={close_to_best} (best_conv_r2={best_conv:.4f}, "
                    f"hb_best_r2={hb_best_r2:.4f}, gap={best_conv-hb_best_r2:.4f}) AND "
                    f"sparse<=50%={sparse_enough} (retained {n_feat_hb}/{p}={retention_frac:.1%}) -> {rule2a}")
    reasons.append(f"2b: beats_naive_by_margin={beats_naive_by_margin} (hb_best_r2={hb_best_r2:.4f} vs "
                    f"naive_r2={naive_r2:.4f}, margin={hb_best_r2-naive_r2:.4f}) AND "
                    f"mechanism_validated={mechanism_validated} (rho flat={mech['rho_theta_vs_distance_flat']:+.3f}, "
                    f"rho spatial={mech['rho_theta_vs_distance_spatial']:+.3f}) -> {rule2b}")

    worse_than_naive = hb_best_r2 < naive_r2
    sparsity_never_engages = sweep_max_frac > 0.9 and not mechanism_validated

    reasons.append(f"FAIL-2: hierboost worse than naive: {worse_than_naive} "
                    f"(hb_best_r2={hb_best_r2:.4f} vs naive_r2={naive_r2:.4f})")
    reasons.append(f"FAIL-3: sparsity never engages (max retained frac across sweep={sweep_max_frac:.1%} > 90%) "
                    f"AND no mechanism validation: {sparsity_never_engages}")

    if not gate_pass:
        verdict = "FAIL"
        reasons.append("-> FAIL rule 1 triggered (gate failed): verdict is FAIL regardless of downstream fit.")
    elif worse_than_naive or sparsity_never_engages:
        verdict = "FAIL"
        reasons.append("-> a FAIL condition (2 or 3) is triggered despite the gate passing: verdict is FAIL.")
    elif rule2a or rule2b:
        verdict = "WORK"
        reasons.append("-> gate passed AND (2a or 2b) satisfied, no FAIL condition triggered: verdict is WORK.")
    else:
        verdict = "MIXED"
        reasons.append("-> gate passed, no FAIL condition triggered, but neither 2a nor 2b cleanly satisfied: MIXED.")

    return verdict, reasons, dict(best_conventional_r2=float(best_conv), hierboost_flat_r2=float(hb_flat_r2),
                                    hierboost_spatial_r2=float(hb_sp_r2), naive_r2=float(naive_r2),
                                    hierboost_n_retained=int(n_feat_hb), p=int(p),
                                    retention_frac=float(retention_frac))


def assess_v2_prediction(cond2, cond5, verdict):
    """Would the two v2 pre-checks ALONE, computed before any fitting, have predicted
    this domain's actual outcome? This -- not the raw verdict -- is the thing this
    prospective re-test is actually for."""
    lines = []
    if cond2["flagged_saturated"] and not cond5["flagged_air_quality_signature"]:
        precheck_read = ("condition-2 (saturation) flags a real warning; condition-5 "
                          "(persistence) does NOT -- i.e. the pre-checks predict a "
                          "streamflow-style saturation risk, not an air-quality-style "
                          "persistence-lag risk.")
        precheck_predicts_fail_risk = True
    elif cond5["flagged_air_quality_signature"] and not cond2["flagged_saturated"]:
        precheck_read = ("condition-5 (persistence) flags a real warning; condition-2 "
                          "(saturation) does NOT -- the pre-checks predict an "
                          "air-quality-style persistence-lag risk, not a streamflow-style "
                          "saturation risk.")
        precheck_predicts_fail_risk = True
    elif cond2["flagged_saturated"] and cond5["flagged_air_quality_signature"]:
        precheck_read = "BOTH pre-checks flag a warning -- the pre-checks predict a high FAIL/MIXED risk."
        precheck_predicts_fail_risk = True
    else:
        precheck_read = "NEITHER pre-check flags a warning -- the pre-checks predict the domain should cleanly WORK."
        precheck_predicts_fail_risk = False

    consistent = (precheck_predicts_fail_risk and verdict in ("FAIL", "MIXED")) or \
                 (not precheck_predicts_fail_risk and verdict == "WORK")
    lines.append(precheck_read)
    lines.append(f"Actual full-pipeline verdict: {verdict}.")
    lines.append(f"Pre-checks alone {'WOULD' if consistent else 'would NOT'} have correctly "
                 f"anticipated this outcome ahead of running the full pipeline.")
    return dict(precheck_read=precheck_read, precheck_predicts_fail_risk=bool(precheck_predicts_fail_risk),
                consistent_with_full_pipeline=bool(consistent), notes=lines)


def main():
    fetch_raw()
    build_daily_matrix()
    d = np.load(PROCESSED_NPZ, allow_pickle=True)
    values, centroids, stations, target_k = (d["values"], d["centroids"], d["stations"],
                                               int(d["target_k"]))

    print("\n" + "=" * 88)
    print("STEP 0: v2 PRE-CHECKS (computed on RAW data, BEFORE any model is fit)")
    print("=" * 88)
    cond2 = condition2_precheck(values, stations, label="snowpack (San Juan SNTL)")
    cond5 = condition5_precheck(values, target_k, stations, label="snowpack (San Juan SNTL)")

    print("\n" + "=" * 88)
    print("STEP 1: GENERALIZATION GATE (condition 4, run before any other model result)")
    print("=" * 88)
    gate = check_generalization(values, target_k, stations, label="snowpack (lag-1)")

    print("\n" + "=" * 88)
    print("STEP 2: hierboost vs. baseline suite, held-out chronological split")
    print("=" * 88)
    main_run = run_models(values, centroids, target_k, stations, window=1)
    print()
    secondary_run = run_models(values, centroids, target_k, stations, window=3)

    print("\n" + "=" * 88)
    print("STEP 3: sparsity-engagement sweep (FAIL-rule check #3)")
    print("=" * 88)
    sweep, sweep_max_frac = sparsity_sweep(values, centroids, target_k, stations, window=1)

    print("\n" + "=" * 88)
    print("STEP 4: apply the preregistered WORK/FAIL/MIXED rule to the numbers above")
    print("=" * 88)
    verdict, reasons, verdict_numbers = apply_outcome_rule(gate, main_run, sweep_max_frac)
    for r in reasons:
        print(" -", r)
    print(f"\n>>> Snowpack (SNOTEL, San Juan Mountains) VERDICT: {verdict}  "
          f"(preregistered prediction was WORK)")

    print("\n" + "=" * 88)
    print("STEP 5: would the v2 pre-checks ALONE have predicted this outcome?")
    print("=" * 88)
    v2_assessment = assess_v2_prediction(cond2, cond5, verdict)
    for n in v2_assessment["notes"]:
        print(" -", n)

    out = dict(
        domain="Snowpack (SWE) across San Juan Mountains SNOTEL network -- prospective v2 re-test",
        data_source=dict(
            url=AWDB_BASE,
            note=("USDA NRCS AWDB REST API (wcc.sc.egov.usda.gov/awdbRestApi), verified "
                  "live 2026-08-30: HTTP 200, no auth. The alternative 'newer' "
                  "awdb.ars.usda.gov API named in the task does not resolve in this "
                  "environment (connection failure / HTTP 000, checked live) -- the "
                  "wcc.sc.egov.usda.gov endpoint is the one that actually works and is "
                  "the one used here. Parameter names (stationTriplets, elements, "
                  "duration, beginDate/endDate) taken from the API's own OpenAPI spec "
                  "at /awdbRestApi/v3/api-docs, not guessed."),
            begin_date=BEGIN_DATE, end_date=END_DATE, element="WTEQ (snow water equivalent, inches)",
        ),
        mountain_range="San Juan Mountains, southwest Colorado",
        candidate_stations=[v[0] for v in CANDIDATES.values()],
        dropped_stations_missing_data=[k for k, v in CANDIDATES.items() if k == "Cascade2"],
        target_station=str(stations[target_k]),
        predictor_stations=[str(s) for s in stations if s != stations[target_k]],
        n_days_total=int(values.shape[0]),
        n_stations_analysis_pool=int(values.shape[1]),
        condition2_saturation_precheck=cond2,
        condition5_persistence_precheck=cond5,
        generalization_gate=gate,
        main_run_window1=main_run,
        secondary_run_window3=secondary_run,
        sparsity_sweep=sweep,
        sparsity_sweep_max_retained_frac=sweep_max_frac,
        verdict=verdict,
        verdict_reasoning=reasons,
        verdict_numbers=verdict_numbers,
        preregistered_prediction="WORK",
        v2_precheck_assessment=v2_assessment,
    )
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[main] results saved -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
