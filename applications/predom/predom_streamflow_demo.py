"""W3, predicted WORK: Delaware River streamflow, four USGS gauges strictly ordered
downstream (1970-2024, post-reservoir regime). First real-data test of
causal_affinity_1d on physical position rather than calendar time: a lag shorter than
water's physical travel time from a gauge gets zero prior credit, mirroring the
anti-look-ahead mechanism it was built for. Also runs the more familiar
decorrelate="ar1" per-gauge trend-latent path."""
import json
import os
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from hierboost.kernels import causal_affinity_1d
from hierboost.spike_slab_gaussian import em_filter_gaussian
from hierboost.state_space import fit_temporal_block_factor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "streamflow_delaware", "data")
CACHE_PATH = os.path.join(CACHE_DIR, "raw_discharge.json")
RESULTS_PATH = os.path.join(ROOT, "results", "predom_streamflow.json")
USER_AGENT = "hierboost-research/0.1 (research use; contact: ian2johnston@gmail.com)"

# site_no -> (name, lat, lon, drainage_area_sqmi)
SITES = {
    "01434000": dict(name="port_jervis", lat=41.3705833, lon=-74.69713889, da=3076),
    "01438500": dict(name="montague", lat=41.30916667, lon=-74.79527778, da=3480),
    "01446500": dict(name="belvidere", lat=40.82638889, lon=-75.0825, da=4535),
    "01463500": dict(name="trenton", lat=40.22166667, lon=-74.77805556, da=6780),
}
TARGET = "trenton"
UPSTREAM = ["port_jervis", "montague", "belvidere"]  # farthest -> closest to target
START_DATE = "1970-01-01"
LAGS = list(range(0, 11))  # days
TRAIN_FRAC = 0.8
MEANDER_FACTOR = 1.3
ASSUMED_VELOCITY_MPH = 2.0
BANDWIDTH_DAYS = 2.0
KAPPA = 100.0
EM_KWARGS = dict(xi0=-1.0, xi1=1.0, nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)


# ---------------------------------------------------------------- data ----

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def travel_time_days(gauge_name):
    """Physically-motivated (not response-fit) estimate of minimum water travel time
    from `gauge_name` to the target, in days. Independent of any correlation computed
    below, by design -- see module docstring."""
    g = next(v for v in SITES.values() if v["name"] == gauge_name)
    t = SITES["01463500"]
    dist_miles = haversine_miles(g["lat"], g["lon"], t["lat"], t["lon"]) * MEANDER_FACTOR
    return dist_miles / ASSUMED_VELOCITY_MPH / 24.0


def distance_miles(gauge_name):
    g = next(v for v in SITES.values() if v["name"] == gauge_name)
    t = SITES["01463500"]
    return haversine_miles(g["lat"], g["lon"], t["lat"], t["lon"]) * MEANDER_FACTOR


def fetch_site(site_no):
    url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_no}"
           f"&parameterCd=00060&statCd=00003&startDT=1930-01-01&endDT={date.today().isoformat()}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read())
    ts = d["value"]["timeSeries"]
    if not ts:
        return {}
    out = {}
    for v in ts[0]["values"][0]["value"]:
        dt = v["dateTime"][:10]
        try:
            q = float(v["value"])
        except ValueError:
            continue
        if q > 0:
            out[dt] = q
    return out


def fetch_all(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {}
    for site_no, meta in SITES.items():
        print(f"fetching {meta['name']} ({site_no}) from USGS NWIS...")
        data[meta["name"]] = fetch_site(site_no)
        print(f"  {len(data[meta['name']])} daily values")
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)
    return data


def build_dataset():
    raw = fetch_all()
    names = [SITES[s]["name"] for s in SITES]
    df = pd.DataFrame({n: pd.Series(raw[n]) for n in names})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()  # restrict to the common gap-free window
    df = df.loc[START_DATE:]
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    n_before = len(df)
    df = df.reindex(full_idx).ffill()  # fill the handful of missing days (checked: <0.01%)
    print(f"{len(df)} daily observations {df.index.min().date()} -> {df.index.max().date()} "
          f"({n_before} present before gap-fill, {len(df) - n_before} filled)")
    return np.log(df)


def make_lag_features(logdf):
    """Returns X (n, 33), y (n,), and parallel length-33 arrays giving each column's
    originating gauge name and lag (days)."""
    max_lag = max(LAGS)
    n_total = len(logdf)
    n = n_total - max_lag
    cols, col_gauge, col_lag = [], [], []
    for g in UPSTREAM:
        v = logdf[g].values
        for L in LAGS:
            cols.append(v[max_lag - L: n_total - L])
            col_gauge.append(g)
            col_lag.append(L)
    X = np.column_stack(cols)
    y = logdf[TARGET].values[max_lag:]
    return X, y, np.array(col_gauge), np.array(col_lag)


# ------------------------------------------------------- gate & helpers ----

def generalization_gate(X, y, n_train, feature_names):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    train_corr = np.array([np.corrcoef(X_train[:, j], y_train)[0, 1] for j in range(X.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = np.corrcoef(X_test[:, best_j], y_test)[0, 1]
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    passed = bool(same_sign and ratio >= 0.5)
    print(f"\n=== GENERALIZATION GATE (run first, per pre-registration) ===")
    print(f"best train-correlated raw feature: {feature_names[best_j]}  "
          f"train r={train_corr[best_j]:+.4f}  held-out test r={test_corr:+.4f}  "
          f"ratio={ratio:.3f}  same_sign={same_sign}  -> {'PASS' if passed else 'FAIL'}")
    return dict(best_feature=feature_names[best_j], train_corr=float(train_corr[best_j]),
                test_corr=float(test_corr), ratio=float(ratio), same_sign=bool(same_sign),
                passed=passed)


def zscore_fit(X_train, X_test):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


# ---------------------------------------------------------- baselines ----

def run_baselines(X, y, n_train):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    results = {}

    mu_naive = np.full_like(y_test, y_train.mean())
    results["naive (train-mean constant)"] = dict(
        r2=r2_score(y_test, mu_naive), corr=float("nan"), n_features=0)

    lasso = LassoCV(alphas=np.logspace(-4, 0, 30), max_iter=20000, cv=5).fit(Xs_train, y_train)
    pred = lasso.predict(Xs_test)
    results["Lasso (raw lag features)"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]),
        n_features=int((lasso.coef_ != 0).sum()), alpha=float(lasso.alpha_))

    pca = PCA(n_components=len(UPSTREAM)).fit(Xs_train)
    lr = LinearRegression().fit(pca.transform(Xs_train), y_train)
    pred = lr.predict(pca.transform(Xs_test))
    results[f"PCA({len(UPSTREAM)})+linear"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]),
        n_features=len(UPSTREAM))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["Random Forest (raw lag features)"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]),
        n_features=X_train.shape[1])

    return results


# --------------------------------------------------------- hierboost ----

def fit_em(X_train, y_train, wr, kappa=KAPPA):
    Xd_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    return em_filter_gaussian(Xd_train, y_train, wr, kappa=kappa, **EM_KWARGS)


def eval_em(filt, X_test, y_test):
    b = filt.best
    Xd_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
    mu = Xd_test[:, np.concatenate([[0], b.retained_idx + 1])] @ b.beta
    r2 = r2_score(y_test, mu)
    corr = float(np.corrcoef(mu, y_test)[0, 1]) if np.std(mu) > 1e-10 else float("nan")
    return r2, corr, b


def causal_boosted_wr(col_gauge, col_lag, bandwidth=BANDWIDTH_DAYS):
    """Per-gauge causal_affinity_1d boost (see module docstring for the mapping):
    feature coordinate = lag L (days); the single group per gauge = a point interval
    at that gauge's physically-estimated minimum travel time. Computed separately per
    gauge so one gauge's lags are never scored against another gauge's travel time."""
    wr = np.zeros(len(col_gauge))
    tt_by_gauge = {}
    for g in UPSTREAM:
        tt = travel_time_days(g)
        tt_by_gauge[g] = tt
        mask = col_gauge == g
        lags_g = col_lag[mask].astype(float)
        aff = causal_affinity_1d(lags_g, group_l=np.array([tt]), group_r=np.array([tt]),
                                  bandwidth=bandwidth)  # (n_lags_g, 1)
        wr[mask] = aff[:, 0]
    m = wr.max()
    if m > 0:
        wr = wr / m
    return wr, tt_by_gauge


def run_hierboost_flat(X, y, n_train):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    wr = np.ones(X.shape[1])
    filt = fit_em(Xs_train, y_train, wr)
    r2, corr, b = eval_em(filt, Xs_test, y_test)
    return dict(r2=r2, corr=corr, n_features=len(b.retained_idx)), filt


def run_hierboost_causal(X, y, n_train, col_gauge, col_lag, bandwidth=BANDWIDTH_DAYS, kappa=KAPPA):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    wr, tt_by_gauge = causal_boosted_wr(col_gauge, col_lag, bandwidth)
    filt = fit_em(Xs_train, y_train, wr, kappa=kappa)
    r2, corr, b = eval_em(filt, Xs_test, y_test)
    return dict(r2=r2, corr=corr, n_features=len(b.retained_idx),
                travel_time_days=tt_by_gauge, wr=wr.tolist()), filt


def run_hierboost_ar1(logdf, y, n_train, col_gauge, col_lag):
    """decorrelate='ar1' companion: one AR(1) block-latent per gauge, fit on the same
    lag-feature blocks used above. Smoother is fit on the full series before the
    train/test split is applied at the regression stage -- same convention as
    temporal_demo.py / uk_weather_star_demo.py (mild in-sample smoothing look-ahead,
    disclosed, not something this demo introduces)."""
    max_lag = max(LAGS)
    latents = {}
    rhos = {}
    for g in UPSTREAM:
        v = logdf[g].values
        n_total = len(v)
        n = n_total - max_lag
        block = np.column_stack([v[max_lag - L: n_total - L] for L in LAGS])
        mu, sd = block.mean(axis=0), block.std(axis=0)
        sd = np.where(sd < 1e-10, 1.0, sd)
        block_z = (block - mu) / sd
        res = fit_temporal_block_factor(block_z)
        latents[g] = res.z
        rhos[g] = res.rho
    Z = np.column_stack([latents[g] for g in UPSTREAM])
    Z_train, Z_test = Z[:n_train], Z[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    wr = np.ones(Z.shape[1])
    filt = fit_em(Z_train, y_train, wr)
    r2, corr, b = eval_em(filt, Z_test, y_test)
    return dict(r2=r2, corr=corr, n_features=len(b.retained_idx), rho=rhos), filt


# ----------------------------------------------------------- outcome ----

def kappa_sweep(X, y, n_train, wr):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    out = []
    for kap in [10.0, 100.0, 1000.0, 10000.0]:
        filt = fit_em(Xs_train, y_train, wr, kappa=kap)
        r2, corr, b = eval_em(filt, Xs_test, y_test)
        out.append(dict(kappa=kap, r2=r2, n_features=len(b.retained_idx)))
        print(f"  kappa={kap:g}: R2={r2:+.4f} n_features={len(b.retained_idx)}/{X.shape[1]}")
    return out


if __name__ == "__main__":
    logdf = build_dataset()
    X, y, col_gauge, col_lag = make_lag_features(logdf)
    feature_names = [f"{g}_lag{L}" for g, L in zip(col_gauge, col_lag)]
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    print(f"{n} daily rows, {X.shape[1]} raw (gauge x lag) features "
          f"({n_train} train / {n - n_train} test, chronological split)")

    gate = generalization_gate(X, y, n_train, feature_names)

    print("\n=== baseline suite ===")
    baselines = run_baselines(X, y, n_train)
    for name, r in baselines.items():
        print(f"{name:<35} R2={r['r2']:+.4f}  n_features={r['n_features']}")

    print("\n=== hierboost, flat prior (uniform boost) ===")
    flat_res, flat_filt = run_hierboost_flat(X, y, n_train)
    print(f"R2={flat_res['r2']:+.4f} corr={flat_res['corr']:+.4f} n_features={flat_res['n_features']}/{X.shape[1]}")

    print("\n=== hierboost, causal-boosted prior (causal_affinity_1d, spatial mapping) ===")
    causal_res, causal_filt = run_hierboost_causal(X, y, n_train, col_gauge, col_lag)
    print(f"R2={causal_res['r2']:+.4f} corr={causal_res['corr']:+.4f} "
          f"n_features={causal_res['n_features']}/{X.shape[1]}")
    print(f"physically-estimated travel times (days): {causal_res['travel_time_days']}")

    print("\n=== hierboost, decorrelate='ar1' per-gauge latent (companion test) ===")
    ar1_res, ar1_filt = run_hierboost_ar1(logdf, y, n_train, col_gauge, col_lag)
    print(f"R2={ar1_res['r2']:+.4f} corr={ar1_res['corr']:+.4f} n_features={ar1_res['n_features']}/{len(UPSTREAM)}")
    print(f"fitted AR(1) persistence rho by gauge: {ar1_res['rho']}")

    print("\n=== sparsity engagement: kappa sweep (flat prior) ===")
    flat_sweep = kappa_sweep(X, y, n_train, np.ones(X.shape[1]))
    print("=== sparsity engagement: kappa sweep (causal-boosted prior) ===")
    wr_causal, _ = causal_boosted_wr(col_gauge, col_lag)
    causal_sweep = kappa_sweep(X, y, n_train, wr_causal)

    # --- mechanism check: does posterior inclusion confidence (theta_hat) track the
    # independently-known structural variable (physical distance to target), and the
    # excess-lag-beyond-minimum-travel-time, in the theoretically expected direction? ---
    print("\n=== mechanism check ===")
    dist_by_col = np.array([distance_miles(g) for g in col_gauge])
    tt_by_col = np.array([travel_time_days(g) for g in col_gauge])
    excess_lag = col_lag - tt_by_col

    b = causal_filt.best
    ridx = b.retained_idx
    theta = b.theta_hat
    rho_dist, p_dist = spearmanr(theta, dist_by_col[ridx])
    rho_excess, p_excess = spearmanr(theta, excess_lag[ridx])
    print(f"causal-boosted fit: spearman(theta_hat, distance-to-target) = {rho_dist:+.3f} (p={p_dist:.3g}) "
          f"[expect negative: nearer gauges -> higher inclusion confidence]")
    print(f"causal-boosted fit: spearman(theta_hat, excess lag beyond min travel time) = {rho_excess:+.3f} "
          f"(p={p_excess:.3g}) [expect negative: fresher-just-past-minimum lags -> higher confidence]")
    print(f"causal-boosted retained features: {[feature_names[i] for i in ridx]}")

    b_flat = flat_filt.best
    theta_flat = b_flat.theta_hat
    ridx_flat = b_flat.retained_idx
    rho_dist_flat, _ = spearmanr(theta_flat, dist_by_col[ridx_flat])
    print(f"[ablation] flat-prior fit: spearman(theta_hat, distance-to-target) = {rho_dist_flat:+.3f} "
          f"(no boosting prior to inject this structure)")

    # --- diagnostic: is the raw lag-correlation structure actually dominated by
    # routed water (multi-day lag peak per gauge, staggered by distance) or by shared
    # regional precipitation hitting every gauge ~simultaneously (lag-0/1 peak
    # regardless of distance)? Reported regardless of which it turns out to be. ---
    print("\n=== diagnostic: train-window correlation profile by lag, per gauge ===")
    corr_profile = {}
    for g in UPSTREAM:
        prof = []
        for L in LAGS:
            j = int(np.where((col_gauge == g) & (col_lag == L))[0][0])
            c = np.corrcoef(X[:n_train, j], y[:n_train])[0, 1]
            prof.append(float(c))
        corr_profile[g] = prof
        best_L = int(np.argmax(np.abs(prof)))
        print(f"{g:<12} travel_time~{travel_time_days(g):.2f}d  best raw-corr lag={LAGS[best_L]}d "
              f"(r={prof[best_L]:+.4f})  profile={[round(c, 3) for c in prof]}")

    # ----------------------------------------------------- outcome rule -----------
    print("\n" + "=" * 78)
    print("APPLYING PRE-REGISTERED OUTCOME RULE")
    print("=" * 78)

    best_baseline_name = max(
        (k for k in baselines if k != "naive (train-mean constant)"),
        key=lambda k: baselines[k]["r2"])
    best_baseline_r2 = baselines[best_baseline_name]["r2"]
    naive_r2 = baselines["naive (train-mean constant)"]["r2"]

    within_5pts = abs(causal_res["r2"] - best_baseline_r2) <= 0.05
    sparsity_frac = causal_res["n_features"] / X.shape[1]
    sparsity_ok = sparsity_frac <= 0.5
    rule2a = within_5pts and sparsity_ok

    beats_naive_by_margin = (causal_res["r2"] - naive_r2) > 0.05
    mechanism_validated = bool(rho_dist < -0.3 and p_dist < 0.10)  # negative & reasonably confident
    rule2b = beats_naive_by_margin and mechanism_validated

    fail_gate = not gate["passed"]
    fail_worse_than_naive = causal_res["r2"] < naive_r2
    max_retained_frac = max(
        max(s["n_features"] for s in flat_sweep) / X.shape[1],
        max(s["n_features"] for s in causal_sweep) / X.shape[1])
    fail_no_sparsity = (max_retained_frac > 0.9) and not mechanism_validated

    is_fail = fail_gate or fail_worse_than_naive or fail_no_sparsity
    is_work = gate["passed"] and (rule2a or rule2b) and not is_fail

    if is_fail:
        verdict = "FAIL"
    elif is_work:
        verdict = "WORK"
    else:
        verdict = "MIXED"

    print(f"gate passed: {gate['passed']}")
    print(f"best conventional baseline: {best_baseline_name} R2={best_baseline_r2:+.4f}")
    print(f"hierboost (causal-boosted) R2={causal_res['r2']:+.4f}, "
          f"within 5 R2 points of best baseline: {within_5pts}, "
          f"retained {causal_res['n_features']}/{X.shape[1]} ({sparsity_frac:.1%}) <=50%: {sparsity_ok}")
    print(f"rule 2a (accuracy parity + sparsity): {rule2a}")
    print(f"naive R2={naive_r2:+.4f}; hierboost beats naive by >0.05: {beats_naive_by_margin}; "
          f"mechanism validated (rho_dist<-0.3, p<0.10): {mechanism_validated}")
    print(f"rule 2b (beats naive + mechanism validation): {rule2b}")
    print(f"FAIL checks -- gate failed: {fail_gate}, worse than naive: {fail_worse_than_naive}, "
          f"sparsity never engages (max retained frac={max_retained_frac:.2f} > 0.9) "
          f"and no mechanism validation: {fail_no_sparsity}")
    print(f"\n>>> VERDICT: {verdict} <<<")

    # -------------------------------------------------------------- save ----------
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out = dict(
        domain="W3 river streamflow network (Delaware River mainstem)",
        predicted="WORK",
        sites={s: v for s, v in SITES.items()},
        target=TARGET, upstream=UPSTREAM,
        start_date=START_DATE, n_obs=n, n_train=n_train, n_test=n - n_train,
        lags=LAGS, n_features=X.shape[1],
        assumed_velocity_mph=ASSUMED_VELOCITY_MPH, meander_factor=MEANDER_FACTOR,
        bandwidth_days=BANDWIDTH_DAYS,
        generalization_gate=gate,
        baselines=baselines,
        hierboost_flat=flat_res,
        hierboost_causal_boosted={k: v for k, v in causal_res.items() if k != "wr"},
        hierboost_ar1=ar1_res,
        kappa_sweep_flat=flat_sweep,
        kappa_sweep_causal=causal_sweep,
        mechanism_check=dict(
            spearman_theta_vs_distance=float(rho_dist), p_value=float(p_dist),
            spearman_theta_vs_excess_lag=float(rho_excess), p_value_excess=float(p_excess),
            spearman_theta_vs_distance_flat_ablation=float(rho_dist_flat),
            retained_features_causal=[feature_names[i] for i in ridx],
        ),
        raw_corr_profile_by_lag_per_gauge=corr_profile,
        outcome_rule=dict(
            best_baseline=best_baseline_name, best_baseline_r2=float(best_baseline_r2),
            naive_r2=float(naive_r2),
            rule2a_accuracy_parity_and_sparsity=bool(rule2a),
            rule2b_beats_naive_and_mechanism=bool(rule2b),
            fail_gate=bool(fail_gate), fail_worse_than_naive=bool(fail_worse_than_naive),
            fail_no_sparsity=bool(fail_no_sparsity),
            max_retained_frac_across_sweep=float(max_retained_frac),
            verdict=verdict,
        ),
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults written to {RESULTS_PATH}")
