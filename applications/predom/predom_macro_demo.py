"""F3, predicted FAIL: cross-country GDP growth (World Bank API), deviation-detrended
against the global business cycle (state_econ's national-deviation trick, applied
globally). Two boosting-prior arms: same-region indicator, and capital-to-capital
distance."""
import json
import os

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr, pointbiserialr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_gaussian import em_filter_gaussian

DATA_DIR = "predom_macro/data"
START_YEAR, END_YEAR = 1962, 2024
TRAIN_FRAC = 0.8
TARGET_ISO3 = "JPN"  # pre-registered: 3rd-largest economy in the complete panel (see docstring)
INDICATOR = "NY.GDP.MKTP.KD.ZG"  # GDP growth (annual %)
LEVEL_INDICATOR = "NY.GDP.MKTP.CD"  # GDP, current US$ -- used ONLY for the target-selection rank


# ---------------------------------------------------------------------------
# Data: fetch (cached) + build panel
# ---------------------------------------------------------------------------

def _fetch_wb(url, cache_path):
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(data, open(cache_path, "w"))
    return data


def fetch_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    meta = _fetch_wb("https://api.worldbank.org/v2/country?format=json&per_page=400",
                      f"{DATA_DIR}/country_meta.json")[1]
    growth = _fetch_wb(f"https://api.worldbank.org/v2/country/all/indicator/{INDICATOR}"
                        f"?date=1960:2024&per_page=20000&format=json",
                        f"{DATA_DIR}/gdp_growth_full.json")[1]
    level = _fetch_wb(f"https://api.worldbank.org/v2/country/all/indicator/{LEVEL_INDICATOR}"
                       f"?date=1960:2024&per_page=20000&format=json",
                       f"{DATA_DIR}/gdp_level_full.json")[1]
    return meta, growth, level


def build_panel():
    meta, growth, level = fetch_all()
    real = {m["id"]: m for m in meta if m["region"]["value"] != "Aggregates" and m["id"]}

    g_rows = [(r["countryiso3code"], int(r["date"]), r["value"])
              for r in growth if r["countryiso3code"] in real]
    dfg = pd.DataFrame(g_rows, columns=["iso3", "year", "value"])
    pivg = dfg.pivot(index="year", columns="iso3", values="value").loc[START_YEAR:END_YEAR]
    complete = pivg.dropna(axis=1, how="any")
    countries = list(complete.columns)
    assert TARGET_ISO3 in countries

    l_rows = [(r["countryiso3code"], int(r["date"]), r["value"])
              for r in level if r["countryiso3code"] in real]
    dfl = pd.DataFrame(l_rows, columns=["iso3", "year", "value"])
    pivl = dfl.pivot(index="year", columns="iso3", values="value").loc[START_YEAR:END_YEAR]
    avg_level = pivl.reindex(columns=countries).mean(axis=0, skipna=True).sort_values(ascending=False)
    assert avg_level.index[2] == TARGET_ISO3, (
        f"target-selection rule (3rd-largest economy) no longer picks {TARGET_ISO3}; "
        f"got {list(avg_level.index[:5])} -- data may have been re-fetched with different coverage")

    lats = np.array([float(real[c]["latitude"]) if real[c]["latitude"] else np.nan for c in countries])
    lons = np.array([float(real[c]["longitude"]) if real[c]["longitude"] else np.nan for c in countries])
    regions = np.array([real[c]["region"]["value"] for c in countries])
    names = np.array([real[c]["name"] for c in countries])

    return dict(values=complete.values, years=complete.index.values, countries=np.array(countries),
                names=names, lats=lats, lons=lons, regions=regions, avg_level=avg_level)


def load():
    d = build_panel()
    values, countries = d["values"], d["countries"]
    target_k = int(np.where(countries == TARGET_ISO3)[0][0])
    global_mean = values.mean(axis=1, keepdims=True)
    dev = values - global_mean  # (T, N) deviation from cross-country mean, per year
    centroids = np.column_stack([d["lats"], d["lons"]])
    same_region = (d["regions"] == d["regions"][target_k]).astype(float)
    region_dist = 1.0 - same_region  # 0 = same WB region as target, 1 = different
    return dev, centroids, region_dist, countries, d["names"], d["regions"], target_k, d["years"]


# ---------------------------------------------------------------------------
# Generalization gate (run FIRST, before anything else -- project discipline)
# ---------------------------------------------------------------------------

def check_generalization(dev, target_k, countries, train_frac=TRAIN_FRAC):
    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_k])
    y = dev[1:, target_k]
    X = dev[:-1][:, other_idx]  # lag-1 year
    n = len(y)
    n_train = int(n * train_frac)
    train_corr = np.array([np.corrcoef(X[:n_train, j], y[:n_train])[0, 1] for j in range(X.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = np.corrcoef(X[n_train:, best_j], y[n_train:])[0, 1]
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    passed = bool(same_sign and ratio >= 0.5)
    print(f"generalization gate: best train-correlated country = {countries[other_idx[best_j]]} "
          f"(train r={train_corr[best_j]:+.3f}) -> held-out test r={test_corr:+.3f} "
          f"(ratio={ratio:.3f}, same_sign={same_sign}) -> {'PASS' if passed else 'FAIL'}")
    return dict(best_country=str(countries[other_idx[best_j]]), train_corr=float(train_corr[best_j]),
                test_corr=float(test_corr), ratio=float(ratio), same_sign=bool(same_sign),
                passed=passed, n_train=n_train, n_test=n - n_train,
                all_train_corr=train_corr.tolist())


# ---------------------------------------------------------------------------
# Baselines: naive, Lasso (L1), PCA+linear, Random Forest
# ---------------------------------------------------------------------------

def run_baselines(X_train, y_train, X_test, y_test, seed=0):
    out = {}

    naive_pred = np.zeros_like(y_test)  # deviation series is ~mean-zero by construction
    out["naive (predict zero deviation)"] = dict(r2=r2_score(y_test, naive_pred),
                                                  corr=float("nan"), n_feat=0)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
    lasso = LassoCV(cv=5, max_iter=20000, n_alphas=50, random_state=seed).fit(Xtr_s, y_train)
    pred = lasso.predict(Xte_s)
    nf = int((lasso.coef_ != 0).sum())
    out[f"Lasso/L1 (alpha={lasso.alpha_:.4g}, {nf} nonzero)"] = dict(
        r2=r2_score(y_test, pred),
        corr=float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-10 else float("nan"),
        n_feat=nf)

    # unsupervised component count: smallest K explaining >=90% variance of X_train, capped
    pca_full = PCA().fit(Xtr_s)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    K = int(np.searchsorted(cum, 0.90) + 1)
    K = min(K, X_train.shape[0] - 2, 30)
    pca = PCA(n_components=K).fit(Xtr_s)
    lr = LinearRegression().fit(pca.transform(Xtr_s), y_train)
    pred = lr.predict(pca.transform(Xte_s))
    out[f"PCA({K})+linear"] = dict(r2=r2_score(y_test, pred),
                                    corr=float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-10 else float("nan"),
                                    n_feat=K)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed).fit(X_train, y_train)
    pred = rf.predict(X_test)
    out[f"Random Forest ({X_train.shape[1]} countries)"] = dict(
        r2=r2_score(y_test, pred),
        corr=float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-10 else float("nan"),
        n_feat=X_train.shape[1])

    return out


# ---------------------------------------------------------------------------
# hierboost: flat prior + region-graph boost + geo-centroid boost
# ---------------------------------------------------------------------------

def hierboost_variant(Xd_train, y_train, Xd_test, wr, kappa, xi0=-1.0, xi1=1.0,
                       nu=1.0, lam=1.0, filter_frac=0.2, min_features=5, max_outer=60):
    filt = em_filter_gaussian(Xd_train, y_train, wr, xi0=xi0, xi1=xi1, kappa=kappa,
                               nu=nu, lam=lam, filter_frac=filter_frac,
                               min_features=min_features, max_outer=max_outer)
    b = filt.best
    pred = Xd_test[:, np.concatenate([[0], b.retained_idx + 1])] @ b.beta
    trace = [int(h.n_features) for h in filt.history]
    return filt, b, pred, trace


def run_all(dev, centroids, region_dist, target_k, kappa, countries,
            geo_bandwidth=None, region_bandwidth=2.0, verbose_header=True):
    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_k])
    y_all = dev[1:, target_k]
    X_all = dev[:-1][:, other_idx]
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]
    p = X_train.shape[1]

    other_centroids = centroids[other_idx]
    target_centroid = centroids[target_k]
    other_region_dist = region_dist[other_idx]

    results = run_baselines(X_train, y_train, X_test, y_test)

    Xd_train = np.column_stack([np.ones(n_train), X_train])
    Xd_test = np.column_stack([np.ones(len(y_test)), X_test])

    hb = {}
    flat_wr = np.ones(p)
    filt_flat, b_flat, mu_flat, trace_flat = hierboost_variant(Xd_train, y_train, Xd_test, flat_wr, kappa)
    hb["flat"] = dict(filt=filt_flat, best=b_flat, pred=mu_flat, trace=trace_flat)

    wr_region = resolve_affinity(other_region_dist, kind="graph",
                                  distance_matrix=other_region_dist[:, None], bandwidth=region_bandwidth)
    filt_region, b_region, mu_region, trace_region = hierboost_variant(Xd_train, y_train, Xd_test, wr_region, kappa)
    hb["region"] = dict(filt=filt_region, best=b_region, pred=mu_region, trace=trace_region)

    wr_geo = resolve_affinity(other_centroids, kind="gaussian", group_coords=target_centroid,
                               bandwidth=geo_bandwidth, X=X_train)
    filt_geo, b_geo, mu_geo, trace_geo = hierboost_variant(Xd_train, y_train, Xd_test, wr_geo, kappa)
    hb["geo"] = dict(filt=filt_geo, best=b_geo, pred=mu_geo, trace=trace_geo)

    for name, h in hb.items():
        r2 = r2_score(y_test, h["pred"])
        corr = float(np.corrcoef(h["pred"], y_test)[0, 1]) if np.std(h["pred"]) > 1e-10 else float("nan")
        results[f"hierboost {name} prior (kappa={kappa:g})"] = dict(
            r2=r2, corr=corr, n_feat=int(len(h["best"].retained_idx)))

    if verbose_header:
        print(f"\n{n} years ({START_YEAR + 1}-{END_YEAR}), {p} predictor countries, "
              f"target={countries[target_k]}, {n_train} train / {n - n_train} test (chronological)")
        print("-" * 96)
        print(f"{'model':<56}{'test R2':>10}{'corr':>10}{'n_feat':>8}")
        print("-" * 96)
    for name, v in results.items():
        print(f"{name:<56}{v['r2']:>10.4f}{v['corr']:>10.4f}{v['n_feat']:>8d}")

    # Mechanism-validation check (state_econ's theta_hat-vs-hop-distance analog):
    # does the boosting prior at least correctly re-rank posterior inclusion confidence
    # toward geographically/regionally closer countries, independent of whether em_filter
    # ever actually drops a feature? theta_hat is only defined over whatever subset of
    # features survived to the "best" step (b.retained_idx into the original 0..p-1
    # predictor indices) -- unlike state_econ, sparsity partially engages here, so
    # theta_hat is shorter than the full predictor count and must be aligned via
    # retained_idx before comparing against the full-length distance arrays.
    geo_dist_scalar = np.linalg.norm(other_centroids - target_centroid[None, :], axis=1)

    def region_rho(b):
        d = other_region_dist[b.retained_idx]
        finite = np.isfinite(d)
        if finite.sum() > 2 and len(np.unique(d[finite])) > 1:
            rho, _ = pointbiserialr(d[finite].astype(int), b.theta_hat[finite])
            return float(rho)
        return float("nan")

    def geo_rho(b):
        d = geo_dist_scalar[b.retained_idx]
        finite = np.isfinite(d)
        if finite.sum() > 2:
            rho, _ = spearmanr(b.theta_hat[finite], d[finite])
            return float(rho)
        return float("nan")

    rho_flat_region, rho_region_region = region_rho(b_flat), region_rho(b_region)
    rho_flat_geo, rho_geo_geo = geo_rho(b_flat), geo_rho(b_geo)

    top5_region = countries[other_idx[b_region.retained_idx[np.argsort(-b_region.theta_hat)[:5]]]]
    top5_geo = countries[other_idx[b_geo.retained_idx[np.argsort(-b_geo.theta_hat)[:5]]]]
    print(f"mechanism check (region): theta_hat vs same-region point-biserial rho -- "
          f"flat prior={rho_flat_region:+.3f}, region boost={rho_region_region:+.3f}; "
          f"region-boost top-5: {list(top5_region)}")
    print(f"mechanism check (geo): theta_hat vs capital-distance spearman rho -- "
          f"flat prior={rho_flat_geo:+.3f}, geo boost={rho_geo_geo:+.3f}; "
          f"geo-boost top-5: {list(top5_geo)}")

    mechanism = dict(rho_flat_region=float(rho_flat_region), rho_region_region=float(rho_region_region),
                      rho_flat_geo=float(rho_flat_geo), rho_geo_geo=float(rho_geo_geo),
                      region_boost_top5=list(top5_region), geo_boost_top5=list(top5_geo))
    return results, hb, mechanism, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dev, centroids, region_dist, countries, names, regions, target_k, years = load()
    print(f"target country: {countries[target_k]} ({names[target_k]}, region={regions[target_k]})")
    print(f"panel: {dev.shape[1]} countries, {dev.shape[0]} years ({years[0]}-{years[-1]})")

    print("\n=== GENERALIZATION GATE (run first, per project discipline) ===")
    gate = check_generalization(dev, target_k, countries)

    print("\n=== main comparison (kappa=100, region bandwidth=2, geo bandwidth=auto) ===")
    results_main, hb_main, mechanism_main, p = run_all(dev, centroids, region_dist, target_k, kappa=100.0,
                                                         countries=countries)

    print("\n=== sparsity-engagement sweep: kappa ===")
    kappa_grid = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    sweep = {}
    for kap in kappa_grid:
        print(f"\n-- kappa={kap:g} --")
        res, hb, mech, _ = run_all(dev, centroids, region_dist, target_k, kappa=kap,
                                    countries=countries, verbose_header=False)
        sweep[str(kap)] = dict(
            results={k: v for k, v in res.items()},
            traces={name: h["trace"] for name, h in hb.items()},
            retained_frac={name: len(h["best"].retained_idx) / p for name, h in hb.items()},
            mechanism=mech,
        )

    # Engagement verdict, per the rule's OWN wording: "never meaningfully engages" is DEFINED
    # as "retains >90% of candidate features ACROSS a reasonable hyperparameter sweep" -- i.e.
    # never-engages holds only if retention stays above 90% at every point in the sweep, for
    # every prior arm. A single sweep point (any arm) dropping to <=90% retained is enough to
    # falsify "never engages". This is the max-strictness reading: we check all three arms
    # (flat/region/geo), not just the flat prior, before concluding sparsity never engages.
    retained_fracs = {k: v["retained_frac"]["flat"] for k, v in sweep.items()}
    retained_fracs_all_arms = {k: v["retained_frac"] for k, v in sweep.items()}
    min_retained_frac = min(f for d in retained_fracs_all_arms.values() for f in d.values())
    never_engages = min_retained_frac > 0.90
    sparsity_engages = not never_engages
    print(f"\nsparsity engagement: retained fraction (flat prior) across kappa sweep = {retained_fracs}")
    print(f"retained fraction, all arms, across sweep = {retained_fracs_all_arms}")
    print(f"minimum retained fraction anywhere in sweep = {min_retained_frac:.3f}")
    print(f"-> sparsity {'ENGAGES' if sparsity_engages else 'DOES NOT ENGAGE'} "
          f"(FAIL-rule-3: 'never engages' requires >90% retained at EVERY sweep point, "
          f"every arm; out of {p} candidate features)")

    # ---------------- outcome rule ----------------
    best_conventional_r2 = max(
        results_main[k]["r2"] for k in results_main
        if k.startswith(("Lasso", "PCA", "Random Forest"))
    )
    naive_r2 = results_main["naive (predict zero deviation)"]["r2"]
    hb_r2s = {k: v["r2"] for k, v in results_main.items() if k.startswith("hierboost")}
    hb_nfeat = {k: v["n_feat"] for k, v in results_main.items() if k.startswith("hierboost")}
    best_hb_name = max(hb_r2s, key=hb_r2s.get)
    best_hb_r2 = hb_r2s[best_hb_name]
    best_hb_nfeat_frac = hb_nfeat[best_hb_name] / p

    mechanism_finding = any(abs(mechanism_main[k]) > 0.3 for k in
                             ("rho_region_region", "rho_geo_geo"))

    gate_pass = gate["passed"]
    rule2a = (abs(best_hb_r2 - best_conventional_r2) <= 0.05) and (best_hb_nfeat_frac <= 0.50)
    rule2b = (best_hb_r2 > naive_r2 + 0.02) and mechanism_finding
    fail_1 = not gate_pass
    fail_2 = best_hb_r2 < naive_r2
    fail_3 = (not sparsity_engages) and (not mechanism_finding)

    if fail_1 or fail_2 or fail_3:
        verdict = "FAIL"
    elif gate_pass and (rule2a or rule2b):
        verdict = "WORK"
    else:
        verdict = "MIXED"

    print("\n=== OUTCOME RULE (applied exactly as pre-registered) ===")
    print(f"gate_pass={gate_pass}  fail_1(gate)={fail_1}  fail_2(worse than naive)={fail_2}  "
          f"fail_3(no sparsity & no mechanism)={fail_3}")
    print(f"best_conventional_r2={best_conventional_r2:.4f}  naive_r2={naive_r2:.4f}  "
          f"best_hierboost_r2={best_hb_r2:.4f} ({best_hb_name}, retains {best_hb_nfeat_frac:.1%} of features)")
    print(f"rule2a (within-5pts & <=50% features)={rule2a}   rule2b (beats naive & mechanism finding)={rule2b}")
    print(f"mechanism_finding={mechanism_finding}")
    print(f"\n>>> VERDICT: {verdict} <<<   (F3 predicted: FAIL)")

    out = dict(
        domain="F3_macro",
        predicted="FAIL",
        verdict=verdict,
        target_country=str(countries[target_k]),
        target_country_name=str(names[target_k]),
        target_region=str(regions[target_k]),
        n_countries=int(dev.shape[1]),
        n_years=int(dev.shape[0]),
        year_range=[int(years[0]), int(years[-1])],
        n_predictors=int(p),
        generalization_gate=gate,
        main_comparison=results_main,
        mechanism_main=mechanism_main,
        sparsity_sweep=sweep,
        sparsity_engages=bool(sparsity_engages),
        min_retained_frac_across_sweep=float(min_retained_frac),
        retained_fracs_by_kappa_flat_arm=retained_fracs,
        retained_fracs_by_kappa_all_arms=retained_fracs_all_arms,
        best_conventional_r2=float(best_conventional_r2),
        naive_r2=float(naive_r2),
        best_hierboost_r2=float(best_hb_r2),
        best_hierboost_name=best_hb_name,
        best_hierboost_retained_frac=float(best_hb_nfeat_frac),
        mechanism_finding=bool(mechanism_finding),
        rule_fail_1_gate=bool(fail_1),
        rule_fail_2_worse_than_naive=bool(fail_2),
        rule_fail_3_no_sparsity_no_mechanism=bool(fail_3),
        rule_2a_close_and_sparse=bool(rule2a),
        rule_2b_beats_naive_and_mechanism=bool(rule2b),
    )
    os.makedirs("results", exist_ok=True)
    with open("results/predom_macro.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved results/predom_macro.json")
