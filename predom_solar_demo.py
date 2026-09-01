"""W1 of preregistration_10domain_test.md: solar power generation across a regional
grid (predicted WORK). Direct spatial-blocking analog of uk_weather/demo.py -- same
"predict target from OTHER regions' preceding-window values, no leakage" design and
the same hierboost.kernels.resolve_affinity(kind="gaussian") boosting-prior machinery
on real lon/lat centroids -- but an independent physical process (photovoltaic output
driven by cloud cover / synoptic weather systems) instead of precipitation.

Data source: Open Power System Data time_series package (data.open-power-system-data.org),
verified live before committing (2026-08-30): the project's maintenance snapshot at
time_series/2020-10-06/ is still served (HTTP 200, ~130MB), and its 60-minute
singleindex CSV genuinely contains solar_generation_actual columns for the 4 German
TSO control zones (50Hertz, Amprion, TenneT, TransnetBW) plus one column per European
country reporting to ENTSO-E -- exactly the "German/European TSO-zone solar
generation" layout the preregistration document names. No substitution was needed.
One honest caveat: OPSD itself stopped actively publishing new snapshots after 2020,
so this is real historical grid-operator data (sourced originally from ENTSO-E's
Transparency Platform) through 2020-09, not a live-updating feed -- the data is
genuine and unmodified, just not current-day.

Target zone: of the 4 German TSO zones (the "regional grid" the domain names), the
3rd-highest by total 2015-2020 generation -- same "well-connected but not the single
most extreme" pre-registered pick rule as earthquake_japan's "3rd-most-active point",
uk_weather's "3rd-most-active grid point", and state_econ's "3rd-highest border-degree
state", applied here before any model is fit.

Predictor pool: the other 3 German TSO zones (the fine-grained regional grid) plus 15
European country-level zones with clean (<2% missing over 2015-2020) reporting --
AT, BE, BG, CH, CZ, DK, EE, ES, FR, GB (UKM), GR, LT, RO, SI, SK. HR/PL/HU (>80%
missing) and NL/IT (10-13% missing, scattered gaps) were dropped for data-quality
reasons, not cherry-picked to help the result. Each zone's coordinate is that
zone's TSO-territory centroid (Germany) or national capital (countries) -- the same
level of approximation as state_econ's "state capital as physical-distance stand-in",
used only to build a distance-based boosting prior and mechanism check, not as a
scientific claim about population-weighted solar centroids.

Task: daily total generation (sum of hourly MW readings, a proxy for daily MWh) at
the target zone on day t, predicted from every OTHER zone's trailing window of daily
totals ending at day t-1 (strictly before t, matching uk_weather's no-leakage
discipline) -- window=1 (yesterday only) is the headline run; window=3 is a secondary
robustness check, exactly mirroring uk_weather/demo.py's lag-1 / trailing-3 pair.
"""
import json
import os
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

RAW_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
DATA_DIR = "predom_solar/data"
RAW_CSV = f"{DATA_DIR}/time_series_60min_singleindex.csv"
PROCESSED_NPZ = f"{DATA_DIR}/daily_solar.npz"
RESULTS_JSON = "results/predom_solar.json"

TRAIN_FRAC = 0.8
MIN_HOURS_PER_DAY = 20  # require a near-complete day (out of 24) to keep it

# German TSO control-zone approximate territory centroids (lon, lat) -- 50Hertz (NE:
# Berlin/Brandenburg/Saxony/Sachsen-Anhalt/Thuringia/MV/Hamburg), Amprion (west: NRW +
# parts of RLP/Saarland), TenneT (north-south corridor: Schleswig-Holstein/Lower
# Saxony/Hesse/Bavaria), TransnetBW (Baden-Wurttemberg, southwest).
DE_TSO_CENTROIDS = {
    "DE_50hertz": (13.0, 52.0),
    "DE_amprion": (7.3, 50.9),
    "DE_tennet": (10.0, 51.0),
    "DE_transnetbw": (9.2, 48.5),
}
# European country capitals as a physical-distance stand-in (state_econ precedent).
COUNTRY_CENTROIDS = {
    "AT": (16.37, 48.21), "BE": (4.35, 50.85), "BG": (23.32, 42.70), "CH": (7.45, 46.95),
    "CZ": (14.42, 50.09), "DK": (12.57, 55.68), "EE": (24.75, 59.44), "ES": (-3.70, 40.42),
    "FR": (2.35, 48.86), "GB_UKM": (-0.13, 51.51), "GR": (23.73, 37.98), "LT": (25.28, 54.69),
    "RO": (26.10, 44.44), "SI": (14.51, 46.06), "SK": (17.11, 48.15),
}
ALL_CENTROIDS = {**DE_TSO_CENTROIDS, **COUNTRY_CENTROIDS}
ZONE_COLS = {z: f"{z}_solar_generation_actual" for z in ALL_CENTROIDS}


def fetch_raw():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 1e8:
        print(f"[fetch] {RAW_CSV} already present ({os.path.getsize(RAW_CSV)/1e6:.1f} MB), skipping download")
        return
    print(f"[fetch] downloading {RAW_URL} -> {RAW_CSV} (~130MB, may take a few minutes)")
    req = urllib.request.Request(RAW_URL, headers={"User-Agent": "hierboost-research/0.1"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(RAW_CSV, "wb") as f:
        f.write(resp.read())
    print(f"[fetch] saved {os.path.getsize(RAW_CSV)/1e6:.1f} MB")


def build_daily_matrix():
    """Hourly MW -> daily total per zone, dropping any day that isn't near-complete
    (>=MIN_HOURS_PER_DAY of 24) for any kept zone, then restricting to the common
    fully-covered date range across all zones -- same discipline as uk_weather's
    "drop partial/missing, use the common well-covered window" approach."""
    if os.path.exists(PROCESSED_NPZ):
        print(f"[build] {PROCESSED_NPZ} already present, skipping rebuild")
        return
    usecols = ["utc_timestamp"] + list(ZONE_COLS.values())
    df = pd.read_csv(RAW_CSV, usecols=usecols, parse_dates=["utc_timestamp"])
    df["day"] = df["utc_timestamp"].dt.date
    grp = df.groupby("day")
    daily_sum = grp[list(ZONE_COLS.values())].sum(min_count=1)
    daily_count = grp[list(ZONE_COLS.values())].count()
    # a day only counts for a zone if it has a near-complete set of hourly readings
    daily_sum = daily_sum.where(daily_count >= MIN_HOURS_PER_DAY)
    # keep only days with ALL zones present -- a clean rectangular panel, no imputation
    complete = daily_sum.dropna(how="any")
    complete = complete.sort_index()
    zones = list(ZONE_COLS.keys())
    values = complete[[ZONE_COLS[z] for z in zones]].to_numpy()
    centroids = np.array([ALL_CENTROIDS[z] for z in zones])
    days = np.array([str(d) for d in complete.index])
    print(f"[build] {values.shape[0]} complete days, {values.shape[1]} zones, "
          f"date range {days[0]} .. {days[-1]}")

    # pre-registered target pick: of the 4 German TSO zones (the domain's "regional
    # grid"), the 3rd-highest by total generation over the sample -- decided before
    # any model is fit, same rule family as earthquake_japan/uk_weather/state_econ.
    de_idx = [zones.index(z) for z in DE_TSO_CENTROIDS]
    de_totals = values[:, de_idx].sum(axis=0)
    order = np.argsort(de_totals)[::-1]
    target_k = de_idx[order[2]]
    print(f"[build] DE TSO zone totals (MW-hours, sample sum): "
          f"{dict(zip(DE_TSO_CENTROIDS, de_totals.round(0)))}")
    print(f"[build] target zone = {zones[target_k]} (3rd-highest of the 4 DE TSO zones)")

    np.savez(PROCESSED_NPZ, values=values, centroids=centroids,
             zones=np.array(zones), days=days, target_k=target_k)
    print(f"[build] saved -> {PROCESSED_NPZ}")


def make_lagged(values, target_k, window):
    """X[t] = trailing `window`-day SUM of every OTHER zone's daily total ending at
    day t-1 (strictly before t); y[t] = target zone's total at day t. Same
    construction as uk_weather/demo.py's make_lagged window logic."""
    n_days, K = values.shape
    other_idx = np.array([k for k in range(K) if k != target_k])
    y_all = values[window:, target_k]
    X_all = np.zeros((n_days - window, len(other_idx)))
    for w in range(1, window + 1):
        X_all += values[window - w: n_days - w][:, other_idx]
    return X_all, y_all, other_idx


def check_generalization(values, target_k, zones, label):
    """PRE-REGISTERED GATE, run first: does the single best train-correlated raw
    predictor's relationship survive a held-out, chronological split? Same-sign and
    held-out |corr| >= 50% of train |corr| required to pass."""
    X_all, y_all, other_idx = make_lagged(values, target_k, window=1)
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    train_corr = np.array([np.corrcoef(X_all[:n_train, j], y_all[:n_train])[0, 1]
                            for j in range(X_all.shape[1])])
    best_j = np.nanargmax(np.abs(train_corr))
    test_corr = np.corrcoef(X_all[n_train:, best_j], y_all[n_train:])[0, 1]
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    passes = bool(same_sign and ratio >= 0.5)
    print(f"[{label}] generalization gate: best train-correlated zone = "
          f"{zones[other_idx[best_j]]} (train r={train_corr[best_j]:+.4f}) -> "
          f"held-out test r={test_corr:+.4f}  (ratio={ratio:.3f}, same_sign={same_sign}) "
          f"-> {'PASS' if passes else 'FAIL'}")
    return dict(best_predictor_zone=str(zones[other_idx[best_j]]),
                train_corr=float(train_corr[best_j]), test_corr=float(test_corr),
                ratio=float(ratio), same_sign=bool(same_sign), passes=passes)


def run_models(values, centroids, target_k, zones, window, kappa=100.0, bandwidth=None,
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
    results[f"Lasso/L1 (raw, {p} zones)"] = (lasso.predict(Xte_s), int((lasso.coef_ != 0).sum()))

    K_pca = min(8, p - 1)
    pca = PCA(n_components=K_pca, random_state=0).fit(Xtr_s)
    lr = sm.OLS(y_train, sm.add_constant(pca.transform(Xtr_s))).fit()
    pred_pca = lr.predict(sm.add_constant(pca.transform(Xte_s), has_constant="add"))
    results[f"PCA({K_pca})+linear"] = (pred_pca, K_pca)

    rf = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=2).fit(X_train, y_train)
    results[f"Random Forest (raw, {p} zones)"] = (rf.predict(X_test), p)

    Xd_train = np.column_stack([np.ones(n_train), Xtr_s])
    Xd_test = np.column_stack([np.ones(len(y_test)), Xte_s])

    # hierboost's spike-slab priors (nu/lam on sigma_y2, slab variance ~1 by default)
    # are calibrated for roughly unit-scale responses -- fed raw MW-scale y (tens of
    # thousands), the EM collapses beta to ~0 and badly misfits (verified: R2 ~ -2.5
    # despite corr ~0.9). Standardizing y for the fit and rescaling predictions back
    # is the fix, exactly analogous to why X is already standardized for Lasso/PCA.
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
              f"chronological), p={p} predictor zones, target={zones[target_k]}")
        print(f"[window={window}d] target train mean={y_train.mean():.1f} MW, test mean={y_test.mean():.1f} MW")
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
    top5 = zones[other_idx[np.argsort(-bs.theta_hat)[:5]]]
    print(f"[window={window}d] mechanism check: theta_hat vs distance-to-target spearman rho -- "
          f"flat prior={rho_flat:+.3f}, spatial boost={rho_sp:+.3f}; "
          f"spatial-boost top-5 by theta_hat: {list(top5)}")

    return dict(window=window, n=n, n_train=n_train, n_test=n - n_train, p=p,
                results=row_results,
                mechanism_check=dict(rho_theta_vs_distance_flat=float(rho_flat),
                                      rho_theta_vs_distance_spatial=float(rho_sp),
                                      spatial_top5_by_theta=[str(z) for z in top5]),
                hierboost_flat_n_retained=len(bf.retained_idx),
                hierboost_spatial_n_retained=len(bs.retained_idx))


def sparsity_sweep(values, centroids, target_k, zones, window=1):
    """FAIL-rule check #3: does em_filter's sparsity ever meaningfully engage across a
    reasonable kappa sweep, or does it retain >90% of candidate zones regardless?"""
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

    print(f"\n[sweep] em_filter feature retention vs kappa (p={p} candidate zones):")
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
        and mech["rho_theta_vs_distance_spatial"] < -0.1  # closer zones (smaller distance) -> higher theta
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


def main():
    fetch_raw()
    build_daily_matrix()
    d = np.load(PROCESSED_NPZ, allow_pickle=True)
    values, centroids, zones, target_k = d["values"], d["centroids"], d["zones"], int(d["target_k"])

    print("\n" + "=" * 88)
    print("STEP 1: GENERALIZATION GATE (run first, per project discipline)")
    print("=" * 88)
    gate = check_generalization(values, target_k, zones, label="W1 solar (lag-1)")

    print("\n" + "=" * 88)
    print("STEP 2: hierboost vs. baseline suite, held-out chronological split")
    print("=" * 88)
    main_run = run_models(values, centroids, target_k, zones, window=1)
    print()
    secondary_run = run_models(values, centroids, target_k, zones, window=3)

    print("\n" + "=" * 88)
    print("STEP 3: sparsity-engagement sweep (FAIL-rule check #3)")
    print("=" * 88)
    sweep, sweep_max_frac = sparsity_sweep(values, centroids, target_k, zones, window=1)

    print("\n" + "=" * 88)
    print("STEP 4: apply the preregistered WORK/FAIL/MIXED rule to the numbers above")
    print("=" * 88)
    verdict, reasons, verdict_numbers = apply_outcome_rule(gate, main_run, sweep_max_frac)
    for r in reasons:
        print(" -", r)
    print(f"\n>>> W1 (solar power generation, regional grid) VERDICT: {verdict}  "
          f"(preregistered prediction was WORK)")

    out = dict(
        domain="W1: solar power generation, regional grid",
        data_source=dict(
            url=RAW_URL,
            note=("Open Power System Data time_series package, verified live 2026-08-30. "
                  "OPSD itself stopped publishing new snapshots after this 2020-10-06 one, "
                  "so this is genuine historical ENTSO-E-sourced grid data through 2020-09, "
                  "not a live feed -- no substitution was needed, flagged here for honesty."),
        ),
        target_zone=str(zones[target_k]),
        predictor_zones=[str(z) for z in zones if z != zones[target_k]],
        n_days_total=int(values.shape[0]),
        generalization_gate=gate,
        main_run_window1=main_run,
        secondary_run_window3=secondary_run,
        sparsity_sweep=sweep,
        sparsity_sweep_max_retained_frac=sweep_max_frac,
        verdict=verdict,
        verdict_reasoning=reasons,
        verdict_numbers=verdict_numbers,
        preregistered_prediction="WORK",
    )
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[main] results saved -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
