"""F5, predicted FAIL: Zillow city-level home prices, deliberately blocked by an
arbitrary, non-causal variable (first letter of city name) instead of true geographic
adjacency -- the point is what happens when a plausible pipeline uses a wrong grouping.
A bonus arm reruns with Census division as the (sensible) blocking variable instead."""
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score

from hierboost.estimator import HierBoostRegressor

DATA_URL = ("https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
DATA_DIR = "predom_homeprice/data"
DATA_PATH = os.path.join(DATA_DIR, "City_zhvi.csv")
RESULTS_PATH = "results/predom_homeprice.json"

TOP_N_CITIES = 150
MIN_DATE = "2012-01-01"  # since this date the top-150-by-SizeRank cities are (with
                          # one exception, dropped) fully complete -- see docstring
TRAIN_FRAC = 0.8
KAPPA_SWEEP = [10.0, 100.0, 1000.0, 10000.0]
MAIN_KAPPA = 100.0
XI0 = float(np.log(0.2 / 0.8))  # prior belief ~20% of blocks matter, same house style
                                 # as movielens_demo's xi0=log(0.3/0.7) for genre blocks

# US Census Bureau divisions (the "true geographic/regional adjacency" variable used
# ONLY in the bonus comparison arm, never in the pre-registered arbitrary-blocking arm).
DIVISION = {}
_DIVISIONS = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "DC"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}
for _div, _states in _DIVISIONS.items():
    for _s in _states:
        DIVISION[_s] = _div


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def ensure_data():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"downloading {DATA_URL} -> {DATA_PATH}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_panel():
    """Top-150-by-SizeRank US cities, restricted to the date window where they are
    (almost) all complete, dropping the rare incomplete straggler. Target city is
    picked deterministically (median SizeRank among the complete set) BEFORE any
    correlation is examined."""
    df = pd.read_csv(DATA_PATH)
    date_cols_all = [c for c in df.columns if c[:4].isdigit()]
    date_cols = [c for c in date_cols_all if c >= MIN_DATE]
    top = df.sort_values("SizeRank").head(TOP_N_CITIES).reset_index(drop=True)
    miss = top[date_cols].isna().sum(axis=1)
    complete = top[miss == 0].reset_index(drop=True)
    complete["division"] = complete["State"].map(DIVISION)
    assert complete["division"].isna().sum() == 0, "unmapped state -> census division"
    target_idx = len(complete) // 2
    print(f"{len(complete)}/{len(top)} top-{TOP_N_CITIES} cities fully complete since "
          f"{MIN_DATE}; target (median SizeRank) = "
          f"{complete.loc[target_idx, 'RegionName']}, {complete.loc[target_idx, 'State']} "
          f"(SizeRank {int(complete.loc[target_idx, 'SizeRank'])}, "
          f"division={complete.loc[target_idx, 'division']})")
    return complete, target_idx, date_cols


def build_deviation_matrix(complete, date_cols):
    """(T, C) matrix of month-over-month log-return deviations from the cross-
    sectional (all-city) mean return at each month -- removes the shared national
    housing-cycle trend, see module docstring."""
    values = complete[date_cols].values.astype(float).T  # (T, C)
    log_v = np.log(values)
    returns = np.diff(log_v, axis=0)  # (T-1, C)
    national_mean = returns.mean(axis=1, keepdims=True)
    dev = returns - national_mean
    return dev


# ---------------------------------------------------------------------------
# Generalization gate -- run FIRST, per the document's rules, independent of blocking
# ---------------------------------------------------------------------------

def check_generalization(dev, target_idx, names, train_frac=TRAIN_FRAC):
    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_idx])
    y = dev[1:, target_idx]
    X = dev[:-1][:, other_idx]  # lag-1, no leakage
    n = len(y)
    n_train = int(n * train_frac)
    train_corr = np.array([np.corrcoef(X[:n_train, j], y[:n_train])[0, 1]
                            for j in range(X.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = np.corrcoef(X[n_train:, best_j], y[n_train:])[0, 1]
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else 0.0
    passed = bool(same_sign and ratio >= 0.5)
    print(f"\n=== GENERALIZATION GATE (run first, per pre-registration rules) ===")
    print(f"best train-correlated predictor city: {names[other_idx[best_j]]!r} "
          f"(train r={train_corr[best_j]:+.4f}) -> held-out test r={test_corr:+.4f} "
          f"(|held-out|/|train| = {ratio:.3f})")
    print(f"GATE {'PASSES' if passed else 'FAILS'} "
          f"(same sign={same_sign}, ratio>=0.5: {ratio >= 0.5})")
    return dict(best_predictor=names[other_idx[best_j]], train_corr=float(train_corr[best_j]),
                test_corr=float(test_corr), ratio=float(ratio), same_sign=bool(same_sign),
                passed=passed)


# ---------------------------------------------------------------------------
# Blocking schemes
# ---------------------------------------------------------------------------

def alpha_block_id(other_names):
    """Arbitrary, non-causal blocking: first letter of the city's name. This is F5's
    load-bearing design choice -- deliberately unrelated to true geography."""
    letters = sorted(set(n[0].upper() for n in other_names))
    letter_to_id = {L: i for i, L in enumerate(letters)}
    return np.array([letter_to_id[n[0].upper()] for n in other_names]), letters


def geo_block_id(other_divisions):
    """BONUS arm only: true geographic/regional adjacency via US Census division."""
    divs = sorted(set(other_divisions))
    div_to_id = {d: i for i, d in enumerate(divs)}
    return np.array([div_to_id[d] for d in other_divisions]), divs


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_hierboost(X_train, y_train, X_test, y_test, block_id, kappa, filter_frac=0.25,
                   min_features=3):
    hb = HierBoostRegressor(decorrelate="ar1", xi0=XI0, xi1=0.0, kappa=kappa, nu=1.0, lam=1.0,
                             fit_method="em_filter", filter_frac=filter_frac,
                             min_features=min_features, max_outer=100)
    hb.fit(X_train, y_train, coords=np.arange(X_train.shape[1]), block_id=block_id)
    pred = np.asarray(hb.predict(X_test))
    r2 = r2_score(y_test, pred)
    n_blocks_total = len(hb.block_ids_)
    n_blocks_retained = len(hb.names_)
    return dict(r2=float(r2), n_blocks_total=n_blocks_total,
                n_blocks_retained=n_blocks_retained,
                retained_frac=n_blocks_retained / n_blocks_total, model=hb)


def fit_baselines(X_train, y_train, X_test, y_test):
    results = {}

    naive_pred = np.full_like(y_test, y_train.mean())
    results["naive (mean)"] = dict(r2=float(r2_score(y_test, naive_pred)),
                                    n_features=0)

    lasso = LassoCV(cv=5, max_iter=20000).fit(X_train, y_train)
    n_lasso = int(np.sum(np.abs(lasso.coef_) > 1e-10))
    results["Lasso (raw cities)"] = dict(r2=float(r2_score(y_test, lasso.predict(X_test))),
                                          n_features=n_lasso)

    K = min(10, X_train.shape[1])
    pca = PCA(n_components=K, random_state=0).fit(X_train)
    lin = LinearRegression().fit(pca.transform(X_train), y_train)
    results[f"PCA({K})+linear"] = dict(
        r2=float(r2_score(y_test, lin.predict(pca.transform(X_test)))), n_features=K)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    rf.fit(X_train, y_train)
    results["Random Forest (raw cities)"] = dict(
        r2=float(r2_score(y_test, rf.predict(X_test))), n_features=X_train.shape[1])

    return results


# ---------------------------------------------------------------------------
# Mechanism diagnostic: does blocking-scheme posterior confidence track true
# geographic relevance to the target (fraction of block members sharing the
# target's own census division)? Meaningful for the geo arm by near-tautology;
# the informative comparison is whether the ARBITRARY arm's theta_hat shows any
# such structure (it should not, if the arbitrary grouping really carries no
# regional information).
# ---------------------------------------------------------------------------

def mechanism_check(block_id, block_ids, theta_hat, other_divisions, target_division):
    is_target_div = np.array([d == target_division for d in other_divisions])
    frac_by_block = []
    for b in block_ids:
        members = is_target_div[block_id == b]
        frac_by_block.append(members.mean() if len(members) else 0.0)
    frac_by_block = np.array(frac_by_block)
    if np.std(frac_by_block) < 1e-12 or np.std(theta_hat) < 1e-12:
        rho = float("nan")
    else:
        rho, _ = spearmanr(theta_hat, frac_by_block)
    return float(rho), frac_by_block.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_data()
    complete, target_idx, date_cols = load_panel()
    names = complete["RegionName"].tolist()
    divisions = complete["division"].tolist()
    target_division = divisions[target_idx]

    dev = build_deviation_matrix(complete, date_cols)
    n_predictors = dev.shape[1] - 1

    gate = check_generalization(dev, target_idx, names)

    other_idx = np.array([k for k in range(dev.shape[1]) if k != target_idx])
    other_names = [names[k] for k in other_idx]
    other_divisions = [divisions[k] for k in other_idx]

    y_all = dev[1:, target_idx]
    X_all = dev[:-1][:, other_idx]
    n = len(y_all)
    n_train = int(n * TRAIN_FRAC)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]
    print(f"\n{n} months total ({n_train} train / {n - n_train} test, chronological), "
          f"{n_predictors} predictor cities, target={names[target_idx]}")

    # ---- baselines (blocking-independent) ----
    baselines = fit_baselines(X_train, y_train, X_test, y_test)
    print("\n=== baselines (raw features, no blocking) ===")
    for k, v in baselines.items():
        print(f"{k:<28} R2={v['r2']:+.4f}  n_features={v['n_features']}")

    # ---- ARBITRARY (alphabetical) blocking -- the pre-registered F5 arm ----
    alpha_ids, alpha_letters = alpha_block_id(other_names)
    print(f"\n=== ARBITRARY blocking (first letter of city name): "
          f"{len(alpha_letters)} blocks, sizes "
          f"{np.bincount(alpha_ids).tolist()} ===")
    alpha_sweep = {}
    for kappa in KAPPA_SWEEP:
        res = fit_hierboost(X_train, y_train, X_test, y_test, alpha_ids, kappa)
        alpha_sweep[kappa] = res
        print(f"  kappa={kappa:<8g} R2={res['r2']:+.4f}  "
              f"retained {res['n_blocks_retained']}/{res['n_blocks_total']} blocks "
              f"({100*res['retained_frac']:.0f}%)")
    main_alpha = alpha_sweep[MAIN_KAPPA]
    rho_alpha, frac_alpha = mechanism_check(
        alpha_ids, main_alpha["model"].block_ids_, main_alpha["model"].theta_hat_,
        other_divisions, target_division)
    print(f"  mechanism check (kappa={MAIN_KAPPA:g}): spearman(theta_hat, "
          f"frac-block-in-target's-own-division) = {rho_alpha:+.3f}")

    # ---- BONUS arm: geographically-sensible (Census division) blocking ----
    geo_ids, geo_divs = geo_block_id(other_divisions)
    print(f"\n=== BONUS arm: geographic (Census division) blocking: "
          f"{len(geo_divs)} blocks, sizes {np.bincount(geo_ids).tolist()} ===")
    geo_sweep = {}
    for kappa in KAPPA_SWEEP:
        res = fit_hierboost(X_train, y_train, X_test, y_test, geo_ids, kappa)
        geo_sweep[kappa] = res
        print(f"  kappa={kappa:<8g} R2={res['r2']:+.4f}  "
              f"retained {res['n_blocks_retained']}/{res['n_blocks_total']} blocks "
              f"({100*res['retained_frac']:.0f}%)")
    main_geo = geo_sweep[MAIN_KAPPA]
    rho_geo, frac_geo = mechanism_check(
        geo_ids, main_geo["model"].block_ids_, main_geo["model"].theta_hat_,
        other_divisions, target_division)
    print(f"  mechanism check (kappa={MAIN_KAPPA:g}): spearman(theta_hat, "
          f"frac-block-in-target's-own-division) = {rho_geo:+.3f}  "
          f"(target's own division = {target_division!r})")

    # ---- Apply the pre-registered outcome rule EXACTLY, using the ARBITRARY arm ----
    best_conventional_r2 = max(v["r2"] for k, v in baselines.items() if k != "naive (mean)")
    naive_r2 = baselines["naive (mean)"]["r2"]
    main_alpha_r2 = main_alpha["r2"]
    within_5pts_r2 = (best_conventional_r2 - main_alpha_r2) <= 0.05
    retained_fracs = [alpha_sweep[k]["retained_frac"] for k in KAPPA_SWEEP]
    sparsity_never_engages = all(f > 0.90 for f in retained_fracs)
    # No plausible mechanism-validation pathway exists for an arbitrary/alphabetical
    # blocking by construction (that is the whole point of F5) -- rho_alpha close to
    # 0 / not meaningfully positive confirms there is no compensating finding.
    mechanism_present = bool(np.isfinite(rho_alpha) and rho_alpha > 0.3)

    fail_1 = not gate["passed"]
    fail_2 = main_alpha_r2 < naive_r2
    fail_3 = sparsity_never_engages and not mechanism_present
    is_fail = fail_1 or fail_2 or fail_3

    work_1 = gate["passed"]
    work_2a = within_5pts_r2 and main_alpha["retained_frac"] <= 0.50
    work_2b = (main_alpha_r2 > naive_r2) and mechanism_present
    is_work = work_1 and (work_2a or work_2b)

    if is_fail and not is_work:
        verdict = "FAIL"
    elif is_work and not is_fail:
        verdict = "WORK"
    else:
        verdict = "MIXED"

    print("\n" + "=" * 78)
    print(f"OUTCOME RULE APPLIED TO ARBITRARY (ALPHABETICAL) BLOCKING ARM -- "
          f"pre-registered F5 verdict")
    print("=" * 78)
    print(f"1. Generalization gate: {'PASS' if gate['passed'] else 'FAIL'} "
          f"(train r={gate['train_corr']:+.3f}, test r={gate['test_corr']:+.3f}, "
          f"ratio={gate['ratio']:.2f})")
    print(f"2. hierboost (kappa={MAIN_KAPPA:g}) R2={main_alpha_r2:+.4f} vs. "
          f"best conventional baseline R2={best_conventional_r2:+.4f} "
          f"(within 5 pts: {within_5pts_r2}); vs. naive R2={naive_r2:+.4f} "
          f"(worse than naive: {fail_2})")
    print(f"3. Sparsity engagement: retained fractions across kappa sweep = "
          f"{[f'{f:.2f}' for f in retained_fracs]} "
          f"(never <=90% retained: {sparsity_never_engages}); "
          f"mechanism-validation finding present: {mechanism_present} "
          f"(spearman rho={rho_alpha:+.3f})")
    print(f"-> FAIL conditions met: {[fail_1, fail_2, fail_3]}")
    print(f"-> WORK conditions met: work_1={work_1}, work_2a={work_2a}, work_2b={work_2b}")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    # ---- Save results ----
    def sweep_to_json(sweep):
        return {str(k): {kk: vv for kk, vv in v.items() if kk != "model"}
                for k, v in sweep.items()}

    out = dict(
        domain="F5",
        description="Home price index (Zillow ZHVI), deliberately mis-blocked "
                     "(alphabetical vs. geographic Census-division blocking)",
        predicted="FAIL",
        target_city=names[target_idx],
        target_state=complete.loc[target_idx, "State"],
        target_division=target_division,
        n_predictor_cities=n_predictors,
        n_months_total=n,
        n_train=n_train,
        n_test=n - n_train,
        data_source=DATA_URL,
        generalization_gate=gate,
        baselines=baselines,
        arbitrary_blocking_arm=dict(
            description="Blocks = first letter of predictor city's name (arbitrary, "
                         "non-causal) -- the pre-registered F5 arm",
            n_blocks=len(alpha_letters),
            block_sizes=np.bincount(alpha_ids).tolist(),
            sweep=sweep_to_json(alpha_sweep),
            main_kappa=MAIN_KAPPA,
            mechanism_check_spearman_rho=rho_alpha,
        ),
        geographic_bonus_arm=dict(
            description="BONUS, NOT part of the pre-registered verdict. Blocks = "
                         "US Census division (true geographic/regional adjacency).",
            n_blocks=len(geo_divs),
            block_sizes=np.bincount(geo_ids).tolist(),
            sweep=sweep_to_json(geo_sweep),
            main_kappa=MAIN_KAPPA,
            mechanism_check_spearman_rho=rho_geo,
        ),
        outcome_rule=dict(
            fail_condition_1_gate_fails=fail_1,
            fail_condition_2_worse_than_naive=fail_2,
            fail_condition_3_sparsity_never_engages_and_no_mechanism=fail_3,
            work_condition_1_gate_passes=work_1,
            work_condition_2a_within_5pts_and_sparse=work_2a,
            work_condition_2b_beats_naive_and_mechanism=work_2b,
            best_conventional_baseline_r2=best_conventional_r2,
            naive_baseline_r2=naive_r2,
            hierboost_arbitrary_r2_main_kappa=main_alpha_r2,
            retained_fracs_across_sweep=retained_fracs,
        ),
        verdict=verdict,
    )
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
