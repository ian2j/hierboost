"""F4, predicted FAIL for the same reason MovieLens failed: NBA player scoring grouped
by position, a plausible-sounding label that individual performance isn't expected to
share much with. Same design as movielens_demo.py (dense core, mean-centering,
genre/position blocks vs. correlation-threshold blocks), season-based train/test split."""
import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from hierboost.estimator import HierBoostRegressor
from hierboost.structure import blocks_from_correlation_threshold

DATA_DIR = os.path.expanduser("~/nba_sports_data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "predom_sports.json")

TRAIN_SEASON = "2022-23"
TEST_SEASON = "2023-24"
MIN_GP = 40          # both-seasons games-played floor -> stable-roster "dense core" candidates
N_CORE = 200         # mirrors movielens_demo.py's N_CORE_MOVIES
CORR_RHO = 0.2        # mirrors movielens_demo.py's CORR_RHO
FILTER_FRAC_SWEEP = [0.1, 0.2, 0.3, 0.4]


# --------------------------------------------------------------------------- fetching
def fetch_gamelog(season):
    """One row per player per game for a full regular season -- verified live: this
    single leaguegamelog call returns clean data with no rate-limiting or blocking."""
    fp = os.path.join(DATA_DIR, f"gamelog_{season}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    os.makedirs(DATA_DIR, exist_ok=True)
    from nba_api.stats.endpoints import leaguegamelog
    print(f"Fetching {season} regular-season player game logs from stats.nba.com ...")
    r = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season",
                                     player_or_team_abbreviation="P", timeout=30)
    df = r.get_data_frames()[0]
    df.to_csv(fp, index=False)
    return df


def fetch_positions(season):
    """Listed roster POSITION per player, one commonteamroster call per team (30 total,
    paced at 0.6s) -- also verified live, no blocking encountered."""
    fp = os.path.join(DATA_DIR, f"positions_{season}.json")
    if os.path.exists(fp):
        with open(fp) as f:
            return {int(k): v for k, v in json.load(f).items()}
    os.makedirs(DATA_DIR, exist_ok=True)
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams
    print(f"Fetching {season} team rosters (position labels) from stats.nba.com ...")
    pos_map = {}
    for tm in teams.get_teams():
        r = commonteamroster.CommonTeamRoster(team_id=tm["id"], season=season, timeout=30)
        df = r.get_data_frames()[0]
        for _, row in df.iterrows():
            pos_map[int(row["PLAYER_ID"])] = row["POSITION"]
        time.sleep(0.6)
    with open(fp, "w") as f:
        json.dump(pos_map, f)
    return pos_map


# --------------------------------------------------------------------------- data build
def build_dense_core(df_train, df_test, positions, n_core=N_CORE, min_gp=MIN_GP):
    gp_tr = df_train.groupby("PLAYER_ID").size()
    gp_te = df_test.groupby("PLAYER_ID").size()
    both = sorted(set(gp_tr[gp_tr >= min_gp].index) & set(gp_te[gp_te >= min_gp].index)
                  & set(positions.keys()))
    combined_gp = (gp_tr.reindex(both) + gp_te.reindex(both)).sort_values(ascending=False)
    core_ids = combined_gp.head(n_core).index.tolist()
    target_id = int(combined_gp.index[0])

    names = df_train.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"]
    names_te = df_test.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"]
    names = names.combine_first(names_te)

    def pivot(df, ids):
        sub = df[df["PLAYER_ID"].isin(ids)]
        return sub.pivot_table(index="GAME_DATE", columns="PLAYER_ID", values="PTS")

    train_pivot = pivot(df_train, core_ids).reindex(columns=core_ids)
    test_pivot = pivot(df_test, core_ids).reindex(columns=core_ids)

    # per-player mean from TRAIN season only, applied to both -- no test-season leakage
    train_mean = train_pivot.mean(axis=0, skipna=True)
    train_c = train_pivot.sub(train_mean, axis=1).fillna(0.0)
    test_c = test_pivot.sub(train_mean, axis=1).fillna(0.0)

    y_tr = train_c[target_id].values
    y_te = test_c[target_id].values
    X_ids = [c for c in core_ids if c != target_id]
    X_tr = train_c[X_ids].values
    X_te = test_c[X_ids].values

    pos_labels = sorted(set(positions[i] for i in X_ids))
    label_to_int = {lab: i for i, lab in enumerate(pos_labels)}
    block_id = np.array([label_to_int[positions[i]] for i in X_ids])

    print(f"Target: {names.get(target_id, target_id)!r} (id {target_id}, position "
          f"{positions[target_id]!r}, {int(combined_gp.iloc[0])} combined GP)")
    print(f"{len(X_ids)} other players in the dense core, {len(pos_labels)} distinct "
          f"position labels: {label_to_int}")
    print(f"train (season {TRAIN_SEASON}): {train_pivot.shape[0]} dates; "
          f"test (season {TEST_SEASON}): {test_pivot.shape[0]} dates")

    info = dict(target_id=target_id, target_name=str(names.get(target_id, target_id)),
                target_position=positions[target_id], target_combined_gp=int(combined_gp.iloc[0]),
                n_core=len(core_ids), n_features=len(X_ids), n_position_blocks=len(pos_labels),
                position_block_sizes={lab: int(np.sum(block_id == i)) for lab, i in label_to_int.items()},
                n_train_dates=int(train_pivot.shape[0]), n_test_dates=int(test_pivot.shape[0]),
                other_player_names={str(i): str(names.get(i, i)) for i in X_ids})
    return X_tr, y_tr, X_te, y_te, block_id, X_ids, info


# --------------------------------------------------------------------------- gate
def generalization_gate(X_tr, y_tr, X_te, y_te, X_ids, names, label=""):
    """The pre-registered gate, run FIRST: single best-correlated raw feature on TRAIN,
    checked for same-sign / >=50% held-out |corr| retention on a genuinely held-out split."""
    corr_tr = np.array([np.corrcoef(X_tr[:, j], y_tr)[0, 1] for j in range(X_tr.shape[1])])
    best_j = int(np.nanargmax(np.abs(corr_tr)))
    corr_tr_best = float(corr_tr[best_j])
    corr_te_best = float(np.corrcoef(X_te[:, best_j], y_te)[0, 1])
    same_sign = bool(np.sign(corr_tr_best) == np.sign(corr_te_best))
    ratio = abs(corr_te_best) / abs(corr_tr_best) if corr_tr_best != 0 else 0.0
    passed = bool(same_sign and ratio >= 0.5)
    result = dict(split=label, best_feature_id=X_ids[best_j],
                  best_feature_name=names.get(str(X_ids[best_j]), str(X_ids[best_j])),
                  corr_train=corr_tr_best, corr_test=corr_te_best,
                  same_sign=same_sign, ratio_test_over_train=float(ratio), gate_pass=passed)
    print(f"\n[Generalization gate -- {label}]")
    print(f"  best raw feature: {result['best_feature_name']} (id {result['best_feature_id']})")
    print(f"  train |corr| = {corr_tr_best:.4f}, held-out corr = {corr_te_best:.4f} "
          f"(same sign: {same_sign}, ratio |test|/|train| = {ratio:.3f})")
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")
    return result


def within_between_block_corr(X_tr, block_id):
    """Same diagnostic movielens_demo.py's writeup used: mean |corr| within a position
    block vs. between blocks, on TRAIN only -- confirms or refutes the weak/absent
    shared-structure mechanism directly, not just via downstream accuracy."""
    C = np.corrcoef(X_tr.T)
    p = X_tr.shape[1]
    within, between = [], []
    for a in range(p):
        for b in range(a + 1, p):
            c = C[a, b]
            if np.isnan(c):
                continue
            (within if block_id[a] == block_id[b] else between).append(abs(c))
    result = dict(mean_abs_corr_within_block=float(np.mean(within)), n_within_pairs=len(within),
                  mean_abs_corr_between_block=float(np.mean(between)), n_between_pairs=len(between))
    print(f"\n[Within vs. between position-block |corr|, train season]")
    print(f"  within-block:  mean |corr| = {result['mean_abs_corr_within_block']:.4f} "
          f"(n={result['n_within_pairs']} pairs)")
    print(f"  between-block: mean |corr| = {result['mean_abs_corr_between_block']:.4f} "
          f"(n={result['n_between_pairs']} pairs)")
    return result


# --------------------------------------------------------------------------- model suite
def fit_hierboost(X_tr, y_tr, X_te, block_id, filter_frac=0.2, min_features=2):
    xi0 = np.log(0.3 / 0.7)  # prior belief ~30% of position blocks matter, same as movielens
    hb = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                             fit_method="em_filter", filter_frac=filter_frac,
                             min_features=min_features, max_outer=30)
    hb.fit(X_tr, y_tr, coords=np.arange(X_tr.shape[1]), block_id=block_id)
    pred = np.asarray(hb.predict(X_te))
    return hb, pred


def run_model_suite(X_tr, y_tr, X_te, y_te, block_id, seed=0):
    rows = []
    K = len(np.unique(block_id))

    hb_pos, pred_pos = fit_hierboost(X_tr, y_tr, X_te, block_id)
    rows.append(dict(method="hierboost (position block-latent + spike-slab)",
                      r2=float(r2_score(y_te, pred_pos)), n_features=len(hb_pos.names_),
                      n_candidate_blocks=K))

    corr_block_id = blocks_from_correlation_threshold(X_tr, rho=CORR_RHO)  # fit on TRAIN only
    hb_corr, pred_corr = fit_hierboost(X_tr, y_tr, X_te, corr_block_id)
    rows.append(dict(method="hierboost (correlation block-latent + spike-slab)",
                      r2=float(r2_score(y_te, pred_corr)), n_features=len(hb_corr.names_),
                      n_candidate_blocks=len(np.unique(corr_block_id))))

    lasso = LassoCV(cv=5, max_iter=20000, random_state=seed).fit(X_tr, y_tr)
    n_lasso_nz = int(np.sum(np.abs(lasso.coef_) > 1e-10))
    rows.append(dict(method="Lasso/L1 (raw players, CV-selected alpha)",
                      r2=float(r2_score(y_te, lasso.predict(X_te))), n_features=n_lasso_nz))

    pca = PCA(n_components=K).fit(X_tr)
    lin_pca = LinearRegression().fit(pca.transform(X_tr), y_tr)
    rows.append(dict(method=f"PCA({K})+Linear", r2=float(r2_score(y_te, lin_pca.predict(pca.transform(X_te)))),
                      n_features=K))

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=seed).fit(X_tr, y_tr)
    rows.append(dict(method="Random Forest (raw players)", r2=float(r2_score(y_te, rf.predict(X_te))),
                      n_features=X_tr.shape[1]))

    naive_pred = np.full(len(y_te), float(np.mean(y_tr)))
    rows.append(dict(method="Naive (train-mean constant)", r2=float(r2_score(y_te, naive_pred)),
                      n_features=0))

    return rows, hb_pos


def sparsity_sweep(X_tr, y_tr, X_te, block_id):
    K = len(np.unique(block_id))
    sweep = []
    for ff in FILTER_FRAC_SWEEP:
        hb, _ = fit_hierboost(X_tr, y_tr, X_te, block_id, filter_frac=ff, min_features=2)
        frac_retained = len(hb.names_) / K
        sweep.append(dict(filter_frac=ff, n_blocks_retained=len(hb.names_), n_candidate_blocks=K,
                           frac_retained=float(frac_retained)))
    print("\n[Sparsity sweep -- hierboost position blocks, em_filter]")
    for row in sweep:
        print(f"  filter_frac={row['filter_frac']}: retained {row['n_blocks_retained']}/"
              f"{row['n_candidate_blocks']} blocks ({row['frac_retained']:.0%})")
    return sweep


# --------------------------------------------------------------------------- robustness: pooled random split
def random_split_check(X_tr, y_tr, X_te, y_te, X_ids, names, seed=0):
    """Secondary robustness check for the gate using a shuffled random split pooling both
    seasons together (no leakage: rows are independent player-dates), to confirm the
    season-based gate result isn't an artifact of that particular split choice."""
    Xp = np.vstack([X_tr, X_te])
    yp = np.concatenate([y_tr, y_te])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(yp))
    n_tr = int(0.7 * len(yp))
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    return generalization_gate(Xp[tr_idx], yp[tr_idx], Xp[te_idx], yp[te_idx], X_ids, names,
                                label="robustness: random 70/30 split, pooled seasons")


# --------------------------------------------------------------------------- verdict
def apply_outcome_rule(gate, model_rows, sweep, n_raw_features):
    """Applies the preregistration's WORK/FAIL/MIXED rule EXACTLY as written (section
    "Falsifiable outcome rules") to this run's actual numbers -- no substituted judgment."""
    reasons = []
    r2_by_method = {r["method"]: r["r2"] for r in model_rows}
    naive_r2 = r2_by_method["Naive (train-mean constant)"]
    hb_r2 = r2_by_method["hierboost (position block-latent + spike-slab)"]
    hb_n_features = next(r["n_features"] for r in model_rows
                          if r["method"] == "hierboost (position block-latent + spike-slab)")
    conventional_methods = ["Lasso/L1 (raw players, CV-selected alpha)",
                             next(k for k in r2_by_method if k.startswith("PCA(")),
                             "Random Forest (raw players)"]
    best_conventional_r2 = max(r2_by_method[m] for m in conventional_methods)
    max_frac_retained = max(row["frac_retained"] for row in sweep)

    # ---- FAIL checks (ANY triggers FAIL) ----
    fail_1 = not gate["gate_pass"]
    if fail_1:
        reasons.append("FAIL-1: best single feature fails the generalization gate "
                        f"(same_sign={gate['same_sign']}, ratio={gate['ratio_test_over_train']:.3f} "
                        "-- needs same-sign AND ratio>=0.5)")
    fail_2 = hb_r2 < naive_r2
    if fail_2:
        reasons.append(f"FAIL-2: hierboost held-out R^2 ({hb_r2:.4f}) is worse than the naive "
                        f"baseline ({naive_r2:.4f})")
    sparsity_never_engages = max_frac_retained > 0.90
    # FAIL-3 also requires "no compensating mechanism-validation finding" -- this run has
    # none on offer (no independent structural variable was tested against posterior
    # inclusion), so that half of the AND is satisfied whenever sparsity fails to engage.
    fail_3 = sparsity_never_engages
    if fail_3:
        reasons.append(f"FAIL-3: em_filter retains >90% of candidate blocks across the "
                        f"filter_frac sweep (max frac retained = {max_frac_retained:.0%}), and "
                        "no compensating mechanism-validation finding was found")

    if fail_1 or fail_2 or fail_3:
        return "FAIL", reasons

    # ---- WORK checks (gate already passed to reach here; need EITHER 2a OR 2b) ----
    work_2a = (hb_r2 >= best_conventional_r2 - 0.05) and (hb_n_features <= 0.5 * n_raw_features)
    if work_2a:
        reasons.append(f"WORK-2a: hierboost R^2 ({hb_r2:.4f}) within 0.05 of best conventional "
                        f"baseline ({best_conventional_r2:.4f}) while retaining {hb_n_features}/"
                        f"{n_raw_features} raw features ({hb_n_features / n_raw_features:.0%} <= 50%)")
    # 2b (beats naive by a real margin AND an independent mechanism-validation finding) was
    # not tested in this run -- no independently-known structural variable (e.g. physical
    # distance) was checked against posterior inclusion confidence, so 2b cannot be claimed.
    if work_2a:
        return "WORK", reasons
    reasons.append("Neither WORK criterion (2a: near-best-baseline with <=50% features; 2b: "
                    "beats naive with mechanism validation) was met, and no FAIL condition "
                    "fired -- reported as MIXED per the document's standing practice.")
    return "MIXED", reasons


if __name__ == "__main__":
    df_train_raw = fetch_gamelog(TRAIN_SEASON)
    df_test_raw = fetch_gamelog(TEST_SEASON)
    positions = fetch_positions(TEST_SEASON)

    X_tr, y_tr, X_te, y_te, block_id, X_ids, info = build_dense_core(
        df_train_raw, df_test_raw, positions)

    gate = generalization_gate(X_tr, y_tr, X_te, y_te, X_ids, info["other_player_names"],
                                label=f"season-based: train {TRAIN_SEASON} -> held-out {TEST_SEASON}")
    gate_random = random_split_check(X_tr, y_tr, X_te, y_te, X_ids, info["other_player_names"])

    wb_corr = within_between_block_corr(X_tr, block_id)

    model_rows, hb_full = run_model_suite(X_tr, y_tr, X_te, y_te, block_id)
    print("\n[Held-out season R^2 -- full model suite]")
    for row in model_rows:
        print(f"  {row['method']:<55} R^2={row['r2']:+.4f}  n_features={row['n_features']}")

    sweep = sparsity_sweep(X_tr, y_tr, X_te, block_id)

    verdict, reasons = apply_outcome_rule(gate, model_rows, sweep, n_raw_features=X_tr.shape[1])
    print(f"\n[VERDICT: {verdict}]")
    for r in reasons:
        print(f"  - {r}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = dict(
        domain="F4: NBA player per-game statistics grouped by position",
        data_source="nba_api (stats.nba.com public JSON endpoints) -- worked on first live "
                     "attempt, no rate-limiting encountered, no MLB fallback needed",
        train_season=TRAIN_SEASON, test_season=TEST_SEASON,
        target_metric="PTS (points per game, per-player mean-centered using train-season means)",
        info=info,
        generalization_gate_season_split=gate,
        generalization_gate_random_split_robustness=gate_random,
        within_between_block_correlation=wb_corr,
        model_suite_held_out_season=model_rows,
        sparsity_sweep=sweep,
        verdict=verdict,
        verdict_reasons=reasons,
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")
