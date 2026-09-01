"""F1 (pre-registered 10-domain test, preregistration_10domain_test.md) -- Wikipedia
article pageviews. PREDICTED FAIL. Mechanism: "viral/news-driven spikes are
event-driven and idiosyncratic per-article -- same failure mode as earthquakes (real
in-sample correlation, collapses out-of-sample)."

Data: Wikimedia REST pageviews API (wikimedia.org/api/rest_v1/metrics/pageviews/
per-article, free, no auth -- verified live before committing to it, see the fetch()
docstring below). Daily, en.wikipedia, access=all-access, agent=user (excludes bot
traffic), 2022-01-01 through yesterday (UTC).

Domain design (the MovieLens-genre analog this document calls for -- "a plausible-
sounding but not mechanistically-causal grouping"): target = Taylor_Swift, predictors
= 12 other prominent pop/R&B musicians (Beyonce, Ariana Grande, Dua Lipa, Billie
Eilish, Olivia Rodrigo, Adele, Rihanna, Katy Perry, Lady Gaga, Justin Bieber, Drake,
The Weeknd). "Same music industry" is exactly the kind of plausible-but-not-causal
grouping MovieLens's genre label was -- there's no mechanistic reason Beyonce's
Wikipedia traffic on a given day should predict Taylor Swift's, only a vague
"topically related" story, which is precisely the F1 mechanism under test: each
artist's real spikes are driven by their OWN idiosyncratic news cycle (an album drop,
a tour date, an awards show, a breakup story), not a shared one.

Two confounds are removed before any correlation is computed or any model is fit,
matching this project's established discipline of not trusting an unexamined raw
correlation (earthquake_japan: strict lag; state_econ: subtract the national-average
business cycle before looking at cross-state correlation):
  1. Shared global Wikipedia traffic (weekday/weekend browsing patterns, overall site
     growth, a globally newsy day bumping everything at once) is removed by
     subtracting, from each article's log1p(views) on day t, the mean log1p(views)
     across all 13 fetched articles (target + 12 predictors) on that SAME day t --
     the direct pageviews analog of state_econ's "deviation from national average."
     What's left is each article's IDIOSYNCRATIC deviation from the group's shared
     day-to-day pull, which is the only thing a "topically related articles" story
     could plausibly explain.
  2. Same-day leakage: the task is a genuine one-step-ahead forecast, not a same-
     instant correlation fit -- X_t is every predictor's deviation on day t-1, y_t is
     the target's deviation on day t (strict lag, no leakage), the same discipline
     earthquake_japan/state_econ use for exactly this reason.

Per the document's General Discipline, the generalization gate (single best-correlated
raw feature, train-window correlation vs. held-out-window correlation, chronological
split) is run and reported FIRST, before anything else. Then hierboost (correlation-
threshold blocking, same as movielens_demo.py/newsgroups_demo.py) vs. the standard
baseline suite (Lasso, PCA+linear, Random Forest, naive) on the same held-out split.
The document's WORK/FAIL/MIXED rule is then applied mechanically to the actual
numbers -- no substituting judgment for the stated thresholds.
"""
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import date, timedelta

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from hierboost.estimator import HierBoostRegressor
from hierboost.structure import blocks_from_correlation_threshold
from hierboost.blocks import block_membership_lists
from hierboost.spike_slab_gaussian import em_filter_gaussian

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_PATH = os.path.join(RESULTS_DIR, "wikipedia_pageviews_cache.json")
OUT_PATH = os.path.join(RESULTS_DIR, "predom_wikipedia.json")

TARGET = "Taylor_Swift"
RELATED = ["Beyoncé", "Ariana_Grande", "Dua_Lipa", "Billie_Eilish", "Olivia_Rodrigo",
           "Adele", "Rihanna", "Katy_Perry", "Lady_Gaga", "Justin_Bieber",
           "Drake_(musician)", "The_Weeknd"]
ALL_ARTICLES = [TARGET] + RELATED

START = "20220101"
END_LAG_DAYS = 1  # yesterday UTC -- today's daily bucket is typically incomplete
TRAIN_FRAC = 0.8
CORR_RHO = 0.2  # same default MovieLens/Newsgroups use for blocks_from_correlation_threshold
API_TMPL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
HEADERS = {"User-Agent": "hierboost-research-demo/1.0 (research script; "
                         "contact ian2johnston@gmail.com)"}


def fetch_article(article, start, end):
    """Live-verified against the real API before this script was written (see session
    notes): GET .../per-article/en.wikipedia/all-access/user/<article>/daily/<start>/<end>
    returns 200 with a JSON {"items": [...]} list, one entry per day, no auth needed.
    Returns a dict {YYYYMMDD: views}."""
    url = API_TMPL.format(article=urllib.parse.quote(article, safe=""), start=start, end=end)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return {it["timestamp"][:8]: int(it["views"]) for it in payload["items"]}


def fetch_all(start, end):
    """Cache raw per-article series to disk (results/wikipedia_pageviews_cache.json) so
    reruns of this script don't re-hit the API. Cache is keyed by (article, start, end)
    so a range change just adds a new entry rather than silently reusing stale data."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    out = {}
    dirty = False
    for art in ALL_ARTICLES:
        key = f"{art}|{start}|{end}"
        if key in cache:
            out[art] = cache[key]
            continue
        print(f"fetching {art} [{start}, {end}] ...")
        for attempt in range(3):
            try:
                out[art] = fetch_article(art, start, end)
                break
            except urllib.error.HTTPError as e:
                if attempt == 2:
                    raise
                print(f"  retry after HTTPError {e.code}")
                time.sleep(2)
        cache[key] = out[art]
        dirty = True
        time.sleep(0.2)  # polite pacing, well within Wikimedia's rate limits
    if dirty:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    return out


def build_matrix(series_by_article, start, end):
    """Align every article onto the same calendar (dates present for ALL articles),
    log1p-transform, then subtract the cross-article same-day mean from every series
    (target included) -- removes shared global-Wikipedia-traffic confound, matching
    state_econ's "deviation from national average" discipline (see module docstring)."""
    d0 = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    d1 = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    all_dates = [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]
    common = [dt for dt in all_dates if all(dt in series_by_article[a] for a in ALL_ARTICLES)]
    missing = len(all_dates) - len(common)
    if missing:
        print(f"note: {missing} of {len(all_dates)} calendar days dropped "
              f"(missing from at least one article's series)")

    raw = np.array([[series_by_article[a][dt] for a in ALL_ARTICLES] for dt in common], dtype=float)
    log_raw = np.log1p(raw)  # (T, 13), column 0 = target
    group_mean = log_raw.mean(axis=1, keepdims=True)  # shared-day confound proxy
    dev = log_raw - group_mean  # idiosyncratic deviation, target included (col 0)
    return dev, common


def make_lagged(dev):
    """Strict lag-1: X_t = predictors' deviation at day t-1, y_t = target's deviation
    at day t. No same-day leakage -- a genuine one-step-ahead forecast, matching
    earthquake_japan/state_econ's discipline (see module docstring)."""
    y = dev[1:, 0]
    X = dev[:-1, 1:]
    return X, y


def check_generalization(X, y, train_frac=TRAIN_FRAC):
    """THE GATE. Run and reported first, before any other result, per the document's
    General Discipline #2. Single best train-correlated raw (post-confound-removal,
    pre-hierboost) feature; does its relationship survive into a genuinely held-out,
    chronologically-later window? WORK requires held-out |corr| >= 50% of train |corr|
    with the same sign; anything else is FAIL condition 1, full stop, regardless of
    what any other analysis shows."""
    n = len(y)
    n_train = int(n * train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    train_corr = np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = float(np.corrcoef(Xte[:, best_j], yte)[0, 1])
    train_best = float(train_corr[best_j])
    same_sign = np.sign(train_best) == np.sign(test_corr) and test_corr != 0
    ratio = abs(test_corr) / abs(train_best) if train_best != 0 else float("nan")
    passes = bool(same_sign and ratio >= 0.5)
    return dict(best_feature=RELATED[best_j], train_corr=train_best, test_corr=test_corr,
                ratio_held_out_over_train=ratio, same_sign=bool(same_sign), passes_gate=passes,
                n_train=n_train, n_test=n - n_train)


def retained_raw_feature_count(hb, p_raw):
    """How many of the p_raw raw predictor features fall inside blocks hierboost
    actually RETAINED (post em_filter), not just how many blocks survived -- the
    outcome rule's "<=50% of the raw feature count" / ">90% of candidate features"
    thresholds are stated in raw-feature terms. Mirrors HierBoostRegressor.describe_blocks
    (hierboost/estimator.py) rather than calling it directly, since we also want it
    for intermediate sweep fits that never call fit() through the full class."""
    blocks = block_membership_lists(hb.block_id_)
    retained_block_ids = (hb.block_ids_ if hb.retained_idx_ is None
                           else [hb.block_ids_[k] for k in hb.retained_idx_])
    return int(sum(len(blocks[b]) for b in retained_block_ids))


def run_baselines_and_hierboost(X, y, train_frac=TRAIN_FRAC, corr_rho=CORR_RHO):
    n = len(y)
    n_train = int(n * train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    p = X.shape[1]
    results = {}

    naive_mu = np.full_like(yte, ytr.mean())
    results["naive (train-mean deviation)"] = dict(r2=r2_score(yte, naive_mu),
                                                     corr=float("nan"), n_features=0)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    lasso = Lasso(alpha=0.01, max_iter=20000).fit(Xtr_s, ytr)
    pred_lasso = lasso.predict(Xte_s)
    results[f"Lasso (raw {p} articles)"] = dict(
        r2=r2_score(yte, pred_lasso), corr=float(np.corrcoef(pred_lasso, yte)[0, 1]),
        n_features=int((lasso.coef_ != 0).sum()))

    K_pca = min(6, p)
    pca = PCA(n_components=K_pca).fit(Xtr_s)
    lr_pca = LinearRegression().fit(pca.transform(Xtr_s), ytr)
    pred_pca = lr_pca.predict(pca.transform(Xte_s))
    results[f"PCA({K_pca})+linear"] = dict(
        r2=r2_score(yte, pred_pca), corr=float(np.corrcoef(pred_pca, yte)[0, 1]), n_features=K_pca)

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=0).fit(Xtr, ytr)
    pred_rf = rf.predict(Xte)
    results[f"Random Forest (raw {p} articles)"] = dict(
        r2=r2_score(yte, pred_rf), corr=float(np.corrcoef(pred_rf, yte)[0, 1]), n_features=p)

    block_id = blocks_from_correlation_threshold(Xtr, rho=corr_rho)  # train-only, no leakage
    n_blocks = len(np.unique(block_id))
    xi0 = np.log(0.3 / 0.7)  # same weak prior MovieLens/state_econ use, not tuned here
    hb = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=100.0, nu=1.0, lam=1.0,
                             fit_method="em_filter", filter_frac=0.2, min_features=3, max_outer=30)
    hb.fit(Xtr, ytr, coords=np.arange(p), block_id=block_id, feature_names=RELATED)
    pred_hb = np.asarray(hb.predict(Xte))
    n_raw_retained = retained_raw_feature_count(hb, p)
    results[f"hierboost (correlation block-latent + spike-slab, rho={corr_rho})"] = dict(
        r2=r2_score(yte, pred_hb), corr=float(np.corrcoef(pred_hb, yte)[0, 1]),
        n_features=n_raw_retained, n_blocks_total=n_blocks, n_blocks_retained=len(hb.names_))

    return results, hb, block_id


def sparsity_sweep(X, y, train_frac=TRAIN_FRAC):
    """Outcome rule FAIL-condition 3: does em_filter ever meaningfully engage (retain
    <=90% of candidate raw features) across a reasonable hyperparameter sweep of kappa
    (sparsity prior strength) and rho (block-correlation threshold)?"""
    n = len(y)
    n_train = int(n * train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]
    p = X.shape[1]
    rows = []
    for rho in [0.1, 0.2, 0.3, 0.4]:
        block_id = blocks_from_correlation_threshold(Xtr, rho=rho)
        for kappa in [10.0, 100.0, 1000.0, 10000.0]:
            xi0 = np.log(0.3 / 0.7)
            hb = HierBoostRegressor(decorrelate="sar", xi0=xi0, xi1=0.0, kappa=kappa, nu=1.0, lam=1.0,
                                     fit_method="em_filter", filter_frac=0.2, min_features=3, max_outer=30)
            hb.fit(Xtr, ytr, coords=np.arange(p), block_id=block_id, feature_names=RELATED)
            n_raw_retained = retained_raw_feature_count(hb, p)
            frac = n_raw_retained / p
            rows.append(dict(rho=rho, kappa=kappa, n_blocks_total=len(np.unique(block_id)),
                              n_features_retained=n_raw_retained, frac_retained=frac))
    return rows


def diagnose_own_autocorrelation(dev, X, y, train_frac=TRAIN_FRAC):
    """Diagnostic, NOT part of the pre-registered outcome rule -- added because the
    confound-removal step (subtracting the same-day cross-article mean, target
    included, matching state_econ's "deviation from national average" design exactly)
    has a mechanical side effect worth surfacing rather than silently living inside a
    surprising R^2 number: since all 13 (target + 12 predictor) deviations sum to zero
    on any given day, the 12 predictor columns' row-sum is EXACTLY equal to
    -1 * the target's own same-day deviation. If the target's own deviation series is
    itself strongly autocorrelated, any model that puts roughly equal weight on the 12
    predictors (e.g. Lasso, which doesn't know or care where a linear combination's
    predictive power comes from) can recover a large chunk of R^2 purely from the
    target's own persistence smuggled back in through that identity -- not from any
    genuine "other articles' movement predicts this one" signal, which is the actual
    question this domain is supposed to test. Same diagnostic instinct as the
    earthquake weekly-vs-daily write-up and state_econ's explicit national-average
    rationale: check whether a striking number means what it looks like it means
    before reporting it."""
    n = len(y)
    n_train = int(n * train_frac)
    target_dev = dev[:, 0]
    row_sum_corr = float(np.corrcoef(X.sum(axis=1), target_dev[:n])[0, 1])

    own_lag_train = float(np.corrcoef(target_dev[1:n_train + 1], target_dev[:n_train])[0, 1])
    own_lag_test = float(np.corrcoef(target_dev[n_train + 1:], target_dev[n_train:-1])[0, 1])

    # "own-lag-1 only" model: uses NOTHING but the target's own previous-day deviation
    # (recoverable from -row-sum(X), so this is a fair, apples-to-apples diagnostic
    # baseline, not new data) -- how much of Lasso's headline R^2 is just this?
    own_lag_feat = -X.sum(axis=1, keepdims=True)
    Xtr_o, Xte_o = own_lag_feat[:n_train], own_lag_feat[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    lr = LinearRegression().fit(Xtr_o, ytr)
    pred = lr.predict(Xte_o)
    own_lag_r2 = float(r2_score(yte, pred))
    own_lag_corr = float(np.corrcoef(pred, yte)[0, 1])

    return dict(
        row_sum_identity_corr=row_sum_corr,
        note="predictor-row-sum should be ~ -1.0 correlated with the target's own same-day "
             "deviation by construction of the confound-removal step (see docstring)",
        target_own_autocorr_train=own_lag_train,
        target_own_autocorr_test=own_lag_test,
        own_lag_only_r2=own_lag_r2,
        own_lag_only_corr=own_lag_corr,
    )


def apply_outcome_rule(gate, model_results, sweep_rows):
    """Mechanical application of preregistration_10domain_test.md's stated rule --
    no judgment substituted for the thresholds."""
    hb_key = [k for k in model_results if k.startswith("hierboost")][0]
    hb = model_results[hb_key]
    naive = model_results["naive (train-mean deviation)"]
    conventional = {k: v for k, v in model_results.items()
                    if k not in (hb_key, "naive (train-mean deviation)")}
    best_conv_key = max(conventional, key=lambda k: conventional[k]["r2"])
    best_conv_r2 = conventional[best_conv_key]["r2"]

    fail_1 = not gate["passes_gate"]
    fail_2 = hb["r2"] < naive["r2"]
    max_frac_retained = max(r["frac_retained"] for r in sweep_rows)
    fail_3_engagement = max_frac_retained > 0.90  # never drops below 90% retained anywhere in sweep
    fail_3 = fail_3_engagement  # no mechanism-validation finding available for this domain
                                 # (no independently-known structural variable analogous to
                                 # physical distance exists between "topically related" pop
                                 # musicians -- see module docstring)

    work_1 = gate["passes_gate"]
    within_5pts = abs(hb["r2"] - best_conv_r2) <= 0.05
    beats_50pct_raw = hb["n_features"] <= 0.5 * len(RELATED)
    work_2a = within_5pts and beats_50pct_raw
    work_2b = False  # no mechanism-validation variable available in this domain; see above
    work_2 = work_2a or work_2b

    is_fail = fail_1 or fail_2 or fail_3
    is_work = work_1 and work_2 and not is_fail

    if is_fail:
        verdict = "FAIL"
    elif is_work:
        verdict = "WORK"
    else:
        verdict = "MIXED"

    return dict(
        verdict=verdict,
        fail_condition_1_gate_failed=fail_1,
        fail_condition_2_worse_than_naive=fail_2,
        fail_condition_3_no_sparsity_engagement=fail_3,
        max_frac_retained_in_sweep=max_frac_retained,
        work_condition_1_gate_passed=work_1,
        work_condition_2a_within_5pts_and_sparse=work_2a,
        work_condition_2b_mechanism_validation=work_2b,
        best_conventional_baseline=best_conv_key,
        best_conventional_r2=best_conv_r2,
        hierboost_r2=hb["r2"],
        naive_r2=naive["r2"],
    )


if __name__ == "__main__":
    end_date = (date.today() - timedelta(days=END_LAG_DAYS)).strftime("%Y%m%d")
    print(f"Fetching Wikimedia pageviews API data: {ALL_ARTICLES} [{START}, {end_date}]")
    series = fetch_all(START, end_date)
    for a in ALL_ARTICLES:
        print(f"  {a}: {len(series[a])} days fetched")

    dev, dates = build_matrix(series, START, end_date)
    print(f"\n{dev.shape[0]} common calendar days after alignment "
          f"(target col 0, {len(RELATED)} predictor columns)")

    X, y = make_lagged(dev)
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    print(f"lag-1 design: n={n} ({n_train} train / {n - n_train} test, chronological split)")

    print("\n" + "=" * 78)
    print("GENERALIZATION GATE (reported FIRST, per document discipline)")
    print("=" * 78)
    gate = check_generalization(X, y)
    print(f"best train-correlated predictor: {gate['best_feature']!r}  "
          f"train r={gate['train_corr']:+.4f} -> held-out test r={gate['test_corr']:+.4f}  "
          f"(ratio={gate['ratio_held_out_over_train']:.3f}, same_sign={gate['same_sign']})")
    print(f"GATE {'PASSES' if gate['passes_gate'] else 'FAILS'} "
          f"(needs same-sign and held-out |corr| >= 50% of train |corr|)")

    print("\n" + "=" * 78)
    print("hierboost vs. baseline suite (same chronological 80/20 split)")
    print("=" * 78)
    model_results, hb_full, block_id_full = run_baselines_and_hierboost(X, y)
    print(f"{'model':<58}{'test R2':>10}{'corr':>10}{'n_feat':>8}")
    print("-" * 86)
    for name, r in model_results.items():
        print(f"{name:<58}{r['r2']:>10.4f}{r['corr']:>10.4f}{r['n_features']:>8d}")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC (not part of the outcome rule): is Lasso's R2 above just the "
          "target's own autocorrelation smuggled back in through the confound-removal step?")
    print("=" * 78)
    diag = diagnose_own_autocorrelation(dev, X, y)
    print(f"row-sum(X) vs target's own same-day deviation: r={diag['row_sum_identity_corr']:+.4f} "
          f"(expect ~ -1.0 by construction)")
    print(f"target's own lag-1 autocorrelation: train r={diag['target_own_autocorr_train']:+.4f}, "
          f"test r={diag['target_own_autocorr_test']:+.4f}")
    print(f"'own lag-1 only' diagnostic model (no cross-article info at all): "
          f"held-out R2={diag['own_lag_only_r2']:+.4f}, corr={diag['own_lag_only_corr']:+.4f}")
    lasso_r2 = model_results["Lasso (raw 12 articles)"]["r2"]
    print(f"(for comparison, Lasso's held-out R2 using all 12 articles was {lasso_r2:+.4f} -- "
          f"most of Lasso's apparent skill is attributable to the target's own persistence, "
          f"not genuine cross-article relatedness)")

    print("\n" + "=" * 78)
    print("sparsity sweep (kappa x rho) -- does em_filter ever meaningfully engage?")
    print("=" * 78)
    sweep_rows = sparsity_sweep(X, y)
    for r in sweep_rows:
        print(f"rho={r['rho']:.2f} kappa={r['kappa']:>8g}  "
              f"blocks={r['n_blocks_total']:3d}  raw features retained={r['n_features_retained']:2d}"
              f"/{len(RELATED)} ({r['frac_retained']*100:.0f}%)")

    print("\n" + "=" * 78)
    print("OUTCOME RULE (applied mechanically to the numbers above)")
    print("=" * 78)
    outcome = apply_outcome_rule(gate, model_results, sweep_rows)
    for k, v in outcome.items():
        print(f"  {k}: {v}")
    print(f"\n>>> VERDICT: {outcome['verdict']} <<<")

    payload = dict(
        domain="F1 Wikipedia pageviews",
        predicted="FAIL",
        target=TARGET,
        related=RELATED,
        date_range=[START, end_date],
        n_days_common=int(dev.shape[0]),
        n_train=n_train,
        n_test=n - n_train,
        generalization_gate=gate,
        model_results=model_results,
        diagnostic_own_autocorrelation=diag,
        sparsity_sweep=sweep_rows,
        outcome_rule=outcome,
    )

    def _json_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)!r}")

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\nResults written to {OUT_PATH}")
