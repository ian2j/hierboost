"""W5 of the pre-registered 10-domain test (preregistration_10domain_test.md): urban
traffic congestion across a road-sensor network, predicted WORK.

Data source, verified live 2026-08-30 (see module history -- both candidates named in
the assignment were actually tried, not assumed):
  - Chicago's data.cityofchicago.org (Traffic Tracker historical congestion,
    resource sxs8-h27x) was tried FIRST, per the assignment's instruction to check both
    -- every request (curl, repeated with 20s/40s timeouts, from this environment)
    timed out completely (HTTP 000, no response), while a Chicago-side dataset-metadata
    request behaved identically. This reads as a network-reachability limitation of the
    current sandbox toward that specific host, not a Socrata/API design problem -- but
    per the assignment's "use whichever genuinely works" instruction, it is not usable
    here and the decision is not revisited.
  - NYC's data.cityofnewyork.us (Socrata, no auth) responded immediately and reliably.
    Dataset used: "DOT Traffic Speeds NBE" (resource i4gi-tjb9) -- NYC DOT's live
    Bluetooth/TRANSCOM travel-time sensor network, one row per (link, ~5-minute poll),
    111M+ rows total, updated continuously, retained with genuine multi-month history
    (verified: the 15 links used below each have ~24,000 polls spanning 2026-06-01
    through 2026-08-29 with no gaps). This is the dataset actually used.

Building a genuinely LOCAL road-network graph (the assignment's explicit design
requirement -- "one city's arterial network, not something diffuse", the specific
fix for state_econ's diffuse-nationwide-graph miss): the dataset's ~125 currently-
polling "link_id" sensors each carry a `link_points` field, an ordered polyline of
lat/lon vertices marking the physical stretch of roadway that TRANSCOM link covers.
Two links are declared graph-adjacent here if one link's END vertex sits within 80m
(haversine) of another link's START vertex -- i.e. adjacency is derived directly from
real road geometry, not from name-matching or any coarser proxy. This is a stricter,
more literal reading of "real road-network adjacency" than the state-border
county_adjacency.txt precedent (hop-count on a literal, physically-verified graph
rather than an administrative one), computed exactly the same way house style
requires: from authoritative structural data, not fit to the response.

Applying this to all ~125 currently-polling links (see explore_corridor() below, run
once and result hard-coded here -- the search itself touches no response data) finds
one clean, non-trivial connected component: 15 links along the Staten Island
Expressway (I-278) / West Shore Expressway interchange in Staten Island, both
directions, hop-distances 1-9. This is picked because it is the largest connected
local cluster the live geometry search actually returned, not hand-selected for a
particular narrative -- smaller components (e.g. a 3-node cluster in Queens) exist
too but 15 nodes is closer to this project's usual predictor-set size (state_econ:
51, streamflow: 4) and gives the sparsity machinery something real to chew on.

Target segment: pre-registered rule, same "well-connected but not the single most
extreme" logic as earthquake_japan's 3rd-most-active pick / state_econ's 3rd-highest
border-degree state -- the 3rd-highest-degree node in the 15-node adjacency graph,
computed from the graph alone before any response data is touched. Five nodes tie at
degree 3 for the 3rd-highest slot (one node has degree 4, the other nine have degree
1-2); the tie is broken by lowest link_id, also fixed before fitting. This lands on
link_id 4616197, "SIE E SOUTH AVENUE - RICHMOND AVENUE" (Staten Island Expressway
eastbound, between the Richmond Ave and South Ave interchanges), degree 3.

Data-quality disclosure (real sensor-network messiness, not smoothed away): each poll
carries a `status` code; 0 means a valid Bluetooth-matched travel-time read, negative
codes (chiefly -101) mean too few vehicle reads were matched in that window to trust
the estimate -- concentrated overnight when traffic volume is low. Coverage varies a
lot by link (36%-99% valid-status rate across the 15 links here, checked live). Only
status=0 reads are used; they are then aggregated to HOURLY MEAN speed per link (also
damps residual 5-minute sensor noise) -- turning a noisy, irregularly-gapped 5-minute
poll stream into a regular grid, the same kind of real-world cleanup step streamflow's
gap-fill and uk_weather's rolling windows already established as this project's norm.
Remaining sparse hourly gaps (mainly the two lower-coverage links) are forward-filled
up to 3 hours, matching streamflow_delaware's disclosed small-gap tolerance; any
hour where a link still has no value after that is dropped from the aligned matrix.

Task: predict the target link's hourly mean speed at hour t from the 14 OTHER links'
hourly mean speed at hour t-1 (strict lag, no leakage), graph-blocked by real hop-
distance via hierboost.kernels.graph_affinity / resolve_affinity(kind="graph") -- the
exact mechanism state_econ used for state-border adjacency, applied here at
arterial-segment scale instead of state-border scale, per the assignment's explicit
instruction to build the graph "genuinely local" this time.
"""
import json
import math
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score

from hierboost.kernels import resolve_affinity
from hierboost.spike_slab_gaussian import em_filter_gaussian

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "predom_traffic", "data")
CACHE_PATH = os.path.join(CACHE_DIR, "raw_speeds.json")
RESULTS_PATH = os.path.join(HERE, "results", "predom_traffic.json")
USER_AGENT = "hierboost-research/0.1 (research use; contact: ian2johnston@gmail.com)"
BASE_URL = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
START_DATE = "2026-06-01T00:00:00"

# The 15-node Staten Island Expressway / West Shore Expressway connected component,
# hard-coded from the one-time live geometry search (explore_corridor(), not re-run
# by default -- see module docstring). link_id -> (name, borough).
SIE_LINKS = {
    "4616193": "SIE E BRADLEY AVENUE - CLOVE ROAD",
    "4616194": "SIE E WOOLEY AVENUE - BRADLEY AVENUE",
    "4616195": "SIE E RICHMOND AVENUE - WOOLEY AVENUE",
    "4616197": "SIE E SOUTH AVENUE - RICHMOND AVENUE",
    "4616198": "WSE N-SIE E SOUTH AVENUE - SOUTH AVENUE",
    "4616199": "SIE W - WSE S SOUTH AVENUE - SOUTH AVENUE",
    "4616200": "WSE S SOUTH AVENUE - VICTORY BOULEVARD",
    "4616201": "WSE S VICTORY BOULEVARD - ARDEN AVENUE",
    "4616202": "WSE S ARDEN AVENUE - BLOOMINGDALE ROAD",
    "4616208": "SIE W BRADLEY AVENUE - WOOLEY AVENUE",
    "4616209": "SIE W WOOLEY AVENUE - RICHMOND AVENUE",
    "4616211": "SIE W RICHMOND AVENUE - SOUTH AVENUE",
    "4616212": "WSE N VICTORY BLVD - SOUTH AVENUE",
    "4616213": "WSE N ARDEN AVENUE - VICTORY BLVD",
    "4616214": "WSE N BLOOMUINGDALE ROAD - ARDEN AVENUE",
}
LINK_IDS = list(SIE_LINKS.keys())
TARGET_ID = "4616197"  # pre-registered: 3rd-highest-degree node, ties broken by lowest link_id
ADJACENCY_THRESH_M = 80.0
FFILL_LIMIT_HOURS = 3
TRAIN_FRAC = 0.8
GRAPH_BANDWIDTH = 2.0  # hops
KAPPA = 100.0
EM_KWARGS = dict(xi0=-1.0, xi1=1.0, nu=1.0, lam=1.0, filter_frac=0.2, min_features=5)

PT_RE = re.compile(r"(-?\d+\.\d+),(-?\d+\.\d+)")


# ------------------------------------------------------ one-time graph search ----

def haversine_m(p1, p2):
    R = 6371000.0
    lat1, lon1 = p1
    lat2, lon2 = p2
    p1r, p2r = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1r) * math.cos(p2r) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def explore_corridor(recent_minutes=90):
    """One-time live search (documented, not re-run by default) that produced
    SIE_LINKS/TARGET_ID above: pull every currently-polling link's endpoint geometry,
    connect END-near-START pairs within ADJACENCY_THRESH_M, and report connected
    components. Kept here for reproducibility/audit, not called by __main__."""
    q = urllib.parse.urlencode({
        "$select": "link_id,link_name,borough,link_points",
        "$where": f"data_as_of > '2026-08-29T22:00:00'",
        "$group": "link_id,link_name,borough,link_points",
        "$limit": "500",
    })
    req = urllib.request.Request(f"{BASE_URL}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read())

    links = {}
    for r in rows:
        pts = [(float(a), float(b)) for a, b in PT_RE.findall(r["link_points"])]
        if len(pts) < 2:
            continue
        links[r["link_id"]] = dict(name=r["link_name"], borough=r["borough"],
                                    start=pts[0], end=pts[-1])

    ids = list(links)
    n = len(ids)
    idx = {lid: i for i, lid in enumerate(ids)}
    A = np.zeros((n, n))
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if haversine_m(links[a]["end"], links[b]["start"]) < ADJACENCY_THRESH_M:
                A[idx[a], idx[b]] = 1
                A[idx[b], idx[a]] = 1
    D = shortest_path(csr_matrix(A), method="D", unweighted=True, directed=False)
    seen, comps = set(), []
    for lid in ids:
        i = idx[lid]
        if lid in seen or A[i].sum() == 0:
            continue
        stack, comp = [lid], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            seen.add(x)
            stack += [ids[j] for j in np.nonzero(A[idx[x]])[0] if ids[j] not in comp]
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return links, A, D, idx, comps


# ----------------------------------------------------------------- fetch/build ----

def fetch_link(link_id, force=False):
    cache_dir = os.path.join(CACHE_DIR, "by_link")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{link_id}.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    q = urllib.parse.urlencode({
        "link_id": link_id,
        "status": "0",
        "$select": "data_as_of,speed",
        "$where": f"data_as_of > '{START_DATE}'",
        "$order": "data_as_of ASC",
        "$limit": "30000",
    })
    req = urllib.request.Request(f"{BASE_URL}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rows = json.loads(resp.read())
    out = [(r["data_as_of"], float(r["speed"])) for r in rows]
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def fetch_all(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {}
    for lid in LINK_IDS:
        print(f"fetching {lid} ({SIE_LINKS[lid]})...")
        rows = fetch_link(lid, force=force)
        print(f"  {len(rows)} valid (status=0) polls")
        data[lid] = rows
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)
    return data


def build_hourly_matrix():
    raw = fetch_all()
    series = {}
    for lid in LINK_IDS:
        df = pd.DataFrame(raw[lid], columns=["ts", "speed"])
        df["ts"] = pd.to_datetime(df["ts"])
        s = df.set_index("ts")["speed"].resample("h").mean()
        series[lid] = s
    mat = pd.concat(series, axis=1)
    full_idx = pd.date_range(mat.index.min(), mat.index.max(), freq="h")
    mat = mat.reindex(full_idx)
    n_before_ffill_na = mat.isna().sum().sum()
    mat = mat.ffill(limit=FFILL_LIMIT_HOURS)
    mat = mat.dropna(how="any")
    print(f"{len(mat)} hourly rows {mat.index.min()} -> {mat.index.max()} "
          f"({n_before_ffill_na} link-hours missing before ffill(limit={FFILL_LIMIT_HOURS}), "
          f"{mat.isna().sum().sum()} still missing after -> dropped)")
    return mat


def build_graph():
    """Build the 15x15 adjacency/hop-distance matrix for SIE_LINKS from each link's
    polyline geometry (live-fetched once and cached -- the unauthenticated Socrata
    endpoint intermittently rate-limits/hangs under repeated re-querying from this
    sandbox, same as the earlier Chicago unreachability, so a cache avoids re-hitting
    it on every run once the geometry has been fetched successfully)."""
    geom_path = os.path.join(CACHE_DIR, "link_geometry.json")
    if os.path.exists(geom_path):
        with open(geom_path) as f:
            geom = json.load(f)
    else:
        q = urllib.parse.urlencode({
            "$select": "link_id,link_points",
            "$where": "link_id in(" + ",".join(f"'{i}'" for i in LINK_IDS) + ")",
            "$group": "link_id,link_points",
            "$limit": "100",
        })
        req = urllib.request.Request(f"{BASE_URL}?{q}", headers={"User-Agent": USER_AGENT})
        rows = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    rows = json.loads(resp.read())
                break
            except Exception:
                if attempt == 2:
                    raise
        geom = {r["link_id"]: r["link_points"] for r in rows}
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(geom_path, "w") as f:
            json.dump(geom, f)
    pts_by_id = {lid: [(float(a), float(b)) for a, b in PT_RE.findall(s)] for lid, s in geom.items()}
    n = len(LINK_IDS)
    idx = {lid: i for i, lid in enumerate(LINK_IDS)}
    A = np.zeros((n, n))
    for a in LINK_IDS:
        for b in LINK_IDS:
            if a == b:
                continue
            if haversine_m(pts_by_id[a][-1], pts_by_id[b][0]) < ADJACENCY_THRESH_M:
                A[idx[a], idx[b]] = 1
                A[idx[b], idx[a]] = 1
    D = shortest_path(csr_matrix(A), method="D", unweighted=True, directed=False)
    deg = A.sum(axis=1)
    return A, D, deg, idx


# ------------------------------------------------------------------- gate ----

def generalization_gate(X, y, n_train, feature_names):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    train_corr = np.array([np.corrcoef(X_train[:, j], y_train)[0, 1] for j in range(X.shape[1])])
    best_j = int(np.nanargmax(np.abs(train_corr)))
    test_corr = np.corrcoef(X_test[:, best_j], y_test)[0, 1]
    ratio = abs(test_corr) / abs(train_corr[best_j]) if train_corr[best_j] != 0 else float("nan")
    same_sign = np.sign(train_corr[best_j]) == np.sign(test_corr)
    passed = bool(same_sign and ratio >= 0.5)
    print("\n=== GENERALIZATION GATE (run first, per pre-registration) ===")
    print(f"best train-correlated raw feature: {feature_names[best_j]}  "
          f"train r={train_corr[best_j]:+.4f}  held-out test r={test_corr:+.4f}  "
          f"ratio={ratio:.3f}  same_sign={same_sign}  -> {'PASS' if passed else 'FAIL'}")
    return dict(best_feature=feature_names[best_j], train_corr=float(train_corr[best_j]),
                test_corr=float(test_corr), ratio=float(ratio) if np.isfinite(ratio) else None,
                same_sign=bool(same_sign), passed=passed)


def zscore_fit(X_train, X_test):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


# -------------------------------------------------------------- baselines ----

def run_baselines(X, y, n_train):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    results = {}

    mu_naive = np.full_like(y_test, y_train.mean())
    results["naive (train-mean constant)"] = dict(r2=r2_score(y_test, mu_naive), corr=None, n_features=0)

    lasso = LassoCV(alphas=np.logspace(-4, 1, 30), max_iter=20000, cv=5).fit(Xs_train, y_train)
    pred = lasso.predict(Xs_test)
    results["Lasso (raw lag features)"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]),
        n_features=int((lasso.coef_ != 0).sum()), alpha=float(lasso.alpha_))

    n_pc = min(8, X_train.shape[1])
    pca = PCA(n_components=n_pc).fit(Xs_train)
    lr = LinearRegression().fit(pca.transform(Xs_train), y_train)
    pred = lr.predict(pca.transform(Xs_test))
    results[f"PCA({n_pc})+linear"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]), n_features=n_pc)

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["Random Forest (raw lag features)"] = dict(
        r2=r2_score(y_test, pred), corr=float(np.corrcoef(pred, y_test)[0, 1]), n_features=X_train.shape[1])

    return results


# ------------------------------------------------------------- hierboost ----

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


def run_hierboost_flat(X, y, n_train):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    wr = np.ones(X.shape[1])
    filt = fit_em(Xs_train, y_train, wr)
    r2, corr, b = eval_em(filt, Xs_test, y_test)
    return dict(r2=r2, corr=corr, n_features=len(b.retained_idx)), filt


def run_hierboost_graph(X, y, n_train, other_hop_dist, bandwidth=GRAPH_BANDWIDTH, kappa=KAPPA):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    wr = resolve_affinity(other_hop_dist, kind="graph",
                           distance_matrix=other_hop_dist[:, None], bandwidth=bandwidth)
    filt = fit_em(Xs_train, y_train, wr, kappa=kappa)
    r2, corr, b = eval_em(filt, Xs_test, y_test)
    return dict(r2=r2, corr=corr, n_features=len(b.retained_idx)), filt, wr


def kappa_sweep(X, y, n_train, wr, label):
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    Xs_train, Xs_test = zscore_fit(X_train, X_test)
    out = []
    print(f"-- kappa sweep ({label}) --")
    for kap in [10.0, 100.0, 1000.0, 10000.0]:
        filt = fit_em(Xs_train, y_train, wr, kappa=kap)
        r2, corr, b = eval_em(filt, Xs_test, y_test)
        frac_retained = len(b.retained_idx) / X.shape[1]
        out.append(dict(kappa=kap, r2=r2, n_features=len(b.retained_idx), frac_retained=frac_retained))
        print(f"  kappa={kap:g}: R2={r2:+.4f} n_features={len(b.retained_idx)}/{X.shape[1]} "
              f"({frac_retained:.0%})")
    return out


# ----------------------------------------------------------------- main ----

if __name__ == "__main__":
    mat = build_hourly_matrix()
    A, D, deg, idx = build_graph()
    print(f"\ncorridor adjacency graph: {len(LINK_IDS)} links, "
          f"degrees={dict(zip(LINK_IDS, deg.astype(int)))}")
    print(f"target link: {TARGET_ID} ({SIE_LINKS[TARGET_ID]}), degree={int(deg[idx[TARGET_ID]])}")

    other_ids = [lid for lid in LINK_IDS if lid != TARGET_ID]
    target_col = mat[TARGET_ID].values
    other_cols = mat[other_ids].values  # (T, 14)
    y = target_col[1:]
    X = other_cols[:-1]  # lag-1 hour
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    feature_names = [f"{lid}_lag1h" for lid in other_ids]
    print(f"{n} hourly rows, {X.shape[1]} predictor links ({n_train} train / {n - n_train} test, "
          f"chronological split)")

    gate = generalization_gate(X, y, n_train, feature_names)

    print("\n=== baseline suite ===")
    baselines = run_baselines(X, y, n_train)
    for name, r in baselines.items():
        print(f"{name:<35} R2={r['r2']:+.4f}  n_features={r['n_features']}")

    print("\n=== hierboost, flat prior (uniform boost) ===")
    flat_res, flat_filt = run_hierboost_flat(X, y, n_train)
    print(f"R2={flat_res['r2']:+.4f} corr={flat_res['corr']:+.4f} n_features={flat_res['n_features']}/{X.shape[1]}")

    other_idx_g = np.array([idx[lid] for lid in other_ids])
    other_hop_dist = D[idx[TARGET_ID]][other_idx_g]
    print(f"\nhop-distances target->others: {dict(zip(other_ids, other_hop_dist.astype(int)))}")

    print("\n=== hierboost, graph-adjacency boost (bandwidth sweep) ===")
    bw_sweep = []
    for bw in [1.0, 2.0, 4.0, 8.0]:
        graph_res, graph_filt, wr_graph = run_hierboost_graph(X, y, n_train, other_hop_dist, bandwidth=bw)
        print(f"  bandwidth={bw:g} hops: R2={graph_res['r2']:+.4f} corr={graph_res['corr']:+.4f} "
              f"n_features={graph_res['n_features']}/{X.shape[1]}")
        bw_sweep.append(dict(bandwidth=bw, **graph_res))
        if bw == GRAPH_BANDWIDTH:
            main_graph_res, main_graph_filt, main_wr_graph = graph_res, graph_filt, wr_graph

    print("\n=== sparsity engagement: kappa sweep ===")
    flat_sweep = kappa_sweep(X, y, n_train, np.ones(X.shape[1]), "flat prior")
    graph_sweep = kappa_sweep(X, y, n_train, main_wr_graph, f"graph boost bw={GRAPH_BANDWIDTH:g}")

    # Mechanism check (same discipline as state_econ/streamflow): does the boosting
    # prior at least correctly re-rank POSTERIOR CONFIDENCE toward literally
    # graph-close segments? Unlike state_econ (where em_filter never dropped a
    # feature, so theta_hat stayed full-length), sparsity DOES engage here -- best.
    # theta_hat is only as long as best.retained_idx, so hop-distances must be
    # gathered through that same index set rather than compared position-for-position.
    flat_b = flat_filt.best
    graph_b = main_graph_filt.best
    rho_flat, _ = spearmanr(flat_b.theta_hat, other_hop_dist[flat_b.retained_idx])
    rho_graph, _ = spearmanr(graph_b.theta_hat, other_hop_dist[graph_b.retained_idx])
    top3_pos = graph_b.retained_idx[np.argsort(-graph_b.theta_hat)[:3]]
    top3 = [other_ids[i] for i in top3_pos]
    print(f"\nmechanism check: theta_hat vs hop-distance spearman rho -- "
          f"flat prior={rho_flat:+.3f}, graph boost={rho_graph:+.3f}; "
          f"graph-boost top-3 by theta_hat: {top3}")

    # ---------------------------------------------------------------- verdict ----
    best_conventional_r2 = max(baselines["Lasso (raw lag features)"]["r2"],
                                baselines["PCA(8)+linear"]["r2"],
                                baselines["Random Forest (raw lag features)"]["r2"])
    naive_r2 = baselines["naive (train-mean constant)"]["r2"]
    hb_r2 = main_graph_res["r2"]
    hb_n_feat = main_graph_res["n_features"]
    n_raw_feat = X.shape[1]
    frac_retained_sweep = [d["frac_retained"] for d in graph_sweep]
    sparsity_engaged = any(f <= 0.90 for f in frac_retained_sweep)

    gate_pass = gate["passed"]
    cond2a = gate_pass and (hb_r2 >= best_conventional_r2 - 0.05) and (hb_n_feat <= 0.5 * n_raw_feat)
    cond2b_mechanism = bool(np.isfinite(rho_graph) and abs(rho_graph) > abs(rho_flat) and rho_graph < 0)
    # (negative rho: theta_hat should be HIGHER for LOWER hop-distance, i.e. closer segments)
    cond2b = gate_pass and (hb_r2 > naive_r2) and cond2b_mechanism

    fail1 = not gate_pass
    fail2 = hb_r2 < naive_r2
    fail3 = (not sparsity_engaged) and (not cond2b_mechanism)

    if fail1 or fail2 or fail3:
        verdict = "FAIL"
    elif cond2a or cond2b:
        verdict = "WORK"
    else:
        verdict = "MIXED"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"gate_pass={gate_pass}  hb_r2={hb_r2:+.4f}  best_conventional_r2={best_conventional_r2:+.4f}  "
          f"naive_r2={naive_r2:+.4f}  hb_n_feat={hb_n_feat}/{n_raw_feat}  "
          f"sparsity_engaged={sparsity_engaged}  rho_flat={rho_flat:+.3f}  rho_graph={rho_graph:+.3f}  "
          f"cond2a={cond2a}  cond2b={cond2b}  fail1={fail1}  fail2={fail2}  fail3={fail3}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out = dict(
        domain="W5 urban traffic congestion network (NYC DOT Traffic Speeds NBE, "
               "Staten Island Expressway / West Shore Expressway interchange)",
        predicted="WORK",
        data_source="https://data.cityofnewyork.us/resource/i4gi-tjb9.json (Socrata SODA, no auth)",
        chicago_attempted=True,
        chicago_reachable=False,
        links=SIE_LINKS,
        target_id=TARGET_ID,
        target_name=SIE_LINKS[TARGET_ID],
        target_degree=int(deg[idx[TARGET_ID]]),
        adjacency_threshold_m=ADJACENCY_THRESH_M,
        n_obs=n,
        n_train=n_train,
        n_test=n - n_train,
        n_predictor_links=X.shape[1],
        hop_distances_target_to_others=dict(zip(other_ids, other_hop_dist.astype(int).tolist())),
        generalization_gate=gate,
        baselines=baselines,
        hierboost_flat=flat_res,
        hierboost_graph_bandwidth_sweep=bw_sweep,
        hierboost_graph_main=dict(bandwidth=GRAPH_BANDWIDTH, **main_graph_res),
        kappa_sweep_flat=flat_sweep,
        kappa_sweep_graph=graph_sweep,
        mechanism_check=dict(rho_flat=float(rho_flat), rho_graph=float(rho_graph),
                              top3_by_theta_hat=top3),
        verdict=verdict,
        verdict_components=dict(
            gate_pass=gate_pass, hb_r2=hb_r2, best_conventional_r2=best_conventional_r2,
            naive_r2=naive_r2, hb_n_feat=hb_n_feat, n_raw_feat=n_raw_feat,
            sparsity_engaged=sparsity_engaged, cond2a=cond2a, cond2b=cond2b,
            fail1=fail1, fail2=fail2, fail3=fail3),
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved {RESULTS_PATH}")
