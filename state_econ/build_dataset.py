"""Build the state-unemployment dataset: monthly LAUS unemployment rate (FRED, no key,
data/*.csv, one file per state postal code + 'UR') aligned across all 50 states + DC,
plus two independent affinity structures for the boosting prior:
  - graph: literal state-border adjacency (derived from the Census Bureau's authoritative
    county_adjacency.txt by collapsing county pairs to their 2-letter state codes, then
    graph shortest-path hop-distance via scipy) -- kernels.graph_affinity, never
    exercised on real data anywhere else in this project (every prior spatial demo used
    a continuous coordinate kernel instead).
  - centroid: state capital lat/lon (public reference data) as a physical-distance stand-in,
    the same kind of kernel earthquake_japan/uk_weather already used -- kept as an
    in-project comparison arm, not the experiment's main point.
"""
import csv
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
          "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
          "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
          "WI","WY","DC"]

NAME_BY_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

TARGET_STATE = "CO"  # pre-registered: 3rd-highest border-degree state (see project memory),
                      # same "not the single most extreme, but well-connected" logic as
                      # earthquake_japan's "3rd-most-active" pick, chosen before any fit


def load_unemployment():
    series = {}
    for st in STATES:
        df = pd.read_csv(f"state_econ/data/{st}UR.csv", parse_dates=["observation_date"])
        df = df.rename(columns={f"{st}UR": st}).set_index("observation_date")
        series[st] = df[st]
    mat = pd.concat(series, axis=1).dropna(how="any")
    return mat  # index=date, columns=state abbr


def build_graph_distance():
    pairs = json.load(open("state_econ/state_adjacency_pairs.json"))
    idx = {s: i for i, s in enumerate(STATES)}
    n = len(STATES)
    rows, cols = [], []
    for a, b in pairs:
        if a in idx and b in idx:
            rows += [idx[a], idx[b]]
            cols += [idx[b], idx[a]]
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    D = shortest_path(A, method="D", unweighted=True, directed=False)
    return D  # (51, 51) hop-count matrix, inf if disconnected (shouldn't happen for CONUS+DC; AK/HI isolated)


def build_centroids():
    caps = {}
    with open("state_econ/state_capitals.csv") as f:
        for row in csv.DictReader(f):
            caps[row["name"]] = (float(row["latitude"]), float(row["longitude"]))
    caps["District of Columbia"] = (38.9072, -77.0369)
    return np.array([caps[NAME_BY_ABBR[s]] for s in STATES])


if __name__ == "__main__":
    mat = load_unemployment()
    D_graph = build_graph_distance()
    centroids = build_centroids()
    target_k = STATES.index(TARGET_STATE)

    print(f"{mat.shape[0]} months ({mat.index[0].date()} to {mat.index[-1].date()}), "
          f"{mat.shape[1]} states, target={TARGET_STATE} (k={target_k})")
    finite = np.isfinite(D_graph)
    print(f"graph hop-distance: min={D_graph[finite & (D_graph>0)].min():.0f}, "
          f"max={D_graph[finite].max():.0f}, disconnected pairs={ (~finite).sum() - n if (n:=D_graph.shape[0]) else 0}")

    np.savez("state_econ/data/dataset.npz",
             values=mat.values, dates=mat.index.values.astype("datetime64[M]").astype(str),
             states=np.array(STATES), graph_dist=D_graph, centroids=centroids,
             target_k=target_k)
    print("saved state_econ/data/dataset.npz")
