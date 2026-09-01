"""Turn the raw USGS event catalog into a weekly cell x count design matrix.

Grid: regular 2.0deg (lon) x 1.5deg (lat) cells over the fetch bounding box --
roughly square (~180km x ~170km at this latitude), an objective, reproducible
partition (not hand-drawn regional boundaries). Cells below a minimum total-event
threshold are dropped (mostly open ocean / aseismic cells that would otherwise be
near-all-zero noise columns).

Target cell selection is a pre-registered rule, decided before looking at any model
result: the 3rd-most-active cell by total event count over the period (not the #1
cell, to avoid the demo being "can you predict the one dominant cluster from itself";
not hand-picked for a specific known sequence, to avoid the appearance of cherry-
picking a favorable target after the fact).
"""
import numpy as np

LON_BIN, LAT_BIN = 2.0, 1.5
MIN_CELL_EVENTS = 80
DAY_MS = 24 * 3600 * 1000
WEEK_MS = 7 * DAY_MS


def build(bin_ms, out_path, label):
    d = np.load("earthquake_japan/data/events_raw.npz")
    lon, lat, time_ms, mag = d["lon"], d["lat"], d["time_ms"], d["mag"]

    lon_idx = np.floor((lon - lon.min()) / LON_BIN).astype(int)
    lat_idx = np.floor((lat - lat.min()) / LAT_BIN).astype(int)
    cell_id = lon_idx * 1000 + lat_idx  # unique combined key

    uniq_cells, counts = np.unique(cell_id, return_counts=True)
    keep_cells = uniq_cells[counts >= MIN_CELL_EVENTS]

    # cell centroids (lon/lat), in the same key order as keep_cells
    centroids = []
    for c in keep_cells:
        li, ai = c // 1000, c % 1000
        centroids.append((lon.min() + (li + 0.5) * LON_BIN, lat.min() + (ai + 0.5) * LAT_BIN))
    centroids = np.array(centroids)

    t0 = time_ms.min()
    bin_idx_all = np.floor((time_ms - t0) / bin_ms).astype(int)
    n_bins = bin_idx_all.max() + 1

    K = len(keep_cells)
    cell_lookup = {c: k for k, c in enumerate(keep_cells)}
    counts_matrix = np.zeros((n_bins, K), dtype=np.int64)
    for i in range(len(lon)):
        c = cell_id[i]
        if c in cell_lookup:
            counts_matrix[bin_idx_all[i], cell_lookup[c]] += 1

    totals = counts_matrix.sum(axis=0)
    order = np.argsort(totals)[::-1]
    target_k = order[2]  # 3rd-most-active cell, pre-registered rule (see docstring) --
                          # same rule/target cell reused at both granularities so they're
                          # directly comparable, not two different cherry-picked stories

    print(f"\n[{label}] {len(uniq_cells)} occupied cells total, {len(keep_cells)} kept "
          f"(>={MIN_CELL_EVENTS} events), {n_bins} bins")
    print(f"[{label}] target cell {target_k} at ({centroids[target_k,0]:.1f}E, "
          f"{centroids[target_k,1]:.1f}N), mean={counts_matrix[:, target_k].mean():.3f}, "
          f"max={counts_matrix[:, target_k].max()}")

    np.savez(out_path, counts=counts_matrix, centroids=centroids, target_k=target_k,
              n_bins=n_bins, bin_ms=bin_ms, t0=t0)
    print(f"[{label}] saved -> {out_path}")


def main():
    build(WEEK_MS, "earthquake_japan/data/weekly_counts.npz", "weekly")
    build(DAY_MS, "earthquake_japan/data/daily_counts.npz", "daily")


if __name__ == "__main__":
    main()
