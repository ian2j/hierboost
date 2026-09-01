"""Turn daily precipitation into weekly/daily wet-day COUNT matrices, matching
earthquake_japan/build_dataset.py's structure exactly for a clean apples-to-apples
comparison. Wet day := precip_sum >= 1.0mm (standard climatological threshold).
"""
import numpy as np

WET_THRESHOLD_MM = 1.0
DAY_MS = 24 * 3600 * 1000
WEEK_MS = 7 * DAY_MS


def build(bin_days, out_path, label):
    d = np.load("uk_weather/data/precip_raw.npz")
    precip, centroids = d["precip"], d["centroids"]  # (n_points, n_days), (n_points, 2) lon/lat
    n_points, n_days = precip.shape

    wet = (precip >= WET_THRESHOLD_MM).astype(np.int64)  # (n_points, n_days), 0/1
    wet[np.isnan(precip)] = 0

    n_bins = n_days // bin_days
    wet_trim = wet[:, :n_bins * bin_days]
    counts_matrix = wet_trim.reshape(n_points, n_bins, bin_days).sum(axis=2).T  # (n_bins, n_points)

    totals = counts_matrix.sum(axis=0)
    order = np.argsort(totals)[::-1]
    target_k = order[2]  # 3rd-most-active point, same pre-registered rule as earthquake_japan

    print(f"[{label}] {n_points} points, {n_bins} bins of {bin_days} day(s)")
    print(f"[{label}] target point {target_k} at ({centroids[target_k,0]:.1f}E, "
          f"{centroids[target_k,1]:.1f}N), mean={counts_matrix[:, target_k].mean():.3f}, "
          f"max={counts_matrix[:, target_k].max()} (out of {bin_days} days/bin)")

    np.savez(out_path, counts=counts_matrix, centroids=centroids, target_k=target_k,
              n_bins=n_bins, bin_days=bin_days)
    print(f"[{label}] saved -> {out_path}\n")


def main():
    build(7, "uk_weather/data/weekly_wetdays.npz", "weekly")
    build(1, "uk_weather/data/daily_wetdays.npz", "daily")


if __name__ == "__main__":
    main()
