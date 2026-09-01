"""Fetch daily precipitation for a regular grid of points across the UK & Ireland
from Open-Meteo's historical archive API (ERA5-based reanalysis, no key needed,
generous free tier). Deliberately picked as the counter-test to earthquake_japan/:
same spatial-count-forecasting design, but a domain where the correlation-generating
mechanism (synoptic-scale weather systems) is physically stable and slow-moving
(days, not hours), unlike earthquake triggering's fast, erratic point process.
"""
import json
import time
import urllib.request
import numpy as np

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2023-12-31"

LAT_MIN, LAT_MAX, LAT_STEP = 50.0, 59.0, 1.2
LON_MIN, LON_MAX, LON_STEP = -8.0, 2.0, 1.5


def fetch_point(lat, lon):
    url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
           f"&daily=precipitation_sum&timezone=UTC")
    req = urllib.request.Request(url, headers={"User-Agent": "hierboost-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["daily"]["time"], data["daily"]["precipitation_sum"]


def main():
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, LAT_STEP)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, LON_STEP)
    grid = [(round(la, 3), round(lo, 3)) for la in lats for lo in lons]
    print(f"{len(grid)} grid points to fetch")

    times_ref = None
    series = []
    centroids = []
    for i, (lat, lon) in enumerate(grid):
        try:
            times, precip = fetch_point(lat, lon)
        except Exception as e:
            print(f"  [{i}] ({lat},{lon}) FAILED: {e}")
            continue
        precip = np.array([np.nan if v is None else v for v in precip], dtype=float)
        if times_ref is None:
            times_ref = times
        elif times != times_ref:
            print(f"  [{i}] ({lat},{lon}) date mismatch, skipping")
            continue
        series.append(precip)
        centroids.append((lon, lat))
        if i % 10 == 0:
            print(f"  [{i}/{len(grid)}] fetched ({lat},{lon}), "
                  f"nan_frac={np.isnan(precip).mean():.3f}")
        time.sleep(0.15)

    precip_matrix = np.array(series)  # (n_points, n_days)
    centroids = np.array(centroids)
    print(f"\nfetched {precip_matrix.shape[0]} points x {precip_matrix.shape[1]} days")

    np.savez("uk_weather/data/precip_raw.npz",
             precip=precip_matrix, centroids=centroids,
             times=np.array(times_ref))
    print("saved to uk_weather/data/precip_raw.npz")


if __name__ == "__main__":
    main()
