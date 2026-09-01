"""Fetch the Japan-region earthquake catalog from USGS (FDSNWS event API, no key
needed) and cache it locally. Bounding box covers the Japanese archipelago and
surrounding subduction zones (Kuril trench to the Ryukyus).

Period starts 2013-01-01, not 2010: the 2011 Tohoku-oki M9.0 event's immediate
aftershock sequence (Omori-law decay, extremely high rate for the first ~1-2 years)
would otherwise dominate every regional count series and swamp everything else --
starting after that decay gives a more stationary background-seismicity regime while
Japan is still, by a wide margin, one of the most seismically active regions on
Earth. Disclosed choice, not hidden.
"""
import json
import time
import urllib.request
import numpy as np

USER_AGENT = "hierboost-research/0.1 (research use; contact: ian2johnston@gmail.com)"
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Japan + surrounding subduction zones
MINLAT, MAXLAT = 24.0, 46.0
MINLON, MAXLON = 122.0, 146.0
MIN_MAG = 3.0
START, END = "2013-01-01", "2023-12-31"


def fetch_year(year):
    url = (f"{BASE_URL}?format=geojson&starttime={year}-01-01&endtime={year}-12-31"
           f"&minlatitude={MINLAT}&maxlatitude={MAXLAT}"
           f"&minlongitude={MINLON}&maxlongitude={MAXLON}&minmagnitude={MIN_MAG}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["features"]


def main():
    all_events = []
    for year in range(2013, 2024):
        feats = fetch_year(year)
        print(f"{year}: {len(feats)} events (M>={MIN_MAG})")
        for f in feats:
            lon, lat, depth = f["geometry"]["coordinates"]
            all_events.append(dict(
                time_ms=f["properties"]["time"],
                mag=f["properties"]["mag"],
                lon=lon, lat=lat, depth=depth,
                place=f["properties"]["place"],
            ))
        time.sleep(0.5)  # polite pacing, not rate-limit-mandated but good practice

    print(f"total: {len(all_events)} events")
    times = np.array([e["time_ms"] for e in all_events], dtype=np.int64)
    mags = np.array([e["mag"] for e in all_events], dtype=np.float64)
    lons = np.array([e["lon"] for e in all_events], dtype=np.float64)
    lats = np.array([e["lat"] for e in all_events], dtype=np.float64)
    depths = np.array([e["depth"] for e in all_events], dtype=np.float64)

    np.savez("earthquake_japan/data/events_raw.npz",
             time_ms=times, mag=mags, lon=lons, lat=lats, depth=depths)
    print("saved to earthquake_japan/data/events_raw.npz")


if __name__ == "__main__":
    main()
