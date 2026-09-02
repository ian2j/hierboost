"""Extends Ian's homotopy-exception zero dataset by scanning the gaps between his
original 10 window centers (reusing his exact scan_window() method), writing to a
separate file rather than editing his original data."""
import sys, time, json
sys.path.insert(0, '/home/ian/Research/Riemann/writeup')
import mpmath as mp
mp.mp.dps = 20
import homotopy_exception_gap_scan as hs

OUT = "/home/ian/Research/Math/prompts/homotopy_exception_gap_data_extra.json"
# midpoints of the gaps between the original scan's 10 centers (110..1200)
centers = [155, 250, 350, 455, 565, 685, 825, 975, 1125]

if __name__ == "__main__":
    all_rows = []
    t_start = time.time()
    for c in centers:
        w0, w1 = c - 11, c + 11
        t0 = time.time()
        rows = hs.scan_window(w0, w1, n_steps=35)
        dt = time.time() - t0
        n_exc = sum(1 for r in rows if not r['on_line'])
        print(f"window [{w0},{w1}] M~{rows[0]['M'] if rows else '?'}: "
              f"{len(rows)} zeros, {n_exc} off-line, {dt:.1f}s "
              f"(total elapsed {time.time()-t_start:.0f}s)", flush=True)
        all_rows.extend(rows)
        with open(OUT, 'w') as f:
            json.dump(all_rows, f, indent=1)

    print(f"\nDONE. {len(all_rows)} total new zeros scanned, "
          f"{sum(1 for r in all_rows if not r['on_line'])} off-line exceptions, "
          f"{time.time()-t_start:.0f}s total")
