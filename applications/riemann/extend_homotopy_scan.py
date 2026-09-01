"""Generate ADDITIONAL homotopy-exception-labeled zeros, filling gaps the original
~/Research/Riemann/writeup/homotopy_exception_gap_scan.py left unscanned between its
10 window centers (110,200,...,1200, width 22 each -- big gaps between them). Reuses
that script's exact scan_window() method (same mpmath dps=20/n_steps=35 settings) so
results are directly poolable with the original 150-zero dataset.

Deliberately writes to a SEPARATE file, not Ian's original homotopy_exception_gap_data.json
-- this is my own analysis extension, not an edit to his research artifact. Sticks to
t<1200 (new centers fill gaps strictly BETWEEN the original windows) since a timing probe
at t~1500 showed cost rises sharply with t (single window >3.5min and still running,
vs ~90s for a similar-width window at t~160) -- likely because M=floor(t/pi) (and hence
the size of every sum in the homotopy tracking) grows with t.
"""
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
