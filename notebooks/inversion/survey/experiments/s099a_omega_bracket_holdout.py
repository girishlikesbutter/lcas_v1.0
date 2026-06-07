"""s099a — held-out validation of the s019 LS-bracket |w| prior + seed-116 detail.

Goal-1 validity gate. s019 confirmed the bracket strategy covers truth-|w| within
5% on 98/100 seeds, but it only ran seeds 0-99. m048 has 120 seeds, so 100-119
are a genuine held-out test set the bracket policy never saw. Seed 116 (the one
s098 inverted) is in that held-out set.

This reuses s019's run_seed UNCHANGED (no re-tuning of any bracket parameter) and:
  1. reports bracket coverage on held-out seeds 100-119 (in_grid + within_5pct),
  2. dumps seed 116's bracket in full,
  3. reports the OPERATIONAL window width factor hi/lo per seed -- the number
     goal 2 hinges on, because at inversion time we cannot collapse the bracket
     to the nearest-to-truth grid point; the blind |w| acceptance window is the
     full [lo, hi] span.

Pure spectral, no surrogate, no propagation. Seconds of wall.
"""
import json
import sys
import time
from pathlib import Path
import importlib

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

# reuse s019's run_seed unchanged (workspace-internal import; not buggy-era)
s019 = importlib.import_module("s019_ls_bracket_omega_mag")

OUT = SURVEY / "results" / "s099"
OUT.mkdir(parents=True, exist_ok=True)

HELDOUT = list(range(100, 120))
FOCUS = 116


def main():
    t0 = time.time()
    print("=== s099a — held-out bracket coverage (seeds 100-119) + seed-116 detail ===", flush=True)

    rows = {}
    for seed in HELDOUT:
        r = s019.run_seed(seed)
        if r is None or r.get("no_significant_peaks"):
            print(f"  seed {seed}: NO significant LS peaks (bracket undefined)", flush=True)
            continue
        rows[seed] = r

    # held-out coverage (bracket strategy only)
    n = len(rows)
    in_grid = sum(r["bracket_in_grid"] for r in rows.values())
    within5 = sum(r["bracket_n_within_5pct"] >= 1 for r in rows.values())
    offsets = np.array([r["bracket_nearest_pct"] for r in rows.values()])
    factors = np.array([r["bracket_hi"] / r["bracket_lo"] for r in rows.values()])

    print(f"\n--- HELD-OUT coverage (n={n} seeds with peaks, bracket strategy) ---", flush=True)
    print(f"  in_grid (truth between lo,hi):  {in_grid}/{n}", flush=True)
    print(f"  >=1 grid pt within 5% of truth: {within5}/{n}", flush=True)
    print(f"  nearest-offset  p50 {np.median(offsets):.2f}%  p90 {np.percentile(offsets,90):.2f}%  "
          f"max {offsets.max():.2f}%", flush=True)
    print(f"  window factor hi/lo  p50 {np.median(factors):.2f}x  p90 {np.percentile(factors,90):.2f}x  "
          f"max {factors.max():.2f}x", flush=True)

    # which seeds (if any) the bracket MISSES
    missed = [s for s, r in rows.items() if not r["bracket_in_grid"]]
    if missed:
        print(f"  MISSED (truth outside [lo,hi]): {missed}", flush=True)

    # ---- seed 116 detail ----
    r = rows.get(FOCUS)
    print(f"\n--- seed {FOCUS} detail ---", flush=True)
    if r is None:
        print("  no significant peaks; bracket undefined.", flush=True)
    else:
        tmag = r["truth_mag_rad_s"]
        lo, hi = r["bracket_lo"], r["bracket_hi"]
        print(f"  truth |w|       : {tmag:.6f} rad/s ({r['truth_mag_dps']:.4f} dps)", flush=True)
        print(f"  bracket [lo,hi] : [{lo:.6f}, {hi:.6f}] rad/s  (factor {hi/lo:.2f}x)", flush=True)
        print(f"  truth in bracket: {r['bracket_in_grid']}  | nearest grid offset {r['bracket_nearest_pct']:.2f}%", flush=True)
        print(f"  n grid pts <=5% : {r['bracket_n_within_5pct']}  | grid size {r['bracket_grid_size']}", flush=True)
        print(f"  LS-top1 base    : {r['ls_top1_base']:.6f} rad/s  (offset to truth {r['ls_nearest_pct']:.2f}%, "
              f"in_grid {r['ls_in_grid']})", flush=True)
        print(f"  n significant peaks: {r['n_significant_peaks']}", flush=True)
        # compare the cheating band vs the blind bracket, as |w| windows
        cheat_lo, cheat_hi = 0.70 * tmag, 1.30 * tmag
        print(f"\n  cheating band  [0.70,1.30]*truth = [{cheat_lo:.6f}, {cheat_hi:.6f}]  (factor 1.86x)", flush=True)
        print(f"  blind bracket  [lo,hi]            = [{lo:.6f}, {hi:.6f}]  (factor {hi/lo:.2f}x)", flush=True)
        print(f"  blind window is {(hi/lo)/1.857:.2f}x WIDER than the cheating band", flush=True)

    meta = dict(
        heldout_seeds=HELDOUT, n_with_peaks=n,
        bracket_in_grid=in_grid, bracket_within_5pct=within5,
        offset_p50=float(np.median(offsets)), offset_p90=float(np.percentile(offsets, 90)),
        offset_max=float(offsets.max()),
        factor_p50=float(np.median(factors)), factor_p90=float(np.percentile(factors, 90)),
        factor_max=float(factors.max()), missed=missed,
        seed116={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                 for k, v in (rows.get(FOCUS) or {}).items()},
        wall_s=time.time() - t0,
    )
    with open(OUT / "bracket_holdout.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'bracket_holdout.json'}\nWall: {meta['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
