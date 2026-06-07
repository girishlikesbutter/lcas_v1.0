#!/usr/bin/env python3
"""
scan_m048_constraints_vs_phase.py — cross-tabulate specular-peak count
against phase angle across all 100 m048 seeds.

QUESTION
========
Seed 23 at 30° phase FAILed with only 2 spec peaks (constraint-poor).
Seeds 69/28 at 68°/87° phase FAILed with alignment-cost flatness.
Are these SEPARATE failure modes, or does phase angle always correlate
with constraint count?

To find out: scan every m048 per-trajectory npz, count spec peaks (same
logic m103 uses: find_peaks on -mag_hifi with distance=5, prominence=0.3,
then filter mag<9), bucket by phase angle, and report each bucket's
constraint-count distribution.

If there exist HIGH-PHASE seeds with MANY spec peaks (say >=7), they're
the cleanest test of whether the high-phase failure mechanism is phase
per se or a phase-correlated constraint shortage.
"""

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
M048_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "per_trajectory"

M046_BASELINE = {0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}
PHASE_B_DONE = {23, 24, 28, 49, 69, 81, 90, 91}  # all phase-B runs so far

SPEC_MAG_MAX = 9.0   # m103's spec threshold
PEAK_DISTANCE = 5
PEAK_PROMINENCE = 0.3

BUCKETS = [
    ("very_low",   (0.0, 20.0)),
    ("low",        (20.0, 35.0)),
    ("mid_low",    (35.0, 50.0)),
    ("mid_high",   (50.0, 60.0)),
    ("high",       (60.0, 75.0)),
    ("very_high",  (75.0, 100.0)),
]


def count_spec_peaks(mag_hifi):
    """Same logic as m103: find minima of mag (= peaks of -mag), spec-filter."""
    peaks_idx, _ = find_peaks(-mag_hifi, distance=PEAK_DISTANCE,
                              prominence=PEAK_PROMINENCE)
    spec = peaks_idx[mag_hifi[peaks_idx] < SPEC_MAG_MAX]
    return len(peaks_idx), len(spec)


def scan_seed(seed):
    path = M048_DIR / f"traj_seed{seed:03d}.npz"
    if not path.exists():
        return None
    z = np.load(str(path), allow_pickle=True)
    pa = z["phase_angle_3d"]
    mag = z["mag_hifi"]
    n_total, n_spec = count_spec_peaks(mag)
    return {
        "seed": seed,
        "pa_med": float(np.median(pa)),
        "pa_min": float(pa.min()),
        "pa_max": float(pa.max()),
        "omega_dps": float(z["omega_mag_dps"]),
        "valid_frac": float((np.isfinite(mag) & (mag < 15)).mean()),
        "n_peaks_total": int(n_total),
        "n_spec_peaks": int(n_spec),
    }


def bucket_label(pa_med):
    for name, (lo, hi) in BUCKETS:
        if lo <= pa_med < hi:
            return name
    return "out_of_range"


def main():
    rows = [r for r in (scan_seed(s) for s in range(100)) if r is not None]
    print(f"Loaded {len(rows)} m048 seeds\n")

    # Overall stats
    n_spec = np.array([r["n_spec_peaks"] for r in rows])
    print(f"Spec-peak count distribution (all 100 seeds):")
    print(f"  min={n_spec.min()}  q25={np.percentile(n_spec, 25):.0f}  "
          f"med={int(np.median(n_spec))}  q75={np.percentile(n_spec, 75):.0f}  "
          f"max={n_spec.max()}  mean={n_spec.mean():.1f}")
    print()

    # Bucketed
    print("=== Spec-peak count by phase-angle bucket ===")
    print(f"{'bucket':<12} {'phase range':<14} {'N':>4} {'spec-peak quartiles (min/q25/med/q75/max)':>45}")
    for name, (lo, hi) in BUCKETS:
        bucket_rows = [r for r in rows if lo <= r["pa_med"] < hi]
        if not bucket_rows:
            print(f"{name:<12} [{lo:4.0f}, {hi:4.0f})   {0:>4}   (empty)")
            continue
        bp = np.array([r["n_spec_peaks"] for r in bucket_rows])
        print(f"{name:<12} [{lo:4.0f}, {hi:4.0f})   {len(bp):>4}   "
              f"{bp.min():>3} / {np.percentile(bp,25):>4.1f} / "
              f"{np.median(bp):>4.1f} / {np.percentile(bp,75):>4.1f} / "
              f"{bp.max():>3}")
    print()

    # Find well-constrained high-phase seeds (excluding m046 baseline + done)
    skip = M046_BASELINE | PHASE_B_DONE
    print(f"=== HIGHLY CONSTRAINED + HIGH PHASE (>=60° med, n_spec >=7, "
          f"excludes m046 baseline + done) ===")
    candidates = sorted(
        [r for r in rows
         if r["pa_med"] >= 60.0
         and r["n_spec_peaks"] >= 7
         and r["seed"] not in skip],
        key=lambda r: (-r["n_spec_peaks"], -r["pa_med"])
    )
    if not candidates:
        print("  NONE — suggests constraint count and high phase are confounded.")
        # Relax: any seed with >=60° and n_spec >= 5
        relaxed = sorted(
            [r for r in rows
             if r["pa_med"] >= 60.0
             and r["n_spec_peaks"] >= 5
             and r["seed"] not in skip],
            key=lambda r: (-r["n_spec_peaks"], -r["pa_med"])
        )
        if relaxed:
            print(f"\n  Relaxed (n_spec >= 5, still high phase):")
            for r in relaxed[:10]:
                print(f"    seed {r['seed']:3d}: pa_med={r['pa_med']:5.2f}°  "
                      f"n_spec={r['n_spec_peaks']:2d}  omega={r['omega_dps']:.3f} dps  "
                      f"valid={r['valid_frac']:.1%}")
    else:
        print(f"  Found {len(candidates)} candidates:")
        for r in candidates[:10]:
            print(f"    seed {r['seed']:3d}: pa_med={r['pa_med']:5.2f}°  "
                  f"pa_range=[{r['pa_min']:4.1f},{r['pa_max']:4.1f}]  "
                  f"n_spec={r['n_spec_peaks']:2d}  "
                  f"omega={r['omega_dps']:.3f} dps  valid={r['valid_frac']:.1%}")
    print()

    # Also: the inverse — low-phase seeds with MANY spec peaks
    print(f"=== LOW PHASE (<35° med) WITH MANY SPEC PEAKS (>=5), excludes m046 baseline + done ===")
    lowphase = sorted(
        [r for r in rows
         if r["pa_med"] < 35.0
         and r["n_spec_peaks"] >= 5
         and r["seed"] not in skip],
        key=lambda r: (-r["n_spec_peaks"], r["pa_med"])
    )
    if not lowphase:
        print("  NONE — low-phase seeds appear to be constraint-poor by geometry.")
    else:
        for r in lowphase[:10]:
            print(f"    seed {r['seed']:3d}: pa_med={r['pa_med']:5.2f}°  "
                  f"n_spec={r['n_spec_peaks']:2d}  omega={r['omega_dps']:.3f} dps  "
                  f"valid={r['valid_frac']:.1%}")


if __name__ == "__main__":
    main()
