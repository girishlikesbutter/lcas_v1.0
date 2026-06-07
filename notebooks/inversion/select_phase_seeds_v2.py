#!/usr/bin/env python3
"""
select_phase_seeds_v2.py — pick 2 m048 seeds for Phase-B pilot expansion.

CONTEXT
=======
The original Phase-B pilot (data/results/inversion_diagnostics/phase_B_pilot_seeds.json)
picked 3 seeds spanning the 9-95° phase-angle distribution:
    seed 024 @ 15.3° phase → PARTIAL
    seed 091 @ 51.7° phase → OK
    seed 028 @ 87.3° phase → FAIL (upstream grid+NM can't find basin)

Highest confirmed OK so far: ~59° (seed 91's max). FAIL at 87°.
27°-wide gap between known-good and known-bad — no intermediate sample.

This script picks two ADDITIONAL m048 seeds to probe that gap:
    "safe":   median phase ~45° (inside the known-good 30-60° band)
    "past":   median phase ~65° (just past seed 91's 58.8° max)

Selection criteria (same as original pilot):
    1. Median phase angle near target
    2. valid_frac > 10% (LC not saturated > 90% of epochs)
    3. omega_mag in [0.5, 1.5] dps (mainstream peak-count range)
    4. NOT in m046 baseline cohort {0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}
    5. NOT already run in Phase-B pilot {24, 28, 91}

Reads all 100 m048 per-trajectory npz files. Saves picks + rationale to
data/results/inversion_diagnostics/phase_B_gap_seeds.json so the choice
is auditable.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
M048_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "per_trajectory"
OUT_JSON = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "phase_B_gap_seeds.json"

# Seeds to skip
M046_BASELINE = {0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}
PHASE_B_DONE = {24, 28, 91}  # 24 is in both, but listed for clarity
SKIP = M046_BASELINE | PHASE_B_DONE

# Validity filters
VALID_FRAC_MIN = 0.10
OMEGA_MAG_LO_DPS = 0.5
OMEGA_MAG_HI_DPS = 1.5

# Targets
TARGET_SAFE = 45.0
TARGET_PAST = 65.0


def scan_seed(seed):
    """Load one m048 per-trajectory file and return summary dict."""
    path = M048_DIR / f"traj_seed{seed:03d}.npz"
    if not path.exists():
        return None
    z = np.load(str(path), allow_pickle=True)
    pa = z["phase_angle_3d"]
    mag = z["mag_hifi"]
    valid = np.isfinite(mag) & (mag < 15.0)
    return {
        "seed": seed,
        "pa_min": float(pa.min()),
        "pa_med": float(np.median(pa)),
        "pa_max": float(pa.max()),
        "valid_frac": float(valid.mean()),
        "omega_dps": float(z["omega_mag_dps"]),
        "omega0_rad": [float(x) for x in z["omega0_rad"]],
        "q0_wxyz": [float(x) for x in z["q0_wxyz"]],
        "start_et": float(z["start_et"]),
    }


def main():
    if not M048_DIR.exists():
        raise FileNotFoundError(f"m048 per-trajectory dir missing: {M048_DIR}")

    print(f"Scanning m048 per-trajectory files at {M048_DIR}")
    rows = []
    for s in range(100):
        r = scan_seed(s)
        if r is None:
            continue
        rows.append(r)
    print(f"  Loaded {len(rows)} seeds\n")

    # Quick distribution sanity print
    pa_meds = np.array([r["pa_med"] for r in rows])
    print("Population phase-angle distribution (median per seed):")
    print(f"  min={pa_meds.min():.2f}°  q25={np.percentile(pa_meds, 25):.2f}°  "
          f"med={np.median(pa_meds):.2f}°  q75={np.percentile(pa_meds, 75):.2f}°  "
          f"max={pa_meds.max():.2f}°\n")

    # Filter to eligibles
    elig = [
        r for r in rows
        if r["seed"] not in SKIP
        and r["valid_frac"] > VALID_FRAC_MIN
        and OMEGA_MAG_LO_DPS <= r["omega_dps"] <= OMEGA_MAG_HI_DPS
    ]
    print(f"After SKIP + valid_frac + omega_mag filters: {len(elig)} eligible\n")

    def pick_nearest(target, pool, label):
        pool_sorted = sorted(pool, key=lambda r: abs(r["pa_med"] - target))
        print(f"=== {label} (target {target:.0f}° phase) ===")
        print(f"  top-5 by |pa_med - target|:")
        for r in pool_sorted[:5]:
            print(f"    seed {r['seed']:3d}: pa_med={r['pa_med']:.2f}°  "
                  f"pa_range=[{r['pa_min']:.1f}, {r['pa_max']:.1f}]  "
                  f"omega={r['omega_dps']:.3f} dps  "
                  f"valid={r['valid_frac']:.1%}")
        winner = pool_sorted[0]
        print(f"  → picked seed {winner['seed']} (pa_med={winner['pa_med']:.2f}°)\n")
        return winner

    safe = pick_nearest(TARGET_SAFE, elig, "SAFE ZONE")
    past = pick_nearest(TARGET_PAST, elig, "PAST-CEILING ZONE")

    payload = {
        "description": (
            "Phase-B gap-probe seed picks (2026-04-17). Expands the original "
            "pilot with one seed in the known-safe ~30-60° band and one just "
            "past seed 091's 58.8° ceiling. Intent: find where the pipeline "
            "transitions from OK→FAIL between 59° and 87°."
        ),
        "skip_criteria": {
            "m046_baseline": sorted(M046_BASELINE),
            "phase_B_pilot_done": sorted(PHASE_B_DONE),
        },
        "validity_filters": {
            "valid_frac_min": VALID_FRAC_MIN,
            "omega_mag_dps_range": [OMEGA_MAG_LO_DPS, OMEGA_MAG_HI_DPS],
        },
        "targets": {
            "safe_target_deg": TARGET_SAFE,
            "past_target_deg": TARGET_PAST,
        },
        "picks": {
            "safe_zone": safe,
            "past_ceiling": past,
        },
        "known_pilot_results_for_reference": {
            "seed_024_pa_15p3": "PARTIAL (q0=5.03°, w=0.99°, hifi=0.034)",
            "seed_091_pa_51p7": "OK (q0=0.05°, w=0.01°, hifi=0.0025)",
            "seed_028_pa_87p3": "FAIL (upstream grid+NM; geo hang in m103 Pool(24))",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved picks: {OUT_JSON}")
    print(f"\nNext: python3 notebooks/inversion/invert.py --seed {safe['seed']} --traj-source m048")
    print(f"Then: python3 notebooks/inversion/invert.py --seed {past['seed']} --traj-source m048")


if __name__ == "__main__":
    main()
