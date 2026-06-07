#!/usr/bin/env python3
"""
select_phase_descending.py — pick 3 m048 seeds at decreasing phase angles
spanning the known-good band, starting just under seed 091's 51.7° median.

CONTEXT
=======
Phase-B pilot + gap-probe results so far:
    seed 024 @ 15.3°  PARTIAL
    seed 081 @ 45.6°  OK
    seed 091 @ 51.7°  OK  <- highest confirmed OK median
    seed 069 @ 67.7°  FAIL (upstream grid+NM)
    seed 028 @ 87.3°  FAIL (upstream grid+NM)

This script picks 3 seeds at descending phase angles to verify consistency
in the known-good band: ~50° (just under 51.7°), ~40°, ~30°.

Selection criteria (same as prior picks):
    1. Median phase angle near target
    2. valid_frac > 10%
    3. omega_mag in [0.5, 1.5] dps
    4. NOT in m046 baseline {0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}
    5. NOT already run {24, 28, 69, 81, 91}

Saves picks + rationale to
data/results/inversion_diagnostics/phase_B_descending_seeds.json.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
M048_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "per_trajectory"
OUT_JSON = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "phase_B_descending_seeds.json"

M046_BASELINE = {0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}
PHASE_B_DONE = {24, 28, 69, 81, 91}
SKIP = M046_BASELINE | PHASE_B_DONE

VALID_FRAC_MIN = 0.10
OMEGA_MAG_LO_DPS = 0.5
OMEGA_MAG_HI_DPS = 1.5

TARGETS = [("~50 deg (just under 51.7)", 50.0),
           ("~40 deg (mid safe-band)",   40.0),
           ("~30 deg (low safe-band)",   30.0)]


def scan_seed(seed):
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
    rows = [r for r in (scan_seed(s) for s in range(100)) if r is not None]
    print(f"Loaded {len(rows)} m048 seeds")

    elig = [
        r for r in rows
        if r["seed"] not in SKIP
        and r["valid_frac"] > VALID_FRAC_MIN
        and OMEGA_MAG_LO_DPS <= r["omega_dps"] <= OMEGA_MAG_HI_DPS
    ]
    print(f"Eligible after filters: {len(elig)}\n")

    picks = {}
    used = set()
    for label, target in TARGETS:
        pool = [r for r in elig if r["seed"] not in used]
        pool.sort(key=lambda r: abs(r["pa_med"] - target))
        print(f"=== {label} ===")
        for r in pool[:5]:
            print(f"  seed {r['seed']:3d}: pa_med={r['pa_med']:5.2f}°  "
                  f"pa_range=[{r['pa_min']:5.2f}, {r['pa_max']:5.2f}]  "
                  f"omega={r['omega_dps']:.3f} dps  valid={r['valid_frac']:.1%}")
        winner = pool[0]
        used.add(winner["seed"])
        picks[label] = winner
        print(f"  -> picked seed {winner['seed']} (pa_med={winner['pa_med']:.2f} deg)\n")

    # Order by descending phase angle (user requested)
    ordered = sorted(picks.items(), key=lambda kv: kv[1]["pa_med"], reverse=True)
    print("Launch order (descending phase):")
    for label, r in ordered:
        print(f"  seed {r['seed']:3d} @ pa_med={r['pa_med']:.2f} deg  ({label})")

    payload = {
        "description": (
            "Phase-B descending-phase run (3 seeds) to verify consistency "
            "inside the known-good safe band. Targets: ~50, ~40, ~30 deg."
        ),
        "skip_criteria": {
            "m046_baseline": sorted(M046_BASELINE),
            "phase_B_previously_run": sorted(PHASE_B_DONE),
        },
        "validity_filters": {
            "valid_frac_min": VALID_FRAC_MIN,
            "omega_mag_dps_range": [OMEGA_MAG_LO_DPS, OMEGA_MAG_HI_DPS],
        },
        "targets_and_picks": {
            label: {"target_deg": t,
                    "seed": picks[label]["seed"],
                    "pa_med_deg": picks[label]["pa_med"],
                    "pa_min_deg": picks[label]["pa_min"],
                    "pa_max_deg": picks[label]["pa_max"],
                    "omega_mag_dps": picks[label]["omega_dps"],
                    "valid_frac": picks[label]["valid_frac"]}
            for label, t in TARGETS
        },
        "launch_order_descending_phase": [
            {"seed": r["seed"], "pa_med_deg": r["pa_med"], "label": label}
            for label, r in ordered
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    print("\nCommands (run in order, each ~13-16 min wall):")
    for label, r in ordered:
        print(f"  python3 notebooks/inversion/invert.py --seed {r['seed']} --traj-source m048")


if __name__ == "__main__":
    main()
