"""s042 — joint-LM basin radius cohort probe (10 stratified seeds).

Extends s040 to 10 new seeds chosen to span the cohort |ω| range plus
multi-solution-rich and narrow-basin candidates. q0 grid is extended to
{2, 5, 10, 15, 20, 25, 30, 45, 60}° based on s041's finding that the seed-23
basin edge sits between 45° and 60°.

Seeds:
  q10 79 (n_rot<2 tail)        |ω|=0.121 dps
  q15 57 (n_rot<2 boundary)    |ω|=0.201
  q25 62                       |ω|=0.436
  q50 6  (s006/s011 narrow)    |ω|=0.713
  q75 14                       |ω|=1.229
  q90 19 (max |ω|)             |ω|=1.476
  multi-sol 84 (s014b class_2) |ω|=0.507
  multi-sol 16 (n_rot<2)       |ω|=0.198
  n_rot<2 42                   |ω|=0.129
  narrow-basin 44 (s011 class) |ω|=1.445

Total LMs = 10 × 2 × (9+9+14+7) = 780. Pool(8) wall ~90 min @ ~7s/LM.
Already-probed seeds (23, 28, 89) NOT re-run; combined with s040 in the
cohort summary.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DOMAIN_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))

from s040_basin_radius_3seed import (
    build_seed_payload,
    rotvec_quat_wxyz,
    quat_mul,
    perp_unit,
    _worker_init,
    _polish_one,
)

OUT_DIR = SURVEY_DIR / "results" / "s042_basin_radius_cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [79, 57, 62, 6, 14, 19, 84, 16, 42, 44]

Q0_DEG_GRID    = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0]
OMEGA_MAG_PCT  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]   # used with both signs
OMEGA_DIR_DEG  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def build_jobs_for_attractor(seed, payload, attractor):
    """Same shape as s040 but with the EXTENDED Q0_DEG_GRID."""
    if attractor == "truth":
        q0_attr = payload["q0_truth"]
        omega_attr = payload["omega_truth"]
    else:
        q0_attr = payload["q0_twin"]
        omega_attr = payload["omega_twin"]

    om_unit = omega_attr / np.linalg.norm(omega_attr)
    perp_omega = perp_unit(omega_attr)

    jobs = []

    for deg in Q0_DEG_GRID:
        rad = np.radians(deg)
        # along ω
        q_pert = rotvec_quat_wxyz(rad * om_unit)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_along_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })
        # perp ω
        q_pert = rotvec_quat_wxyz(rad * perp_omega)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_perp_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })

    for sign in (+1.0, -1.0):
        for pct in OMEGA_MAG_PCT:
            mag = sign * pct
            omega_init = omega_attr * (1.0 + mag / 100.0)
            jobs.append({
                "seed": seed, "attractor": attractor,
                "axis": "omega_mag_pct", "mag": mag,
                "q0_init": q0_attr.copy(), "omega_init": omega_init,
                "q0_attr": q0_attr, "omega_attr": omega_attr,
            })

    for deg in OMEGA_DIR_DEG:
        rad = np.radians(deg)
        omega_init = Rotation.from_rotvec(rad * perp_omega).apply(omega_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "omega_dir_deg", "mag": deg,
            "q0_init": q0_attr.copy(), "omega_init": omega_init,
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })

    return jobs


def basin_radius_per_axis(results, axis):
    sub = [r for r in results if r["axis"] == axis]
    if axis == "omega_mag_pct":
        radii = {}
        for sign_label, predicate in [("pos", lambda r: r["mag"] > 0),
                                       ("neg", lambda r: r["mag"] < 0)]:
            sub_signed = [r for r in sub if predicate(r)]
            band_a = [r for r in sub_signed if r["band"] == "A"]
            radii[f"radius_{sign_label}_pct"] = (
                max(abs(r["mag"]) for r in band_a) if band_a else None)
        return radii
    band_a = [r for r in sub if r["band"] == "A"]
    return {"radius": max(r["mag"] for r in band_a) if band_a else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    print(f"=== s042 basin cohort | seeds={SEEDS} | out={OUT_DIR} ===", flush=True)
    t_start = time.time()

    payloads = {}
    for seed in SEEDS:
        payloads[seed] = build_seed_payload(seed)
        p = payloads[seed]
        om_dps = float(np.linalg.norm(p["omega_truth"])) * 180.0 / np.pi
        print(f"[s042] seed {seed}: |ω|={om_dps:.3f} dps "
              f"({len(p['obs_times'])} epochs, {p['valid_mask'].sum()} valid)",
              flush=True)

    jobs = []
    for seed in SEEDS:
        for attractor in ("truth", "twin"):
            jobs.extend(build_jobs_for_attractor(seed, payloads[seed], attractor))
    n_per_pair = 2 * len(Q0_DEG_GRID) + 2 * len(OMEGA_MAG_PCT) + len(OMEGA_DIR_DEG)
    expected = len(SEEDS) * 2 * n_per_pair
    print(f"[s042] total jobs = {len(jobs)} (expected {len(SEEDS)}×2×{n_per_pair} = {expected})",
          flush=True)
    assert len(jobs) == expected

    print(f"[s042] launching Pool({args.n_workers}) ...", flush=True)
    if args.n_workers <= 1:
        _worker_init(payloads)
        results = [_polish_one(j) for j in jobs]
    else:
        from multiprocessing import get_context
        ctx = get_context("fork")
        with ctx.Pool(args.n_workers,
                      initializer=_worker_init,
                      initargs=(payloads,)) as pool:
            results = pool.map(_polish_one, jobs)
    wall = time.time() - t_start
    print(f"[s042] LM batch wall: {wall/60:.1f} min", flush=True)

    summary = {
        "seeds": SEEDS,
        "n_jobs": len(jobs),
        "wall_s": wall,
        "lm_max_nfev": 200,
        "q0_grid": Q0_DEG_GRID,
        "omega_mag_pct_grid": OMEGA_MAG_PCT,
        "omega_dir_deg_grid": OMEGA_DIR_DEG,
        "by_seed": {},
    }

    for seed in SEEDS:
        seed_dir = OUT_DIR / f"seed{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_summary = {}
        seed_summary["omega_mag_dps"] = float(
            np.linalg.norm(payloads[seed]["omega_truth"])) * 180.0 / np.pi
        for attractor in ("truth", "twin"):
            sub = [r for r in results
                   if r["seed"] == seed and r["attractor"] == attractor]
            radii = {}
            for axis in ("q0_along_omega", "q0_perp_omega",
                         "omega_mag_pct", "omega_dir_deg"):
                radii[axis] = basin_radius_per_axis(sub, axis)
            payload_save = {
                "seed": seed,
                "attractor": attractor,
                "n_results": len(sub),
                "basin_radii": radii,
                "results": sub,
            }
            with open(seed_dir / f"{attractor}_basin.json", "w") as f:
                json.dump(payload_save, f, indent=2)
            seed_summary[attractor] = {
                "n_results": len(sub),
                "n_band_a": sum(1 for r in sub if r["band"] == "A"),
                "n_band_b": sum(1 for r in sub if r["band"] == "B"),
                "n_band_c": sum(1 for r in sub if r["band"] == "C"),
                "n_band_d": sum(1 for r in sub if r["band"] == "D"),
                "basin_radii": radii,
            }
        summary["by_seed"][f"seed{seed:03d}"] = seed_summary
        print(f"\n[s042] seed {seed} (|ω|={seed_summary['omega_mag_dps']:.3f} dps):",
              flush=True)
        for attractor in ("truth", "twin"):
            ss = seed_summary[attractor]
            print(f"   {attractor:5s}  Band A/B/C/D = "
                  f"{ss['n_band_a']}/{ss['n_band_b']}/{ss['n_band_c']}/{ss['n_band_d']}",
                  flush=True)
            for axis, val in ss["basin_radii"].items():
                print(f"     {axis}: {val}", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[s042] Saved: {OUT_DIR / 'summary.json'}", flush=True)
    print(f"[s042] Total wall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
