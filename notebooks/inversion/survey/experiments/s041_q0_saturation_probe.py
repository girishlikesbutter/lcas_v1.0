"""s041 — q0-saturation probe on seed 23 (truth + twin), q0 axes only.

s040 saw q0 basins saturate at the 30° grid limit on every seed/attractor
except seed 89-truth-perp. This probe pushes the q0 axes to {45°, 60°, 90°,
120°, 150°, 175°} on seed 23 (the seed where 30° was clearly inside the basin)
to find the true edge or confirm it's effectively global.

Reuses s040's job-builder structure but only the q0 axes. ω is held at the
attractor (no ω perturbation), so this isolates the q0 basin width.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DOMAIN_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

# Reuse s040's helpers + worker
sys.path.insert(0, str(SURVEY_DIR / "experiments"))
from s040_basin_radius_3seed import (
    build_seed_payload,
    build_jobs_for_attractor,
    rotvec_quat_wxyz,
    quat_mul,
    perp_unit,
    _worker_init,
    _polish_one,
)

OUT_DIR = SURVEY_DIR / "results" / "s041_q0_saturation_seed023"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 23
Q0_DEG_GRID_EXT = [45.0, 60.0, 90.0, 120.0, 150.0, 175.0]


def build_q0_only_jobs(seed, payload, attractor):
    """Like s040.build_jobs_for_attractor but ONLY q0 axes at extended grid."""
    if attractor == "truth":
        q0_attr = payload["q0_truth"]
        omega_attr = payload["omega_truth"]
    else:
        q0_attr = payload["q0_twin"]
        omega_attr = payload["omega_twin"]

    om_unit = omega_attr / np.linalg.norm(omega_attr)
    perp_omega = perp_unit(omega_attr)

    jobs = []
    for deg in Q0_DEG_GRID_EXT:
        rad = np.radians(deg)
        # q0_along_omega
        q_pert = rotvec_quat_wxyz(rad * om_unit)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_along_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })
        # q0_perp_omega
        q_pert = rotvec_quat_wxyz(rad * perp_omega)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_perp_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })
    return jobs


def main():
    print(f"=== s041 q0 saturation | seed={SEED} | out={OUT_DIR} ===", flush=True)
    t_start = time.time()

    payloads = {SEED: build_seed_payload(SEED)}
    jobs = []
    for attr in ("truth", "twin"):
        jobs.extend(build_q0_only_jobs(SEED, payloads[SEED], attr))
    print(f"[s041] {len(jobs)} jobs (2 attractors × 2 axes × 6 mags)", flush=True)

    from multiprocessing import get_context
    ctx = get_context("fork")
    n_workers = 8
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(payloads,)) as pool:
        results = pool.map(_polish_one, jobs)
    wall = time.time() - t_start
    print(f"[s041] wall: {wall/60:.1f} min", flush=True)

    # Pretty print
    for attr in ("truth", "twin"):
        print(f"\n--- seed {SEED} {attr} ---", flush=True)
        for axis in ("q0_along_omega", "q0_perp_omega"):
            print(f"  {axis}:", flush=True)
            sub = sorted([r for r in results
                          if r["attractor"] == attr and r["axis"] == axis],
                         key=lambda r: r["mag"])
            for r in sub:
                rho = r["rho_final"]
                rho_str = f"{rho:.2f}" if rho < 1e6 else "inf"
                print(f"    mag={r['mag']:6.1f}°  ρ={rho_str:>8s}  band={r['band']}  "
                      f"q0_err={r['q0_err_deg']:6.2f}°  "
                      f"ω-dir={r['omega_dir_err_deg']:5.2f}°  "
                      f"ω-mag={r['omega_mag_pct']:+8.4f}%  nfev={r['n_iter']}",
                      flush=True)

    summary = {
        "seed": SEED,
        "wall_s": wall,
        "extended_q0_grid_deg": Q0_DEG_GRID_EXT,
        "results": results,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[s041] Saved: {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
