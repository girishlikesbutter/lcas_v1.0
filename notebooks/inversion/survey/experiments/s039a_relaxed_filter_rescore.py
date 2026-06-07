"""s039a — relaxed-filter re-score on the s032 cohort.

For each seed, recategorise the cached candidates_meta.npz at relaxed geo
thresholds. Compute per-seed:
  - survivor counts at strict 1.0, half (>=0.5), low (>=1/n_spec), any positive
  - min q0_err to truth/twin among each survivor set
  - n surviving ω-cells (cells with at least one candidate above threshold)

Limitation: align_score is NaN for geo-fail candidates (s032 ran with
MEASURE_GEO_FAIL_ALIGN=False). So we can only relax geo here; relaxing
align would require surrogate re-eval (deferred to s039b).

Output: per-seed table, cohort summary.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

S032_DIR = SURVEY_DIR / "results" / "s032_cohort_fast"
TRAJ_DIR = SURVEY_DIR / "data" / "trajectories"
OUT_DIR = SURVEY_DIR / "results" / "s039a_relaxed_filter_rescore"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def angular_dist_batch(q_arr, q_ref):
    """Vectorised q0_err in degrees from each q in q_arr to q_ref."""
    dots = np.abs(q_arr @ q_ref)
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def rescore_seed(seed: int):
    seed_dir = S032_DIR / f"seed{seed:03d}"
    if not seed_dir.exists():
        return None
    summ_path = seed_dir / "summary.json"
    if not summ_path.exists():
        return None
    summary = json.loads(summ_path.read_text())
    n_spec = int(summary["scale"]["n_spec_events"])
    n_bright = int(summary["scale"]["n_bright_peaks"])
    n_classifiable = int(summary["scale"]["n_classifiable_peaks"])
    M_q = int(summary["scale"]["M_q_target"])
    N_dir = int(summary["scale"]["N_dir"])

    cands = np.load(seed_dir / "candidates_meta.npz", mmap_mode="r")
    geo_score = np.array(cands["geo_score"])
    align_score = np.array(cands["align_score"])
    omega_cell_idx = np.array(cands["omega_cell_idx"])
    q0_arr = np.array(cands["q0"]).astype(np.float64)

    truth = np.load(TRAJ_DIR / f"traj_seed{seed:03d}.npz")
    q0_truth = truth["q0_wxyz"].astype(np.float64)
    twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q0_truth)

    # Relaxation grid for geo
    if n_spec >= 1:
        relax_levels = {
            "strict_1.0":   1.0,
            "half_0.5":     0.5,
            "one_over_n":   1.0 / n_spec - 1e-6,   # 1/n_spec floor (≥1/n_spec)
            "any_positive": 0.0 + 1e-6,            # > 0
        }
    else:
        # No spec events — geo is NaN throughout. Skip.
        return {"seed": seed, "n_spec": 0, "skipped": "no_spec_events"}

    results = {}
    for label, thr in relax_levels.items():
        mask = np.isfinite(geo_score) & (geo_score >= thr)
        n_pass = int(mask.sum())
        if n_pass == 0:
            results[label] = {"n_survivors": 0, "n_cells": 0,
                              "min_q0_err_truth_deg": None,
                              "min_q0_err_twin_deg": None,
                              "min_q0_err_min_deg": None}
            continue
        q_sub = q0_arr[mask]
        # Vectorised: q dot q_truth (or twin) gives cos(geodesic/2); take abs
        e_t = angular_dist_batch(q_sub, q0_truth)
        e_w = angular_dist_batch(q_sub, twin_q0)
        e_min = np.minimum(e_t, e_w)

        cells_pass = np.unique(omega_cell_idx[mask])
        results[label] = {
            "n_survivors": n_pass,
            "n_cells": int(cells_pass.size),
            "min_q0_err_truth_deg": float(e_t.min()),
            "min_q0_err_twin_deg": float(e_w.min()),
            "min_q0_err_min_deg": float(e_min.min()),
            "n_in_basin_30deg": int((e_min < 30).sum()),
            "n_in_basin_15deg": int((e_min < 15).sum()),
            "n_in_basin_5deg": int((e_min < 5).sum()),
        }

    return {
        "seed": seed,
        "n_spec": n_spec,
        "n_bright": n_bright,
        "n_classifiable": n_classifiable,
        "nearest_cell_pct": float(summary["bracket"]["nearest_cell_pct"]),
        "strict_survivors_baseline": int(summary["categorisation"]["passed_both"]),
        "relaxed": results,
    }


def main():
    print(f"=== s039a relaxed filter re-score | s032 cohort | out={OUT_DIR} ===",
          flush=True)
    seed_dirs = sorted(S032_DIR.glob("seed*/"))
    seeds = [int(d.name[4:]) for d in seed_dirs]
    print(f"  seeds available: {len(seeds)}", flush=True)

    cohort = []
    t0 = time.time()
    for seed in seeds:
        t_s = time.time()
        r = rescore_seed(seed)
        if r is None:
            continue
        if r.get("skipped"):
            print(f"  seed {seed}: SKIPPED ({r['skipped']})", flush=True)
            cohort.append(r)
            continue
        relaxed = r["relaxed"]
        n_strict = relaxed["strict_1.0"]["n_survivors"]
        n_low = relaxed["one_over_n"]["n_survivors"]
        min_q_low = relaxed["one_over_n"]["min_q0_err_min_deg"]
        n_basin_low = relaxed["one_over_n"]["n_in_basin_30deg"]
        print(f"  seed {seed:>2}: n_spec={r['n_spec']:>2}  "
              f"strict_1.0={n_strict:>5}  "
              f"one_over_n={n_low:>5} (min_q={min_q_low:>6.2f}° basin30={n_basin_low})  "
              f"[{time.time()-t_s:.2f}s]",
              flush=True)
        cohort.append(r)

    print(f"\nCohort total wall: {time.time()-t0:.1f} s ({len(cohort)} seeds)", flush=True)

    # Save per-seed table + cohort summary
    out_json = OUT_DIR / "cohort_relaxed_rescore.json"
    with open(out_json, "w") as f:
        json.dump({"cohort": cohort}, f, indent=2)
    print(f"Saved: {out_json}")

    # Cohort headline numbers
    valid = [c for c in cohort if not c.get("skipped")]
    print(f"\n=== Cohort summary (n={len(valid)} valid seeds) ===")
    for label in ["strict_1.0", "half_0.5", "one_over_n", "any_positive"]:
        n_with = sum(1 for c in valid if c["relaxed"][label]["n_survivors"] > 0)
        n_basin30 = sum(1 for c in valid
                        if c["relaxed"][label]["n_survivors"] > 0
                        and c["relaxed"][label].get("n_in_basin_30deg", 0) > 0)
        n_basin15 = sum(1 for c in valid
                        if c["relaxed"][label]["n_survivors"] > 0
                        and c["relaxed"][label].get("n_in_basin_15deg", 0) > 0)
        n_basin5 = sum(1 for c in valid
                       if c["relaxed"][label]["n_survivors"] > 0
                       and c["relaxed"][label].get("n_in_basin_5deg", 0) > 0)
        print(f"  {label:>14}: {n_with:>3}/{len(valid)} seeds with survivors,  "
              f"basin30°={n_basin30:>3}, basin15°={n_basin15:>3}, basin5°={n_basin5:>3}")


if __name__ == "__main__":
    main()
