"""s016 — ω-mag-frozen 5-DOF polish from harvested s015 ICs (Option C).

Tests the hypothesis that ω-magnitude-correct ICs (within 10% of truth)
are close enough to the truth-ω tube for a 5-DOF LM (q0 + ω-direction
only) to bridge the remaining gap to the truth basin.

Harvested ICs: 41 across 7/8 seeds (seed 42 = 0 harvest, excluded). Per-
seed counts: 6=1, 13=1, 28=9, 41=7, 49=12, 79=1, 91=10. Source:
results/s015_diagnostic_n64/runs.npz, mask = |ω_mag_err_pct| < 10.

Crucial framing: every harvested IC has q0_err ∈ [29°, 180°]. The
hypothesis being tested is NOT "ω-mag-correct ICs are near truth" — it
is "right-ω-mag is sufficient prior to bridge the q0 + ω-dir search via
5-DOF LM."

5-DOF parameterization:
  x = (δθ_3, theta, phi)
  q0 = quat_from_rotvec(δθ) · q0_seed
  ω  = ω_mag_fixed · (sin(theta)·cos(phi), sin(theta)·sin(phi), cos(theta))

ω-magnitude is frozen at the IC's harvested ω-mag value (NOT truth).
Within ±10% of truth by construction.

Polish: scipy least_squares(method='lm', max_nfev=200), residuals =
surrogate full-LC residuals.

Pool(8), BLAS=1 + torch_threads=1.

Decision criteria:
  Per-seed in-basin (loose: q0<10°, ω_dir<2°, |ω_mag|<10%):
    ≥1/seed = bridge succeeded for that seed.
  Cohort: ≥4/7 seeds in-basin → s016-C viable, propose two-stage pipeline.
  Cohort: <4/7 → s016-C falsifies hypothesis; ω-mag prior insufficient,
    move to S016-B (LC-feature ω-mag bias) or S016-A (ω-grid).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
S015_RUNS = SURVEY_DIR / "results" / "s015_diagnostic_n64" / "runs.npz"
OUT_DIR = SURVEY_DIR / "results" / "s016c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_MAG_HARVEST_PCT = 10.0
MAX_NFEV = 200
N_WORKERS = 8

Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

STRICT_Q0 = 5.0
STRICT_OD = 1.0
STRICT_OM_PCT = 5.0
LOOSE_Q0 = 10.0
LOOSE_OD = 2.0
LOOSE_OM_PCT = 10.0


# ── quaternion utilities ─────────────────────────────────────────────────


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_rotvec_wxyz(rotvec):
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0 - 0.125 * angle * angle,
                         0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), s * rotvec[0], s * rotvec[1], s * rotvec[2]])


def vec_to_spherical(v):
    """Cartesian unit vector → (theta in [0, π], phi in [-π, π])."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return 0.0, 0.0
    vx, vy, vz = v / n
    theta = float(np.arccos(np.clip(vz, -1.0, 1.0)))
    phi = float(np.arctan2(vy, vx))
    return theta, phi


def spherical_to_vec(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


# ── worker setup ─────────────────────────────────────────────────────────

_W_TIMES_BY_SEED = {}
_W_SUN_BY_SEED = {}
_W_OBS_BY_SEED = {}
_W_SAT_BY_SEED = {}
_W_OBS_DIST_BY_SEED = {}
_W_MAG_HIFI_BY_SEED = {}
_W_INERTIA_TENSOR = None


def init_worker(seed_data, inertia_tensor):
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    global _W_INERTIA_TENSOR
    _W_INERTIA_TENSOR = inertia_tensor
    for s, d in seed_data.items():
        _W_TIMES_BY_SEED[s] = d["observation_times"]
        _W_SUN_BY_SEED[s] = d["sun_pos"]
        _W_OBS_BY_SEED[s] = d["obs_pos"]
        _W_SAT_BY_SEED[s] = d["sat_pos"]
        _W_OBS_DIST_BY_SEED[s] = d["obs_dist"]
        _W_MAG_HIFI_BY_SEED[s] = d["mag_hifi"]
    surrogate_eval.get_model()


def make_residual_fn(seed, q0_seed_wxyz, omega_mag_fixed):
    times = _W_TIMES_BY_SEED[seed]
    sun = _W_SUN_BY_SEED[seed]
    obs = _W_OBS_BY_SEED[seed]
    sat = _W_SAT_BY_SEED[seed]
    obs_dist = _W_OBS_DIST_BY_SEED[seed]
    mag_truth = _W_MAG_HIFI_BY_SEED[seed]
    inertia = _W_INERTIA_TENSOR
    finite_truth = np.isfinite(mag_truth)

    def residuals(x):
        delta_theta = x[:3]
        theta, phi = float(x[3]), float(x[4])
        delta_q = quat_from_rotvec_wxyz(delta_theta)
        q0 = quat_multiply_wxyz(delta_q, q0_seed_wxyz)
        q0 = q0 / np.linalg.norm(q0)
        omega = omega_mag_fixed * spherical_to_vec(theta, phi)
        try:
            k1b, k2b, _ = propagate_to_body_frame(
                q0, omega, times, sun, obs, sat, inertia,
            )
            pred = surrogate_eval.predict(k1b, k2b, obs_dist)
        except Exception:
            return np.full_like(mag_truth, 1e3)
        r = pred - mag_truth
        bad = ~(finite_truth & np.isfinite(r))
        r = np.where(bad, 0.0, r)
        return r

    return residuals


def run_lm_for_ic(args):
    ic_dict, max_nfev = args
    seed = int(ic_dict["seed"])
    s015_ic_idx = int(ic_dict["s015_ic_idx"])
    s016_ic_idx = int(ic_dict["s016_ic_idx"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    omega_seed_rad = np.asarray(ic_dict["omega_seed_rad"], dtype=float)
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    omega_mag_fixed = float(np.linalg.norm(omega_seed_rad))
    omega_mag_fixed_err_pct = (
        100.0 * (omega_mag_fixed - omega_truth_mag) / omega_truth_mag
    )

    theta0, phi0 = vec_to_spherical(omega_seed_rad)

    residual_fn = make_residual_fn(seed, q0_seed_wxyz, omega_mag_fixed)
    x0 = np.array([0.0, 0.0, 0.0, theta0, phi0])

    initial_residual = residual_fn(x0)
    initial_mse = float(np.mean(initial_residual ** 2))
    initial_q0_err = quat_geodesic_deg(q0_seed_wxyz, q0_truth)

    t0 = time.time()
    try:
        result = least_squares(
            residual_fn, x0, method="lm",
            max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
        success = bool(result.success)
        status = int(result.status)
        n_fev = int(result.nfev)
        x_final = result.x.copy()
    except Exception:
        success = False
        status = -99
        n_fev = 0
        x_final = x0.copy()
    wall = time.time() - t0

    delta_theta_final = x_final[:3]
    theta_f, phi_f = float(x_final[3]), float(x_final[4])
    delta_q_final = quat_from_rotvec_wxyz(delta_theta_final)
    q0_final = quat_multiply_wxyz(delta_q_final, q0_seed_wxyz)
    q0_final = q0_final / np.linalg.norm(q0_final)

    omega_dir_final = spherical_to_vec(theta_f, phi_f)
    omega_final = omega_mag_fixed * omega_dir_final  # mag stays at fixed

    final_residual = residual_fn(x_final)
    final_mse = float(np.mean(final_residual ** 2))

    q0_err = quat_geodesic_deg(q0_final, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    twin_err = quat_geodesic_deg(q0_final, q_twin)
    omega_dir_err = float(np.degrees(np.arccos(
        float(np.clip(np.dot(omega_dir_final, omega_truth_dir), -1.0, 1.0))
    )))
    # ω-mag err remains at IC's harvested value (frozen) — record for clarity.
    omega_mag_err_pct = omega_mag_fixed_err_pct

    truth_basin_strict = (q0_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)
    truth_basin_loose = (q0_err < LOOSE_Q0) and (omega_dir_err < LOOSE_OD) \
        and (abs(omega_mag_err_pct) < LOOSE_OM_PCT)
    twin_basin_strict = (twin_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)

    return {
        "seed": seed,
        "s015_ic_idx": s015_ic_idx,
        "s016_ic_idx": s016_ic_idx,
        "omega_mag_fixed_rad": omega_mag_fixed,
        "omega_mag_fixed_err_pct": omega_mag_fixed_err_pct,
        "initial_q0_err_deg": float(initial_q0_err),
        "initial_mse": initial_mse,
        "q0_seed_wxyz": q0_seed_wxyz,
        "omega_seed_rad": omega_seed_rad,
        "q0_final_wxyz": q0_final,
        "omega_final_rad": omega_final,
        "q0_err_deg": float(q0_err),
        "twin_err_deg": float(twin_err),
        "omega_dir_err_deg": float(omega_dir_err),
        "omega_mag_err_pct": float(omega_mag_err_pct),
        "final_mse": final_mse,
        "truth_basin_strict": bool(truth_basin_strict),
        "truth_basin_loose": bool(truth_basin_loose),
        "twin_basin_strict": bool(twin_basin_strict),
        "n_fev": n_fev,
        "wall_s": float(wall),
        "success": bool(success),
        "status": int(status),
    }


# ── harvest from s015 ──────────────────────────────────────────────────────


def harvest_ics():
    r = np.load(str(S015_RUNS))
    ics = []
    s016_idx = 0
    for s in sorted(set(r["seed"].tolist())):
        mask_seed = r["seed"] == s
        mask_omag = np.abs(r["omega_mag_err_pct"]) < OMEGA_MAG_HARVEST_PCT
        mask = mask_seed & mask_omag
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        omega_truth_mag = float(np.linalg.norm(omega_truth))
        omega_truth_dir = omega_truth / omega_truth_mag
        for s015_i in idxs:
            ics.append({
                "seed": int(s),
                "s015_ic_idx": int(r["ic_idx"][s015_i]),
                "s016_ic_idx": s016_idx,
                "q0_seed_wxyz": r["q0_final_wxyz"][s015_i].copy(),
                "omega_seed_rad": r["omega_final_rad"][s015_i].copy(),
                "q0_truth_wxyz": q0_truth,
                "omega_truth_dir": omega_truth_dir,
                "omega_truth_mag": omega_truth_mag,
            })
            s016_idx += 1
    return ics


def collect_seed_data(seeds):
    seed_data = {}
    for s in seeds:
        d = traj_load.load_truth(s)
        seed_data[s] = {
            "observation_times": d["observation_times"],
            "sun_pos": d["sun_pos"],
            "obs_pos": d["obs_pos"],
            "sat_pos": d["sat_pos"],
            "obs_dist": d["obs_dist"],
            "mag_hifi": d["mag_hifi"],
        }
    return seed_data


# ── summary + decision ─────────────────────────────────────────────────────


def summarise(results, all_seeds_planned):
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)

    per_seed = {}
    for s in all_seeds_planned:
        runs = by_seed.get(s, [])
        if not runs:
            per_seed[f"seed_{s:03d}"] = {
                "seed": s,
                "n_starts": 0,
                "status": "no_harvest",
                "note": "no s015 IC landed |ω_mag_err|<10% — excluded",
            }
            continue
        n_strict = sum(1 for r in runs if r["truth_basin_strict"])
        n_loose = sum(1 for r in runs if r["truth_basin_loose"])
        n_twin = sum(1 for r in runs if r["twin_basin_strict"])
        idx_min_mse = int(np.argmin([r["final_mse"] for r in runs]))
        argmin_run = runs[idx_min_mse]
        per_seed[f"seed_{s:03d}"] = {
            "seed": s,
            "n_starts": len(runs),
            "n_truth_basin_strict": n_strict,
            "n_truth_basin_loose": n_loose,
            "n_twin_basin_strict": n_twin,
            "min_q0_err_deg": float(min(r["q0_err_deg"] for r in runs)),
            "min_final_mse": float(min(r["final_mse"] for r in runs)),
            "min_omega_dir_err_deg": float(min(r["omega_dir_err_deg"] for r in runs)),
            "argmin_mse_q0_err_deg": float(argmin_run["q0_err_deg"]),
            "argmin_mse_omega_dir_err_deg": float(argmin_run["omega_dir_err_deg"]),
            "argmin_mse_omega_mag_err_pct": float(argmin_run["omega_mag_err_pct"]),
            "median_wall_s": float(np.median([r["wall_s"] for r in runs])),
        }

    seeds_with_starts = [s for s in all_seeds_planned
                         if per_seed[f"seed_{s:03d}"].get("n_starts", 0) > 0]
    n_loose = sum(1 for s in seeds_with_starts
                  if per_seed[f"seed_{s:03d}"]["n_truth_basin_loose"] >= 1)
    n_strict = sum(1 for s in seeds_with_starts
                   if per_seed[f"seed_{s:03d}"]["n_truth_basin_strict"] >= 1)

    n_tested = len(seeds_with_starts)
    if n_loose >= max(1, int(0.6 * n_tested)):
        decision = (f"S016-C VIABLE: {n_loose}/{n_tested} seeds with starts get "
                    f"≥1 in-basin (loose) landing. Two-stage pipeline confirmed.")
    elif n_loose >= 1:
        decision = (f"S016-C PARTIAL: {n_loose}/{n_tested} seeds with starts get "
                    f"≥1 in-basin (loose); insufficient cohort coverage for production. "
                    f"Move to S016-B (LC ω-mag bias).")
    else:
        decision = (f"S016-C REFUTED: 0/{n_tested} seeds with starts reach loose basin. "
                    f"ω-mag-correct is NOT a sufficient bridge. Move to S016-B "
                    f"(LC ω-mag bias) or S016-A (ω-grid).")

    return {
        "harvest_threshold_pct": OMEGA_MAG_HARVEST_PCT,
        "max_nfev": MAX_NFEV,
        "all_seeds_planned": all_seeds_planned,
        "seeds_with_starts": seeds_with_starts,
        "n_total_starts": len(results),
        "per_seed": per_seed,
        "n_seeds_with_loose_basin": n_loose,
        "n_seeds_with_strict_basin": n_strict,
        "decision": decision,
    }


def save_npz(results):
    arrs = {
        "seed": np.array([r["seed"] for r in results], dtype=int),
        "s015_ic_idx": np.array([r["s015_ic_idx"] for r in results], dtype=int),
        "s016_ic_idx": np.array([r["s016_ic_idx"] for r in results], dtype=int),
        "omega_mag_fixed_rad": np.array([r["omega_mag_fixed_rad"] for r in results]),
        "omega_mag_fixed_err_pct":
            np.array([r["omega_mag_fixed_err_pct"] for r in results]),
        "initial_q0_err_deg": np.array([r["initial_q0_err_deg"] for r in results]),
        "initial_mse": np.array([r["initial_mse"] for r in results]),
        "q0_seed_wxyz": np.stack([r["q0_seed_wxyz"] for r in results]),
        "omega_seed_rad": np.stack([r["omega_seed_rad"] for r in results]),
        "q0_final_wxyz": np.stack([r["q0_final_wxyz"] for r in results]),
        "omega_final_rad": np.stack([r["omega_final_rad"] for r in results]),
        "q0_err_deg": np.array([r["q0_err_deg"] for r in results]),
        "twin_err_deg": np.array([r["twin_err_deg"] for r in results]),
        "omega_dir_err_deg": np.array([r["omega_dir_err_deg"] for r in results]),
        "omega_mag_err_pct": np.array([r["omega_mag_err_pct"] for r in results]),
        "final_mse": np.array([r["final_mse"] for r in results]),
        "truth_basin_strict": np.array([r["truth_basin_strict"] for r in results]),
        "truth_basin_loose": np.array([r["truth_basin_loose"] for r in results]),
        "twin_basin_strict": np.array([r["twin_basin_strict"] for r in results]),
        "n_fev": np.array([r["n_fev"] for r in results], dtype=int),
        "wall_s": np.array([r["wall_s"] for r in results]),
        "success": np.array([r["success"] for r in results]),
        "status": np.array([r["status"] for r in results], dtype=int),
    }
    out = OUT_DIR / "runs.npz"
    np.savez(out, **arrs)
    print(f"Saved: {out}")


def save_summary(summary):
    out = OUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    all_seeds_planned = [6, 13, 28, 41, 42, 49, 79, 91]
    ics = harvest_ics()
    seeds_with_starts = sorted(set(ic["seed"] for ic in ics))
    seed_data = collect_seed_data(seeds_with_starts)

    print(f"s016c: {len(ics)} starts harvested across "
          f"{len(seeds_with_starts)}/8 seeds (|ω_mag_err|<{OMEGA_MAG_HARVEST_PCT}%).",
          flush=True)
    print(f"  seeds with starts: {seeds_with_starts}", flush=True)
    print(f"  Pool({N_WORKERS}), BLAS=1, max_nfev={MAX_NFEV}.", flush=True)
    print(f"  Out dir: {OUT_DIR}", flush=True)

    args_list = [(ic, MAX_NFEV) for ic in ics]
    t0 = time.time()
    results = []
    last_print = t0
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_lm_for_ic, args_list,
                                                  chunksize=1)):
            results.append(r)
            now = time.time()
            if now - last_print > 30.0 or i + 1 == len(ics):
                done = i + 1
                rate = done / (now - t0) if now > t0 else 0.0
                eta = (len(ics) - done) / rate if rate > 0 else 0.0
                print(f"  [{done:3d}/{len(ics)}] rate={rate:.2f}/s, eta={eta:.0f} s",
                      flush=True)
                last_print = now
    wall = time.time() - t0
    print(f"\nAll runs done. Total wall: {wall:.1f} s.\n", flush=True)

    save_npz(results)
    summary = summarise(results, all_seeds_planned)
    save_summary(summary)

    print("\n=== s016c per-seed yield ===")
    print(f"  {'seed':>5s} {'starts':>7s} {'in_loose':>9s} {'in_strict':>10s} "
          f"{'min_q0':>7s} {'min_od':>7s} {'min_mse':>10s}")
    for s in all_seeds_planned:
        cell = summary["per_seed"][f"seed_{s:03d}"]
        if cell.get("n_starts", 0) == 0:
            print(f"  {s:5d} {'0':>7s} {'-':>9s} {'-':>10s} "
                  f"{'-':>7s} {'-':>7s} {'-':>10s}    (no harvest)")
        else:
            print(f"  {s:5d} {cell['n_starts']:7d} "
                  f"{cell['n_truth_basin_loose']:9d} "
                  f"{cell['n_truth_basin_strict']:10d} "
                  f"{cell['min_q0_err_deg']:7.2f} "
                  f"{cell['min_omega_dir_err_deg']:7.2f} "
                  f"{cell['min_final_mse']:10.3e}")

    print(f"\n  loose-basin yield (cohort): "
          f"{summary['n_seeds_with_loose_basin']}/{len(summary['seeds_with_starts'])} "
          f"seeds with starts")
    print(f"  → DECISION: {summary['decision']}")
    print(f"\nTotal wall: {time.time() - t_main:.1f} s")


if __name__ == "__main__":
    main()
