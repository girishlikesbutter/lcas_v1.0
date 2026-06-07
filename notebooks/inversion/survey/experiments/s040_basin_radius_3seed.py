"""s040 — joint-LM basin radius probe on seeds {23, 28, 89} × {truth, twin}.

For each seed × attractor we perturb the initial state in 4 independent axes
and at multiple magnitudes, then run scipy LM polish (`least_squares`,
`method='lm'`, max_nfev=200, xtol=ftol=1e-8) — the same recipe as s034/s038.
We record ρ_final + q0_err + ω-dir + ω-mag to the SAME attractor we perturbed
from. Per axis, the basin radius is the largest perturbation that still
lands Band A (ρ<2).

Axes:
  axis_q0_along_omega  : rotvec parallel to attractor ω-axis
                         {2, 5, 10, 15, 20, 25, 30}°
  axis_q0_perp_omega   : rotvec along an arbitrary perpendicular axis
                         {2, 5, 10, 15, 20, 25, 30}°
  axis_omega_mag       : ω → ω·(1+δ), δ in {±0.5, ±1, ±2, ±3, ±5, ±7, ±10}%
  axis_omega_dir       : rotate ω about a perpendicular axis
                         {0.5, 1, 2, 3, 5, 7, 10}°

Total = 3 seeds × 2 attractors × (7+7+14+7) = 210 LMs. ~13 min Pool(8) at
~30 s/LM. Kill at 45 min.

Outputs:
  results/s040_basin_radius_3seed/seed{XXX}/{truth,twin}_basin.json
  results/s040_basin_radius_3seed/summary.json
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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry

OUT_DIR = SURVEY_DIR / "results" / "s040_basin_radius_3seed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [23, 28, 89]
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
LM_MAX_NFEV = 200

Q0_DEG_GRID    = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
OMEGA_MAG_PCT  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]   # used with both signs
OMEGA_DIR_DEG  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


# --- quaternion helpers (identical to s034/s038) ----------------------------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_R_i2b_batch(q_arr_wxyz):
    qxyzw = q_arr_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def rotvec_quat_wxyz(rotvec):
    q_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def perp_unit(v):
    """An arbitrary unit vector perpendicular to v."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    assert n > 0
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v / n, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    p = np.cross(v, candidate)
    return p / np.linalg.norm(p)


# --- residual (joint 6-DOF surrogate-MSE; matches s034/s038) ----------------
def build_residual(q0_init, omega_init, obs_times, inertia,
                   sun_unit, obs_unit, obs_dist, mag_truth, valid_mask, surrogate):
    mag_truth_valid = mag_truth[valid_mask]

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert_wxyz = rotvec_quat_wxyz(rotvec)
        q0_new = quat_mul(q_pert_wxyz, q0_init)
        omega_new = omega_init + omega_delta
        q_traj, _ = propagate_attitude(
            q0_new, omega_new, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        R_full = quat_to_R_i2b_batch(q_traj)
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)
        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid_mask] - mag_truth_valid).astype(np.float64)

    return residual


# --- worker plumbing --------------------------------------------------------
_WORKERS = {}    # seed -> static payload


def _worker_init(payloads_by_seed):
    global _WORKERS
    _WORKERS = payloads_by_seed
    from lib.surrogate_eval import get_model
    _WORKERS["__surrogate__"] = get_model()


def _polish_one(job):
    """job = dict(seed, attractor, axis, mag, q0_init, omega_init,
                  q0_attr, omega_attr)
    """
    s = _WORKERS[job["seed"]]
    surrogate = _WORKERS["__surrogate__"]

    q0_init = np.asarray(job["q0_init"], dtype=np.float64)
    q0_init = q0_init / np.linalg.norm(q0_init)
    omega_init = np.asarray(job["omega_init"], dtype=np.float64)

    residual = build_residual(
        q0_init, omega_init,
        s["obs_times"], s["inertia"],
        s["sun_unit"], s["obs_unit"], s["obs_dist"],
        s["mag_truth"], s["valid_mask"], surrogate,
    )

    x0 = np.zeros(6)
    t0 = time.time()
    converged = True
    try:
        result = least_squares(
            residual, x0, method='lm',
            max_nfev=LM_MAX_NFEV, xtol=1e-8, ftol=1e-8,
        )
        n_iter = int(result.nfev)
        mse_final = float(np.mean(result.fun ** 2))
        rotvec = result.x[:3]
        omega_delta = result.x[3:]
        q_pert_wxyz = rotvec_quat_wxyz(rotvec)
        q0_final = quat_mul(q_pert_wxyz, q0_init)
        omega_final = omega_init + omega_delta
    except Exception as e:
        converged = False
        n_iter = -1
        mse_final = float("inf")
        q0_final = q0_init.copy()
        omega_final = omega_init.copy()
    dt = time.time() - t0

    # errors back to the attractor we perturbed AROUND
    q0_attr = np.asarray(job["q0_attr"], dtype=np.float64)
    omega_attr = np.asarray(job["omega_attr"], dtype=np.float64)
    om_attr_norm = float(np.linalg.norm(omega_attr))
    om_final_norm = float(np.linalg.norm(omega_final))

    q0_err = angular_dist_deg(q0_final, q0_attr)
    omega_dir_err = float(np.degrees(np.arccos(np.clip(
        np.dot(omega_final, omega_attr) / (om_final_norm * om_attr_norm),
        -1.0, 1.0))))
    omega_mag_pct = float((om_final_norm - om_attr_norm) / om_attr_norm * 100.0)

    rho_final = float(np.sqrt(max(mse_final, 0.0)) / 0.05)
    band = (
        "A" if rho_final < 2 else
        "B" if rho_final < 4 else
        "C" if rho_final < 8 else
        "D"
    )

    return {
        "seed": int(job["seed"]),
        "attractor": str(job["attractor"]),
        "axis": str(job["axis"]),
        "mag": float(job["mag"]),
        "q0_init": q0_init.tolist(),
        "omega_init": omega_init.tolist(),
        "q0_final": q0_final.tolist(),
        "omega_final": omega_final.tolist(),
        "rho_final": rho_final,
        "band": band,
        "q0_err_deg": float(q0_err),
        "omega_dir_err_deg": float(omega_dir_err),
        "omega_mag_pct": float(omega_mag_pct),
        "n_iter": int(n_iter),
        "wall_s": float(dt),
        "converged": bool(converged),
    }


# --- per-seed static payload ------------------------------------------------
def build_seed_payload(seed):
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    truth = np.load(traj_path)
    obs_times = truth["observation_times"].astype(np.float64)
    sun_pos = truth["sun_pos"].astype(np.float64)
    obs_pos = truth["obs_pos"].astype(np.float64)
    sat_pos = truth["sat_pos"].astype(np.float64)
    obs_dist = truth["obs_dist"].astype(np.float64)
    mag_truth = truth["mag_hifi"].astype(np.float64)
    valid_mask = np.isfinite(mag_truth)

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)).astype(np.float64)
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)).astype(np.float64)

    q0_truth = truth["q0_wxyz"].astype(np.float64)
    omega_truth = truth["omega0_rad"].astype(np.float64)

    # body-twin: q_180x ⊗ q0_truth, R_180x @ ω_truth
    q_180x_quat = np.array([0.0, 1.0, 0.0, 0.0])
    R_180x = np.diag([1.0, -1.0, -1.0])
    q0_twin = quat_mul(q_180x_quat, q0_truth)
    omega_twin = R_180x @ omega_truth

    inertia = load_static_geometry()["inertia_tensor"].astype(np.float64)

    return {
        "obs_times": obs_times,
        "sun_unit": sun_unit,
        "obs_unit": obs_unit,
        "obs_dist": obs_dist,
        "mag_truth": mag_truth,
        "valid_mask": valid_mask,
        "inertia": inertia,
        "q0_truth": q0_truth,
        "omega_truth": omega_truth,
        "q0_twin": q0_twin,
        "omega_twin": omega_twin,
    }


# --- job builder ------------------------------------------------------------
def build_jobs_for_attractor(seed, payload, attractor):
    """attractor in {'truth','twin'}.  Returns list of job dicts."""
    if attractor == "truth":
        q0_attr = payload["q0_truth"]
        omega_attr = payload["omega_truth"]
    else:
        q0_attr = payload["q0_twin"]
        omega_attr = payload["omega_twin"]

    om_norm = np.linalg.norm(omega_attr)
    om_unit = omega_attr / om_norm
    perp_omega = perp_unit(omega_attr)   # one arbitrary perpendicular axis

    jobs = []

    # axis 1: q0 along ω axis
    for deg in Q0_DEG_GRID:
        rad = np.radians(deg)
        rotvec = rad * om_unit
        q_pert = rotvec_quat_wxyz(rotvec)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_along_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })

    # axis 2: q0 along ω-perpendicular axis
    for deg in Q0_DEG_GRID:
        rad = np.radians(deg)
        rotvec = rad * perp_omega
        q_pert = rotvec_quat_wxyz(rotvec)
        q0_init = quat_mul(q_pert, q0_attr)
        jobs.append({
            "seed": seed, "attractor": attractor,
            "axis": "q0_perp_omega", "mag": deg,
            "q0_init": q0_init, "omega_init": omega_attr.copy(),
            "q0_attr": q0_attr, "omega_attr": omega_attr,
        })

    # axis 3: ω-mag (signed)
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

    # axis 4: ω-direction (rotate ω about a perpendicular axis)
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


# --- per-axis basin-radius extraction ---------------------------------------
def basin_radius_per_axis(results, axis):
    """Largest |mag| at which Band A. Splits omega_mag_pct by sign.
    Returns dict: {"radius_neg":..., "radius_pos":...} for signed axes,
    {"radius":...} for unsigned. None means even smallest mag failed.
    """
    sub = [r for r in results if r["axis"] == axis]
    if axis == "omega_mag_pct":
        radii = {}
        for sign_label, predicate in [("pos", lambda r: r["mag"] > 0),
                                       ("neg", lambda r: r["mag"] < 0)]:
            sub_signed = [r for r in sub if predicate(r)]
            band_a = [r for r in sub_signed if r["band"] == "A"]
            if band_a:
                radii[f"radius_{sign_label}_pct"] = max(abs(r["mag"]) for r in band_a)
            else:
                radii[f"radius_{sign_label}_pct"] = None
        return radii
    else:
        band_a = [r for r in sub if r["band"] == "A"]
        if band_a:
            return {"radius": max(r["mag"] for r in band_a)}
        return {"radius": None}


# --- main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="One LM at smallest perturbation on seed 23 truth.")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    print(f"=== s040 basin radius probe | seeds={SEEDS} | out={OUT_DIR} ===",
          flush=True)
    t_start = time.time()

    # Build per-seed payloads
    payloads = {}
    for seed in SEEDS:
        payloads[seed] = build_seed_payload(seed)
        p = payloads[seed]
        print(f"[s040] seed {seed}: |ω_truth|={np.linalg.norm(p['omega_truth']):.6f} rad/s "
              f"({len(p['obs_times'])} epochs, {p['valid_mask'].sum()} valid)",
              flush=True)

    # Build jobs
    if args.smoke:
        seed = 23
        jobs = build_jobs_for_attractor(seed, payloads[seed], "truth")
        # smallest q0_along perturbation = 2°
        jobs = [j for j in jobs if j["axis"] == "q0_along_omega" and j["mag"] == 2.0]
        print(f"[s040-smoke] running {len(jobs)} job(s)", flush=True)
    else:
        jobs = []
        for seed in SEEDS:
            for attractor in ("truth", "twin"):
                jobs.extend(build_jobs_for_attractor(seed, payloads[seed], attractor))
        print(f"[s040] total jobs = {len(jobs)} "
              f"(expected 3×2×35 = 210)", flush=True)

    # Run
    print(f"[s040] launching Pool({args.n_workers}) ...", flush=True)
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
    print(f"[s040] LM batch wall: {wall/60:.1f} min", flush=True)

    if args.smoke:
        for r in results:
            print(f"  smoke: seed={r['seed']} attr={r['attractor']} axis={r['axis']} "
                  f"mag={r['mag']:.2f} → ρ={r['rho_final']:.3f} band={r['band']} "
                  f"q0_err={r['q0_err_deg']:.3f}° ω-dir={r['omega_dir_err_deg']:.3f}° "
                  f"ω-mag={r['omega_mag_pct']:+.4f}% nfev={r['n_iter']} wall={r['wall_s']:.1f}s",
                  flush=True)
        return

    # Per-seed-per-attractor save
    summary = {
        "seeds": SEEDS,
        "n_jobs": len(jobs),
        "wall_s": wall,
        "lm_max_nfev": LM_MAX_NFEV,
        "by_seed": {},
    }
    for seed in SEEDS:
        seed_dir = OUT_DIR / f"seed{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_summary = {}
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
        print(f"\n[s040] seed {seed}: ", flush=True)
        for attractor in ("truth", "twin"):
            ss = seed_summary[attractor]
            print(f"   {attractor:5s}  Band A/B/C/D = "
                  f"{ss['n_band_a']}/{ss['n_band_b']}/{ss['n_band_c']}/{ss['n_band_d']}",
                  flush=True)
            for axis, val in ss["basin_radii"].items():
                print(f"     {axis}: {val}", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[s040] Saved: {OUT_DIR / 'summary.json'}", flush=True)
    print(f"[s040] Total wall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
