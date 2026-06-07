"""s016-C' — fresh Sobol(q0 × ω-dir) at fixed harvested ω-mag.

s016 (first attempt at C) restarted LM from already-converged s015 ICs;
LM didn't move because the starts were already at local minima. C' tests
the actual hypothesis with FRESH starts.

For each seed, pick the top-3-by-s015-final_mse ICs with |ω_mag_err|<10%.
Use each picked IC's ω-mag as the FIXED ω-mag for a fresh search round.
Within that round: Sobol-Shoemake(q0) on SO(3) at N=64, paired stitched
with Sobol(ω-dir) on S² at N=64. LM polish at fixed ω-mag.

Per-seed harvest (from s015_diagnostic_n64):
  6: 1, 13: 1, 28: 3, 41: 3, 42: 0, 49: 3, 79: 1, 91: 3
  Total: 15 ω-mag values × 64 fresh ICs = 960 LM runs.

5-DOF parameterization (same as s016): x = (δθ_3, theta, phi).
LM: max_nfev=120 (fresh starts, give it room but bound stalls).
Pool(8), BLAS=1.

Decision criteria:
  Per-seed in-basin (loose: q0<10°, ω_dir<2°, |ω_mag|<10%):
    ≥1 IC across any of the seed's harvest ω-mag rounds = bridge succeeded.
  Cohort: ≥4/7 seeds with starts in-basin → s016-C' viable, propose
    two-stage pipeline (random search → ω-mag harvest → fresh fixed-ω-mag).
  Cohort: 0/7 → ω-mag prior structurally insufficient; pivot to S016-B
    (LC ω-mag bias) or S016-A (ω-grid).
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
from scipy.stats import qmc

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
OUT_DIR = SURVEY_DIR / "results" / "s016c_prime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_MAG_HARVEST_PCT = 10.0
TOP_K_PER_SEED = 3
N_PER_HARVEST = 64
SOBOL_SEED_Q0 = 50
SOBOL_SEED_WDIR = 51
MAX_NFEV = 120
N_WORKERS = 8

Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
STRICT_Q0 = 5.0
STRICT_OD = 1.0
STRICT_OM_PCT = 5.0
LOOSE_Q0 = 10.0
LOOSE_OD = 2.0
LOOSE_OM_PCT = 10.0


# ── quaternion / sphere utilities ─────────────────────────────────────────


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


def shoemake_to_quat(u):
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a2 = 2.0 * np.pi * u2
    a3 = 2.0 * np.pi * u3
    return np.column_stack([s2 * np.cos(a3), s1 * np.sin(a2),
                             s1 * np.cos(a2), s2 * np.sin(a3)])


def s2_uniform_to_spherical(u):
    """2-D Sobol → (theta in [0, π], phi in [-π, π])."""
    u1, u2 = u[:, 0], u[:, 1]
    z = 2.0 * u1 - 1.0
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = 2.0 * np.pi * u2 - np.pi
    return theta, phi


def spherical_to_vec(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


# ── workers (same pattern as s016) ────────────────────────────────────────

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
    harvest_idx = int(ic_dict["harvest_idx"])  # which top-K within the seed
    sobol_idx = int(ic_dict["sobol_idx"])
    omega_mag_fixed = float(ic_dict["omega_mag_fixed"])
    omega_mag_fixed_err_pct = float(ic_dict["omega_mag_fixed_err_pct"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    theta_init = float(ic_dict["theta_init"])
    phi_init = float(ic_dict["phi_init"])
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    residual_fn = make_residual_fn(seed, q0_seed_wxyz, omega_mag_fixed)
    x0 = np.array([0.0, 0.0, 0.0, theta_init, phi_init])

    initial_q0_err = quat_geodesic_deg(q0_seed_wxyz, q0_truth)
    initial_residual = residual_fn(x0)
    initial_mse = float(np.mean(initial_residual ** 2))

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
    omega_final = omega_mag_fixed * omega_dir_final

    final_residual = residual_fn(x_final)
    final_mse = float(np.mean(final_residual ** 2))

    q0_err = quat_geodesic_deg(q0_final, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    twin_err = quat_geodesic_deg(q0_final, q_twin)
    omega_dir_err = float(np.degrees(np.arccos(
        float(np.clip(np.dot(omega_dir_final, omega_truth_dir), -1.0, 1.0))
    )))
    omega_mag_err_pct = omega_mag_fixed_err_pct

    truth_basin_strict = (q0_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)
    truth_basin_loose = (q0_err < LOOSE_Q0) and (omega_dir_err < LOOSE_OD) \
        and (abs(omega_mag_err_pct) < LOOSE_OM_PCT)
    twin_basin_strict = (twin_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)

    return {
        "seed": seed,
        "harvest_idx": harvest_idx,
        "sobol_idx": sobol_idx,
        "omega_mag_fixed_rad": omega_mag_fixed,
        "omega_mag_fixed_err_pct": omega_mag_fixed_err_pct,
        "initial_q0_err_deg": float(initial_q0_err),
        "initial_mse": initial_mse,
        "q0_seed_wxyz": q0_seed_wxyz,
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


# ── harvest top-K ω-mag values per seed ───────────────────────────────────


def harvest_top_k_omega_mag_values():
    r = np.load(str(S015_RUNS))
    harvest = {}  # seed → list of (omega_mag_rad, omega_mag_err_pct, source_s015_ic)
    for s in sorted(set(r["seed"].tolist())):
        mask_seed = r["seed"] == s
        mask_omag = np.abs(r["omega_mag_err_pct"]) < OMEGA_MAG_HARVEST_PCT
        mask = mask_seed & mask_omag
        if not mask.any():
            continue
        candidate_idxs = np.where(mask)[0]
        # Sort by final_mse ascending; take top K (or all if fewer)
        mse = r["final_mse"][candidate_idxs]
        order = np.argsort(mse)
        picked = candidate_idxs[order[:TOP_K_PER_SEED]]
        omega_mag_values = []
        for idx in picked:
            omag = float(np.linalg.norm(r["omega_final_rad"][idx]))
            omag_err_pct = float(r["omega_mag_err_pct"][idx])
            omega_mag_values.append({
                "omega_mag_rad": omag,
                "omega_mag_err_pct": omag_err_pct,
                "source_s015_ic_idx": int(r["ic_idx"][idx]),
                "source_s015_final_mse": float(r["final_mse"][idx]),
            })
        harvest[int(s)] = omega_mag_values
    return harvest


def build_fresh_ics(harvest):
    sobol_q0 = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED_Q0)
    q0_set = shoemake_to_quat(sobol_q0.random(N_PER_HARVEST))
    sobol_dir = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED_WDIR)
    theta_set, phi_set = s2_uniform_to_spherical(sobol_dir.random(N_PER_HARVEST))

    ics = []
    for s, harvest_vals in harvest.items():
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        omega_truth_mag = float(np.linalg.norm(omega_truth))
        omega_truth_dir = omega_truth / omega_truth_mag
        for h_idx, hv in enumerate(harvest_vals):
            for sob_idx in range(N_PER_HARVEST):
                ics.append({
                    "seed": s,
                    "harvest_idx": h_idx,
                    "sobol_idx": sob_idx,
                    "omega_mag_fixed": hv["omega_mag_rad"],
                    "omega_mag_fixed_err_pct": hv["omega_mag_err_pct"],
                    "q0_seed_wxyz": q0_set[sob_idx],
                    "theta_init": float(theta_set[sob_idx]),
                    "phi_init": float(phi_set[sob_idx]),
                    "q0_truth_wxyz": q0_truth,
                    "omega_truth_dir": omega_truth_dir,
                    "omega_truth_mag": omega_truth_mag,
                })
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


def summarise(results, all_seeds_planned, harvest):
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)

    per_seed = {}
    for s in all_seeds_planned:
        runs = by_seed.get(s, [])
        if not runs:
            per_seed[f"seed_{s:03d}"] = {
                "seed": s,
                "n_harvests": 0,
                "n_starts": 0,
                "status": "no_harvest",
            }
            continue
        n_strict = sum(1 for r in runs if r["truth_basin_strict"])
        n_loose = sum(1 for r in runs if r["truth_basin_loose"])
        n_twin = sum(1 for r in runs if r["twin_basin_strict"])
        idx_min_mse = int(np.argmin([r["final_mse"] for r in runs]))
        argmin_run = runs[idx_min_mse]
        per_seed[f"seed_{s:03d}"] = {
            "seed": s,
            "n_harvests": len(harvest.get(s, [])),
            "n_starts": len(runs),
            "harvest_omega_mag_values_dps": [
                np.degrees(h["omega_mag_rad"]) for h in harvest.get(s, [])
            ],
            "harvest_omega_mag_err_pct": [
                h["omega_mag_err_pct"] for h in harvest.get(s, [])
            ],
            "n_truth_basin_strict": n_strict,
            "n_truth_basin_loose": n_loose,
            "n_twin_basin_strict": n_twin,
            "min_q0_err_deg": float(min(r["q0_err_deg"] for r in runs)),
            "min_omega_dir_err_deg": float(min(r["omega_dir_err_deg"] for r in runs)),
            "min_final_mse": float(min(r["final_mse"] for r in runs)),
            "argmin_mse_q0_err_deg": float(argmin_run["q0_err_deg"]),
            "argmin_mse_omega_dir_err_deg": float(argmin_run["omega_dir_err_deg"]),
            "argmin_mse_omega_mag_err_pct": float(argmin_run["omega_mag_err_pct"]),
        }

    seeds_with_starts = [s for s in all_seeds_planned
                         if per_seed[f"seed_{s:03d}"].get("n_starts", 0) > 0]
    n_loose = sum(1 for s in seeds_with_starts
                  if per_seed[f"seed_{s:03d}"]["n_truth_basin_loose"] >= 1)
    n_strict = sum(1 for s in seeds_with_starts
                   if per_seed[f"seed_{s:03d}"]["n_truth_basin_strict"] >= 1)
    n_tested = len(seeds_with_starts)

    if n_tested == 0:
        decision = "No seeds tested — harvest empty."
    elif n_loose >= max(1, int(0.6 * n_tested)):
        decision = (f"S016-C' VIABLE: {n_loose}/{n_tested} seeds get ≥1 in-basin "
                    f"(loose). Two-stage pipeline confirmed.")
    elif n_loose >= 1:
        decision = (f"S016-C' PARTIAL: {n_loose}/{n_tested} seeds in-basin (loose). "
                    f"Bridge works on some seeds; pipeline needs refinement (which "
                    f"seeds work, what's different).")
    else:
        decision = (f"S016-C' REFUTED: 0/{n_tested} seeds in-basin. ω-mag prior "
                    f"alone is insufficient. Pivot to S016-B (LC bias) or S016-A "
                    f"(ω-grid).")

    return {
        "harvest_threshold_pct": OMEGA_MAG_HARVEST_PCT,
        "top_k_per_seed": TOP_K_PER_SEED,
        "n_per_harvest": N_PER_HARVEST,
        "max_nfev": MAX_NFEV,
        "sobol_seeds": {"q0": SOBOL_SEED_Q0, "wdir": SOBOL_SEED_WDIR},
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
        "harvest_idx": np.array([r["harvest_idx"] for r in results], dtype=int),
        "sobol_idx": np.array([r["sobol_idx"] for r in results], dtype=int),
        "omega_mag_fixed_rad": np.array([r["omega_mag_fixed_rad"] for r in results]),
        "omega_mag_fixed_err_pct": np.array(
            [r["omega_mag_fixed_err_pct"] for r in results]),
        "initial_q0_err_deg": np.array([r["initial_q0_err_deg"] for r in results]),
        "initial_mse": np.array([r["initial_mse"] for r in results]),
        "q0_seed_wxyz": np.stack([r["q0_seed_wxyz"] for r in results]),
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
    harvest = harvest_top_k_omega_mag_values()
    seeds_with_harvest = sorted(harvest.keys())
    seed_data = collect_seed_data(seeds_with_harvest)

    print(f"s016-C': harvest summary (top-{TOP_K_PER_SEED} ω-mag values per seed,"
          f" |ω_mag_err|<{OMEGA_MAG_HARVEST_PCT}%)", flush=True)
    for s in seeds_with_harvest:
        vals = harvest[s]
        omags = [f"{np.degrees(h['omega_mag_rad']):.4f}dps({h['omega_mag_err_pct']:+.2f}%)"
                 for h in vals]
        print(f"  seed {s:>3d}: {len(vals)} ω-mag value(s) — {', '.join(omags)}",
              flush=True)

    ics = build_fresh_ics(harvest)
    n_total = len(ics)
    print(f"\nTotal LM runs: {n_total} ({len(seeds_with_harvest)} seeds × "
          f"varied harvests × N={N_PER_HARVEST} fresh Sobol(q0)×Sobol(ω-dir) ICs)",
          flush=True)
    print(f"Pool({N_WORKERS}), BLAS=1, max_nfev={MAX_NFEV}.", flush=True)
    print(f"Out dir: {OUT_DIR}", flush=True)
    print(f"Expected wall ≈ {n_total * 30.0 / N_WORKERS:.0f} s ≈ "
          f"{n_total * 30.0 / N_WORKERS / 60:.1f} min.\n", flush=True)

    args_list = [(ic, MAX_NFEV) for ic in ics]
    t0 = time.time()
    results = []
    last_print = t0
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_lm_for_ic, args_list,
                                                  chunksize=8)):
            results.append(r)
            now = time.time()
            if now - last_print > 30.0 or i + 1 == n_total:
                done = i + 1
                rate = done / (now - t0) if now > t0 else 0.0
                eta = (n_total - done) / rate if rate > 0 else 0.0
                print(f"  [{done:4d}/{n_total}] rate={rate:.2f}/s, eta={eta:.0f} s",
                      flush=True)
                last_print = now
    wall = time.time() - t0
    print(f"\nAll runs done. Total wall: {wall:.1f} s = {wall/60:.1f} min.\n",
          flush=True)

    save_npz(results)
    summary = summarise(results, all_seeds_planned, harvest)
    save_summary(summary)

    print("\n=== s016-C' per-seed yield ===")
    print(f"  {'seed':>5s} {'harvests':>9s} {'starts':>7s} {'in_loose':>9s} "
          f"{'in_strict':>10s} {'min_q0':>7s} {'min_od':>7s} {'min_mse':>10s}")
    for s in all_seeds_planned:
        cell = summary["per_seed"][f"seed_{s:03d}"]
        if cell.get("n_starts", 0) == 0:
            print(f"  {s:5d} {'0':>9s} {'0':>7s} {'-':>9s} {'-':>10s} "
                  f"{'-':>7s} {'-':>7s} {'-':>10s}    (no harvest)")
        else:
            print(f"  {s:5d} {cell['n_harvests']:9d} {cell['n_starts']:7d} "
                  f"{cell['n_truth_basin_loose']:9d} "
                  f"{cell['n_truth_basin_strict']:10d} "
                  f"{cell['min_q0_err_deg']:7.2f} "
                  f"{cell['min_omega_dir_err_deg']:7.2f} "
                  f"{cell['min_final_mse']:10.3e}")

    print(f"\n  loose-basin yield (cohort): "
          f"{summary['n_seeds_with_loose_basin']}/"
          f"{len(summary['seeds_with_starts'])} seeds with starts")
    print(f"  → DECISION: {summary['decision']}")
    print(f"\nTotal wall: {time.time() - t_main:.1f} s")


if __name__ == "__main__":
    main()
