"""s011 — Q4c-ii cohort pilot: multi-axis Sobol(q0) on SO(3) × LM polish at truth-ω.

Survey question Q4c. s005 settled joint (q0, ω) LM as the inversion architecture
inside the s003 truth-ω tube. s006 + s010 confirmed that several seeds (28, 44)
have sub-Sobol-resolution narrow basins — bare Sobol density alone won't
seed them. s009 measured cohort basin distribution (45% wide / 52% mid / 3%
tight) but only along body-X axis; multi-axis Sobol is needed for full
coverage of seeds 44/76.

This pilot answers: at what Sobol(q0) density on SO(3) (uniform Shoemake)
does LM polish from each candidate land ≥1 IC inside the per-seed truth
basin for ≥90% of pilot seeds, AT FIXED TRUTH-ω?

Decoupling: ω is held at truth in this round. The next round (s012) adds
the ω-grid axis (5 dirs × 5 mags = 25 cells) once we know how the
Sobol(q0) density scales with in-basin yield.

Method:
  - 10 pilot seeds: 8 PA-stratified s002 anchors + 2 cohort tail seeds.
      anchors:   6, 10, 21, 41, 48, 60, 84, 91
      cohort tail: 28 (sub-Sobol-narrow truth basin), 44 (sub-Sobol-narrow
                   competing basin)
  - 1 q0 density round 1: N=64.
      Sobol-Shoemake uniform on SO(3), SOBOL_SEED=42 (matches s002/s006/s010).
      If round-1 cohort yield <9/10, follow up with N=256 on the failing
      seeds only (deferred experiment, not in this script).
  - ω fixed at truth-ω (per seed, from traj_load.load_truth(seed)).
  - Polish: scipy least_squares(method='lm', max_nfev=60), 6-DOF
    parameterization x = (δθ, ω); residuals = surrogate full-LC residuals.
    Same code path as s005's run_lm_for_ic; max_nfev=60 (vs s005's 200)
    is calibrated against s005's in-basin nfev distribution (max=66, p90=47)
    + a smoke test on seed 41 random Sobol ICs (deep-far stalls hit nfev=200
    burning 80 s with no useful work; at nfev=60 they bail at ~25 s).
  - Pool(8), BLAS=1.

Reporting (per-IC, all four metrics + ρ-band-eligible final state):
  - q0_err, twin_err, ω_dir_err, ω_mag_err_pct
  - final_mse, n_fev, wall_s
  - in-basin flags (strict 5°/1°/5%, loose 10°/2°/10%)
  - twin-basin flag

Per-(seed, density) summary:
  - n_in_basin_strict / N: how many ICs landed in the truth basin
  - n_in_basin_strict_unique: how many DISTINCT in-basin convergence
    points (geodesic-cluster at 1°). Density doesn't help if all ICs
    converge to the same basin point only when one of them is already
    inside the basin.
  - min_q0_err: best LM landing per seed (sanity check on sub-Sobol-narrow
    seeds).
  - n_competing_basins: clusters of LM landings ≥ 30° from truth with
    final_mse < 0.5 mag² (the s009/s010 competing-basin signature).

Decision criteria (per density):
  - per_seed_in_basin_strict_yield: n_seeds with ≥1 in-basin landing / 10
  - This is the cohort-scale Q4c-ii success rate at this density.
  - ≥9/10 at N=64 → Sobol-q0 is cheap; Q4c works at any reasonable budget.
  - ≥9/10 at N=256, <9/10 at N=64 → moderate density required.
  - <9/10 at N=256 → Q4c needs much higher density OR partial-cohort
    acceptance.

Compute budget (Pool(8), max_nfev=60, ~20 s/run mean from smoke test):
  - N=64:  10 seeds × 64 ICs × 20 s / 8 workers ≈ 1600 s ≈ 27 min wall.
  - Kill criterion: 60 min.

Outputs:
  - results/s011/runs.npz       per-IC final state arrays (all densities)
  - results/s011/summary.json   per-(seed, density) yield summary + decision
  - results/s011/yield_vs_density.png  per-seed yield + cohort yield bars
  - results/s011/q0_err_distribution.png  CDF of final q0_err per density
"""

# BLAS=1 BEFORE Pool fork.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s011"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 pilot seeds: s002 anchors + cohort-tail seeds 28 (sub-Sobol-narrow truth)
# and 44 (sub-Sobol-narrow competing basin).
SEEDS = [6, 10, 21, 28, 41, 44, 48, 60, 84, 91]
DENSITIES = [64]  # round 1: N=64 only; if cohort yield <9/10, follow up with N=256 on failures
SOBOL_SEED = 42  # matches s002 / s006 / s010
N_WORKERS = 8
# max_nfev=60 covers s005's in-basin nfev p90=47 / max=66 with margin while
# capping deep-stall walls. Smoke test on seed 41 / 4 random ICs showed:
#   - in-basin LM uses ~11 nfev (5 s); stalls hit max_nfev=200 (80+ s).
#   - At max_nfev=60, stall walls drop to ~25 s — 3-4× cheaper.
MAX_NFEV = 60

# Per-seed cached truth surrogate-MSE (from s001 / per_seed.csv).
TRUTH_MSE_REF = {
    6:  2.92451e-3,
    10: 3.59353e-3,
    18: 3.10499e-4,  # not used here; for cross-reference
    21: 5.76963e-4,
    28: 5.84697e-4,
    41: 1.33258e-4,
    44: 3.35654e-4,
    48: 3.07952e-4,
    60: 1.55543e-3,
    84: 2.56333e-4,
    91: 2.03195e-4,
}

# Twin: 180° body-X rotation.
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# In-basin definitions (match s005 / s009).
STRICT_Q0 = 5.0
STRICT_OD = 1.0
STRICT_OM_PCT = 5.0
LOOSE_Q0 = 10.0
LOOSE_OD = 2.0
LOOSE_OM_PCT = 10.0


# ────────────────────────────────────────────────────────────────────────────
# Quaternion utilities (cloned from s005 / s006 to avoid cross-experiment
# refactor — survey contract: thin lib, self-contained scripts).
# ────────────────────────────────────────────────────────────────────────────


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_rotvec_wxyz(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0 - 0.125 * angle * angle,
                         0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), s * rotvec[0], s * rotvec[1], s * rotvec[2]])


def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    """Shoemake's uniform-on-S^3 mapping (matches s002 / s006 / s010)."""
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a2 = 2.0 * np.pi * u2
    a3 = 2.0 * np.pi * u3
    x = s1 * np.sin(a2)
    y = s1 * np.cos(a2)
    z = s2 * np.sin(a3)
    w = s2 * np.cos(a3)
    return np.column_stack([w, x, y, z])


def build_sobol_q0(n: int, sobol_seed: int) -> np.ndarray:
    """N uniform Sobol-Shoemake quaternions, (N, 4) wxyz."""
    sobol = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    u = sobol.random(n)
    return shoemake_to_quat(u)


# ────────────────────────────────────────────────────────────────────────────
# Worker globals + LM run (pattern from s005)
# ────────────────────────────────────────────────────────────────────────────

_W_TIMES_BY_SEED: dict[int, np.ndarray] = {}
_W_SUN_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_BY_SEED: dict[int, np.ndarray] = {}
_W_SAT_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_DIST_BY_SEED: dict[int, np.ndarray] = {}
_W_MAG_HIFI_BY_SEED: dict[int, np.ndarray] = {}
_W_INERTIA_TENSOR = None


def init_worker(seed_data: dict, inertia_tensor: np.ndarray):
    # Critical: torch ignores OMP_NUM_THREADS by default and defaults to
    # cpu_count // 2 (16 threads on this box). With Pool(8) that's 128 threads
    # competing for 8 cores — ~8× per-worker slowdown vs single-thread. Project
    # memory `feedback_blas_threads_for_pool.md` listed OMP/OPENBLAS/MKL but
    # missed torch. Explicitly pin torch threads here, in the worker, after
    # surrogate import (torch is imported transitively by surrogate_eval).
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


def make_residual_fn(seed: int, q0_seed_wxyz: np.ndarray):
    times = _W_TIMES_BY_SEED[seed]
    sun = _W_SUN_BY_SEED[seed]
    obs = _W_OBS_BY_SEED[seed]
    sat = _W_SAT_BY_SEED[seed]
    obs_dist = _W_OBS_DIST_BY_SEED[seed]
    mag_truth = _W_MAG_HIFI_BY_SEED[seed]
    inertia = _W_INERTIA_TENSOR
    finite_truth = np.isfinite(mag_truth)

    def residuals(x: np.ndarray) -> np.ndarray:
        delta_theta = x[:3]
        omega = x[3:6]
        delta_q = quat_from_rotvec_wxyz(delta_theta)
        q0 = quat_multiply_wxyz(delta_q, q0_seed_wxyz)
        q0 = q0 / np.linalg.norm(q0)
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


def run_lm_for_ic(args: tuple) -> dict:
    """Worker: args = (ic_dict, max_nfev). Returns serialised result dict."""
    ic_dict, max_nfev = args
    seed = int(ic_dict["seed"])
    density = int(ic_dict["density"])
    ic_idx = int(ic_dict["ic_idx"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    omega_seed_rad = np.asarray(ic_dict["omega_seed_rad"], dtype=float)  # = truth-ω
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    residual_fn = make_residual_fn(seed, q0_seed_wxyz)
    x0 = np.concatenate([np.zeros(3), omega_seed_rad])

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
    except Exception as e:
        success = False
        status = -99
        n_fev = 0
        x_final = x0.copy()
    wall = time.time() - t0

    delta_theta_final = x_final[:3]
    omega_final = x_final[3:6]
    delta_q_final = quat_from_rotvec_wxyz(delta_theta_final)
    q0_final = quat_multiply_wxyz(delta_q_final, q0_seed_wxyz)
    q0_final = q0_final / np.linalg.norm(q0_final)

    final_residual = residual_fn(x_final)
    final_mse = float(np.mean(final_residual ** 2))

    q0_err = quat_geodesic_deg(q0_final, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    twin_err = quat_geodesic_deg(q0_final, q_twin)
    omega_final_mag = float(np.linalg.norm(omega_final))
    if omega_final_mag < 1e-12:
        omega_dir_err = 180.0
    else:
        omega_dir_err = float(np.degrees(np.arccos(
            float(np.clip(np.dot(omega_final / omega_final_mag, omega_truth_dir),
                          -1.0, 1.0))
        )))
    omega_mag_err_pct = 100.0 * (omega_final_mag - omega_truth_mag) / omega_truth_mag

    truth_basin_strict = (q0_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)
    truth_basin_loose = (q0_err < LOOSE_Q0) and (omega_dir_err < LOOSE_OD) \
        and (abs(omega_mag_err_pct) < LOOSE_OM_PCT)
    twin_basin_strict = (twin_err < STRICT_Q0) and (omega_dir_err < STRICT_OD) \
        and (abs(omega_mag_err_pct) < STRICT_OM_PCT)

    return {
        "seed": seed,
        "density": density,
        "ic_idx": ic_idx,
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


# ────────────────────────────────────────────────────────────────────────────
# IC construction
# ────────────────────────────────────────────────────────────────────────────


def build_ics_for_seed_density(seed: int, density: int,
                                q0_truth: np.ndarray,
                                omega_truth: np.ndarray) -> list[dict]:
    """Build `density` ICs for one (seed, density) cell — Sobol-Shoemake on SO(3),
    ω at truth."""
    omega_mag = float(np.linalg.norm(omega_truth))
    omega_dir = omega_truth / omega_mag
    q0_set = build_sobol_q0(density, SOBOL_SEED)
    ics = []
    for i in range(density):
        ics.append({
            "seed": seed,
            "density": density,
            "ic_idx": i,
            "q0_seed_wxyz": q0_set[i],
            "omega_seed_rad": omega_truth.copy(),
            "q0_truth_wxyz": q0_truth.copy(),
            "omega_truth_dir": omega_dir.copy(),
            "omega_truth_mag": omega_mag,
        })
    return ics


def collect_seed_data() -> dict:
    seed_data = {}
    for s in SEEDS:
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


# ────────────────────────────────────────────────────────────────────────────
# Aggregation + decision
# ────────────────────────────────────────────────────────────────────────────


def cluster_in_basin_landings(runs: list[dict], cluster_deg: float = 1.0) -> int:
    """Count distinct in-basin landings via greedy 1°-geodesic clustering."""
    in_basin = [r for r in runs if r["truth_basin_strict"]]
    if not in_basin:
        return 0
    qs = [r["q0_final_wxyz"] for r in in_basin]
    clusters = [qs[0]]
    for q in qs[1:]:
        if all(quat_geodesic_deg(q, c) > cluster_deg for c in clusters):
            clusters.append(q)
    return len(clusters)


def count_competing_basins(runs: list[dict],
                           min_geo_from_truth: float = 30.0,
                           max_mse: float = 0.5) -> int:
    """Count distinct competing-basin landings (LM-discovered competing basins
    at low MSE far from truth — the s009/s010 signature)."""
    candidates = [r for r in runs
                  if r["q0_err_deg"] >= min_geo_from_truth
                  and r["final_mse"] < max_mse]
    if not candidates:
        return 0
    qs = [r["q0_final_wxyz"] for r in candidates]
    clusters = [qs[0]]
    for q in qs[1:]:
        if all(quat_geodesic_deg(q, c) > 5.0 for c in clusters):
            clusters.append(q)
    return len(clusters)


def summarise(results: list[dict]) -> dict:
    """Per-(seed, density) yield summary + cohort decision."""
    by_cell: dict = {}
    for r in results:
        by_cell.setdefault((r["seed"], r["density"]), []).append(r)

    per_cell = {}
    for (seed, density), runs in by_cell.items():
        n = len(runs)
        n_strict = sum(1 for r in runs if r["truth_basin_strict"])
        n_loose = sum(1 for r in runs if r["truth_basin_loose"])
        n_twin = sum(1 for r in runs if r["twin_basin_strict"])
        n_strict_unique = cluster_in_basin_landings(runs, cluster_deg=1.0)
        n_competing = count_competing_basins(runs, min_geo_from_truth=30.0,
                                             max_mse=0.5)
        per_cell[f"seed_{seed:03d}_N{density:04d}"] = {
            "seed": seed,
            "density": density,
            "n_ics": n,
            "n_truth_basin_strict": n_strict,
            "n_truth_basin_loose": n_loose,
            "n_twin_basin_strict": n_twin,
            "n_truth_basin_strict_unique_clusters": n_strict_unique,
            "n_competing_basins_below_mse_0_5": n_competing,
            "min_q0_err_deg": float(min(r["q0_err_deg"] for r in runs)),
            "min_final_mse": float(min(r["final_mse"] for r in runs)),
            "median_wall_s": float(np.median([r["wall_s"] for r in runs])),
            "truth_mse_ref": TRUTH_MSE_REF.get(seed),
        }

    # Cohort yield per density.
    cohort_yield = {}
    for d in DENSITIES:
        n_seeds_with_basin = 0
        for s in SEEDS:
            cell = per_cell.get(f"seed_{s:03d}_N{d:04d}")
            if cell and cell["n_truth_basin_strict"] >= 1:
                n_seeds_with_basin += 1
        cohort_yield[f"N{d}"] = {
            "density": d,
            "n_seeds_with_at_least_one_in_basin": n_seeds_with_basin,
            "n_seeds_total": len(SEEDS),
            "cohort_in_basin_yield_pct": 100.0 * n_seeds_with_basin / len(SEEDS),
        }

    # Decision (round 1: only N=64 measured).
    n64 = cohort_yield[f"N{DENSITIES[0]}"]["n_seeds_with_at_least_one_in_basin"]
    if n64 >= 9:
        decision = f"Sobol-q0 cheap (N={DENSITIES[0]} achieves {n64}/10); Q4c works at low density"
    elif n64 >= 7:
        decision = (f"N={DENSITIES[0]} yields {n64}/10; partial-cohort acceptance OR follow-up "
                    f"N=256 on the {10 - n64} failing seeds")
    else:
        decision = (f"N={DENSITIES[0]} yields only {n64}/10; Q4c needs much higher density "
                    f"OR architectural shift")

    return {
        "seeds": SEEDS,
        "densities": DENSITIES,
        "sobol_seed": SOBOL_SEED,
        "n_runs_total": len(results),
        "per_cell": per_cell,
        "cohort_yield_by_density": cohort_yield,
        "decision": decision,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    seed_data = collect_seed_data()

    # Build all ICs across (seed, density).
    all_ics: list[dict] = []
    for s in SEEDS:
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        for density in DENSITIES:
            all_ics.extend(build_ics_for_seed_density(s, density, q0_truth, omega_truth))

    n_total = len(all_ics)
    expected_wall = n_total * 20.0 / N_WORKERS  # ~20 s/run mean (in-basin 5 s, stall 25 s @ nfev=60)
    print(f"s011: {len(SEEDS)} seeds × {DENSITIES} densities → "
          f"{n_total} LM runs  (expected wall ≈ {expected_wall:.0f} s).",
          flush=True)
    print(f"Pool({N_WORKERS}), BLAS=1, SOBOL_SEED={SOBOL_SEED}, max_nfev={MAX_NFEV}.",
          flush=True)

    args_list = [(ic, MAX_NFEV) for ic in all_ics]

    t0 = time.time()
    results = []
    last_print = t0
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_lm_for_ic, args_list, chunksize=8)):
            results.append(r)
            now = time.time()
            # Print incremental progress every 30 s.
            if now - last_print > 30.0 or i + 1 == n_total:
                done = i + 1
                rate = done / (now - t0) if now > t0 else 0.0
                eta = (n_total - done) / rate if rate > 0 else 0.0
                print(f"  [{done:5d}/{n_total}] rate={rate:.1f}/s, eta={eta:.0f} s",
                      flush=True)
                last_print = now
    wall = time.time() - t0
    print(f"\nAll runs done. Total wall: {wall:.1f} s.\n", flush=True)

    save_npz(results)
    summary = summarise(results)
    save_summary(summary, results)
    save_plots(results, summary)

    # Decision + per-seed table.
    print("\n=== s011 cohort yield by density ===")
    for d in DENSITIES:
        cy = summary["cohort_yield_by_density"][f"N{d}"]
        print(f"  N={d:4d}: {cy['n_seeds_with_at_least_one_in_basin']}/"
              f"{cy['n_seeds_total']} seeds with ≥1 in-basin landing  "
              f"({cy['cohort_in_basin_yield_pct']:.1f}%)")

    print("\n=== per-(seed, density) yield ===")
    print(f"  {'seed':>5s} {'N':>5s} {'in-basin':>9s} {'unique':>7s} "
          f"{'competing':>10s} {'min_q0_err':>11s} {'min_mse':>10s}")
    for s in SEEDS:
        for d in DENSITIES:
            cell = summary["per_cell"][f"seed_{s:03d}_N{d:04d}"]
            print(f"  {s:5d} {d:5d} {cell['n_truth_basin_strict']:9d} "
                  f"{cell['n_truth_basin_strict_unique_clusters']:7d} "
                  f"{cell['n_competing_basins_below_mse_0_5']:10d} "
                  f"{cell['min_q0_err_deg']:11.3f} "
                  f"{cell['min_final_mse']:10.3e}")

    print(f"\n  → DECISION: {summary['decision']}")
    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


def save_npz(results: list[dict]):
    arrs = {
        "seed": np.array([r["seed"] for r in results], dtype=int),
        "density": np.array([r["density"] for r in results], dtype=int),
        "ic_idx": np.array([r["ic_idx"] for r in results], dtype=int),
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
        "truth_basin_loose":  np.array([r["truth_basin_loose"] for r in results]),
        "twin_basin_strict":  np.array([r["twin_basin_strict"] for r in results]),
        "n_fev": np.array([r["n_fev"] for r in results], dtype=int),
        "wall_s": np.array([r["wall_s"] for r in results]),
        "success": np.array([r["success"] for r in results]),
        "status": np.array([r["status"] for r in results], dtype=int),
    }
    out = OUT_DIR / "runs.npz"
    np.savez(out, **arrs)
    print(f"Saved: {out}")


def save_summary(summary: dict, results: list[dict]):
    out = OUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


def save_plots(results: list[dict], summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── yield_vs_density.png ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    width = 0.4
    x = np.arange(len(SEEDS))
    for di, d in enumerate(DENSITIES):
        yields = []
        for s in SEEDS:
            cell = summary["per_cell"][f"seed_{s:03d}_N{d:04d}"]
            yields.append(cell["n_truth_basin_strict"])
        ax.bar(x + (di - 0.5) * width, yields, width,
               label=f"N={d}", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("# Sobol candidates landing in truth basin (strict)")
    ax.set_title("s011 — per-seed in-basin-landing count vs Sobol density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    cohort = [summary["cohort_yield_by_density"][f"N{d}"]["cohort_in_basin_yield_pct"]
              for d in DENSITIES]
    ax.bar([str(d) for d in DENSITIES], cohort, color="steelblue", alpha=0.85)
    for i, c in enumerate(cohort):
        ax.text(i, c + 1, f"{c:.0f}%", ha="center")
    ax.axhline(90.0, color="red", linestyle="--", alpha=0.5,
               label="≥90% cohort decision bar")
    ax.set_ylabel("cohort in-basin yield  (% seeds with ≥1 in-basin)")
    ax.set_xlabel("Sobol(q0) density on SO(3)")
    ax.set_ylim(0, 110)
    ax.set_title("s011 — cohort yield vs density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = OUT_DIR / "yield_vs_density.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── q0_err_distribution.png ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in DENSITIES:
        errs = sorted(r["q0_err_deg"] for r in results if r["density"] == d)
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, label=f"N={d} (n={len(errs)})", linewidth=2)
    ax.axvline(STRICT_Q0, color="green", linestyle="--", alpha=0.5,
               label=f"strict basin = {STRICT_Q0}°")
    ax.axvline(LOOSE_Q0, color="olive", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("final q0 error after LM polish  [deg]")
    ax.set_ylabel("CDF over (seed × IC) cells")
    ax.set_title("s011 — q0-error CDF after LM polish, per Sobol density")
    ax.legend()
    ax.grid(alpha=0.3)
    p = OUT_DIR / "q0_err_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
