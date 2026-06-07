"""s015 — joint (q0, ω) inversion pilot: random-ω stitched-Sobol + 6-DOF LM.

This is the first time we hide ω. Up to s014 the algorithm got truth-ω free
and only had to find q0; s015 samples ω alongside q0 and asks LM to find
both. This is the actual inversion problem.

Architecture (decision A2-stitched, locked 2026-05-01):
  - q0:    Sobol-Shoemake on SO(3), SOBOL_SEED_Q0 = 42
              (matches s002 / s006 / s010 / s011; first N=64 ICs are
               bit-identical to s011 N=64 → cross-experiment comparison)
  - ω-dir: uniform on S² via inverse-CDF on 2-D Sobol, SOBOL_SEED_WDIR = 43
  - ω-mag: uniform-linear in [0.1, 1.5] dps, 1-D Sobol, SOBOL_SEED_WMAG = 44
  - Three independent Sobol streams, stitched per-IC (NOT a 6-D Sobol):
    qualitative argument — q0/ω-dir/ω-mag drive different LC features
    (orientation / glint axis / temporal compression); independent streams
    preserve marginal coverage of each.

Polish:
  - scipy least_squares(method='lm', max_nfev=120), 6-DOF parameterization
    x = (δθ, ω) (δθ in tangent space around q0_seed; ω absolute 3-vec).
    Residuals = surrogate full-LC residuals.
  - max_nfev=120 (vs s011's 60): ω is no longer truth — LM has 6 dimensions
    to navigate from a random start, not 3. s005 in-basin nfev p90=47 was
    measured INSIDE the s003 truth-ω tube; outside-tube ICs need more iter.
    Cap at 120 to bound deep-stall walls (~50 s @ nfev=120 vs ~25 s @ 60).

Pool(8), BLAS=1 + torch_threads=1.

Seeds (8 total):
  Anchors (s011 pilot, known recoverable at fixed truth-ω):
    6, 28, 41, 91   — 28 narrow-basin, 91 m115's old failure.
  n_rot<2 tail (s014b cohort prediction):
    13, 42, 79      — predicted seed-10-class; expected to fail.
  Fresh average (untouched, n_rot ≈ 11):
    49              — clean median pick from the 56 untouched n_rot ≥ 5.

Density:
  N = 512 (smoke pilot). Calibrated to:
    - s006 / s010 measurement: bare-Sobol density needed to seed narrow
      basins is impractical (~1e5 q0 ICs alone). LM-grab radius is wide
      (~50-100° in q0 at fixed truth-ω); we expect joint search to need
      2-4× s011's N=64 density, not 16×.
    - Wall budget: 8 seeds × 512 ICs × ~35 s mean / 8 workers ≈ 14k s ≈
      3.9 h. Fits an overnight run; can drop to N=128 for fast iteration
      after the architecture validates.

Reporting (per-IC, all four metrics + ρ-band-eligible final state):
  - q0_err, twin_err, ω_dir_err, ω_mag_err_pct
  - final_mse, n_fev, wall_s
  - in-basin flags (strict 5°/1°/5%, loose 10°/2°/10%)
  - twin-basin flag

Per-seed summary:
  - n_in_basin_strict: ICs landing inside the truth basin
  - n_unique_clusters: distinct in-basin convergence points (1° geo)
  - n_competing_basins: clusters of LM landings ≥30° from truth at low MSE
  - min_q0_err, min_final_mse, surrogate-best-IC index
  - per-seed wall

Decision criteria (cohort-scale at N=512):
  - ≥6/8 anchors+49 in-basin → joint search at N=512 viable; cohort scan
    next at N=128-256 to map yield distribution.
  - 4-5/8 → architecture works on easy seeds, struggles on n_rot<2 tail;
    confirm tail prediction, refine on the gap.
  - <4/8 → joint search at N=512 insufficient; architecture revision
    (ω-grid stratification, basin-hopping outer loop, hierarchical search).

The 1-seed smoke test (seed 6 only, ~25 min) precedes the full pilot —
purpose is to validate the script lands cleanly + checkpoint the predicted-
easy seed before committing 4 hours.

Outputs:
  - results/s015/runs.npz       per-IC final state arrays
  - results/s015/summary.json   per-seed yield summary + decision
  - results/s015/yield.png      per-seed in-basin / competing / min-q0-err bars
  - results/s015/q0_err_cdf.png CDF of final q0_err per seed
  - results/s015/omega_recovery.png ω-dir vs ω-mag err scatter
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s015"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pilot seed mix (8 total, locked 2026-05-01).
SEEDS_FULL = [6, 13, 28, 41, 42, 49, 79, 91]
SEED_ROLE = {
    6: "anchor",
    28: "anchor_narrow_basin",
    41: "anchor_multi_solution",
    91: "anchor_m115_failure",
    13: "n_rot_lt_2_tail",
    42: "n_rot_lt_2_tail",
    79: "n_rot_lt_2_tail",
    49: "fresh_average",
}

# Three independent Sobol streams.
SOBOL_SEED_Q0 = 42       # matches s002/s006/s010/s011 → first N=64 q0s identical
SOBOL_SEED_WDIR = 43
SOBOL_SEED_WMAG = 44

# ω-mag bounds (deg/s) — confirmed against truth cohort 0.106-1.476 dps.
W_MAG_MIN_DPS = 0.1
W_MAG_MAX_DPS = 1.5

N_DEFAULT = 512
N_WORKERS = 8
MAX_NFEV = 120  # higher than s011's 60: joint 6-DOF search needs more iter

# Per-seed cached truth surrogate-MSE (extends s011 with new seeds).
TRUTH_MSE_REF = {
    6:  2.92451e-3,
    13: None,  # not in s011 cache; leave null, will be computed from runs
    28: 5.84697e-4,
    41: 1.33258e-4,
    42: None,
    49: None,
    79: None,
    91: 2.03195e-4,
}

Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

STRICT_Q0 = 5.0
STRICT_OD = 1.0
STRICT_OM_PCT = 5.0
LOOSE_Q0 = 10.0
LOOSE_OD = 2.0
LOOSE_OM_PCT = 10.0


# ────────────────────────────────────────────────────────────────────────────
# Quaternion utilities (cloned from s011 — survey contract: thin lib).
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
    """Shoemake uniform-on-S^3 from 3-D Sobol points in [0,1)^3."""
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


def s2_uniform_from_sobol(u: np.ndarray) -> np.ndarray:
    """Uniform-on-S^2 from 2-D Sobol points in [0,1)^2 (inverse CDF on
    spherical coords). Returns (N, 3) unit vectors."""
    u1, u2 = u[:, 0], u[:, 1]
    z = 2.0 * u1 - 1.0           # cos(θ) uniform on [-1, 1]
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = 2.0 * np.pi * u2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def build_sobol_q0(n: int) -> np.ndarray:
    sobol = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED_Q0)
    return shoemake_to_quat(sobol.random(n))


def build_sobol_omega(n: int) -> np.ndarray:
    """Stitched ω sampler: dir from S², mag from uniform [W_MIN, W_MAX] dps,
    returned as (N, 3) ω-vectors in rad/s."""
    sobol_dir = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED_WDIR)
    sobol_mag = qmc.Sobol(d=1, scramble=True, seed=SOBOL_SEED_WMAG)
    dirs = s2_uniform_from_sobol(sobol_dir.random(n))
    mags_dps = (W_MAG_MIN_DPS
                + (W_MAG_MAX_DPS - W_MAG_MIN_DPS) * sobol_mag.random(n).flatten())
    mags_rad = np.radians(mags_dps)
    return dirs * mags_rad[:, None]


# ────────────────────────────────────────────────────────────────────────────
# Worker globals + LM run (pattern from s011)
# ────────────────────────────────────────────────────────────────────────────

_W_TIMES_BY_SEED: dict[int, np.ndarray] = {}
_W_SUN_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_BY_SEED: dict[int, np.ndarray] = {}
_W_SAT_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_DIST_BY_SEED: dict[int, np.ndarray] = {}
_W_MAG_HIFI_BY_SEED: dict[int, np.ndarray] = {}
_W_INERTIA_TENSOR = None


def init_worker(seed_data: dict, inertia_tensor: np.ndarray):
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
    ic_dict, max_nfev = args
    seed = int(ic_dict["seed"])
    ic_idx = int(ic_dict["ic_idx"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    omega_seed_rad = np.asarray(ic_dict["omega_seed_rad"], dtype=float)
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    residual_fn = make_residual_fn(seed, q0_seed_wxyz)
    x0 = np.concatenate([np.zeros(3), omega_seed_rad])

    initial_q0_err = quat_geodesic_deg(q0_seed_wxyz, q0_truth)
    omega_seed_mag = float(np.linalg.norm(omega_seed_rad))
    initial_omega_dir_err = (180.0 if omega_seed_mag < 1e-12 else
        float(np.degrees(np.arccos(float(np.clip(
            np.dot(omega_seed_rad / omega_seed_mag, omega_truth_dir),
            -1.0, 1.0,
        ))))))
    initial_omega_mag_err_pct = (
        100.0 * (omega_seed_mag - omega_truth_mag) / omega_truth_mag
    )
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
        "ic_idx": ic_idx,
        "initial_q0_err_deg": float(initial_q0_err),
        "initial_omega_dir_err_deg": float(initial_omega_dir_err),
        "initial_omega_mag_err_pct": float(initial_omega_mag_err_pct),
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


# ────────────────────────────────────────────────────────────────────────────
# IC construction
# ────────────────────────────────────────────────────────────────────────────


def build_ics(seeds: list[int], n: int) -> list[dict]:
    """Build n joint (q0, ω) ICs per seed via stitched Sobol streams.

    All seeds share the same stitched (q0, ω) IC table — the variation
    across seeds comes from the truth-state metadata and the seed-specific
    light curve, not from per-seed IC randomization. This makes seeds
    comparable at fixed IC indices."""
    q0_set = build_sobol_q0(n)
    omega_set = build_sobol_omega(n)
    ics = []
    for s in seeds:
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        omega_truth_mag = float(np.linalg.norm(omega_truth))
        omega_truth_dir = omega_truth / omega_truth_mag
        for i in range(n):
            ics.append({
                "seed": s,
                "ic_idx": i,
                "q0_seed_wxyz": q0_set[i],
                "omega_seed_rad": omega_set[i],
                "q0_truth_wxyz": q0_truth.copy(),
                "omega_truth_dir": omega_truth_dir.copy(),
                "omega_truth_mag": omega_truth_mag,
            })
    return ics


def collect_seed_data(seeds: list[int]) -> dict:
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


# ────────────────────────────────────────────────────────────────────────────
# Aggregation + decision
# ────────────────────────────────────────────────────────────────────────────


def cluster_in_basin_landings(runs: list[dict], cluster_deg: float = 1.0) -> int:
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


def summarise(results: list[dict], seeds: list[int], n: int) -> dict:
    by_seed: dict = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)

    per_seed = {}
    for seed, runs in by_seed.items():
        n_strict = sum(1 for r in runs if r["truth_basin_strict"])
        n_loose = sum(1 for r in runs if r["truth_basin_loose"])
        n_twin = sum(1 for r in runs if r["twin_basin_strict"])
        n_strict_unique = cluster_in_basin_landings(runs, cluster_deg=1.0)
        n_competing = count_competing_basins(runs, min_geo_from_truth=30.0,
                                             max_mse=0.5)
        idx_min_mse = int(np.argmin([r["final_mse"] for r in runs]))
        argmin_run = runs[idx_min_mse]
        per_seed[f"seed_{seed:03d}"] = {
            "seed": seed,
            "role": SEED_ROLE.get(seed, "unknown"),
            "n_ics": len(runs),
            "n_truth_basin_strict": n_strict,
            "n_truth_basin_loose": n_loose,
            "n_twin_basin_strict": n_twin,
            "n_truth_basin_strict_unique_clusters": n_strict_unique,
            "n_competing_basins_below_mse_0_5": n_competing,
            "min_q0_err_deg": float(min(r["q0_err_deg"] for r in runs)),
            "min_final_mse": float(min(r["final_mse"] for r in runs)),
            "argmin_mse_q0_err_deg": float(argmin_run["q0_err_deg"]),
            "argmin_mse_omega_dir_err_deg": float(argmin_run["omega_dir_err_deg"]),
            "argmin_mse_omega_mag_err_pct": float(argmin_run["omega_mag_err_pct"]),
            "median_wall_s": float(np.median([r["wall_s"] for r in runs])),
            "p90_wall_s": float(np.percentile([r["wall_s"] for r in runs], 90)),
            "truth_mse_ref": TRUTH_MSE_REF.get(seed),
        }

    n_in_basin = sum(1 for s in seeds
                     if per_seed[f"seed_{s:03d}"]["n_truth_basin_strict"] >= 1)
    cohort_pct = 100.0 * n_in_basin / len(seeds)
    if n_in_basin >= 6:
        decision = (f"Joint search at N={n} viable: {n_in_basin}/{len(seeds)} "
                    f"seeds in-basin. Cohort scan next at N=128-256.")
    elif n_in_basin >= 4:
        decision = (f"{n_in_basin}/{len(seeds)} seeds in-basin. Architecture works "
                    f"on easy seeds; characterise the gap before scaling.")
    else:
        decision = (f"Only {n_in_basin}/{len(seeds)} seeds in-basin at N={n}. "
                    f"Joint search insufficient; architecture revision needed.")

    return {
        "seeds": seeds,
        "n_per_seed": n,
        "max_nfev": MAX_NFEV,
        "sobol_seeds": {"q0": SOBOL_SEED_Q0, "wdir": SOBOL_SEED_WDIR,
                        "wmag": SOBOL_SEED_WMAG},
        "omega_mag_bounds_dps": [W_MAG_MIN_DPS, W_MAG_MAX_DPS],
        "per_seed": per_seed,
        "cohort_in_basin_count": n_in_basin,
        "cohort_in_basin_pct": cohort_pct,
        "decision": decision,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: full pilot list).")
    parser.add_argument("--n", type=int, default=N_DEFAULT,
                        help=f"ICs per seed (default {N_DEFAULT}).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 1 seed (6) at N=512 with output to "
                             "results/s015_smoke/.")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Append to output dir name (e.g. '_smoke').")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS_FULL
    if args.smoke:
        seeds = [6]
        out_dir = SURVEY_DIR / "results" / "s015_smoke"
    elif args.out_suffix:
        out_dir = SURVEY_DIR / "results" / f"s015{args.out_suffix}"
    else:
        out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n

    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    seed_data = collect_seed_data(seeds)
    all_ics = build_ics(seeds, n)

    n_total = len(all_ics)
    expected_wall = n_total * 35.0 / N_WORKERS
    print(f"s015: {len(seeds)} seeds × {n} ICs = {n_total} LM runs.", flush=True)
    print(f"  seeds: {seeds}", flush=True)
    print(f"  Pool({N_WORKERS}), BLAS=1, max_nfev={MAX_NFEV}.", flush=True)
    print(f"  ω-mag bounds: [{W_MAG_MIN_DPS}, {W_MAG_MAX_DPS}] dps", flush=True)
    print(f"  Sobol seeds: q0={SOBOL_SEED_Q0}, wdir={SOBOL_SEED_WDIR}, "
          f"wmag={SOBOL_SEED_WMAG}", flush=True)
    print(f"  Expected wall ≈ {expected_wall:.0f} s ≈ {expected_wall/60:.1f} min.",
          flush=True)
    print(f"  Out dir: {out_dir}", flush=True)

    args_list = [(ic, MAX_NFEV) for ic in all_ics]

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
                print(f"  [{done:5d}/{n_total}] rate={rate:.1f}/s, eta={eta:.0f} s",
                      flush=True)
                last_print = now
    wall = time.time() - t0
    print(f"\nAll runs done. Total wall: {wall:.1f} s = {wall/60:.1f} min.\n",
          flush=True)

    save_npz(results, out_dir)
    summary = summarise(results, seeds, n)
    save_summary(summary, out_dir)
    save_plots(results, summary, seeds, out_dir)

    print("\n=== s015 per-seed yield ===")
    print(f"  {'seed':>5s} {'role':>22s} {'in-basin':>9s} {'unique':>7s} "
          f"{'comp':>5s} {'min_q0_err':>11s} {'min_mse':>10s}")
    for s in seeds:
        cell = summary["per_seed"][f"seed_{s:03d}"]
        print(f"  {s:5d} {cell['role']:>22s} {cell['n_truth_basin_strict']:9d} "
              f"{cell['n_truth_basin_strict_unique_clusters']:7d} "
              f"{cell['n_competing_basins_below_mse_0_5']:5d} "
              f"{cell['min_q0_err_deg']:11.3f} "
              f"{cell['min_final_mse']:10.3e}")
    print(f"\n  Cohort in-basin: {summary['cohort_in_basin_count']}/{len(seeds)} "
          f"({summary['cohort_in_basin_pct']:.1f}%)")
    print(f"  → DECISION: {summary['decision']}")
    print(f"\nTotal wall: {time.time() - t_main:.1f} s")


def save_npz(results: list[dict], out_dir: Path):
    arrs = {
        "seed": np.array([r["seed"] for r in results], dtype=int),
        "ic_idx": np.array([r["ic_idx"] for r in results], dtype=int),
        "initial_q0_err_deg": np.array([r["initial_q0_err_deg"] for r in results]),
        "initial_omega_dir_err_deg":
            np.array([r["initial_omega_dir_err_deg"] for r in results]),
        "initial_omega_mag_err_pct":
            np.array([r["initial_omega_mag_err_pct"] for r in results]),
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
        "truth_basin_loose":  np.array([r["truth_basin_loose"] for r in results]),
        "twin_basin_strict":  np.array([r["twin_basin_strict"] for r in results]),
        "n_fev": np.array([r["n_fev"] for r in results], dtype=int),
        "wall_s": np.array([r["wall_s"] for r in results]),
        "success": np.array([r["success"] for r in results]),
        "status": np.array([r["status"] for r in results], dtype=int),
    }
    out = out_dir / "runs.npz"
    np.savez(out, **arrs)
    print(f"Saved: {out}")


def save_summary(summary: dict, out_dir: Path):
    out = out_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


def save_plots(results: list[dict], summary: dict, seeds: list[int],
               out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── yield.png ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(seeds))
    labels = [f"{s}\n{SEED_ROLE.get(s, '?')[:8]}" for s in seeds]

    ax = axes[0]
    in_basin = [summary["per_seed"][f"seed_{s:03d}"]["n_truth_basin_strict"]
                for s in seeds]
    ax.bar(x, in_basin, color="seagreen", alpha=0.85)
    for i, v in enumerate(in_basin):
        ax.text(i, v + 0.5, str(v), ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"# ICs in truth basin (strict, of {summary['n_per_seed']})")
    ax.set_title("s015 — per-seed in-basin landings")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    competing = [summary["per_seed"][f"seed_{s:03d}"]["n_competing_basins_below_mse_0_5"]
                 for s in seeds]
    ax.bar(x, competing, color="indianred", alpha=0.85)
    for i, v in enumerate(competing):
        ax.text(i, v + 0.2, str(v), ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("# competing basins (≥30° from truth, mse<0.5)")
    ax.set_title("s015 — competing basins per seed")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    min_q0_err = [summary["per_seed"][f"seed_{s:03d}"]["min_q0_err_deg"]
                  for s in seeds]
    ax.bar(x, min_q0_err, color="steelblue", alpha=0.85)
    ax.axhline(STRICT_Q0, color="green", linestyle="--", alpha=0.5,
               label=f"strict basin = {STRICT_Q0}°")
    ax.axhline(LOOSE_Q0, color="olive", linestyle="--", alpha=0.5,
               label=f"loose basin = {LOOSE_Q0}°")
    for i, v in enumerate(min_q0_err):
        ax.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("min q0_err_deg across all ICs")
    ax.set_title("s015 — best q0 recovery per seed")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = out_dir / "yield.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── q0_err_cdf.png ──
    fig, ax = plt.subplots(figsize=(10, 6))
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r["q0_err_deg"])
    for s in seeds:
        errs = sorted(by_seed[s])
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, label=f"seed {s} ({SEED_ROLE.get(s, '?')})",
                linewidth=1.5)
    ax.axvline(STRICT_Q0, color="green", linestyle="--", alpha=0.5,
               label=f"strict = {STRICT_Q0}°")
    ax.axvline(LOOSE_Q0, color="olive", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("final q0_err_deg")
    ax.set_ylabel("CDF over ICs")
    ax.set_title("s015 — final q0_err CDF per seed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out_dir / "q0_err_cdf.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── omega_recovery.png ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in seeds:
        runs_s = [r for r in results if r["seed"] == s]
        wd = [r["omega_dir_err_deg"] for r in runs_s]
        wm = [abs(r["omega_mag_err_pct"]) for r in runs_s]
        ax.scatter(wd, wm, label=f"seed {s}", s=8, alpha=0.5)
    ax.axhline(STRICT_OM_PCT, color="green", linestyle="--", alpha=0.5)
    ax.axvline(STRICT_OD, color="green", linestyle="--", alpha=0.5,
               label="strict basin")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ω_dir_err [deg]")
    ax.set_ylabel("|ω_mag_err| [%]")
    ax.set_title("s015 — ω recovery per IC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out_dir / "omega_recovery.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
