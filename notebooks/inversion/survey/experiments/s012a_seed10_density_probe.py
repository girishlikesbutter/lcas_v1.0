"""s012a — seed-10 single-seed N=256 probe (Q4c-ii diagnostic).

s011 cohort pilot found cohort yield 9/10 at N=64 with seed 10 = 0/64. s011
identified 49 distinct competing-basin clusters on seed 10 (q0_err 28°-180°,
all final_mse < 0.5 mag²; 5/64 with final_mse < 0.05 same order as truth_mse_ref
3.59e-3, but none near truth). s009 (body-X-axis only) reported seed 10 as T1
strict pass — multi-axis Sobol exposes that body-X is a special axis, not
representative of generic SO(3) ICs.

This is the diagnostic that decides whether seed-10 failure is:
  H1 — N-recoverable: basin exists but is sub-N=64-resolution. Extending Sobol
       to N=256 hits ≥1 in-basin candidate.
  H2 — architectural: 0/256 even at 4× density. Either (a) basin volume too
       small for any feasible Sobol density on SO(3), or (b) generic SO(3) ICs
       cannot LM-funnel into the truth basin (LM-grab radius is small for
       seed 10 specifically). Either way, basin-hopping or a different
       architectural class is needed for seed-10-class seeds.

Method:
  - Seed 10 only.
  - Extend s011's N=64 by 192 new ICs to reach N=256 (Sobol fast-forward
    past first 64 → indices 64..255). Same SOBOL_SEED=42, same scrambled
    Sobol-Shoemake parameterization. Verified scipy fast_forward semantics:
    fast_forward(64) + random(192) == random(256)[64:].
  - Same LM polish: scipy least_squares(method='lm', max_nfev=60). Same
    6-DOF parameterization (δθ tangent + ω-3-vec) as s005/s011.
  - ω fixed at truth-ω.
  - Pool(8), BLAS=1, torch threads=1 (s011 critical-fix path).

Reporting (per-IC):
  - q0_err, twin_err, ω_dir_err, ω_mag_err_pct, final_mse, n_fev, wall_s
  - in-basin flags (strict 5°/1°/5%, loose 10°/2°/10%), twin-basin flag

Per-N analysis (cumulative N=64 / 128 / 256, since Sobol is sequential):
  - n_in_basin_strict per N (yield curve)
  - n_competing_basins (≥30° from truth, final_mse < 0.5 mag²) per N
    — does it saturate or grow?
  - min_q0_err per N (closest LM landing per N — does adding ICs reduce it?)
  - min_final_mse_at_q0_below_30deg per N (best near-truth final MSE — gives
    a "did the basin even get seeded" measurement)

Decision criteria:
  - n_in_basin_strict_N256 ≥ 1 → H1 (N-recoverable). DECISION: bump cohort
    Sobol density to N=128 or 256 on failing seeds during s012 cohort scan.
  - n_in_basin_strict_N256 = 0 AND min_q0_err_N256 > 5° → H2 (architectural).
    DECISION: basin-hopping or partial-cohort acceptance for seed-10-class.
  - Edge case: 0/256 in-basin but min_q0_err_N256 < 10° → near-miss; basin
    seeded but LM stalls just outside. Suggests Q4c-ii needs LM tuning
    (longer max_nfev or trust-region method) for narrow / oddly-shaped
    basins, not pure architectural failure.

Compute budget:
  - 192 new ICs × ~26 s/run / 8 workers ≈ 624 s ≈ 10.4 min wall.
  - Kill criterion: 25 min.

Outputs:
  - results/s012a/runs.npz             192 new IC results (full state)
  - results/s012a/runs_combined.npz    256 IC results (s011 first 64 + new 192)
  - results/s012a/summary.json         per-N yield + decision
  - results/s012a/yield_curve.png      cumulative yield + competing basins vs N
  - results/s012a/q0_err_distribution.png  CDF of q0_err at N=64/128/256
"""

# BLAS=1 BEFORE Pool fork.
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
S011_RUNS = SURVEY_DIR / "results" / "s011" / "runs.npz"
OUT_DIR = SURVEY_DIR / "results" / "s012a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 10
N_EXISTING = 64    # already in s011
N_TARGET = 256     # cumulative after s012a
N_NEW = N_TARGET - N_EXISTING  # 192 new ICs at Sobol indices 64..255
SOBOL_SEED = 42    # matches s002 / s006 / s010 / s011
N_WORKERS = 8
MAX_NFEV = 60      # matches s011

# Per-N reporting checkpoints (cumulative).
N_CHECKPOINTS = [64, 128, 256]

# In-basin definitions (match s005 / s009 / s011).
STRICT_Q0 = 5.0
STRICT_OD = 1.0
STRICT_OM_PCT = 5.0
LOOSE_Q0 = 10.0
LOOSE_OD = 2.0
LOOSE_OM_PCT = 10.0

# From s001 cache.
TRUTH_MSE_REF = 3.59353e-3

# Twin (180° body-X).
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


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


def build_sobol_q0_extension(n_skip: int, n_new: int, sobol_seed: int) -> np.ndarray:
    """Sobol-Shoemake quaternions at indices [n_skip, n_skip + n_new), continuing
    the same scrambled sequence used in s011. Verified: scipy Sobol with
    scramble=True + seed is deterministic, and fast_forward(n) + random(m)
    yields the same points as random(n+m)[n:]."""
    sobol = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    sobol.fast_forward(n_skip)
    u = sobol.random(n_new)
    return shoemake_to_quat(u)


# ────────────────────────────────────────────────────────────────────────────
# Worker (cloned from s011; single seed simplifies state).
# ────────────────────────────────────────────────────────────────────────────

_W_TIMES = None
_W_SUN = None
_W_OBS = None
_W_SAT = None
_W_OBS_DIST = None
_W_MAG_HIFI = None
_W_INERTIA = None


def init_worker(seed_data: dict, inertia_tensor: np.ndarray):
    # torch threads MUST be pinned in the worker (s011 finding).
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    global _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_OBS_DIST, _W_MAG_HIFI, _W_INERTIA
    _W_INERTIA = inertia_tensor
    _W_TIMES = seed_data["observation_times"]
    _W_SUN = seed_data["sun_pos"]
    _W_OBS = seed_data["obs_pos"]
    _W_SAT = seed_data["sat_pos"]
    _W_OBS_DIST = seed_data["obs_dist"]
    _W_MAG_HIFI = seed_data["mag_hifi"]
    surrogate_eval.get_model()


def make_residual_fn(q0_seed_wxyz: np.ndarray):
    times = _W_TIMES
    sun = _W_SUN
    obs = _W_OBS
    sat = _W_SAT
    obs_dist = _W_OBS_DIST
    mag_truth = _W_MAG_HIFI
    inertia = _W_INERTIA
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
    ic_idx = int(ic_dict["ic_idx"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    omega_seed_rad = np.asarray(ic_dict["omega_seed_rad"], dtype=float)
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    residual_fn = make_residual_fn(q0_seed_wxyz)
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
# Combine s011 N=64 + s012a N=192 into a 256-IC array.
# ────────────────────────────────────────────────────────────────────────────


def load_s011_seed10_first64() -> dict:
    """Pull the first 64 ICs for seed 10 from the s011 npz, sorted by ic_idx."""
    d = np.load(str(S011_RUNS))
    mask = d["seed"] == SEED
    if mask.sum() != 64:
        raise RuntimeError(f"expected 64 seed-10 runs in s011, got {mask.sum()}")
    order = np.argsort(d["ic_idx"][mask])
    keys = [
        "ic_idx", "initial_q0_err_deg", "initial_mse", "q0_seed_wxyz",
        "q0_final_wxyz", "omega_final_rad", "q0_err_deg", "twin_err_deg",
        "omega_dir_err_deg", "omega_mag_err_pct", "final_mse",
        "truth_basin_strict", "truth_basin_loose", "twin_basin_strict",
        "n_fev", "wall_s", "success", "status",
    ]
    out = {}
    for k in keys:
        v = d[k][mask]
        out[k] = v[order]
    return out


def cluster_q0_landings(q0_finals: np.ndarray, cluster_deg: float) -> int:
    """Greedy 1D geodesic clustering."""
    if len(q0_finals) == 0:
        return 0
    clusters = [q0_finals[0]]
    for q in q0_finals[1:]:
        if all(quat_geodesic_deg(q, c) > cluster_deg for c in clusters):
            clusters.append(q)
    return len(clusters)


def per_n_summary(combined: dict, n: int) -> dict:
    """Summary across the first n ICs (cumulative)."""
    sl = slice(0, n)
    q0_err = combined["q0_err_deg"][sl]
    final_mse = combined["final_mse"][sl]
    omega_dir_err = combined["omega_dir_err_deg"][sl]
    omega_mag_err_pct = combined["omega_mag_err_pct"][sl]
    twin_err = combined["twin_err_deg"][sl]
    q0_final = combined["q0_final_wxyz"][sl]

    in_basin_strict = (q0_err < STRICT_Q0) & (omega_dir_err < STRICT_OD) \
        & (np.abs(omega_mag_err_pct) < STRICT_OM_PCT)
    in_basin_loose = (q0_err < LOOSE_Q0) & (omega_dir_err < LOOSE_OD) \
        & (np.abs(omega_mag_err_pct) < LOOSE_OM_PCT)
    twin_strict = (twin_err < STRICT_Q0) & (omega_dir_err < STRICT_OD) \
        & (np.abs(omega_mag_err_pct) < STRICT_OM_PCT)

    n_in_strict = int(in_basin_strict.sum())
    n_in_strict_unique = cluster_q0_landings(q0_final[in_basin_strict], 1.0)

    competing_mask = (q0_err >= 30.0) & (final_mse < 0.5)
    competing_q0 = q0_final[competing_mask]
    n_competing = cluster_q0_landings(competing_q0, 5.0)

    near_mask = q0_err < 30.0
    if near_mask.sum() > 0:
        min_mse_near = float(final_mse[near_mask].min())
        min_q0_at_near = float(q0_err[near_mask].min())
    else:
        min_mse_near = float("nan")
        min_q0_at_near = float("nan")

    return {
        "n_ics": int(n),
        "n_truth_basin_strict": n_in_strict,
        "n_truth_basin_loose": int(in_basin_loose.sum()),
        "n_twin_basin_strict": int(twin_strict.sum()),
        "n_truth_basin_strict_unique_clusters": n_in_strict_unique,
        "n_competing_basins_below_mse_0_5": n_competing,
        "min_q0_err_deg": float(q0_err.min()),
        "min_q0_err_within_30deg": min_q0_at_near,
        "min_final_mse": float(final_mse.min()),
        "min_final_mse_within_30deg": min_mse_near,
        "frac_final_mse_below_0_5": float(np.mean(final_mse < 0.5)),
        "frac_final_mse_below_truth": float(np.mean(final_mse < TRUTH_MSE_REF)),
    }


def decide(per_n: dict) -> str:
    n256 = per_n["N256"]
    if n256["n_truth_basin_strict"] >= 1:
        return ("H1 confirmed: seed-10 basin reached at N=256 with "
                f"{n256['n_truth_basin_strict']}/256 in-basin landings. Increase "
                "Sobol density on failing seeds in s012 cohort scan.")
    if n256["min_q0_err_deg"] < LOOSE_Q0:
        return (f"Edge case: 0/256 strict in-basin but min_q0_err = "
                f"{n256['min_q0_err_deg']:.2f}° < {LOOSE_Q0}°. Basin seeded; LM "
                "stalls just outside. Suggests max_nfev or method tuning, not "
                "pure architectural failure.")
    return ("H2 confirmed: 0/256 in-basin AND min_q0_err = "
            f"{n256['min_q0_err_deg']:.2f}° far above basin. Either basin "
            "volume too small for SO(3) Sobol at feasible density, OR "
            "LM-grab radius is small for seed-10 specifically. Cohort needs "
            "basin-hopping or partial-cohort acceptance for seed-10-class.")


# ────────────────────────────────────────────────────────────────────────────
# Output
# ────────────────────────────────────────────────────────────────────────────


def save_runs_npz(new_results: list[dict], path: Path):
    arrs = {
        "ic_idx": np.array([r["ic_idx"] for r in new_results], dtype=int),
        "initial_q0_err_deg": np.array([r["initial_q0_err_deg"] for r in new_results]),
        "initial_mse": np.array([r["initial_mse"] for r in new_results]),
        "q0_seed_wxyz": np.stack([r["q0_seed_wxyz"] for r in new_results]),
        "q0_final_wxyz": np.stack([r["q0_final_wxyz"] for r in new_results]),
        "omega_final_rad": np.stack([r["omega_final_rad"] for r in new_results]),
        "q0_err_deg": np.array([r["q0_err_deg"] for r in new_results]),
        "twin_err_deg": np.array([r["twin_err_deg"] for r in new_results]),
        "omega_dir_err_deg": np.array([r["omega_dir_err_deg"] for r in new_results]),
        "omega_mag_err_pct": np.array([r["omega_mag_err_pct"] for r in new_results]),
        "final_mse": np.array([r["final_mse"] for r in new_results]),
        "truth_basin_strict": np.array([r["truth_basin_strict"] for r in new_results]),
        "truth_basin_loose": np.array([r["truth_basin_loose"] for r in new_results]),
        "twin_basin_strict": np.array([r["twin_basin_strict"] for r in new_results]),
        "n_fev": np.array([r["n_fev"] for r in new_results], dtype=int),
        "wall_s": np.array([r["wall_s"] for r in new_results]),
        "success": np.array([r["success"] for r in new_results]),
        "status": np.array([r["status"] for r in new_results], dtype=int),
    }
    np.savez(path, **arrs)
    print(f"Saved: {path}")


def save_combined_npz(combined: dict, path: Path):
    np.savez(path, **{k: v for k, v in combined.items() if k != "_n_existing"})
    print(f"Saved: {path}")


def save_summary(summary: dict, path: Path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")


def save_yield_curve(combined: dict, summary: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Yield curve at every cumulative N from 1..256.
    n_grid = np.arange(1, N_TARGET + 1)
    in_basin = combined["truth_basin_strict"][:N_TARGET].astype(int)
    cum_in_basin = np.cumsum(in_basin)
    cum_competing = np.zeros_like(n_grid, dtype=int)
    q0_finals = combined["q0_final_wxyz"][:N_TARGET]
    q0_errs = combined["q0_err_deg"][:N_TARGET]
    final_mses = combined["final_mse"][:N_TARGET]
    competing_idx = (q0_errs >= 30.0) & (final_mses < 0.5)
    cum_competing_mask = np.cumsum(competing_idx)

    competing_clusters_at_n = []
    for n in n_grid:
        mask = competing_idx[:n]
        if mask.sum() == 0:
            competing_clusters_at_n.append(0)
        else:
            competing_clusters_at_n.append(cluster_q0_landings(q0_finals[:n][mask], 5.0))
    competing_clusters_at_n = np.array(competing_clusters_at_n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(n_grid, cum_in_basin, color="seagreen", linewidth=2, label="cumulative in-basin (strict)")
    for n in N_CHECKPOINTS:
        ax.axvline(n, color="grey", linestyle="--", alpha=0.4)
        ax.text(n, ax.get_ylim()[1] * 0.05, f"N={n}", ha="center", fontsize=8)
    ax.set_xlabel("Sobol candidates evaluated (cumulative N)")
    ax.set_ylabel("# in-basin landings (strict 5°/1°/5%)")
    ax.set_title(f"s012a — seed {SEED}: in-basin yield vs Sobol density")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(n_grid, cum_competing_mask, color="firebrick", linewidth=2,
            label="cumulative competing landings (≥30°, mse<0.5)")
    ax.plot(n_grid, competing_clusters_at_n, color="darkorange", linewidth=2,
            label="distinct competing-basin clusters")
    for n in N_CHECKPOINTS:
        ax.axvline(n, color="grey", linestyle="--", alpha=0.4)
    ax.set_xlabel("Sobol candidates evaluated (cumulative N)")
    ax.set_ylabel("count")
    ax.set_title(f"s012a — seed {SEED}: competing-basin growth vs density")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


def save_q0_err_distribution(combined: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in N_CHECKPOINTS:
        errs = np.sort(combined["q0_err_deg"][:n])
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, linewidth=2, label=f"N={n}")
    ax.axvline(STRICT_Q0, color="green", linestyle="--", alpha=0.5,
               label=f"strict basin = {STRICT_Q0}°")
    ax.axvline(LOOSE_Q0, color="olive", linestyle="--", alpha=0.5,
               label=f"loose basin = {LOOSE_Q0}°")
    ax.set_xscale("log")
    ax.set_xlabel("final q0 error after LM polish  [deg]")
    ax.set_ylabel("CDF over IC")
    ax.set_title(f"s012a — seed {SEED}: q0-error CDF, cumulative N")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    t_main = time.time()

    if not S011_RUNS.exists():
        raise RuntimeError(f"missing s011 runs.npz at {S011_RUNS}")

    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    # Truth state for seed 10.
    truth = traj_load.load_truth(SEED)
    q0_truth = np.asarray(truth["q0_wxyz"], dtype=float)
    omega_truth = np.asarray(truth["omega0_rad"], dtype=float)
    omega_mag = float(np.linalg.norm(omega_truth))
    omega_dir = omega_truth / omega_mag

    seed_data = {
        "observation_times": truth["observation_times"],
        "sun_pos": truth["sun_pos"],
        "obs_pos": truth["obs_pos"],
        "sat_pos": truth["sat_pos"],
        "obs_dist": truth["obs_dist"],
        "mag_hifi": truth["mag_hifi"],
    }

    # Verify s011 and s012a use compatible Sobol points: re-build first-64 and
    # compare to s011's stored q0_seed_wxyz.
    s011_first64 = load_s011_seed10_first64()
    sobol_check = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u_first64 = sobol_check.random(64)
    q0_first64_recomputed = shoemake_to_quat(u_first64)
    consistency = np.allclose(s011_first64["q0_seed_wxyz"], q0_first64_recomputed)
    print(f"s011 first-64 Sobol consistency check: {consistency}", flush=True)
    if not consistency:
        raise RuntimeError("s011 first-64 Sobol points do NOT match recomputed "
                           "Sobol(seed=42, scramble=True).random(64). Refuse to "
                           "extend without consistent IC space.")

    # Build new ICs at indices 64..255.
    q0_set_new = build_sobol_q0_extension(N_EXISTING, N_NEW, SOBOL_SEED)
    new_ics = []
    for i in range(N_NEW):
        new_ics.append({
            "ic_idx": N_EXISTING + i,
            "q0_seed_wxyz": q0_set_new[i],
            "omega_seed_rad": omega_truth.copy(),
            "q0_truth_wxyz": q0_truth.copy(),
            "omega_truth_dir": omega_dir.copy(),
            "omega_truth_mag": omega_mag,
        })

    expected_wall = N_NEW * 26.0 / N_WORKERS
    print(f"s012a: seed {SEED}, {N_NEW} new LM runs at Sobol indices "
          f"[{N_EXISTING}, {N_TARGET}). expected wall ≈ {expected_wall:.0f} s.",
          flush=True)
    print(f"Pool({N_WORKERS}), BLAS=1, SOBOL_SEED={SOBOL_SEED}, max_nfev={MAX_NFEV}.",
          flush=True)

    args_list = [(ic, MAX_NFEV) for ic in new_ics]
    t0 = time.time()
    new_results = []
    last_print = t0
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_lm_for_ic, args_list, chunksize=4)):
            new_results.append(r)
            now = time.time()
            if now - last_print > 30.0 or i + 1 == N_NEW:
                done = i + 1
                rate = done / (now - t0) if now > t0 else 0.0
                eta = (N_NEW - done) / rate if rate > 0 else 0.0
                print(f"  [{done:3d}/{N_NEW}] rate={rate:.1f}/s, eta={eta:.0f} s",
                      flush=True)
                last_print = now
    wall = time.time() - t0
    print(f"\nNew runs done. Total wall: {wall:.1f} s.\n", flush=True)

    # Sort new_results by ic_idx (Pool returns unordered).
    new_results = sorted(new_results, key=lambda r: r["ic_idx"])

    save_runs_npz(new_results, OUT_DIR / "runs.npz")

    # Combine s011 first-64 + s012a 192 → 256-IC array, sorted by ic_idx.
    combined = {
        "ic_idx": np.concatenate([
            s011_first64["ic_idx"],
            np.array([r["ic_idx"] for r in new_results], dtype=int),
        ]),
        "initial_q0_err_deg": np.concatenate([
            s011_first64["initial_q0_err_deg"],
            np.array([r["initial_q0_err_deg"] for r in new_results]),
        ]),
        "initial_mse": np.concatenate([
            s011_first64["initial_mse"],
            np.array([r["initial_mse"] for r in new_results]),
        ]),
        "q0_seed_wxyz": np.concatenate([
            s011_first64["q0_seed_wxyz"],
            np.stack([r["q0_seed_wxyz"] for r in new_results]),
        ], axis=0),
        "q0_final_wxyz": np.concatenate([
            s011_first64["q0_final_wxyz"],
            np.stack([r["q0_final_wxyz"] for r in new_results]),
        ], axis=0),
        "omega_final_rad": np.concatenate([
            s011_first64["omega_final_rad"],
            np.stack([r["omega_final_rad"] for r in new_results]),
        ], axis=0),
        "q0_err_deg": np.concatenate([
            s011_first64["q0_err_deg"],
            np.array([r["q0_err_deg"] for r in new_results]),
        ]),
        "twin_err_deg": np.concatenate([
            s011_first64["twin_err_deg"],
            np.array([r["twin_err_deg"] for r in new_results]),
        ]),
        "omega_dir_err_deg": np.concatenate([
            s011_first64["omega_dir_err_deg"],
            np.array([r["omega_dir_err_deg"] for r in new_results]),
        ]),
        "omega_mag_err_pct": np.concatenate([
            s011_first64["omega_mag_err_pct"],
            np.array([r["omega_mag_err_pct"] for r in new_results]),
        ]),
        "final_mse": np.concatenate([
            s011_first64["final_mse"],
            np.array([r["final_mse"] for r in new_results]),
        ]),
        "truth_basin_strict": np.concatenate([
            s011_first64["truth_basin_strict"],
            np.array([r["truth_basin_strict"] for r in new_results]),
        ]),
        "truth_basin_loose": np.concatenate([
            s011_first64["truth_basin_loose"],
            np.array([r["truth_basin_loose"] for r in new_results]),
        ]),
        "twin_basin_strict": np.concatenate([
            s011_first64["twin_basin_strict"],
            np.array([r["twin_basin_strict"] for r in new_results]),
        ]),
        "n_fev": np.concatenate([
            s011_first64["n_fev"],
            np.array([r["n_fev"] for r in new_results], dtype=int),
        ]),
        "wall_s": np.concatenate([
            s011_first64["wall_s"],
            np.array([r["wall_s"] for r in new_results]),
        ]),
        "success": np.concatenate([
            s011_first64["success"],
            np.array([r["success"] for r in new_results]),
        ]),
        "status": np.concatenate([
            s011_first64["status"],
            np.array([r["status"] for r in new_results], dtype=int),
        ]),
    }
    save_combined_npz(combined, OUT_DIR / "runs_combined.npz")

    per_n = {f"N{n}": per_n_summary(combined, n) for n in N_CHECKPOINTS}
    summary = {
        "seed": SEED,
        "sobol_seed": SOBOL_SEED,
        "n_existing_from_s011": N_EXISTING,
        "n_new_in_s012a": N_NEW,
        "n_total": N_TARGET,
        "max_nfev": MAX_NFEV,
        "truth_mse_ref": TRUTH_MSE_REF,
        "per_n": per_n,
        "decision": decide(per_n),
    }
    save_summary(summary, OUT_DIR / "summary.json")

    save_yield_curve(combined, summary, OUT_DIR / "yield_curve.png")
    save_q0_err_distribution(combined, OUT_DIR / "q0_err_distribution.png")

    # Console report.
    print(f"\n=== s012a per-N yield (seed {SEED}) ===")
    print(f"  {'N':>5s} {'in-basin':>9s} {'unique':>7s} {'comp':>5s} "
          f"{'min_q0':>8s} {'min_q0<30':>10s} {'min_mse':>10s} {'min_mse<30':>11s}")
    for n in N_CHECKPOINTS:
        s = per_n[f"N{n}"]
        mq = s.get("min_q0_err_within_30deg", float("nan"))
        mm = s.get("min_final_mse_within_30deg", float("nan"))
        print(f"  {n:5d} {s['n_truth_basin_strict']:9d} "
              f"{s['n_truth_basin_strict_unique_clusters']:7d} "
              f"{s['n_competing_basins_below_mse_0_5']:5d} "
              f"{s['min_q0_err_deg']:8.2f} {mq:10.2f} "
              f"{s['min_final_mse']:10.3e} {mm:11.3e}")
    print(f"\n  → DECISION: {summary['decision']}")
    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


if __name__ == "__main__":
    main()
