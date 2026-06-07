"""s002 — surrogate landscape probe (PA-stratified seeds, Sobol-SO(3) grid).

Survey question Q2. Does the surrogate full-LC MSE landscape's argmin sit at
truth-q0 (or its antipode), or somewhere structurally else? s001 established
that surrogate-MSE-at-truth is small everywhere (Band-A on all 100 seeds),
but cannot tell us whether truth is the global minimum or merely a local
small value with a deeper basin elsewhere.

Decision tree:
  - argmin at truth-q0 (or antipode q_180x · q0) on most seeds:
      m145's "deceptive q0=135° attractor on seed 91" is seed-specific;
      surrogate-MSE is structurally honest and any solver failure is solver-
      side (DE escape, polish from input, etc.).
  - argmin away from truth on a non-trivial fraction:
      surrogate-MSE is structurally deceptive under correct truth; need
      surrogate correction or a different cost substrate.

Method:
  - Pick 8 PA-stratified seeds: anchors 6 (m141), 10 (s001 max-MSE), 91
    (m145), plus PA-stratified 21, 41, 48, 60, 84.
  - Generate 2046 Sobol-Shoemake quaternions for low-discrepancy SO(3)
    coverage; insert truth-q0 at index 0 and twin (q_180x · q0_truth) at
    index 1 → 2048 candidates total.
  - For each candidate q0_c, propagate (q0_c, ω_truth) under post-fix
    propagator using cached observation_times + sun/obs/sat positions.
  - Score surrogate full-LC MSE vs cached mag_hifi.
  - Save per-seed candidate arrays + summary stats + plots.

Pool(8) per-seed since per-candidate cost is ~89 ms (28 ms propagation +
56 ms surrogate predict). Per project memory: BLAS threads = 1 in workers.

Outputs:
  - results/s002/per_seed_landscape.npz   all candidate quats + MSEs per seed
  - results/s002/summary.json             per-seed argmin / truth-rank stats
  - results/s002/landscape_grid.png       MSE vs q0_err panel (one per seed)
"""

# BLAS=1 BEFORE Pool fork (per project memory).
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
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg_batch  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s002"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [6, 10, 21, 41, 48, 60, 84, 91]
N_SOBOL = 2046  # +2 for truth-q0 and twin → 2048 total candidates
SOBOL_SEED = 42
N_WORKERS = 8


# Twin: q_180x rotation about body x (the standard convention from
# `concepts/twin_degeneracy.md`: LEFT-multiply q_180x · q0; same omega).
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])  # 180° about body X


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    """Shoemake's uniform-on-S^3 mapping: (N, 3) Sobol → (N, 4) quat (wxyz).

    Reference: Shoemake, "Uniform Random Rotations" (Graphics Gems III).
    Given u1, u2, u3 ∈ [0, 1):
      x = sqrt(1 - u1) sin(2π u2)
      y = sqrt(1 - u1) cos(2π u2)
      z = sqrt(u1)     sin(2π u3)
      w = sqrt(u1)     cos(2π u3)
    Returns wxyz order (scalar-first) to match the propagator.
    """
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


def build_grid(q0_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truth + twin + 2046 Sobol-Shoemake quats → (2048, 4) array.

    Returns:
      q_grid: (2048, 4) wxyz quaternions.
      grid_kind: (2048,) int array — 0=truth, 1=twin, 2=sobol.
    """
    sobol = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u = sobol.random(N_SOBOL)
    q_sobol = shoemake_to_quat(u)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    q_grid = np.vstack([q0_truth[None, :], q_twin[None, :], q_sobol])
    kind = np.concatenate([
        np.array([0], dtype=int),
        np.array([1], dtype=int),
        np.full(N_SOBOL, 2, dtype=int),
    ])
    return q_grid, kind


# Worker globals — populated in init_worker per process.
_W_OMEGA = None
_W_TIMES = None
_W_SUN = None
_W_OBS = None
_W_SAT = None
_W_INERTIA = None
_W_OBS_DIST = None
_W_MAG_HIFI = None


def init_worker(omega0_rad, observation_times, sun_pos, obs_pos, sat_pos,
                inertia_tensor, obs_dist, mag_hifi):
    """Pre-fork (technically per-fork) state initialiser.

    Loads the surrogate model lazily on first predict per worker. Stashes
    the per-seed arrays in process-globals so the work-fn can be pickled
    cheaply (only the candidate quaternion crosses the IPC boundary).
    """
    global _W_OMEGA, _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA, _W_OBS_DIST, _W_MAG_HIFI
    _W_OMEGA = omega0_rad
    _W_TIMES = observation_times
    _W_SUN = sun_pos
    _W_OBS = obs_pos
    _W_SAT = sat_pos
    _W_INERTIA = inertia_tensor
    _W_OBS_DIST = obs_dist
    _W_MAG_HIFI = mag_hifi
    surrogate_eval.get_model()  # warm cache in this worker


def score_candidate(q0_wxyz: np.ndarray) -> tuple[float, float]:
    """Run propagation + surrogate predict + MSE for one candidate q0.

    Returns:
      (full_lc_mse_vs_hifi, bright_mse_vs_hifi)
    """
    k1b, k2b, _ = propagate_to_body_frame(
        q0_wxyz, _W_OMEGA, _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA,
    )
    pred = surrogate_eval.predict(k1b, k2b, _W_OBS_DIST)
    full = surrogate_eval.full_lc_mse(pred, _W_MAG_HIFI)
    bright = surrogate_eval.bright_mse(pred, _W_MAG_HIFI, 11.0)
    return float(full), float(bright)


def run_seed(seed: int, inertia_tensor: np.ndarray) -> dict:
    """Run the landscape probe for one seed."""
    d = traj_load.load_truth(seed)
    q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
    omega0 = np.asarray(d["omega0_rad"], dtype=float)

    q_grid, kind = build_grid(q0_truth)
    n_cand = q_grid.shape[0]

    t0 = time.time()
    init_args = (
        omega0, d["observation_times"], d["sun_pos"], d["obs_pos"], d["sat_pos"],
        inertia_tensor, d["obs_dist"], d["mag_hifi"],
    )
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=init_args) as pool:
        # imap_unordered would scramble the candidate order; use map to
        # preserve q_grid index ↔ MSE alignment.
        results = pool.map(score_candidate, list(q_grid), chunksize=16)
    wall = time.time() - t0

    full_mse = np.array([r[0] for r in results], dtype=float)
    bright_mse = np.array([r[1] for r in results], dtype=float)

    geo_to_truth = quat_geodesic_deg_batch(q_grid, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    geo_to_twin = quat_geodesic_deg_batch(q_grid, q_twin)
    geo_to_truth_or_twin = np.minimum(geo_to_truth, geo_to_twin)

    argmin = int(np.argmin(full_mse))
    truth_mse = float(full_mse[0])
    twin_mse = float(full_mse[1])
    argmin_geo = float(geo_to_truth[argmin])
    argmin_geo_twin = float(geo_to_twin[argmin])

    print(
        f"seed {seed:3d}  N={n_cand}  "
        f"truth_mse={truth_mse:.4e}  twin_mse={twin_mse:.4e}  "
        f"argmin={argmin} ({'truth' if argmin == 0 else 'twin' if argmin == 1 else 'sobol'})  "
        f"argmin_mse={float(full_mse[argmin]):.4e}  "
        f"argmin_geo_to_truth={argmin_geo:.1f}°  "
        f"argmin_geo_to_twin={argmin_geo_twin:.1f}°  "
        f"wall={wall:.1f}s",
        flush=True,
    )

    return {
        "seed": int(seed),
        "n_candidates": int(n_cand),
        "q_grid": q_grid,
        "kind": kind,
        "full_mse": full_mse,
        "bright_mse": bright_mse,
        "geo_to_truth_deg": geo_to_truth,
        "geo_to_twin_deg": geo_to_twin,
        "geo_to_truth_or_twin_deg": geo_to_truth_or_twin,
        "argmin_idx": argmin,
        "argmin_full_mse": float(full_mse[argmin]),
        "argmin_geo_to_truth_deg": argmin_geo,
        "argmin_geo_to_twin_deg": argmin_geo_twin,
        "truth_full_mse": truth_mse,
        "twin_full_mse": twin_mse,
        "wall_s": wall,
    }


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    print(f"Running s002 surrogate-landscape probe on {len(SEEDS)} seeds "
          f"({N_SOBOL + 2} candidates each, Pool({N_WORKERS}), BLAS=1).", flush=True)

    per_seed = []
    for seed in SEEDS:
        per_seed.append(run_seed(seed, inertia_tensor))

    # ---- Save per-seed NPZ ----
    npz_path = OUT_DIR / "per_seed_landscape.npz"
    save_dict = {}
    for r in per_seed:
        s = r["seed"]
        save_dict[f"seed{s:03d}_q_grid"] = r["q_grid"]
        save_dict[f"seed{s:03d}_kind"] = r["kind"]
        save_dict[f"seed{s:03d}_full_mse"] = r["full_mse"]
        save_dict[f"seed{s:03d}_bright_mse"] = r["bright_mse"]
        save_dict[f"seed{s:03d}_geo_to_truth_deg"] = r["geo_to_truth_deg"]
        save_dict[f"seed{s:03d}_geo_to_twin_deg"] = r["geo_to_twin_deg"]
    save_dict["seeds"] = np.array(SEEDS, dtype=int)
    save_dict["n_sobol"] = N_SOBOL
    save_dict["sobol_seed"] = SOBOL_SEED
    np.savez(npz_path, **save_dict)
    print(f"Saved: {npz_path}")

    # ---- Save summary JSON ----
    summary = build_summary(per_seed)
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ---- Save landscape grid plot ----
    plot_path = OUT_DIR / "landscape_grid.png"
    save_landscape_grid(per_seed, plot_path)
    print(f"Saved: {plot_path}")

    print(f"\nTotal wall: {time.time() - t_main:.1f}s for {len(SEEDS)} seeds.")


def build_summary(per_seed):
    """One row per seed plus population-level decision stats."""
    rows = []
    for r in per_seed:
        # Truth's rank in the sorted-MSE list (0 = best). Compute on the
        # sobol subset so truth's hard-coded position at index 0 doesn't
        # bias things — i.e. "where does truth_mse fall vs all 2046 sobols?"
        sobol_mask = r["kind"] == 2
        sobol_full = r["full_mse"][sobol_mask]
        n_sobol_below = int(np.sum(sobol_full < r["truth_full_mse"]))
        n_sobol_below_twin = int(np.sum(sobol_full < r["twin_full_mse"]))
        rows.append({
            "seed": r["seed"],
            "n_candidates": r["n_candidates"],
            "truth_full_mse": r["truth_full_mse"],
            "twin_full_mse": r["twin_full_mse"],
            "argmin_idx": r["argmin_idx"],
            "argmin_kind": int(r["kind"][r["argmin_idx"]]),  # 0=truth 1=twin 2=sobol
            "argmin_full_mse": r["argmin_full_mse"],
            "argmin_geo_to_truth_deg": r["argmin_geo_to_truth_deg"],
            "argmin_geo_to_twin_deg": r["argmin_geo_to_twin_deg"],
            "n_sobol_below_truth": n_sobol_below,
            "n_sobol_below_twin": n_sobol_below_twin,
            "wall_s": r["wall_s"],
        })

    # Population-level decision: how many seeds have truth (or twin) as the global argmin?
    n_truth_global = sum(1 for r in rows if r["argmin_kind"] == 0)
    n_twin_global = sum(1 for r in rows if r["argmin_kind"] == 1)
    n_sobol_global = sum(1 for r in rows if r["argmin_kind"] == 2)
    n_truth_or_twin_within_5deg = sum(
        1 for r in rows
        if min(r["argmin_geo_to_truth_deg"], r["argmin_geo_to_twin_deg"]) < 5.0
    )

    return {
        "seeds": SEEDS,
        "n_sobol": N_SOBOL,
        "sobol_seed": SOBOL_SEED,
        "per_seed": rows,
        "n_seeds_argmin_at_truth": n_truth_global,
        "n_seeds_argmin_at_twin": n_twin_global,
        "n_seeds_argmin_elsewhere": n_sobol_global,
        "n_seeds_argmin_within_5deg_of_truth_or_twin": n_truth_or_twin_within_5deg,
    }


def save_landscape_grid(per_seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(per_seed)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for i, r in enumerate(per_seed):
        ax = axes[i]
        sobol_mask = r["kind"] == 2
        x = r["geo_to_truth_or_twin_deg"][sobol_mask]
        y = r["full_mse"][sobol_mask]
        ax.scatter(x, y, s=3, alpha=0.45, color="steelblue", linewidths=0)

        ax.scatter([0], [r["truth_full_mse"]], marker="*", s=130,
                   color="crimson", zorder=10,
                   label=f"truth={r['truth_full_mse']:.2e}")
        ax.scatter([0], [r["twin_full_mse"]], marker="D", s=55,
                   color="darkorange", zorder=10,
                   label=f"twin={r['twin_full_mse']:.2e}")
        ax.scatter(
            [r["geo_to_truth_or_twin_deg"][r["argmin_idx"]]],
            [r["argmin_full_mse"]],
            marker="X", s=85, color="black", zorder=11,
            label=f"argmin={r['argmin_full_mse']:.2e}",
        )

        ax.set_yscale("log")
        ax.set_xlabel("min geodesic to truth or twin [deg]")
        ax.set_ylabel("surrogate full-LC MSE  [mag²]")
        ax.set_title(f"seed {r['seed']}  (Δargmin–truth={r['argmin_geo_to_truth_deg']:.1f}°)")
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "s002 — surrogate full-LC MSE landscape vs q0 distance from truth/twin "
        f"({N_SOBOL} Sobol-Shoemake quats per seed, ω fixed at truth)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
