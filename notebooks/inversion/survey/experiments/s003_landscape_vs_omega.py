"""s003 — surrogate landscape under ω-misspecification.

Survey question Q3. s002 settled: at fixed truth-ω the surrogate full-LC MSE
landscape over q0 has its argmin at truth-q0 on 8/8 PA-stratified seeds, with
no competing basin at sobol resolution. But m115's solver failure on seed 91
was during ω SEARCH (per-ω DE), not after ω was nailed. So the realistic
landscape question is: as ω drifts off truth-ω, does the q0 argmin slide
smoothly, or does it discontinuously jump to a different basin?

Method:
  - 3 seeds: 91 (m145 anchor), 6 (m141 elevated-floor), 41 (PA-low control).
  - Build a 24-element ω perturbation ladder per seed (1 baseline + 9 mag-
    only + 8 dir-only + 6 combined). Mag perturbations are scalar multiples
    of ω_truth-mag; dir perturbations rotate ω_truth-dir by ε around a
    fixed perpendicular axis (reproducible across seeds via a deterministic
    perpendicular-axis pick).
  - For each (seed, ω_perturbed), Sobol-Shoemake 512 quats + truth-q0 at
    index 0 (so we can track how truth-q0's MSE moves as ω drifts).
  - Score full-LC MSE per candidate against the cached truth `mag_hifi`
    (which was generated at (q0_truth, ω_truth) — that's the reference LC
    we're trying to fit).
  - Per (seed, ω_perturbed): record argmin's geodesic distance to truth-q0,
    truth-q0's MSE under perturbed ω, best-sobol vs truth-MSE ratio.

Key observable: argmin_q0_geodesic_to_truth as a function of ω perturbation.

Decision tree:
  - argmin stays within ~10° of truth-q0 across most perturbations: surrogate
    landscape is robust under ω-misspecification → m115 failure is solver-
    side. Local polish from a near-truth initial seed should work even if ω
    is initialised off-truth.
  - argmin jumps far from truth-q0 at modest ω perturbations: surrogate
    landscape is ω-fragile → joint (q0, ω) optimisation needs careful
    coupling, can't decouple ω-search from q0-search.

Pool(8) per seed, BLAS=1 in workers. Same per-candidate cost as s002 (~89 ms).

Outputs:
  - results/s003/per_seed_omega.npz       per-seed × per-perturbation arrays
  - results/s003/summary.json             argmin drift + truth-cost growth
  - results/s003/argmin_drift.png         drift curves
  - results/s003/landscape_panels.png     6-panel: per-seed landscape at 2
                                          representative perturbations each
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
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg_batch  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s003"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [6, 41, 91]
N_SOBOL = 511  # +1 truth-q0 → 512 total per perturbation
SOBOL_SEED = 42
N_WORKERS = 8


# ω perturbation ladder.
# (label, mag_factor, dir_rot_deg) — dir_rot_deg is rotation of ω_dir around
# a deterministic perpendicular axis; mag_factor scales ω_mag.
PERTURBATIONS = [
    ("baseline",        1.00,   0.0),
    # Pure mag perturbations.
    ("mag_x0.5",        0.50,   0.0),
    ("mag_x0.8",        0.80,   0.0),
    ("mag_x0.9",        0.90,   0.0),
    ("mag_x0.95",       0.95,   0.0),
    ("mag_x1.05",       1.05,   0.0),
    ("mag_x1.1",        1.10,   0.0),
    ("mag_x1.2",        1.20,   0.0),
    ("mag_x1.5",        1.50,   0.0),
    ("mag_x2.0",        2.00,   0.0),
    # Pure dir perturbations.
    ("dir_0.5deg",      1.00,   0.5),
    ("dir_1deg",        1.00,   1.0),
    ("dir_2deg",        1.00,   2.0),
    ("dir_5deg",        1.00,   5.0),
    ("dir_10deg",       1.00,  10.0),
    ("dir_30deg",       1.00,  30.0),
    ("dir_60deg",       1.00,  60.0),
    ("dir_90deg",       1.00,  90.0),
    # Combined mag+dir perturbations.
    ("mag_x0.5_dir10",  0.50,  10.0),
    ("mag_x0.5_dir30",  0.50,  30.0),
    ("mag_x2.0_dir10",  2.00,  10.0),
    ("mag_x2.0_dir30",  2.00,  30.0),
    ("mag_x1.1_dir5",   1.10,   5.0),
    ("mag_x1.1_dir30",  1.10,  30.0),
]


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


def build_grid(q0_truth: np.ndarray) -> np.ndarray:
    """Truth-q0 at index 0 + 511 Sobol-Shoemake quats → (512, 4)."""
    sobol = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u = sobol.random(N_SOBOL)
    q_sobol = shoemake_to_quat(u)
    return np.vstack([q0_truth[None, :], q_sobol])


def perpendicular_axis(omega_dir: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to omega_dir, deterministic.

    Uses cross product with body x; if degenerate (omega_dir || x), fall
    back to body y. Same axis is used across all dir perturbations for a
    given seed, so dir_5deg and dir_10deg sit on a continuous 1-parameter
    arc of ω-dir misspecification.
    """
    refs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ref in refs:
        a = np.cross(omega_dir, ref)
        n = np.linalg.norm(a)
        if n > 1e-6:
            return a / n
    raise RuntimeError("could not find perpendicular axis")


def perturb_omega(omega_truth: np.ndarray, mag_factor: float, dir_rot_deg: float) -> np.ndarray:
    """Apply (mag_factor, dir_rot_deg) perturbation to a body-frame ω vector."""
    mag = float(np.linalg.norm(omega_truth))
    if mag < 1e-12:
        return omega_truth.copy()
    direction = omega_truth / mag

    if dir_rot_deg != 0.0:
        axis = perpendicular_axis(direction)
        rot = Rotation.from_rotvec(axis * np.radians(dir_rot_deg))
        direction = rot.apply(direction)

    return direction * (mag * mag_factor)


# Worker globals.
_W_TIMES = None
_W_SUN = None
_W_OBS = None
_W_SAT = None
_W_INERTIA = None
_W_OBS_DIST = None
_W_MAG_HIFI = None


def init_worker(times, sun_pos, obs_pos, sat_pos, inertia_tensor, obs_dist, mag_hifi):
    global _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA, _W_OBS_DIST, _W_MAG_HIFI
    _W_TIMES = times
    _W_SUN = sun_pos
    _W_OBS = obs_pos
    _W_SAT = sat_pos
    _W_INERTIA = inertia_tensor
    _W_OBS_DIST = obs_dist
    _W_MAG_HIFI = mag_hifi
    surrogate_eval.get_model()


def score_candidate(args):
    """Worker function — args is (q0_wxyz, omega_rad)."""
    q0, omega = args
    k1b, k2b, _ = propagate_to_body_frame(
        q0, omega, _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA,
    )
    pred = surrogate_eval.predict(k1b, k2b, _W_OBS_DIST)
    return float(surrogate_eval.full_lc_mse(pred, _W_MAG_HIFI))


def run_seed(seed: int, inertia_tensor: np.ndarray) -> dict:
    d = traj_load.load_truth(seed)
    q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
    omega_truth = np.asarray(d["omega0_rad"], dtype=float)

    q_grid = build_grid(q0_truth)
    n_cand = q_grid.shape[0]
    n_pert = len(PERTURBATIONS)

    # Precompute perturbed omegas per perturbation entry.
    omegas = np.array([perturb_omega(omega_truth, m, dr) for _, m, dr in PERTURBATIONS])

    # Per-perturbation results.
    full_mse = np.full((n_pert, n_cand), np.nan)
    truth_q0_mse = np.full(n_pert, np.nan)
    argmin_idx = np.full(n_pert, -1, dtype=int)
    argmin_mse = np.full(n_pert, np.nan)
    argmin_geo_to_truth = np.full(n_pert, np.nan)
    second_best_mse = np.full(n_pert, np.nan)
    n_sobol_below_truth = np.full(n_pert, -1, dtype=int)

    geo_to_truth = quat_geodesic_deg_batch(q_grid, q0_truth)

    init_args = (
        d["observation_times"], d["sun_pos"], d["obs_pos"], d["sat_pos"],
        inertia_tensor, d["obs_dist"], d["mag_hifi"],
    )

    t0 = time.time()
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=init_args) as pool:
        for pi, (label, mf, dr) in enumerate(PERTURBATIONS):
            omega_p = omegas[pi]
            args_list = [(q_grid[i], omega_p) for i in range(n_cand)]
            mses = pool.map(score_candidate, args_list, chunksize=16)
            mses = np.array(mses, dtype=float)
            full_mse[pi] = mses

            truth_q0_mse[pi] = float(mses[0])
            am = int(np.argmin(mses))
            argmin_idx[pi] = am
            argmin_mse[pi] = float(mses[am])
            argmin_geo_to_truth[pi] = float(geo_to_truth[am])
            sorted_mses = np.sort(mses)
            second_best_mse[pi] = float(sorted_mses[1])
            sobol_mse = mses[1:]
            n_sobol_below_truth[pi] = int(np.sum(sobol_mse < truth_q0_mse[pi]))

            print(
                f"  seed {seed:3d}  pert={label:18s}  ω_mag×{mf:.2f} dir{dr:+5.1f}°   "
                f"truth_q0_mse={truth_q0_mse[pi]:.3e}  "
                f"argmin_mse={argmin_mse[pi]:.3e}  "
                f"argmin_geo→truth={argmin_geo_to_truth[pi]:6.1f}°  "
                f"n_sobol<truth={n_sobol_below_truth[pi]:3d}",
                flush=True,
            )
    wall = time.time() - t0
    print(f"  seed {seed:3d}  total wall {wall:.1f}s for {n_pert} perturbations\n", flush=True)

    return {
        "seed": int(seed),
        "q_grid": q_grid,
        "omegas": omegas,
        "full_mse": full_mse,
        "geo_to_truth_deg": geo_to_truth,
        "truth_q0_mse": truth_q0_mse,
        "argmin_idx": argmin_idx,
        "argmin_mse": argmin_mse,
        "argmin_geo_to_truth_deg": argmin_geo_to_truth,
        "second_best_mse": second_best_mse,
        "n_sobol_below_truth": n_sobol_below_truth,
        "wall_s": wall,
    }


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    print(f"s003: {len(SEEDS)} seeds × {len(PERTURBATIONS)} ω perturbations × "
          f"{N_SOBOL + 1} candidates, Pool({N_WORKERS}), BLAS=1.\n", flush=True)

    per_seed = [run_seed(s, inertia_tensor) for s in SEEDS]

    # ---- Save per-seed NPZ ----
    npz_path = OUT_DIR / "per_seed_omega.npz"
    save = {}
    for r in per_seed:
        s = r["seed"]
        save[f"seed{s:03d}_q_grid"] = r["q_grid"]
        save[f"seed{s:03d}_omegas"] = r["omegas"]
        save[f"seed{s:03d}_full_mse"] = r["full_mse"]
        save[f"seed{s:03d}_geo_to_truth_deg"] = r["geo_to_truth_deg"]
        save[f"seed{s:03d}_truth_q0_mse"] = r["truth_q0_mse"]
        save[f"seed{s:03d}_argmin_idx"] = r["argmin_idx"]
        save[f"seed{s:03d}_argmin_mse"] = r["argmin_mse"]
        save[f"seed{s:03d}_argmin_geo_to_truth_deg"] = r["argmin_geo_to_truth_deg"]
        save[f"seed{s:03d}_second_best_mse"] = r["second_best_mse"]
        save[f"seed{s:03d}_n_sobol_below_truth"] = r["n_sobol_below_truth"]
    save["seeds"] = np.array(SEEDS, dtype=int)
    save["pert_labels"] = np.array([p[0] for p in PERTURBATIONS])
    save["pert_mag_factor"] = np.array([p[1] for p in PERTURBATIONS])
    save["pert_dir_rot_deg"] = np.array([p[2] for p in PERTURBATIONS])
    save["n_sobol"] = N_SOBOL
    save["sobol_seed"] = SOBOL_SEED
    np.savez(npz_path, **save)
    print(f"Saved: {npz_path}")

    summary = build_summary(per_seed)
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    drift_path = OUT_DIR / "argmin_drift.png"
    save_argmin_drift(per_seed, drift_path)
    print(f"Saved: {drift_path}")

    panels_path = OUT_DIR / "landscape_panels.png"
    save_landscape_panels(per_seed, panels_path)
    print(f"Saved: {panels_path}")

    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


def build_summary(per_seed):
    rows_per_seed = []
    for r in per_seed:
        rows = []
        for pi, (label, mf, dr) in enumerate(PERTURBATIONS):
            rows.append({
                "perturbation": label,
                "mag_factor": mf,
                "dir_rot_deg": dr,
                "truth_q0_mse": float(r["truth_q0_mse"][pi]),
                "argmin_mse": float(r["argmin_mse"][pi]),
                "argmin_geo_to_truth_deg": float(r["argmin_geo_to_truth_deg"][pi]),
                "argmin_idx": int(r["argmin_idx"][pi]),
                "second_best_mse": float(r["second_best_mse"][pi]),
                "n_sobol_below_truth": int(r["n_sobol_below_truth"][pi]),
            })
        rows_per_seed.append({"seed": r["seed"], "perturbations": rows})

    # Population-level: how many (seed, perturbation) cells have argmin within
    # 10° of truth-q0 (i.e. local-polish-friendly)?
    n_cells = len(SEEDS) * len(PERTURBATIONS)
    drift = np.array([r["argmin_geo_to_truth_deg"] for r in per_seed])  # (n_seeds, n_pert)
    n_within_5 = int(np.sum(drift < 5.0))
    n_within_10 = int(np.sum(drift < 10.0))
    n_within_30 = int(np.sum(drift < 30.0))
    n_far = int(np.sum(drift >= 30.0))

    return {
        "seeds": SEEDS,
        "n_perturbations": len(PERTURBATIONS),
        "perturbation_labels": [p[0] for p in PERTURBATIONS],
        "per_seed": rows_per_seed,
        "n_cells_total": n_cells,
        "n_cells_argmin_within_5deg_of_truth": n_within_5,
        "n_cells_argmin_within_10deg_of_truth": n_within_10,
        "n_cells_argmin_within_30deg_of_truth": n_within_30,
        "n_cells_argmin_30deg_or_more_from_truth": n_far,
    }


def save_argmin_drift(per_seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"6": "tab:red", "41": "tab:green", "91": "tab:blue"}

    # Panel 1: argmin_geo_to_truth vs mag_factor (dir=0)
    ax = axes[0, 0]
    mag_indices = [i for i, p in enumerate(PERTURBATIONS) if p[2] == 0.0]
    mag_factors = np.array([PERTURBATIONS[i][1] for i in mag_indices])
    for r in per_seed:
        y = r["argmin_geo_to_truth_deg"][mag_indices]
        ax.plot(mag_factors, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("ω-mag factor (1.0 = truth)")
    ax.set_ylabel("argmin geodesic to truth-q0  [deg]")
    ax.set_title("argmin drift under pure ω-mag perturbation")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: argmin_geo_to_truth vs dir_rot_deg (mag=1.0)
    ax = axes[0, 1]
    dir_indices = [i for i, p in enumerate(PERTURBATIONS) if p[1] == 1.0 and p[2] != 0.0]
    dir_rots = np.array([PERTURBATIONS[i][2] for i in dir_indices])
    for r in per_seed:
        y = r["argmin_geo_to_truth_deg"][dir_indices]
        ax.plot(dir_rots, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.set_xlabel("ω-dir rotation [deg]")
    ax.set_ylabel("argmin geodesic to truth-q0  [deg]")
    ax.set_title("argmin drift under pure ω-dir perturbation")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: truth_q0_mse growth vs mag_factor
    ax = axes[1, 0]
    for r in per_seed:
        y = r["truth_q0_mse"][mag_indices]
        ax.plot(mag_factors, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("ω-mag factor")
    ax.set_ylabel("truth-q0 MSE under perturbed ω  [mag²]")
    ax.set_title("how badly truth-q0 fits as ω drifts (mag-only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: truth_q0_mse growth vs dir_rot
    ax = axes[1, 1]
    for r in per_seed:
        y = r["truth_q0_mse"][dir_indices]
        ax.plot(dir_rots, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.set_xlabel("ω-dir rotation [deg]")
    ax.set_ylabel("truth-q0 MSE under perturbed ω  [mag²]")
    ax.set_title("how badly truth-q0 fits as ω drifts (dir-only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "s003 — surrogate landscape under ω-misspecification",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


def save_landscape_panels(per_seed, path):
    """Per-seed landscape at 2 representative perturbations: dir_5deg + dir_30deg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representative = [
        ("baseline", "ω = ω_truth"),
        ("dir_5deg", "ω-dir +5°"),
        ("dir_30deg", "ω-dir +30°"),
        ("mag_x0.5", "ω-mag × 0.5"),
        ("mag_x2.0", "ω-mag × 2.0"),
        ("mag_x0.5_dir30", "ω-mag×0.5 + dir+30°"),
    ]
    pert_names = [p[0] for p in PERTURBATIONS]
    rep_indices = [pert_names.index(name) for name, _ in representative]

    n_seeds = len(per_seed)
    n_pert = len(representative)
    fig, axes = plt.subplots(n_seeds, n_pert,
                             figsize=(3.6 * n_pert, 3.0 * n_seeds),
                             sharey="row")
    axes = np.atleast_2d(axes)

    for si, r in enumerate(per_seed):
        for pi, (name, label) in enumerate(representative):
            ax = axes[si, pi]
            idx = rep_indices[pi]
            mse = r["full_mse"][idx]
            geo = r["geo_to_truth_deg"]

            # Sobol cloud
            ax.scatter(geo[1:], mse[1:], s=3, alpha=0.45, color="steelblue",
                       linewidths=0)
            # Truth-q0 (now under perturbed ω)
            ax.scatter([0], [mse[0]], marker="*", s=100, color="crimson",
                       zorder=10, label=f"truth-q0={mse[0]:.2e}")
            # Argmin
            am = int(r["argmin_idx"][idx])
            ax.scatter([geo[am]], [mse[am]], marker="X", s=70, color="black",
                       zorder=11, label=f"argmin={mse[am]:.2e} @ {geo[am]:.0f}°")

            ax.set_yscale("log")
            if si == n_seeds - 1:
                ax.set_xlabel("geodesic to truth-q0  [deg]")
            if pi == 0:
                ax.set_ylabel(f"seed {r['seed']}\nMSE [mag²]")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "s003 — q0 landscape per seed at representative ω perturbations",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
