"""s004 — m103 alignment-cost landscape under ω-misspecification.

Survey question Q5. s003 established that surrogate-MSE has an extremely narrow
truth-ω tube (~1° dir / ~2-5% mag) on 3 seeds — outside the tube the q0 argmin
slides far from truth. m115's per-ω DE on the surrogate is therefore fighting
an essentially invisible target in ω-space. The s003 strategic recommendation
was joint (q0, ω) local descent. But before we commit to that architecture,
ask: does the m103 alignment cost have a wider ω-tube? If yes, alignment cost
could feed an outer ω-search that the surrogate can't, and a decoupled
alignment-ω-outer / surrogate-q0-inner architecture might still be viable.

Method (parallels s003 in shape, replaces cost surface):
  - 3 seeds (6 / 18 / 28). Seeds 41 and 91 from s003 cannot be reused: both
    have zero constraint epochs (≤1 spec peak after the m103 anchor strip),
    so the alignment cost is undefined. Replacements were chosen to span the
    same PA stratification (low / mid / high) while having usable constraint
    sets:
      seed 6:  PA_med  62.5°, ω 0.71°/s, 3 constraints  (carry from s003)
      seed 18: PA_med  14.2°, ω 1.13°/s, 14 constraints (PA-low)
      seed 28: PA_med  87.3°, ω 1.44°/s, 6 constraints  (PA-high)
  - Same 24-element ω perturbation ladder as s003 (1 baseline + 9 mag + 8 dir
    + 6 combined). Same Sobol-Shoemake 512 quats (truth-q0 + 511 sobol) so
    candidate quaternions are pairwise comparable to s003 results.
  - For each (seed, ω_perturbed, candidate q0), propagate to get full
    quaternion trajectory, derive pab_body[t] at constraint epochs, evaluate
    m103 alignment cost. **No surrogate is invoked here — pure peak-bisector
    geometry. Per-candidate cost is therefore much cheaper than s003.**

Faithful reproduction of m103 alignment-cost machinery (copied + adapted from
s001, which itself copied from `m103_hybrid.py`, NOT imported):
  - canonical observed LC: rng=default_rng(42), sigma=0.05
  - peak detection: find_peaks(-observed_lc, distance=5, prominence=0.3)
  - spec peaks: subset where observed_lc < 9.0
  - anchor: savgol-smoothed brightest spec peak (tiebreak on epoch index)
  - constraint epochs: spec peaks excluding anchor
  - alignment cost = sum_ci w * (1 - max_a (pab_body[ci] · normals[allowed_a]))^2
    with w = CONSTRAINT_WEIGHT = 10.0, allowed = get_allowed_normals(observed[ci])

Key observable: argmin_q0_geodesic_to_truth as a function of ω perturbation,
exactly as s003. Decision tree for Q5:

  - argmin tube width comparable to or NARROWER than s003's surrogate (~1° dir
    / ~2-5% mag): alignment cost adds NO ω-search value the surrogate doesn't
    already provide → m103 ω-outer / surrogate-q0-inner is a dead architecture
    → joint local descent (Q4) is the only path forward.
  - argmin tube width substantially WIDER (e.g. ≥10° dir or ≥30% mag): there's
    a hybrid worth designing — alignment-cost outer ω-search + surrogate q0
    inner polish, with the alignment cost providing the gradient signal in
    ω-space that the surrogate cannot.

Pool(8), BLAS=1 in workers. No surrogate forward pass per candidate, so each
score is dominated by the propagator (~10-20 ms estimated vs s003's ~89 ms).
Wall budget: ~5 min total expected.

Outputs (mirroring s003 layout):
  - results/s004/per_seed_omega.npz       per-seed × per-perturbation arrays
  - results/s004/summary.json             argmin drift + truth-cost growth
  - results/s004/argmin_drift.png         drift curves vs s003 overlay
  - results/s004/landscape_panels.png     per-seed landscape at 6 perturbations
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
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib import traj_load  # noqa: E402
from lib.forward import quat_geodesic_deg_batch  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s004"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# m103-era constants (copied from s001, NOT imported from m103).
CONSTRAINT_WEIGHT = 10.0
NOISE_SEED = 42
NOISE_SIGMA = 0.05
SPEC_THRESHOLD = 9.0
OBS_PEAK_DISTANCE = 5
OBS_PEAK_PROMINENCE = 0.3
SAVGOL_WINDOW = 7
SAVGOL_POLY = 3

SEEDS = [6, 18, 28]
N_SOBOL = 511
SOBOL_SEED = 42
N_WORKERS = 8

# Same perturbation ladder as s003, so cost surfaces are comparable cell-by-cell.
PERTURBATIONS = [
    ("baseline",        1.00,   0.0),
    ("mag_x0.5",        0.50,   0.0),
    ("mag_x0.8",        0.80,   0.0),
    ("mag_x0.9",        0.90,   0.0),
    ("mag_x0.95",       0.95,   0.0),
    ("mag_x1.05",       1.05,   0.0),
    ("mag_x1.1",        1.10,   0.0),
    ("mag_x1.2",        1.20,   0.0),
    ("mag_x1.5",        1.50,   0.0),
    ("mag_x2.0",        2.00,   0.0),
    ("dir_0.5deg",      1.00,   0.5),
    ("dir_1deg",        1.00,   1.0),
    ("dir_2deg",        1.00,   2.0),
    ("dir_5deg",        1.00,   5.0),
    ("dir_10deg",       1.00,  10.0),
    ("dir_30deg",       1.00,  30.0),
    ("dir_60deg",       1.00,  60.0),
    ("dir_90deg",       1.00,  90.0),
    ("mag_x0.5_dir10",  0.50,  10.0),
    ("mag_x0.5_dir30",  0.50,  30.0),
    ("mag_x2.0_dir10",  2.00,  10.0),
    ("mag_x2.0_dir30",  2.00,  30.0),
    ("mag_x1.1_dir5",   1.10,   5.0),
    ("mag_x1.1_dir30",  1.10,  30.0),
]


# --- m103 anchor / constraint plumbing (copied from s001) -------------------
def get_allowed_normals(mag: float):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


def canonical_observed_lc(mag_hifi: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(NOISE_SEED)
    return mag_hifi + rng.normal(0.0, NOISE_SIGMA, mag_hifi.shape[0])


def select_anchor(observed_lc: np.ndarray, spec_peaks: np.ndarray) -> int:
    smoothed = savgol_filter(observed_lc, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    smooth_mags = smoothed[spec_peaks]
    sr = np.argsort(smooth_mags)
    if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
        anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
    else:
        anchor_rank = int(sr[0])
    return int(spec_peaks[anchor_rank])


def derive_constraint_epochs(observed_lc: np.ndarray) -> tuple[int, np.ndarray]:
    """Replicate s001/m103 anchor + constraint selection."""
    peaks_idx, _ = find_peaks(
        -observed_lc, distance=OBS_PEAK_DISTANCE, prominence=OBS_PEAK_PROMINENCE
    )
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < SPEC_THRESHOLD]
    if len(spec_peaks) <= 1:
        return -1, np.array([], dtype=int)
    anchor_idx = select_anchor(observed_lc, spec_peaks)
    constraint_epochs = spec_peaks[spec_peaks != anchor_idx]
    return int(anchor_idx), constraint_epochs


def derive_pab_J2000(quats_truth_wxyz: np.ndarray, pab_body_truth: np.ndarray) -> np.ndarray:
    """pab_J2000[t] = R(q_truth_t).T @ pab_body_truth[t].

    Bypasses the bisector-sign convention question entirely by inverting the
    cached truth body-frame pab using the cached truth quaternions.
    """
    quats_xyzw = quats_truth_wxyz[:, [1, 2, 3, 0]]
    R_truth = Rotation.from_quat(quats_xyzw)
    return R_truth.inv().apply(pab_body_truth)


def perpendicular_axis(omega_dir: np.ndarray) -> np.ndarray:
    refs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ref in refs:
        a = np.cross(omega_dir, ref)
        n = np.linalg.norm(a)
        if n > 1e-6:
            return a / n
    raise RuntimeError("could not find perpendicular axis")


def perturb_omega(omega_truth: np.ndarray, mag_factor: float, dir_rot_deg: float) -> np.ndarray:
    mag = float(np.linalg.norm(omega_truth))
    if mag < 1e-12:
        return omega_truth.copy()
    direction = omega_truth / mag
    if dir_rot_deg != 0.0:
        axis = perpendicular_axis(direction)
        rot = Rotation.from_rotvec(axis * np.radians(dir_rot_deg))
        direction = rot.apply(direction)
    return direction * (mag * mag_factor)


# --- Worker plumbing --------------------------------------------------------
_W_TIMES = None
_W_INERTIA = None
_W_PAB_J2000_AT_CON = None      # (n_con, 3) — pab_J2000 at constraint epochs only
_W_CON_EPOCHS = None             # (n_con,) — indices into the full N-epoch array
_W_ALLOWED_NORMALS = None        # list of np.ndarray, one per constraint epoch
                                 # (n_normals_at_ci, 3) of unit normals available
                                 # at that constraint's magnitude band


def init_worker(times, inertia_tensor, pab_J2000_at_con, con_epochs, allowed_normals):
    global _W_TIMES, _W_INERTIA, _W_PAB_J2000_AT_CON, _W_CON_EPOCHS, _W_ALLOWED_NORMALS
    _W_TIMES = times
    _W_INERTIA = inertia_tensor
    _W_PAB_J2000_AT_CON = pab_J2000_at_con
    _W_CON_EPOCHS = con_epochs
    _W_ALLOWED_NORMALS = allowed_normals


def score_candidate(args):
    """Worker function — args is (q0_wxyz, omega_rad). Returns alignment cost."""
    q0, omega = args
    quats, _ = propagate_attitude(
        q0=q0, omega0=omega, times=_W_TIMES,
        mode="tumbling", inertia_tensor=_W_INERTIA,
    )
    # Pull rotations only at constraint epochs.
    quats_at_con = quats[_W_CON_EPOCHS]
    quats_xyzw = quats_at_con[:, [1, 2, 3, 0]]
    R_at_con = Rotation.from_quat(quats_xyzw)
    pab_body_at_con = R_at_con.apply(_W_PAB_J2000_AT_CON)  # (n_con, 3)

    cost = 0.0
    for k in range(pab_body_at_con.shape[0]):
        bds = float(np.max(pab_body_at_con[k] @ _W_ALLOWED_NORMALS[k].T))
        cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
    return cost


# --- Per-seed driver --------------------------------------------------------
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
    sobol = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u = sobol.random(N_SOBOL)
    q_sobol = shoemake_to_quat(u)
    return np.vstack([q0_truth[None, :], q_sobol])


def run_seed(seed: int, inertia_tensor: np.ndarray, unique_normals: np.ndarray) -> dict:
    d = traj_load.load_truth(seed)
    q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
    omega_truth = np.asarray(d["omega0_rad"], dtype=float)
    times = np.asarray(d["observation_times"], dtype=float)
    quats_truth = np.asarray(d["quaternions"], dtype=float)
    pab_body_truth = np.asarray(d["pab_body"], dtype=float)
    mag_hifi = np.asarray(d["mag_hifi"], dtype=float)

    observed_lc = canonical_observed_lc(mag_hifi)
    anchor_idx, con_epochs = derive_constraint_epochs(observed_lc)
    n_con = len(con_epochs)
    if n_con == 0:
        raise RuntimeError(f"seed {seed} has zero constraint epochs — cannot evaluate alignment cost")

    # pab_J2000 only at constraint epochs (we never need it elsewhere).
    pab_J2000_full = derive_pab_J2000(quats_truth, pab_body_truth)
    pab_J2000_at_con = pab_J2000_full[con_epochs]

    # Allowed normals per constraint epoch — magnitude-banded lookup against
    # the OBSERVED magnitude (not the predicted one). Static per seed.
    allowed_normals = [
        np.asarray(unique_normals[get_allowed_normals(observed_lc[ci])], dtype=float)
        for ci in con_epochs
    ]

    # Smoke test: at (q0_truth, omega_truth), the alignment cost must equal the
    # value reported by s001 (pab_body_cand[ci] = pab_body_truth[ci] exactly,
    # so the cost is a function of cached pab_body and unique_normals only).
    smoke_cost = 0.0
    for k, ci in enumerate(con_epochs):
        bds = float(np.max(pab_body_truth[ci] @ allowed_normals[k].T))
        smoke_cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
    print(f"  seed {seed:3d}  smoke (cost at truth from cached pab_body): {smoke_cost:.4e}  (n_con={n_con})", flush=True)

    q_grid = build_grid(q0_truth)
    n_cand = q_grid.shape[0]
    n_pert = len(PERTURBATIONS)
    omegas = np.array([perturb_omega(omega_truth, m, dr) for _, m, dr in PERTURBATIONS])

    full_cost = np.full((n_pert, n_cand), np.nan)
    truth_q0_cost = np.full(n_pert, np.nan)
    argmin_idx = np.full(n_pert, -1, dtype=int)
    argmin_cost = np.full(n_pert, np.nan)
    argmin_geo_to_truth = np.full(n_pert, np.nan)
    second_best_cost = np.full(n_pert, np.nan)
    n_sobol_below_truth = np.full(n_pert, -1, dtype=int)

    geo_to_truth = quat_geodesic_deg_batch(q_grid, q0_truth)

    init_args = (times, inertia_tensor, pab_J2000_at_con, con_epochs, allowed_normals)

    t0 = time.time()
    with Pool(processes=N_WORKERS, initializer=init_worker, initargs=init_args) as pool:
        for pi, (label, mf, dr) in enumerate(PERTURBATIONS):
            omega_p = omegas[pi]
            args_list = [(q_grid[i], omega_p) for i in range(n_cand)]
            costs = pool.map(score_candidate, args_list, chunksize=16)
            costs = np.array(costs, dtype=float)
            full_cost[pi] = costs
            truth_q0_cost[pi] = float(costs[0])
            am = int(np.argmin(costs))
            argmin_idx[pi] = am
            argmin_cost[pi] = float(costs[am])
            argmin_geo_to_truth[pi] = float(geo_to_truth[am])
            sorted_costs = np.sort(costs)
            second_best_cost[pi] = float(sorted_costs[1])
            sobol_costs = costs[1:]
            n_sobol_below_truth[pi] = int(np.sum(sobol_costs < truth_q0_cost[pi]))
            print(
                f"  seed {seed:3d}  pert={label:18s}  ω_mag×{mf:.2f} dir{dr:+5.1f}°   "
                f"truth_q0_cost={truth_q0_cost[pi]:.3e}  "
                f"argmin_cost={argmin_cost[pi]:.3e}  "
                f"argmin_geo→truth={argmin_geo_to_truth[pi]:6.1f}°  "
                f"n_sobol<truth={n_sobol_below_truth[pi]:3d}",
                flush=True,
            )
    wall = time.time() - t0
    print(f"  seed {seed:3d}  total wall {wall:.1f}s for {n_pert} perturbations\n", flush=True)

    return {
        "seed": int(seed),
        "n_constraints": int(n_con),
        "anchor_idx": int(anchor_idx),
        "constraint_epochs": np.asarray(con_epochs, dtype=int),
        "smoke_cost_at_truth": float(smoke_cost),
        "q_grid": q_grid,
        "omegas": omegas,
        "full_cost": full_cost,
        "geo_to_truth_deg": geo_to_truth,
        "truth_q0_cost": truth_q0_cost,
        "argmin_idx": argmin_idx,
        "argmin_cost": argmin_cost,
        "argmin_geo_to_truth_deg": argmin_geo_to_truth,
        "second_best_cost": second_best_cost,
        "n_sobol_below_truth": n_sobol_below_truth,
        "wall_s": wall,
    }


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)
    unique_normals = np.asarray(master["unique_normals"], dtype=float)
    assert unique_normals.shape == (10, 3), f"unexpected normals: {unique_normals.shape}"

    print(f"s004: {len(SEEDS)} seeds × {len(PERTURBATIONS)} ω perturbations × "
          f"{N_SOBOL + 1} candidates, Pool({N_WORKERS}), BLAS=1.\n"
          f"Cost: m103 alignment cost (w={CONSTRAINT_WEIGHT}); "
          f"no surrogate forward pass per candidate.\n", flush=True)

    per_seed = [run_seed(s, inertia_tensor, unique_normals) for s in SEEDS]

    npz_path = OUT_DIR / "per_seed_omega.npz"
    save = {}
    for r in per_seed:
        s = r["seed"]
        save[f"seed{s:03d}_q_grid"] = r["q_grid"]
        save[f"seed{s:03d}_omegas"] = r["omegas"]
        save[f"seed{s:03d}_full_cost"] = r["full_cost"]
        save[f"seed{s:03d}_geo_to_truth_deg"] = r["geo_to_truth_deg"]
        save[f"seed{s:03d}_truth_q0_cost"] = r["truth_q0_cost"]
        save[f"seed{s:03d}_argmin_idx"] = r["argmin_idx"]
        save[f"seed{s:03d}_argmin_cost"] = r["argmin_cost"]
        save[f"seed{s:03d}_argmin_geo_to_truth_deg"] = r["argmin_geo_to_truth_deg"]
        save[f"seed{s:03d}_second_best_cost"] = r["second_best_cost"]
        save[f"seed{s:03d}_n_sobol_below_truth"] = r["n_sobol_below_truth"]
        save[f"seed{s:03d}_constraint_epochs"] = r["constraint_epochs"]
        save[f"seed{s:03d}_n_constraints"] = np.array([r["n_constraints"]], dtype=int)
        save[f"seed{s:03d}_smoke_cost_at_truth"] = np.array([r["smoke_cost_at_truth"]], dtype=float)
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
                "truth_q0_cost": float(r["truth_q0_cost"][pi]),
                "argmin_cost": float(r["argmin_cost"][pi]),
                "argmin_geo_to_truth_deg": float(r["argmin_geo_to_truth_deg"][pi]),
                "argmin_idx": int(r["argmin_idx"][pi]),
                "second_best_cost": float(r["second_best_cost"][pi]),
                "n_sobol_below_truth": int(r["n_sobol_below_truth"][pi]),
            })
        rows_per_seed.append({
            "seed": r["seed"],
            "n_constraints": r["n_constraints"],
            "smoke_cost_at_truth": r["smoke_cost_at_truth"],
            "perturbations": rows,
        })

    n_cells = len(SEEDS) * len(PERTURBATIONS)
    drift = np.array([r["argmin_geo_to_truth_deg"] for r in per_seed])
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
        "note_seeds_undefined_in_s003_overlap": (
            "s003 used seeds 6/41/91; alignment cost is undefined on 41 and 91 "
            "(both have ≤1 spec peak after m103 anchor strip → 0 constraints). "
            "s004 substituted 18 (PA-low) and 28 (PA-high) to preserve PA-spread "
            "while keeping the alignment cost defined. Seed 6 carries over."
        ),
    }


def save_argmin_drift(per_seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"6": "tab:red", "18": "tab:green", "28": "tab:blue"}

    mag_indices = [i for i, p in enumerate(PERTURBATIONS) if p[2] == 0.0]
    mag_factors = np.array([PERTURBATIONS[i][1] for i in mag_indices])
    dir_indices = [i for i, p in enumerate(PERTURBATIONS) if p[1] == 1.0 and p[2] != 0.0]
    dir_rots = np.array([PERTURBATIONS[i][2] for i in dir_indices])

    ax = axes[0, 0]
    for r in per_seed:
        y = r["argmin_geo_to_truth_deg"][mag_indices]
        ax.plot(mag_factors, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']} (n_con={r['n_constraints']})")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("ω-mag factor (1.0 = truth)")
    ax.set_ylabel("argmin geodesic to truth-q0  [deg]")
    ax.set_title("alignment-cost argmin drift under pure ω-mag perturbation")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for r in per_seed:
        y = r["argmin_geo_to_truth_deg"][dir_indices]
        ax.plot(dir_rots, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']} (n_con={r['n_constraints']})")
    ax.set_xlabel("ω-dir rotation [deg]")
    ax.set_ylabel("argmin geodesic to truth-q0  [deg]")
    ax.set_title("alignment-cost argmin drift under pure ω-dir perturbation")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for r in per_seed:
        y = r["truth_q0_cost"][mag_indices]
        ax.plot(mag_factors, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("ω-mag factor")
    ax.set_ylabel("truth-q0 alignment cost  [unitless]")
    ax.set_title("how badly truth-q0 fits as ω drifts (mag-only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for r in per_seed:
        y = r["truth_q0_cost"][dir_indices]
        ax.plot(dir_rots, y, "o-", color=colors[str(r["seed"])],
                label=f"seed {r['seed']}")
    ax.set_xlabel("ω-dir rotation [deg]")
    ax.set_ylabel("truth-q0 alignment cost  [unitless]")
    ax.set_title("how badly truth-q0 fits as ω drifts (dir-only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "s004 — m103 alignment-cost landscape under ω-misspecification",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


def save_landscape_panels(per_seed, path):
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
            cost = r["full_cost"][idx]
            geo = r["geo_to_truth_deg"]

            ax.scatter(geo[1:], cost[1:], s=3, alpha=0.45, color="darkorange",
                       linewidths=0)
            ax.scatter([0], [cost[0]], marker="*", s=100, color="crimson",
                       zorder=10, label=f"truth-q0={cost[0]:.2e}")
            am = int(r["argmin_idx"][idx])
            ax.scatter([geo[am]], [cost[am]], marker="X", s=70, color="black",
                       zorder=11, label=f"argmin={cost[am]:.2e} @ {geo[am]:.0f}°")

            ax.set_yscale("log")
            if si == n_seeds - 1:
                ax.set_xlabel("geodesic to truth-q0  [deg]")
            if pi == 0:
                ax.set_ylabel(f"seed {r['seed']}\nalignment cost")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "s004 — q0 alignment-cost landscape per seed at representative ω perturbations",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
