"""s024 — Alignment-cost as ω-prior (NEW USE — localizer not just filter).

Idea: alignment cost depends primarily on ω (peak timing). If we marginalise
over q0 — take the BEST alignment cost over a small Sobol-q0 sample — we
get a per-ω-cell score. If truth-ω lights up cleanly, alignment cost is
not just a filter but a localizer for ω. Could be a stronger ω-prior than
the LS-bracket (s019), which only addresses ω-mag.

Pilot scope:
  3 stratified seeds (small / large / mid n_rotations + tier diversity).
  ω-grid: ω-dir perturbation around truth (50 cells from S²-Sobol +
          shells at radii [2°, 5°, 15°, 30°, 90°, 180°]) × 3 ω-mag values
          (0.5×, 1×, 2× truth-ω-mag).
  q0 ICs: 32 Sobol-Shoemake per ω-cell, score MAX alignment_cost.

Output: heatmap of ω-cell scores; peak location vs truth-ω.
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

from lib.traj_load import load_truth                # noqa: E402
from lib import filter_costs as fc                   # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s024"
RESULTS.mkdir(parents=True, exist_ok=True)

# Stratified pilot seeds
PILOT_SEEDS = [6, 28, 44]  # density-rec / sub-Sobol / tier-rich

N_Q0_PER_CELL = 32
N_WORKERS = 8
ALIGN_BRIGHT_MAG = 11.0
ALIGN_WINDOW_EPOCHS = 3
GEO_THRESHOLD_DEG = 5.0

_STATIC = None
_TIER = None


def _init_worker():
    global _STATIC, _TIER
    _STATIC = fc.load_static_geometry()
    _TIER = fc.load_tier_table()
    from lib.surrogate_eval import get_model
    get_model()


def _build_omega_grid(omega_truth_rad: np.ndarray):
    """Construct the ω-cell grid: perturbation shells + Sobol-S² + truth.

    Returns (n_cells, 3) array of omega vectors in rad/s, plus per-cell
    metadata (dir_radius_deg, mag_ratio).
    """
    omega_mag_rad = float(np.linalg.norm(omega_truth_rad))
    if omega_mag_rad < 1e-9:
        raise ValueError("zero truth omega")
    truth_dir = omega_truth_rad / omega_mag_rad

    # ω-dir perturbations
    rng = np.random.default_rng(20260504)
    dirs = [truth_dir]
    radii_deg = [0.0]
    # Shells
    for r_deg in [2, 5, 15, 30, 90, 180]:
        # Sample 8 azimuths around truth at fixed perturbation angle
        # Build orthonormal frame with truth_dir as one axis
        if abs(truth_dir[2]) < 0.9:
            ortho1 = np.cross(truth_dir, [0, 0, 1])
        else:
            ortho1 = np.cross(truth_dir, [1, 0, 0])
        ortho1 /= np.linalg.norm(ortho1)
        ortho2 = np.cross(truth_dir, ortho1)
        for k in range(8):
            azim = 2 * np.pi * k / 8
            r = np.deg2rad(r_deg)
            d = (np.cos(r) * truth_dir
                 + np.sin(r) * (np.cos(azim) * ortho1 + np.sin(azim) * ortho2))
            d /= np.linalg.norm(d)
            dirs.append(d)
            radii_deg.append(r_deg)
    # Add random S²-Sobol coverage (8 extra)
    for _ in range(8):
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v)
        dirs.append(v)
        ang = np.degrees(np.arccos(np.clip(np.dot(v, truth_dir), -1, 1)))
        radii_deg.append(float(ang))
    dirs = np.array(dirs)
    radii_deg = np.array(radii_deg)

    # ω-mag perturbations
    mag_ratios = np.array([0.5, 1.0, 2.0])

    cells = []
    cell_dir_rad_deg = []
    cell_mag_ratio = []
    for i, d in enumerate(dirs):
        for r in mag_ratios:
            cells.append(omega_mag_rad * r * d)
            cell_dir_rad_deg.append(radii_deg[i])
            cell_mag_ratio.append(r)
    return (np.array(cells),
            np.array(cell_dir_rad_deg),
            np.array(cell_mag_ratio))


def _generate_q0_sobol(n: int, seed: int = 0):
    """Sobol-Shoemake on SO(3): n quaternions in (w, x, y, z) order."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    u = sampler.random(n)  # (n, 3) in [0,1]
    sqrt_1mu0 = np.sqrt(1.0 - u[:, 0])
    sqrt_u0 = np.sqrt(u[:, 0])
    qs = np.stack([
        sqrt_1mu0 * np.sin(2 * np.pi * u[:, 1]),
        sqrt_1mu0 * np.cos(2 * np.pi * u[:, 1]),
        sqrt_u0 * np.sin(2 * np.pi * u[:, 2]),
        sqrt_u0 * np.cos(2 * np.pi * u[:, 2]),
    ], axis=1)
    # Convert from (xi, xj, xk, xs) to (w, x, y, z) — spec is (z3=cos(2π u2)*√u0
    # is the scalar w; rearrange)
    # Per Shoemake (1992), components are (s1*sin(2π u1), s1*cos(2π u1),
    #                                       s2*sin(2π u2), s2*cos(2π u2))
    # where s1=√(1-u0), s2=√u0; the scalar is s2*cos(2π u2) → component idx 3.
    # So (w, x, y, z) = (qs[:,3], qs[:,0], qs[:,1], qs[:,2]).
    qwxyz = np.stack([qs[:, 3], qs[:, 0], qs[:, 1], qs[:, 2]], axis=1)
    qwxyz /= np.linalg.norm(qwxyz, axis=1, keepdims=True)
    return qwxyz


def _score_one_cell(args):
    """Score MAX alignment cost over q0 ICs at a single ω-cell."""
    seed, omega_cell, q0_panel = args

    truth = load_truth(seed)
    seed_data = fc.precompute_seed_filter_data(
        truth, _TIER, bright_mag_threshold=ALIGN_BRIGHT_MAG,
    )

    align_scores = np.zeros(q0_panel.shape[0])
    geo_scores = np.zeros(q0_panel.shape[0])
    for i, q0 in enumerate(q0_panel):
        res = fc.evaluate_candidate(
            q0, omega_cell, seed_data,
            _STATIC["inertia_tensor"], _STATIC["face_normals"], _TIER["tier_face_idx"],
            align_window_epochs=ALIGN_WINDOW_EPOCHS,
            align_bright_mag=ALIGN_BRIGHT_MAG,
            geo_threshold_deg=GEO_THRESHOLD_DEG,
        )
        align_scores[i] = res["score_alignment"] if np.isfinite(res["score_alignment"]) else 0.0
        geo_scores[i] = res["score_geo"] if np.isfinite(res["score_geo"]) else 0.0
    return {
        "seed": seed,
        "max_align": float(np.max(align_scores)),
        "mean_align": float(np.mean(align_scores)),
        "max_geo": float(np.max(geo_scores)),
        "mean_geo": float(np.mean(geo_scores)),
    }


def main():
    print("=" * 72)
    print(f"s024 — Alignment-cost ω-prior pilot ({len(PILOT_SEEDS)} seeds)")
    print("=" * 72)

    # Build q0 panel (shared across seeds and cells)
    q0_panel = _generate_q0_sobol(N_Q0_PER_CELL)
    print(f"  q0 panel size: {q0_panel.shape}")

    all_results = {}

    for seed in PILOT_SEEDS:
        truth = load_truth(seed)
        omega_truth = truth["omega0_rad"]
        cells, radii_deg, mag_ratios = _build_omega_grid(omega_truth)
        print(f"\n--- seed {seed}: |ω|={float(truth['omega_mag_dps']):.3f} dps "
              f"| {cells.shape[0]} ω-cells × {N_Q0_PER_CELL} q0 ICs "
              f"= {cells.shape[0]*N_Q0_PER_CELL} evals ---")

        args_list = [(seed, cells[i], q0_panel) for i in range(cells.shape[0])]
        t0 = time.time()
        with Pool(N_WORKERS, initializer=_init_worker) as pool:
            results = []
            for j, r in enumerate(pool.imap(_score_one_cell, args_list)):
                results.append(r)
        wall = time.time() - t0
        print(f"  wall: {wall:.1f}s")

        max_align = np.array([r["max_align"] for r in results])
        max_geo = np.array([r["max_geo"] for r in results])

        # Save per-seed
        np.savez_compressed(
            RESULTS / f"seed{seed:03d}.npz",
            cells=cells,
            radii_deg=radii_deg,
            mag_ratios=mag_ratios,
            max_align=max_align,
            max_geo=max_geo,
            mean_align=np.array([r["mean_align"] for r in results]),
            mean_geo=np.array([r["mean_geo"] for r in results]),
            omega_truth=omega_truth,
        )

        # Locate best cell
        best_idx = int(np.argmax(max_align))
        best_align = max_align[best_idx]
        best_radius = radii_deg[best_idx]
        best_mag_ratio = mag_ratios[best_idx]

        # Truth cell index (radius=0 AND mag_ratio=1.0)
        truth_cell = np.where((radii_deg == 0.0) & (mag_ratios == 1.0))[0]
        truth_align = float(max_align[truth_cell[0]]) if truth_cell.size else float("nan")

        all_results[seed] = {
            "n_cells": int(cells.shape[0]),
            "wall_s": float(wall),
            "max_align_at_truth_cell": truth_align,
            "max_align_anywhere": float(best_align),
            "best_cell_radius_deg": float(best_radius),
            "best_cell_mag_ratio": float(best_mag_ratio),
            "best_aligns_with_truth": bool(best_radius < 5.0 and 0.8 < best_mag_ratio < 1.2),
        }
        print(f"  truth-cell max_align: {truth_align:.3f}")
        print(f"  global max_align:     {best_align:.3f} at "
              f"radius={best_radius:.0f}° mag_ratio={best_mag_ratio:.2f}")
        print(f"  best aligns with truth: {all_results[seed]['best_aligns_with_truth']}")

    summary = {
        "pilot_seeds": PILOT_SEEDS,
        "n_q0_per_cell": N_Q0_PER_CELL,
        "params": {
            "align_bright_mag": ALIGN_BRIGHT_MAG,
            "align_window_epochs": ALIGN_WINDOW_EPOCHS,
        },
        "per_seed": all_results,
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
