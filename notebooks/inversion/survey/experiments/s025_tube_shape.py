"""s025 — Tube-shape characterization.

For 5 stratified seeds, perturb truth (q0, ω) on each axis independently
and measure how each filter's score rolls off. Defines the natural cell
size for any filter-based grid search.

Perturbation axes (one at a time, others at truth):
  Δq0:      {0°, 1°, 5°, 15°, 45°, 90°, 180°}, 8 random directions per shell
  Δω-dir:   {0°, 1°, 5°, 30°, 90°, 180°}, 8 azimuths per shell
  Δω-mag:   {-50%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +50%}

For each cell, score (alignment, geo) from a single evaluation (no Sobol).
Plot: per-axis roll-off curves for each metric.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.traj_load import load_truth                # noqa: E402
from lib import filter_costs as fc                   # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s025"
RESULTS.mkdir(parents=True, exist_ok=True)

PILOT_SEEDS = [6, 28, 41, 44, 91]
ALIGN_BRIGHT_MAG = 11.0
ALIGN_WINDOW_EPOCHS = 3
GEO_THRESHOLD_DEG = 5.0
N_WORKERS = 8

DQ0_DEG = [0, 1, 5, 15, 45, 90, 180]
DOMEGA_DIR_DEG = [0, 1, 5, 30, 90, 180]
DOMEGA_MAG_PCT = [-50, -20, -10, -5, 0, 5, 10, 20, 50]
N_AZIM = 8

_STATIC = None
_TIER = None


def _init_worker():
    global _STATIC, _TIER
    _STATIC = fc.load_static_geometry()
    _TIER = fc.load_tier_table()
    from lib.surrogate_eval import get_model
    get_model()


def _perturb_q0(q0_truth_wxyz, angle_deg, axis):
    """Apply a rotation of `angle_deg` about `axis` (3-vec, body-frame) to q0."""
    if angle_deg == 0:
        return q0_truth_wxyz.copy()
    half = np.deg2rad(angle_deg) / 2
    s = np.sin(half)
    q_pert = np.array([np.cos(half), s * axis[0], s * axis[1], s * axis[2]])
    # LEFT-multiply (body-frame perturbation per quaternion convention)
    import quaternion as q_pkg
    q1 = q_pkg.quaternion(*q_pert)
    q2 = q_pkg.quaternion(*q0_truth_wxyz)
    out = q1 * q2
    return np.array([out.w, out.x, out.y, out.z])


def _perturb_omega_dir(omega_truth_rad, angle_deg, azim_idx):
    """Rotate ω-direction by angle_deg, keep magnitude."""
    if angle_deg == 0:
        return omega_truth_rad.copy()
    omega_mag = np.linalg.norm(omega_truth_rad)
    truth_dir = omega_truth_rad / omega_mag
    if abs(truth_dir[2]) < 0.9:
        ortho1 = np.cross(truth_dir, [0, 0, 1])
    else:
        ortho1 = np.cross(truth_dir, [1, 0, 0])
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(truth_dir, ortho1)
    azim = 2 * np.pi * azim_idx / N_AZIM
    r = np.deg2rad(angle_deg)
    new_dir = (np.cos(r) * truth_dir
               + np.sin(r) * (np.cos(azim) * ortho1 + np.sin(azim) * ortho2))
    new_dir /= np.linalg.norm(new_dir)
    return omega_mag * new_dir


def _score_one(args):
    seed, q0, omega = args
    truth = load_truth(seed)
    seed_data = fc.precompute_seed_filter_data(
        truth, _TIER, bright_mag_threshold=ALIGN_BRIGHT_MAG,
    )
    res = fc.evaluate_candidate(
        q0, omega, seed_data,
        _STATIC["inertia_tensor"], _STATIC["face_normals"], _TIER["tier_face_idx"],
        align_window_epochs=ALIGN_WINDOW_EPOCHS,
        align_bright_mag=ALIGN_BRIGHT_MAG,
        geo_threshold_deg=GEO_THRESHOLD_DEG,
    )
    return (res["score_alignment"] if np.isfinite(res["score_alignment"]) else 0.0,
            res["score_geo"] if np.isfinite(res["score_geo"]) else 0.0)


def main():
    print("=" * 72)
    print(f"s025 — Tube-shape characterization ({len(PILOT_SEEDS)} seeds)")
    print("=" * 72)

    rng = np.random.default_rng(20260504)

    per_seed = {}
    for seed in PILOT_SEEDS:
        truth = load_truth(seed)
        q0_t = truth["q0_wxyz"]
        omega_t = truth["omega0_rad"]
        omega_mag_t = float(np.linalg.norm(omega_t))

        # Build job list
        jobs = []
        meta = []
        # Δq0 axis
        for ang in DQ0_DEG:
            for k in range(N_AZIM):
                axis = rng.standard_normal(3)
                axis /= np.linalg.norm(axis)
                q0 = _perturb_q0(q0_t, ang, axis)
                jobs.append((seed, q0, omega_t))
                meta.append({"axis": "q0", "angle_deg": ang, "azim_idx": k})
        # Δω-dir axis
        for ang in DOMEGA_DIR_DEG:
            for k in range(N_AZIM):
                omega_p = _perturb_omega_dir(omega_t, ang, k)
                jobs.append((seed, q0_t, omega_p))
                meta.append({"axis": "omega_dir", "angle_deg": ang, "azim_idx": k})
        # Δω-mag axis
        for pct in DOMEGA_MAG_PCT:
            omega_p = omega_t * (1.0 + pct / 100.0)
            jobs.append((seed, q0_t, omega_p))
            meta.append({"axis": "omega_mag", "pct": pct, "azim_idx": 0})

        print(f"\nseed {seed}: {len(jobs)} perturbation cells")
        t0 = time.time()
        with Pool(N_WORKERS, initializer=_init_worker) as pool:
            scores = pool.map(_score_one, jobs)
        wall = time.time() - t0
        scores = np.array(scores)  # (n, 2)

        # Bin by axis + angle
        rolloff = {"q0": {}, "omega_dir": {}, "omega_mag": {}}
        for ang in DQ0_DEG:
            mask = np.array([m["axis"] == "q0" and m["angle_deg"] == ang for m in meta])
            if mask.sum() > 0:
                rolloff["q0"][ang] = {
                    "align_mean": float(np.mean(scores[mask, 0])),
                    "align_min": float(np.min(scores[mask, 0])),
                    "align_max": float(np.max(scores[mask, 0])),
                    "geo_mean": float(np.mean(scores[mask, 1])),
                    "geo_max": float(np.max(scores[mask, 1])),
                }
        for ang in DOMEGA_DIR_DEG:
            mask = np.array([m["axis"] == "omega_dir" and m["angle_deg"] == ang for m in meta])
            if mask.sum() > 0:
                rolloff["omega_dir"][ang] = {
                    "align_mean": float(np.mean(scores[mask, 0])),
                    "align_min": float(np.min(scores[mask, 0])),
                    "align_max": float(np.max(scores[mask, 0])),
                    "geo_mean": float(np.mean(scores[mask, 1])),
                    "geo_max": float(np.max(scores[mask, 1])),
                }
        for pct in DOMEGA_MAG_PCT:
            mask = np.array([m["axis"] == "omega_mag" and m.get("pct") == pct for m in meta])
            if mask.sum() > 0:
                rolloff["omega_mag"][pct] = {
                    "align_mean": float(np.mean(scores[mask, 0])),
                    "geo_mean": float(np.mean(scores[mask, 1])),
                }

        per_seed[seed] = {"wall_s": float(wall), "rolloff": rolloff}
        print(f"  wall: {wall:.1f}s")

        # Save per-seed scores
        np.savez_compressed(
            RESULTS / f"seed{seed:03d}.npz",
            scores=scores,
            meta_axis=np.array([m["axis"] for m in meta]),
            meta_angle=np.array([m.get("angle_deg", m.get("pct", 0)) for m in meta]),
            meta_azim=np.array([m["azim_idx"] for m in meta]),
        )

    with open(RESULTS / "summary.json", "w") as f:
        json.dump({"pilot_seeds": PILOT_SEEDS, "per_seed": per_seed}, f, indent=2)
    print(f"\nSaved: {RESULTS / 'summary.json'}")

    # Plot roll-off
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, (axis, x_vals, x_label) in enumerate([
        ("q0", DQ0_DEG, "Δq0 (°)"),
        ("omega_dir", DOMEGA_DIR_DEG, "Δω-dir (°)"),
        ("omega_mag", DOMEGA_MAG_PCT, "Δω-mag (%)"),
    ]):
        for seed in PILOT_SEEDS:
            roll = per_seed[seed]["rolloff"][axis]
            xs = sorted(roll.keys())
            align = [roll[x]["align_mean"] for x in xs]
            geo = [roll[x].get("geo_mean", np.nan) for x in xs]
            axs[0, col].plot(xs, align, marker="o", label=f"seed {seed}")
            axs[1, col].plot(xs, geo, marker="o", label=f"seed {seed}")
        axs[0, col].set_xlabel(x_label)
        axs[0, col].set_ylabel("alignment (mean)")
        axs[0, col].set_title(f"alignment vs {axis}")
        axs[0, col].set_ylim(-0.05, 1.05)
        axs[1, col].set_xlabel(x_label)
        axs[1, col].set_ylabel("geo (mean)")
        axs[1, col].set_title(f"geo vs {axis}")
        axs[1, col].set_ylim(-0.05, 1.05)
        axs[0, col].legend(fontsize=8)
        axs[1, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "tube_shape.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'tube_shape.png'}")


if __name__ == "__main__":
    main()
