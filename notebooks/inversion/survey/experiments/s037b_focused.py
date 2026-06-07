"""s037b focused: just seed 23 L1 + seed 28 L0 (the two critical tests).

Seed 23 L0 already confirmed (9/64 Band A at truth-ω). Remaining questions:
- Seed 23 L1: does the 5.64% bracket offset break convergence?
- Seed 28 L0: can Sobol+LM recover the narrow-basin seed at all?

Usage:
    cd notebooks/inversion/survey
    python experiments/s037b_focused.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "false"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.traj_load import load_truth
from lib.filter_costs import load_static_geometry
from lib.surrogate_eval import get_model

N_SOBOL = 64
SOBOL_SEED = 42
LM_MAX_NFEV = 200
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0

RESULTS_DIR = SURVEY_DIR / "results" / "s037b_sobol_lm_pilot"


def shoemake_to_quat(u):
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


def build_sobol_q0(n, sobol_seed):
    sobol = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    u = sobol.random(n)
    return shoemake_to_quat(u)


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def run_lm_level(label, q0_batch, omega_start, truth, inertia, surrogate):
    """Run LM on all Sobol ICs at given omega. Print per-IC progress."""
    times = truth["observation_times"]
    sun_pos = truth["sun_pos"]
    obs_pos = truth["obs_pos"]
    sat_pos = truth["sat_pos"]
    obs_dist = truth["obs_dist"]
    mag_truth = truth["mag_hifi"]
    valid = np.isfinite(mag_truth)
    mag_truth_valid = mag_truth[valid]
    q0_truth = truth["q0_wxyz"]
    omega_truth = truth["omega0_rad"]
    omega_truth_mag = np.linalg.norm(omega_truth)

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True))
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True))

    results = []
    t_total = time.time()

    for i in range(len(q0_batch)):
        q0_ic = q0_batch[i] / np.linalg.norm(q0_batch[i])

        def residual(x, _q0_ic=q0_ic):
            rotvec = x[:3]
            omega = x[3:6]
            q_pert = Rotation.from_rotvec(rotvec).as_quat()
            q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
            q0 = quat_mul(q_pert_wxyz, _q0_ic)
            try:
                q_traj, _ = propagate_attitude(
                    q0, omega, times, mode="tumbling", inertia_tensor=inertia)
                qxyzw = q_traj[:, [1, 2, 3, 0]]
                R_full = Rotation.from_quat(qxyzw).as_matrix()
                k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
                k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)
                mag_pred = surrogate.predict_magnitude(
                    k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist)
                return (mag_pred[valid] - mag_truth_valid).astype(np.float64)
            except Exception:
                return np.full(valid.sum(), 1e3, dtype=np.float64)

        x0 = np.concatenate([np.zeros(3), omega_start])
        t0 = time.time()
        try:
            result = least_squares(residual, x0, method='lm',
                                   max_nfev=LM_MAX_NFEV, xtol=1e-8, ftol=1e-8)
            mse_f = float(np.mean(result.fun ** 2))
            rotvec = result.x[:3]
            omega_f = result.x[3:6]
            q_pert = Rotation.from_rotvec(rotvec).as_quat()
            q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
            q0_f = quat_mul(q_pert_wxyz, q0_ic)
            nfev = result.nfev
        except Exception:
            mse_f = np.inf
            q0_f = q0_ic
            omega_f = omega_start
            nfev = -1
        dt = time.time() - t0

        rho_f = float(np.sqrt(mse_f) / 0.05) if mse_f < np.inf else np.inf
        q0_err = angular_dist_deg(q0_f, q0_truth)
        omega_dir_err = np.degrees(np.arccos(np.clip(
            np.dot(omega_f, omega_truth) /
            (np.linalg.norm(omega_f) * np.linalg.norm(omega_truth) + 1e-30), -1, 1)))
        omega_mag_pct = (np.linalg.norm(omega_f) - omega_truth_mag) / omega_truth_mag * 100

        results.append({
            "ic_idx": i, "rho_final": rho_f,
            "q0_to_truth_deg": q0_err,
            "omega_dir_to_truth_deg": float(omega_dir_err),
            "omega_mag_pct": float(omega_mag_pct),
            "nfev": nfev, "wall_s": dt,
            "q0_final": q0_f.tolist(),
            "omega_final": omega_f.tolist(),
        })

        if rho_f < 2.0 or i % 16 == 0:
            print(f"    IC {i:>2}: ρ={rho_f:>7.3f} q0→t={q0_err:>6.2f}° "
                  f"ω-dir={omega_dir_err:>5.2f}° ω-mag={omega_mag_pct:>+6.2f}% "
                  f"nfev={nfev:>3d} {dt:.1f}s"
                  f"{' *** BAND A ***' if rho_f < 2.0 else ''}", flush=True)

    wall = time.time() - t_total
    rhos = np.array([r["rho_final"] for r in results])
    n_a = int(np.sum(rhos < 2))
    n_sub15 = int(np.sum(rhos < 15))
    print(f"\n  [{label}] DONE wall={wall:.0f}s | min ρ={rhos.min():.3f} | "
          f"Band A={n_a}/64 | ρ<15={n_sub15}/64", flush=True)

    return results, wall


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    geo = load_static_geometry()
    inertia = geo["inertia_tensor"].astype(np.float64)
    surrogate = get_model()
    q0_sobol = build_sobol_q0(N_SOBOL, SOBOL_SEED)

    print(f"s037b focused: seed 23 L1 + seed 28 L0")
    print(f"  N_sobol={N_SOBOL}, LM_max_nfev={LM_MAX_NFEV}\n", flush=True)

    # ─── Seed 23 L1: bracket cell at 5.64% off truth ─���───────────────────────
    print(f"{'='*70}")
    print(f"  SEED 23 — L1 (nearest bracket cell)")
    print(f"{'='*70}", flush=True)

    truth_23 = load_truth(23)
    omega_truth_23 = truth_23["omega0_rad"]
    omega_truth_mag_23 = np.linalg.norm(omega_truth_23)
    omega_truth_dir_23 = omega_truth_23 / omega_truth_mag_23

    # Load bracket from s036/s032
    bracket_path = RESULTS_DIR.parent / "s036_multi_seed_pilot" / "seed023" / "bracket.npz"
    if not bracket_path.exists():
        bracket_path = RESULTS_DIR.parent / "s032_cohort_fast" / "seed023" / "bracket.npz"
    bracket = np.load(bracket_path)
    cells_23 = bracket["bracket_cells"]
    nearest_idx = np.argmin(np.abs(cells_23 - omega_truth_mag_23))
    nearest_cell = cells_23[nearest_idx]
    nearest_pct = abs(nearest_cell - omega_truth_mag_23) / omega_truth_mag_23 * 100
    omega_l1_23 = omega_truth_dir_23 * nearest_cell

    print(f"  omega truth: {np.degrees(omega_truth_mag_23):.4f} dps")
    print(f"  nearest bracket cell: {np.degrees(nearest_cell):.4f} dps ({nearest_pct:.2f}% off)")
    print(f"  omega L1 = truth-dir × bracket-mag\n", flush=True)

    res_23_l1, wall_23 = run_lm_level("S23-L1", q0_sobol, omega_l1_23,
                                       truth_23, inertia, surrogate)

    # Save
    out_23 = RESULTS_DIR / "seed023_L1_result.json"
    with open(out_23, "w") as f:
        json.dump({"seed": 23, "level": "L1", "nearest_cell_pct": nearest_pct,
                   "wall_s": wall_23, "results": res_23_l1}, f, indent=2, default=float)
    print(f"  Saved: {out_23}", flush=True)

    # ─── Seed 28 L0: truth-ω (narrow basin test) ──��──────────────────────────
    print(f"\n{'='*70}")
    print(f"  SEED 28 — L0 (truth-ω)")
    print(f"{'='*70}", flush=True)

    truth_28 = load_truth(28)
    omega_truth_28 = truth_28["omega0_rad"]
    omega_truth_mag_28 = np.linalg.norm(omega_truth_28)

    print(f"  omega truth: {np.degrees(omega_truth_mag_28):.4f} dps")
    print(f"  (known narrow-basin seed — s006 ~2° radius)\n", flush=True)

    res_28_l0, wall_28 = run_lm_level("S28-L0", q0_sobol, omega_truth_28,
                                       truth_28, inertia, surrogate)

    # Save
    out_28 = RESULTS_DIR / "seed028_L0_result.json"
    with open(out_28, "w") as f:
        json.dump({"seed": 28, "level": "L0", "wall_s": wall_28,
                   "results": res_28_l0}, f, indent=2, default=float)
    print(f"  Saved: {out_28}", flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    rhos_23 = [r["rho_final"] for r in res_23_l1]
    rhos_28 = [r["rho_final"] for r in res_28_l0]
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Seed 23 L1 (bracket {nearest_pct:.1f}% off): "
          f"min ρ={min(rhos_23):.3f}, Band A={sum(1 for r in rhos_23 if r<2)}/64")
    print(f"  Seed 28 L0 (truth-ω): "
          f"min ρ={min(rhos_28):.3f}, Band A={sum(1 for r in rhos_28 if r<2)}/64")
    print(f"  Total wall: {wall_23 + wall_28:.0f}s", flush=True)


if __name__ == "__main__":
    main()
