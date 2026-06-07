"""s034 — LM polish on top-50 surrogate-MSE survivors of seed 89.

Loads the s033_n2000_m20_smoke run (N_dir=2000, N_mag=20, N_phi=12, 4170
survivors), ranks by surrogate-MSE, takes top-50, runs joint (q0, omega)
LM polish per candidate (scipy.least_squares method='lm'), then hi-fi
reranks the polished candidates.

Decision-cost test: does LM polish bridge the 7.86° q0 floor we hit at
this density level, or is the off-circle distance structurally
unbridgeable?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry

SEED = 89
RUN_DIR = SURVEY_DIR / "results" / "s033_n2000_m20_smoke" / f"seed{SEED:03d}"
TRAJ_PATH = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz"
OUT_DIR = SURVEY_DIR / "results" / "s034_lm_polish_seed089"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOP = 50
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
LM_MAX_NFEV = 500


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def quat_to_R_i2b_batch(q_arr):
    """(M, 4) wxyz → (M, 3, 3) inertial→body rotation matrices.

    Matches s020's helper: scipy returns R_i2b directly under the post-fix
    quaternion convention; no transpose.
    """
    qxyzw = q_arr[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def build_residual(q0_init, omega_init, obs_times, inertia,
                   sun_unit, obs_unit, obs_dist, mag_truth, valid_mask,
                   surrogate):
    """Build residual function (q0_pert, omega_delta) → (N_valid,) residuals.

    Parameterisation: x = [rx, ry, rz, dwx, dwy, dwz]
      - q0_new = Rotation.from_rotvec([rx,ry,rz]) ⊗ q0_init
      - omega_new = omega_init + [dwx, dwy, dwz]
    """
    mag_truth_valid = mag_truth[valid_mask]

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        # LEFT multiply: q0_new = q_pert ⊗ q0_init
        q0_new = quat_mul(q_pert_wxyz, q0_init)
        omega_new = omega_init + omega_delta

        # Forward propagate
        q_traj, _ = propagate_attitude(
            q0_new, omega_new, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        # q_traj is (N_obs, 4) wxyz
        R_full = quat_to_R_i2b_batch(q_traj)
        # k1, k2 in body frame: R @ vec_inertial
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)

        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid_mask] - mag_truth_valid).astype(np.float64)

    return residual


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def main():
    t_start = time.time()

    # --- Load survivor data ---
    print(f"[s034] Loading survivor data from {RUN_DIR}...", flush=True)
    surv = np.load(RUN_DIR / "survivor_lcs.npz")
    cands = np.load(RUN_DIR / "candidates_meta.npz")
    omega_grid = np.load(RUN_DIR / "omega_grid.npz")
    truth_full = np.load(TRAJ_PATH)

    surv_idx = surv["survivor_cand_idx"]
    surv_mag_pred = surv["survivor_mag_pred"]
    truth_mag_hifi = surv["truth_mag_hifi"]
    obs_times = surv["observation_times"]
    valid_mask = np.isfinite(truth_mag_hifi)
    print(f"  {len(surv_idx)} survivors, {valid_mask.sum()}/{len(truth_mag_hifi)} valid epochs")

    # Surrogate MSE per survivor
    mse_all = np.mean((surv_mag_pred[:, valid_mask]
                       - truth_mag_hifi[None, valid_mask])**2, axis=1)
    order = np.argsort(mse_all)[:N_TOP]
    print(f"  top-{N_TOP} surrogate-MSE range: "
          f"{mse_all[order[0]]:.4f} - {mse_all[order[-1]]:.4f} "
          f"(predicted ρ {np.sqrt(mse_all[order[0]])/0.05:.2f} - "
          f"{np.sqrt(mse_all[order[-1]])/0.05:.2f})")

    # Pull q0 + omega for each top candidate
    cand_q0_all = cands["q0"]
    cand_omega_idx = cands["omega_cell_idx"]
    omega_vectors = omega_grid["omega_vectors"]

    top_global_idx = surv_idx[order]
    top_q0_init = cand_q0_all[top_global_idx].astype(np.float64)
    top_omega_init = omega_vectors[cand_omega_idx[top_global_idx]].astype(np.float64)
    top_mse_init = mse_all[order]

    # --- Truth references ---
    q0_truth = truth_full["q0_wxyz"].astype(np.float64)
    omega_truth = truth_full["omega0_rad"].astype(np.float64)
    print(f"  truth q0: {q0_truth}")
    print(f"  truth omega: {omega_truth} (mag {np.linalg.norm(omega_truth)*180/np.pi:.4f} dps)")

    # --- Load forward-model inputs ---
    sun_pos = truth_full["sun_pos"]
    obs_pos = truth_full["obs_pos"]
    sat_pos = truth_full["sat_pos"]
    obs_dist = truth_full["obs_dist"].astype(np.float64)
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)).astype(np.float64)
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)).astype(np.float64)

    # Inertia
    geo = load_static_geometry()
    inertia = geo["inertia_tensor"].astype(np.float64)
    print(f"  inertia diagonal: {np.diag(inertia)}")

    # Surrogate
    from lib.surrogate_eval import get_model
    surrogate = get_model()
    print(f"  surrogate loaded")

    # --- LM polish each top candidate ---
    print(f"\n[s034] Running LM polish on top-{N_TOP} candidates...", flush=True)
    polished = []
    for i in range(N_TOP):
        q0_init = top_q0_init[i] / np.linalg.norm(top_q0_init[i])
        omega_init = top_omega_init[i]
        mse_init = top_mse_init[i]

        residual = build_residual(
            q0_init, omega_init, obs_times, inertia,
            sun_unit, obs_unit, obs_dist, truth_mag_hifi, valid_mask, surrogate,
        )

        x0 = np.zeros(6)
        t_lm = time.time()
        try:
            result = least_squares(
                residual, x0, method='lm',
                max_nfev=LM_MAX_NFEV,
                xtol=1e-8, ftol=1e-8,
            )
            converged = True
            n_iter = result.nfev
            mse_final = float(np.mean(result.fun ** 2))
            # Reconstruct optimized state
            rotvec = result.x[:3]
            omega_delta = result.x[3:]
            q_pert = Rotation.from_rotvec(rotvec).as_quat()
            q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
            q0_final = quat_mul(q_pert_wxyz, q0_init)
            omega_final = omega_init + omega_delta
        except Exception as e:
            converged = False
            n_iter = -1
            mse_final = mse_init
            q0_final = q0_init
            omega_final = omega_init
            print(f"  cand {i} FAILED: {e}")

        dt_lm = time.time() - t_lm
        # Twin: X-flip + omega-transform
        q_180x = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)  # rotation by pi about X
        twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q0_truth)
        R_180x = np.diag([1.0, -1.0, -1.0])
        twin_omega = R_180x @ omega_truth

        q0_to_truth = angular_dist_deg(q0_final, q0_truth)
        q0_to_twin = angular_dist_deg(q0_final, twin_q0)
        omega_dir_truth = np.degrees(np.arccos(np.clip(
            np.dot(omega_final, omega_truth) /
            (np.linalg.norm(omega_final) * np.linalg.norm(omega_truth)),
            -1, 1)))
        omega_mag_pct = (np.linalg.norm(omega_final) - np.linalg.norm(omega_truth)) \
                        / np.linalg.norm(omega_truth) * 100

        polished.append({
            "rank": i,
            "global_idx": int(top_global_idx[i]),
            "mse_init": float(mse_init),
            "mse_final": float(mse_final),
            "rho_pred_init": float(np.sqrt(mse_init) / 0.05),
            "rho_pred_final": float(np.sqrt(mse_final) / 0.05),
            "n_iter": int(n_iter),
            "lm_wall_s": float(dt_lm),
            "q0_init": q0_init.tolist(),
            "q0_final": q0_final.tolist(),
            "omega_init": omega_init.tolist(),
            "omega_final": omega_final.tolist(),
            "q0_to_truth_deg": float(q0_to_truth),
            "q0_to_twin_deg": float(q0_to_twin),
            "omega_dir_to_truth_deg": float(omega_dir_truth),
            "omega_mag_pct": float(omega_mag_pct),
            "converged": converged,
        })

        if i < 5 or i % 10 == 0:
            print(f"  rank {i:>2}: ρ {np.sqrt(mse_init)/0.05:>6.2f} → {np.sqrt(mse_final)/0.05:>6.2f}  "
                  f"q0→truth {q0_to_truth:>5.2f}°  ω-dir {omega_dir_truth:>5.2f}°  "
                  f"ω-mag {omega_mag_pct:+5.2f}%  iter={n_iter:>3d}  wall={dt_lm:.1f}s",
                  flush=True)

    # --- Save ---
    summary = {
        "seed": SEED,
        "source_run": str(RUN_DIR),
        "n_top": N_TOP,
        "n_total_survivors": len(surv_idx),
        "wall_total_s": time.time() - t_start,
        "polished": polished,
    }
    out_json = OUT_DIR / "polish_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[s034] Saved: {out_json}")

    # Quick verdict
    rho_finals = sorted(p["rho_pred_final"] for p in polished)
    rho_inits = sorted(p["rho_pred_init"] for p in polished)
    q0_finals = sorted(min(p["q0_to_truth_deg"], p["q0_to_twin_deg"]) for p in polished)
    print(f"\n=== VERDICT ===")
    print(f"Predicted ρ (init→final): best {rho_inits[0]:.2f} → {rho_finals[0]:.2f}, "
          f"median {rho_inits[N_TOP//2]:.2f} → {rho_finals[N_TOP//2]:.2f}")
    print(f"q0→nearest(truth/twin) (final): best {q0_finals[0]:.2f}°, "
          f"median {q0_finals[N_TOP//2]:.2f}°, max {q0_finals[-1]:.2f}°")
    n_band_a = sum(1 for r in rho_finals if r < 2)
    n_band_b = sum(1 for r in rho_finals if 2 <= r < 4)
    n_band_c = sum(1 for r in rho_finals if 4 <= r < 8)
    n_band_d = sum(1 for r in rho_finals if r >= 8)
    print(f"Predicted ρ-band (final): A={n_band_a} B={n_band_b} C={n_band_c} D={n_band_d}")
    print(f"\nTotal wall: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
