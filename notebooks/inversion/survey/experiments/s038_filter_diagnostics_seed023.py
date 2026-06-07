"""s038 — phi-sweep IC + filter diagnostic on seed 23 at the near-truth ω-cell.

Answers the gaps from s038_handoff_diagnosis_gaps.md without an architecture change:

  1. At the bracket cell nearest truth-ω (mag 5.64% off, dir 2.26° off), score the
     528 phi-sweep ICs on geo + align + pre-LM surrogate-MSE. Tabulate vs q0_err
     to truth/twin. Did the filter reject any IC inside the LM convergence basin?

  2. Bypass the filter. Run LM polish on the top-K ICs by closeness-to-truth-q0
     and top-K by lowest pre-LM surrogate-MSE. Count Band A converged.

  3. Compare to s037b L1 (Sobol N=64 + LM at the same ω-cell) which got 9/64 Band A.

Strict checkpointing: per-IC table CSV + summary JSON + diagnostic plot.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DOMAIN_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
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

SEED = 23
S036_DIR = SURVEY_DIR / "results" / "s036_multi_seed_pilot" / f"seed{SEED:03d}"
TRAJ_PATH = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz"
OUT_DIR = SURVEY_DIR / "results" / f"s038_filter_diagnostics_seed{SEED:03d}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
LM_MAX_NFEV = 200          # match s037b ceiling


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_R_i2b_batch(q_arr_wxyz):
    qxyzw = q_arr_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def fmt_state(label, q0, omega):
    """Quick state summary for log lines."""
    return (f"{label}: q={q0[0]:+.3f}{q0[1]:+.3f}{q0[2]:+.3f}{q0[3]:+.3f} "
            f"ω={omega[0]:+.5f}{omega[1]:+.5f}{omega[2]:+.5f}")


# ------------- Stage A: per-IC scoring at the near-truth ω-cell -------------
def stage_a_score_ics(verbose=True):
    """Slice cached pool for the near-truth ω-cell, build a per-IC table.

    Outputs:
      ic_table: structured arrays of length 528 with columns
        ic_idx, peak_epoch, face_idx, phi_idx, tier_idx,
        q0(4), q0_err_truth_deg, q0_err_twin_deg, geo_score, align_score,
        surrogate_mse, rho_pred.
    """
    if verbose:
        print("[s038-A] Loading cached seed-23 IC pool...", flush=True)
    bracket = np.load(S036_DIR / "bracket.npz")
    omega_grid = np.load(S036_DIR / "omega_grid.npz")
    q_pool = np.load(S036_DIR / "q_target_pool.npz")
    spec_geo = np.load(S036_DIR / "spec_geometry.npz")
    cands = np.load(S036_DIR / "candidates_meta.npz", mmap_mode="r")
    truth_full = np.load(TRAJ_PATH)

    obs_times = truth_full["observation_times"].astype(np.float64)
    sun_pos = truth_full["sun_pos"].astype(np.float64)
    obs_pos = truth_full["obs_pos"].astype(np.float64)
    sat_pos = truth_full["sat_pos"].astype(np.float64)
    obs_dist = truth_full["obs_dist"].astype(np.float64)
    mag_truth = truth_full["mag_hifi"].astype(np.float64)
    valid_mask = np.isfinite(mag_truth)
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)).astype(np.float64)
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)).astype(np.float64)

    inertia = load_static_geometry()["inertia_tensor"].astype(np.float64)

    q0_truth = truth_full["q0_wxyz"].astype(np.float64)
    omega_truth = truth_full["omega0_rad"].astype(np.float64)
    omega_mag_truth = float(np.linalg.norm(omega_truth))
    q_180x_quat = np.array([0.0, 1.0, 0.0, 0.0])
    twin_q0 = quat_mul(q_180x_quat, q0_truth)

    # Identify near-truth ω-cell
    bracket_cells = bracket["bracket_cells"]
    bracket_pct = bracket["bracket_dist_to_truth_pct"]
    mag_idx = int(np.argmin(bracket_pct))
    nearest_dir_idx = int(omega_grid["nearest_dir_idx"])
    nearest_dir_deg = float(omega_grid["nearest_dir_deg"])
    N_dir = int(omega_grid["omega_dirs"].shape[0])
    cell_idx = mag_idx * N_dir + nearest_dir_idx
    omega_cell = omega_grid["omega_vectors"][cell_idx].astype(np.float64)
    omega_cell_mag_pct = float(bracket_pct[mag_idx])
    if verbose:
        print(f"  near-truth ω-cell: mag_idx={mag_idx} ({omega_cell_mag_pct:.3f}% off), "
              f"dir_idx={nearest_dir_idx} ({nearest_dir_deg:.3f}° off)", flush=True)
        print(f"  cell global idx = {cell_idx}", flush=True)
        print(f"  ω_cell = {omega_cell} (|ω|={np.linalg.norm(omega_cell):.6f} rad/s)", flush=True)
        print(f"  ω_truth = {omega_truth} (|ω|={omega_mag_truth:.6f} rad/s)", flush=True)

    # Slice the cached arrays for this cell
    M_q = int(q_pool["q_target"].shape[0])
    sl = slice(cell_idx * M_q, (cell_idx + 1) * M_q)
    q0_arr_cached = np.array(cands["q0"][sl]).astype(np.float64)
    geo_score = np.array(cands["geo_score"][sl]).astype(np.float64)
    align_score = np.array(cands["align_score"][sl]).astype(np.float64)

    # Compute pre-LM surrogate MSE for all 528 ICs at this cell
    if verbose:
        print(f"[s038-A] Propagating + surrogate-evaluating {M_q} ICs at near-truth cell...",
              flush=True)
    from lib.surrogate_eval import get_model
    surrogate = get_model()

    # Propagate Δ once, build q(t) = Δ(t) ⊗ q0, get k1/k2, surrogate eval per IC.
    # Match s020/s034 forward.
    delta_q_traj, _ = propagate_attitude(
        np.array([1.0, 0.0, 0.0, 0.0]), omega_cell, obs_times,
        mode="tumbling", inertia_tensor=inertia,
    )
    N_obs = obs_times.size

    # q_full[e, q] = delta_q_traj[e] ⊗ q0_arr[q]
    dw = delta_q_traj[:, 0:1]; dx = delta_q_traj[:, 1:2]
    dy = delta_q_traj[:, 2:3]; dz = delta_q_traj[:, 3:4]
    qw = q0_arr_cached[None, :, 0]; qx = q0_arr_cached[None, :, 1]
    qy = q0_arr_cached[None, :, 2]; qz = q0_arr_cached[None, :, 3]
    q_full = np.stack([
        dw*qw - dx*qx - dy*qy - dz*qz,
        dw*qx + dx*qw + dy*qz - dz*qy,
        dw*qy - dx*qz + dy*qw + dz*qx,
        dw*qz + dx*qy - dy*qx + dz*qw,
    ], axis=-1)   # (N_obs, M_q, 4)

    q_flat = q_full.reshape(N_obs * M_q, 4)
    R_flat = quat_to_R_i2b_batch(q_flat)
    R_full = R_flat.reshape(N_obs, M_q, 3, 3)
    k1_body = np.einsum('eqij,ej->qei', R_full, sun_unit)
    k2_body = np.einsum('eqij,ej->qei', R_full, obs_unit)

    obs_dist_flat = np.broadcast_to(obs_dist[None, :], (M_q, N_obs)).reshape(-1)
    mag_pred_flat = surrogate.predict_magnitude(
        k1_body.reshape(-1, 3), k2_body.reshape(-1, 3),
        SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist_flat,
    )
    mag_pred = mag_pred_flat.reshape(M_q, N_obs)

    # Surrogate MSE vs truth (valid epochs only)
    diff = mag_pred[:, valid_mask] - mag_truth[None, valid_mask]
    surrogate_mse = np.mean(diff ** 2, axis=1)
    rho_pred = np.sqrt(surrogate_mse) / 0.05

    # q0 errors to truth and twin
    q0_err_truth = np.array([angular_dist_deg(q, q0_truth) for q in q0_arr_cached])
    q0_err_twin = np.array([angular_dist_deg(q, twin_q0) for q in q0_arr_cached])
    q0_err_min = np.minimum(q0_err_truth, q0_err_twin)

    if verbose:
        print(f"  q0_err_truth: min={q0_err_truth.min():.2f}° "
              f"median={np.median(q0_err_truth):.2f}° max={q0_err_truth.max():.2f}°",
              flush=True)
        print(f"  q0_err_twin : min={q0_err_twin.min():.2f}° "
              f"median={np.median(q0_err_twin):.2f}° max={q0_err_twin.max():.2f}°",
              flush=True)
        print(f"  q0_err_min  : min={q0_err_min.min():.2f}°", flush=True)
        print(f"  geo_score: min={geo_score.min():.3f} max={geo_score.max():.3f} "
              f"n>=1.0: {int((geo_score >= 1.0).sum())}/{M_q}", flush=True)
        print(f"  align_score: min={align_score.min():.3f} max={align_score.max():.3f} "
              f"n>=1.0: {int((align_score >= 1.0).sum())}/{M_q}", flush=True)
        print(f"  pre-LM ρ: min={rho_pred.min():.2f} median={np.median(rho_pred):.2f}",
              flush=True)

    # Save table
    table = np.zeros(M_q, dtype=[
        ("ic_idx", "i4"),
        ("peak_epoch", "i4"),
        ("face_idx", "i1"),
        ("phi_idx", "i1"),
        ("tier_idx", "i1"),
        ("q0_w", "f4"), ("q0_x", "f4"), ("q0_y", "f4"), ("q0_z", "f4"),
        ("q0_err_truth_deg", "f4"),
        ("q0_err_twin_deg", "f4"),
        ("q0_err_min_deg", "f4"),
        ("geo_score", "f4"),
        ("align_score", "f4"),
        ("surrogate_mse", "f4"),
        ("rho_pred", "f4"),
    ])
    table["ic_idx"] = np.arange(M_q, dtype=np.int32)
    table["peak_epoch"] = q_pool["peak_epoch_idx"]
    table["face_idx"] = q_pool["face_idx"]
    table["phi_idx"] = q_pool["phi_idx"]
    table["tier_idx"] = q_pool["tier_idx"]
    table["q0_w"] = q0_arr_cached[:, 0]
    table["q0_x"] = q0_arr_cached[:, 1]
    table["q0_y"] = q0_arr_cached[:, 2]
    table["q0_z"] = q0_arr_cached[:, 3]
    table["q0_err_truth_deg"] = q0_err_truth
    table["q0_err_twin_deg"] = q0_err_twin
    table["q0_err_min_deg"] = q0_err_min
    table["geo_score"] = geo_score
    table["align_score"] = align_score
    table["surrogate_mse"] = surrogate_mse
    table["rho_pred"] = rho_pred

    np.savez_compressed(OUT_DIR / "stage_a_ic_scores.npz",
                        table=table,
                        cell_idx=cell_idx,
                        omega_cell=omega_cell,
                        omega_cell_mag_pct=omega_cell_mag_pct,
                        omega_dir_to_truth_deg=nearest_dir_deg,
                        omega_truth=omega_truth,
                        q0_truth=q0_truth,
                        twin_q0=twin_q0,
                        bracket_cells_rad=bracket_cells,
                        spec_event_eps=spec_geo["spec_event_eps"],
                        spec_event_tier=spec_geo["spec_event_tier"],
                        bright_peak_idx=spec_geo["bright_peak_idx"],
                        valid_mask=valid_mask,
                        truth_mag_hifi=mag_truth,
                        # We also store the per-IC mag_pred for offline comparison
                        # but only the K closest-to-truth + K best-MSE, to keep size sane.
                        )

    # Also save the truth mag_pred (via surrogate at truth (q0, ω))
    truth_q_traj, _ = propagate_attitude(
        q0_truth, omega_truth, obs_times, mode="tumbling", inertia_tensor=inertia,
    )
    truth_R = quat_to_R_i2b_batch(truth_q_traj)
    truth_k1 = np.einsum('eij,ej->ei', truth_R, sun_unit)
    truth_k2 = np.einsum('eij,ej->ei', truth_R, obs_unit)
    truth_mag_via_surrogate = surrogate.predict_magnitude(
        truth_k1, truth_k2, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
    )
    truth_mse = float(np.mean((truth_mag_via_surrogate[valid_mask] -
                                mag_truth[valid_mask])**2))
    truth_rho = float(np.sqrt(truth_mse) / 0.05)
    if verbose:
        print(f"  truth surrogate MSE = {truth_mse:.6e} (ρ = {truth_rho:.3f})  "
              "[surrogate noise floor]", flush=True)

    if verbose:
        print(f"[s038-A] Saved: {OUT_DIR / 'stage_a_ic_scores.npz'}", flush=True)

    return {
        "table": table,
        "cell_idx": cell_idx,
        "omega_cell": omega_cell,
        "omega_truth": omega_truth,
        "q0_truth": q0_truth,
        "twin_q0": twin_q0,
        "obs_times": obs_times,
        "sun_unit": sun_unit,
        "obs_unit": obs_unit,
        "obs_dist": obs_dist,
        "mag_truth": mag_truth,
        "valid_mask": valid_mask,
        "inertia": inertia,
        "q0_arr": q0_arr_cached,
        "spec_event_eps": spec_geo["spec_event_eps"],
        "spec_event_tier": spec_geo["spec_event_tier"],
        "truth_rho": truth_rho,
        "truth_mse": truth_mse,
    }


# ------------- Stage B: LM polish, bypassing the filter ---------------------
def build_residual(q0_init, omega_init, obs_times, inertia,
                   sun_unit, obs_unit, obs_dist, mag_truth, valid_mask, surrogate):
    mag_truth_valid = mag_truth[valid_mask]

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_new = quat_mul(q_pert_wxyz, q0_init)
        omega_new = omega_init + omega_delta
        q_traj, _ = propagate_attitude(
            q0_new, omega_new, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        R_full = quat_to_R_i2b_batch(q_traj)
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)
        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid_mask] - mag_truth_valid).astype(np.float64)

    return residual


_WORKER = None

def _worker_init(payload):
    global _WORKER
    _WORKER = payload
    from lib.surrogate_eval import get_model
    _WORKER["surrogate"] = get_model()


def _polish_one(args):
    """Pool worker: polish one IC at the near-truth ω-cell."""
    ic_idx, q0_init, omega_init = args
    s = _WORKER
    surrogate = s["surrogate"]

    q0_init = q0_init / np.linalg.norm(q0_init)
    residual = build_residual(
        q0_init, omega_init, s["obs_times"], s["inertia"],
        s["sun_unit"], s["obs_unit"], s["obs_dist"],
        s["mag_truth"], s["valid_mask"], surrogate,
    )
    x0 = np.zeros(6)
    t0 = time.time()
    converged = True
    try:
        result = least_squares(
            residual, x0, method='lm',
            max_nfev=LM_MAX_NFEV, xtol=1e-8, ftol=1e-8,
        )
        n_iter = int(result.nfev)
        mse_final = float(np.mean(result.fun ** 2))
        rotvec = result.x[:3]
        omega_delta = result.x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_final = quat_mul(q_pert_wxyz, q0_init)
        omega_final = omega_init + omega_delta
    except Exception as e:
        converged = False
        n_iter = -1
        mse_final = float('inf')
        q0_final = q0_init.copy()
        omega_final = omega_init.copy()
    dt = time.time() - t0
    return {
        "ic_idx": int(ic_idx),
        "q0_init": q0_init.tolist(),
        "omega_init": omega_init.tolist(),
        "q0_final": q0_final.tolist(),
        "omega_final": omega_final.tolist(),
        "mse_final": float(mse_final),
        "rho_final": float(np.sqrt(max(mse_final, 0.0)) / 0.05),
        "n_iter": int(n_iter),
        "wall_s": float(dt),
        "converged": bool(converged),
    }


def stage_b_polish(stage_a, n_top_q0=30, n_top_rho=30, n_workers=8, verbose=True):
    """Pick top-K ICs by q0_err_min and top-K by pre-LM ρ_pred. Run LM on the union."""
    table = stage_a["table"]
    omega_cell = stage_a["omega_cell"]
    q0_arr = stage_a["q0_arr"]
    q0_truth = stage_a["q0_truth"]
    twin_q0 = stage_a["twin_q0"]
    omega_truth = stage_a["omega_truth"]

    # Top-K by closest q0 (truth or twin)
    order_q0 = np.argsort(table["q0_err_min_deg"])[:n_top_q0]
    # Top-K by best pre-LM ρ
    order_rho = np.argsort(table["rho_pred"])[:n_top_rho]
    union = np.unique(np.concatenate([order_q0, order_rho]))
    if verbose:
        print(f"[s038-B] LM panel: {len(union)} ICs "
              f"(top-{n_top_q0} q0_err ∪ top-{n_top_rho} ρ_pred). "
              f"q0-rank overlap = {len(set(order_q0)) & set(order_rho).__len__() if False else len(np.intersect1d(order_q0, order_rho))}",
              flush=True)
        print(f"  closest q0 IC: ic={int(order_q0[0])} "
              f"q0_err_min={table['q0_err_min_deg'][order_q0[0]]:.2f}° "
              f"geo={table['geo_score'][order_q0[0]]:.3f} "
              f"align={table['align_score'][order_q0[0]]:.3f} "
              f"ρ_pre={table['rho_pred'][order_q0[0]]:.2f}", flush=True)
        print(f"  best ρ IC:     ic={int(order_rho[0])} "
              f"q0_err_min={table['q0_err_min_deg'][order_rho[0]]:.2f}° "
              f"geo={table['geo_score'][order_rho[0]]:.3f} "
              f"align={table['align_score'][order_rho[0]]:.3f} "
              f"ρ_pre={table['rho_pred'][order_rho[0]]:.2f}", flush=True)

    panel_args = [(int(i), q0_arr[i], omega_cell.copy()) for i in union]

    payload = {
        "obs_times": stage_a["obs_times"],
        "sun_unit": stage_a["sun_unit"],
        "obs_unit": stage_a["obs_unit"],
        "obs_dist": stage_a["obs_dist"],
        "mag_truth": stage_a["mag_truth"],
        "valid_mask": stage_a["valid_mask"],
        "inertia": stage_a["inertia"],
    }

    if verbose:
        print(f"[s038-B] launching Pool({n_workers}) over {len(panel_args)} LM polishes...",
              flush=True)
    t0 = time.time()
    if n_workers <= 1:
        _worker_init(payload)
        results = [_polish_one(a) for a in panel_args]
    else:
        from multiprocessing import get_context
        ctx = get_context("fork")
        with ctx.Pool(n_workers, initializer=_worker_init, initargs=(payload,)) as pool:
            results = pool.map(_polish_one, panel_args)
    wall = time.time() - t0

    # Annotate with q0_err_final / ω_err_final
    for r in results:
        q_final = np.array(r["q0_final"])
        om_final = np.array(r["omega_final"])
        r["q0_err_truth_deg"] = angular_dist_deg(q_final, q0_truth)
        r["q0_err_twin_deg"] = angular_dist_deg(q_final, twin_q0)
        r["omega_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            np.dot(om_final, omega_truth) /
            (np.linalg.norm(om_final) * np.linalg.norm(omega_truth)),
            -1, 1))))
        r["omega_mag_pct"] = float(
            (np.linalg.norm(om_final) - np.linalg.norm(omega_truth))
            / np.linalg.norm(omega_truth) * 100
        )
        r["q0_err_init_truth_deg"] = float(table["q0_err_truth_deg"][r["ic_idx"]])
        r["q0_err_init_min_deg"] = float(table["q0_err_min_deg"][r["ic_idx"]])
        r["geo_score_init"] = float(table["geo_score"][r["ic_idx"]])
        r["align_score_init"] = float(table["align_score"][r["ic_idx"]])
        r["rho_pred_init"] = float(table["rho_pred"][r["ic_idx"]])
        r["ic_in_top_q0"] = bool(r["ic_idx"] in order_q0)
        r["ic_in_top_rho"] = bool(r["ic_idx"] in order_rho)

    rho_finals = [r["rho_final"] for r in results]
    n_band_a = sum(1 for r in rho_finals if r < 2)
    n_band_b = sum(1 for r in rho_finals if 2 <= r < 4)
    n_band_c = sum(1 for r in rho_finals if 4 <= r < 8)
    n_band_d = sum(1 for r in rho_finals if r >= 8)
    if verbose:
        print(f"[s038-B] LM wall: {wall/60:.1f} min", flush=True)
        print(f"  ρ-band (final, predicted): A={n_band_a} B={n_band_b} "
              f"C={n_band_c} D={n_band_d}  (panel n={len(results)})", flush=True)
        rho_min = min(rho_finals)
        rho_med = sorted(rho_finals)[len(rho_finals)//2]
        print(f"  ρ_final: min={rho_min:.2f} median={rho_med:.2f}", flush=True)
        # closest-to-truth final
        order_post = sorted(range(len(results)),
                            key=lambda i: min(results[i]["q0_err_truth_deg"],
                                              results[i]["q0_err_twin_deg"]))
        for i in order_post[:5]:
            r = results[i]
            print(f"  ic {r['ic_idx']:>3}: "
                  f"q0_init→truth/twin {r['q0_err_init_min_deg']:.2f}° "
                  f"→ {min(r['q0_err_truth_deg'], r['q0_err_twin_deg']):.2f}°  "
                  f"ρ_pre={r['rho_pred_init']:.2f}→{r['rho_final']:.3f}  "
                  f"ω-dir={r['omega_dir_err_deg']:.2f}° "
                  f"ω-mag={r['omega_mag_pct']:+.2f}% "
                  f"nfev={r['n_iter']} wall={r['wall_s']:.1f}s "
                  f"[topQ0={r['ic_in_top_q0']} topρ={r['ic_in_top_rho']}]",
                  flush=True)

    summary = {
        "seed": SEED,
        "n_panel": len(results),
        "n_top_q0": n_top_q0,
        "n_top_rho": n_top_rho,
        "lm_max_nfev": LM_MAX_NFEV,
        "wall_s": wall,
        "rho_band": {"A": n_band_a, "B": n_band_b, "C": n_band_c, "D": n_band_d},
        "results": results,
    }
    with open(OUT_DIR / "stage_b_lm_panel.json", "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"[s038-B] Saved: {OUT_DIR / 'stage_b_lm_panel.json'}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lm", action="store_true",
                        help="Skip stage B (LM panel); only score ICs.")
    parser.add_argument("--n-top-q0", type=int, default=30)
    parser.add_argument("--n-top-rho", type=int, default=30)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    print(f"=== s038 filter diagnostics | seed={SEED} | out={OUT_DIR} ===",
          flush=True)
    t0 = time.time()
    stage_a = stage_a_score_ics()
    print(f"\n[s038-A] wall: {(time.time()-t0)/60:.2f} min", flush=True)

    if args.no_lm:
        print("[s038] --no-lm; stopping after stage A.", flush=True)
        return

    print(f"\n=== stage B: LM panel ===", flush=True)
    t1 = time.time()
    stage_b_polish(stage_a, n_top_q0=args.n_top_q0,
                   n_top_rho=args.n_top_rho, n_workers=args.n_workers)
    print(f"\n[s038-B] wall: {(time.time()-t1)/60:.2f} min", flush=True)
    print(f"[s038] TOTAL: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
