"""s020 — bracket-augmented filter pipeline (NO LM polish).

Run for one seed at a time. Stops at the alignment-filter survivor stage.
Checkpoints EVERYTHING — every cell, every q0, every score, every timing.

Pipeline:
  1. LS-bracket → ω-mag cells (typically ~5).
  2. Fibonacci sphere → ω-dir cells (~1500 at 3° spacing).
  3. For each ω-cell (mag × dir):
     a. Propagate Δ(t) once.
     b. Generate phi-sweep q0 IC pool from bright-peak tier classifier.
  4. Geo cost on every candidate. Threshold = truth's geo score (per-seed).
     Survivors and rejects categorised; thresholds stored.
  5. Alignment cost on EVERY candidate (including geo-rejects), with timing.
     Threshold = truth's alignment score (per-seed). Survivor LCs stored;
     reject LCs discarded (regenerable from stored q0/ω).
  6. Final categorisation: passed_both / passed_geo_only / passed_align_only /
     rejected_both. Stats reported.

Usage:
  python notebooks/inversion/survey/experiments/s020_seed_pipeline.py 6
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle, find_peaks
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.surrogate_eval import predict as surrogate_predict, get_model
from lib.filter_costs import (
    load_static_geometry, load_tier_table, assign_tier,
    GROUP_NAMES, D_REF_KM,
)


# -------- Configuration --------
N_LS_FREQS = 4000
LS_THRESHOLD_FRAC = 0.1
LS_MIN_PEAK_DIST = 5

N_OMEGA_DIR = 300           # Fibonacci sphere — ~5° spacing on S² (covers 3-5° basin)
N_OMEGA_MAG_CELLS = 5       # bracket subsampled to this many cells
N_PHI_STEPS = 12            # phi-sweep per face per peak
PHI_DEG_STEP = 360.0 / N_PHI_STEPS
N_WORKERS = 8               # multiprocessing pool size

BRIGHT_MAG_THRESHOLD = 11.0
SPEC_THRESHOLD_DEG = 5.0
ALIGN_WINDOW_EPOCHS = 3
GEO_THRESHOLD_DEG = 5.0
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0

MEASURE_GEO_FAIL_ALIGN = False


# -------- Helpers --------
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply_batch_left(delta_q, q0_arr):
    """LEFT multiply: q(t) = delta_q ⊗ q0 for each q0 in q0_arr.

    delta_q: (4,) — single quaternion.
    q0_arr: (M, 4) — batch of q0s.
    Returns: (M, 4)
    """
    dw, dx, dy, dz = delta_q
    w = q0_arr[:, 0]; x = q0_arr[:, 1]; y = q0_arr[:, 2]; z = q0_arr[:, 3]
    return np.stack([
        dw*w - dx*x - dy*y - dz*z,
        dw*x + dx*w + dy*z - dz*y,
        dw*y - dx*z + dy*w + dz*x,
        dw*z + dx*y - dy*x + dz*w,
    ], axis=1)


def quat_mul_outer_left(delta_arr, q0_arr):
    """Outer-product LEFT multiply: q[n, m] = delta_arr[n] ⊗ q0_arr[m].

    delta_arr: (N, 4)
    q0_arr:    (M, 4)
    Returns:   (N, M, 4)
    """
    dw = delta_arr[:, 0:1]; dx = delta_arr[:, 1:2]
    dy = delta_arr[:, 2:3]; dz = delta_arr[:, 3:4]
    w = q0_arr[None, :, 0]; x = q0_arr[None, :, 1]
    y = q0_arr[None, :, 2]; z = q0_arr[None, :, 3]
    return np.stack([
        dw*w - dx*x - dy*y - dz*z,
        dw*x + dx*w + dy*z - dz*y,
        dw*y - dx*z + dy*w + dz*x,
        dw*z + dx*y - dy*x + dz*w,
    ], axis=-1)


def quat_to_R_i2b_batch(q_arr_wxyz):
    """Batch convert (M, 4) quaternions to (M, 3, 3) inertial→body matrices."""
    qxyzw = q_arr_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_to_R_i2b_single(q_wxyz):
    qxyzw = q_wxyz[[1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_to_R_b2i_single(q_wxyz):
    return quat_to_R_i2b_single(q_wxyz).T


def fibonacci_sphere(n):
    """Generate n unit vectors approximately uniformly on S²."""
    phi = np.pi * (np.sqrt(5.0) - 1.0)
    i = np.arange(n)
    y = 1.0 - 2.0 * i / (n - 1)
    r = np.sqrt(1.0 - y * y)
    theta = phi * i
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def rotation_aligning(a, b):
    """Rotation matrix R such that R @ a = b. Unit a, b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        ortho = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def rotation_about_axis(axis_unit, angle_rad):
    return Rotation.from_rotvec(angle_rad * axis_unit).as_matrix()


def matrix_to_quat_wxyz(R_b2i):
    R_i2b = R_b2i.T
    qxyzw = Rotation.from_matrix(R_i2b).as_quat()
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


# -------- Bracket --------
def run_bracket(mag_hifi, observation_times):
    """Lomb-Scargle periodogram + bracket on bright-mask transitions."""
    valid = np.isfinite(mag_hifi)
    valid_lc = mag_hifi[valid]
    valid_t = observation_times[valid]

    s = -valid_lc
    s = s - np.mean(s)
    dt = float(np.median(np.diff(valid_t)))
    f_min = 1.0 / (valid_t[-1] - valid_t[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_LS_FREQS)
    ang = 2 * np.pi * freqs
    power = lombscargle(valid_t, s, ang, normalize=True)

    pmax = float(power.max())
    idx, _ = find_peaks(power, distance=LS_MIN_PEAK_DIST,
                        height=LS_THRESHOLD_FRAC * pmax)
    if len(idx) == 0:
        return None
    # Sort peaks by descending power
    order_by_power = idx[np.argsort(power[idx])[::-1]]
    peak_freqs = freqs[order_by_power]
    peak_powers = power[order_by_power]
    peak_omegas = 2 * np.pi * peak_freqs   # rad/s, sorted by descending power

    lo = 0.5 * peak_omegas.min()
    hi = 2.0 * peak_omegas.max()
    n_full = max(20, int(np.ceil(np.log(hi / max(lo, 1e-12)) / np.log(1.05))))
    full_grid = np.geomspace(lo, hi, n_full)

    # Take top-N_OMEGA_MAG_CELLS LS peaks by raw power. Truth ω is most often
    # equal to one of the LS peaks (per s019 cohort coverage). When N_mag exceeds
    # the LS-peak count, pad with cells from the full geometric grid (excluding
    # cells already within 5% of an LS peak) so densification beyond the peak
    # count actually adds coverage instead of capping silently.
    n_take = min(N_OMEGA_MAG_CELLS, peak_omegas.size)
    bracket_cells = peak_omegas[:n_take]
    if N_OMEGA_MAG_CELLS > n_take:
        n_extra = N_OMEGA_MAG_CELLS - n_take
        # Filter grid cells that aren't already within 5% of a chosen LS peak
        rel = np.abs(full_grid[:, None] - bracket_cells[None, :]) / bracket_cells[None, :]
        far_mask = (rel.min(axis=1) > 0.05)
        candidates = full_grid[far_mask]
        if candidates.size > 0:
            stride = max(1, candidates.size // n_extra)
            extras = candidates[::stride][:n_extra]
            bracket_cells = np.concatenate([bracket_cells, extras])
    bracket_cells = np.sort(bracket_cells)

    return {
        "ls_freqs": freqs,
        "ls_power": power,
        "ls_peak_freqs": peak_freqs,
        "ls_peak_powers": peak_powers,
        "ls_peak_omegas": peak_omegas,
        "bracket_lo": float(lo),
        "bracket_hi": float(hi),
        "bracket_full_grid": full_grid,
        "bracket_cells": bracket_cells,
    }


# -------- Phi-sweep IC pool --------
def build_q_target_pool(truth_data, tier_table):
    """Build the fixed (face × phi) q_target list anchored on bright peaks.

    Returns:
      q_target_arr: (M, 4) — q_target candidates (independent of ω-cell).
      meta: list of (peak_idx, face_idx, phi_idx, t_peak_array_idx) for each.
      t_peak_indices: (n_peaks_used,) array indices into observation_times.
    """
    obs_times = truth_data["observation_times"]
    pab_j2000 = truth_data["pab_j2000"]
    obs_dist = truth_data["obs_dist"]
    mag_hifi = truth_data["mag_hifi"]
    min_ang = truth_data["min_ang_dist"]
    hifi_peak_eps = truth_data["hifi_peak_epochs"]

    face_normals = load_static_geometry()["face_normals"]

    mag_abs_at_peak = mag_hifi[hifi_peak_eps] - 5.0 * np.log10(
        obs_dist[hifi_peak_eps] / D_REF_KM
    )
    tier_at_peak = assign_tier(mag_abs_at_peak, tier_table)

    pool_q_target = []
    pool_meta = []
    used_peak_indices = []

    for k, ep in enumerate(hifi_peak_eps):
        ti = int(tier_at_peak[k])
        if ti < 0:
            continue
        face_shortlist = tier_table["tier_face_idx"][ti]
        pab_at_peak = pab_j2000[ep]
        used_peak_indices.append(int(ep))

        for face_idx in face_shortlist:
            n_g = face_normals[face_idx]
            R_align = rotation_aligning(n_g, pab_at_peak)
            for phi_idx in range(N_PHI_STEPS):
                phi_rad = np.radians(phi_idx * PHI_DEG_STEP)
                R_phi = rotation_about_axis(pab_at_peak, phi_rad)
                R_b2i = R_phi @ R_align
                q_target = matrix_to_quat_wxyz(R_b2i)
                pool_q_target.append(q_target)
                pool_meta.append((int(ep), int(face_idx), int(phi_idx),
                                  int(ti), int(k)))

    if not pool_q_target:
        return None

    return {
        "q_target": np.array(pool_q_target),       # (M, 4)
        "peak_epoch_idx": np.array([m[0] for m in pool_meta]),
        "face_idx":       np.array([m[1] for m in pool_meta]),
        "phi_idx":        np.array([m[2] for m in pool_meta]),
        "tier_idx":       np.array([m[3] for m in pool_meta]),
        "peak_array_idx": np.array([m[4] for m in pool_meta]),  # idx into hifi_peak_epochs
        "tier_at_peak":   tier_at_peak,
        "mag_abs_at_peak": mag_abs_at_peak,
        "n_peaks_classifiable": len(used_peak_indices),
        "n_q_target": len(pool_q_target),
    }


# -------- Geo cost (vectorised over candidates per cell) --------
def geo_cost_batch(pab_body_per_q, spec_event_array_idx, spec_tier,
                   face_normals, tier_face_idx, geo_threshold_deg=5.0):
    """Vectorised geo cost across many candidates for ONE ω-cell.

    pab_body_per_q: (M, n_spec, 3) — body-frame PAB for each candidate
                     at each spec event epoch.
    Returns scores: (M,) — fraction of spec events with a tier-allowed face
                    within geo_threshold_deg of pab_body.
    """
    M, n_spec, _ = pab_body_per_q.shape
    if n_spec == 0:
        return np.full(M, np.nan)
    cos_thresh = np.cos(np.deg2rad(geo_threshold_deg))
    hits = np.zeros(M, dtype=int)
    for k in range(n_spec):
        ti = int(spec_tier[k])
        allowed = tier_face_idx[ti]
        # (M, 3) @ (3, n_allowed) -> (M, n_allowed)
        dots = pab_body_per_q[:, k, :] @ face_normals[allowed].T
        # max dot per candidate at this epoch
        max_dots = dots.max(axis=1)
        hits += (max_dots >= cos_thresh).astype(int)
    return hits / float(n_spec)


# -------- Alignment cost (per candidate scalar from candidate LC) --------
def alignment_cost_one(mag_pred, bright_peak_idx, window_epochs=3,
                       bright_mag_threshold=11.0):
    if bright_peak_idx.size == 0:
        return float("nan")
    N = mag_pred.shape[0]
    W = window_epochs
    hits = 0
    for tp in bright_peak_idx:
        lo = max(0, int(tp) - W)
        hi = min(N, int(tp) + W + 1)
        window = mag_pred[lo:hi]
        local_min_idx = lo + int(np.argmin(window))
        is_min = True
        if local_min_idx > 0:
            is_min &= mag_pred[local_min_idx] <= mag_pred[local_min_idx - 1]
        if local_min_idx < N - 1:
            is_min &= mag_pred[local_min_idx] <= mag_pred[local_min_idx + 1]
        if is_min and mag_pred[local_min_idx] < bright_mag_threshold:
            hits += 1
    return hits / float(bright_peak_idx.size)


# -------- Pool worker (per-cell processing) --------
_WORKER_STATE = None
_WORKER_SURROGATE = None


def _pool_init(state):
    """Initializer: set per-worker globals once. Surrogate is loaded lazily."""
    global _WORKER_STATE, _WORKER_SURROGATE
    _WORKER_STATE = state
    _WORKER_SURROGATE = get_model()    # warm cache


def _process_cell(args):
    cell_idx, omega_cell = args
    s = _WORKER_STATE
    M_q = s["M_q"]; N_obs = s["N_obs"]; n_spec = s["n_spec"]

    delta_q_traj, _ = propagate_attitude(
        np.array([1.0, 0.0, 0.0, 0.0]), omega_cell, s["obs_times"],
        mode="tumbling", inertia_tensor=s["inertia"],
    )

    # q0 = Phi(t_peak)^-1 ⊗ q_target  (vectorised over q_target)
    phi_at_peaks = delta_q_traj[s["q_target_peak_eps"]]
    phi_inv = phi_at_peaks * np.array([1, -1, -1, -1])[None]
    q_target = s["q_target"]
    pw = phi_inv[:, 0]; px = phi_inv[:, 1]; py = phi_inv[:, 2]; pz = phi_inv[:, 3]
    tw = q_target[:, 0]; tx = q_target[:, 1]
    ty = q_target[:, 2]; tz = q_target[:, 3]
    q0_arr = np.stack([
        pw*tw - px*tx - py*ty - pz*tz,
        pw*tx + px*tw + py*tz - pz*ty,
        pw*ty - px*tz + py*tw + pz*tx,
        pw*tz + px*ty - py*tx + pz*tw,
    ], axis=1)

    # Geo cost
    if n_spec > 0:
        t_g = time.time()
        delta_at_spec = delta_q_traj[s["spec_event_eps"]]
        q_at_spec = quat_mul_outer_left(delta_at_spec, q0_arr)
        q_flat = q_at_spec.reshape(n_spec * M_q, 4)
        R_flat = quat_to_R_i2b_batch(q_flat)
        R_at_spec = R_flat.reshape(n_spec, M_q, 3, 3)
        pab_body_per_q = np.einsum(
            'sqij,sj->qsi', R_at_spec, s["pab_at_spec_inertial"]
        )
        geo_scores = geo_cost_batch(
            pab_body_per_q, s["spec_event_eps"], s["spec_event_tier"],
            s["face_normals"], s["tier_face_idx"],
            geo_threshold_deg=GEO_THRESHOLD_DEG,
        )
        t_geo = time.time() - t_g
    else:
        geo_scores = np.full(M_q, np.nan)
        t_geo = 0.0

    # Body-frame k1/k2 for all (epoch, candidate)
    q_full = quat_mul_outer_left(delta_q_traj, q0_arr)
    q_full_flat = q_full.reshape(N_obs * M_q, 4)
    R_full_flat = quat_to_R_i2b_batch(q_full_flat)
    R_full = R_full_flat.reshape(N_obs, M_q, 3, 3)
    k1_body_all = np.einsum('eqij,ej->qei', R_full, s["sun_unit"])
    k2_body_all = np.einsum('eqij,ej->qei', R_full, s["obs_unit"])

    # Geo PASS / FAIL split
    if n_spec > 0:
        geo_pass_mask = (geo_scores >= s["geo_threshold"]) & np.isfinite(geo_scores)
    else:
        geo_pass_mask = np.ones(M_q, dtype=bool)
    n_pass = int(geo_pass_mask.sum())
    n_fail = M_q - n_pass

    align_scores = np.full(M_q, np.nan, dtype=np.float64)

    def _surrogate_batch(mask):
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return idxs, np.zeros((0, N_obs), dtype=np.float32)
        k1_sub = k1_body_all[idxs]
        k2_sub = k2_body_all[idxs]
        n_sel = idxs.size
        k1_flat = k1_sub.reshape(n_sel * N_obs, 3)
        k2_flat = k2_sub.reshape(n_sel * N_obs, 3)
        obs_dist_flat = np.broadcast_to(
            s["obs_dist"][None, :], (n_sel, N_obs)
        ).reshape(-1)
        mag_flat = _WORKER_SURROGATE.predict_magnitude(
            k1_flat, k2_flat, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist_flat,
        )
        return idxs, mag_flat.reshape(n_sel, N_obs).astype(np.float32)

    # PASS
    t_p = time.time()
    pass_idxs, mag_pred_pass = _surrogate_batch(geo_pass_mask)
    for k_local, j in enumerate(pass_idxs):
        align_scores[j] = alignment_cost_one(
            mag_pred_pass[k_local], s["bright_peak_idx_arr"],
            ALIGN_WINDOW_EPOCHS, BRIGHT_MAG_THRESHOLD,
        )
    dt_pass = time.time() - t_p

    # FAIL — only compute when measuring "would-have-saved" instrumentation;
    # production mode skips the surrogate eval on geo-rejected candidates.
    t_f = time.time()
    if s["measure_geo_fail"]:
        fail_mask = ~geo_pass_mask if n_spec > 0 else np.zeros(M_q, dtype=bool)
        fail_idxs, mag_pred_fail = _surrogate_batch(fail_mask)
        for k_local, j in enumerate(fail_idxs):
            align_scores[j] = alignment_cost_one(
                mag_pred_fail[k_local], s["bright_peak_idx_arr"],
                ALIGN_WINDOW_EPOCHS, BRIGHT_MAG_THRESHOLD,
            )
    else:
        fail_idxs = np.array([], dtype=np.int64)
    dt_fail = time.time() - t_f

    # Survivor LCs
    survivor_mask = (
        np.isfinite(geo_scores) & (geo_scores >= s["geo_threshold"]) &
        np.isfinite(align_scores) & (align_scores >= s["align_threshold"])
    )
    survivor_lcs = []
    for j in np.where(survivor_mask)[0]:
        local = np.where(pass_idxs == j)[0]
        if local.size > 0:
            survivor_lcs.append((int(j), mag_pred_pass[local[0]].copy()))

    per_pass_time = []
    if n_pass > 0:
        avg_p = dt_pass / n_pass
        per_pass_time = [(int(j), float(avg_p)) for j in pass_idxs]
    per_fail_time = []
    if n_fail > 0:
        avg_f = dt_fail / max(n_fail, 1)
        per_fail_time = [(int(j), float(avg_f)) for j in fail_idxs]

    return {
        "cell_idx": int(cell_idx),
        "delta_quats": delta_q_traj,
        "q0_arr": q0_arr,
        "geo_scores": geo_scores,
        "align_scores": align_scores,
        "survivor_lcs": survivor_lcs,
        "per_pass_time": per_pass_time,
        "per_fail_time": per_fail_time,
        "t_geo": float(t_geo),
        "t_align_pass": float(dt_pass),
        "t_align_fail": float(dt_fail),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
    }


# -------- The main per-seed driver --------
def run_seed(seed: int, out_dir: Path, verbose: bool = True):
    t_total_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load truth + static data ----
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    if verbose:
        print(f"[seed {seed}] loading {traj_path}", flush=True)
    truth_npz = np.load(traj_path)
    truth = {k: truth_npz[k] for k in truth_npz.files}
    obs_times = truth["observation_times"].astype(float)
    obs_dist = truth["obs_dist"].astype(float)
    mag_hifi = truth["mag_hifi"].astype(float)
    sun_pos = truth["sun_pos"].astype(float)
    obs_pos = truth["obs_pos"].astype(float)
    sat_pos = truth["sat_pos"].astype(float)
    pab_j2000 = truth["pab_j2000"].astype(float)
    min_ang = truth["min_ang_dist"].astype(float)
    hifi_peak_eps = truth["hifi_peak_epochs"].astype(int)
    q0_truth = truth["q0_wxyz"].astype(float)
    omega0_truth = truth["omega0_rad"].astype(float)
    omega_mag_truth = float(np.linalg.norm(omega0_truth))
    N_obs = obs_times.size

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)

    geo = load_static_geometry()
    face_normals = geo["face_normals"]
    inertia = geo["inertia_tensor"]
    tier_table = load_tier_table()

    # ---- Step 1: bracket ----
    if verbose:
        print(f"[seed {seed}] step 1: LS-bracket", flush=True)
    t_bracket = time.time()
    bracket = run_bracket(mag_hifi, obs_times)
    if bracket is None:
        raise RuntimeError("No significant LS peaks — bracket failed for this seed.")
    bracket["truth_omega_mag_rad"] = omega_mag_truth
    bracket["truth_omega_mag_dps"] = float(np.degrees(omega_mag_truth))
    bracket["bracket_dist_to_truth_pct"] = (
        np.abs(bracket["bracket_cells"] - omega_mag_truth) / omega_mag_truth * 100
    )
    bracket["nearest_cell_pct"] = float(bracket["bracket_dist_to_truth_pct"].min())
    bracket_wall = time.time() - t_bracket
    np.savez_compressed(out_dir / "bracket.npz", **{
        k: v for k, v in bracket.items()
        if isinstance(v, (np.ndarray, float, int))
    })
    if verbose:
        print(f"  bracket cells (rad/s): {bracket['bracket_cells']}", flush=True)
        print(f"  truth ω-mag: {omega_mag_truth:.6f} rad/s "
              f"({np.degrees(omega_mag_truth):.4f} dps)", flush=True)
        print(f"  nearest bracket cell to truth: "
              f"{bracket['nearest_cell_pct']:.2f}%", flush=True)
        print(f"  bracket wall: {bracket_wall:.3f} s", flush=True)

    # ---- Step 2a: ω-grid ----
    if verbose:
        print(f"[seed {seed}] step 2a: ω-grid construction", flush=True)
    omega_dirs = fibonacci_sphere(N_OMEGA_DIR)         # (N_dir, 3)
    omega_mags = bracket["bracket_cells"]              # (N_mag,)
    N_mag = omega_mags.size
    N_dir = omega_dirs.shape[0]
    N_cells = N_mag * N_dir
    omega_vectors = (omega_mags[:, None, None] * omega_dirs[None, :, :]).reshape(N_cells, 3)

    # Truth ω as a "ghost" reference cell (NOT part of the search grid; just for diagnostics).
    truth_dir = omega0_truth / omega_mag_truth
    omega_dir_dist_to_truth = np.degrees(np.arccos(
        np.clip(omega_dirs @ truth_dir, -1, 1)
    ))
    nearest_dir_idx = int(np.argmin(omega_dir_dist_to_truth))
    nearest_dir_deg = float(omega_dir_dist_to_truth[nearest_dir_idx])
    if verbose:
        print(f"  ω-mag cells: {N_mag}, ω-dir cells: {N_dir}, total: {N_cells}",
              flush=True)
        print(f"  nearest ω-dir cell to truth: {nearest_dir_deg:.3f}° "
              f"(idx {nearest_dir_idx})", flush=True)

    # ---- Step 2b: q_target pool ----
    if verbose:
        print(f"[seed {seed}] step 2b: q_target pool from phi-sweep", flush=True)
    pool = build_q_target_pool(truth, tier_table)
    if pool is None:
        raise RuntimeError("Zero classifiable peaks — phi-sweep cannot generate ICs.")
    M_q = pool["n_q_target"]
    n_peaks_used = pool["n_peaks_classifiable"]
    if verbose:
        print(f"  classifiable peaks: {n_peaks_used} of {hifi_peak_eps.size}",
              flush=True)
        print(f"  q_target pool size: {M_q} ({n_peaks_used} peaks × shortlist × "
              f"{N_PHI_STEPS} phi)", flush=True)
        print(f"  total candidates this seed: {N_cells * M_q}", flush=True)

    # ---- Step 2c: precompute spec-event scaffold (for geo cost) ----
    spec_threshold = SPEC_THRESHOLD_DEG
    is_spec_at_peak = (
        (min_ang[hifi_peak_eps] < spec_threshold)
        & (pool["tier_at_peak"] >= 0)
    )
    spec_event_eps = hifi_peak_eps[is_spec_at_peak]
    spec_event_tier = pool["tier_at_peak"][is_spec_at_peak]
    n_spec = spec_event_eps.size
    pab_at_spec_inertial = pab_j2000[spec_event_eps]   # (n_spec, 3)
    if verbose:
        print(f"  spec events for geo cost: {n_spec}", flush=True)

    bright_mask = mag_hifi[hifi_peak_eps] < BRIGHT_MAG_THRESHOLD
    bright_peak_idx_arr = hifi_peak_eps[bright_mask]
    if verbose:
        print(f"  bright peaks for align cost: {bright_peak_idx_arr.size}",
              flush=True)

    # ---- Bring up surrogate ----
    _ = get_model()
    if verbose:
        print(f"[seed {seed}] surrogate loaded", flush=True)

    # ---- Step 3 + 4: per-cell processing ----
    # Storage: per-candidate arrays of length N_total = N_cells * M_q.
    # Stored in CELL-MAJOR order: candidate index = cell_idx * M_q + q_target_idx.
    N_total = N_cells * M_q

    cand_q0 = np.empty((N_total, 4), dtype=np.float32)
    cand_omega_cell = np.repeat(np.arange(N_cells, dtype=np.int32), M_q)
    cand_qt_idx = np.tile(np.arange(M_q, dtype=np.int32), N_cells)
    cand_geo_score = np.full(N_total, np.nan, dtype=np.float32)
    cand_align_score = np.full(N_total, np.nan, dtype=np.float32)

    # Δ trajectories per ω-cell (compressed at the end). Store at full epoch
    # resolution so we can recompute body-frame fields if needed.
    delta_quats_all = np.empty((N_cells, N_obs, 4), dtype=np.float32)

    # Survivor LCs (filled at end-of-loop with the survivor mask).
    candidate_lc_buffer = []   # list of (cand_idx, mag_pred[N_obs])
    align_eval_time_per_cand = np.zeros(N_total, dtype=np.float32)

    # Truth's own scores (compute once, used as per-seed thresholds).
    if verbose:
        print(f"[seed {seed}] computing truth's scores for thresholds", flush=True)
    truth_q_traj, _ = propagate_attitude(
        q0_truth, omega0_truth, obs_times, mode="tumbling",
        inertia_tensor=inertia,
    )
    truth_R_i2b = quat_to_R_i2b_batch(truth_q_traj)        # (N_obs, 3, 3)
    truth_pab_body = np.einsum('nij,nj->ni', truth_R_i2b,
                               pab_j2000)                  # (N_obs, 3)
    truth_pab_body_at_spec = truth_pab_body[spec_event_eps]
    # geo
    truth_geo_score = float(geo_cost_batch(
        truth_pab_body_at_spec[None, :, :], spec_event_eps, spec_event_tier,
        face_normals, tier_table["tier_face_idx"], GEO_THRESHOLD_DEG,
    )[0])
    # align (use cached truth k1/k2 for fidelity, but recompute via R_i2b for parity)
    truth_k1 = np.einsum('nij,nj->ni', truth_R_i2b, sun_unit)
    truth_k2 = np.einsum('nij,nj->ni', truth_R_i2b, obs_unit)
    truth_mag_pred = surrogate_predict(truth_k1, truth_k2, obs_dist,
                                        SP_ANGLE_DEG, AD_ANGLE_DEG)
    truth_align_score = float(alignment_cost_one(
        truth_mag_pred, bright_peak_idx_arr, ALIGN_WINDOW_EPOCHS,
        BRIGHT_MAG_THRESHOLD,
    ))
    if verbose:
        print(f"  truth_geo_score   = {truth_geo_score:.4f}", flush=True)
        print(f"  truth_align_score = {truth_align_score:.4f}", flush=True)
    geo_threshold = truth_geo_score
    align_threshold = truth_align_score

    if verbose:
        print(f"[seed {seed}] step 3+4: looping over {N_cells} ω-cells × "
              f"{M_q} q_targets each", flush=True)
    t_geo_total = 0.0
    t_align_total = 0.0
    t_align_geo_pass = 0.0       # only on candidates that passed geo
    t_align_geo_fail = 0.0       # on candidates that failed geo (the "would-have-saved" measurement)
    n_align_geo_pass = 0
    n_align_geo_fail = 0
    t_loop_start = time.time()
    last_print = time.time()

    # Pre-extract peak epochs for each q_target (so we can index Δ later)
    q_target_peak_eps = pool["peak_epoch_idx"]   # (M_q,) int

    obs_dist_per_epoch = obs_dist                  # (N_obs,)

    # Pool-friendly worker state passed once per worker.
    worker_state = {
        "obs_times": obs_times, "inertia": inertia,
        "q_target": pool["q_target"], "q_target_peak_eps": q_target_peak_eps,
        "sun_unit": sun_unit, "obs_unit": obs_unit,
        "obs_dist": obs_dist_per_epoch,
        "spec_event_eps": spec_event_eps,
        "spec_event_tier": spec_event_tier,
        "pab_at_spec_inertial": pab_at_spec_inertial,
        "face_normals": face_normals,
        "tier_face_idx": tier_table["tier_face_idx"],
        "geo_threshold": float(geo_threshold),
        "align_threshold": float(align_threshold),
        "bright_peak_idx_arr": bright_peak_idx_arr,
        "M_q": M_q, "N_obs": N_obs, "n_spec": n_spec,
        "measure_geo_fail": bool(MEASURE_GEO_FAIL_ALIGN),
    }

    if N_WORKERS > 1 and N_cells > N_WORKERS:
        if verbose:
            print(f"[seed {seed}] using Pool({N_WORKERS}) over {N_cells} cells",
                  flush=True)
        from multiprocessing import Pool
        cell_args = [(i, omega_vectors[i]) for i in range(N_cells)]
        with Pool(N_WORKERS, initializer=_pool_init,
                  initargs=(worker_state,)) as pool_obj:
            results_iter = pool_obj.imap_unordered(_process_cell, cell_args,
                                                   chunksize=4)
            results = []
            done = 0
            for r in results_iter:
                results.append(r)
                done += 1
                if verbose and (done % max(1, N_cells // 20) == 0
                                or done == N_cells):
                    elapsed = time.time() - t_loop_start
                    rate = done / max(elapsed, 1e-6)
                    eta = (N_cells - done) / max(rate, 1e-6)
                    surv_so_far = sum(len(rr["survivor_lcs"]) for rr in results)
                    print(f"  cell {done}/{N_cells} ({100*done/N_cells:.1f}%); "
                          f"elapsed {elapsed/60:.1f} min; "
                          f"survivors {surv_so_far}; "
                          f"ETA {eta/60:.1f} min", flush=True)

        # Aggregate
        for r in results:
            ci = r["cell_idx"]
            cs = slice(ci * M_q, (ci + 1) * M_q)
            cand_q0[cs] = r["q0_arr"].astype(np.float32)
            cand_geo_score[cs] = r["geo_scores"].astype(np.float32)
            cand_align_score[cs] = r["align_scores"].astype(np.float32)
            delta_quats_all[ci] = r["delta_quats"].astype(np.float32)
            t_geo_total += r["t_geo"]
            t_align_geo_pass += r["t_align_pass"]
            t_align_geo_fail += r["t_align_fail"]
            t_align_total += r["t_align_pass"] + r["t_align_fail"]
            n_align_geo_pass += r["n_pass"]
            n_align_geo_fail += r["n_fail"]
            for j, dt in r["per_pass_time"]:
                align_eval_time_per_cand[ci * M_q + j] = dt
            for j, dt in r["per_fail_time"]:
                align_eval_time_per_cand[ci * M_q + j] = dt
            for j, lc in r["survivor_lcs"]:
                candidate_lc_buffer.append((ci * M_q + j, lc))
        loop_wall = time.time() - t_loop_start
        # Jump to finalization
        cell_loop_done = True
    else:
        cell_loop_done = False

    if cell_loop_done:
        pass  # serial loop below is skipped

    for cell_idx in (range(N_cells) if not cell_loop_done else range(0)):
        omega_cell = omega_vectors[cell_idx]

        # --- Δ propagation ---
        delta_q_traj, _ = propagate_attitude(
            np.array([1.0, 0.0, 0.0, 0.0]), omega_cell, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        delta_quats_all[cell_idx] = delta_q_traj.astype(np.float32)

        # --- Build q0 pool: q0 = Phi(t_peak)^-1 ⊗ q_target ---
        # Vectorised: gather Φ at each q_target's peak epoch, conjugate, multiply.
        phi_at_peaks = delta_q_traj[q_target_peak_eps]            # (M_q, 4)
        phi_inv = phi_at_peaks * np.array([1, -1, -1, -1])[None]  # conjugate
        # q0[j] = phi_inv[j] ⊗ q_target[j]
        q_target = pool["q_target"]
        pw = phi_inv[:, 0]; px = phi_inv[:, 1]; py = phi_inv[:, 2]; pz = phi_inv[:, 3]
        tw = q_target[:, 0]; tx = q_target[:, 1]; ty = q_target[:, 2]; tz = q_target[:, 3]
        q0_arr = np.stack([
            pw*tw - px*tx - py*ty - pz*tz,
            pw*tx + px*tw + py*tz - pz*ty,
            pw*ty - px*tz + py*tw + pz*tx,
            pw*tz + px*ty - py*tx + pz*tw,
        ], axis=1)

        cand_slice = slice(cell_idx * M_q, (cell_idx + 1) * M_q)
        cand_q0[cand_slice] = q0_arr.astype(np.float32)

        # --- Geo cost (vectorised over candidates × spec epochs) ---
        if n_spec > 0:
            t_g = time.time()
            delta_at_spec = delta_q_traj[spec_event_eps]          # (n_spec, 4)
            q_at_spec = quat_mul_outer_left(delta_at_spec, q0_arr)  # (n_spec, M_q, 4)
            q_flat = q_at_spec.reshape(n_spec * M_q, 4)
            R_flat = quat_to_R_i2b_batch(q_flat)
            R_at_spec = R_flat.reshape(n_spec, M_q, 3, 3)
            pab_body_per_q = np.einsum(
                'sqij,sj->qsi', R_at_spec, pab_at_spec_inertial
            )                                                      # (M_q, n_spec, 3)
            geo_scores = geo_cost_batch(
                pab_body_per_q, spec_event_eps, spec_event_tier,
                face_normals, tier_table["tier_face_idx"], GEO_THRESHOLD_DEG,
            )
            cand_geo_score[cand_slice] = geo_scores.astype(np.float32)
            t_geo_total += time.time() - t_g
        else:
            geo_scores = np.full(M_q, np.nan)
            cand_geo_score[cand_slice] = np.nan

        # --- Body-frame k1/k2 for every (epoch, candidate) ---
        # q_full[e, q] = delta_q_traj[e] ⊗ q0_arr[q]
        q_full = quat_mul_outer_left(delta_q_traj, q0_arr)         # (N_obs, M_q, 4)
        q_full_flat = q_full.reshape(N_obs * M_q, 4)
        R_full_flat = quat_to_R_i2b_batch(q_full_flat)             # (N_obs*M_q, 3, 3)
        R_full = R_full_flat.reshape(N_obs, M_q, 3, 3)
        # k1_body[q, e, :] = R_full[e, q] @ sun_unit[e]
        k1_body_all = np.einsum('eqij,ej->qei', R_full, sun_unit)
        k2_body_all = np.einsum('eqij,ej->qei', R_full, obs_unit)

        # --- Alignment cost: split into geo-PASS and geo-FAIL groups; time each ---
        # In production we'd ONLY surrogate-eval the PASS group; we eval BOTH
        # so we can measure "time we would have saved by skipping FAILs."
        if n_spec > 0:
            geo_pass_mask = (geo_scores >= geo_threshold) & np.isfinite(geo_scores)
        else:
            # Geo undefined for this seed → treat all as "pass"; alignment is the
            # only filter. This matches the s023 framework for zero-classifiable seeds.
            geo_pass_mask = np.ones(M_q, dtype=bool)
        n_pass = int(geo_pass_mask.sum())
        n_fail = M_q - n_pass

        align_scores = np.full(M_q, np.nan, dtype=np.float64)
        # We allocate but only fill mag_pred for survivors after both filters.

        def _surrogate_batch(mask):
            """Run surrogate over candidates selected by mask; return mag_pred."""
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return idxs, np.zeros((0, N_obs), dtype=np.float32)
            k1_sub = k1_body_all[idxs]           # (n_sel, N_obs, 3)
            k2_sub = k2_body_all[idxs]
            # Concatenate along candidate axis for one big surrogate call
            n_sel = idxs.size
            k1_flat = k1_sub.reshape(n_sel * N_obs, 3)
            k2_flat = k2_sub.reshape(n_sel * N_obs, 3)
            obs_dist_flat = np.broadcast_to(
                obs_dist_per_epoch[None, :], (n_sel, N_obs)
            ).reshape(-1)
            mag_flat = surrogate_predict(
                k1_flat, k2_flat, obs_dist_flat,
                SP_ANGLE_DEG, AD_ANGLE_DEG,
            )
            return idxs, mag_flat.reshape(n_sel, N_obs).astype(np.float32)

        # PASS group first
        t_p = time.time()
        pass_idxs, mag_pred_pass = _surrogate_batch(geo_pass_mask)
        for k_local, j in enumerate(pass_idxs):
            align_scores[j] = alignment_cost_one(
                mag_pred_pass[k_local], bright_peak_idx_arr,
                ALIGN_WINDOW_EPOCHS, BRIGHT_MAG_THRESHOLD,
            )
        dt_pass = time.time() - t_p
        if n_pass > 0:
            per_cand_pass = dt_pass / n_pass
            for j in pass_idxs:
                align_eval_time_per_cand[cell_idx * M_q + j] = per_cand_pass
        t_align_geo_pass += dt_pass
        n_align_geo_pass += n_pass

        # FAIL group — production skips this; only run for instrumentation.
        t_f = time.time()
        if MEASURE_GEO_FAIL_ALIGN:
            fail_mask = ~geo_pass_mask if n_spec > 0 else np.zeros(M_q, dtype=bool)
            fail_idxs, mag_pred_fail = _surrogate_batch(fail_mask)
            for k_local, j in enumerate(fail_idxs):
                align_scores[j] = alignment_cost_one(
                    mag_pred_fail[k_local], bright_peak_idx_arr,
                    ALIGN_WINDOW_EPOCHS, BRIGHT_MAG_THRESHOLD,
                )
            dt_fail = time.time() - t_f
            if n_fail > 0:
                per_cand_fail = dt_fail / max(n_fail, 1)
                for j in fail_idxs:
                    align_eval_time_per_cand[cell_idx * M_q + j] = per_cand_fail
        else:
            dt_fail = time.time() - t_f
        t_align_geo_fail += dt_fail
        n_align_geo_fail += n_fail

        t_align_total += dt_pass + dt_fail
        cand_align_score[cand_slice] = align_scores.astype(np.float32)

        # --- Buffer survivor LCs (passes BOTH thresholds) ---
        cell_survivor_mask = (
            np.isfinite(geo_scores) & (geo_scores >= geo_threshold) &
            np.isfinite(align_scores) & (align_scores >= align_threshold)
        )
        for j in np.where(cell_survivor_mask)[0]:
            # Need this candidate's mag_pred — it must be in the PASS batch
            # (since survivors must pass geo). Look it up.
            local_idx = np.where(pass_idxs == j)[0]
            if local_idx.size > 0:
                cand_global = cell_idx * M_q + j
                candidate_lc_buffer.append(
                    (cand_global, mag_pred_pass[local_idx[0]].copy())
                )
            elif n_spec == 0:
                # geo undefined → survivor was eval'd as PASS by convention
                local_idx = np.where(pass_idxs == j)[0]
                if local_idx.size > 0:
                    cand_global = cell_idx * M_q + j
                    candidate_lc_buffer.append(
                        (cand_global, mag_pred_pass[local_idx[0]].copy())
                    )

        # Print progress every 30 sec
        if verbose and (time.time() - last_print) > 30.0:
            done = cell_idx + 1
            elapsed = time.time() - t_loop_start
            rate = done / elapsed
            eta = (N_cells - done) / rate
            print(f"  cell {done}/{N_cells} ({100*done/N_cells:.1f}%); "
                  f"geo_total={t_geo_total:.1f}s, align_total={t_align_total:.1f}s; "
                  f"survivors_so_far={len(candidate_lc_buffer)}; "
                  f"ETA={eta/60:.1f} min", flush=True)
            last_print = time.time()

    if not cell_loop_done:
        loop_wall = time.time() - t_loop_start
    if verbose:
        print(f"[seed {seed}] loop wall: {loop_wall:.1f} s", flush=True)
        print(f"  geo_total: {t_geo_total:.1f} s; align_total: {t_align_total:.1f} s",
              flush=True)
        print(f"  align time on geo-PASS candidates: {t_align_geo_pass:.1f} s "
              f"({n_align_geo_pass} cands)", flush=True)
        print(f"  align time on geo-FAIL candidates: {t_align_geo_fail:.1f} s "
              f"({n_align_geo_fail} cands)", flush=True)
        if t_align_geo_fail > 0:
            print(f"  WOULD-HAVE-SAVED by skipping geo-fails: "
                  f"{t_align_geo_fail:.1f} s "
                  f"({100*t_align_geo_fail/t_align_total:.1f}% of align time)",
                  flush=True)

    # ---- Categorise candidates ----
    geo_pass = (cand_geo_score >= geo_threshold) & np.isfinite(cand_geo_score)
    align_pass = (cand_align_score >= align_threshold) & np.isfinite(cand_align_score)
    cat_both = geo_pass & align_pass
    cat_geo_only = geo_pass & ~align_pass
    cat_align_only = ~geo_pass & align_pass
    cat_neither = ~geo_pass & ~align_pass
    if verbose:
        print(f"\n[seed {seed}] CATEGORISATION ({N_total} candidates):", flush=True)
        print(f"  passed_both       : {cat_both.sum():>10d} "
              f"({100*cat_both.sum()/N_total:.4f}%)", flush=True)
        print(f"  passed_geo_only   : {cat_geo_only.sum():>10d} "
              f"({100*cat_geo_only.sum()/N_total:.4f}%)", flush=True)
        print(f"  passed_align_only : {cat_align_only.sum():>10d} "
              f"({100*cat_align_only.sum()/N_total:.4f}%)", flush=True)
        print(f"  rejected_both     : {cat_neither.sum():>10d} "
              f"({100*cat_neither.sum()/N_total:.4f}%)", flush=True)

    # ---- Truth proximity diagnostics for survivors ----
    # Body-twin (X-flip) is independent of survivors; compute unconditionally
    # so it's available for the no-survivors checkpoint path.
    q_180x = np.array([0.0, 1.0, 0.0, 0.0])
    twin_q0 = quat_multiply(q_180x, q0_truth)
    survivor_idx = np.where(cat_both)[0]
    if survivor_idx.size > 0:
        surv_q0 = cand_q0[survivor_idx].astype(np.float64)
        geos_truth = np.array([angular_dist_deg(q, q0_truth) for q in surv_q0])
        geos_twin = np.array([angular_dist_deg(q, twin_q0) for q in surv_q0])
        if verbose:
            print(f"\n[seed {seed}] SURVIVOR Q0 DIAGNOSTICS (n={survivor_idx.size}):",
                  flush=True)
            print(f"  q0_geodesic_to_truth: min={geos_truth.min():.2f}°, "
                  f"median={np.median(geos_truth):.2f}°, "
                  f"max={geos_truth.max():.2f}°", flush=True)
            print(f"  q0_geodesic_to_twin : min={geos_twin.min():.2f}°, "
                  f"median={np.median(geos_twin):.2f}°, "
                  f"max={geos_twin.max():.2f}°", flush=True)
    else:
        geos_truth = np.array([])
        geos_twin = np.array([])
        if verbose:
            print(f"\n[seed {seed}] NO SURVIVORS in cat_both. Filter is too strict, "
                  "or truth-IC was not in pool.", flush=True)

    # ---- Save checkpoints ----
    if verbose:
        print(f"\n[seed {seed}] saving checkpoints", flush=True)

    np.savez_compressed(out_dir / "omega_grid.npz",
                        omega_dirs=omega_dirs,
                        omega_mags=omega_mags,
                        omega_vectors=omega_vectors,
                        omega_dir_dist_to_truth_deg=omega_dir_dist_to_truth,
                        truth_omega_dir=truth_dir,
                        truth_omega_mag_rad=omega_mag_truth,
                        nearest_dir_idx=nearest_dir_idx,
                        nearest_dir_deg=nearest_dir_deg)

    np.savez_compressed(out_dir / "delta_trajectories.npz",
                        delta_quats=delta_quats_all)

    np.savez_compressed(out_dir / "q_target_pool.npz",
                        q_target=pool["q_target"],
                        peak_epoch_idx=pool["peak_epoch_idx"],
                        face_idx=pool["face_idx"],
                        phi_idx=pool["phi_idx"],
                        tier_idx=pool["tier_idx"],
                        peak_array_idx=pool["peak_array_idx"],
                        tier_at_peak=pool["tier_at_peak"],
                        mag_abs_at_peak=pool["mag_abs_at_peak"],
                        n_peaks_classifiable=pool["n_peaks_classifiable"])

    np.savez_compressed(out_dir / "candidates_meta.npz",
                        q0=cand_q0,
                        omega_cell_idx=cand_omega_cell,
                        q_target_idx=cand_qt_idx,
                        geo_score=cand_geo_score,
                        align_score=cand_align_score,
                        align_eval_time_s=align_eval_time_per_cand,
                        cat_both=cat_both, cat_geo_only=cat_geo_only,
                        cat_align_only=cat_align_only, cat_neither=cat_neither)

    np.savez_compressed(out_dir / "thresholds.npz",
                        geo_threshold=geo_threshold,
                        align_threshold=align_threshold,
                        truth_geo_score=truth_geo_score,
                        truth_align_score=truth_align_score,
                        geo_threshold_deg=GEO_THRESHOLD_DEG,
                        align_window_epochs=ALIGN_WINDOW_EPOCHS,
                        bright_mag_threshold=BRIGHT_MAG_THRESHOLD,
                        spec_threshold_deg=SPEC_THRESHOLD_DEG)

    np.savez_compressed(out_dir / "spec_geometry.npz",
                        spec_event_eps=spec_event_eps,
                        spec_event_tier=spec_event_tier,
                        bright_peak_idx=bright_peak_idx_arr,
                        face_normals=face_normals,
                        tier_face_idx_T1=tier_table["tier_face_idx"][0],
                        tier_face_idx_T2=tier_table["tier_face_idx"][1],
                        tier_face_idx_T3=tier_table["tier_face_idx"][2],
                        tier_face_idx_T4=tier_table["tier_face_idx"][3],
                        truth_pab_body=truth_pab_body)

    # Survivor LCs only (per the user's instruction).
    if candidate_lc_buffer:
        surv_idx_arr = np.array([t[0] for t in candidate_lc_buffer], dtype=np.int64)
        surv_lc_arr = np.array([t[1] for t in candidate_lc_buffer], dtype=np.float32)
    else:
        surv_idx_arr = np.zeros(0, dtype=np.int64)
        surv_lc_arr = np.zeros((0, N_obs), dtype=np.float32)
    np.savez_compressed(out_dir / "survivor_lcs.npz",
                        survivor_cand_idx=surv_idx_arr,
                        survivor_mag_pred=surv_lc_arr,
                        truth_mag_hifi=mag_hifi.astype(np.float32),
                        truth_mag_pred_via_surrogate=truth_mag_pred.astype(np.float32),
                        observation_times=obs_times)

    if survivor_idx.size > 0:
        np.savez_compressed(out_dir / "survivor_diagnostics.npz",
                            survivor_q0=cand_q0[survivor_idx],
                            survivor_omega_cell=cand_omega_cell[survivor_idx],
                            survivor_q_target=cand_qt_idx[survivor_idx],
                            survivor_geo_score=cand_geo_score[survivor_idx],
                            survivor_align_score=cand_align_score[survivor_idx],
                            survivor_geodesic_to_truth_deg=geos_truth.astype(np.float32),
                            survivor_geodesic_to_twin_deg=geos_twin.astype(np.float32),
                            truth_q0=q0_truth, twin_q0=twin_q0,
                            truth_omega=omega0_truth)
    else:
        np.savez_compressed(out_dir / "survivor_diagnostics.npz",
                            survivor_q0=np.zeros((0, 4)),
                            survivor_omega_cell=np.zeros(0, dtype=np.int32),
                            survivor_q_target=np.zeros(0, dtype=np.int32),
                            survivor_geo_score=np.zeros(0),
                            survivor_align_score=np.zeros(0),
                            survivor_geodesic_to_truth_deg=np.zeros(0),
                            survivor_geodesic_to_twin_deg=np.zeros(0),
                            truth_q0=q0_truth, twin_q0=twin_q0,
                            truth_omega=omega0_truth)

    summary = {
        "seed": seed,
        "config": {
            "N_OMEGA_DIR": N_OMEGA_DIR,
            "N_OMEGA_MAG_CELLS": N_OMEGA_MAG_CELLS,
            "N_PHI_STEPS": N_PHI_STEPS,
            "PHI_DEG_STEP": PHI_DEG_STEP,
            "BRIGHT_MAG_THRESHOLD": BRIGHT_MAG_THRESHOLD,
            "SPEC_THRESHOLD_DEG": SPEC_THRESHOLD_DEG,
            "GEO_THRESHOLD_DEG": GEO_THRESHOLD_DEG,
            "ALIGN_WINDOW_EPOCHS": ALIGN_WINDOW_EPOCHS,
        },
        "scale": {
            "N_obs": int(N_obs),
            "N_dir": int(N_dir),
            "N_mag": int(N_mag),
            "N_cells": int(N_cells),
            "M_q_target": int(M_q),
            "N_total_candidates": int(N_total),
            "n_classifiable_peaks": int(n_peaks_used),
            "n_spec_events": int(n_spec),
            "n_bright_peaks": int(bright_peak_idx_arr.size),
        },
        "bracket": {
            "n_full_cells": int(bracket["bracket_full_grid"].size),
            "selected_cells_rad_s": bracket["bracket_cells"].tolist(),
            "selected_cells_dps": np.degrees(bracket["bracket_cells"]).tolist(),
            "truth_omega_mag_rad": omega_mag_truth,
            "truth_omega_mag_dps": float(np.degrees(omega_mag_truth)),
            "nearest_cell_pct": float(bracket["nearest_cell_pct"]),
        },
        "thresholds": {
            "truth_geo_score": float(truth_geo_score),
            "truth_align_score": float(truth_align_score),
            "geo_threshold": float(geo_threshold),
            "align_threshold": float(align_threshold),
        },
        "categorisation": {
            "passed_both": int(cat_both.sum()),
            "passed_geo_only": int(cat_geo_only.sum()),
            "passed_align_only": int(cat_align_only.sum()),
            "rejected_both": int(cat_neither.sum()),
        },
        "survivor_diagnostics": {
            "n_survivors": int(survivor_idx.size),
            "min_geodesic_to_truth_deg": (float(geos_truth.min())
                                          if geos_truth.size else None),
            "median_geodesic_to_truth_deg": (float(np.median(geos_truth))
                                              if geos_truth.size else None),
            "min_geodesic_to_twin_deg": (float(geos_twin.min())
                                         if geos_twin.size else None),
            "median_geodesic_to_twin_deg": (float(np.median(geos_twin))
                                             if geos_twin.size else None),
        },
        "timing": {
            "bracket_wall_s": float(bracket_wall),
            "loop_wall_s": float(loop_wall),
            "geo_total_s": float(t_geo_total),
            "align_total_s": float(t_align_total),
            "align_on_geo_pass_s": float(t_align_geo_pass),
            "align_on_geo_fail_s": float(t_align_geo_fail),
            "n_align_geo_pass": int(n_align_geo_pass),
            "n_align_geo_fail": int(n_align_geo_fail),
            "would_have_saved_s": float(t_align_geo_fail),
            "total_wall_s": float(time.time() - t_total_start),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n[seed {seed}] Saved checkpoints to {out_dir}", flush=True)
        for fn in sorted(out_dir.glob("*.npz")):
            print(f"  {fn.name}: {fn.stat().st_size / 1e6:.2f} MB", flush=True)
        print(f"  summary.json", flush=True)
        print(f"\n[seed {seed}] TOTAL WALL: {time.time() - t_total_start:.1f} s",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="m048 seed (0-99)")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Output root (default: results/s020/)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny grid (N_dir=20, N_mag=2) for mechanics check")
    parser.add_argument("--n-dir", type=int, default=None,
                        help="Override N_OMEGA_DIR")
    parser.add_argument("--n-mag", type=int, default=None,
                        help="Override N_OMEGA_MAG_CELLS")
    parser.add_argument("--n-phi", type=int, default=None,
                        help="Override N_PHI_STEPS (phi-sweep granularity)")
    parser.add_argument("--measure-geo-fail", action="store_true",
                        help="Run alignment cost on geo-FAIL candidates too "
                             "(instrumentation, ~6.7x slower; default OFF)")
    args = parser.parse_args()

    global N_OMEGA_DIR, N_OMEGA_MAG_CELLS, N_PHI_STEPS, PHI_DEG_STEP, MEASURE_GEO_FAIL_ALIGN
    if args.smoke:
        N_OMEGA_DIR = 20
        N_OMEGA_MAG_CELLS = 2
    if args.n_dir is not None:
        N_OMEGA_DIR = args.n_dir
    if args.n_mag is not None:
        N_OMEGA_MAG_CELLS = args.n_mag
    if args.n_phi is not None:
        N_PHI_STEPS = args.n_phi
        PHI_DEG_STEP = 360.0 / N_PHI_STEPS
    if args.measure_geo_fail:
        MEASURE_GEO_FAIL_ALIGN = True

    if args.out_root is None:
        out_root = SURVEY_DIR / "results" / "s020"
    else:
        out_root = Path(args.out_root)
    sub = "smoke" if args.smoke else f"seed{args.seed:03d}"
    out_dir = out_root / sub if args.smoke else out_root / f"seed{args.seed:03d}"
    print(f"=== s020 pipeline | seed={args.seed} | "
          f"N_dir={N_OMEGA_DIR}, N_mag={N_OMEGA_MAG_CELLS} | "
          f"measure_geo_fail={MEASURE_GEO_FAIL_ALIGN} | out={out_dir} ===",
          flush=True)
    run_seed(args.seed, out_dir, verbose=True)


if __name__ == "__main__":
    main()
