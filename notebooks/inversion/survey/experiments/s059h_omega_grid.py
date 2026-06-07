"""s059h — ω-grid scoring with ψ-factorization and surrogate-MSE local-window cost.

The architectural reform after s059g:

  Old generator: all-pairs finite-diff ω over (C_a × C_b). Produces 184k
  candidates with ω errors ~17% on high-|ω| seeds. Score is binary "hits
  validator pool within 5°" — clipped at pool resolution (~7°).

  New generator (this script):
    - C_a: ~hundreds of q candidates from cloud at anchor.
    - ω_body grid: Fibonacci sphere × magnitudes within polhode-prior bracket.
    - Per ω in grid: ONE real-dynamics ODE solve from (identity, ω_body) →
      ψ_body(t) at every epoch in local window.
    - Per (q_a, ω) candidate: q(t) = q_a ⊗ ψ(t) (batched matmul, free).
    - Score: surrogate-MSE between (q, ω)-implied LC and observed LC over
      local window only. Continuous, no pool quantization.

ψ-factorization saves |C_a|× duplicate ODE solves. With |C_a|=200 and
N_ω=5000, that's 5000 ODE solves instead of 1M.

Truth-ω is in the grid by construction (modulo magnitude/direction grid
spacing), so we don't depend on noisy finite-diff ω estimates.

For now |ω|-bracket uses truth's |ω| ± 30% as a placeholder for the
polhode-prior-derived bracket (s055a, ~25% MAPE). This is the same oracle
the existing pipeline already uses.

Usage:
    python experiments/s059h_omega_grid.py --seed 28
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import (
    fd_omega_passive, quat_ang_deg_batch, ang_to_axis,
    wxyz_to_xyzw, xyzw_to_wxyz, quat_ang_deg, back_propagate,
    LM_MAX_NFEV, LM_FTOL, LM_XTOL, RESIDUAL_CAP,
)
from experiments.s059d_early_anchor import stage_pick_early_anchor
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.surrogate_eval import predict as surrogate_predict
from lib.traj_load import load_truth as _load_traj
from src.dynamics.attitude_propagator import propagate_attitude


# ---- defaults -----------------------------------------------------------

WINDOW_W = 5              # ±W epochs (= 11 epochs) — small window keeps eval count under control
N_DIRECTIONS = 600        # Fibonacci sphere points (~7° spacing)
N_MAGNITUDES = 8          # magnitude steps in |ω|-prior bracket
OMEGA_MAG_BRACKET = (0.7, 1.3)
OMEGA_PRIOR_PCT = 30      # placeholder; in production use polhode-prior |ω|
TOP_K_POLISH = 5
RTOL_PSI = 1e-6
ATOL_PSI = 1e-8
SURROGATE_BATCH_OMEGA = 20  # smaller chunks for visible progress
N_WORKERS = 24            # 32-core Ryzen 9950X3D; surrogate is ~14k/s single-thread → 24 workers ≈ 280k/s aggregate

# Worker globals (forked from parent — CoW)
_W_INERTIA = None
_W_DT_WINDOW_SIGNED = None
_W_C_A = None
_W_SUN_WIN = None
_W_OBS_WIN = None
_W_OBS_DIST_WIN = None
_W_MAG_TRUTH_WIN = None


def _worker_init_psi(inertia, dt_window_signed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _W_INERTIA, _W_DT_WINDOW_SIGNED
    _W_INERTIA = inertia
    _W_DT_WINDOW_SIGNED = dt_window_signed


def _worker_psi_chunk(args):
    chunk_id, omega_chunk = args
    psi_chunk = compute_psi_grid(omega_chunk, _W_DT_WINDOW_SIGNED, _W_INERTIA)
    return chunk_id, psi_chunk


def _worker_init_score(inertia, dt_window_signed, C_a, sun_win,
                        obs_win, obs_dist_win, mag_truth_win):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _W_INERTIA, _W_DT_WINDOW_SIGNED, _W_C_A
    global _W_SUN_WIN, _W_OBS_WIN, _W_OBS_DIST_WIN, _W_MAG_TRUTH_WIN
    _W_INERTIA = inertia
    _W_DT_WINDOW_SIGNED = dt_window_signed
    _W_C_A = C_a
    _W_SUN_WIN = sun_win
    _W_OBS_WIN = obs_win
    _W_OBS_DIST_WIN = obs_dist_win
    _W_MAG_TRUTH_WIN = mag_truth_win
    # warm surrogate
    from lib.surrogate_eval import get_model
    get_model()


def _worker_score_chunk(args):
    chunk_id, omega_chunk, psi_chunk = args
    scores_chunk = score_window_mse(
        _W_C_A, omega_chunk, psi_chunk,
        _W_SUN_WIN, _W_OBS_WIN, _W_OBS_DIST_WIN, _W_MAG_TRUTH_WIN, log=None,
    )
    return chunk_id, scores_chunk

# ---- helpers ------------------------------------------------------------


def fibonacci_sphere(n):
    """Approximately uniform points on the unit sphere via Fibonacci spiral."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def build_omega_grid(om_mag_estimate, n_dir, n_mag, mag_bracket):
    """Build (n_dir × n_mag, 3) body-frame ω grid in rad/s."""
    dirs = fibonacci_sphere(n_dir)
    mags = np.linspace(mag_bracket[0], mag_bracket[1], n_mag) * om_mag_estimate
    grid = (dirs[:, None, :] * mags[None, :, None]).reshape(-1, 3)
    return grid, dirs, mags


def compute_psi_grid(omega_body_grid, dt_window_signed, inertia, log=None):
    """Per-ω: ODE-integrate (identity, ω_body) over window times.

    Returns ψ_grid (N_omega, n_window, 4) wxyz.
    """
    fwd_dt = sorted({t for t in dt_window_signed if t > 0})
    bwd_dt_pos = sorted({-t for t in dt_window_signed if t < 0})  # absolute, ascending
    fwd_times = np.concatenate([[0.0], fwd_dt]) if fwd_dt else np.array([0.0])
    bwd_times = np.concatenate([[0.0], bwd_dt_pos]) if bwd_dt_pos else np.array([0.0])
    fwd_idx = {t: 1 + i for i, t in enumerate(fwd_dt)}
    bwd_idx = {t: 1 + i for i, t in enumerate(bwd_dt_pos)}

    N = omega_body_grid.shape[0]
    n_window = len(dt_window_signed)
    psi_grid = np.empty((N, n_window, 4), dtype=np.float64)
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    t0 = time.time()
    for n in range(N):
        omega = omega_body_grid[n]
        # forward
        if len(fwd_times) > 1:
            quats_fwd, _ = propagate_attitude(
                q0=identity, omega0=omega, times=fwd_times,
                mode="tumbling", inertia_tensor=inertia,
                rtol=RTOL_PSI, atol=ATOL_PSI,
            )
        else:
            quats_fwd = identity[None]
        # backward: time-reversal symmetry — propagate (identity, -ω) forward
        if len(bwd_times) > 1:
            quats_bwd, _ = propagate_attitude(
                q0=identity, omega0=-omega, times=bwd_times,
                mode="tumbling", inertia_tensor=inertia,
                rtol=RTOL_PSI, atol=ATOL_PSI,
            )
        else:
            quats_bwd = identity[None]

        for ti, t in enumerate(dt_window_signed):
            if t > 0:
                psi_grid[n, ti] = quats_fwd[fwd_idx[t]]
            elif t < 0:
                psi_grid[n, ti] = quats_bwd[bwd_idx[-t]]
            else:
                psi_grid[n, ti] = identity

        if log is not None and n > 0 and n % 500 == 0:
            log(f"    ψ {n}/{N} ({time.time()-t0:.1f}s)")
    return psi_grid


def quat_mul_batch(a, b):
    """Batched quaternion product (wxyz). a, b shape (..., 4)."""
    aw = a[..., 0]; ax = a[..., 1]; ay = a[..., 2]; az = a[..., 3]
    bw = b[..., 0]; bx = b[..., 1]; by = b[..., 2]; bz = b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def quat_rotate(q, v):
    """v_rot = q v q^* — active rotation by q. Matches
    scipy.spatial.transform.Rotation.from_quat([qx,qy,qz,qw]).apply(v).
    In this codebase, propagate_attitude's output q is used by lib.forward
    as `R(q) @ v_inertial = v_body`, so this function maps inertial → body.
    """
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2 * np.cross(qxyz, v)
    return v + qw * t + np.cross(qxyz, t)


def score_window_mse(C_a_wxyz, omega_body_grid, psi_grid,
                     sun_inertial_window, obs_inertial_window,
                     obs_dist_window, mag_truth_window, log=None):
    """Score every (q_a, ω) by surrogate-MSE on the local window — raw numpy.

    Returns: scores (n_a, n_omega) — surrogate MSE per (q_a, ω).
    """
    n_a = C_a_wxyz.shape[0]
    n_omega = omega_body_grid.shape[0]
    n_window = sun_inertial_window.shape[0]
    scores = np.empty((n_a, n_omega), dtype=np.float64)

    # sun_inertial_window / obs_inertial_window are EXPECTED to be unit vectors
    # in inertial frame (sun-relative-to-sat / obs-relative-to-sat, normalised).
    # Caller must precompute these.
    sun_w = sun_inertial_window[None, None, :, :]  # (1, 1, n_window, 3)
    obs_w = obs_inertial_window[None, None, :, :]
    obs_dist_b = np.broadcast_to(obs_dist_window[None, None, :],
                                  (n_a, SURROGATE_BATCH_OMEGA, n_window))

    t0 = time.time()
    for j_start in range(0, n_omega, SURROGATE_BATCH_OMEGA):
        j_end = min(j_start + SURROGATE_BATCH_OMEGA, n_omega)
        chunk_size = j_end - j_start

        psi_chunk = psi_grid[j_start:j_end]  # (chunk, n_window, 4)

        # q_t[a, j, t] = q_a[a] ⊗ ψ[j, t] — raw numpy quaternion product.
        # Broadcast: (n_a, 1, 1, 4) and (1, chunk, n_window, 4) → (n_a, chunk, n_window, 4)
        q_a_brd = C_a_wxyz[:, None, None, :]
        psi_brd = psi_chunk[None, :, :, :]
        q_t = quat_mul_batch(q_a_brd, psi_brd)  # (n_a, chunk, n_window, 4)

        # k1, k2 in body frame: R(q) @ v_inertial_unit = v_body
        k1 = quat_rotate(q_t, sun_w)  # (n_a, chunk, n_window, 3)
        k2 = quat_rotate(q_t, obs_w)
        k1_flat = k1.reshape(-1, 3)
        k2_flat = k2.reshape(-1, 3)

        # surrogate predict
        if chunk_size != SURROGATE_BATCH_OMEGA:
            od_flat = np.broadcast_to(obs_dist_window[None, None, :],
                                       (n_a, chunk_size, n_window)).reshape(-1)
        else:
            od_flat = obs_dist_b.reshape(-1)
        mag_pred = surrogate_predict(k1_flat, k2_flat, od_flat)
        mag_pred = mag_pred.reshape(n_a, chunk_size, n_window)

        # MSE per (q_a, ω) across window
        diff = mag_pred - mag_truth_window[None, None, :]
        scores[:, j_start:j_end] = np.mean(diff ** 2, axis=2)

        if log is not None:
            log(f"    score chunk {j_end}/{n_omega} ({time.time()-t0:.1f}s)")
    return scores


def make_residual_local_omega_body(q_a_rep_wxyz, om_body_rep, t_a_idx, W,
                                    ctx, target):
    """LM polish residual at anchor; params = (rotvec_a, ω_body_a).

    Each call: build (q_a, ω_body), back-prop to (q_0, ω_0_body) at lo,
    forward-prop to fill window, surrogate-render, residual on window.
    """
    q_a_rep_xyzw = wxyz_to_xyzw(q_a_rep_wxyz)
    R_seed = Rotation.from_quat(q_a_rep_xyzw)
    inertia = ctx["inertia_tensor"]
    obs_times = ctx["observation_times"]
    n = len(obs_times)
    lo = max(0, t_a_idx - W); hi = min(n, t_a_idx + W + 1)
    times_local = obs_times[lo:hi]
    sun_local = ctx["sun_pos"][lo:hi]
    obs_local = ctx["obs_pos"][lo:hi]
    sat_local = ctx["sat_pos"][lo:hi]
    od_local = ctx["obs_dist"][lo:hi]
    target_local = target[lo:hi]
    dt_back = float(obs_times[t_a_idx] - obs_times[lo])

    from lib.forward import propagate_to_body_frame  # avoid top-level shadow

    def residual(params):
        rotvec_a = params[:3]
        om_body = params[3:]
        try:
            q_a_xyzw = (Rotation.from_rotvec(rotvec_a) * R_seed).as_quat()
            q_a_wxyz = xyzw_to_wxyz(q_a_xyzw)
            if dt_back > 0:
                q_lo, om_lo = back_propagate(q_a_wxyz, om_body, dt_back, inertia)
            else:
                q_lo, om_lo = q_a_wxyz, om_body
            k1, k2, _ = propagate_to_body_frame(
                q0_wxyz=q_lo, omega0_rad=om_lo,
                observation_times=times_local,
                sun_pos=sun_local, obs_pos=obs_local, sat_pos=sat_local,
                inertia_tensor=inertia, mode="tumbling",
            )
            pred = surrogate_predict(k1, k2, od_local)
            r = pred - target_local
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full(2 * W + 1, RESIDUAL_CAP)
    return residual


def lm_polish(q_a_rep_wxyz, om_body_rep, t_a_idx, W, ctx, target):
    residual = make_residual_local_omega_body(q_a_rep_wxyz, om_body_rep,
                                               t_a_idx, W, ctx, target)
    x0 = np.concatenate([np.zeros(3), om_body_rep])
    r_seed = residual(x0)
    mse_seed = float(np.mean(r_seed ** 2))
    t0 = time.time()
    res = least_squares(residual, x0, method="lm",
                        max_nfev=LM_MAX_NFEV, ftol=LM_FTOL, xtol=LM_XTOL)
    wall = time.time() - t0
    rotvec = res.x[:3]
    om_body_pol = res.x[3:]
    q_a_pol_xyzw = (Rotation.from_rotvec(rotvec) *
                    Rotation.from_quat(wxyz_to_xyzw(q_a_rep_wxyz))).as_quat()
    q_a_pol = xyzw_to_wxyz(q_a_pol_xyzw)
    t_a_seconds = float(ctx["observation_times"][t_a_idx] -
                         ctx["observation_times"][0])
    q0_pol, om0_pol_body = back_propagate(q_a_pol, om_body_pol, t_a_seconds,
                                            ctx["inertia_tensor"])
    mse_pol = float(np.mean(res.fun ** 2))
    return {
        "q_a_pol_wxyz": q_a_pol, "om_body_pol": om_body_pol,
        "q0_pol_wxyz": q0_pol, "om0_pol_body": om0_pol_body,
        "rotvec_pol_mag_deg": float(np.degrees(np.linalg.norm(rotvec))),
        "surrogate_mse_local_seed": mse_seed,
        "surrogate_mse_local_polished": mse_pol,
        "surrogate_rho_local_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_local_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(res.nfev), "wall_s": wall,
    }


# ---- main ---------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--anchor-window", type=int, default=30)
    p.add_argument("--window", type=int, default=WINDOW_W)
    p.add_argument("--n-dir", type=int, default=N_DIRECTIONS)
    p.add_argument("--n-mag", type=int, default=N_MAGNITUDES)
    p.add_argument("--top-k", type=int, default=TOP_K_POLISH)
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059h_seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir = SURVEY / "results" / f"s059_seed{args.seed:03d}"
    log_lines = []
    log_path = out_dir / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)
        log_path.write_text("\n".join(log_lines) + "\n")

    t_overall = time.time()
    log(f"=== s059h ω-grid + ψ-factorize — seed {args.seed} ===")

    # cloud (cache)
    log("\n[1/5] cloud (cache)")
    z = np.load(cloud_dir / "cloud.npz")
    q_pool = z["q_pool_wxyz"]
    survive_all = z["survive_all"]
    obs_times = z["obs_times"]
    n_epochs = len(obs_times)
    dt_epoch = float(np.median(np.diff(obs_times)))

    # truth
    truth = _load_traj(args.seed)
    q_truth_t = truth["quaternions"]
    om_truth_body_t0 = truth["omega0_rad"]  # body-frame ω at t=0
    log(f"truth |ω|_0 (body) = {np.linalg.norm(om_truth_body_t0):.6f} rad/s")

    ctx = build_context(seed=args.seed)
    inertia = ctx["inertia_tensor"]
    log(f"inertia eigvals: {np.linalg.eigvalsh(inertia)}")

    # anchor
    log("\n[2/5] anchor")
    T_A = stage_pick_early_anchor(survive_all, log, window=args.anchor_window)
    idx_a = np.where(survive_all[T_A])[0]
    C_a = q_pool[idx_a]
    log(f"  C_a: {len(C_a)} candidates at T_A={T_A}")

    # local window
    W = args.window
    lo = max(0, T_A - W); hi = min(n_epochs, T_A + W + 1)
    window_idx = np.arange(lo, hi)
    dt_window_signed = (window_idx - T_A) * dt_epoch
    n_window = len(window_idx)
    log(f"  window: [{lo}, {hi}) — {n_window} epochs around T_A")

    # truth's body-frame ω at T_A (for grid centering — use polhode prior in production)
    # Propagate truth from t=0 to T_A to get ω at T_A.
    times_to_TA = np.array([0.0, float(obs_times[T_A] - obs_times[0])])
    quats_truth_segment, omegas_truth_segment = propagate_attitude(
        q0=truth["q0_wxyz"], omega0=om_truth_body_t0, times=times_to_TA,
        mode="tumbling", inertia_tensor=inertia, rtol=1e-9, atol=1e-12,
    )
    om_truth_body_TA = omegas_truth_segment[1]
    om_truth_mag_TA = float(np.linalg.norm(om_truth_body_TA))
    log(f"  truth |ω|_body at T_A: {om_truth_mag_TA:.6f} rad/s")

    # ω-grid
    log("\n[3/5] ω-grid + ψ-factorize")
    omega_grid, dirs, mags = build_omega_grid(
        om_truth_mag_TA, args.n_dir, args.n_mag, OMEGA_MAG_BRACKET)
    n_omega = omega_grid.shape[0]
    log(f"  ω grid: {args.n_dir} dirs × {args.n_mag} mags = {n_omega} ω's "
        f"(|ω| range: {mags.min():.6f} to {mags.max():.6f})")

    t_psi = time.time()
    # Pool(N_WORKERS) parallel ψ-grid computation
    omega_chunks_psi = np.array_split(omega_grid, N_WORKERS)
    psi_pieces = [None] * N_WORKERS
    with get_context("fork").Pool(
        N_WORKERS, initializer=_worker_init_psi,
        initargs=(inertia, dt_window_signed),
    ) as pool:
        for chunk_id, psi_chunk in pool.imap_unordered(
            _worker_psi_chunk,
            [(i, omega_chunks_psi[i]) for i in range(N_WORKERS)],
        ):
            psi_pieces[chunk_id] = psi_chunk
            log(f"    ψ chunk {chunk_id} done")
    psi_grid = np.concatenate(psi_pieces, axis=0)
    log(f"  ψ-grid: {time.time()-t_psi:.1f}s ({psi_grid.shape[0]} ω, "
        f"Pool({N_WORKERS}))")

    # context for surrogate — sun/obs unit vectors in INERTIAL frame, relative to satellite
    sun_vec = ctx["sun_pos"][lo:hi] - ctx["sat_pos"][lo:hi]
    obs_vec = ctx["obs_pos"][lo:hi] - ctx["sat_pos"][lo:hi]
    sun_inertial_window = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_inertial_window = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
    obs_dist_window = ctx["obs_dist"][lo:hi]
    mag_truth_window = ctx["mag_hifi_truth"][lo:hi]

    log("\n[4/5] score grid (surrogate-MSE) — Pool({})".format(N_WORKERS))
    t_score = time.time()
    omega_chunks_score = np.array_split(omega_grid, N_WORKERS)
    psi_chunks_score = np.array_split(psi_grid, N_WORKERS)
    score_pieces = [None] * N_WORKERS
    with get_context("fork").Pool(
        N_WORKERS, initializer=_worker_init_score,
        initargs=(inertia, dt_window_signed, C_a, sun_inertial_window,
                   obs_inertial_window, obs_dist_window, mag_truth_window),
    ) as pool:
        for chunk_id, sc in pool.imap_unordered(
            _worker_score_chunk,
            [(i, omega_chunks_score[i], psi_chunks_score[i])
             for i in range(N_WORKERS)],
        ):
            score_pieces[chunk_id] = sc
            log(f"    score chunk {chunk_id} done")
    scores = np.concatenate(score_pieces, axis=1)  # (n_a, n_omega)
    log(f"  scoring: {time.time()-t_score:.1f}s")
    log(f"  MSE range [{scores.min():.4e}, {scores.max():.4e}]")
    log(f"  ρ_local range [{np.sqrt(scores.min())/0.05:.3f}, "
        f"{np.sqrt(scores.max())/0.05:.3f}]")

    # truth localization
    qa_dist_C_a = quat_ang_deg_batch(C_a, q_truth_t[T_A])
    truth_a_idx = int(np.argmin(qa_dist_C_a))
    log(f"  closest C_a to truth at T_A: idx={truth_a_idx}, qa_d={qa_dist_C_a[truth_a_idx]:.2f}°")

    # find the closest ω in the grid to truth's actual ω at T_A
    om_d_grid = ang_to_axis(omega_grid, om_truth_body_TA)
    om_mag_pct_grid = (np.linalg.norm(omega_grid, axis=1) - om_truth_mag_TA) / om_truth_mag_TA * 100
    om_truth_idx = int(np.argmin(om_d_grid + np.abs(om_mag_pct_grid)))
    log(f"  closest ω in grid to truth: idx={om_truth_idx}, "
        f"om_d={om_d_grid[om_truth_idx]:.2f}°, "
        f"|ω|Δ={om_mag_pct_grid[om_truth_idx]:+.2f}%")

    # truth's score in the grid
    truth_score = scores[truth_a_idx, om_truth_idx]
    truth_rho = float(np.sqrt(truth_score) / 0.05)
    flat_scores = scores.flatten()
    truth_flat_idx = truth_a_idx * n_omega + om_truth_idx
    truth_rank = int((flat_scores < truth_score).sum()) + 1  # ascending (lower MSE better)
    log(f"  TRUTH-CLOSEST GRID POINT: score={truth_score:.4e}, "
        f"ρ={truth_rho:.3f}, rank {truth_rank}/{flat_scores.size}")

    # top-K (lowest MSE) for polishing
    K = args.top_k
    flat_order = np.argsort(flat_scores)
    top_K_flat = flat_order[:K]
    top_K = [(idx // n_omega, idx % n_omega) for idx in top_K_flat]
    log(f"\n=== TOP {K} by surrogate-MSE on local window ===")
    log(f"{'rank':<5}{'a_idx':<7}{'om_idx':<8}{'qa_d°':<10}{'om_d°':<10}{'|ω|Δ%':<10}{'ρ_local':<10}")
    for r, (ai, oj) in enumerate(top_K):
        is_truth = " *" if ai == truth_a_idx and oj == om_truth_idx else ""
        log(f"{r+1:<5}{ai:<7}{oj:<8}"
            f"{qa_dist_C_a[ai]:<10.2f}{om_d_grid[oj]:<10.2f}"
            f"{om_mag_pct_grid[oj]:<10.2f}"
            f"{float(np.sqrt(scores[ai, oj])/0.05):<10.3f}{is_truth}")
    # ensure truth-closest grid point is in the polish set (diagnostic)
    if (truth_a_idx, om_truth_idx) not in top_K:
        log(f"  truth-closest grid point not in top-{K}; ranked {truth_rank}")

    # LM polish each top-K candidate (+ truth-closest as diagnostic)
    log(f"\n[5/5] LM polish + hi-fi classify")
    target = ctx["mag_hifi_truth"]
    polish_set = list(top_K)
    if (truth_a_idx, om_truth_idx) not in polish_set:
        polish_set.append((truth_a_idx, om_truth_idx))

    polished = []
    for r, (ai, oj) in enumerate(polish_set):
        is_truth_inj = (ai == truth_a_idx and oj == om_truth_idx)
        rank_label = (r + 1) if (ai, oj) in top_K else f"truth-inj"
        q_a_seed = C_a[ai]
        om_body_seed = omega_grid[oj]
        result = lm_polish(q_a_seed, om_body_seed, T_A, W, ctx, target)
        result["rank"] = rank_label
        result["is_truth_grid"] = is_truth_inj
        result["a_idx"] = int(ai); result["om_idx"] = int(oj)

        # errors vs truth at t=0
        result["q0_err_polished_deg"] = quat_ang_deg(
            result["q0_pol_wxyz"], ctx["q0_truth"])
        # om0_pol_body vs truth body-frame at t=0
        om_pol_body = result["om0_pol_body"]
        om_truth_body = ctx["omega0_truth_rad"]
        ot_mag = float(np.linalg.norm(om_truth_body))
        result["om_mag_err_pct"] = float(
            (np.linalg.norm(om_pol_body) - ot_mag) / ot_mag * 100)
        result["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om_pol_body / np.linalg.norm(om_pol_body),
                       om_truth_body / ot_mag)), 0, 1))))

        log(f"  rank {rank_label}  ai={ai}  oj={oj}{' (TRUTH-grid)' if is_truth_inj else ''}  "
            f"local ρ_seed={result['surrogate_rho_local_seed']:.3f} → "
            f"ρ_pol={result['surrogate_rho_local_polished']:.3f}  "
            f"q0_err={result['q0_err_polished_deg']:.2f}°  "
            f"|ω|_err={result['om_mag_err_pct']:+.2f}%  "
            f"ω_dir_err={result['om_dir_err_deg']:.2f}°  "
            f"n_eval={result['n_eval']}  wall={result['wall_s']:.1f}s")
        polished.append(result)

    # hi-fi each
    log("\nhi-fi rendering full trajectory for each polish...")
    for p in polished:
        t0 = time.time()
        try:
            pred = render_hifi(p["q0_pol_wxyz"], p["om0_pol_body"], ctx)
            rho_h = rho_from_hifi(pred, target)
            band = rho_band(rho_h)
        except Exception as e:
            log(f"  hi-fi FAILED: {e}")
            rho_h, band = float("nan"), "ERR"
        p["rho_hifi"] = float(rho_h); p["band_hifi"] = band
        marker = " (TRUTH-grid)" if p["is_truth_grid"] else ""
        log(f"  HI-FI rank {p['rank']}{marker}  ρ={rho_h:.3f}  band={band}  "
            f"(wall {time.time()-t0:.1f}s)")

    bands = [pp["band_hifi"] for pp in polished if not pp["is_truth_grid"]]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "ERR": 0}
    for b in bands: counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    log(f"\n=== HEADLINE: seed {args.seed} (top-K only, NO oracle) ===")
    log(f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']} "
        f"ERR={counts['ERR']}")
    log(f"  Band A∪B yield: {n_AB}/{len(bands)}")
    log(f"  Truth-grid diagnostic: rank={truth_rank}/{flat_scores.size}, "
        f"ρ_local at grid={truth_rho:.3f}")
    truth_inj = next((p for p in polished if p["is_truth_grid"]), None)
    if truth_inj is not None:
        log(f"  Truth-grid polished: ρ_hifi={truth_inj['rho_hifi']:.3f}  "
            f"band={truth_inj['band_hifi']}  (DIAGNOSTIC, not part of yield)")
    log(f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)")

    summary = {
        "seed": args.seed, "T_A": T_A, "window": W,
        "n_directions": args.n_dir, "n_magnitudes": args.n_mag, "n_omega": n_omega,
        "n_C_a": int(len(C_a)),
        "om_truth_mag_TA": om_truth_mag_TA,
        "truth_a_idx": truth_a_idx,
        "truth_om_grid_idx": om_truth_idx,
        "truth_qa_d_C_a": float(qa_dist_C_a[truth_a_idx]),
        "truth_om_d_grid": float(om_d_grid[om_truth_idx]),
        "truth_om_mag_err_pct_grid": float(om_mag_pct_grid[om_truth_idx]),
        "truth_grid_rho_local": truth_rho, "truth_grid_rank": truth_rank,
        "n_grid_total": int(flat_scores.size),
        "band_counts_top_K": counts, "n_band_AB_top_K": n_AB,
        "wall_total_s": time.time() - t_overall,
        "polished": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in p.items()} for p in polished],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "scores.npz", scores=scores,
                        omega_grid=omega_grid, C_a=C_a)
    log(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
