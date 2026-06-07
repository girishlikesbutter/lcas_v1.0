"""s059g — real-dynamics scoring (replace const-ω with rigid-body Euler).

s059f diagnosed that the const-ω scoring rewards constant-axis trajectories
that graze validator clouds, not dynamics-consistent (q, ω) pairs. Truth
hits 0/8 high-weight (small-|C_v|) validators because its finite-diff ω is
~17% off truth's instantaneous ω, and propagated 10 epochs forward at this
wrong ω lands 30°+ from truth's actual q at those epochs.

This script swaps const-ω for real rigid-body dynamics in the scoring:
each (q_a, ω_inertial) candidate gets ω_body = q_a^* · ω_inertial · q_a,
then `propagate_attitude(mode="tumbling")` integrates Euler's equations
+ quaternion kinematics to validator epochs. The candidate's body-frame
precession is correctly modeled.

Implementation: Pool(N_WORKERS) over candidate chunks. Per-candidate cost
is one ODE solve (rtol=1e-6 — scoring doesn't need 1e-10 like rendering).

If truth's rank lifts substantially under real-dynamics scoring, we know
the architecture's failure was const-ω, not anything else. Then we can
refactor the generator to ω-grid + ψ-factorization for speed.

Usage:
    python experiments/s059g_realdyn_score.py --seed 28 --t-a 312
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

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import (
    fd_omega_passive, propagate_const_omega, quat_ang_deg_batch, ang_to_axis,
    DELTA_GEN, PRIOR_BRACKET, HIT_THRESHOLD_DEG, CONST_OMEGA_RELIABLE_MAX_DEG,
    SMOKE_DELTAS, wxyz_to_xyzw, xyzw_to_wxyz,
)
from lib.hifi_render import build_context
from lib.traj_load import load_truth as _load_traj
from src.dynamics.attitude_propagator import propagate_attitude


N_WORKERS = 8
RTOL_SCORE = 1e-6
ATOL_SCORE = 1e-8


# Worker globals (forked from parent)
_INERTIA = None
_VALIDATORS_DATA = None  # list of (delta_signed, C_vT (4, n_v), weight)
_DT_EPOCH = None
_COS_THRESH = None


def _worker_init(inertia, validators_data, dt_epoch, cos_thresh):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _INERTIA, _VALIDATORS_DATA, _DT_EPOCH, _COS_THRESH
    _INERTIA = inertia
    _VALIDATORS_DATA = validators_data
    _DT_EPOCH = dt_epoch
    _COS_THRESH = cos_thresh


def _score_chunk(args):
    chunk_id, Q_A_chunk, om_inertial_chunk = args
    n = len(Q_A_chunk)
    n_val = len(_VALIDATORS_DATA)
    scores = np.zeros(n)
    hits = np.zeros((n, n_val), dtype=bool)

    # Group validators by sign of delta for two-sided propagation
    fwd_v = [(i, v) for i, v in enumerate(_VALIDATORS_DATA) if v[0] > 0]
    bwd_v = [(i, v) for i, v in enumerate(_VALIDATORS_DATA) if v[0] < 0]
    fwd_dt_sorted = sorted(set(v[1][0] * _DT_EPOCH for v in fwd_v))
    bwd_dt_sorted = sorted(set(abs(v[1][0]) * _DT_EPOCH for v in bwd_v))
    fwd_times = np.concatenate([[0.0], fwd_dt_sorted]) if fwd_dt_sorted else np.array([0.0])
    bwd_times = np.concatenate([[0.0], bwd_dt_sorted]) if bwd_dt_sorted else np.array([0.0])
    fwd_dt_to_idx = {dt: i for i, dt in enumerate(fwd_dt_sorted)}
    bwd_dt_to_idx = {dt: i for i, dt in enumerate(bwd_dt_sorted)}

    for k in range(n):
        q_a = Q_A_chunk[k]
        om_inert = om_inertial_chunk[k]
        # convert ω inertial → body
        R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
        om_body = R_a.inv().apply(om_inert)

        # forward propagation
        if len(fwd_times) > 1:
            try:
                quats_fwd, _ = propagate_attitude(
                    q0=q_a, omega0=om_body, times=fwd_times,
                    mode="tumbling", inertia_tensor=_INERTIA,
                    rtol=RTOL_SCORE, atol=ATOL_SCORE,
                )
            except Exception:
                quats_fwd = None
        else:
            quats_fwd = None
        # backward propagation (forward integrate with -ω over |Δ|)
        if len(bwd_times) > 1:
            try:
                quats_bwd, _ = propagate_attitude(
                    q0=q_a, omega0=-om_body, times=bwd_times,
                    mode="tumbling", inertia_tensor=_INERTIA,
                    rtol=RTOL_SCORE, atol=ATOL_SCORE,
                )
            except Exception:
                quats_bwd = None
        else:
            quats_bwd = None

        for vi, (delta_eps, _, weight) in enumerate(_VALIDATORS_DATA):
            dt_v = abs(delta_eps) * _DT_EPOCH
            if delta_eps > 0:
                if quats_fwd is None: continue
                q_pred = quats_fwd[1 + fwd_dt_to_idx[dt_v]]
            else:
                if quats_bwd is None: continue
                q_pred = quats_bwd[1 + bwd_dt_to_idx[dt_v]]
            C_vT = _VALIDATORS_DATA[vi][1]
            max_dot = np.abs(q_pred @ C_vT).max()
            hit = max_dot > _COS_THRESH
            hits[k, vi] = hit
            if hit:
                scores[k] += weight

    return chunk_id, scores, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--t-a", type=int, required=True)
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059g_seed{args.seed:03d}_T_A{args.t_a:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"=== s059g real-dynamics scoring — seed {args.seed} T_A={args.t_a} ===")
    cloud_dir = SURVEY / "results" / f"s059_seed{args.seed:03d}"
    z = np.load(cloud_dir / "cloud.npz")
    q_pool = z["q_pool_wxyz"]
    survive_all = z["survive_all"]
    obs_times = z["obs_times"]
    n_epochs = len(obs_times)
    dt_epoch = float(np.median(np.diff(obs_times)))

    truth = _load_traj(args.seed)
    q_truth_t = truth["quaternions"]
    T_A = args.t_a

    om_truth = fd_omega_passive(
        q_truth_t[T_A:T_A+1], q_truth_t[T_A+1:T_A+2], dt_epoch)[0]
    om_truth_mag = float(np.linalg.norm(om_truth))
    log(f"truth |ω| at T_A (Δ=1 estimate): {om_truth_mag:.6f} rad/s")

    ctx = build_context(seed=args.seed)
    inertia = ctx["inertia_tensor"]
    log(f"inertia eigenvals: {np.linalg.eigvalsh(inertia)}")

    # validators (same logic as s059_pilot)
    valid_max_delta = 1
    for d in SMOKE_DELTAS:
        if T_A + d >= n_epochs: continue
        from experiments.s059_pilot import quat_ang_deg as _q
        q_pred = propagate_const_omega(q_truth_t[T_A], om_truth, d * dt_epoch)
        err = _q(q_pred, q_truth_t[T_A + d])
        if err <= CONST_OMEGA_RELIABLE_MAX_DEG:
            valid_max_delta = d
        else:
            break
    log(f"const-ω reliable to Δ = {valid_max_delta}")

    validators_data = []
    lo_v = max(0, T_A - valid_max_delta)
    hi_v = min(n_epochs, T_A + valid_max_delta + 1)
    for t_v in range(lo_v, hi_v):
        if t_v == T_A: continue
        idx_v = np.where(survive_all[t_v])[0]
        if len(idx_v) == 0: continue
        validators_data.append((
            t_v - T_A, q_pool[idx_v].T, float(np.log(100000.0 / len(idx_v))),
        ))
    log(f"{len(validators_data)} validators")

    # candidate generation (same as s059)
    Δgen_t_b = T_A + DELTA_GEN
    if Δgen_t_b >= n_epochs:
        Δgen_t_b = T_A - DELTA_GEN
        Δt_gen = -DELTA_GEN * dt_epoch
    else:
        Δt_gen = DELTA_GEN * dt_epoch
    idx_a = np.where(survive_all[T_A])[0]
    idx_b = np.where(survive_all[Δgen_t_b])[0]
    C_a = q_pool[idx_a]
    C_b = q_pool[idx_b]
    n_a, n_b = len(C_a), len(C_b)
    Q_A = np.repeat(C_a, n_b, axis=0)
    Q_B = np.tile(C_b, (n_a, 1))
    om_all = fd_omega_passive(Q_A, Q_B, Δt_gen)
    om_mag_all = np.linalg.norm(om_all, axis=1)
    mask = (om_mag_all >= om_truth_mag * PRIOR_BRACKET[0]) & \
           (om_mag_all <= om_truth_mag * PRIOR_BRACKET[1])
    Q_A_pass = Q_A[mask]
    om_pass = om_all[mask]
    n_cand = len(Q_A_pass)
    log(f"candidates: {n_cand}")

    qa_dist = quat_ang_deg_batch(Q_A_pass, q_truth_t[T_A])
    om_dist = ang_to_axis(om_pass, om_truth)
    om_mag_pct = (om_mag_all[mask] - om_truth_mag) / om_truth_mag * 100

    cos_thresh = float(np.cos(np.radians(HIT_THRESHOLD_DEG / 2.0)))

    # Pool(8) — split candidates into chunks
    chunk_size = (n_cand + N_WORKERS - 1) // N_WORKERS
    chunks = [(i, Q_A_pass[i*chunk_size:(i+1)*chunk_size],
               om_pass[i*chunk_size:(i+1)*chunk_size])
              for i in range(N_WORKERS)
              if i*chunk_size < n_cand]
    log(f"scoring with Pool({N_WORKERS}), chunk size ≈ {chunk_size}...")

    t_score = time.time()
    scores = np.zeros(n_cand)
    hits = np.zeros((n_cand, len(validators_data)), dtype=bool)
    with get_context("fork").Pool(
        N_WORKERS, initializer=_worker_init,
        initargs=(inertia, validators_data, dt_epoch, cos_thresh),
    ) as pool:
        for chunk_id, sc, h in pool.imap_unordered(_score_chunk, chunks):
            s = chunk_id * chunk_size
            e = s + len(sc)
            scores[s:e] = sc
            hits[s:e] = h
    log(f"  scoring wall: {time.time()-t_score:.1f}s")
    log(f"  score range [{scores.min():.2f}, {scores.max():.2f}]")

    # truth rank
    truth_idx = int(np.argmin(qa_dist + om_dist))
    truth_score = float(scores[truth_idx])
    truth_rank = int((scores > truth_score).sum()) + 1
    truth_n_hits = int(hits[truth_idx].sum())
    log(f"truth-candidate idx={truth_idx}: qa_d={qa_dist[truth_idx]:.2f}°, "
        f"ω_d={om_dist[truth_idx]:.2f}°, |ω|Δ={om_mag_pct[truth_idx]:+.2f}%")
    log(f"  REAL-DYN score: {truth_score:.2f}, rank {truth_rank}/{n_cand}, "
        f"hits {truth_n_hits}/{len(validators_data)}")

    # Top 20
    top20 = np.argsort(-scores)[:20]
    log("\n=== TOP 20 by REAL-DYN score ===")
    log(f"{'rank':<5}{'idx':<10}{'qa_d°':<10}{'om_d°':<10}{'|ω|Δ%':<10}{'score':<10}{'n_hits':<8}")
    for r, idx in enumerate(top20):
        is_truth = " (TRUTH)" if idx == truth_idx else ""
        log(f"{r+1:<5}{idx:<10}{qa_dist[idx]:<10.2f}{om_dist[idx]:<10.2f}"
            f"{om_mag_pct[idx]:<10.2f}{scores[idx]:<10.2f}"
            f"{int(hits[idx].sum()):<8}{is_truth}")

    summary = {
        "seed": args.seed, "T_A": T_A, "n_candidates": int(n_cand),
        "n_validators": len(validators_data),
        "score_range": [float(scores.min()), float(scores.max())],
        "truth_score": truth_score, "truth_rank": truth_rank,
        "truth_n_validators_hit": truth_n_hits,
        "truth_qa_d": float(qa_dist[truth_idx]),
        "truth_om_d": float(om_dist[truth_idx]),
        "truth_om_mag_err_pct": float(om_mag_pct[truth_idx]),
        "top20_idx": top20.tolist(),
        "top20_qa_d": qa_dist[top20].tolist(),
        "top20_om_d": om_dist[top20].tolist(),
        "top20_score": scores[top20].tolist(),
        "wall_score_s": time.time() - t_score,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "scores.npz",
                        scores=scores, hits=hits, qa_dist=qa_dist,
                        om_dist=om_dist, om_mag_pct=om_mag_pct)
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    log(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
