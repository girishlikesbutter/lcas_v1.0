"""s059 — single-seed pilot of the cloud-data → polish → ρ-band pipeline.

Combines the s057g forward-prop discrimination, s057h canonicalisation +
clustering, s058 LM-polish-on-surrogate, and the new surrogate-ρ gate
before hi-fi classification (per `feedback_surrogate_first_hifi_last.md`).

Pipeline at a glance:

    1. Build q_pool (100k Sobol on SO(3))  — one-time, reused across seeds
    2. Cloud generation: surrogate-v1 predict + survival mask per epoch
    3. Auto-pick T_A = argmin |C_t| (excluding the first/last 5 epochs)
    4. Forward-prop scoring (s057g): const-ω propagate q_a candidates
       to validator epochs in [T_A - max_delta, T_A + max_delta]
    5. Canonicalise + greedy cluster (s057h)
    6. Back-propagate top-K cluster reps + truth cluster to t=0
    7. LM polish on surrogate-v2 cost (max_nfev=200), same as s058
    8. Gate: keep polished candidates with surrogate_ρ < 4
    9. Hi-fi render gated subset (single render each)
   10. ρ-band classify, save NPZ + JSON + figure

Usage:
    python experiments/s059_pilot.py --seed 28
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# NOTE: do NOT set OMP/OPENBLAS/MKL_NUM_THREADS=1 here. This script is a
# single-process sequential pipeline (no Pool), so multi-threaded BLAS
# parallelises the surrogate matmul + finite-diff Jacobian across all
# cores. The single-thread setting only matters when spawning Pool(N)
# workers (see feedback_blas_threads_for_pool.md) — that's not the case
# here. Forcing single-thread BLAS turns 8-core into a 1-core run and
# was the bottleneck during the first launch.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from multiprocessing import get_context

from lib.c_t_pipeline import (
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
)
from lib.twin import canonical_batch
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import predict as surrogate_predict
from lib.traj_load import load_truth as _load_traj
from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1
from src.dynamics.attitude_propagator import propagate_attitude

# ----- parameters ------------------------------------------------------------

N_SAMPLES = 100_000
TOL_MAG = 0.10                  # cloud-survival tolerance (s048c default)
SP_DEG = 0.0
AD_DEG = 15.0
DELTA_GEN = 15                  # s057g default
PRIOR_BRACKET = (0.75, 1.25)    # s057g default ±25% |ω|-prior
HIT_THRESHOLD_DEG = 5.0
CONST_OMEGA_RELIABLE_MAX_DEG = 20.0
SMOKE_DELTAS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
CAND_CHUNK = 10_000               # cap per-validator dots matrix at chunk × n_surv
CLUSTER_Q_DEG = 8.0
CLUSTER_OM_DEG = 15.0
CLUSTER_OM_MAG_PCT = 25.0
TOP_K_CLUSTERS = 5              # number of cluster reps to polish (+ truth)
LM_MAX_NFEV = 200
LM_FTOL = LM_XTOL = 1e-8
SURROGATE_RHO_HIFI_GATE = 4.0   # only hi-fi candidates with surrogate ρ < this
RESIDUAL_CAP = 5.0
ANCHOR_EXCLUDE_EDGE = 5         # don't pick T_A in the first/last 5 epochs

V1_DIR = Path("/home/girish/surrogate_model/surrogate_model")
N_WORKERS = 8                   # cloud-gen Pool size; BLAS=1 in workers

# ----- helpers ---------------------------------------------------------------


def wxyz_to_xyzw(q): return q[..., [1, 2, 3, 0]]
def xyzw_to_wxyz(q): return q[..., [3, 0, 1, 2]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def propagate_const_omega(q_a, omega_rad_s, dt):
    rotvec = omega_rad_s * dt
    R_om = Rotation.from_rotvec(rotvec)
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    return xyzw_to_wxyz((R_om * R_a).as_quat())


def quat_ang_deg(q1, q2):
    return float(2 * np.degrees(np.arccos(
        np.clip(abs(np.dot(q1, q2)), 0, 1))))


def quat_ang_deg_batch(q_arr, q_ref):
    dots = np.abs(q_arr @ q_ref)
    return 2 * np.degrees(np.arccos(np.clip(dots, 0, 1)))


def ang_to_axis(om_arr, ref):
    om_norm = np.linalg.norm(om_arr, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_arr / safe
    r_hat = ref / np.linalg.norm(ref)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, r_hat))
    return np.degrees(np.arccos(np.clip(cos_a, 0, 1)))


def back_propagate(q_a_wxyz, omega_a_rad, t_back_s, inertia_tensor):
    """Back-propagate (q_a, ω_a) at t_back to (q_0, ω_0) at t=0 via
    time-reversal symmetry (s057i:96-110).
    """
    times = np.array([0.0, t_back_s])
    quats, omegas = propagate_attitude(
        q0=q_a_wxyz, omega0=-omega_a_rad, times=times,
        mode="tumbling", inertia_tensor=inertia_tensor,
    )
    return quats[-1], -omegas[-1]


def make_residual_fn(q0_seed_wxyz, ctx, target):
    q0_seed_xyzw = wxyz_to_xyzw(q0_seed_wxyz)
    R_seed = Rotation.from_quat(q0_seed_xyzw)

    def residual(params):
        rotvec = params[:3]
        omega0 = params[3:]
        q_xyzw = (Rotation.from_rotvec(rotvec) * R_seed).as_quat()
        q_wxyz = xyzw_to_wxyz(q_xyzw)
        try:
            k1, k2, _ = propagate_to_body_frame(
                q0_wxyz=q_wxyz, omega0_rad=omega0,
                observation_times=ctx["observation_times"],
                sun_pos=ctx["sun_pos"], obs_pos=ctx["obs_pos"],
                sat_pos=ctx["sat_pos"], inertia_tensor=ctx["inertia_tensor"],
                mode="tumbling",
            )
            pred = surrogate_predict(k1, k2, ctx["obs_dist"])
            r = pred - target
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full_like(target, RESIDUAL_CAP)

    return residual


def lm_polish(q0_seed_wxyz, om0_seed_rad, ctx, target):
    residual = make_residual_fn(q0_seed_wxyz, ctx, target)
    x0 = np.concatenate([np.zeros(3), np.asarray(om0_seed_rad)])
    r_seed = residual(x0)
    mse_seed = float(np.mean(r_seed ** 2))
    t0 = time.time()
    result = least_squares(
        residual, x0, method="lm",
        max_nfev=LM_MAX_NFEV, ftol=LM_FTOL, xtol=LM_XTOL,
    )
    wall_s = time.time() - t0
    rotvec_pol = result.x[:3]
    om0_pol = result.x[3:]
    q_xyzw_seed = wxyz_to_xyzw(q0_seed_wxyz)
    q_pol_xyzw = (Rotation.from_rotvec(rotvec_pol) *
                  Rotation.from_quat(q_xyzw_seed)).as_quat()
    q0_pol_wxyz = xyzw_to_wxyz(q_pol_xyzw)
    mse_pol = float(np.mean(result.fun ** 2))
    return {
        "q0_pol_wxyz": q0_pol_wxyz,
        "om0_pol_rad": om0_pol,
        "rotvec_pol_mag_deg": float(np.degrees(np.linalg.norm(rotvec_pol))),
        "om_change_pct": float(
            np.linalg.norm(om0_pol - np.asarray(om0_seed_rad)) /
            max(1e-12, np.linalg.norm(om0_seed_rad)) * 100),
        "surrogate_mse_seed": mse_seed,
        "surrogate_mse_polished": mse_pol,
        "surrogate_rho_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(result.nfev),
        "wall_s": wall_s,
    }


# ----- Pool workers for cloud generation -------------------------------------
#
# Strategy: parent loads the heavy state (R_cache, per-epoch SPICE units,
# truth mag, surrogate weights) into module-level globals BEFORE spawning the
# Pool. On Linux, the default `fork` start-method gives every worker a CoW
# view of the parent's memory, so we pay zero pickle/transfer cost for the
# 100k-sample R_cache.
#
# Inside each worker we then PIN BLAS to 1 thread per process (8 workers ×
# 1 thread = 8 cores actively working). With BLAS=multithreaded in workers
# we'd get 8 × 32 = 256-thread oversubscription on a 32-core box and the
# thread-pool contention destroys throughput.

_R_CACHE = None      # (N, 3, 3) body→inertial rotation matrices
_SUN_UNIT = None     # (n_obs, 3) per-epoch sun unit vector in J2000
_OBS_UNIT = None     # (n_obs, 3) per-epoch observer unit vector in J2000
_OBS_DIST = None     # (n_obs,) observer distance in km
_MAG_TARGET = None   # (n_obs,) measured (truth) magnitudes
_SURROGATE = None    # surrogate v1 model instance


def _worker_init(weights_path: str, norm_path: str):
    """Run ONCE per Pool worker on startup. Pins BLAS to 1 thread and
    instantiates the surrogate model in the worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURROGATE
    from surrogate_model.surrogate_v1 import SurrogateModel as _V1
    _SURROGATE = _V1(weights_path, norm_path)


def _worker_cloud_chunk(epoch_indices):
    """Process a contiguous chunk of epochs in this worker. Reads
    R_cache / sun_unit / obs_unit / obs_dist / mag_target from the
    parent-inherited module globals (CoW on Linux fork)."""
    n_pool = _R_CACHE.shape[0]
    out = np.empty((len(epoch_indices), n_pool), dtype=bool)
    obs_dist_arr = np.empty(n_pool, dtype=np.float64)
    for j, ep in enumerate(epoch_indices):
        k1 = np.einsum("nij,j->ni", _R_CACHE, _SUN_UNIT[ep])
        k2 = np.einsum("nij,j->ni", _R_CACHE, _OBS_UNIT[ep])
        obs_dist_arr.fill(_OBS_DIST[ep])
        pred = _SURROGATE.predict_magnitude(
            k1, k2, SP_DEG, AD_DEG, obs_dist_arr
        )
        out[j] = np.abs(pred - _MAG_TARGET[ep]) < TOL_MAG
    return epoch_indices, out


# ----- pipeline stages -------------------------------------------------------


def stage_cloud_generation(seed, out_dir, log):
    """Generate or load 100k × 500 cloud. Pool(N_WORKERS) over epoch chunks."""
    cache = out_dir / "cloud.npz"
    if cache.exists():
        log(f"cloud cache hit: {cache.name}")
        z = np.load(cache)
        # NOTE: don't materialise R_cache (7.2 GB) on the cache-hit path —
        # forward_prop only needs q_pool_wxyz + survive_all.
        return {
            "q_pool_wxyz": z["q_pool_wxyz"],
            "q_pool_xyzw": z["q_pool_wxyz"][:, [1, 2, 3, 0]],
            "survive_all": z["survive_all"],
            "obs_times": z["obs_times"],
            "wall_s_total": float(z["wall_s_total"]),
        }

    t0 = time.time()
    log(f"loading Sobol pool + truth NPZ...")
    pool = sample_so3_pool(N_SAMPLES, sample_seed=42)
    truth = _load_traj(seed)
    sun_unit, obs_unit = compute_j2000_units(
        truth["sun_pos"], truth["obs_pos"], truth["sat_pos"])
    n_obs = len(truth["observation_times"])

    # Stash heavy state into module globals so forked workers inherit via CoW.
    global _R_CACHE, _SUN_UNIT, _OBS_UNIT, _OBS_DIST, _MAG_TARGET
    _R_CACHE = pool["R_cache"]
    _SUN_UNIT = sun_unit
    _OBS_UNIT = obs_unit
    _OBS_DIST = truth["obs_dist"]
    _MAG_TARGET = truth["mag_hifi"]

    weights_path = str(V1_DIR / "s10_5M_weights.npz")
    norm_path = str(V1_DIR / "s10_5M_normalization.npz")

    # Slice 0..n_obs into N_WORKERS contiguous chunks.
    epoch_chunks = np.array_split(np.arange(n_obs), N_WORKERS)
    log(f"cloud generation: {n_obs} epochs × {N_SAMPLES} samples, "
        f"Pool({N_WORKERS}) over {len(epoch_chunks)} chunks "
        f"(chunk size ≈ {len(epoch_chunks[0])} epochs each)...")

    survive_all = np.empty((n_obs, N_SAMPLES), dtype=bool)
    ctx = get_context("fork")
    t_gen = time.time()
    with ctx.Pool(
        N_WORKERS,
        initializer=_worker_init,
        initargs=(weights_path, norm_path),
    ) as p:
        for chunk_indices, chunk_out in p.imap_unordered(
            _worker_cloud_chunk, epoch_chunks
        ):
            survive_all[chunk_indices] = chunk_out
    wall_gen = time.time() - t_gen
    cv = survive_all.sum(axis=1)
    log(f"  done in {wall_gen:.1f}s; |C_t| min={cv.min()} "
        f"median={int(np.median(cv))} max={cv.max()}")
    wall_total = time.time() - t0
    np.savez_compressed(
        cache,
        q_pool_wxyz=pool["q_pool_wxyz"],
        survive_all=survive_all,
        obs_times=truth["observation_times"],
        wall_s_total=wall_total,
    )
    log(f"  cached: {cache}")
    return {
        "q_pool_wxyz": pool["q_pool_wxyz"],
        "q_pool_xyzw": pool["q_pool_xyzw"],
        "survive_all": survive_all,
        "obs_times": truth["observation_times"],
        "wall_s_total": wall_total,
    }


def stage_pick_anchor(survive_all, log, edge=ANCHOR_EXCLUDE_EDGE):
    cv = survive_all.sum(axis=1)
    cv_masked = cv.copy()
    cv_masked[:edge] = 1 << 30
    cv_masked[-edge:] = 1 << 30
    T_A = int(np.argmin(cv_masked))
    log(f"anchor T_A = {T_A}, |C_{{T_A}}| = {cv[T_A]} "
        f"(cohort min |C_t| = {cv[edge:-edge].min()})")
    return T_A


def stage_forward_prop(seed, cloud, T_A, ctx, log):
    """s057g: generate (q_a, ω) candidates, score by forward-prop hits."""
    survive_all = cloud["survive_all"]
    q_pool = cloud["q_pool_wxyz"]
    obs_times = cloud["obs_times"]
    dt_epoch = float(np.median(np.diff(obs_times)))
    n_epochs = len(survive_all)

    # smoke: const-ω validity range using truth (if known) — diagnostic only,
    # uses ω derived from truth-q at consecutive epochs
    q_truth_t = _load_traj(seed)["quaternions"]
    om_inst = fd_omega_passive(
        q_truth_t[T_A:T_A+1], q_truth_t[T_A+1:T_A+2], dt_epoch)[0]
    valid_max_delta = 1
    for d in SMOKE_DELTAS:
        if T_A + d >= n_epochs: continue
        q_pred = propagate_const_omega(q_truth_t[T_A], om_inst, d * dt_epoch)
        err = quat_ang_deg(q_pred, q_truth_t[T_A + d])
        if err <= CONST_OMEGA_RELIABLE_MAX_DEG:
            valid_max_delta = d
        else:
            break
    log(f"const-ω reliable to Δ = {valid_max_delta} epochs ({valid_max_delta*dt_epoch:.1f}s)")

    validators = []
    lo_v = max(0, T_A - valid_max_delta)
    hi_v = min(n_epochs, T_A + valid_max_delta + 1)
    for t_v in range(lo_v, hi_v):
        if t_v == T_A: continue
        idx_v = np.where(survive_all[t_v])[0]
        if len(idx_v) == 0: continue
        validators.append({
            "delta": t_v - T_A, "C_v": q_pool[idx_v],
            "n_surv": int(len(idx_v)),
            "weight": float(np.log(100000.0 / len(idx_v))),
        })
    log(f"{len(validators)} validators, |C_v| range "
        f"[{min(v['n_surv'] for v in validators)}, {max(v['n_surv'] for v in validators)}]")

    # candidate generation
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

    om_truth = fd_omega_passive(
        q_truth_t[T_A:T_A+1], q_truth_t[Δgen_t_b:Δgen_t_b+1], Δt_gen)[0]
    om_truth_mag = float(np.linalg.norm(om_truth))

    mask = (om_mag_all >= om_truth_mag * PRIOR_BRACKET[0]) & \
           (om_mag_all <= om_truth_mag * PRIOR_BRACKET[1])
    Q_A_pass = Q_A[mask]
    om_pass = om_all[mask]
    n_cand = len(Q_A_pass)
    log(f"{n_a} × {n_b} = {n_a*n_b} pairs → {n_cand} candidates after |ω|-prior ±25%")

    scores = np.zeros(n_cand)
    cos_thresh = float(np.cos(np.radians(HIT_THRESHOLD_DEG / 2.0)))
    for v in validators:
        d = v["delta"]
        Δt = d * dt_epoch
        C_vT = v["C_v"].T
        weight = v["weight"]
        for s in range(0, n_cand, CAND_CHUNK):
            e = min(s + CAND_CHUNK, n_cand)
            q_pred_chunk = propagate_const_omega(
                Q_A_pass[s:e], om_pass[s:e], Δt)
            # max |dot| per row without materialising full (chunk × n_surv) ang grid
            max_dot = np.abs(q_pred_chunk @ C_vT).max(axis=1)
            scores[s:e] += weight * (max_dot > cos_thresh)

    # null expectation
    cap_frac = 1.0 - np.cos(np.radians(HIT_THRESHOLD_DEG))
    null_score = sum(v["weight"] * min(cap_frac * v["n_surv"] / 100000.0, 1.0)
                     for v in validators)
    log(f"score range [{scores.min():.2f}, {scores.max():.2f}], "
        f"null E[score] = {null_score:.4f}, "
        f"discrimination = {scores.max() / max(null_score, 1e-6):.0f}×")

    qa_dist = quat_ang_deg_batch(Q_A_pass, q_truth_t[T_A])
    om_dist = ang_to_axis(om_pass, om_truth)
    truth_idx = int(np.argmin(qa_dist + om_dist))
    rank_truth = int((scores > scores[truth_idx]).sum()) + 1
    log(f"truth-candidate rank {rank_truth}/{n_cand} "
        f"(qa_d={qa_dist[truth_idx]:.2f}°, ω_d={om_dist[truth_idx]:.2f}°, "
        f"score={scores[truth_idx]:.2f})")

    return {
        "Q_A_pass": Q_A_pass, "om_pass": om_pass, "scores": scores,
        "qa_dist_to_truth": qa_dist, "om_dist_to_truth": om_dist,
        "truth_idx": truth_idx, "om_truth": om_truth, "om_truth_mag": om_truth_mag,
        "T_A": T_A, "Δgen_t_b": Δgen_t_b, "Δt_gen": Δt_gen,
        "validators_n": len(validators), "null_score": null_score,
    }


def stage_cluster(fp, log):
    """s057h: canonicalise + greedy-cluster by score."""
    Q_A_pass = fp["Q_A_pass"]
    om_pass = fp["om_pass"]
    scores = fp["scores"]
    n_cand = len(Q_A_pass)

    Q_A_canon, om_canon = canonical_batch(Q_A_pass, om_pass)
    om_mag_canon = np.linalg.norm(om_canon, axis=1)
    order = np.argsort(-scores)
    assigned = np.full(n_cand, -1, dtype=int)
    clusters = []
    for seed_i in order:
        if assigned[seed_i] != -1: continue
        unassigned = np.where(assigned == -1)[0]
        d_q = quat_ang_deg_batch(Q_A_canon[unassigned], Q_A_canon[seed_i])
        d_om = ang_to_axis(om_canon[unassigned], om_canon[seed_i])
        d_mag = np.abs(om_mag_canon[unassigned] - om_mag_canon[seed_i]) / \
                om_mag_canon[seed_i] * 100
        in_clu = (d_q < CLUSTER_Q_DEG) & (d_om < CLUSTER_OM_DEG) & \
                 (d_mag < CLUSTER_OM_MAG_PCT)
        members = unassigned[in_clu]
        cluster_id = len(clusters)
        assigned[members] = cluster_id
        best_member = int(members[np.argmax(scores[members])])
        clusters.append({
            "cluster_id": cluster_id,
            "n_members": int(len(members)),
            "best_member_idx": best_member,
            "score_sum": float(scores[members].sum()),
            "score_max": float(scores[members].max()),
            "min_qa_dist_to_truth": float(np.min(fp["qa_dist_to_truth"][members])),
            "min_om_dist_to_truth": float(np.min(fp["om_dist_to_truth"][members])),
        })
    clusters_sorted = sorted(clusters, key=lambda c: -c["score_sum"])
    truth_cluster_id = int(assigned[fp["truth_idx"]])
    truth_rank = next(i for i, c in enumerate(clusters_sorted)
                      if c["cluster_id"] == truth_cluster_id) + 1
    log(f"{n_cand} candidates → {len(clusters)} clusters; "
        f"truth cluster rank {truth_rank}/{len(clusters)}")
    return {
        "clusters": clusters,
        "clusters_sorted": clusters_sorted,
        "truth_cluster_id": truth_cluster_id,
        "truth_cluster_rank": truth_rank,
        "assigned": assigned,
    }


def stage_polish_and_classify(seed, fp, cl, ctx, log):
    """s058: back-prop top-K + truth, LM polish, surrogate-ρ gate, hi-fi."""
    target = ctx["mag_hifi_truth"]
    obs_times = ctx["observation_times"]
    t_a_seconds = float(obs_times[fp["T_A"]] - obs_times[0])
    log(f"T_A = {fp['T_A']}, t_a_seconds = {t_a_seconds:.1f}")

    to_polish = list(cl["clusters_sorted"][:TOP_K_CLUSTERS])
    truth_cl = next(c for c in cl["clusters"]
                    if c["cluster_id"] == cl["truth_cluster_id"])
    if truth_cl not in to_polish:
        to_polish.append(truth_cl)
    log(f"polishing {len(to_polish)} clusters (top {TOP_K_CLUSTERS} + truth)")

    polished = []
    for c in to_polish:
        rank = next(i for i, cc in enumerate(cl["clusters_sorted"])
                    if cc["cluster_id"] == c["cluster_id"]) + 1
        is_truth = c["cluster_id"] == cl["truth_cluster_id"]
        bm = c["best_member_idx"]
        q_a = fp["Q_A_pass"][bm]
        om_a = fp["om_pass"][bm]
        try:
            q0_back, om0_back = back_propagate(
                q_a, om_a, t_a_seconds, ctx["inertia_tensor"])
        except Exception as e:
            log(f"  cluster_id={c['cluster_id']} back-prop FAILED: {e}")
            continue
        result = lm_polish(q0_back, om0_back, ctx, target)
        result["cluster_rank"] = rank
        result["cluster_id"] = c["cluster_id"]
        result["is_truth_cluster"] = is_truth
        result["score_sum"] = c["score_sum"]
        result["qa_dist_t_a"] = c["min_qa_dist_to_truth"]
        result["om_dist_t_a"] = c["min_om_dist_to_truth"]
        result["q0_back_wxyz"] = q0_back
        result["om0_back_rad"] = om0_back

        # errors vs truth
        result["q0_err_polished_deg"] = quat_ang_deg(
            result["q0_pol_wxyz"], ctx["q0_truth"])
        om_truth = ctx["omega0_truth_rad"]
        om_pol = result["om0_pol_rad"]
        om_truth_mag = float(np.linalg.norm(om_truth))
        result["om_mag_err_pct"] = float(
            (np.linalg.norm(om_pol) - om_truth_mag) / om_truth_mag * 100)
        result["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om_pol / np.linalg.norm(om_pol),
                       om_truth / om_truth_mag)), 0, 1))))

        marker = " (TRUTH)" if is_truth else ""
        log(f"  rank {rank:3d}/{len(cl['clusters'])}  cluster_id={c['cluster_id']:3d}{marker}  "
            f"surrogate ρ_seed={result['surrogate_rho_seed']:6.2f} → "
            f"ρ_polished={result['surrogate_rho_polished']:6.3f}  "
            f"(rotvec={result['rotvec_pol_mag_deg']:6.2f}°, "
            f"|ω|Δ={result['om_change_pct']:+.2f}%, n_eval={result['n_eval']:3d}, "
            f"wall={result['wall_s']:.1f}s)")
        polished.append(result)

    # surrogate-ρ gate before hi-fi
    log("\napplying surrogate-ρ < {} gate before hi-fi...".format(
        SURROGATE_RHO_HIFI_GATE))
    n_pass_gate = sum(1 for p in polished
                      if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE)
    log(f"  {n_pass_gate}/{len(polished)} candidates pass; hi-fi-rendering only those")

    for p in polished:
        if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE:
            t0 = time.time()
            try:
                pred = render_hifi(p["q0_pol_wxyz"], p["om0_pol_rad"], ctx)
                rho_h = rho_from_hifi(pred, target)
                band = rho_band(rho_h)
            except Exception as e:
                log(f"  cluster_id={p['cluster_id']}: hi-fi FAILED: {e}")
                rho_h, band = float("nan"), "ERR"
                pred = np.full_like(target, np.nan)
            p["pred_hifi"] = pred
            p["rho_polished_hifi"] = float(rho_h)
            p["band_polished_hifi"] = band
            p["hifi_render_s"] = time.time() - t0
            marker = " (TRUTH)" if p["is_truth_cluster"] else ""
            log(f"  HI-FI cluster_id={p['cluster_id']:3d}{marker}  "
                f"surrogate ρ={p['surrogate_rho_polished']:.3f} → "
                f"hi-fi ρ={rho_h:.3f}  band={band}  "
                f"(q0_err={p['q0_err_polished_deg']:.2f}°, "
                f"|ω|err={p['om_mag_err_pct']:+.3f}%, "
                f"ω_dir_err={p['om_dir_err_deg']:.2f}°)")
        else:
            p["pred_hifi"] = None
            p["rho_polished_hifi"] = float("nan")
            p["band_polished_hifi"] = "GATED"
            p["hifi_render_s"] = 0.0

    return polished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059_seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059 pilot — seed {args.seed} ===")

    log("\n[1/4] cloud generation")
    cloud = stage_cloud_generation(args.seed, out_dir, log)

    # release heavy state no longer needed downstream (~7.2 GB R_cache plus
    # SPICE arrays kept alive for the cloud-gen Pool workers via fork CoW).
    global _R_CACHE, _SUN_UNIT, _OBS_UNIT, _OBS_DIST, _MAG_TARGET, _SURROGATE
    _R_CACHE = _SUN_UNIT = _OBS_UNIT = _OBS_DIST = _MAG_TARGET = _SURROGATE = None
    import gc
    gc.collect()

    log("\n[2/4] anchor + forward-prop scoring")
    ctx = build_context(seed=args.seed)
    T_A = stage_pick_anchor(cloud["survive_all"], log)
    fp = stage_forward_prop(args.seed, cloud, T_A, ctx, log)

    log("\n[3/4] canonicalise + cluster")
    cl = stage_cluster(fp, log)

    log("\n[4/4] back-prop + LM polish + surrogate-ρ gate + hi-fi classify")
    polished = stage_polish_and_classify(args.seed, fp, cl, ctx, log)

    # aggregate
    bands = [p["band_polished_hifi"] for p in polished]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in bands:
        counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    log(f"\n=== HEADLINE: seed {args.seed} ===")
    log(f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']} "
        f"GATED={counts['GATED']} ERR={counts['ERR']}")
    log(f"  Band A∪B yield: {n_AB}/{len(polished)} polished candidates")
    log(f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)")

    # save NPZ
    np.savez_compressed(
        out_dir / "polished_states.npz",
        q0_pol_wxyz=np.array([p["q0_pol_wxyz"] for p in polished]),
        om0_pol_rad=np.array([p["om0_pol_rad"] for p in polished]),
        cluster_rank=np.array([p["cluster_rank"] for p in polished]),
        cluster_id=np.array([p["cluster_id"] for p in polished]),
        is_truth=np.array([p["is_truth_cluster"] for p in polished]),
        surrogate_rho_polished=np.array(
            [p["surrogate_rho_polished"] for p in polished]),
        rho_polished_hifi=np.array(
            [p["rho_polished_hifi"] for p in polished]),
        band=np.array([p["band_polished_hifi"] for p in polished]),
        q0_err_deg=np.array([p["q0_err_polished_deg"] for p in polished]),
        om_mag_err_pct=np.array([p["om_mag_err_pct"] for p in polished]),
        om_dir_err_deg=np.array([p["om_dir_err_deg"] for p in polished]),
        pred_hifi=np.array([p["pred_hifi"] if p["pred_hifi"] is not None
                            else np.full_like(ctx["mag_hifi_truth"], np.nan)
                            for p in polished]),
    )
    summary = {
        "seed": args.seed,
        "T_A": T_A,
        "n_candidates": int(len(fp["Q_A_pass"])),
        "n_clusters": int(len(cl["clusters"])),
        "truth_cluster_rank": cl["truth_cluster_rank"],
        "discrimination_ratio": float(
            fp["scores"].max() / max(fp["null_score"], 1e-6)),
        "band_counts": counts,
        "n_band_AB": n_AB,
        "n_polished": len(polished),
        "n_passed_surrogate_gate": sum(
            1 for p in polished if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE),
        "wall_total_s": time.time() - t_overall,
        "polished": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in p.items() if k != "pred_hifi"}
            for p in polished
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved: {out_dir / 'summary.json'}")
    log(f"Saved: {out_dir / 'polished_states.npz'}")


if __name__ == "__main__":
    main()
