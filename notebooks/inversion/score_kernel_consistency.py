#!/usr/bin/env python3
"""Multi-epoch surrogate-consistency cost on m118-style kernel factorisation.

Replaces the full-LC surrogate MSE in score_constrained_anchor.py with a
K-epoch consistency cost. Same anchor-selection step (Step 1), but the
ranking step (Step 2) propagates only at constraint epochs and scores
surrogate magnitude residuals there — no full-LC propagation per candidate.

Algorithm:
  1. Anchor selection (Step 1, unchanged): per epoch in middle 30% of LC,
     sample 64k q's broadly on SO(3) (PAB-body fibonacci × phi azimuth),
     count how many produce surrogate magnitude within ±tol of observed.
     Pick smallest count subject to floor. NO BRIGHT FILTER — established
     analysis: bright = wide cone = LOW info; mid-brightness is constraining.

  2. Constraint epoch set: spec_peaks ∪ (uniform sample of K_DENSE epochs
     where observed_lc < 11 in middle 60% of LC). No IPL dependency.

  3. Build kernel q_delta[N_DIRS=2000, N_MAGS=20, K_EPS, 4]: propagate
     identity quaternion under each (dir, mag) over the constraint times,
     parallel over directions. Pattern from m118_kernel.py / kernel-
     factorization.md. Cheap (~100s on Pool(8)).

  4. Score: for each (dir, mag, q_anchor in Q_a), q_world[ep] = q_anchor ·
     q_delta[dir, mag, ep]. Surrogate mag at each epoch; MSE over K_EPS.
     Per (dir, mag): min over anchor → cost. Per dir: min over mag → cost.

  5. Rank top-K by cost; report ω-direction error vs truth (NO oracle).

Defaults sized for the seed-91 dry run before the failure-seed battery.
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402
from surrogate_model.surrogate import SurrogateModel  # noqa: E402
from surrogate_model import surrogate_v1 as _v1  # noqa: E402

DIAG = PROJECT_ROOT.parent / "data" / "results" / "inversion_diagnostics"
_V1_PKG_DIR = Path("/home/girish/surrogate_model/surrogate_model")


def _load_v1_surrogate():
    return _v1.SurrogateModel(str(_V1_PKG_DIR / "s10_5M_weights.npz"),
                              str(_V1_PKG_DIR / "s10_5M_normalization.npz"))


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])


def quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000):
    N = pab_body_dirs.shape[0]
    P = phi_arr.shape[0]
    out = np.zeros((N * P, 4))
    pj = pab_j2000 / np.linalg.norm(pab_j2000)
    for i, b in enumerate(pab_body_dirs):
        b = b / np.linalg.norm(b)
        R0, _ = Rotation.align_vectors([b], [pj])
        for j, phi in enumerate(phi_arr):
            R_twist = Rotation.from_rotvec(phi * b)
            R_total = R_twist * R0
            q_xyzw = R_total.as_quat()
            out[i * P + j] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return out


def quat_mul_v(qa, qb):
    """wxyz quat mul. qa (...,4), qb (...,4) → broadcast result."""
    aw, ax, ay, az = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    bw, bx, by, bz = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def setup_seed(seed, source):
    if source == "m048":
        master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"][seed]
        start_et = float(master["start_ets"][seed])
        end_time_utc = None
    else:
        master = np.load(str(DIAG / "m046_trajectories" / "m046_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"]
        start_et = None
        end_time_utc = "2020-02-05T11:00:00"

    I_tensor = master["inertia_tensor"]
    true_q0 = master["q0s"][seed]
    true_omega0 = master["omega0s"][seed]
    true_lc = master["mag_hifi"][seed]
    observed_lc = canonical_observed_lc(true_lc)

    geom_ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                start_et=start_et, end_time_utc=end_time_utc,
                                skip_true_lc=True)
    sun_vecs = geom_ctx.sun_pos - geom_ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = geom_ctx.obs_pos - geom_ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)

    return dict(
        seed=seed, source=source,
        obs_times=obs_times, observed_lc=observed_lc,
        true_q0=true_q0, true_omega0=true_omega0,
        I_tensor=I_tensor,
        sun_dirs=sun_dirs, obs_dirs=obs_dirs, obs_dist=geom_ctx.obs_dist,
    )


# ── Step 1: anchor selection (no bright filter) ─────────────────────────
_W = {}
def _init_step1(state):
    global _W
    _W.update(state)
    _W["model"] = _load_v1_surrogate()


def _scan_one_epoch(args):
    epoch_idx, sun_j2000, obs_j2000, obs_dist, observed_mag = args
    pab_body_dirs = _W["pab_body_dirs"]
    phi_arr = _W["phi_arr"]
    pab_j2000 = (sun_j2000 + obs_j2000) / np.linalg.norm(sun_j2000 + obs_j2000)
    qs_wxyz = quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000)
    qs_xyzw = qs_wxyz[:, [1, 2, 3, 0]]
    R_all = Rotation.from_quat(qs_xyzw).as_matrix()
    sj = sun_j2000 / np.linalg.norm(sun_j2000)
    oj = obs_j2000 / np.linalg.norm(obs_j2000)
    k1 = R_all @ sj
    k2 = R_all @ oj
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    od = np.full(R_all.shape[0], float(obs_dist))
    surr = _W["model"].predict_magnitude(k1, k2, 0.0, 15.0, od)
    diff = np.abs(surr - observed_mag)
    consistent = np.where((diff < _W["tolerance"]) & np.isfinite(surr))[0]
    return epoch_idx, len(consistent), consistent.astype(np.int32)


def step1_pick_anchor(ctx, n_pab=800, n_phi=80, tolerance=0.15,
                     middle_frac=0.3, n_workers=8, min_count=6,
                     max_obs_mag=11.0):
    """Returns (anchor_epoch, q_anchor_set_wxyz, anchor_meta).

    `max_obs_mag` is a SURROGATE-VALIDITY guard, NOT a brightness heuristic:
    above ~11 mag the v1 surrogate saturates near its dim ceiling and Q-counts
    become artifact-dominated (v3 of original pilot picked ep=283 mag=15.73,
    Q=48 spuriously). Mid-brightness epochs (5-11 mag) are still admitted.
    """
    n_t = len(ctx["obs_times"])
    lo = int(0.5 * (1 - middle_frac) * n_t)
    hi = int(0.5 * (1 + middle_frac) * n_t)
    cand_eps = np.arange(lo, hi)
    obs = ctx["observed_lc"]
    cand_eps = cand_eps[np.isfinite(obs[cand_eps]) & (obs[cand_eps] < max_obs_mag)]
    print(f"  Step 1: {len(cand_eps)} candidate epochs in middle {middle_frac*100:.0f}% "
          f"(mag<{max_obs_mag} for surrogate validity)")

    pab_body_dirs = fibonacci_sphere(n_pab)
    phi_arr = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    state = dict(pab_body_dirs=pab_body_dirs, phi_arr=phi_arr,
                 tolerance=tolerance)
    args_list = [(int(ep),
                  ctx["sun_dirs"][ep] * ctx["obs_dist"][ep],
                  ctx["obs_dirs"][ep] * ctx["obs_dist"][ep],
                  float(ctx["obs_dist"][ep]),
                  float(ctx["observed_lc"][ep]))
                 for ep in cand_eps]
    n_consistent = np.full(n_t, -1, dtype=int)
    consistent_q_idx = {}
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_step1, initargs=(state,)) as pool:
        for ep, n, idx in pool.imap_unordered(_scan_one_epoch, args_list,
                                              chunksize=1):
            n_consistent[ep] = n
            consistent_q_idx[ep] = idx
    print(f"  Step 1: scan done in {time.time()-t0:.1f}s")

    valid = (n_consistent >= min_count)
    if not valid.any():
        raise RuntimeError(f"No epoch has ≥{min_count} consistent q's")
    candidates = np.where(valid)[0]
    best = int(candidates[np.argmin(n_consistent[candidates])])
    print(f"  Step 1: anchor=ep{best} (Q={n_consistent[best]}, mag={ctx['observed_lc'][best]:.2f})")

    sun_b = ctx["sun_dirs"][best]
    obs_b = ctx["obs_dirs"][best]
    pab_j2000 = (sun_b + obs_b) / np.linalg.norm(sun_b + obs_b)
    q_anchor_set = quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000)[
        consistent_q_idx[best]]

    return best, q_anchor_set, dict(
        anchor_epoch=best,
        n_consistent=n_consistent,
        Q=int(n_consistent[best]),
        anchor_mag=float(ctx["observed_lc"][best]),
        n_pab=n_pab, n_phi=n_phi, tolerance=tolerance,
    )


# ── Constraint epoch set (no IPL — surrogate-driven) ────────────────────

def select_constraint_epochs(ctx, anchor_epoch, target_count=100,
                             dim_threshold=11.0, middle_frac=0.6):
    """spec_peaks ∪ (uniform sample of mid-LC informative epochs).

    No IPL data. Informative ≈ observed_lc < dim_threshold AND middle 60%
    of LC (manage propagation error at edges). Spec peaks always included.
    """
    obs = ctx["observed_lc"]
    n_t = len(obs)
    lo = int(0.5 * (1 - middle_frac) * n_t)
    hi = int(0.5 * (1 + middle_frac) * n_t)

    peaks_idx, _ = find_peaks(-obs, distance=5, prominence=0.3)
    spec_peaks = peaks_idx[obs[peaks_idx] < 9.0]

    informative_mask = np.zeros(n_t, dtype=bool)
    informative_mask[lo:hi] = True
    informative_mask &= np.isfinite(obs)
    informative_mask &= (obs < dim_threshold)
    informative_eps = np.where(informative_mask)[0]
    if len(informative_eps) > target_count:
        sel = np.linspace(0, len(informative_eps) - 1, target_count).astype(int)
        informative_eps = informative_eps[sel]

    union = np.sort(np.unique(np.concatenate(
        [np.asarray(spec_peaks, dtype=int),
         np.asarray(informative_eps, dtype=int)])))
    constraint_eps = union[union != anchor_epoch]
    return constraint_eps, spec_peaks


# ── Step 2: kernel build + scoring ──────────────────────────────────────
_K = {}
def _init_kernel_worker(state):
    global _K
    _K.update(state)


def _kernel_one_dir(args):
    """Propagate identity quaternion under (omega_dir × omega_mag) to each
    constraint dt. dt's may be negative (epoch before anchor); propagate
    those backward via the involution q(-t, ω) = q(t, -ω).
    """
    di, omega_dir = args
    omega_mags = _K["omega_mags"]
    dt_arr = _K["dt_constraints"]            # (K,) signed
    I_tensor = _K["I_tensor"]
    n_mags = len(omega_mags)
    n_eps = len(dt_arr)
    out = np.zeros((n_mags, n_eps, 4), dtype=np.float32)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])

    fwd_mask = dt_arr > 0
    bwd_mask = dt_arr < 0
    fwd_dts = dt_arr[fwd_mask]
    bwd_dts_pos = -dt_arr[bwd_mask]          # positive magnitudes for back-prop

    fwd_times = np.concatenate([[0.0], np.sort(fwd_dts)]) if len(fwd_dts) else None
    bwd_times = np.concatenate([[0.0], np.sort(bwd_dts_pos)]) if len(bwd_dts_pos) else None

    fwd_order = np.argsort(fwd_dts) if len(fwd_dts) else None
    bwd_order = np.argsort(bwd_dts_pos) if len(bwd_dts_pos) else None
    fwd_idx = np.where(fwd_mask)[0]
    bwd_idx = np.where(bwd_mask)[0]
    zero_idx = np.where(dt_arr == 0)[0]

    for mi, mag in enumerate(omega_mags):
        omega_vec = omega_dir * mag
        # Forward branch
        if fwd_times is not None:
            quats_f, _ = propagate_attitude(q_id, omega_vec, fwd_times,
                                            "tumbling", I_tensor)
            # quats_f[0] is at t=0, quats_f[1:] correspond to sorted fwd_dts
            for k, src in enumerate(fwd_order):
                out[mi, fwd_idx[src]] = quats_f[1 + k].astype(np.float32)
        # Backward branch (use -omega for time-reversal)
        if bwd_times is not None:
            quats_b, _ = propagate_attitude(q_id, -omega_vec, bwd_times,
                                            "tumbling", I_tensor)
            for k, src in enumerate(bwd_order):
                out[mi, bwd_idx[src]] = quats_b[1 + k].astype(np.float32)
        # Zero (anchor itself, shouldn't happen since we exclude it but defensive)
        for zi in zero_idx:
            out[mi, zi] = q_id.astype(np.float32)
    return di, out


def build_kernel(ctx, anchor_epoch, constraint_eps, omega_dirs, omega_mags,
                 n_workers=8):
    obs_times = ctx["obs_times"]
    anchor_time = float(obs_times[anchor_epoch])
    dt_constraints = obs_times[constraint_eps] - anchor_time
    print(f"  Kernel: {len(omega_dirs)}d × {len(omega_mags)}m × "
          f"{len(constraint_eps)} eps  ({len(omega_dirs)*len(omega_mags)*len(constraint_eps)*16/1e6:.1f} MB)")
    state = dict(omega_mags=np.asarray(omega_mags),
                 dt_constraints=np.asarray(dt_constraints),
                 I_tensor=ctx["I_tensor"])
    q_delta = np.zeros((len(omega_dirs), len(omega_mags), len(constraint_eps), 4),
                       dtype=np.float32)
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_kernel_worker, initargs=(state,)) as pool:
        args_list = [(di, omega_dirs[di]) for di in range(len(omega_dirs))]
        for di, block in pool.imap_unordered(_kernel_one_dir, args_list, chunksize=8):
            q_delta[di] = block
    print(f"  Kernel built in {time.time()-t0:.1f}s")
    return q_delta, dt_constraints


# ── Scorer ───────────────────────────────────────────────────────────────
_S = {}
def _init_scorer(state):
    global _S
    _S.update(state)
    _S["model"] = _load_v1_surrogate()


def _score_one_dir(args):
    """For one ω-direction, score every (mag, q_anchor) and return min cost."""
    di = args
    q_delta = _S["q_delta"][di]                         # (M, K, 4)
    q_anchor_set = _S["q_anchor_set"]                   # (Q, 4) wxyz
    sun_dirs_eps = _S["sun_dirs_eps"]                   # (K, 3) j2000
    obs_dirs_eps = _S["obs_dirs_eps"]                   # (K, 3) j2000
    obs_dist_eps = _S["obs_dist_eps"]                   # (K,)
    observed_eps = _S["observed_eps"]                   # (K,)
    M, K, _ = q_delta.shape
    Q = q_anchor_set.shape[0]

    best_mse = np.inf
    best_mi = -1
    best_qi = -1
    best_q0 = np.zeros(4)

    for mi in range(M):
        # q_world[Q, K, 4] = q_anchor[Q, 1, 4] · q_delta[mi, K, 4]
        qd = q_delta[mi][None, :, :]                    # (1, K, 4)
        qa = q_anchor_set[:, None, :]                   # (Q, 1, 4)
        q_world = quat_mul_v(qa, qd)                    # (Q, K, 4)
        q_world /= np.linalg.norm(q_world, axis=-1, keepdims=True)
        q_xyzw = q_world[..., [1, 2, 3, 0]].reshape(-1, 4)
        R = Rotation.from_quat(q_xyzw).as_matrix().reshape(Q, K, 3, 3)
        # rotate j2000 sun/obs into body frame using R
        k1 = np.einsum("qkij,kj->qki", R, sun_dirs_eps)
        k2 = np.einsum("qkij,kj->qki", R, obs_dirs_eps)
        k1 /= np.linalg.norm(k1, axis=-1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=-1, keepdims=True)
        k1f = k1.reshape(-1, 3)
        k2f = k2.reshape(-1, 3)
        od_tile = np.tile(obs_dist_eps, Q)
        pred = _S["model"].predict_magnitude(k1f, k2f, 0.0, 15.0, od_tile)
        pred = pred.reshape(Q, K)
        valid = np.isfinite(pred)
        # MSE per anchor over its valid epochs
        for qi in range(Q):
            v = valid[qi]
            if v.sum() < 10:
                continue
            mse = float(np.mean((pred[qi, v] - observed_eps[v]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_mi = mi
                best_qi = qi
                # q0 = q_anchor · q_delta_anchor^{-1}, but we don't have
                # q_delta at anchor (it's identity by construction). So
                # q0 = back-prop q_anchor to t=0 under -ω_vec; expensive.
                # Skip for now — return q_world[ep0] proxy. Caller can
                # re-derive by propagating ω forward from anchor_time.
                best_q0 = q_anchor_set[qi]
    return di, best_mse, best_mi, best_qi, best_q0


def score(q_delta, q_anchor_set, ctx, constraint_eps, n_workers=8):
    sun_dirs_eps = (ctx["sun_dirs"][constraint_eps]
                    * ctx["obs_dist"][constraint_eps][:, None])
    obs_dirs_eps = (ctx["obs_dirs"][constraint_eps]
                    * ctx["obs_dist"][constraint_eps][:, None])
    sun_dirs_eps /= np.linalg.norm(sun_dirs_eps, axis=1, keepdims=True)
    obs_dirs_eps /= np.linalg.norm(obs_dirs_eps, axis=1, keepdims=True)

    state = dict(
        q_delta=q_delta,
        q_anchor_set=q_anchor_set,
        sun_dirs_eps=sun_dirs_eps,
        obs_dirs_eps=obs_dirs_eps,
        obs_dist_eps=ctx["obs_dist"][constraint_eps],
        observed_eps=ctx["observed_lc"][constraint_eps],
    )
    n_dirs = q_delta.shape[0]
    cost = np.full(n_dirs, np.inf)
    best_mi_arr = np.full(n_dirs, -1, dtype=int)
    best_qi_arr = np.full(n_dirs, -1, dtype=int)
    t0 = time.time()
    print(f"  Scoring {n_dirs} dirs (Pool({n_workers}))...")
    with mp.Pool(n_workers, initializer=_init_scorer, initargs=(state,)) as pool:
        n_done = 0
        for di, mse, mi, qi, _q0 in pool.imap_unordered(
                _score_one_dir, range(n_dirs), chunksize=8):
            cost[di] = mse
            best_mi_arr[di] = mi
            best_qi_arr[di] = qi
            n_done += 1
            if n_done % max(1, n_dirs // 10) == 0:
                print(f"    {n_done}/{n_dirs} ({time.time()-t0:.0f}s)")
    print(f"  Score complete in {time.time()-t0:.1f}s")
    return cost, best_mi_arr, best_qi_arr


# ── Main ────────────────────────────────────────────────────────────────

def run(seed, source, n_pab=800, n_phi=80, tolerance=0.15, middle_frac=0.3,
        min_count=6, n_dirs=2000, n_mags=20, n_constraint_target=100,
        n_workers=8, dim_threshold=11.0, max_obs_mag=11.0):
    print(f"\n=== seed {seed} ({source}) — kernel-consistency ===")
    t_global = time.time()
    ctx = setup_seed(seed, source)

    # Step 1: anchor + q-set
    anchor_epoch, q_anchor_set, anchor_meta = step1_pick_anchor(
        ctx, n_pab=n_pab, n_phi=n_phi, tolerance=tolerance,
        middle_frac=middle_frac, n_workers=n_workers, min_count=min_count,
        max_obs_mag=max_obs_mag)

    # Constraint epochs
    constraint_eps, spec_peaks = select_constraint_epochs(
        ctx, anchor_epoch, target_count=n_constraint_target,
        dim_threshold=dim_threshold, middle_frac=0.6)
    print(f"  Constraint set: {len(constraint_eps)} eps "
          f"({len(spec_peaks)} spec_peaks ∪ informative-mid-LC)")

    # |ω| grid (peak-count formula × ±30%, N_MAGS=20 — established)
    peaks_idx, _ = find_peaks(-ctx["observed_lc"], distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = float(np.deg2rad(omega_est_dps))
    omega_mags = omega_est_rad * np.linspace(0.70, 1.30, n_mags)
    omega_dirs = fibonacci_sphere(n_dirs)
    print(f"  |ω| grid: {n_mags} mags spanning [{omega_mags.min():.4f}, "
          f"{omega_mags.max():.4f}] rad/s; est={omega_est_rad:.4f}, "
          f"true=N/A (no oracle)")

    # Build kernel
    q_delta, dt_constraints = build_kernel(
        ctx, anchor_epoch, constraint_eps, omega_dirs, omega_mags,
        n_workers=n_workers)

    # Score
    cost, best_mi, best_qi = score(
        q_delta, q_anchor_set, ctx, constraint_eps, n_workers=n_workers)

    # Truth-distance for ranking
    true_w_mag = float(np.linalg.norm(ctx["true_omega0"]))
    true_w_dir = ctx["true_omega0"] / true_w_mag
    cos = np.clip(omega_dirs @ true_w_dir, -1, 1)
    err_signed_deg = np.degrees(np.arccos(cos))

    order = np.argsort(cost)
    print(f"\n  Top-5 by cost:")
    for r in range(5):
        di = order[r]
        chosen_mag = omega_mags[best_mi[di]] if best_mi[di] >= 0 else np.nan
        print(f"    rank{r+1}: dir_idx={di} dir_err={err_signed_deg[di]:6.2f}°  "
              f"|ω|={chosen_mag:.4f} (true={true_w_mag:.4f})  "
              f"mse={cost[di]:.4f}")

    print(f"\n  Closest dir in grid to truth: "
          f"{err_signed_deg.min():.2f}°  "
          f"(rank by cost: {(cost <= cost[np.argmin(err_signed_deg)]).sum()}/{n_dirs})")

    # Save
    out_dir = DIAG / "kernel_consistency" / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "result.npz",
             omega_dirs=omega_dirs, omega_mags=omega_mags,
             cost=cost, best_mag_idx=best_mi, best_anchor_idx=best_qi,
             anchor_epoch=anchor_epoch,
             constraint_eps=constraint_eps,
             q_anchor_set=q_anchor_set,
             omega_est_rad=omega_est_rad,
             err_signed_deg=err_signed_deg,
             true_omega0=ctx["true_omega0"], true_q0=ctx["true_q0"])
    summary = dict(
        seed=int(seed), source=source,
        anchor_epoch=int(anchor_epoch), Q=int(anchor_meta["Q"]),
        anchor_mag=float(anchor_meta["anchor_mag"]),
        n_constraint_eps=int(len(constraint_eps)),
        n_dirs=int(n_dirs), n_mags=int(n_mags),
        omega_est_rad=float(omega_est_rad),
        true_w_mag=float(true_w_mag),
        rank1_dir_err_deg=float(err_signed_deg[order[0]]),
        rank1_chosen_mag=float(omega_mags[best_mi[order[0]]]) if best_mi[order[0]] >= 0 else None,
        rank1_mse=float(cost[order[0]]),
        closest_to_truth_dir_err_deg=float(err_signed_deg.min()),
        closest_to_truth_rank_by_cost=int(
            (cost <= cost[np.argmin(err_signed_deg)]).sum()),
        wall_clock_s=float(time.time() - t_global),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_dir / 'result.npz'}")
    print(f"  Saved: {out_dir / 'summary.json'}")
    print(f"  Total wall: {time.time()-t_global:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-pab", type=int, default=800)
    ap.add_argument("--n-phi", type=int, default=80)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--middle-frac", type=float, default=0.3)
    ap.add_argument("--min-count", type=int, default=6)
    ap.add_argument("--n-dirs", type=int, default=2000)
    ap.add_argument("--n-mags", type=int, default=20)
    ap.add_argument("--n-constraint-target", type=int, default=100)
    ap.add_argument("--dim-threshold", type=float, default=11.0)
    ap.add_argument("--max-obs-mag", type=float, default=11.0,
                    help="Anchor candidate cap. Surrogate-validity guard "
                         "(NOT brightness-as-constraint). Default 11.")
    args = ap.parse_args()
    for seed in args.seeds:
        run(seed, args.traj_source,
            n_pab=args.n_pab, n_phi=args.n_phi,
            tolerance=args.tolerance, middle_frac=args.middle_frac,
            min_count=args.min_count, n_dirs=args.n_dirs, n_mags=args.n_mags,
            n_constraint_target=args.n_constraint_target,
            n_workers=args.workers, dim_threshold=args.dim_threshold,
            max_obs_mag=args.max_obs_mag)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
