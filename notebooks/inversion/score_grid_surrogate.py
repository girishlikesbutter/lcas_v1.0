#!/usr/bin/env python3
"""Surrogate-cost grid sweep over 2000 Fibonacci omega-directions.

For each direction:
  - sweep |omega| over a small range around the peak-counting estimate
    (or include truth |omega| if --include-truth-mag),
  - for each anchor-allowed normal at the m103 anchor epoch,
    sweep phi over N_PHI_GRID values and back-propagate q_anchor to t=0,
  - forward-model the LC via the surrogate, score against canonical-noise
    observed LC,
  - keep per-direction min over (mag, normal, phi).

Output: m103_hybrid_<source>/seed_NNN/grid_surr_ckpt.npz with
    min_surr_mse  (N_DIRS,)
    best_mag_idx  (N_DIRS,)
    best_phi_idx  (N_DIRS,)
    best_normal   (N_DIRS,)
    best_q0       (N_DIRS, 4)
    omega_dirs    (N_DIRS, 3)
    omega_mags_searched
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Limit BLAS threads BEFORE numpy import — workers will inherit. Without this,
# 8 workers × 4 BLAS threads = 32 threads thrashing 16 cores; surrogate eval
# slowdown destroys throughput.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402
from surrogate_model.surrogate import SurrogateModel  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

N_DIRS = 2000
N_PHI = 12  # coarse first cut; m103 uses 36 internally
Z_NORMALS = {4, 5}


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])


def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def setup_seed(seed, source):
    if source == "m048":
        master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"][seed]
        pab_j2000 = master["pab_j2000"][seed]
        start_et = float(master["start_ets"][seed])
        end_time_utc = None
    else:
        master = np.load(str(DIAG / "m046_trajectories" / "m046_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"]
        pab_j2000 = master["pab_j2000"]
        start_et = None
        end_time_utc = "2020-02-05T11:00:00"

    unique_normals = master["unique_normals"]
    I_tensor = master["inertia_tensor"]
    true_q0 = master["q0s"][seed]
    true_omega0 = master["omega0s"][seed]
    true_lc = master["mag_hifi"][seed]
    observed_lc = canonical_observed_lc(true_lc)

    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = float(np.deg2rad(omega_est_dps))
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

    smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
    smooth_mags = smoothed_lc[spec_peaks]
    sr = np.argsort(smooth_mags)
    if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
        anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
    else:
        anchor_rank = sr[0]
    anchor_idx = int(spec_peaks[anchor_rank])
    anchor_time = obs_times[anchor_idx]
    anchor_mag = observed_lc[anchor_idx]
    anchor_allowed = get_allowed_normals(anchor_mag)
    pab_anchor = pab_j2000[anchor_idx]

    # True omega in body frame at anchor epoch
    _, w_hist = propagate_attitude(true_q0, true_omega0,
                                   np.array([0.0, anchor_time]),
                                   "tumbling", I_tensor)
    true_omega_at_anchor = w_hist[1]
    true_w_mag = float(np.linalg.norm(true_omega_at_anchor))

    return dict(
        seed=seed, source=source,
        unique_normals=unique_normals, I_tensor=I_tensor,
        true_q0=true_q0, true_omega0=true_omega0,
        anchor_idx=anchor_idx, anchor_time=anchor_time,
        anchor_mag=anchor_mag, anchor_allowed=anchor_allowed,
        pab_anchor=pab_anchor,
        obs_times=obs_times,
        observed_lc=observed_lc,
        omega_est_rad=omega_est_rad,
        true_w_mag=true_w_mag,
        start_et=start_et, end_time_utc=end_time_utc,
    )


# ── Worker state ────────────────────────────────────────────────────────
_W = {}


def _init_worker(state):
    """Each worker loads the surrogate once and caches the per-anchor q
    candidates (one per (normal, phi)). Worker also sets up the SPICE/geom
    context."""
    global _W
    _W.update(state)
    _W["model"] = SurrogateModel.load_default()


def _quat_mul_one_v(q1_arr, q2):
    """q1_arr (N, 4) wxyz × q2 (4,) wxyz → (N, 4) wxyz."""
    w1, x1, y1, z1 = q1_arr[:, 0], q1_arr[:, 1], q1_arr[:, 2], q1_arr[:, 3]
    w2, x2, y2, z2 = q2
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _quat_mul_v_batch(q1_arr, q2_arr):
    """q1_arr (P, 4) × q2_arr (T, 4) → (P, T, 4)."""
    w1, x1, y1, z1 = (q1_arr[:, 0:1], q1_arr[:, 1:2],
                      q1_arr[:, 2:3], q1_arr[:, 3:4])  # (P, 1)
    w2, x2, y2, z2 = (q2_arr[:, 0], q2_arr[:, 1],
                      q2_arr[:, 2], q2_arr[:, 3])      # (T,)
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _score_one_dir(args):
    """Compute min surrogate-MSE for one omega-direction over (|w|, normal, phi).

    Vectorised: per direction we propagate omega ONCE per (mag) — getting
    delta_qs over the full 500-epoch window AND at the anchor epoch — then
    derive q0 per phi via cheap quaternion multiplication, batch the
    surrogate forward-LC over (P phi values × N_t epochs) into a single
    predict_magnitude call. ~25× faster than the per-(q0, ω) loop.
    """
    di, omega_dir = args
    obs_times = _W["obs_times"]
    I_tensor = _W["I_tensor"]
    sun_dirs = _W["sun_dirs"]
    obs_dirs = _W["obs_dirs"]
    obs_dist_km = _W["obs_dist_km"]
    observed_lc = _W["observed_lc"]
    obs_valid = _W["obs_valid"]
    qa_cache = _W["qa_cache"]
    anchor_time = _W["anchor_time"]
    omega_mags = _W["omega_mags_searched"]
    model = _W["model"]
    n_t = len(obs_times)

    best_mse = np.inf
    best_mag_idx = -1
    best_phi_idx = -1
    best_normal = -1
    best_q0 = np.zeros(4)

    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    obs_dist_tile_cache = {}  # P -> (P*n_t,) tile

    for mi, mag in enumerate(omega_mags):
        omega_vec = omega_dir * mag

        # 1 ODE: identity propagated over the 500 obs epochs → delta_qs (n_t, 4)
        delta_qs, _ = propagate_attitude(q_id, omega_vec, obs_times,
                                         "tumbling", I_tensor)
        # 1 ODE: identity propagated over [0, anchor_time] → delta_q_anchor
        anchor_traj, _ = propagate_attitude(q_id, omega_vec,
                                            np.array([0.0, anchor_time]),
                                            "tumbling", I_tensor)
        d_q_a = anchor_traj[-1]
        # Conjugate (wxyz)
        d_q_a_conj = np.array([d_q_a[0], -d_q_a[1], -d_q_a[2], -d_q_a[3]])

        for ni, (qa_wxyz_arr, _phi_arr) in qa_cache.items():
            P = qa_wxyz_arr.shape[0]
            # q0 = qa_anchor · delta_q_anchor^{-1} per phi
            q0_arr = _quat_mul_one_v(qa_wxyz_arr, d_q_a_conj)
            q0_arr = q0_arr / np.linalg.norm(q0_arr, axis=1, keepdims=True)

            # q_world(t) = q0 ⊗ delta_q(t) → (P, n_t, 4)
            q_world = _quat_mul_v_batch(q0_arr, delta_qs)
            q_world_xyzw = q_world[..., [1, 2, 3, 0]]  # scipy ordering

            # Rotation matrices (P, n_t, 3, 3). Reshape→batch→reshape.
            R_all = (Rotation.from_quat(q_world_xyzw.reshape(-1, 4))
                     .as_matrix().reshape(P, n_t, 3, 3))
            k1 = np.einsum("ptij,tj->pti", R_all, sun_dirs)
            k2 = np.einsum("ptij,tj->pti", R_all, obs_dirs)
            k1 = k1 / np.linalg.norm(k1, axis=2, keepdims=True)
            k2 = k2 / np.linalg.norm(k2, axis=2, keepdims=True)

            # One batched surrogate call across all phi × epochs
            k1_flat = k1.reshape(-1, 3)
            k2_flat = k2.reshape(-1, 3)
            if P not in obs_dist_tile_cache:
                obs_dist_tile_cache[P] = np.tile(obs_dist_km, P)
            obs_dist_tile = obs_dist_tile_cache[P]
            pred_flat = model.predict_magnitude(k1_flat, k2_flat,
                                                0.0, 15.0, obs_dist_tile)
            pred = pred_flat.reshape(P, n_t)

            # Per-phi MSE; broadcasting obs_valid across phi.
            for pi in range(P):
                p = pred[pi]
                v = obs_valid & np.isfinite(p)
                if v.sum() < 10:
                    continue
                mse = float(np.mean((p[v] - observed_lc[v]) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_mag_idx = mi
                    best_phi_idx = pi
                    best_normal = ni
                    best_q0 = q0_arr[pi].copy()

    return di, best_mse, best_mag_idx, best_phi_idx, best_normal, best_q0


def score_seed(seed, source, n_workers=8, include_truth_mag=False, n_mags=5):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    ctx = setup_seed(seed, source)
    print(f"  anchor ep={ctx['anchor_idx']}, mag={ctx['anchor_mag']:.2f}, "
          f"allowed_normals={ctx['anchor_allowed']}", flush=True)

    omega_dirs = fibonacci_sphere(N_DIRS)

    # |omega| sweep — m103 uses linear over [0.7, 1.3] of omega_est. We add
    # one bonus sample at truth |omega| if requested.
    omega_est = ctx["omega_est_rad"]
    omega_mags = list(np.linspace(0.7 * omega_est, 1.3 * omega_est, n_mags))
    if include_truth_mag:
        omega_mags = sorted(set(omega_mags + [ctx["true_w_mag"]]))
    omega_mags = np.array(omega_mags)
    print(f"  |w| sweep ({len(omega_mags)} mags): {omega_mags}", flush=True)

    # Build qa cache per normal at the anchor.
    phi_xy = np.linspace(0, np.pi, N_PHI, endpoint=False)
    phi_z = np.linspace(0, 2*np.pi, 2*N_PHI, endpoint=False)
    qa_cache = {}
    for ni in ctx["anchor_allowed"]:
        phi_arr = phi_z if ni in Z_NORMALS else phi_xy
        n_body = ctx["unique_normals"][ni]
        qa_wxyz_arr = np.array([anchor_q_from_phi(p, n_body, ctx["pab_anchor"])
                                for p in phi_arr])
        qa_cache[int(ni)] = (qa_wxyz_arr, phi_arr)

    # SPICE/geom setup
    print(f"  SPICE setup...", flush=True)
    geom_ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                start_et=ctx["start_et"], end_time_utc=ctx["end_time_utc"],
                                skip_true_lc=True)
    sun_vecs = geom_ctx.sun_pos - geom_ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = geom_ctx.obs_pos - geom_ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)

    state = dict(
        obs_times=ctx["obs_times"], I_tensor=ctx["I_tensor"],
        sun_dirs=sun_dirs, obs_dirs=obs_dirs, obs_dist_km=geom_ctx.obs_dist,
        observed_lc=ctx["observed_lc"],
        obs_valid=np.isfinite(ctx["observed_lc"]),
        qa_cache=qa_cache,
        anchor_time=ctx["anchor_time"],
        omega_mags_searched=omega_mags,
    )

    n_per_dir = sum(qa_cache[ni][0].shape[0] for ni in qa_cache) * len(omega_mags)
    print(f"  scoring {N_DIRS} dirs × {n_per_dir} (q0,w) per dir = "
          f"{N_DIRS * n_per_dir} total surrogate evals "
          f"on {n_workers} workers...", flush=True)

    args_iter = [(di, omega_dirs[di]) for di in range(N_DIRS)]
    min_mse = np.full(N_DIRS, np.inf)
    best_mag_idx = np.full(N_DIRS, -1, dtype=int)
    best_phi_idx = np.full(N_DIRS, -1, dtype=int)
    best_normal = np.full(N_DIRS, -1, dtype=int)
    best_q0 = np.zeros((N_DIRS, 4))

    t_score = time.time()
    n_done = 0
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(state,)) as pool:
        for di, mse, mi, pi, ni, q0 in pool.imap_unordered(
                _score_one_dir, args_iter, chunksize=4):
            min_mse[di] = mse
            best_mag_idx[di] = mi
            best_phi_idx[di] = pi
            best_normal[di] = ni
            best_q0[di] = q0
            n_done += 1
            if n_done % 200 == 0 or n_done == N_DIRS:
                el = time.time() - t_score
                eta = el / n_done * (N_DIRS - n_done)
                print(f"    [{n_done}/{N_DIRS}] elapsed {el:.1f}s, eta {eta:.1f}s",
                      flush=True)
    elapsed_score = time.time() - t_score
    print(f"  scoring done in {elapsed_score:.1f}s "
          f"({elapsed_score/N_DIRS*1000:.0f}ms / direction)", flush=True)

    out_path = m103_dir(seed, source) / "grid_surr_ckpt.npz"
    np.savez(out_path,
             N_DIRS=N_DIRS, N_PHI=N_PHI, n_mags=len(omega_mags),
             min_surr_mse=min_mse,
             best_mag_idx=best_mag_idx, best_phi_idx=best_phi_idx,
             best_normal=best_normal, best_q0=best_q0,
             omega_dirs=omega_dirs,
             omega_mags_searched=omega_mags,
             include_truth_mag=include_truth_mag)
    print(f"  saved {out_path}", flush=True)
    print(f"    min_surr_mse: best={np.nanmin(min_mse):.4f}, "
          f"median={np.nanmedian(min_mse):.4f}, "
          f"max={np.nanmax(min_mse[np.isfinite(min_mse)]):.4f}",
          flush=True)
    print(f"  total wall {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-truth-mag", action="store_true",
                    help="Include exact truth |omega| as an extra magnitude sample (oracle).")
    ap.add_argument("--n-mags", type=int, default=5)
    args = ap.parse_args()
    for seed in args.seeds:
        score_seed(seed, args.traj_source, n_workers=args.workers,
                   include_truth_mag=args.include_truth_mag, n_mags=args.n_mags)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
