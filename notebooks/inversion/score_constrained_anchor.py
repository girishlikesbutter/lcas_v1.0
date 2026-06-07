#!/usr/bin/env python3
"""User's algorithm — full implementation.

1. For each obs epoch (near middle of LC), sample q broadly in SO(3) and ask
   the surrogate which q's produce the observed magnitude at that epoch's
   real phase / shadowing geometry. The COUNT of consistent q's is the
   constraint metric. Pick the epoch with the smallest count.

2. ω grid: 2000 Fibonacci × small |ω| range.

3. δq-factorize: for each (q_anchor in the consistent-q set from step 1, ω),
   back-propagate to t=0 to get q0.

4. Rank all (q0, ω) candidates by surrogate full-LC MSE.

Output:
  data/.../m103_hybrid_<src>/seed_NNN/constrained_anchor_ckpt.npz
"""
import argparse
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Single-threaded BLAS for clean parallelism
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402
from surrogate_model.surrogate import SurrogateModel  # noqa: E402
from surrogate_model import surrogate_v1 as _v1  # noqa: E402

# v1 surrogate file paths (single direct net, ~3× faster than v2 ensemble).
_V1_PKG_DIR = Path("/home/girish/surrogate_model/surrogate_model")
_V1_WEIGHTS = _V1_PKG_DIR / "s10_5M_weights.npz"
_V1_NORM = _V1_PKG_DIR / "s10_5M_normalization.npz"


def _load_v1_surrogate():
    return _v1.SurrogateModel(str(_V1_WEIGHTS), str(_V1_NORM))

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])


def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


# ── Quaternion sampler on SO(3) ──────────────────────────────────────────
# We parametrise q by (PAB_body unit vector ∈ S² × azimuth around it). The
# unit vector is a fibonacci-sphere; azimuth is a phi grid. This gives
# uniform-ish SO(3) coverage without prejudice toward any facet normal.

def sample_q_so3(n_pab=200, n_phi=36):
    """Returns q_arr (N=n_pab*n_phi, 4) wxyz. q_i represents a body-from-J2000
    rotation that places PAB_J2000=(0,0,1) into a sampled body-frame direction
    `b`, twisted by phi around b. The actual application below builds a
    per-epoch q from each sample by aligning the epoch's PAB_J2000 with each
    sampled body-direction `b`, twisted by phi around `b`.

    Here we just produce the (b, phi) pairs; the caller composes with the
    epoch-specific PAB.
    """
    pab_dirs = fibonacci_sphere(n_pab)
    phi_arr = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    return pab_dirs, phi_arr


def quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000):
    """For each (body-direction b, phi), produce q wxyz that rotates
    pab_j2000 to b in the body frame, twisted by phi around b.

    Returns q_arr (N_pab*N_phi, 4).
    """
    N = pab_body_dirs.shape[0]
    P = phi_arr.shape[0]
    out = np.zeros((N * P, 4))
    pj = pab_j2000 / np.linalg.norm(pab_j2000)
    for i, b in enumerate(pab_body_dirs):
        b = b / np.linalg.norm(b)
        # base rotation aligning pj → b in body frame
        R0, _ = Rotation.align_vectors([b], [pj])
        for j, phi in enumerate(phi_arr):
            R_twist = Rotation.from_rotvec(phi * b)
            R_total = R_twist * R0
            q_xyzw = R_total.as_quat()
            out[i * P + j] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return out


# ── Worker state ────────────────────────────────────────────────────────
_W = {}


def _init_worker(state):
    global _W
    _W.update(state)
    # Step-1 worker uses v1 surrogate (faster, less accurate). Wider
    # tolerance absorbs both noise σ and v1's larger residual MAE.
    _W["model"] = _load_v1_surrogate()


def _scan_one_epoch(args):
    """For one obs epoch, evaluate surrogate magnitude at every sampled q and
    return:
      (epoch_idx, n_consistent, q_consistent_indices)

    A q is 'consistent' if |surr_mag - observed_mag| < tol.
    """
    epoch_idx, sun_j2000, obs_j2000, obs_dist, observed_mag = args
    pab_body_dirs = _W["pab_body_dirs"]
    phi_arr = _W["phi_arr"]
    pab_j2000 = (sun_j2000 + obs_j2000) / np.linalg.norm(sun_j2000 + obs_j2000)
    qs_wxyz = quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000)
    qs_xyzw = qs_wxyz[:, [1, 2, 3, 0]]
    R_all = Rotation.from_quat(qs_xyzw).as_matrix()  # (N, 3, 3)
    # rotate sun, obs into body frame for each q
    sj = sun_j2000 / np.linalg.norm(sun_j2000)
    oj = obs_j2000 / np.linalg.norm(obs_j2000)
    k1 = R_all @ sj  # (N, 3)
    k2 = R_all @ oj
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    od = np.full(R_all.shape[0], float(obs_dist))
    surr_mag = _W["model"].predict_magnitude(k1, k2, 0.0, 15.0, od)
    diff = np.abs(surr_mag - observed_mag)
    tol = _W["tolerance"]
    consistent = np.where((diff < tol) & np.isfinite(surr_mag))[0]
    return epoch_idx, len(consistent), consistent.astype(np.int32)


# ── Main pipeline ────────────────────────────────────────────────────────

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
        start_et=start_et, end_time_utc=end_time_utc,
    )


def run(seed, source, n_pab=200, n_phi=36, tolerance=0.05,
        middle_frac=0.6, n_workers=8, n_dirs=2000, n_mags=1,
        include_truth_mag=True, max_obs_mag=11.0):
    print(f"\n=== seed {seed} ({source}) ===", flush=True)
    t_global = time.time()
    ctx = setup_seed(seed, source)
    n_t = len(ctx["observed_lc"])

    # Step 1: scan candidate epochs (middle window) for q-set narrowness.
    pab_body_dirs, phi_arr = sample_q_so3(n_pab=n_pab, n_phi=n_phi)
    n_q_per_epoch = n_pab * n_phi
    half = int(n_t * middle_frac / 2)
    epoch_lo = max(0, n_t // 2 - half)
    epoch_hi = min(n_t, n_t // 2 + half)
    middle = np.arange(epoch_lo, epoch_hi)
    # Filter to epochs where the surrogate is in its well-modeled regime.
    # Above ~13 mag the surrogate saturates, so any "tight q-set" is the
    # dim-saturation degeneracy, not a real attitude constraint.
    bright_mask = ctx["observed_lc"][middle] < max_obs_mag
    cand_epochs = middle[bright_mask]
    print(f"  Step 1: middle window [{epoch_lo}, {epoch_hi}) has {len(middle)} epochs; "
          f"{len(cand_epochs)} have observed_mag < {max_obs_mag} (well-modeled regime)", flush=True)
    if len(cand_epochs) == 0:
        raise RuntimeError(f"No epoch in middle has obs_mag < {max_obs_mag}; widen max_obs_mag.")
    print(f"  scanning {len(cand_epochs)} epochs, "
          f"{n_q_per_epoch} q-samples each, tol=±{tolerance:.3f} mag (using v1 surrogate)", flush=True)

    state = dict(
        pab_body_dirs=pab_body_dirs, phi_arr=phi_arr,
        tolerance=tolerance,
    )

    # Build per-epoch arg tuples
    args_list = []
    for ep in cand_epochs:
        sun_j2000 = ctx["sun_dirs"][ep]
        obs_j2000 = ctx["obs_dirs"][ep]
        args_list.append((int(ep), sun_j2000, obs_j2000,
                          float(ctx["obs_dist"][ep]),
                          float(ctx["observed_lc"][ep])))

    n_consistent = np.zeros(n_t, dtype=int)
    consistent_q_idx = {}
    t1 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(state,)) as pool:
        n_done = 0
        for ep, n_c, c_idx in pool.imap_unordered(_scan_one_epoch, args_list,
                                                  chunksize=4):
            n_consistent[ep] = n_c
            consistent_q_idx[ep] = c_idx
            n_done += 1
            if n_done % 50 == 0 or n_done == len(args_list):
                el = time.time() - t1
                eta = el / n_done * (len(args_list) - n_done)
                print(f"    [{n_done}/{len(args_list)}] {el:.1f}s elapsed, "
                      f"ETA {eta:.1f}s", flush=True)
    print(f"  Step 1 done in {time.time()-t1:.1f}s", flush=True)

    # Pick the most-constrained epoch — smallest n_consistent that is at
    # least min_count. Sub-min_count values are rejected as undersampling
    # artifacts (Q=1 with 7200 samples on SO(3) is more likely a coverage
    # gap than a genuinely tight isoshell).
    min_count = max(3, int(0.0001 * n_q_per_epoch))
    in_window = np.zeros(n_t, dtype=bool)
    in_window[cand_epochs] = True
    eligible = in_window & (n_consistent >= min_count)
    if not eligible.any():
        # fall back: relax floor to >=1, take smallest non-zero
        print(f"    (no epoch hit min_count={min_count}; falling back to "
              f"smallest non-zero count)", flush=True)
        eligible = in_window & (n_consistent > 0)
    masked_n = np.where(eligible, n_consistent, 1 << 30)
    best_epoch = int(np.argmin(masked_n))
    n_at_best = int(n_consistent[best_epoch])
    print(f"  Most-constrained epoch: ep={best_epoch}, "
          f"n_consistent_q={n_at_best}, "
          f"observed_mag={ctx['observed_lc'][best_epoch]:.3f}, "
          f"min_count_floor={min_count}", flush=True)

    # Sanity: where does truth's actual q at best_epoch sit relative to our
    # sample set? Useful for diagnosing whether truth was even in our SO(3)
    # coverage at that epoch.
    quats_truth, _ = propagate_attitude(ctx["true_q0"], ctx["true_omega0"],
                                        np.array([0.0, ctx["obs_times"][best_epoch]]),
                                        "tumbling", ctx["I_tensor"])
    truth_q_at_anchor = quats_truth[-1]  # wxyz

    # Materialise the q-set for that epoch
    sun_b = ctx["sun_dirs"][best_epoch]
    obs_b = ctx["obs_dirs"][best_epoch]
    pab_j2000 = (sun_b + obs_b) / np.linalg.norm(sun_b + obs_b)
    q_anchor_set_wxyz = quats_from_pab_phi(pab_body_dirs, phi_arr, pab_j2000)[
        consistent_q_idx[best_epoch]]
    print(f"  q_anchor_set: {q_anchor_set_wxyz.shape[0]} candidates", flush=True)

    # Step 2-4: ω grid sweep × q_anchor set → q0 → surrogate full-LC MSE.
    # We REUSE the precompute trick: per ω, propagate identity once over the
    # full obs window AND to the anchor epoch.
    omega_dirs = fibonacci_sphere(n_dirs)
    # |ω| sweep — use a peak-counting estimate plus optional truth-mag
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(-ctx["observed_lc"], distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = float(np.deg2rad(omega_est_dps))
    omega_mags = list(np.linspace(0.7 * omega_est_rad, 1.3 * omega_est_rad, n_mags))
    if include_truth_mag:
        true_w_mag = float(np.linalg.norm(ctx["true_omega0"]))  # at t=0
        omega_mags = sorted(set(omega_mags + [true_w_mag]))
    omega_mags = np.array(omega_mags)

    anchor_time = float(ctx["obs_times"][best_epoch])
    print(f"\n  Step 2-4: ω-grid sweep ({n_dirs} dirs × {len(omega_mags)} mags × "
          f"{q_anchor_set_wxyz.shape[0]} q_anchors)", flush=True)

    state2 = dict(
        obs_times=ctx["obs_times"],
        I_tensor=ctx["I_tensor"],
        sun_dirs=ctx["sun_dirs"],
        obs_dirs=ctx["obs_dirs"],
        obs_dist_km=ctx["obs_dist"],
        observed_lc=ctx["observed_lc"],
        obs_valid=np.isfinite(ctx["observed_lc"]),
        anchor_time=anchor_time,
        omega_mags=omega_mags,
        q_anchor_set=q_anchor_set_wxyz,
    )

    args2 = [(di, omega_dirs[di]) for di in range(n_dirs)]
    min_mse = np.full(n_dirs, np.inf)
    best_mag_idx = np.full(n_dirs, -1, dtype=int)
    best_qa_idx = np.full(n_dirs, -1, dtype=int)
    best_q0 = np.zeros((n_dirs, 4))

    t2 = time.time()
    n_done = 0
    with mp.Pool(n_workers, initializer=_init_worker_step2,
                 initargs=(state2,)) as pool:
        for di, mse, mi, qi, q0 in pool.imap_unordered(_score_one_dir_step2,
                                                       args2, chunksize=4):
            min_mse[di] = mse
            best_mag_idx[di] = mi
            best_qa_idx[di] = qi
            best_q0[di] = q0
            n_done += 1
            if n_done % 200 == 0 or n_done == n_dirs:
                el = time.time() - t2
                eta = el / n_done * (n_dirs - n_done)
                print(f"    [{n_done}/{n_dirs}] {el:.1f}s, ETA {eta:.1f}s", flush=True)
    print(f"  Step 2-4 done in {time.time()-t2:.1f}s", flush=True)

    # Save
    out_dir = m103_dir(seed, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constrained_anchor_ckpt.npz"
    np.savez(out_path,
             # Step-1 results
             best_epoch=best_epoch,
             n_consistent=n_consistent,
             cand_epochs=cand_epochs,
             observed_mag_at_anchor=ctx["observed_lc"][best_epoch],
             q_anchor_set_wxyz=q_anchor_set_wxyz,
             truth_q_at_anchor_wxyz=truth_q_at_anchor,
             tolerance=tolerance,
             middle_frac=middle_frac,
             n_pab=n_pab, n_phi=n_phi,
             # Step 2-4 results
             omega_dirs=omega_dirs,
             omega_mags=omega_mags,
             include_truth_mag=include_truth_mag,
             min_surr_mse=min_mse,
             best_mag_idx=best_mag_idx,
             best_qa_idx=best_qa_idx,
             best_q0=best_q0,
             anchor_time=anchor_time)
    print(f"  saved {out_path}", flush=True)

    # Quick summary
    rank1 = int(np.argmin(min_mse))
    true_w_hat = ctx["true_omega0"] / np.linalg.norm(ctx["true_omega0"])
    offset = float(np.degrees(np.arccos(np.clip(
        float(omega_dirs[rank1] @ true_w_hat), -1, 1))))
    print(f"\n  Rank-1 ω direction: dir #{rank1}, offset from truth = "
          f"{offset:.2f}°, surr_mse = {min_mse[rank1]:.4f}", flush=True)
    print(f"  Total wall: {time.time()-t_global:.1f}s")


# ── Step 2-4 worker ──────────────────────────────────────────────────────

def _init_worker_step2(state):
    global _W
    _W.clear()
    _W.update(state)
    _W["model"] = SurrogateModel.load_default()


def _quat_mul_one_v(q1_arr, q2):
    w1, x1, y1, z1 = q1_arr[:, 0], q1_arr[:, 1], q1_arr[:, 2], q1_arr[:, 3]
    w2, x2, y2, z2 = q2
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _quat_mul_v_batch(q1_arr, q2_arr):
    w1 = q1_arr[:, 0:1]; x1 = q1_arr[:, 1:2]; y1 = q1_arr[:, 2:3]; z1 = q1_arr[:, 3:4]
    w2 = q2_arr[:, 0]; x2 = q2_arr[:, 1]; y2 = q2_arr[:, 2]; z2 = q2_arr[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _score_one_dir_step2(args):
    di, omega_dir = args
    obs_times = _W["obs_times"]
    I_tensor = _W["I_tensor"]
    sun_dirs = _W["sun_dirs"]
    obs_dirs = _W["obs_dirs"]
    obs_dist_km = _W["obs_dist_km"]
    observed_lc = _W["observed_lc"]
    obs_valid = _W["obs_valid"]
    anchor_time = _W["anchor_time"]
    omega_mags = _W["omega_mags"]
    qa_set = _W["q_anchor_set"]    # (Q, 4) wxyz
    model = _W["model"]
    n_t = len(obs_times)
    Q = qa_set.shape[0]

    best_mse = np.inf
    best_mag_idx = -1
    best_qa_idx = -1
    best_q0 = np.zeros(4)

    q_id = np.array([1.0, 0.0, 0.0, 0.0])

    for mi, mag in enumerate(omega_mags):
        omega_vec = omega_dir * mag
        delta_qs, _ = propagate_attitude(q_id, omega_vec, obs_times, "tumbling", I_tensor)
        anchor_traj, _ = propagate_attitude(q_id, omega_vec,
                                            np.array([0.0, anchor_time]),
                                            "tumbling", I_tensor)
        d_q_a = anchor_traj[-1]
        d_q_a_conj = np.array([d_q_a[0], -d_q_a[1], -d_q_a[2], -d_q_a[3]])

        # q0 = qa · delta_q_anchor^{-1}
        q0_arr = _quat_mul_one_v(qa_set, d_q_a_conj)
        q0_arr = q0_arr / np.linalg.norm(q0_arr, axis=1, keepdims=True)

        # q_world (Q, n_t, 4)
        q_world = _quat_mul_v_batch(q0_arr, delta_qs)
        q_world_xyzw = q_world[..., [1, 2, 3, 0]]
        R_all = (Rotation.from_quat(q_world_xyzw.reshape(-1, 4))
                 .as_matrix().reshape(Q, n_t, 3, 3))
        k1 = np.einsum("qtij,tj->qti", R_all, sun_dirs)
        k2 = np.einsum("qtij,tj->qti", R_all, obs_dirs)
        k1 = k1 / np.linalg.norm(k1, axis=2, keepdims=True)
        k2 = k2 / np.linalg.norm(k2, axis=2, keepdims=True)

        k1_flat = k1.reshape(-1, 3)
        k2_flat = k2.reshape(-1, 3)
        od_tile = np.tile(obs_dist_km, Q)
        pred_flat = model.predict_magnitude(k1_flat, k2_flat, 0.0, 15.0, od_tile)
        pred = pred_flat.reshape(Q, n_t)

        for qi in range(Q):
            v = obs_valid & np.isfinite(pred[qi])
            if v.sum() < 10:
                continue
            mse = float(np.mean((pred[qi, v] - observed_lc[v]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_mag_idx = mi
                best_qa_idx = qi
                best_q0 = q0_arr[qi].copy()

    return di, best_mse, best_mag_idx, best_qa_idx, best_q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-pab", type=int, default=500)
    ap.add_argument("--n-phi", type=int, default=72)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="|surr_mag - observed| tolerance in mag (1× noise σ).")
    ap.add_argument("--middle-frac", type=float, default=0.3)
    ap.add_argument("--n-dirs", type=int, default=2000)
    ap.add_argument("--n-mags", type=int, default=1)
    ap.add_argument("--include-truth-mag", dest="include_truth_mag",
                    action="store_true", default=False,
                    help="ORACLE: inject truth |omega| into mag grid. "
                         "Default OFF — only enable for diagnostic ceiling probes.")
    ap.add_argument("--no-include-truth-mag", dest="include_truth_mag",
                    action="store_false",
                    help="Honest mode (default).")
    ap.add_argument("--max-obs-mag", type=float, default=11.0,
                    help="Cap on observed magnitude — epochs dimmer than this "
                         "are excluded from Step 1 (surrogate saturates above ~13 mag).")
    args = ap.parse_args()
    for seed in args.seeds:
        run(seed, args.traj_source, n_pab=args.n_pab, n_phi=args.n_phi,
            tolerance=args.tolerance, middle_frac=args.middle_frac,
            n_workers=args.workers, n_dirs=args.n_dirs,
            n_mags=args.n_mags, include_truth_mag=args.include_truth_mag,
            max_obs_mag=args.max_obs_mag)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
