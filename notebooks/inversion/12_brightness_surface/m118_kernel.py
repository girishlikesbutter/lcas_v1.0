#!/usr/bin/env python3
"""
m118 kernel — Geometric kernel builder for seed 14 (and any seed).

Materializes q_delta[direction_i, magnitude_j, epoch_k] so that multiple
alignment-cost variants can be scored post-hoc without re-propagating.

Constraint epochs = union(spec_peaks, tight_ipl_epochs), where
tight_ipl_epochs = { ep : loop_count[ep] >= 1 AND length[ep] < median(length over all 500 epochs) }.
This ensures score_ipl_centroid_weighted_extended can be scored from the
same kernel.

Anchor epoch rule: minimum IPL length over epochs with loop_count >= 1.

Output NPZ schema (see experiment spec): kernel.npz in
data/results/inversion_diagnostics/m118/seed_<s>/.

Usage:
  MICRO118_SEED=14 MICRO118_POOL_SIZE=24 python3 m118_kernel.py
"""

import sys
import os
import time
import json
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
IPL_NPZ = RESULTS_DIR / "isoshell_viewer" / "ipl_all_epochs.npz"

SEED = int(os.environ.get('MICRO118_SEED', '14'))
POOL_SIZE = int(os.environ.get('MICRO118_POOL_SIZE', '24'))
OUTPUT_DIR = Path(os.environ.get('MICRO118_OUTPUT_DIR',
                  str(RESULTS_DIR / "m118")))

N_DIRS = 2000
N_MAGS = 20
MAG_LO = 0.70
MAG_HI = 1.30

CKPT_DIR = OUTPUT_DIR / f"seed_{SEED:03d}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


_log_file = open(str(CKPT_DIR / "run.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ---- Verbatim copies from m103_hybrid --------------------------------
def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


_I_TENSOR_GLOBAL = None


def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
    """Copied verbatim from m103_hybrid.py (with I_tensor threaded)."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
                                   np.concatenate([[0.0], fwd_dt]),
                                   "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
                                   np.concatenate([[0.0], bwd_dt]),
                                   "tumbling", I_tensor)
        dq_c = dq[1:].copy()
        dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


# ---- Worker -----------------------------------------------------------------
_WORKER_STATE = {}


def _init_worker(omega_dirs, omega_mags, dt_constraints, I_tensor):
    _WORKER_STATE['omega_dirs'] = omega_dirs
    _WORKER_STATE['omega_mags'] = omega_mags
    _WORKER_STATE['dt_constraints'] = dt_constraints
    _WORKER_STATE['I_tensor'] = I_tensor


def _compute_direction_kernel(di):
    dirs = _WORKER_STATE['omega_dirs']
    mags = _WORKER_STATE['omega_mags']
    dt = _WORKER_STATE['dt_constraints']
    I = _WORKER_STATE['I_tensor']
    n_mags = len(mags)
    n_ep = len(dt)
    out = np.zeros((n_mags, n_ep, 4), dtype=np.float32)
    wd = dirs[di]
    for mi, m in enumerate(mags):
        omega_test = wd * m
        out[mi] = propagate_delta_qs(omega_test, dt, I).astype(np.float32)
    return di, out


# ---- Main ------------------------------------------------------------------
def main():
    t_global = time.time()
    print("=" * 70)
    print(f"m118 kernel — seed {SEED}")
    print(f"  N_DIRS={N_DIRS}  N_MAGS={N_MAGS}  POOL={POOL_SIZE}")
    print(f"  OUTPUT: {CKPT_DIR}")
    print("=" * 70)

    # --- Load trajectory master ---
    master = np.load(str(DATA_DIR / "m046_trajectories.npz"),
                     allow_pickle=True)
    obs_times = master['observation_times']
    pab_j2000 = master['pab_j2000']
    unique_normals = master['unique_normals']
    I_tensor = master['inertia_tensor']
    true_q0 = master['q0s'][SEED]
    true_omega0 = master['omega0s'][SEED]
    true_lc = master['mag_hifi'][SEED]

    rng = np.random.default_rng(42)
    observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

    # --- Spec-peak detection (copied verbatim from m103_hybrid) ---
    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = np.deg2rad(omega_est_dps)
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
    print(f"\nSpec peaks: {len(spec_peaks)} (|omega| est = {omega_est_dps:.3f} dps)")

    # --- Load IPL data ---
    ipl_npz = np.load(str(IPL_NPZ), allow_pickle=True)
    lengths = ipl_npz[f"s{SEED:03d}_lengths"]       # (500,)
    loop_counts = ipl_npz[f"s{SEED:03d}_loop_counts"]  # (500,)
    centroids = ipl_npz[f"s{SEED:03d}_centroids"]   # (500,) object array

    n_obs = len(lengths)
    assert n_obs == len(obs_times), f"IPL/obs mismatch: {n_obs} vs {len(obs_times)}"

    # --- Anchor epoch: cascade by ascending loop count, min-length within tier ---
    # Rationale: fewer loops = less anchor branching = stronger constraint. Within
    # a tier, pick min IPL length. Ignore epochs with length in the top quartile
    # (across all 500 epochs) even at favourable loop counts — they aren't
    # constraint-tight. Skip loop_count == 0 (no IPL exists).
    length_q75 = float(np.quantile(lengths[loop_counts >= 1], 0.75))
    anchor_trace = []
    anchor_epoch = -1
    max_lc = int(loop_counts.max())
    for tier_lc in range(1, max_lc + 1):
        tier_mask = (loop_counts == tier_lc) & (lengths < length_q75)
        n_tier = int(tier_mask.sum())
        n_tier_unfiltered = int((loop_counts == tier_lc).sum())
        if n_tier == 0:
            anchor_trace.append({
                'loop_count': tier_lc,
                'n_epochs_total': n_tier_unfiltered,
                'n_epochs_below_q75': 0,
                'min_length': None,
                'chosen': False,
            })
            continue
        tier_lengths = np.where(tier_mask, lengths, np.inf)
        chosen_ep = int(np.argmin(tier_lengths))
        anchor_trace.append({
            'loop_count': tier_lc,
            'n_epochs_total': n_tier_unfiltered,
            'n_epochs_below_q75': n_tier,
            'min_length': float(lengths[chosen_ep]),
            'chosen_epoch': chosen_ep,
            'chosen': True,
        })
        if anchor_epoch == -1:
            anchor_epoch = chosen_ep
            break  # take lowest-loop-count tier with a valid epoch

    if anchor_epoch == -1:
        # Fallback: global min length over loop_count >= 1 (no length filter)
        valid_anchor_mask = loop_counts >= 1
        if not np.any(valid_anchor_mask):
            raise RuntimeError(f"Seed {SEED}: no epoch with loop_count >= 1")
        length_masked = np.where(valid_anchor_mask, lengths, np.inf)
        anchor_epoch = int(np.argmin(length_masked))
        anchor_trace.append({
            'loop_count': int(loop_counts[anchor_epoch]),
            'fallback': True,
            'chosen_epoch': anchor_epoch,
            'chosen': True,
        })

    anchor_time = float(obs_times[anchor_epoch])
    anchor_ipl_length = float(lengths[anchor_epoch])
    anchor_loop_count = int(loop_counts[anchor_epoch])
    anchor_centroids_raw = centroids[anchor_epoch]
    anchor_centroids = np.array(anchor_centroids_raw, dtype=np.float32).reshape(-1, 3)
    n_anchor_centroids = anchor_centroids.shape[0]

    print(f"\nAnchor selection trace (length q75 = {length_q75:.4f}):")
    for t in anchor_trace:
        if t.get('fallback'):
            print(f"  FALLBACK: loop_count={t['loop_count']} ep={t['chosen_epoch']}")
        elif t['chosen']:
            print(f"  loop_count={t['loop_count']:>2}: {t['n_epochs_total']:>3} eps total, "
                  f"{t['n_epochs_below_q75']:>3} below q75, min length={t['min_length']:.4f} "
                  f"@ ep {t['chosen_epoch']} <-- CHOSEN")
        else:
            print(f"  loop_count={t['loop_count']:>2}: {t['n_epochs_total']:>3} eps total, "
                  f"{t['n_epochs_below_q75']:>3} below q75 (skip)")

    print(f"\nAnchor epoch: {anchor_epoch} (t={anchor_time:.1f}s)")
    print(f"  IPL length:  {anchor_ipl_length:.4f} rad")
    print(f"  loop_count:  {anchor_loop_count}")
    print(f"  centroids:   {n_anchor_centroids}")

    # --- Constraint epoch set: union(spec_peaks, tight_ipl_epochs) ---
    median_length = float(np.median(lengths))
    tight_mask = (loop_counts >= 1) & (lengths < median_length)
    tight_epochs = np.where(tight_mask)[0]
    union_eps = np.sort(np.unique(np.concatenate(
        [np.asarray(spec_peaks, dtype=int),
         np.asarray(tight_epochs, dtype=int)])))

    # Exclude anchor from constraint epochs
    constraint_epochs = union_eps[union_eps != anchor_epoch]
    n_constraint = len(constraint_epochs)

    print(f"\nConstraint epochs:")
    print(f"  spec_peaks:      {len(spec_peaks)}")
    print(f"  tight_ipl_eps:   {len(tight_epochs)} (median length = {median_length:.4f})")
    print(f"  union (ex-anchor): {n_constraint}")

    # --- Build omega grid ---
    omega_dirs = fibonacci_sphere(N_DIRS)
    omega_mags = omega_est_rad * np.linspace(MAG_LO, MAG_HI, N_MAGS)

    dt_constraints = obs_times[constraint_epochs] - anchor_time

    # --- Truth body-frame PAB at each constraint epoch ---
    # Propagate truth from t=0 to each constraint epoch, rotate J2000 PAB into body frame
    q_truth_all, _ = propagate_attitude(
        true_q0, true_omega0, obs_times, "tumbling", I_tensor)
    truth_pab_body_at_constraints = np.zeros((n_constraint, 3), dtype=np.float64)
    for i, ep in enumerate(constraint_epochs):
        q = q_truth_all[ep]
        # wxyz -> xyzw for scipy
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        truth_pab_body_at_constraints[i] = R @ pab_j2000[ep]

    # --- Build kernel in parallel over directions ---
    print(f"\n--- Building kernel: {N_DIRS} dirs x {N_MAGS} mags x {n_constraint} eps ---")
    t_kernel = time.time()

    q_delta = np.zeros((N_DIRS, N_MAGS, n_constraint, 4), dtype=np.float32)

    import multiprocessing as mp
    with mp.Pool(POOL_SIZE, initializer=_init_worker,
                 initargs=(omega_dirs, omega_mags, dt_constraints, I_tensor)) as pool:
        for di, block in pool.imap_unordered(_compute_direction_kernel,
                                             range(N_DIRS), chunksize=8):
            q_delta[di] = block

    t_kernel_elapsed = time.time() - t_kernel
    print(f"Kernel built in {t_kernel_elapsed:.1f}s")

    # --- Save kernel ---
    out_path = CKPT_DIR / "kernel.npz"
    np.savez_compressed(
        str(out_path),
        q_delta=q_delta,
        omega_dirs=omega_dirs.astype(np.float64),
        omega_mags=omega_mags.astype(np.float64),
        constraint_epochs=constraint_epochs.astype(np.int64),
        pab_j2000_at_constraints=pab_j2000[constraint_epochs].astype(np.float64),
        pab_anchor_j2000=pab_j2000[anchor_epoch].astype(np.float64),
        anchor_epoch=np.int64(anchor_epoch),
        anchor_centroids=anchor_centroids,
        anchor_ipl_length=np.float64(anchor_ipl_length),
        anchor_loop_count=np.int64(anchor_loop_count),
        anchor_time=np.float64(anchor_time),
        anchor_selection_trace=np.array(anchor_trace, dtype=object),
        anchor_length_q75=np.float64(length_q75),
        truth_q0=true_q0.astype(np.float64),
        truth_omega0=true_omega0.astype(np.float64),
        truth_pab_body_at_constraints=truth_pab_body_at_constraints,
        observed_lc=observed_lc.astype(np.float64),
        obs_times=obs_times.astype(np.float64),
        unique_normals=np.asarray(unique_normals, dtype=np.float64),
        median_length=np.float64(median_length),
        spec_peaks=np.asarray(spec_peaks, dtype=np.int64),
        tight_epochs=tight_epochs.astype(np.int64),
        seed=np.int64(SEED),
    )
    print(f"\nSaved: {out_path}")
    print(f"  q_delta shape: {q_delta.shape}  dtype={q_delta.dtype}  "
          f"size={q_delta.nbytes / 1024 / 1024:.1f} MB")

    # --- Summary JSON ---
    summary = {
        'seed': int(SEED),
        'anchor_epoch': int(anchor_epoch),
        'anchor_time_s': float(anchor_time),
        'anchor_ipl_length_rad': float(anchor_ipl_length),
        'anchor_loop_count': int(anchor_loop_count),
        'n_anchor_centroids': int(n_anchor_centroids),
        'n_spec_peaks': int(len(spec_peaks)),
        'n_tight_epochs': int(len(tight_epochs)),
        'n_constraint_epochs': int(n_constraint),
        'median_length_rad': float(median_length),
        'omega_est_dps': float(omega_est_dps),
        'N_DIRS': int(N_DIRS),
        'N_MAGS': int(N_MAGS),
        'mag_lo': float(MAG_LO),
        'mag_hi': float(MAG_HI),
        'kernel_build_time_s': float(t_kernel_elapsed),
        'total_time_s': float(time.time() - t_global),
    }
    tmp = CKPT_DIR / "kernel_summary.json.tmp"
    final = CKPT_DIR / "kernel_summary.json"
    with open(tmp, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, final)
    print(f"Saved: {final}")

    print(f"\nTotal time: {time.time() - t_global:.1f}s")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
    sys.stdout = sys.__stdout__
    _log_file.close()
