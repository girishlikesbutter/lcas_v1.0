#!/usr/bin/env python3
"""Micro-40 — Bridge solver n_starts coverage sweep (leg 1).

Question: How many random starts per 0.5 dps band are needed on leg 1
(peaks 260->360, dt=721s) for the true omega to appear in the candidate set?

Method:
  - Use oracle q_B, q_C at peaks 260 and 360
  - Sweep n_starts = [10, 20, 50, 100, 200] per band
  - For each level, run the full band-sweep bridge solver with direction-aware dedup
  - Check: does the true omega appear? Report nearest candidate distance + magnitude error
  - Report: total candidates after dedup, wall time, band occupancy
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = None
if __name__ == '__main__' or __name__ == '__mp_main__':
    pass  # Tee setup deferred to main block to avoid issues in worker processes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARRIVAL_THRESH = 1e-6
MAG_TOL = 0.05         # dedup: magnitude tolerance (deg/s)
DIR_TOL = 5.0          # dedup: angular distance tolerance (degrees)
N_WORKERS = 8

_SHARED = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ang_dist_deg(a, b):
    """Angular distance between two vectors in degrees."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1., 1.))))


# ---------------------------------------------------------------------------
# Bridge solver (runs in worker processes)
# ---------------------------------------------------------------------------
def _init_worker(shared):
    global _SHARED
    _SHARED = shared


def _solve_band(args):
    """Solve bridge problem for one random start in one magnitude band."""
    w0, lb_rad, ub_rad = args
    qs = _SHARED['qs']
    qe = _SHARED['qe']
    dt = _SHARED['dt']
    I  = _SHARED['I']

    def obj(w):
        wn2 = float(np.dot(w, w))
        pen = (max(0., lb_rad**2 - wn2)**2 * 1e4
             + max(0., wn2 - ub_rad**2)**2 * 1e4)
        qp, _ = propagate_attitude(qs, w, np.array([0., dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], qe), -1., 1.)
        return (1. - d*d) + pen

    res = minimize(obj, w0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9})
    wk = res.x
    qp, _ = propagate_attitude(qs, wk, np.array([0., dt]), "tumbling", I)
    d = np.clip(np.dot(qp[-1], qe), -1., 1.)
    return {
        'omega': wk.tolist(),
        'mag_degs': float(np.rad2deg(np.linalg.norm(wk))),
        'arrival_err': float(1. - d*d),
    }


def run_band_sweep(q_start, q_end, dt, I, n_starts_per_band, rng_seed):
    """Run band-sweep bridge solve, return deduplicated candidates."""
    bands = [(0.5 * i, 0.5 * (i + 1)) for i in range(13)]  # 0-6.5 deg/s
    rng = np.random.RandomState(rng_seed)

    tasks = []
    for lb_d, ub_d in bands:
        lb_r = np.deg2rad(lb_d) + 1e-6
        ub_r = np.deg2rad(ub_d)
        for _ in range(n_starts_per_band):
            d = rng.randn(3)
            d /= np.linalg.norm(d)
            mag = lb_r + (ub_r - lb_r) * rng.rand()
            tasks.append((mag * d, lb_r, ub_r))

    shared = {'qs': q_start, 'qe': q_end, 'dt': dt, 'I': I}
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        raw = pool.map(_solve_band, tasks, chunksize=4)

    valid = sorted(
        [r for r in raw if r['arrival_err'] < ARRIVAL_THRESH],
        key=lambda r: r['mag_degs']
    )

    # Direction-aware dedup
    deduped = []
    for r in valid:
        w_r = np.array(r['omega'])
        is_dup = False
        for d in deduped:
            if abs(r['mag_degs'] - d['mag_degs']) < MAG_TOL:
                if _ang_dist_deg(w_r, np.array(d['omega'])) < DIR_TOL:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(r)

    return deduped, len(valid)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    # --- Stdout capture (must be in main block, not module level) ---
    LOG_PATH = RESULTS_DIR / "micro40_stdout.txt"
    _log_file = open(LOG_PATH, 'w')
    sys.stdout = Tee(sys.__stdout__, _log_file)

    t0_global = time.time()

    # --- Setup ---
    print("Setting up experiment...")
    CTX = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=42,
        true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00'
    )

    # Peak indices
    PEAKS = [183, 260, 360]

    # Propagate true attitude to get oracle quaternions at peaks
    OBS_T = CTX.observation_times
    times_to_peaks = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS])
    q_traj, om_traj = propagate_attitude(
        CTX.true_q0, CTX.true_omega0, times_to_peaks, "tumbling", CTX.inertia_tensor
    )

    # Leg 1: q_B -> q_C (peaks 260 -> 360)
    q_B = q_traj[2]          # quaternion at peak 260
    q_C = q_traj[3]          # quaternion at peak 360
    omega_at_B = om_traj[2]  # true omega at peak B (rad/s)
    dt_leg1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])

    true_omega_mag_dps = float(np.rad2deg(np.linalg.norm(omega_at_B)))
    true_omega_dir = omega_at_B / np.linalg.norm(omega_at_B)

    print(f"Setup complete in {time.time() - t0_global:.1f}s")
    print(f"Peaks: {PEAKS}")
    print(f"dt_leg1 = {dt_leg1:.1f}s")
    print(f"True omega at B: {np.rad2deg(omega_at_B)} deg/s")
    print(f"True |omega| = {true_omega_mag_dps:.4f} deg/s")

    # --- Sweep over n_starts levels ---
    N_STARTS_SWEEP = [10, 20, 50, 100, 200]
    sweep_results = []

    # Storage for NPZ: omega vectors for each n_starts level
    npz_data = {}

    for n_starts in N_STARTS_SWEEP:
        print(f"\n{'='*60}")
        print(f"Running n_starts = {n_starts} per band...")
        t_start = time.time()

        candidates, n_valid_before_dedup = run_band_sweep(
            q_B, q_C, dt_leg1, CTX.inertia_tensor,
            n_starts_per_band=n_starts,
            rng_seed=42  # Same seed so n_starts=20 includes all starts from n_starts=10
        )

        wall_time = time.time() - t_start
        n_candidates = len(candidates)

        # Find nearest candidate to true omega
        nearest_dist = 180.0
        nearest_mag_err = 999.0
        nearest_idx = -1
        for i, c in enumerate(candidates):
            w = np.array(c['omega'])
            dist = _ang_dist_deg(w, omega_at_B)
            mag_err = abs(c['mag_degs'] - true_omega_mag_dps)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_mag_err = mag_err
                nearest_idx = i

        # Count direction families per band (for diagnostics)
        band_counts = {}
        for c in candidates:
            band_idx = int(c['mag_degs'] / 0.5)
            band_counts[band_idx] = band_counts.get(band_idx, 0) + 1

        result = {
            'n_starts': n_starts,
            'n_total_solves': n_starts * 13,
            'n_valid_before_dedup': n_valid_before_dedup,
            'n_candidates_after_dedup': n_candidates,
            'nearest_angular_dist_deg': nearest_dist,
            'nearest_mag_error_dps': nearest_mag_err,
            'nearest_idx': nearest_idx,
            'wall_time_s': wall_time,
            'band_counts': {str(k): v for k, v in band_counts.items()},
            'true_found': nearest_dist < 5.0,  # within 5 deg = "found"
        }
        sweep_results.append(result)

        # Save omega vectors for this n_starts level
        omega_array = np.array([c['omega'] for c in candidates])
        npz_data[f'omegas_n{n_starts}'] = omega_array
        npz_data[f'mags_n{n_starts}'] = np.array([c['mag_degs'] for c in candidates])

        print(f"  Candidates: {n_candidates} (from {n_valid_before_dedup} valid)")
        print(f"  Nearest to truth: {nearest_dist:.2f} deg, mag_err={nearest_mag_err:.4f} dps")
        print(f"  True found (< 5 deg): {nearest_dist < 5.0}")
        print(f"  Wall time: {wall_time:.1f}s")
        print(f"  Band occupancy: {band_counts}")

    total_time = time.time() - t0_global

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"=== micro40 SUMMARY ===")
    print(f"True omega at B: {np.rad2deg(omega_at_B)} deg/s, |omega|={true_omega_mag_dps:.4f} dps")
    print(f"dt_leg1 = {dt_leg1:.1f}s")
    print(f"")
    print(f"{'n_starts':>10}  {'candidates':>12}  {'nearest_deg':>12}  {'mag_err':>10}  {'found':>6}  {'time_s':>8}")
    for r in sweep_results:
        print(f"{r['n_starts']:>10}  {r['n_candidates_after_dedup']:>12}  "
              f"{r['nearest_angular_dist_deg']:>12.2f}  {r['nearest_mag_error_dps']:>10.4f}  "
              f"{str(r['true_found']):>6}  {r['wall_time_s']:>8.1f}")
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"{'='*60}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Micro-40: Bridge solver n_starts coverage sweep (leg 1)", fontsize=12)

    n_starts_vals = [r['n_starts'] for r in sweep_results]
    nearest_dists = [r['nearest_angular_dist_deg'] for r in sweep_results]
    n_candidates_vals = [r['n_candidates_after_dedup'] for r in sweep_results]
    wall_times = [r['wall_time_s'] for r in sweep_results]

    # Panel 1: Nearest-to-truth angular distance vs n_starts
    ax1.plot(n_starts_vals, nearest_dists, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=5.0, color='r', linestyle='--', linewidth=1.5, label='"found" threshold (5 deg)')
    ax1.set_xlabel('n_starts per band')
    ax1.set_ylabel('Nearest candidate angular distance (deg)')
    ax1.set_title('Distance to true omega')
    ax1.set_xscale('log')
    ax1.set_xticks(n_starts_vals)
    ax1.set_xticklabels([str(n) for n in n_starts_vals])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Color points by found/not found
    for i, r in enumerate(sweep_results):
        color = 'green' if r['true_found'] else 'red'
        ax1.plot(r['n_starts'], r['nearest_angular_dist_deg'], 'o', color=color,
                 markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)

    # Panel 2: Number of candidates + wall time
    color_left = 'steelblue'
    color_right = 'darkorange'

    ax2.bar(range(len(n_starts_vals)), n_candidates_vals, color=color_left, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('n_starts per band')
    ax2.set_ylabel('Unique candidates after dedup', color=color_left)
    ax2.set_title('Candidates & wall time')
    ax2.set_xticks(range(len(n_starts_vals)))
    ax2.set_xticklabels([str(n) for n in n_starts_vals])
    ax2.tick_params(axis='y', labelcolor=color_left)
    ax2.grid(True, alpha=0.3, axis='y')

    ax2r = ax2.twinx()
    ax2r.plot(range(len(n_starts_vals)), wall_times, 'o-', color=color_right,
              linewidth=2, markersize=8)
    ax2r.set_ylabel('Wall time (s)', color=color_right)
    ax2r.tick_params(axis='y', labelcolor=color_right)

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro40_bridge_nstarts_sweep.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    # --- Save JSON ---
    json_results = {
        'experiment': 'micro40',
        'description': 'Bridge solver n_starts coverage sweep — leg 1 (peaks 260->360)',
        'peaks': PEAKS,
        'dt_leg1_s': round(dt_leg1, 2),
        'true_omega_at_B_rad_s': omega_at_B.tolist(),
        'true_omega_at_B_deg_s': np.rad2deg(omega_at_B).tolist(),
        'true_omega_mag_dps': round(true_omega_mag_dps, 6),
        'dedup': {'mag_tol_dps': MAG_TOL, 'dir_tol_deg': DIR_TOL},
        'arrival_thresh': ARRIVAL_THRESH,
        'n_bands': 13,
        'band_width_dps': 0.5,
        'sweep': sweep_results,
        'total_runtime_s': round(total_time, 1),
    }
    out_json = RESULTS_DIR / "micro40_bridge_nstarts_sweep.json"
    save_results(out_json, json_results)
    print(f"Saved: {out_json}")

    # --- Save NPZ ---
    npz_data['true_omega_at_B'] = omega_at_B
    npz_data['n_starts_sweep'] = np.array(N_STARTS_SWEEP)
    npz_data['nearest_dists'] = np.array(nearest_dists)
    npz_data['n_candidates'] = np.array(n_candidates_vals)
    npz_data['wall_times'] = np.array(wall_times)
    out_npz = RESULTS_DIR / "micro40_bridge_nstarts_sweep.npz"
    np.savez(out_npz, **npz_data)
    print(f"Saved: {out_npz}")

    _log_file.close()
