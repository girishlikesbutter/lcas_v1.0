#!/usr/bin/env python3
"""Micro-19 — Dense |ω| magnitude sweep to map ALL valid bridge solutions.

For both legs (using oracle attitudes at 3 peaks from micro13_stage1.npz):
- Sweep |ω| from 0.1 to 6.0 deg/s in 0.02 deg/s steps (~296 bins)
- At each magnitude, minimise arrival error over ω direction using Nelder-Mead
  in 2D spherical coordinates (theta, phi) with 3 random starts per bin
- Plot arrival_error vs |ω| — valleys ARE the valid winding solutions
- Diagnoses the leg-1 staircase gap (micro18): does a solution near 2.08 deg/s exist?
"""
import sys, time, json, numpy as np
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

SEED = 42
MAG_MIN, MAG_MAX, MAG_STEP = 0.1, 6.0, 0.02  # deg/s
N_STARTS = 3
N_WORKERS = 8
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

_LEG_DATA = {}


def _init_worker(leg_data):
    global _LEG_DATA
    _LEG_DATA = leg_data


def _angles_to_omega(theta, phi, mag_rad):
    """Spherical (theta, phi) + fixed magnitude → ω vector."""
    return mag_rad * np.array([np.sin(theta)*np.cos(phi),
                                np.sin(theta)*np.sin(phi),
                                np.cos(theta)])


def _solve_one_bin(args):
    """Worker: find best ω direction at fixed magnitude for one leg."""
    leg_key, mag_degs, seed = args
    q_start = _LEG_DATA[leg_key + '_qs']
    q_end = _LEG_DATA[leg_key + '_qe']
    dt = _LEG_DATA[leg_key + '_dt']
    I = _LEG_DATA['I']
    mag_rad = np.deg2rad(mag_degs)
    rng = np.random.RandomState(seed)

    def arrival_err_2d(angles):
        theta, phi = angles
        w = _angles_to_omega(theta, phi, mag_rad)
        qp, _ = propagate_attitude(q_start, w, np.array([0.0, dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        return float(1.0 - d * d)

    best_val, best_angles = 1e30, (0.0, 0.0)
    for _ in range(N_STARTS):
        theta0 = np.arccos(2.0 * rng.rand() - 1.0)  # uniform on sphere
        phi0 = 2.0 * np.pi * rng.rand()
        res = minimize(arrival_err_2d, [theta0, phi0], method='Nelder-Mead',
                       options={'maxiter': 60, 'xatol': 1e-6, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_angles = (res.x[0], res.x[1])

    w_best = _angles_to_omega(best_angles[0], best_angles[1], mag_rad)
    return {'target_mag': round(mag_degs, 4), 'arrival_err': float(best_val),
            'omega': w_best.tolist()}


if __name__ == '__main__':
    t0 = time.time()
    print("Setting up experiment ...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_TIMES = CTX.observation_times
    print(f"  Setup: {time.time()-t0:.1f}s", flush=True)

    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    times_prop = np.array([0.0] + [float(OBS_TIMES[p]) for p in PEAKS])
    q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_prop, "tumbling", I)
    q_peaks = [q_traj[i+1] for i in range(3)]

    dt_leg0 = float(OBS_TIMES[PEAKS[1]] - OBS_TIMES[PEAKS[0]])
    dt_leg1 = float(OBS_TIMES[PEAKS[2]] - OBS_TIMES[PEAKS[1]])
    true_omega_mag_degs = float(np.rad2deg(np.linalg.norm(CTX.true_omega0)))
    print(f"  Peaks: {PEAKS}, dt_leg0={dt_leg0:.1f}s, dt_leg1={dt_leg1:.1f}s")
    print(f"  True |ω| = {true_omega_mag_degs:.3f} deg/s", flush=True)

    mags = np.arange(MAG_MIN, MAG_MAX + MAG_STEP/2, MAG_STEP)
    n_bins = len(mags)
    print(f"  {n_bins} bins × 2 legs × {N_STARTS} starts = {n_bins*2*N_STARTS} optimisations",
          flush=True)

    leg_data = {
        'leg0_qs': q_peaks[0], 'leg0_qe': q_peaks[1], 'leg0_dt': dt_leg0,
        'leg1_qs': q_peaks[1], 'leg1_qe': q_peaks[2], 'leg1_dt': dt_leg1,
        'I': I,
    }

    tasks = []
    for leg in ['leg0', 'leg1']:
        for i, m in enumerate(mags):
            tasks.append((leg, float(m), SEED + i * 7 + (0 if leg == 'leg0' else 10000)))

    print("  Running sweep ...", flush=True)
    t1 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(leg_data,)) as pool:
        results_raw = pool.map(_solve_one_bin, tasks, chunksize=4)
    print(f"  Sweep done: {time.time()-t1:.1f}s", flush=True)

    results = {'leg0': results_raw[:n_bins], 'leg1': results_raw[n_bins:]}

    # Identify valleys (arrival_err < threshold)
    VALLEY_THRESH = 0.01  # slightly looser to catch near-misses
    summary = {}
    for leg in ['leg0', 'leg1']:
        errs = np.array([r['arrival_err'] for r in results[leg]])
        tgt_mags = np.array([r['target_mag'] for r in results[leg]])
        valley_mask = errs < VALLEY_THRESH
        valley_mags = tgt_mags[valley_mask]
        # Cluster valleys (group within 0.15 deg/s)
        clusters = []
        if len(valley_mags) > 0:
            sorted_m = np.sort(valley_mags)
            cluster = [sorted_m[0]]
            for m in sorted_m[1:]:
                if m - cluster[-1] < 0.15:
                    cluster.append(m)
                else:
                    clusters.append({'center': round(float(np.mean(cluster)), 3),
                                     'width': round(float(cluster[-1] - cluster[0]), 3),
                                     'n_bins': len(cluster)})
                    cluster = [m]
            clusters.append({'center': round(float(np.mean(cluster)), 3),
                             'width': round(float(cluster[-1] - cluster[0]), 3),
                             'n_bins': len(cluster)})
        centers = [c['center'] for c in clusters]
        summary[leg] = {
            'n_valleys': len(clusters),
            'valleys': clusters,
            'min_err': float(np.min(errs)),
            'near_true': any(abs(c - true_omega_mag_degs) < 0.3 for c in centers),
        }
        print(f"\n  {leg}: {len(clusters)} valleys")
        for c in clusters:
            near = " *** NEAR TRUE ***" if abs(c['center'] - true_omega_mag_degs) < 0.3 else ""
            print(f"    {c['center']:.3f} deg/s (width={c['width']:.3f}, bins={c['n_bins']}){near}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Micro-19: ω magnitude landscape (arrival error vs |ω|)", fontsize=13)

    for ax, leg, label in zip(axes, ['leg0', 'leg1'],
                               [f'Leg 0 (pk {PEAKS[0]}→{PEAKS[1]}, dt={dt_leg0:.0f}s)',
                                f'Leg 1 (pk {PEAKS[1]}→{PEAKS[2]}, dt={dt_leg1:.0f}s)']):
        errs = np.array([r['arrival_err'] for r in results[leg]])
        tgt_mags = np.array([r['target_mag'] for r in results[leg]])
        ax.semilogy(tgt_mags, np.clip(errs, 1e-16, None), 'b-', lw=0.6, alpha=0.8)
        ax.axhline(VALLEY_THRESH, color='gray', ls=':', lw=0.8, label=f'threshold={VALLEY_THRESH}')
        ax.axvline(true_omega_mag_degs, color='firebrick', ls='--', lw=1.5,
                   label=f'true |ω|={true_omega_mag_degs:.2f}')
        for c in summary[leg]['valleys']:
            ax.axvline(c['center'], color='green', ls=':', lw=1.0, alpha=0.7)
        ax.set_ylabel('Arrival error (log)')
        ax.set_title(label)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-16, 2)

    axes[1].set_xlabel('|ω| (deg/s)')
    plt.tight_layout()
    out_png = RESULTS_DIR / "micro19_winding_landscape.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    json_data = {
        'n_bins': n_bins, 'mag_range_degs': [MAG_MIN, MAG_MAX], 'mag_step_degs': MAG_STEP,
        'n_starts': N_STARTS, 'true_omega_mag_degs': round(true_omega_mag_degs, 5),
        'valley_threshold': VALLEY_THRESH,
        'peaks': PEAKS, 'dt_leg0_s': dt_leg0, 'dt_leg1_s': dt_leg1,
        'summary': summary,
        'leg0_errs': [round(r['arrival_err'], 10) for r in results['leg0']],
        'leg1_errs': [round(r['arrival_err'], 10) for r in results['leg1']],
        'runtime_s': round(time.time() - t0, 1),
    }
    save_results(RESULTS_DIR / "micro19_winding_landscape.json", json_data)
    print(f"Saved: {RESULTS_DIR / 'micro19_winding_landscape.json'}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
