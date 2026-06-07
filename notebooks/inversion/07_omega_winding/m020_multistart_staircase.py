#!/usr/bin/env python3
"""Micro-20 — Multi-start staircase to find missed winding solutions on leg 1.

m018 showed leg 1 (peaks 260→360) has a huge gap from 0.25 to 3.28 deg/s
in the staircase — the true ω (~2.08 deg/s) was never found. This script:
- Runs the staircase with 10 random initial guesses per winding band
- Uses 0.5 deg/s bands (or m019 valleys if available) as targets
- Compares: which solutions does multi-start find that the original missed?
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
N_STARTS_PER_BAND = 10
N_WORKERS = 8
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

_SHARED = {}


def _init_worker(shared):
    global _SHARED
    _SHARED = shared


def _solve_band(args):
    """Find best ω in a magnitude band [lb, ub] from a given initial guess."""
    w0, lb_rad, ub_rad, seed = args
    q_start, q_end, dt, I = _SHARED['qs'], _SHARED['qe'], _SHARED['dt'], _SHARED['I']

    def objective(w):
        wn2 = float(np.dot(w, w))
        lb_bar = max(0.0, lb_rad**2 - wn2) ** 2 * 1e4
        ub_bar = max(0.0, wn2 - ub_rad**2) ** 2 * 1e4
        qp, _ = propagate_attitude(q_start, w, np.array([0.0, dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        return (1.0 - d * d) + lb_bar + ub_bar

    res = minimize(objective, w0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9})
    w_k = res.x
    mag_rad = float(np.linalg.norm(w_k))

    # Pure arrival error
    qp, _ = propagate_attitude(q_start, w_k, np.array([0.0, dt]), "tumbling", I)
    d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
    arr_err = float(1.0 - d * d)
    return {'omega': w_k.tolist(), 'mag_degs': float(np.rad2deg(mag_rad)),
            'arrival_err': arr_err}


if __name__ == '__main__':
    t0 = time.time()
    print("Setting up experiment ...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_TIMES = CTX.observation_times
    print(f"  Setup: {time.time()-t0:.1f}s", flush=True)

    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "m013_stage1.npz")["peaks"]]
    times_prop = np.array([0.0] + [float(OBS_TIMES[p]) for p in PEAKS])
    q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_prop, "tumbling", I)
    q_mid, q_B = q_traj[2], q_traj[3]

    dt_leg1 = float(OBS_TIMES[PEAKS[2]] - OBS_TIMES[PEAKS[1]])
    true_omega_mag_degs = float(np.rad2deg(np.linalg.norm(CTX.true_omega0)))
    delta_per_rev = 2.0 * np.pi / dt_leg1  # rad/s per extra revolution
    print(f"  Leg 1: peaks {PEAKS[1]}→{PEAKS[2]}, dt={dt_leg1:.1f}s")
    print(f"  True |ω| = {true_omega_mag_degs:.3f} deg/s")
    print(f"  delta_per_rev = {np.rad2deg(delta_per_rev):.3f} deg/s", flush=True)

    # Define magnitude bands: 0.5 deg/s wide, from 0 to 6 deg/s
    band_edges = np.arange(0, 6.5, 0.5)  # 13 bands
    bands = [(band_edges[i], band_edges[i+1]) for i in range(len(band_edges)-1)]
    print(f"  {len(bands)} bands × {N_STARTS_PER_BAND} starts = {len(bands)*N_STARTS_PER_BAND} tasks",
          flush=True)

    shared = {'qs': q_mid, 'qe': q_B, 'dt': dt_leg1, 'I': I}
    rng = np.random.RandomState(SEED)

    # Build tasks: for each band, generate N_STARTS random initial guesses
    tasks = []
    for i, (lb_degs, ub_degs) in enumerate(bands):
        lb_rad = np.deg2rad(lb_degs) + 1e-6
        ub_rad = np.deg2rad(ub_degs)
        mid_rad = (lb_rad + ub_rad) / 2
        for j in range(N_STARTS_PER_BAND):
            # Random direction, magnitude uniformly in [lb, ub]
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            mag = lb_rad + (ub_rad - lb_rad) * rng.rand()
            w0 = mag * direction
            tasks.append((w0, lb_rad, ub_rad, SEED + i * 100 + j))

    with Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        results_raw = pool.map(_solve_band, tasks, chunksize=4)

    # Group by band and find best per band
    ARRIVAL_THRESH = 0.001
    band_results = []
    for i, (lb_degs, ub_degs) in enumerate(bands):
        band_res = results_raw[i * N_STARTS_PER_BAND: (i + 1) * N_STARTS_PER_BAND]
        # Filter valid solutions (arrival_err < threshold)
        valid = [r for r in band_res if r['arrival_err'] < ARRIVAL_THRESH]
        best = min(band_res, key=lambda r: r['arrival_err'])
        band_results.append({
            'band_degs': [lb_degs, ub_degs],
            'n_valid': len(valid),
            'best_mag_degs': round(best['mag_degs'], 4),
            'best_arrival_err': best['arrival_err'],
            'valid_mags': sorted([round(r['mag_degs'], 4) for r in valid]),
        })
        status = f"{len(valid)}/{N_STARTS_PER_BAND} valid" if valid else "NO SOLUTION"
        print(f"  Band [{lb_degs:.1f}, {ub_degs:.1f}]: {status}"
              f"  best |ω|={best['mag_degs']:.3f}, err={best['arrival_err']:.2e}", flush=True)

    # Compare with m018 staircase
    m018_mags = [0.249, 3.279, 3.509, 3.995, 4.486, 4.980, 5.477, 5.976]  # from JSON
    found_mags = sorted(set(round(r['best_mag_degs'], 1)
                            for r in band_results if r['n_valid'] > 0))
    missed_by_staircase = [m for m in found_mags
                           if not any(abs(m - s) < 0.3 for s in m018_mags)]
    found_near_true = any(abs(r['best_mag_degs'] - true_omega_mag_degs) < 0.3
                          for r in band_results if r['n_valid'] > 0)

    print(f"\n{'='*60}")
    print(f"m018 staircase found:  {m018_mags}")
    print(f"Multi-start found:        {found_mags}")
    print(f"Missed by staircase:      {missed_by_staircase}")
    print(f"Solution near true ω:     {found_near_true}")
    print(f"{'='*60}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Micro-20: Multi-start staircase — Leg 1 winding recovery", fontsize=13)

    for i, br in enumerate(band_results):
        lb, ub = br['band_degs']
        color = 'green' if br['n_valid'] > 0 else 'lightcoral'
        ax.barh(i, br['n_valid'], color=color, edgecolor='k', lw=0.5,
                height=0.7, alpha=0.8)
        if br['n_valid'] > 0:
            ax.text(br['n_valid'] + 0.2, i, f"|ω|={br['best_mag_degs']:.2f}",
                    va='center', fontsize=7)

    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([f"[{b[0]:.1f}, {b[1]:.1f})" for b in bands], fontsize=8)
    ax.set_xlabel(f'Valid solutions (of {N_STARTS_PER_BAND} starts)')
    ax.set_ylabel('Magnitude band (deg/s)')

    # Mark true omega band
    true_band = int(true_omega_mag_degs / 0.5)
    ax.axhline(true_band, color='firebrick', ls='--', lw=1.5,
               label=f'true |ω|={true_omega_mag_degs:.2f} deg/s')

    # Mark m018 staircase solutions
    for sm in m018_mags:
        band_idx = min(int(sm / 0.5), len(bands) - 1)
        ax.plot(-0.3, band_idx, 'b^', ms=8)
    ax.plot([], [], 'b^', ms=8, label='m018 staircase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out_png = RESULTS_DIR / "m020_multistart_staircase.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    save_results(RESULTS_DIR / "m020_multistart_staircase.json", {
        'leg': 1, 'peaks': [PEAKS[1], PEAKS[2]], 'dt_s': dt_leg1,
        'n_starts_per_band': N_STARTS_PER_BAND,
        'true_omega_mag_degs': round(true_omega_mag_degs, 5),
        'm018_staircase_mags': m018_mags,
        'bands': band_results,
        'found_near_true': found_near_true,
        'missed_by_staircase': missed_by_staircase,
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Saved: {RESULTS_DIR / 'm020_multistart_staircase.json'}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
