#!/usr/bin/env python3
"""Micro-28 — Integration pipeline with nudged attitudes (1, 2, 5 deg).

Strategy: Run ONE full tumbling bridge sweep with oracle attitudes to find all
winding candidates. Then for each nudged trial, warm-start refine each candidate
with the nudged endpoints (short maxiter) and apply L-conservation scoring.
"""
import sys, time, numpy as np, multiprocessing as mp
from pathlib import Path
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# --- Configuration ---
SEED = 42
N_STARTS = 10           # random starts per band in oracle sweep
N_WORKERS = 8
TRUE_MAG = 2.083266665599966   # |omega_true| in deg/s
ARRIVAL_THRESH = 1e-6
DEDUP_TOL = 0.05               # deg/s
NUDGE_LEVELS = [1.0, 2.0, 5.0]
N_TRIALS = 10
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

_mp_ctx = mp.get_context('fork')
_SHARED = {}


# --- Helper functions ---

def compute_L(q_wxyz, omega_body, I):
    """L_inertial = R(q) @ (I @ omega)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    return R @ (I @ omega_body)


def nudge_quaternion(q_wxyz, angle_deg, rng):
    """Apply a random rotation of given magnitude to a quaternion."""
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    half = np.deg2rad(angle_deg) / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) * axis)])
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = q_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def dir_err_deg(a, b):
    """Angle between two vectors in degrees."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return float(np.rad2deg(np.arccos(np.clip(cos, -1, 1))))


# --- Oracle bridge sweep (full, done once) ---

def _init_worker(shared):
    global _SHARED
    _SHARED = shared


def _solve_band(args):
    """Full tumbling bridge solve for oracle sweep."""
    w0, lb_rad, ub_rad = args
    qs, qe, dt, I = _SHARED['qs'], _SHARED['qe'], _SHARED['dt'], _SHARED['I']
    def obj(w):
        wn2 = float(np.dot(w, w))
        pen = max(0., lb_rad**2 - wn2)**2 * 1e4 + max(0., wn2 - ub_rad**2)**2 * 1e4
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
        'arrival_err': float(1. - d*d)
    }


def oracle_sweep(q_start, q_end, dt, I, rng_seed):
    """Full 13-band x N_STARTS bridge sweep with Pool. Returns deduplicated candidates."""
    bands = [(0.5 * i, 0.5 * (i + 1)) for i in range(13)]
    rng = np.random.RandomState(rng_seed)
    tasks = []
    for lb_d, ub_d in bands:
        lb_r = np.deg2rad(lb_d) + 1e-6
        ub_r = np.deg2rad(ub_d)
        for _ in range(N_STARTS):
            d = rng.randn(3); d /= np.linalg.norm(d)
            tasks.append(((lb_r + (ub_r - lb_r) * rng.rand()) * d, lb_r, ub_r))
    shared = {'qs': q_start, 'qe': q_end, 'dt': dt, 'I': I}
    with _mp_ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        raw = pool.map(_solve_band, tasks, chunksize=4)
    valid = sorted(
        [r for r in raw if r['arrival_err'] < ARRIVAL_THRESH],
        key=lambda r: r['mag_degs']
    )
    deduped = []
    for r in valid:
        if not deduped or abs(r['mag_degs'] - deduped[-1]['mag_degs']) > DEDUP_TOL:
            deduped.append(r)
    return deduped


# --- Nudged refinement (short, warm-started from oracle) ---

def _refine_nudged(args):
    """Short tumbling refinement with nudged endpoints, warm-started from oracle."""
    w0, lb_rad, ub_rad = args
    qs, qe, dt, I = _SHARED['qs'], _SHARED['qe'], _SHARED['dt'], _SHARED['I']
    def obj(w):
        wn2 = float(np.dot(w, w))
        pen = max(0., lb_rad**2 - wn2)**2 * 1e4 + max(0., wn2 - ub_rad**2)**2 * 1e4
        qp, _ = propagate_attitude(qs, w, np.array([0., dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], qe), -1., 1.)
        return (1. - d*d) + pen
    res = minimize(obj, w0, method='L-BFGS-B',
                   options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-9})
    wk = res.x
    qp, _ = propagate_attitude(qs, wk, np.array([0., dt]), "tumbling", I)
    d = np.clip(np.dot(qp[-1], qe), -1., 1.)
    return {
        'omega': wk.tolist(),
        'mag_degs': float(np.rad2deg(np.linalg.norm(wk))),
        'arrival_err': float(1. - d*d)
    }


def refine_leg(oracle_cands, q_start_nudged, q_end_nudged, dt, I):
    """Refine oracle candidates with nudged endpoints (short maxiter, parallel)."""
    if not oracle_cands:
        return []
    tasks = []
    for c in oracle_cands:
        w = np.array(c['omega'])
        mag = np.linalg.norm(w)
        # Band bounds: keep original band (0.5 deg/s wide)
        band_idx = int(np.rad2deg(mag) / 0.5)
        lb = np.deg2rad(band_idx * 0.5) + 1e-6
        ub = np.deg2rad((band_idx + 1) * 0.5)
        tasks.append((w, lb, ub))
    shared = {'qs': q_start_nudged, 'qe': q_end_nudged, 'dt': dt, 'I': I}
    with _mp_ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        raw = pool.map(_refine_nudged, tasks, chunksize=1)
    # Keep converged results, deduplicate
    valid = sorted(
        [r for r in raw if r['arrival_err'] < ARRIVAL_THRESH],
        key=lambda r: r['mag_degs']
    )
    out = []
    for r in valid:
        if not out or abs(r['mag_degs'] - out[-1]['mag_degs']) > DEDUP_TOL:
            out.append(r)
    return out


# --- Trial execution ---

def run_trial(nudge_deg, trial_idx, oracle_leg0, oracle_leg1,
              q_A, q_B, q_C, om_A, dt_0, dt_1, I):
    rng = np.random.RandomState(SEED + trial_idx * 100 + int(nudge_deg * 10))

    # Apply independent random nudges
    qA_n = nudge_quaternion(q_A, nudge_deg, rng)
    qB_n = nudge_quaternion(q_B, nudge_deg, rng)
    qC_n = nudge_quaternion(q_C, nudge_deg, rng)

    # Refine oracle candidates with nudged endpoints
    leg0 = refine_leg(oracle_leg0, qA_n, qB_n, dt_0, I)
    leg1 = refine_leg(oracle_leg1, qB_n, qC_n, dt_1, I)
    n0, n1 = len(leg0), len(leg1)

    if n0 == 0 or n1 == 0:
        return dict(nudge_deg=nudge_deg, trial_idx=trial_idx,
                    n_cands_leg0=n0, n_cands_leg1=n1,
                    true_pair_rank=-1, L_gap=0.,
                    omega_mag_error_degs=99., omega_dir_error_deg=99.,
                    correct_winding=False)

    # L-conservation at junction B
    L_leg0 = []
    for c in leg0:
        _, om_prop = propagate_attitude(
            qA_n, np.array(c['omega']), np.array([0., dt_0]), "tumbling", I)
        L_leg0.append(compute_L(qB_n, om_prop[-1], I))
    L_leg1 = [compute_L(qB_n, np.array(c['omega']), I) for c in leg1]

    L_err = np.array([
        [np.linalg.norm(L_leg0[k] - L_leg1[j]) for j in range(n1)]
        for k in range(n0)
    ])
    flat = L_err.ravel()
    best_k, best_j = divmod(int(flat.argmin()), n1)

    true_k = int(np.argmin([abs(c['mag_degs'] - TRUE_MAG) for c in leg0]))
    true_j = int(np.argmin([abs(c['mag_degs'] - TRUE_MAG) for c in leg1]))
    ranks = flat.argsort().argsort()
    true_rank = int(ranks[true_k * n1 + true_j]) + 1
    sorted_flat = np.sort(flat)
    gap = float(sorted_flat[1] - sorted_flat[0]) if len(sorted_flat) > 1 else 0.
    correct = (best_k == true_k and best_j == true_j)

    best_om0 = np.array(leg0[best_k]['omega'])
    mag_err = abs(np.rad2deg(np.linalg.norm(best_om0)) - TRUE_MAG)
    d_err = dir_err_deg(best_om0, om_A)

    return {
        'nudge_deg': nudge_deg, 'trial_idx': trial_idx,
        'n_cands_leg0': n0, 'n_cands_leg1': n1,
        'true_pair_rank': true_rank, 'L_gap': round(gap, 8),
        'omega_mag_error_degs': round(mag_err, 6),
        'omega_dir_error_deg': round(d_err, 4),
        'correct_winding': correct
    }


# --- Main ---

if __name__ == '__main__':
    t0 = time.time()

    CTX = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times

    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "m013_stage1.npz")["peaks"]]
    times_p = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS])
    q_traj, om_traj = propagate_attitude(
        CTX.true_q0, CTX.true_omega0, times_p, "tumbling", I)
    q_A, q_B, q_C = q_traj[1], q_traj[2], q_traj[3]
    om_A = om_traj[1]
    dt_0 = float(OBS_T[PEAKS[1]] - OBS_T[PEAKS[0]])
    dt_1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])
    print(f"Setup ({time.time()-t0:.1f}s). Peaks {PEAKS}, dt0={dt_0:.1f}s dt1={dt_1:.1f}s")

    # Phase 1: Oracle sweep (one time, full tumbling bridge)
    print("Running oracle sweep (leg 0)...", flush=True)
    oracle_leg0 = oracle_sweep(q_A, q_B, dt_0, I, SEED)
    print(f"  Leg 0: {len(oracle_leg0)} candidates ({time.time()-t0:.1f}s)", flush=True)
    print("Running oracle sweep (leg 1)...", flush=True)
    oracle_leg1 = oracle_sweep(q_B, q_C, dt_1, I, SEED + 1000)
    print(f"  Leg 1: {len(oracle_leg1)} candidates ({time.time()-t0:.1f}s)", flush=True)

    # Phase 2: Nudged trials (warm-started from oracle)
    all_results = []
    for nudge in NUDGE_LEVELS:
        for ti in range(N_TRIALS):
            r = run_trial(nudge, ti, oracle_leg0, oracle_leg1,
                          q_A, q_B, q_C, om_A, dt_0, dt_1, I)
            all_results.append(r)
            tag = "OK" if r['correct_winding'] else "MISS"
            print(f"  {nudge:.0f}d t={ti} [{tag}] rk={r['true_pair_rank']} "
                  f"gap={r['L_gap']:.2e} |w|e={r['omega_mag_error_degs']:.4f} "
                  f"dir={r['omega_dir_error_deg']:.2f}d ({time.time()-t0:.0f}s)",
                  flush=True)

    # Summary
    runtime = time.time() - t0
    print(f"\n{'='*60}\n=== m028 SUMMARY ===")
    for n in NUDGE_LEVELS:
        sub = [r for r in all_results if r['nudge_deg'] == n]
        nc = sum(r['correct_winding'] for r in sub)
        gaps = [r['L_gap'] for r in sub if r['correct_winding']]
        merrs = [r['omega_mag_error_degs'] for r in sub]
        derrs = [r['omega_dir_error_deg'] for r in sub]
        mg = f"{np.median(gaps):.2e}" if gaps else "N/A"
        print(f"Nudge {n:.1f} deg: {nc}/{N_TRIALS} correct, "
              f"median gap = {mg} kg*m^2/s, "
              f"median |w| err = {np.median(merrs):.4f} dps, "
              f"median dir err = {np.median(derrs):.2f} deg")
    print(f"Runtime: {runtime:.1f}s\n{'='*60}")

    # Save
    save_results(RESULTS_DIR / "m028_integration_nudged.json", {
        'trials': all_results,
        'nudge_levels': NUDGE_LEVELS,
        'n_trials': N_TRIALS,
        'n_oracle_leg0': len(oracle_leg0),
        'n_oracle_leg1': len(oracle_leg1),
        'runtime_s': round(runtime, 1)
    })

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Micro-28: Integration pipeline with nudged attitudes", fontsize=13)

    pcts = [sum(r['correct_winding'] for r in all_results if r['nudge_deg'] == n) / N_TRIALS
            for n in NUDGE_LEVELS]
    axes[0, 0].bar([str(n) for n in NUDGE_LEVELS], pcts, color='steelblue', edgecolor='k')
    axes[0, 0].set(ylabel='P(correct)', xlabel='Nudge (deg)',
                   ylim=(0, 1.1), title='Winding selection accuracy')
    for i, p in enumerate(pcts):
        axes[0, 0].text(i, p + 0.03, f"{p:.0%}", ha='center', fontsize=10)

    gap_data = [[r['L_gap'] for r in all_results if r['nudge_deg'] == n]
                for n in NUDGE_LEVELS]
    axes[0, 1].boxplot(gap_data, tick_labels=[str(n) for n in NUDGE_LEVELS])
    axes[0, 1].set(ylabel='L-gap (kg*m^2/s)', xlabel='Nudge (deg)',
                   title='L-gap (best vs next-best)', yscale='log')

    for ax, key, ylabel in [
        (axes[1, 0], 'omega_mag_error_degs', '|omega| error (deg/s)'),
        (axes[1, 1], 'omega_dir_error_deg', 'Direction error (deg)')
    ]:
        for n in NUDGE_LEVELS:
            vals = [r[key] for r in all_results if r['nudge_deg'] == n]
            ax.scatter([n] * len(vals), vals, alpha=0.7, s=40,
                       edgecolors='k', lw=0.5)
        ax.set(xlabel='Nudge (deg)', ylabel=ylabel, title=ylabel)

    plt.tight_layout()
    out_png = RESULTS_DIR / "m028_integration_nudged.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {RESULTS_DIR / 'm028_integration_nudged.json'}")
