#!/usr/bin/env python3
"""Micro-17 — Roberto's staircase omega: multi-winding bridge disambiguation.

For a fast tumbler bridging two attitude endpoints over ~554s, the minimum-|ω|
solver always returns the lowest winding number — which is wrong. This script
generates a staircase family (ω_1, ω_2, ...) by repeatedly finding the next
valid ω above the previous magnitude, then scores each family member against
the trough brightness. The correct winding should have the lowest trough error.
"""
import sys, time, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
N_STEPS = 8
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()

    print("Setting up experiment ...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_TIMES = CTX.observation_times
    print(f"  Setup done: {time.time()-t0:.1f}s", flush=True)

    # ── Load peaks from stage-1 checkpoint ──────────────────────────────────
    stage1 = np.load(RESULTS_DIR / "micro13_stage1.npz")
    PEAKS = [int(x) for x in stage1["peaks"]]   # [183, 260, 360]

    t_A = float(OBS_TIMES[PEAKS[0]])   # time from epoch 0 to peak 183
    t_B = float(OBS_TIMES[PEAKS[1]])   # time from epoch 0 to peak 260
    dt_AB = t_B - t_A
    print(f"  Peaks: {PEAKS},  dt_AB = {dt_AB:.1f}s", flush=True)

    # ── Get exact true attitudes at the two peaks ────────────────────────────
    times_prop = np.array([0.0, t_A, t_B])
    q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_prop,
                                   "tumbling", I)
    q_A_true = q_traj[1]   # wxyz at peak 183
    q_B_true = q_traj[2]   # wxyz at peak 260

    # ── True ω at peak A (reference for comparison) ──────────────────────────
    _, w_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0,
                                   np.array([0.0, t_A]), "tumbling", I)
    true_omega_at_A = w_traj[-1]
    true_omega_mag_degs = float(np.rad2deg(np.linalg.norm(true_omega_at_A)))
    print(f"  True |ω| at peak A: {true_omega_mag_degs:.3f} deg/s", flush=True)

    # ── Trough reference ─────────────────────────────────────────────────────
    # Trough = argmax of observed LC (magnitude, higher=dimmer) between peaks
    seg = CTX.observed_lc[PEAKS[0]:PEAKS[1] + 1]
    TROUGH_IDX = PEAKS[0] + int(np.argmax(seg))
    t_trough_from_A = float(OBS_TIMES[TROUGH_IDX] - OBS_TIMES[PEAKS[0]])
    print(f"  Trough idx: {TROUGH_IDX},  t_trough_from_A = {t_trough_from_A:.1f}s",
          flush=True)

    # Lo-fi reference brightness at trough using true trajectory
    times_ref = np.array([0.0, t_A, t_A + t_trough_from_A])
    q_ref_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_ref,
                                       "tumbling", I)
    lofi_ref = brightness_single_epoch(q_ref_traj[-1], TROUGH_IDX, CTX,
                                       use_shadows=False)
    print(f"  Lo-fi reference brightness at trough: {lofi_ref:.4f}", flush=True)

    # Scipy Rotation objects for endpoint comparison
    R_A = Rotation.from_quat([q_A_true[1], q_A_true[2], q_A_true[3], q_A_true[0]])
    R_B = Rotation.from_quat([q_B_true[1], q_B_true[2], q_B_true[3], q_B_true[0]])

    def arrival_error(w):
        """Geodesic distance (rad^2) between propagated arrival and q_B_true."""
        tb = np.array([0.0, dt_AB])
        qp, _ = propagate_attitude(q_A_true, w, tb, "tumbling", I)
        Rp = Rotation.from_quat([qp[-1][1], qp[-1][2], qp[-1][3], qp[-1][0]])
        d = np.clip(np.dot(qp[-1], q_B_true), -1.0, 1.0)
        return float(1.0 - d * d)

    def obj_with_barrier(w, lb_sq):
        """Arrival error + soft lower-bound barrier to avoid regressing."""
        barrier = max(0.0, lb_sq - float(np.dot(w, w))) ** 2 * 1e4
        return arrival_error(w) + barrier

    def trough_score(w):
        """Lo-fi brightness error at trough epoch when bridging from q_A_true."""
        times_t = np.array([0.0, t_trough_from_A])
        qp, _ = propagate_attitude(q_A_true, w, times_t, "tumbling", I)
        b = brightness_single_epoch(qp[-1], TROUGH_IDX, CTX, use_shadows=False)
        return float(abs(b - lofi_ref))

    # ── Staircase ────────────────────────────────────────────────────────────
    # Step 0 initial guess: minimum-winding rotvec / dt
    w0_rotvec = (R_A.inv() * R_B).as_rotvec() / dt_AB
    # Rotation axis for extra-revolution increments (unit vector along w0)
    w0_norm = float(np.linalg.norm(w0_rotvec))
    rot_axis = w0_rotvec / w0_norm if w0_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    delta_per_rev = (2.0 * np.pi / dt_AB) * rot_axis  # one extra full rotation

    print(f"\n{'─'*65}", flush=True)
    print(f"{'Step':>4}  {'|ω| deg/s':>10}  {'arr_err':>8}  {'trough_err':>11}", flush=True)
    print(f"{'─'*65}", flush=True)

    steps = []
    lb_sq = 0.0   # initial lower bound: 0 (allow any magnitude)
    w_prev = w0_rotvec.copy()

    for k in range(N_STEPS):
        ts = time.time()

        res = minimize(
            obj_with_barrier,
            w_prev,
            args=(lb_sq,),
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9},
        )
        w_k = res.x.copy()
        mag_k = float(np.linalg.norm(w_k))
        arr_err = float(arrival_error(w_k))
        tr_err = float(trough_score(w_k))
        mag_degs = float(np.rad2deg(mag_k))

        steps.append({
            'step': k,
            'omega_rad_s': w_k.tolist(),
            'mag_degs': round(mag_degs, 5),
            'arrival_err': round(arr_err, 8),
            'trough_err': round(tr_err, 6),
        })

        print(f"{k:>4}  {mag_degs:>10.3f}  {arr_err:>8.2e}  {tr_err:>11.5f}  "
              f"[{time.time()-ts:.1f}s]", flush=True)

        # Advance lower bound strictly above found solution
        lb_sq = mag_k ** 2 * 1.01
        # Initial guess for next step: add one full extra revolution along rotation axis
        w_prev = w_k + delta_per_rev

    print(f"{'─'*65}", flush=True)

    # ── Identify correct and best step ───────────────────────────────────────
    mags_degs = np.array([s['mag_degs'] for s in steps])
    trough_errs = np.array([s['trough_err'] for s in steps])
    arr_errs = np.array([s['arrival_err'] for s in steps])

    # Correct step: closest |ω| to true
    correct_step_idx = int(np.argmin(np.abs(mags_degs - true_omega_mag_degs)))
    best_step_by_trough = int(np.argmin(trough_errs))
    matched = (correct_step_idx == best_step_by_trough)

    print(f"\nTrue |ω|:           {true_omega_mag_degs:.3f} deg/s")
    print(f"Correct step:       {correct_step_idx}  "
          f"(|ω| = {mags_degs[correct_step_idx]:.3f} deg/s, "
          f"trough_err = {trough_errs[correct_step_idx]:.5f})")
    print(f"Best by trough:     {best_step_by_trough}  "
          f"(|ω| = {mags_degs[best_step_by_trough]:.3f} deg/s, "
          f"trough_err = {trough_errs[best_step_by_trough]:.5f})")
    print(f"Trough scoring {'IDENTIFIED' if matched else 'MISSED'} correct step.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    step_idx = np.arange(N_STEPS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Micro-17: Staircase ω family — true pair, lo-fi trough scoring",
                 fontsize=12)

    # Left: ω magnitude per step
    ax1.plot(step_idx, mags_degs, 'o-', color='steelblue', ms=7, lw=1.8,
             label='staircase |ω|')
    ax1.axhline(true_omega_mag_degs, color='firebrick', lw=1.5, ls='--',
                label=f'true |ω| = {true_omega_mag_degs:.2f} deg/s')
    ax1.axvline(correct_step_idx, color='firebrick', lw=1.0, ls=':',
                label=f'correct step = {correct_step_idx}')
    ax1.set_xlabel('Staircase step')
    ax1.set_ylabel('|ω| (deg/s)')
    ax1.set_title('ω family magnitude')
    ax1.legend(fontsize=8)
    ax1.set_xticks(step_idx)
    ax1.grid(True, alpha=0.3)

    # Right: trough error per step
    colors = ['gold' if i == best_step_by_trough else 'steelblue' for i in step_idx]
    ax2.bar(step_idx, trough_errs, color=colors, edgecolor='k', linewidth=0.6)
    ax2.axvline(correct_step_idx, color='firebrick', lw=1.5, ls='--',
                label=f'correct step = {correct_step_idx}')
    ax2.set_xlabel('Staircase step')
    ax2.set_ylabel('|predicted − reference| (mag)')
    ax2.set_title('Trough brightness error\n(gold = best)')
    ax2.legend(fontsize=8)
    ax2.set_xticks(step_idx)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro17_staircase_omega.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "micro17_staircase_omega.json"
    save_results(out_json, {
        'steps': steps,
        'true_omega_mag_degs': round(true_omega_mag_degs, 5),
        'best_step_by_trough': best_step_by_trough,
        'correct_step_idx': correct_step_idx,
        'trough_scoring_correct': bool(matched),
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Saved: {out_json}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
