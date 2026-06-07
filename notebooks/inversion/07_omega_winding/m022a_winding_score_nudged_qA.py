#!/usr/bin/env python3
"""Micro-22a — Winding score robustness: nudge q_A only (no re-staircasing).

Uses m017's original 8 omega solutions (found from true endpoints) and just
nudges q_A when propagating/scoring. This avoids staircase cost entirely.
Tests whether correct winding rank degrades with imperfect start attitude.

Given m021's finding that even oracle q_A gives rank=3/8, this script
tests whether the ranking is at least STABLE under small perturbations.
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
from scipy.spatial.transform import Rotation

from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
NUDGE_DEGS = [1, 3, 5]
N_DIRS = 5
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

def nudge_quaternion(q_wxyz, angle_deg, rng):
    """Apply a random rotation of given magnitude to a quaternion."""
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    R_nudge = Rotation.from_rotvec(axis * angle_rad)
    R_orig = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    R_new = R_nudge * R_orig
    xyzw = R_new.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


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

    # ── Load peaks and staircase ──────────────────────────────────────────
    stage1 = np.load(RESULTS_DIR / "m013_stage1.npz")
    PEAKS = [int(x) for x in stage1["peaks"]]
    pk_A, pk_B = PEAKS[0], PEAKS[1]

    with open(RESULTS_DIR / "m017_staircase_omega.json") as f:
        staircase = json.load(f)
    steps = staircase['steps']
    correct_step = staircase['correct_step_idx']
    N_STEPS = len(steps)

    # True q_A
    t_A = float(OBS_TIMES[pk_A])
    q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0,
                                   np.array([0.0, t_A]), "tumbling", I)
    q_A_true = q_traj[1]

    epoch_indices = list(range(pk_A, pk_B + 1))
    N_epochs = len(epoch_indices)
    epoch_times_from_A = np.array([float(OBS_TIMES[i] - OBS_TIMES[pk_A])
                                   for i in epoch_indices])
    observed_seg = CTX.observed_lc[pk_A:pk_B + 1]
    print(f"  Leg 0: {pk_A}->{pk_B}, {N_epochs} epochs", flush=True)

    # ── Sweep ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    all_results = []

    # Baseline (0 deg nudge)
    print(f"\nNudge=0 deg (baseline):", flush=True)
    mse_base = []
    for step_data in steps:
        w = np.array(step_data['omega_rad_s'])
        q_prop, _ = propagate_attitude(q_A_true, w, epoch_times_from_A,
                                       "tumbling", I)
        pred = np.array([brightness_single_epoch(q, eidx, CTX, use_shadows=False)
                         for q, eidx in zip(q_prop, epoch_indices)])
        mse_base.append(float(np.mean((pred - observed_seg) ** 2)))
    rank_base = np.argsort(mse_base)
    cr_base = int(np.where(rank_base == correct_step)[0][0]) + 1
    print(f"  correct rank: {cr_base}/{N_STEPS}, winner: step {rank_base[0]}")
    all_results.append({
        'nudge_deg': 0, 'direction': 'baseline', 'trials': [{
            'correct_rank': cr_base, 'winner': int(rank_base[0]),
            'mse_correct': round(mse_base[correct_step], 6),
        }]
    })

    for nudge_deg in NUDGE_DEGS:
        print(f"\nNudge={nudge_deg} deg, {N_DIRS} directions:", flush=True)
        t1 = time.time()
        trials = []
        for d in range(N_DIRS):
            q_A_nudged = nudge_quaternion(q_A_true, nudge_deg, rng)
            mse_list = []
            for step_data in steps:
                w = np.array(step_data['omega_rad_s'])
                q_prop, _ = propagate_attitude(q_A_nudged, w, epoch_times_from_A,
                                               "tumbling", I)
                pred = np.array([
                    brightness_single_epoch(q, eidx, CTX, use_shadows=False)
                    for q, eidx in zip(q_prop, epoch_indices)])
                mse_list.append(float(np.mean((pred - observed_seg) ** 2)))
            rank = np.argsort(mse_list)
            cr = int(np.where(rank == correct_step)[0][0]) + 1
            trials.append({
                'correct_rank': cr, 'winner': int(rank[0]),
                'mse_correct': round(mse_list[correct_step], 6),
            })
            print(f"  dir {d}: rank={cr}, winner=step {rank[0]}", flush=True)

        ranks = [t['correct_rank'] for t in trials]
        p_rank1 = sum(1 for r in ranks if r == 1) / len(ranks)
        mean_rank = np.mean(ranks)
        print(f"  => P(rank=1)={p_rank1:.0%}, mean_rank={mean_rank:.1f} "
              f"[{time.time()-t1:.1f}s]")

        all_results.append({
            'nudge_deg': nudge_deg, 'n_directions': N_DIRS,
            'p_rank1': round(p_rank1, 3),
            'mean_rank': round(mean_rank, 2),
            'trials': trials,
        })

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    nudges = [0] + NUDGE_DEGS
    mean_ranks = [all_results[0]['trials'][0]['correct_rank']]
    for r in all_results[1:]:
        mean_ranks.append(r['mean_rank'])
    p_rank1 = [1.0 if mean_ranks[0] == 1 else 0.0]
    for r in all_results[1:]:
        p_rank1.append(r.get('p_rank1', 0.0))

    ax.plot(nudges, mean_ranks, 'o-', color='steelblue', lw=2, ms=8,
            label='Mean correct rank')
    ax.axhline(1, color='green', ls='--', lw=1, alpha=0.5, label='Ideal (rank=1)')
    ax.axhline(cr_base, color='firebrick', ls=':', lw=1,
               label=f'Baseline rank={cr_base}')
    ax.set_xlabel('Attitude nudge (deg)')
    ax.set_ylabel('Correct winding rank')
    ax.set_title('Micro-22a: Winding score stability under q_A nudge\n'
                 f'(8 steps, correct=step {correct_step})')
    ax.set_xticks(nudges)
    ax.set_ylim(0.5, N_STEPS + 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / "m022a_winding_nudge.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    # ── JSON ──────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "m022a_winding_nudge.json"
    save_results(out_json, {
        'correct_step': correct_step,
        'nudge_levels': nudges,
        'results': all_results,
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Saved: {out_json}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
