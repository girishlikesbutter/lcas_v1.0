#!/usr/bin/env python3
"""Micro-21 — Multi-epoch LC shape scoring as winding discriminator.

Hypothesis: different winding numbers produce radically different attitude
trajectories between peaks. Evaluating brightness at MULTIPLE intermediate
epochs should uniquely identify the correct winding.

Uses micro17's 8 staircase omega solutions for leg 0 (peaks 183->260, ~554s).
For each winding, propagate from q_A (true attitude at peak 183), compute
lo-fi brightness at all ~77 intermediate epochs, and rank by MSE vs observed.

Also computes lo-fi reference LC from the TRUE trajectory to separate:
  (a) omega direction mismatch (staircase omega != true omega)
  (b) lo-fi vs hi-fi model mismatch (no shadows vs shadows)
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

from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SUBSAMPLE_RATES = [1, 2, 5, 10, 20, 40]

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

    # ── Load peaks and staircase solutions ────────────────────────────────
    stage1 = np.load(RESULTS_DIR / "micro13_stage1.npz")
    PEAKS = [int(x) for x in stage1["peaks"]]  # [183, 260, 360]
    pk_A, pk_B = PEAKS[0], PEAKS[1]

    with open(RESULTS_DIR / "micro17_staircase_omega.json") as f:
        staircase = json.load(f)
    steps = staircase['steps']
    true_omega_mag_degs = staircase['true_omega_mag_degs']
    correct_step = staircase['correct_step_idx']  # 3
    N_STEPS = len(steps)

    # ── True attitude at peak A + true omega at A ─────────────────────────
    t_A = float(OBS_TIMES[pk_A])
    t_B = float(OBS_TIMES[pk_B])
    dt_AB = t_B - t_A
    q_traj_to_A, w_traj_to_A = propagate_attitude(
        CTX.true_q0, CTX.true_omega0, np.array([0.0, t_A]), "tumbling", I)
    q_A_true = q_traj_to_A[1]
    true_omega_at_A = w_traj_to_A[-1]
    true_omega_at_A_degs = np.rad2deg(true_omega_at_A)
    print(f"  True omega at A: [{true_omega_at_A_degs[0]:.3f}, "
          f"{true_omega_at_A_degs[1]:.3f}, {true_omega_at_A_degs[2]:.3f}] deg/s "
          f"(|w|={np.linalg.norm(true_omega_at_A_degs):.3f})", flush=True)

    # Epoch indices between peaks (inclusive of endpoints)
    epoch_indices = list(range(pk_A, pk_B + 1))
    N_epochs = len(epoch_indices)
    epoch_times_from_A = np.array([float(OBS_TIMES[i] - OBS_TIMES[pk_A])
                                   for i in epoch_indices])
    observed_seg = CTX.observed_lc[pk_A:pk_B + 1]
    print(f"  Leg 0: peaks {pk_A}->{pk_B}, dt={dt_AB:.1f}s, "
          f"{N_epochs} epochs", flush=True)

    # ── Lo-fi reference: true trajectory brightness ──────────────────────
    print(f"\nComputing lo-fi reference from true trajectory ...", flush=True)
    t1 = time.time()
    lofi_ref = np.empty(N_epochs)
    for j, eidx in enumerate(epoch_indices):
        lofi_ref[j] = brightness_single_epoch(
            CTX.true_quaternions[eidx], eidx, CTX, use_shadows=False)
    lofi_hifi_mse = float(np.mean((lofi_ref - observed_seg) ** 2))
    print(f"  Lo-fi ref MSE vs observed (hi-fi+noise): {lofi_hifi_mse:.4f}")
    print(f"  Done in {time.time()-t1:.1f}s", flush=True)

    # ── Propagate all windings and compute brightness ─────────────────────
    print(f"\nEvaluating {N_STEPS} windings x {N_epochs} epochs ...", flush=True)
    t1 = time.time()

    predicted_lcs = []
    omega_dir_errors = []
    for k, step_data in enumerate(steps):
        w = np.array(step_data['omega_rad_s'])
        # Omega direction error vs true
        w_unit = w / (np.linalg.norm(w) + 1e-15)
        true_unit = true_omega_at_A / (np.linalg.norm(true_omega_at_A) + 1e-15)
        dot = np.clip(np.dot(w_unit, true_unit), -1, 1)
        dir_err_deg = float(np.rad2deg(np.arccos(abs(dot))))
        omega_dir_errors.append(dir_err_deg)

        q_prop, _ = propagate_attitude(q_A_true, w, epoch_times_from_A,
                                       "tumbling", I)
        predicted = np.empty(N_epochs)
        for j, (q, eidx) in enumerate(zip(q_prop, epoch_indices)):
            predicted[j] = brightness_single_epoch(q, eidx, CTX,
                                                    use_shadows=False)
        predicted_lcs.append(predicted)
        w_degs = np.rad2deg(w)
        print(f"  step {k}: |w|={step_data['mag_degs']:.2f} d/s, "
              f"dir_err={dir_err_deg:.1f} deg  [{time.time()-t1:.1f}s]", flush=True)

    print(f"  All done in {time.time()-t1:.1f}s", flush=True)

    # ── MSE ranking vs observed (hi-fi + noise) ──────────────────────────
    print(f"\n{'='*60}")
    print("RANKING vs OBSERVED LC (hi-fi + noise)")
    print(f"{'='*60}")
    mse_vs_obs = []
    for plc in predicted_lcs:
        mse_vs_obs.append(float(np.mean((plc - observed_seg) ** 2)))

    rank_obs = np.argsort(mse_vs_obs)
    cr_obs = int(np.where(rank_obs == correct_step)[0][0]) + 1

    print(f"{'Step':>4}  {'|w| d/s':>8}  {'dir_err':>8}  {'MSE':>10}  {'Rank':>4}")
    print(f"{'─'*46}")
    for pos, idx in enumerate(rank_obs):
        marker = " <-- correct" if idx == correct_step else ""
        print(f"{idx:>4}  {steps[idx]['mag_degs']:>8.3f}  "
              f"{omega_dir_errors[idx]:>7.1f}°  "
              f"{mse_vs_obs[idx]:>10.4f}  {pos+1:>4}{marker}")
    gap_obs = mse_vs_obs[rank_obs[1]] - mse_vs_obs[rank_obs[0]] if cr_obs == 1 else 0.0
    print(f"\nCorrect rank: {cr_obs}/{N_STEPS}, gap: {gap_obs:.4f}")
    print(f"Lo-fi ref MSE (true traj, no shadows): {lofi_hifi_mse:.4f}")

    # ── MSE ranking vs lo-fi reference (removes shadow mismatch) ─────────
    print(f"\n{'='*60}")
    print("RANKING vs LO-FI REFERENCE (no shadow mismatch)")
    print(f"{'='*60}")
    mse_vs_lofi = []
    for plc in predicted_lcs:
        mse_vs_lofi.append(float(np.mean((plc - lofi_ref) ** 2)))

    rank_lofi = np.argsort(mse_vs_lofi)
    cr_lofi = int(np.where(rank_lofi == correct_step)[0][0]) + 1

    print(f"{'Step':>4}  {'|w| d/s':>8}  {'dir_err':>8}  {'MSE':>10}  {'Rank':>4}")
    print(f"{'─'*46}")
    for pos, idx in enumerate(rank_lofi):
        marker = " <-- correct" if idx == correct_step else ""
        print(f"{idx:>4}  {steps[idx]['mag_degs']:>8.3f}  "
              f"{omega_dir_errors[idx]:>7.1f}°  "
              f"{mse_vs_lofi[idx]:>10.4f}  {pos+1:>4}{marker}")
    gap_lofi = mse_vs_lofi[rank_lofi[1]] - mse_vs_lofi[rank_lofi[0]] if cr_lofi == 1 else 0.0
    print(f"\nCorrect rank: {cr_lofi}/{N_STEPS}, gap: {gap_lofi:.4f}")

    # ── Subsampling sensitivity (vs observed) ─────────────────────────────
    print(f"\nSubsampling sensitivity (vs observed):")
    subsample_results = []
    for N in SUBSAMPLE_RATES:
        sub_idx = np.arange(0, N_epochs, N)
        mse_sub = [float(np.mean((plc[sub_idx] - observed_seg[sub_idx]) ** 2))
                   for plc in predicted_lcs]
        r = np.argsort(mse_sub)
        cr = int(np.where(r == correct_step)[0][0]) + 1
        g = mse_sub[r[1]] - mse_sub[r[0]] if cr == 1 else 0.0
        subsample_results.append({
            'every_N': N, 'n_points': len(sub_idx),
            'correct_rank': cr, 'mse_gap': round(g, 6), 'winner': int(r[0]),
        })
        status = "CORRECT" if cr == 1 else f"WRONG (winner=step {r[0]})"
        print(f"  every {N:>2}: {len(sub_idx):>3} pts, rank={cr}, {status}")

    # ── Plot ──────────────────────────────────────────────────────────────
    epoch_times_s = epoch_times_from_A
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])
    fig.suptitle("Micro-21: Multi-epoch winding score — 8 predicted LCs vs observed",
                 fontsize=13, y=0.98)

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, N_STEPS))
    for k in range(N_STEPS):
        lw = 2.5 if k == correct_step else 1.2
        ls = '-' if k == correct_step else '--'
        alpha = 1.0 if k == correct_step else 0.7
        label = (f"step {k}: {steps[k]['mag_degs']:.2f}d/s "
                 f"MSE={mse_vs_obs[k]:.3f}")
        if k == correct_step:
            label += " (correct)"
        ax.plot(epoch_times_s, predicted_lcs[k], ls=ls, lw=lw, alpha=alpha,
                color=colors[k], label=label)
    ax.plot(epoch_times_s, lofi_ref, 'g-', lw=2, alpha=0.8,
            label=f'lo-fi truth (MSE={lofi_hifi_mse:.3f})')
    ax.scatter(epoch_times_s, observed_seg, s=8, c='black', zorder=5,
               alpha=0.6, label='observed (hi-fi + noise)')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Leg 0: peaks {pk_A}->{pk_B} ({dt_AB:.0f}s)')
    ax.legend(fontsize=6.5, loc='upper right', ncol=2)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for k in range(N_STEPS):
        lw = 2.0 if k == correct_step else 0.8
        alpha = 1.0 if k == correct_step else 0.5
        ax2.plot(epoch_times_s, predicted_lcs[k] - observed_seg,
                 lw=lw, alpha=alpha, color=colors[k])
    ax2.plot(epoch_times_s, lofi_ref - observed_seg, 'g-', lw=2, alpha=0.8)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlabel('Time from peak A (s)')
    ax2.set_ylabel('Residual (pred - obs)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro21_multi_epoch_winding_score.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    # ── JSON ──────────────────────────────────────────────────────────────
    results = {
        'n_steps': N_STEPS, 'n_epochs': N_epochs,
        'correct_step': correct_step,
        'true_omega_mag_degs': true_omega_mag_degs,
        'true_omega_at_A_degs': [round(x, 5) for x in true_omega_at_A_degs],
        'lofi_hifi_mismatch_mse': round(lofi_hifi_mse, 6),
        'vs_observed': {
            'correct_rank': cr_obs,
            'mse_gap': round(gap_obs, 6),
            'ranking': [{'step': int(rank_obs[i]),
                          'mag_degs': steps[int(rank_obs[i])]['mag_degs'],
                          'dir_err_deg': round(omega_dir_errors[int(rank_obs[i])], 2),
                          'mse': round(mse_vs_obs[int(rank_obs[i])], 6),
                          'rank': i + 1}
                         for i in range(N_STEPS)],
        },
        'vs_lofi_ref': {
            'correct_rank': cr_lofi,
            'mse_gap': round(gap_lofi, 6),
            'ranking': [{'step': int(rank_lofi[i]),
                          'mag_degs': steps[int(rank_lofi[i])]['mag_degs'],
                          'dir_err_deg': round(omega_dir_errors[int(rank_lofi[i])], 2),
                          'mse': round(mse_vs_lofi[int(rank_lofi[i])], 6),
                          'rank': i + 1}
                         for i in range(N_STEPS)],
        },
        'subsample_sensitivity': subsample_results,
        'omega_direction_errors_deg': [round(x, 2) for x in omega_dir_errors],
        'runtime_s': round(time.time() - t0, 1),
    }
    out_json = RESULTS_DIR / "micro21_multi_epoch_winding_score.json"
    save_results(out_json, results)
    print(f"Saved: {out_json}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
