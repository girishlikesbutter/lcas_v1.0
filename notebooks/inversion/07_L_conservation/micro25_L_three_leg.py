#!/usr/bin/env python3
"""Micro-25 — Three-leg L-consistency: does a 3rd leg improve discrimination?

With 3 peaks (A,B,C) we have 2 legs and 1 shared node (B).
With 4 peaks (A,B,C,D) we get 3 legs and 2 shared nodes (B,C).
The correct winding triple must satisfy L-consistency at BOTH B and C.
"""
import sys, time, json, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

SEED, N_STEPS = 42, 8
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def compute_L(q_wxyz, omega_body, I):
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R @ (I @ omega_body)


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times
    lc = CTX.observed_lc  # magnitudes (higher = dimmer, peaks = local minima)

    # Existing peaks (brightness maxima = magnitude minima)
    PEAKS_3 = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    print(f"Existing peaks: {PEAKS_3}")

    # Search for a 4th peak: local minima in magnitude outside existing peaks
    from scipy.signal import argrelmin
    local_mins = argrelmin(lc, order=15)[0]
    # Filter: must be outside existing peaks by at least 30 epochs
    candidates = [p for p in local_mins
                  if all(abs(p - ep) > 30 for ep in PEAKS_3) and p < 490]
    if candidates:
        # Pick the deepest (brightest = lowest magnitude)
        best = min(candidates, key=lambda p: lc[p])
        PEAKS_4 = sorted(PEAKS_3 + [best])
        print(f"Found 4th peak at epoch {best} (mag={lc[best]:.2f})")
    else:
        print("No suitable 4th peak found — trying wider search (order=10)")
        local_mins = argrelmin(lc, order=10)[0]
        candidates = [p for p in local_mins
                      if all(abs(p - ep) > 20 for ep in PEAKS_3) and p < 490]
        if candidates:
            best = min(candidates, key=lambda p: lc[p])
            PEAKS_4 = sorted(PEAKS_3 + [best])
            print(f"Found 4th peak at epoch {best} (mag={lc[best]:.2f})")
        else:
            print("NO 4TH PEAK FOUND. Using a non-peak epoch (ep=100) as synthetic anchor.")
            best = 100
            PEAKS_4 = sorted(PEAKS_3 + [best])

    print(f"4 peaks: {PEAKS_4}")

    # Oracle: propagate true state to all 4 peaks
    times_5 = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS_4])
    q_traj, om_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_5, "tumbling", I)
    q_peaks = q_traj[1:]   # shape (4, 4)
    om_peaks = om_traj[1:]  # shape (4, 3)

    # Build 3 legs between consecutive peaks
    dts = [float(OBS_T[PEAKS_4[i+1]] - OBS_T[PEAKS_4[i]]) for i in range(3)]
    print(f"Leg dts: {[f'{d:.1f}s' for d in dts]}")

    # For each leg: construct 8 omega candidates at the START of the leg
    all_leg_omegas = []
    true_indices = []
    for leg_idx in range(3):
        om_start = om_peaks[leg_idx]  # true omega at start of this leg
        delta = 2.0 * np.pi / dts[leg_idx]
        om_hat = om_start / np.linalg.norm(om_start)
        winding_ns = np.array([-3, -2, -1, 0, 1, 2, 3, 4])
        candidates = [om_start + n * delta * om_hat for n in winding_ns]
        all_leg_omegas.append(candidates)
        true_indices.append(int(np.where(winding_ns == 0)[0][0]))
        print(f"Leg {leg_idx}: delta={np.rad2deg(delta):.3f} dps, "
              f"true_idx={true_indices[-1]}, "
              f"mags={[f'{np.rad2deg(np.linalg.norm(w)):.2f}' for w in candidates]}")

    # Propagate each leg's start omega to get arriving omega at end
    all_arrived = []
    for leg_idx in range(3):
        arrived = []
        q_start = q_peaks[leg_idx]
        dt = dts[leg_idx]
        for w in all_leg_omegas[leg_idx]:
            _, om_prop = propagate_attitude(q_start, w, np.array([0.0, dt]), "tumbling", I)
            arrived.append(om_prop[-1])
        all_arrived.append(arrived)
    print(f"Propagation done: {time.time()-t0:.1f}s")

    # 2-leg consistency: L at shared node B (PEAKS_4[1]) for legs 0-1
    # L at shared node C (PEAKS_4[2]) for legs 1-2
    # For leg k arriving at node: L = compute_L(q_node, omega_arrived_k, I)
    # For leg k departing from node: L = compute_L(q_node, omega_start_k, I)

    # Node B = PEAKS_4[1]: leg 0 arrives, leg 1 departs
    # Node C = PEAKS_4[2]: leg 1 arrives, leg 2 departs
    q_B = q_peaks[1]
    q_C = q_peaks[2]

    # 8x8x8 = 512 triples: (i=leg0 step, j=leg1 step, k=leg2 step)
    L_B_arr = [compute_L(q_B, all_arrived[0][i], I) for i in range(N_STEPS)]  # leg 0 arriving at B
    L_B_dep = [compute_L(q_B, all_leg_omegas[1][j], I) for j in range(N_STEPS)]  # leg 1 departing B
    L_C_arr = [compute_L(q_C, all_arrived[1][j], I) for j in range(N_STEPS)]  # leg 1 arriving at C
    L_C_dep = [compute_L(q_C, all_leg_omegas[2][k], I) for k in range(N_STEPS)]  # leg 2 departing C

    scores = np.zeros((N_STEPS, N_STEPS, N_STEPS))
    for i in range(N_STEPS):
        for j in range(N_STEPS):
            dL_B = np.linalg.norm(L_B_arr[i] - L_B_dep[j])
            dL_C_arr_j = L_C_arr[j]
            for k in range(N_STEPS):
                dL_C = np.linalg.norm(dL_C_arr_j - L_C_dep[k])
                scores[i, j, k] = dL_B + dL_C

    true_triple = (true_indices[0], true_indices[1], true_indices[2])
    true_score = float(scores[true_triple])
    flat_scores = scores.ravel()
    sorted_scores = np.sort(flat_scores)
    true_rank = int(flat_scores.argsort().argsort()[
        true_triple[0] * N_STEPS**2 + true_triple[1] * N_STEPS + true_triple[2]]) + 1
    gap = float(sorted_scores[1] - sorted_scores[0]) if true_rank == 1 else 0.0

    best_flat = int(np.argmin(flat_scores))
    best_i = best_flat // (N_STEPS**2)
    best_j = (best_flat % (N_STEPS**2)) // N_STEPS
    best_k = best_flat % N_STEPS

    print(f"\nTrue triple: {true_triple}, score: {true_score:.6e}")
    print(f"Best triple: ({best_i},{best_j},{best_k}), score: {scores[best_i, best_j, best_k]:.6e}")
    print(f"True rank: {true_rank}/512, gap: {gap:.4e}")

    # Compare with 2-leg only (just node B)
    L_err_2leg = np.array([[np.linalg.norm(L_B_arr[i] - L_B_dep[j])
                            for j in range(N_STEPS)] for i in range(N_STEPS)])
    true_2leg = float(L_err_2leg[true_indices[0], true_indices[1]])
    flat_2 = L_err_2leg.ravel()
    true_rank_2 = int(flat_2.argsort().argsort()[true_indices[0] * N_STEPS + true_indices[1]]) + 1
    print(f"\n2-leg only (node B): true rank {true_rank_2}/64")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Micro-25: 3-leg L-consistency (peaks {PEAKS_4})", fontsize=12)

    # Panel 1: 2-leg heatmap at node B
    log_2 = np.log10(np.where(L_err_2leg > 0, L_err_2leg, 1e-20))
    im = axes[0].imshow(log_2, aspect='auto', origin='lower', cmap='viridis_r')
    plt.colorbar(im, ax=axes[0], label='log10(||DL||)')
    axes[0].set_xlabel('Leg 1 step'); axes[0].set_ylabel('Leg 0 step')
    axes[0].set_title(f'2-leg at node B (rank {true_rank_2}/64)')
    axes[0].plot(true_indices[1], true_indices[0], 'r*', ms=16)

    # Panel 2: 3-leg marginalised over best k for each (i,j)
    best_over_k = scores.min(axis=2)
    log_3 = np.log10(np.where(best_over_k > 0, best_over_k, 1e-20))
    im2 = axes[1].imshow(log_3, aspect='auto', origin='lower', cmap='viridis_r')
    plt.colorbar(im2, ax=axes[1], label='log10(min_k score)')
    axes[1].set_xlabel('Leg 1 step'); axes[1].set_ylabel('Leg 0 step')
    axes[1].set_title(f'3-leg min over k (rank {true_rank}/512)')
    axes[1].plot(true_indices[1], true_indices[0], 'r*', ms=16)

    # Panel 3: sorted scores with true marked
    axes[2].plot(sorted_scores, 'k-', linewidth=0.5)
    axes[2].axvline(true_rank - 1, color='red', linewidth=2, label=f'true (rank {true_rank})')
    axes[2].set_xlabel('Rank'); axes[2].set_ylabel('Score (||DL_B|| + ||DL_C||)')
    axes[2].set_title('All 512 triples sorted')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro25_L_three_leg.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out_png}")

    save_results(RESULTS_DIR / "micro25_L_three_leg.json", {
        'peaks_4': [int(p) for p in PEAKS_4],
        'dts': [round(d, 1) for d in dts],
        'true_triple': [int(x) for x in true_triple],
        'true_score': round(true_score, 6),
        'true_rank_3leg': true_rank,
        'true_rank_2leg': true_rank_2,
        'gap_3leg': round(gap, 6),
        'best_triple': [int(best_i), int(best_j), int(best_k)],
        'best_score': round(float(scores[best_i, best_j, best_k]), 6),
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Total: {time.time()-t0:.1f}s")
