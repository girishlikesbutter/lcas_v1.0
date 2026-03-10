#!/usr/bin/env python3
"""Micro-23 — Oracle L-consistency test: does angular momentum conservation
identify the correct winding pair when both legs include the true omega?

Micro-18 FAILED because leg 1's staircase missed the true omega entirely.
Here we inject the truth into both legs and measure discrimination power.
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


def compute_T(omega_body, I):
    return 0.5 * omega_body @ (I @ omega_body)


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times
    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    print(f"Peaks: {PEAKS}, setup: {time.time()-t0:.1f}s")

    # Oracle: propagate true state to all peaks
    times_4 = np.array([0.0, OBS_T[PEAKS[0]], OBS_T[PEAKS[1]], OBS_T[PEAKS[2]]])
    q_traj, om_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_4, "tumbling", I)
    q_A, q_mid, q_B = q_traj[1], q_traj[2], q_traj[3]
    om_A, om_mid, om_B = om_traj[1], om_traj[2], om_traj[3]
    dt_0 = float(OBS_T[PEAKS[1]] - OBS_T[PEAKS[0]])
    dt_1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])
    print(f"dt_0={dt_0:.1f}s  dt_1={dt_1:.1f}s  true|omega|={np.rad2deg(np.linalg.norm(CTX.true_omega0)):.3f} dps")

    # --- LEG 0: load micro17 staircase, inject true omega ---
    with open(RESULTS_DIR / "micro17_staircase_omega.json") as f:
        m17 = json.load(f)
    leg0_omegas = [np.array(s['omega_rad_s']) for s in m17['steps']]
    # Replace closest step to true with EXACT true omega at peak 183
    mags_leg0 = [np.linalg.norm(w) for w in leg0_omegas]
    true_mag = np.linalg.norm(om_A)
    true_idx_leg0 = int(np.argmin([abs(m - true_mag) for m in mags_leg0]))
    leg0_omegas[true_idx_leg0] = om_A.copy()
    print(f"Leg 0: injected true omega at step {true_idx_leg0} (|omega|={np.rad2deg(true_mag):.3f} dps)")

    # Propagate each leg-0 omega from q_A → arriving omega at peak 260
    leg0_arrived = []
    for w_k in leg0_omegas:
        _, om_prop = propagate_attitude(q_A, w_k, np.array([0.0, dt_0]), "tumbling", I)
        leg0_arrived.append(om_prop[-1])

    # --- LEG 1: construct 8 candidates at peak 260 ---
    delta_1 = 2.0 * np.pi / dt_1
    om_hat = om_mid / np.linalg.norm(om_mid)
    winding_ns = np.array([-3, -2, -1, 0, 1, 2, 3, 4])
    true_idx_leg1 = int(np.where(winding_ns == 0)[0][0])
    leg1_omegas = [om_mid + n * delta_1 * om_hat for n in winding_ns]
    mags_leg1 = [np.rad2deg(np.linalg.norm(w)) for w in leg1_omegas]
    print(f"Leg 1: true omega at step {true_idx_leg1}, mags: {[f'{m:.3f}' for m in mags_leg1]}")

    # --- L and T at peak 260 for all 8x8 pairs ---
    L_leg0 = [compute_L(q_mid, leg0_arrived[k], I) for k in range(N_STEPS)]
    L_leg1 = [compute_L(q_mid, leg1_omegas[j], I) for j in range(N_STEPS)]
    T_leg0 = [compute_T(leg0_arrived[k], I) for k in range(N_STEPS)]
    T_leg1 = [compute_T(leg1_omegas[j], I) for j in range(N_STEPS)]

    L_err = np.array([[np.linalg.norm(L_leg0[k] - L_leg1[j])
                        for j in range(N_STEPS)] for k in range(N_STEPS)])
    T_err = np.array([[abs(T_leg0[k] - T_leg1[j])
                        for j in range(N_STEPS)] for k in range(N_STEPS)])

    # Rank
    flat = L_err.ravel()
    ranks = flat.argsort().argsort()
    true_flat_idx = true_idx_leg0 * N_STEPS + true_idx_leg1
    true_rank_L = int(ranks[true_flat_idx]) + 1
    sorted_flat = np.sort(flat)
    gap_L = float(sorted_flat[1] - sorted_flat[0]) if true_rank_L == 1 else 0.0

    flat_T = T_err.ravel()
    ranks_T = flat_T.argsort().argsort()
    true_rank_T = int(ranks_T[true_flat_idx]) + 1
    sorted_T = np.sort(flat_T)
    gap_T = float(sorted_T[1] - sorted_T[0]) if true_rank_T == 1 else 0.0

    # Combined: ||DL|| + lambda*|DT| (normalized)
    L_norm = L_err / (L_err.max() + 1e-30)
    T_norm = T_err / (T_err.max() + 1e-30)
    combined = L_norm + T_norm
    flat_C = combined.ravel()
    ranks_C = flat_C.argsort().argsort()
    true_rank_C = int(ranks_C[true_flat_idx]) + 1

    best_k, best_j = divmod(int(np.argmin(L_err)), N_STEPS)
    mags_l0 = [np.rad2deg(np.linalg.norm(w)) for w in leg0_omegas]
    print(f"\nTrue pair: ({true_idx_leg0},{true_idx_leg1}), L_err={L_err[true_idx_leg0, true_idx_leg1]:.6e}")
    print(f"Best pair: ({best_k},{best_j}), L_err={L_err[best_k, best_j]:.6e}")
    print(f"True rank by ||DL||: {true_rank_L}/64,  gap: {gap_L:.4e}")
    print(f"True rank by |DT|:   {true_rank_T}/64,  gap: {gap_T:.4e}")
    print(f"True rank by combined: {true_rank_C}/64")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Micro-23: L-consistency oracle test (truth injected)", fontsize=12)

    for ax, data, title, cblabel in [
        (axes[0], L_err, '||DL|| (kg*m^2/s)', '||DL||'),
        (axes[1], T_err, '|DT| (kg*m^2*rad^2/s^2)', '|DT|'),
        (axes[2], combined, 'Combined (normalized)', 'score'),
    ]:
        log_data = np.log10(np.where(data > 0, data, 1e-20))
        im = ax.imshow(log_data, aspect='auto', origin='lower', cmap='viridis_r')
        plt.colorbar(im, ax=ax, label=f'log10({cblabel})')
        ax.set_xlabel('Leg 1 step j'); ax.set_ylabel('Leg 0 step k')
        ax.set_title(title)
        ax.set_xticks(range(N_STEPS))
        ax.set_xticklabels([f"{mags_leg1[j]:.1f}" for j in range(N_STEPS)], fontsize=6)
        ax.set_yticks(range(N_STEPS))
        ax.set_yticklabels([f"{mags_l0[k]:.1f}" for k in range(N_STEPS)], fontsize=6)
        ax.plot(true_idx_leg1, true_idx_leg0, 'r*', ms=16, label='true')
        bk, bj = divmod(int(np.argmin(data)), N_STEPS)
        ax.plot(bj, bk, 'go', ms=12, mfc='none', mew=2.5, label='best')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro23_L_oracle_test.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out_png}")

    save_results(RESULTS_DIR / "micro23_L_oracle_test.json", {
        'true_pair': {'k': true_idx_leg0, 'j': true_idx_leg1},
        'true_L_err': float(L_err[true_idx_leg0, true_idx_leg1]),
        'true_T_err': float(T_err[true_idx_leg0, true_idx_leg1]),
        'true_rank_L': true_rank_L, 'gap_L': gap_L,
        'true_rank_T': true_rank_T, 'gap_T': gap_T,
        'true_rank_combined': true_rank_C,
        'best_pair_L': {'k': best_k, 'j': best_j},
        'L_err_matrix': [[round(float(v), 6) for v in row] for row in L_err],
        'T_err_matrix': [[round(float(v), 6) for v in row] for row in T_err],
        'leg0_mags_degs': [round(m, 5) for m in mags_l0],
        'leg1_mags_degs': [round(m, 5) for m in mags_leg1],
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Total: {time.time()-t0:.1f}s")
