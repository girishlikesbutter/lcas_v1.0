#!/usr/bin/env python3
"""Micro-18 — L-consistency constraint across two legs sharing a peak node.
Staircase generates N_STEPS valid ω per leg (different winding numbers).
At shared peak 260, L must be conserved: L_leg0 == L_leg1.
"""
import sys, time, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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


def run_staircase(q_start, q_end, dt, I, n_steps, label):
    """Staircase: N_STEPS ω solutions with increasing winding numbers.
    Lower + upper barriers confine to one winding band per step.
    """
    from scipy.spatial.transform import Rotation as _Rot
    delta_rad = 2.0 * np.pi / dt

    def obj(w, lb_sq, ub_sq):
        lb_bar = max(0.0, lb_sq - float(np.dot(w, w))) ** 2 * 1e4
        ub_bar = max(0.0, float(np.dot(w, w)) - ub_sq) ** 2 * 1e4
        qp, _ = propagate_attitude(q_start, w, np.array([0.0, dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        return (1.0 - d * d) + lb_bar + ub_bar

    R_s = _Rot.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    R_e = _Rot.from_quat([q_end[1],   q_end[2],   q_end[3],   q_end[0]])
    w_prev = (R_s.inv() * R_e).as_rotvec() / dt
    print(f"\n{label} (dt={dt:.1f}s, delta={np.rad2deg(delta_rad):.3f} deg/s)")
    print(f"  {'Step':>4}  {'|ω| deg/s':>10}  {'arr_err':>8}")
    omegas, lb_sq = [], 0.0

    for k in range(n_steps):
        ub_sq = (float(np.linalg.norm(w_prev)) + 0.8 * delta_rad) ** 2
        res = minimize(obj, w_prev, args=(lb_sq, ub_sq), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9})
        w_k = res.x.copy()
        mag_k = float(np.linalg.norm(w_k))
        qp2, _ = propagate_attitude(q_start, w_k, np.array([0.0, dt]), "tumbling", I)
        arr_err = float(1.0 - np.clip(np.dot(qp2[-1], q_end), -1.0, 1.0) ** 2)
        print(f"  {k:>4}  {np.rad2deg(mag_k):>10.3f}  {arr_err:>8.2e}", flush=True)
        omegas.append(w_k)
        lb_sq = mag_k ** 2 * 1.01
        rot_axis = w_k / mag_k if mag_k > 1e-12 else np.array([0., 0., 1.])
        w_prev = w_k + delta_rad * rot_axis

    return omegas


if __name__ == '__main__':
    t0 = time.time()
    print("Setting up experiment ...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    INERTIA, OBS_TIMES = CTX.inertia_tensor, CTX.observation_times
    print(f"  Setup done: {time.time()-t0:.1f}s", flush=True)

    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    print(f"  Peaks: {PEAKS}", flush=True)
    times_3 = np.array([0.0, float(OBS_TIMES[PEAKS[0]]),
                        float(OBS_TIMES[PEAKS[1]]), float(OBS_TIMES[PEAKS[2]])])
    q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_3, "tumbling", INERTIA)
    q_A, q_mid, q_B = q_traj[1], q_traj[2], q_traj[3]
    dt_0 = float(OBS_TIMES[PEAKS[1]] - OBS_TIMES[PEAKS[0]])
    dt_1 = float(OBS_TIMES[PEAKS[2]] - OBS_TIMES[PEAKS[1]])
    true_omega_mag_degs = float(np.rad2deg(np.linalg.norm(CTX.true_omega0)))
    print(f"  dt_0={dt_0:.1f}s  dt_1={dt_1:.1f}s  true|ω|={true_omega_mag_degs:.3f} deg/s")

    omegas_leg0 = run_staircase(q_A,   q_mid, dt_0, INERTIA, N_STEPS, "Leg 0 (183→260)")
    omegas_leg1 = run_staircase(q_mid, q_B,   dt_1, INERTIA, N_STEPS, "Leg 1 (260→360)")

    # L at shared node for each leg
    print("\nComputing L vectors at peak 260 ...", flush=True)
    L_leg0 = []
    for w_k in omegas_leg0:
        _, om_traj = propagate_attitude(q_A, w_k, np.array([0.0, dt_0]), "tumbling", INERTIA)
        L_leg0.append(compute_L(q_mid, om_traj[-1], INERTIA))
    L_leg1 = [compute_L(q_mid, w_j, INERTIA) for w_j in omegas_leg1]

    # L-error matrix
    L_err = np.array([[np.linalg.norm(L_leg0[k] - L_leg1[j])
                       for j in range(N_STEPS)] for k in range(N_STEPS)])
    mags_leg0 = np.array([np.rad2deg(np.linalg.norm(w)) for w in omegas_leg0])
    mags_leg1 = np.array([np.rad2deg(np.linalg.norm(w)) for w in omegas_leg1])

    # Print matrix
    print("\nL_err matrix (kg·m²/s):")
    print("       " + "".join(f" j={j}({mags_leg1[j]:5.2f})" for j in range(N_STEPS)))
    for k in range(N_STEPS):
        print(f"k={k}({mags_leg0[k]:5.2f})" +
              "".join(f" {L_err[k,j]:>10.4f}" for j in range(N_STEPS)))

    best_k, best_j = divmod(int(np.argmin(L_err)), N_STEPS)
    best_L_err = float(L_err[best_k, best_j])
    true_k = int(np.argmin(np.abs(mags_leg0 - true_omega_mag_degs)))
    true_j = int(np.argmin(np.abs(mags_leg1 - true_omega_mag_degs)))
    correct = bool(best_k == true_k and best_j == true_j)

    print(f"\nTrue |ω| = {true_omega_mag_degs:.3f} deg/s")
    print(f"True pair:  k={true_k} (|ω|={mags_leg0[true_k]:.3f}),  "
          f"j={true_j} (|ω|={mags_leg1[true_j]:.3f})")
    print(f"Best pair:  k={best_k} (|ω|={mags_leg0[best_k]:.3f}),  "
          f"j={best_j} (|ω|={mags_leg1[best_j]:.3f}),  L_err={best_L_err:.4e}")
    print(f"True-pair L_err = {L_err[true_k, true_j]:.4e}")
    verdict = ("CORRECT — L-conservation identifies true winding" if correct else "FAILED")
    print(f"\n{verdict}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Micro-18: L-consistency across two legs (oracle attitudes)", fontsize=12)

    log_L = np.log10(np.where(L_err > 0, L_err, 1e-20))
    im = ax1.imshow(log_L, aspect='auto', origin='lower', cmap='viridis_r',
                    interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='log10(||ΔL||)')
    ax1.set_xlabel('Leg 1 step j');  ax1.set_ylabel('Leg 0 step k')
    ax1.set_title('L-error heatmap (log10 scale)')
    ax1.set_xticks(range(N_STEPS)); ax1.set_yticks(range(N_STEPS))
    ax1.set_xticklabels([f"j={j}\n{mags_leg1[j]:.1f}" for j in range(N_STEPS)], fontsize=7)
    ax1.set_yticklabels([f"k={k}\n{mags_leg0[k]:.1f}" for k in range(N_STEPS)], fontsize=7)
    ax1.plot(best_j, best_k, 'go', ms=14, markerfacecolor='none',
             markeredgewidth=2.5, label=f'min ({best_k},{best_j})')
    ax1.plot(true_j, true_k, 'r*', ms=16, label=f'true ({true_k},{true_j})')
    ax1.legend(fontsize=8, loc='upper right')

    min_per_k = L_err.min(axis=1)
    bar_colors = ['gold' if k == true_k else 'steelblue' for k in range(N_STEPS)]
    ax2.bar(range(N_STEPS), min_per_k, color=bar_colors, edgecolor='k', linewidth=0.6)
    ax2.axvline(true_k, color='firebrick', lw=2.0, ls='--', label=f'true k={true_k}')
    ax2.set_xlabel('Leg 0 step k')
    ax2.set_ylabel('min_j ||L_leg0[k] − L_leg1[j]||  (kg·m²/s)')
    ax2.set_title('Best L-match per leg-0 step\n(gold = true step)')
    ax2.set_xticks(range(N_STEPS))
    ax2.set_xticklabels([f"k={k}\n{mags_leg0[k]:.1f}" for k in range(N_STEPS)], fontsize=7)
    ax2.legend(fontsize=8);  ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro18_L_consistency.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    save_results(RESULTS_DIR / "micro18_L_consistency.json", {
        'n_steps': N_STEPS,
        'true_omega_mag_degs': round(true_omega_mag_degs, 5),
        'leg0_mags_degs': [round(float(m), 5) for m in mags_leg0],
        'leg1_mags_degs': [round(float(m), 5) for m in mags_leg1],
        'L_err_matrix': [[round(float(L_err[k, j]), 6) for j in range(N_STEPS)]
                         for k in range(N_STEPS)],
        'best_pair': {'k': int(best_k), 'j': int(best_j), 'L_err': best_L_err},
        'true_pair': {'k': int(true_k), 'j': int(true_j)},
        'correct': correct,
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Saved: {RESULTS_DIR / 'micro18_L_consistency.json'}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
