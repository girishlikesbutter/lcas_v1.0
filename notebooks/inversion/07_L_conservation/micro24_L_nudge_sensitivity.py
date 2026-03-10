#!/usr/bin/env python3
"""Micro-24 — L-consistency sensitivity to attitude errors.

Part A: Nudging q_mid (shared node) — trivially invariant because R cancels.
Part B: Nudging endpoints q_A, q_B — changes the bridging omega, which IS
        the relevant sensitivity for the real pipeline. Uses analytic
        approximation: delta_omega ~ rotvec(dq) / dt.
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
from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

SEED, N_STEPS, N_TRIALS = 42, 8, 30
NUDGE_DEGS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def compute_L(q_wxyz, omega_body, I):
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R @ (I @ omega_body)


def random_axis(rng):
    a = rng.randn(3)
    return a / np.linalg.norm(a)


def endpoint_nudge_trial(args):
    """Nudge q_A and q_B, apply analytic omega perturbation, recheck L."""
    nudge_deg, q_mid, leg0_arrived, leg1_omegas, I, dt_0, dt_1, true_k, true_j, seed = args
    rng = np.random.RandomState(seed)
    nudge_rad = np.deg2rad(nudge_deg)

    # Perturb leg 0 arrived omega: nudging q_A changes bridging omega
    # delta_omega ~ rotvec(dq_A) / dt_0, propagated to arrival
    dq_A_rotvec = nudge_rad * random_axis(rng)
    perturbed_leg0 = [leg0_arrived[k] - dq_A_rotvec / dt_0 for k in range(N_STEPS)]

    # Perturb leg 1 departing omega: nudging q_B changes bridging omega
    dq_B_rotvec = nudge_rad * random_axis(rng)
    perturbed_leg1 = [leg1_omegas[j] + dq_B_rotvec / dt_1 for j in range(N_STEPS)]

    L0 = [compute_L(q_mid, perturbed_leg0[k], I) for k in range(N_STEPS)]
    L1 = [compute_L(q_mid, perturbed_leg1[j], I) for j in range(N_STEPS)]
    L_err = np.array([[np.linalg.norm(L0[k] - L1[j])
                        for j in range(N_STEPS)] for k in range(N_STEPS)])

    best_k, best_j = divmod(int(np.argmin(L_err)), N_STEPS)
    correct = bool(best_k == true_k and best_j == true_j)
    true_val = float(L_err[true_k, true_j])
    flat = L_err.ravel()
    true_rank = int(flat.argsort().argsort()[true_k * N_STEPS + true_j]) + 1
    sorted_flat = np.sort(flat)
    gap = float(sorted_flat[1] - sorted_flat[0]) if true_rank == 1 else 0.0
    return nudge_deg, correct, true_rank, gap, true_val


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times
    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    times_4 = np.array([0.0, OBS_T[PEAKS[0]], OBS_T[PEAKS[1]], OBS_T[PEAKS[2]]])
    q_traj, om_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_4, "tumbling", I)
    q_A, q_mid, q_B = q_traj[1], q_traj[2], q_traj[3]
    om_A, om_mid = om_traj[1], om_traj[2]
    dt_0 = float(OBS_T[PEAKS[1]] - OBS_T[PEAKS[0]])
    dt_1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])
    print(f"Setup: {time.time()-t0:.1f}s, dt_0={dt_0:.1f}s, dt_1={dt_1:.1f}s")

    # Leg 0: micro17 staircase with true omega injected
    with open(RESULTS_DIR / "micro17_staircase_omega.json") as f:
        m17 = json.load(f)
    leg0_omegas = [np.array(s['omega_rad_s']) for s in m17['steps']]
    true_mag = np.linalg.norm(om_A)
    true_k = int(np.argmin([abs(np.linalg.norm(w) - true_mag) for w in leg0_omegas]))
    leg0_omegas[true_k] = om_A.copy()
    leg0_arrived = []
    for w_k in leg0_omegas:
        _, om_prop = propagate_attitude(q_A, w_k, np.array([0.0, dt_0]), "tumbling", I)
        leg0_arrived.append(om_prop[-1])

    # Leg 1: true omega + 7 wrong windings
    delta_1 = 2.0 * np.pi / dt_1
    om_hat = om_mid / np.linalg.norm(om_mid)
    winding_ns = np.array([-3, -2, -1, 0, 1, 2, 3, 4])
    true_j = int(np.where(winding_ns == 0)[0][0])
    leg1_omegas = [om_mid + n * delta_1 * om_hat for n in winding_ns]

    # Part A: q_mid nudge is trivially invariant (R cancels in ||DL||)
    print("\nPart A: q_mid nudge — TRIVIALLY INVARIANT")
    print("  ||DL|| = ||R(q_mid) @ I @ (om_arr - om_dep)|| = ||I @ (om_arr - om_dep)||")
    print("  Since same R on both sides, nudging q_mid has zero effect on ranking.")

    # Part B: endpoint nudge trials
    tasks = []
    for nudge_deg in NUDGE_DEGS:
        for trial in range(N_TRIALS):
            tasks.append((nudge_deg, q_mid, leg0_arrived, leg1_omegas, I,
                          dt_0, dt_1, true_k, true_j, 2000 + len(tasks)))

    print(f"\nPart B: endpoint nudge — {len(tasks)} trials ...")
    with Pool(8) as pool:
        results_raw = pool.map(endpoint_nudge_trial, tasks)

    agg = {d: {'correct': [], 'rank': [], 'gap': [], 'true_Lerr': []} for d in NUDGE_DEGS}
    for nudge_deg, correct, rank, gap, true_val in results_raw:
        agg[nudge_deg]['correct'].append(correct)
        agg[nudge_deg]['rank'].append(rank)
        agg[nudge_deg]['gap'].append(gap)
        agg[nudge_deg]['true_Lerr'].append(true_val)

    print(f"\n{'Nudge':>7}  {'P(correct)':>10}  {'Avg rank':>9}  {'Avg gap':>12}  {'Avg |DL|_true':>13}")
    for d in NUDGE_DEGS:
        p = np.mean(agg[d]['correct'])
        r = np.mean(agg[d]['rank'])
        g = np.mean(agg[d]['gap'])
        v = np.mean(agg[d]['true_Lerr'])
        print(f"{d:>6.1f}°  {p:>10.2f}  {r:>9.1f}  {g:>12.2e}  {v:>13.2e}")

    # --- Plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Micro-24: L-consistency vs endpoint attitude error", fontsize=12)

    p_correct = [np.mean(agg[d]['correct']) for d in NUDGE_DEGS]
    ax1.plot(NUDGE_DEGS, p_correct, 'o-', color='steelblue', linewidth=2)
    ax1.axhline(1.0, color='green', ls='--', alpha=0.5)
    ax1.set_xlabel('Endpoint nudge (deg)'); ax1.set_ylabel('P(true pair selected)')
    ax1.set_title('Selection accuracy'); ax1.set_ylim(-0.05, 1.1); ax1.grid(True, alpha=0.3)

    avg_rank = [np.mean(agg[d]['rank']) for d in NUDGE_DEGS]
    ax2.plot(NUDGE_DEGS, avg_rank, 's-', color='firebrick', linewidth=2)
    ax2.axhline(1.0, color='green', ls='--', alpha=0.5)
    ax2.set_xlabel('Endpoint nudge (deg)'); ax2.set_ylabel('Mean rank of true pair')
    ax2.set_title('True pair rank (1=best)'); ax2.grid(True, alpha=0.3)

    avg_gap = [np.mean(agg[d]['gap']) for d in NUDGE_DEGS]
    ax3.plot(NUDGE_DEGS, avg_gap, 'D-', color='darkorange', linewidth=2)
    ax3.set_xlabel('Endpoint nudge (deg)'); ax3.set_ylabel('Mean gap (kg*m^2/s)')
    ax3.set_title('Gap to next-best pair'); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / "micro24_L_nudge_sensitivity.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\nSaved: {out_png}")

    save_results(RESULTS_DIR / "micro24_L_nudge_sensitivity.json", {
        'nudge_degs': NUDGE_DEGS, 'n_trials': N_TRIALS,
        'true_pair': {'k': true_k, 'j': true_j},
        'part_A': 'q_mid nudge is trivially invariant — R(q) cancels in DL',
        'part_B_results': {str(d): {
            'p_correct': round(float(np.mean(agg[d]['correct'])), 3),
            'avg_rank': round(float(np.mean(agg[d]['rank'])), 2),
            'avg_gap': round(float(np.mean(agg[d]['gap'])), 4),
            'avg_true_Lerr': round(float(np.mean(agg[d]['true_Lerr'])), 4),
        } for d in NUDGE_DEGS},
        'runtime_s': round(time.time() - t0, 1),
    })
    print(f"Total: {time.time()-t0:.1f}s")
