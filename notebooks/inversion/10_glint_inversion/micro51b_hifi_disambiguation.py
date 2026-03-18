#!/usr/bin/env python3
"""Micro-51b -- Hi-fi disambiguation test (oracle omega, 10 trajectories).

Focused test: does hi-fi (shadow) LC residual break the antiparallel degeneracy?
For each trajectory with oracle omega:
1. Phi sweep (10 hyp x 36 phi) → top 4 by glint cost
2. Propagate each to t=0 → full lo-fi LC residual → top 2
3. Full hi-fi LC residual on top 2 → pick winner

Reports: correct hypothesis selection rate, q0 error, omega error.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"


def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    dt = target_times - anchor_time
    fwd = dt > 1e-6
    bwd = dt < -1e-6
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    if fwd.any():
        ft = np.concatenate([[0.0], dt[fwd]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        qb, _ = propagate_attitude(q_anchor, -omega, np.concatenate([[0.0], bt]),
                                   "tumbling", I_tensor)
        tq[bwd] = qb[1:][::-1]
    return tq

def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    total = 0.0
    for i in range(len(glint_quats)):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pab_arr[i])
                       for j in range(len(all_normals)))
        total += (1.0 - best_dot) ** 2
    return total

def omega_direction_error(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("micro51b -- Hi-fi disambiguation (oracle omega, 10 trajectories)")
print("=" * 70)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(unique_normals)

# Select 10 test trajectories with >=3 bright peaks
omega_sorted = np.argsort(omega_mags_arr)
candidates = [idx for idx in omega_sorted
              if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                        prominence=0.3)[0]] < 9.0) >= 3]
sel = np.linspace(0, len(candidates) - 1, 10, dtype=int)
TEST = [candidates[i] for i in sel]
print(f"Test: {TEST}, omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")


# ===========================================================================
# Run pipeline
# ===========================================================================
all_results = []

for traj_idx in TEST:
    print(f"\n{'='*50}")
    print(f"Traj {traj_idx} (omega={omega_mags_arr[traj_idx]:.3f} deg/s)")
    t0 = time.time()

    mags = mag_hifi[traj_idx]
    ff = group_frac_flux[traj_idx]
    quats = quaternions[traj_idx]

    _, omega_hist = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], obs_times, "tumbling", I_tensor)

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    if len(bright) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        continue

    # Oracle labels
    labels = [int(np.argmax(ff[:, p])) for p in bright]
    confs = [float(ff[labels[i], bright[i]]) for i in range(len(bright))]
    conf_idx = [i for i, c in enumerate(confs) if c > 0.77]
    conf_peaks = bright[conf_idx]
    if len(conf_peaks) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few confident'})
        continue

    # Anchor
    anch = int(conf_peaks[np.argmin(mags[conf_peaks])])
    q_true_anch = quats[anch]
    omega_anch = omega_hist[anch]
    og = int(np.argmax(ff[:, anch]))

    non_anch = conf_peaks[conf_peaks != anch]
    g_pabs = pab_j2000[non_anch]
    g_times = obs_times[non_anch]

    print(f"  Anchor={anch}, group={group_names[og]}, "
          f"glints={len(non_anch)}, |omega|={np.rad2deg(np.linalg.norm(omega_anch)):.3f}")

    # --- Step 1: Phi sweep ---
    hyp_results = []
    for hi in range(n_normals):
        nb = unique_normals[hi]
        bc = np.inf
        bp = 0.0
        for phi in np.linspace(0, 2*np.pi, 36, endpoint=False):
            qa = anchor_q_from_phi(phi, nb, pab_j2000[anch])
            try:
                gq = propagate_sparse(qa, omega_anch, obs_times[anch], g_times, I_tensor)
                c = min_over_normals_cost(gq, g_pabs, unique_normals)
            except Exception:
                c = 1e10
            if c < bc:
                bc = c
                bp = phi
        hyp_results.append({'idx': hi, 'cost': bc, 'phi': bp, 'correct': hi == og})

    sorted_h = sorted(hyp_results, key=lambda h: h['cost'])

    # --- Step 2: Lo-fi LC residual on top 4 ---
    obj_lofi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    for h in sorted_h[:4]:
        qa = anchor_q_from_phi(h['phi'], unique_normals[h['idx']], pab_j2000[anch])
        try:
            bt = np.array([0.0, obs_times[anch]])
            qb, ob = propagate_attitude(qa, -omega_anch, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -ob[-1]
            rv = Rotation.from_quat([q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            h['lofi'] = float(obj_lofi.evaluate(np.concatenate([rv, o0c])))
            h['q0'] = q0c; h['o0'] = o0c
        except Exception:
            h['lofi'] = 1e10

    sorted_lofi = sorted(sorted_h[:4], key=lambda h: h.get('lofi', 1e10))
    print(f"  Lo-fi top 2: G{sorted_lofi[0]['idx']}({sorted_lofi[0]['lofi']:.3f}) "
          f"vs G{sorted_lofi[1]['idx']}({sorted_lofi[1]['lofi']:.3f})")

    # --- Step 3: Hi-fi on top 2 ---
    obj_hifi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    for h in sorted_lofi[:2]:
        if 'q0' not in h:
            h['hifi'] = 1e10; continue
        try:
            rv = Rotation.from_quat([h['q0'][1], h['q0'][2], h['q0'][3],
                                     h['q0'][0]]).as_rotvec()
            h['hifi'] = float(obj_hifi.evaluate(np.concatenate([rv, h['o0']])))
        except Exception:
            h['hifi'] = 1e10

    winner = min(sorted_lofi[:2], key=lambda h: h.get('hifi', 1e10))
    print(f"  Hi-fi: G{sorted_lofi[0]['idx']}({sorted_lofi[0].get('hifi','?'):.3f}) "
          f"vs G{sorted_lofi[1]['idx']}({sorted_lofi[1].get('hifi','?'):.3f}) "
          f"→ winner G{winner['idx']}")

    # Evaluate result
    if 'q0' in winner:
        q0_err = attitude_error_deg(winner['q0'], q0s[traj_idx])
        od_err = omega_direction_error(winner['o0'], omega0s[traj_idx])
        om_err = (abs(np.rad2deg(np.linalg.norm(winner['o0'])) -
                      omega_mags_arr[traj_idx]) / omega_mags_arr[traj_idx] * 100)
        conv = q0_err < 5.0 and od_err < 5.0 and om_err < 15.0
        status = "CONVERGED" if conv else "FAILED"
        print(f"  → q0_err={q0_err:.2f}, omega_dir={od_err:.2f}, "
              f"correct_hyp={'Y' if winner['correct'] else 'N'} [{status}]")
    else:
        q0_err = 180.0; od_err = 180.0; om_err = 100.0; conv = False

    dt = time.time() - t0
    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags_arr[traj_idx]),
        'q0_err': float(q0_err),
        'omega_dir_err': float(od_err),
        'omega_mag_err': float(om_err),
        'winner_idx': winner['idx'],
        'winner_correct': winner.get('correct', False),
        'lofi_gap': float(sorted_lofi[0].get('lofi', 0)) - float(sorted_lofi[1].get('lofi', 0))
            if len(sorted_lofi) >= 2 else 0,
        'hifi_gap': float(sorted_lofi[0].get('hifi', 0)) - float(sorted_lofi[1].get('hifi', 0))
            if len(sorted_lofi) >= 2 else 0,
        'converged': bool(conv),
        'runtime_s': float(dt),
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['converged'])
n_correct = sum(1 for r in valid if r.get('winner_correct', False))

print(f"\nConverged (q<5, omega_dir<5, mag<15%): {n_conv}/{len(valid)}")
print(f"Correct hypothesis: {n_correct}/{len(valid)}")

for r in valid:
    s = "OK" if r['converged'] else "FAIL"
    c = "Y" if r['winner_correct'] else "N"
    print(f"  Traj {r['traj_idx']} (ω={r['omega_dps']:.3f}): "
          f"q0={r['q0_err']:.1f}°, ωdir={r['omega_dir_err']:.1f}°, "
          f"hyp={c}, {r['runtime_s']:.0f}s [{s}]")

if [r for r in valid if r['converged']]:
    conv_errs = [r['q0_err'] for r in valid if r['converged']]
    print(f"\nConverged: median q0_err={np.median(conv_errs):.2f}°, "
          f"max={max(conv_errs):.2f}°")


# Save
plot_path = RESULTS_DIR / "micro51b_hifi_disambiguation.png"
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Micro-51b: Hi-fi Disambiguation", fontsize=14, fontweight='bold')
if valid:
    omegas = [r['omega_dps'] for r in valid]
    q0errs = [r['q0_err'] for r in valid]
    colors = ['green' if r['converged'] else 'red' for r in valid]
    ax.scatter(omegas, q0errs, c=colors, s=80, edgecolors='black', zorder=3)
    ax.axhline(5.0, color='gray', linestyle='--', alpha=0.5)
    for r in valid:
        lbl = f"G{r['winner_idx']}{'*' if r['winner_correct'] else ''}"
        ax.annotate(lbl, (r['omega_dps'], r['q0_err']), fontsize=7, ha='center')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('q0 error (deg)')
ax.set_title(f'Hi-fi disambiguation: {n_conv}/{len(valid)} converged')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot: {plot_path}")

json_path = RESULTS_DIR / "micro51b_hifi_disambiguation.json"
with open(str(json_path), 'w') as f:
    json.dump({'experiment': 'micro51b_hifi_disambiguation',
               'results': all_results,
               'summary': {'converged': n_conv, 'total': len(valid),
                           'correct_hyp': n_correct},
               'total_time_s': time.time() - t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating) else
                      int(x) if isinstance(x, np.integer) else x)
print(f"JSON: {json_path}")
print(f"\nTotal: {time.time() - t_global:.0f}s")
