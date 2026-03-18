#!/usr/bin/env python3
"""Micro-51 -- End-to-end glint-anchored inversion pipeline.

Architecture (building on micro42b):
1. Peak detection from observed LC
2. Phi sweep: 10 hypotheses x 36 phi (oracle omega) → best hypothesis
3. LC residual at 10 sampled epochs → break antiparallel degeneracy
4. Nelder-Mead refinement on 4D (phi + omega)
5. Propagate to t=0 → (q0, omega0) estimate

Part A: Characterise omega convergence basin.  For 10 trajectories with oracle
        phi, perturb omega by 0%, 1%, 2%, 5%, 10%, 20%, 50% and run NM.
        Find the maximum perturbation at which convergence still occurs.

Part B: Full pipeline on 10 trajectories with oracle omega.  Report success
        rate (attitude < 5 deg, omega direction < 5 deg).

Part C: Full pipeline with perturbed omega.  Use 5% perturbation on each
        trajectory and test if the pipeline still converges.
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

from lib.experiment_setup import (
    setup_experiment, brightness_single_epoch, attitude_error_deg, save_results,
)
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"


# ===========================================================================
# Helpers
# ===========================================================================

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
    near = np.abs(dt) <= 1e-6
    tq[near] = q_anchor

    if fwd.any():
        ft = np.concatenate([[0.0], dt[fwd]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        bt_full = np.concatenate([[0.0], bt])
        qb, _ = propagate_attitude(q_anchor, -omega, bt_full, "tumbling", I_tensor)
        tq[bwd] = qb[1:][::-1]
    return tq


def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    n_glints = len(glint_quats)
    total_cost = 0.0
    for i in range(n_glints):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pab_arr[i])
                       for j in range(len(all_normals)))
        total_cost += (1.0 - best_dot) ** 2
    return total_cost


def omega_direction_error(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def lc_residual_at_samples(q_anchor, omega, anchor_time, sample_epochs,
                           obs_times, I_tensor, ctx, observed_lc):
    """Evaluate lo-fi brightness at sampled epochs and compare with observed."""
    sample_times = obs_times[sample_epochs]
    quats = propagate_sparse(q_anchor, omega, anchor_time, sample_times, I_tensor)

    mse = 0.0
    for i, ep in enumerate(sample_epochs):
        mag_pred = brightness_single_epoch(quats[i], ep, ctx, use_shadows=False)
        mse += (mag_pred - observed_lc[ep]) ** 2
    return mse / len(sample_epochs)


# ===========================================================================
# Setup satellite model (shared across all trajectories)
# ===========================================================================
print("=" * 70)
print("micro51 -- End-to-end glint-anchored inversion pipeline")
print("=" * 70)
t_global = time.time()

print("\n--- Setting up satellite model ---")
# Use setup_experiment just for satellite/SPICE/geometry (not the truth trajectory)
CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Satellite model ready. {CTX.n_observations} epochs, dt={CTX.dt_sampling:.2f}s")

# Load micro46 data
print("\n--- Loading micro46 data ---")
master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)
observation_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
inertia_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']

n_normals = len(unique_normals)
# Verify geometry matches
assert np.allclose(observation_times, CTX.observation_times, atol=0.1), \
    "micro46 and CTX observation times don't match!"


# ===========================================================================
# Pipeline function
# ===========================================================================

def run_pipeline(traj_idx, omega_perturbation_pct=0.0, rng_seed=42):
    """Run the full glint-anchored inversion pipeline on one trajectory.

    Parameters
    ----------
    traj_idx : int
        Index into micro46 dataset.
    omega_perturbation_pct : float
        Percentage perturbation on oracle omega (0 = oracle, 10 = 10%).
    rng_seed : int
        Random seed for perturbation.

    Returns dict with all pipeline results.
    """
    t0 = time.time()
    mags_t = mag_hifi[traj_idx]
    frac_flux_t = group_frac_flux[traj_idx]
    quats_t = quaternions[traj_idx]
    q0_true = q0s[traj_idx]
    omega0_true = omega0s[traj_idx]

    # Get omega history
    _, omega_hist = propagate_attitude(
        q0_true, omega0_true, observation_times, "tumbling", inertia_tensor)

    # --- Step 1: Peak detection ---
    peaks_idx, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_peaks = peaks_idx[mags_t[peaks_idx] < 9.0]

    if len(bright_peaks) < 2:
        return {'traj_idx': int(traj_idx), 'error': f'only {len(bright_peaks)} bright peaks'}

    # Oracle labels and confident filtering
    oracle_labels = [int(np.argmax(frac_flux_t[:, p])) for p in bright_peaks]
    oracle_conf = [float(frac_flux_t[oracle_labels[i], bright_peaks[i]])
                   for i in range(len(bright_peaks))]
    conf_mask = [c > 0.77 for c in oracle_conf]
    confident_peaks = bright_peaks[[i for i, c in enumerate(conf_mask) if c]]

    if len(confident_peaks) < 2:
        return {'traj_idx': int(traj_idx), 'error': 'too few confident peaks'}

    # --- Step 2: Choose anchor ---
    anchor_epoch = int(confident_peaks[np.argmin(mags_t[confident_peaks])])
    q_true_anchor = quats_t[anchor_epoch]
    omega_true_anchor = omega_hist[anchor_epoch]
    true_mag_dps = np.rad2deg(np.linalg.norm(omega_true_anchor))
    oracle_group = int(np.argmax(frac_flux_t[:, anchor_epoch]))
    anchor_time = observation_times[anchor_epoch]

    # Non-anchor glint epochs
    non_anchor = confident_peaks[confident_peaks != anchor_epoch]
    glint_pabs = pab_j2000[non_anchor]
    glint_times = observation_times[non_anchor]

    # Prepare omega (with optional perturbation)
    rng = np.random.RandomState(rng_seed)
    if omega_perturbation_pct > 0:
        omega_mag = np.linalg.norm(omega_true_anchor)
        perturbation = rng.randn(3) * omega_mag * (omega_perturbation_pct / 100.0)
        omega_start = omega_true_anchor + perturbation
    else:
        omega_start = omega_true_anchor.copy()

    omega_dir_err_start = omega_direction_error(omega_start, omega_true_anchor)

    # --- Step 3: Phi sweep (10 hypotheses x 36 phi) ---
    N_PHI = 36
    phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    hypothesis_results = []

    for hyp_idx in range(n_normals):
        n_body = unique_normals[hyp_idx]
        best_cost = np.inf
        best_phi = 0.0

        for phi in phi_values:
            q_anchor = anchor_q_from_phi(phi, n_body, pab_j2000[anchor_epoch])
            try:
                gq = propagate_sparse(q_anchor, omega_start, anchor_time,
                                      glint_times, inertia_tensor)
                cost = min_over_normals_cost(gq, glint_pabs, unique_normals)
            except Exception:
                cost = 1e10

            if cost < best_cost:
                best_cost = cost
                best_phi = phi

        hypothesis_results.append({
            'hyp_idx': hyp_idx,
            'best_cost': best_cost,
            'best_phi': best_phi,
            'is_correct': hyp_idx == oracle_group,
        })

    # Sort by cost
    sorted_hyps = sorted(hypothesis_results, key=lambda h: h['best_cost'])

    # --- Step 4: LC residual for antiparallel disambiguation ---
    # Top 2 hypotheses — evaluate LC residual to break tie
    top2 = sorted_hyps[:2]
    # Sample 10 non-glint epochs spread across the observation window
    non_glint_epochs = np.array([e for e in range(0, 500, 50)
                                  if e not in confident_peaks])[:10]

    for hyp in top2:
        q_test = anchor_q_from_phi(hyp['best_phi'], unique_normals[hyp['hyp_idx']],
                                   pab_j2000[anchor_epoch])
        try:
            lc_mse = lc_residual_at_samples(
                q_test, omega_start, anchor_time, non_glint_epochs,
                observation_times, inertia_tensor, CTX, mags_t)
        except Exception:
            lc_mse = 1e10
        hyp['lc_mse'] = lc_mse

    # Re-rank by combined score (glint cost primary, LC residual for tie-breaking)
    # If top2 have similar glint cost (within 2x), use LC to break tie
    if top2[0]['best_cost'] > 0 and top2[1]['best_cost'] / top2[0]['best_cost'] < 2.0:
        top2_sorted = sorted(top2, key=lambda h: h.get('lc_mse', 1e10))
        winner = top2_sorted[0]
    else:
        winner = top2[0]

    winner_hyp = winner['hyp_idx']
    winner_phi = winner['best_phi']
    q_winner = anchor_q_from_phi(winner_phi, unique_normals[winner_hyp],
                                 pab_j2000[anchor_epoch])
    att_err_anchor = attitude_error_deg(q_winner, q_true_anchor)

    # --- Step 5: Nelder-Mead refinement on 4D (phi + omega) ---
    def nm_cost(params):
        phi = params[0]
        omega = params[1:4]
        q_a = anchor_q_from_phi(phi, unique_normals[winner_hyp],
                                pab_j2000[anchor_epoch])
        try:
            gq = propagate_sparse(q_a, omega, anchor_time, glint_times,
                                  inertia_tensor)
            return min_over_normals_cost(gq, glint_pabs, unique_normals)
        except Exception:
            return 1e10

    x0 = np.array([winner_phi, *omega_start])
    try:
        nm_result = minimize(nm_cost, x0, method='Nelder-Mead',
                             options={'maxfev': 400, 'xatol': 1e-8,
                                      'fatol': 1e-12, 'adaptive': True})
        phi_refined = nm_result.x[0]
        omega_refined = nm_result.x[1:4]
    except Exception:
        phi_refined = winner_phi
        omega_refined = omega_start

    q_refined = anchor_q_from_phi(phi_refined, unique_normals[winner_hyp],
                                  pab_j2000[anchor_epoch])
    att_err_refined = attitude_error_deg(q_refined, q_true_anchor)
    omega_dir_err_refined = omega_direction_error(omega_refined, omega_true_anchor)
    omega_mag_err_refined = (abs(np.rad2deg(np.linalg.norm(omega_refined)) - true_mag_dps)
                             / true_mag_dps * 100)

    # --- Step 6: Propagate back to t=0 ---
    # q_refined is the attitude at anchor_epoch. Propagate backward to t=0.
    dt_to_start = -anchor_time  # anchor_time is relative to epoch 0
    times_back = np.array([0.0, -dt_to_start])  # [0, anchor_time] reversed
    quats_back, omegas_back = propagate_attitude(
        q_refined, -omega_refined, times_back, "tumbling", inertia_tensor)
    q0_est = quats_back[-1]
    omega0_est = -omegas_back[-1]  # reverse the sign back

    q0_err = attitude_error_deg(q0_est, q0_true)
    omega0_dir_err = omega_direction_error(omega0_est, omega0_true)
    omega0_mag_err = (abs(np.rad2deg(np.linalg.norm(omega0_est)) -
                          np.rad2deg(np.linalg.norm(omega0_true)))
                      / np.rad2deg(np.linalg.norm(omega0_true)) * 100)

    converged = (q0_err < 5.0 and omega0_dir_err < 5.0 and omega0_mag_err < 15.0)

    dt_total = time.time() - t0

    return {
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags_arr[traj_idx]),
        'true_mag_dps': float(true_mag_dps),
        'n_confident_peaks': len(confident_peaks),
        'anchor_epoch': anchor_epoch,
        'oracle_group': oracle_group,
        'winner_hyp': winner_hyp,
        'winner_correct': winner_hyp == oracle_group,
        'omega_pert_pct': omega_perturbation_pct,
        'omega_dir_err_start': float(omega_dir_err_start),
        'att_err_anchor_pre_nm': float(att_err_anchor),
        'att_err_anchor_post_nm': float(att_err_refined),
        'omega_dir_err_post_nm': float(omega_dir_err_refined),
        'omega_mag_err_post_nm': float(omega_mag_err_refined),
        'q0_err_deg': float(q0_err),
        'omega0_dir_err_deg': float(omega0_dir_err),
        'omega0_mag_err_pct': float(omega0_mag_err),
        'converged': bool(converged),
        'runtime_s': float(dt_total),
    }


# ===========================================================================
# Select test trajectories
# ===========================================================================
omega_sorted = np.argsort(omega_mags_arr)
candidates = []
for idx in omega_sorted:
    peaks, _ = find_peaks(-mag_hifi[idx], distance=5, prominence=0.3)
    if np.sum(mag_hifi[idx][peaks] < 9.0) >= 3:
        candidates.append(idx)
select_idx = np.linspace(0, len(candidates) - 1, 10, dtype=int)
TEST_TRAJS = [candidates[i] for i in select_idx]
print(f"\nTest trajectories: {TEST_TRAJS}")
print(f"Omega mags: {[f'{omega_mags_arr[t]:.3f}' for t in TEST_TRAJS]}")


# ===========================================================================
# Part A: Omega convergence basin characterisation
# ===========================================================================
print("\n" + "=" * 70)
print("PART A: Omega convergence basin (10 trajectories, varying perturbation)")
print("=" * 70)

PERT_LEVELS = [0, 1, 2, 5, 10, 20, 50]
basin_results = []

for traj_idx in TEST_TRAJS[:5]:  # 5 trajectories for speed
    print(f"\n  Traj {traj_idx} (omega={omega_mags_arr[traj_idx]:.3f} deg/s):")
    for pert in PERT_LEVELS:
        result = run_pipeline(traj_idx, omega_perturbation_pct=pert, rng_seed=42)
        if 'error' in result:
            print(f"    {pert:>3d}%: SKIP ({result['error']})")
            continue
        status = "OK" if result['converged'] else "FAIL"
        print(f"    {pert:>3d}%: q0_err={result['q0_err_deg']:>6.2f}, "
              f"omega_dir={result['omega0_dir_err_deg']:>6.2f}, "
              f"omega_mag={result['omega0_mag_err_pct']:>5.1f}% [{status}]")
        basin_results.append(result)


# ===========================================================================
# Part B: Full pipeline with oracle omega (10 trajectories)
# ===========================================================================
print("\n" + "=" * 70)
print("PART B: Full pipeline, oracle omega (10 trajectories)")
print("=" * 70)

part_b_results = []
for traj_idx in TEST_TRAJS:
    result = run_pipeline(traj_idx, omega_perturbation_pct=0)
    part_b_results.append(result)

    if 'error' in result:
        print(f"  Traj {traj_idx}: SKIP ({result['error']})")
    else:
        status = "CONVERGED" if result['converged'] else "FAILED"
        print(f"  Traj {traj_idx} (omega={result['omega_dps']:.3f}): "
              f"q0={result['q0_err_deg']:.2f} deg, "
              f"omega_dir={result['omega0_dir_err_deg']:.2f} deg, "
              f"correct_hyp={'Y' if result['winner_correct'] else 'N'} "
              f"[{status}] {result['runtime_s']:.1f}s")

valid_b = [r for r in part_b_results if 'error' not in r]
n_conv_b = sum(1 for r in valid_b if r['converged'])
print(f"\nOracle omega: {n_conv_b}/{len(valid_b)} converged")


# ===========================================================================
# Part C: Pipeline with 5% omega perturbation (10 trajectories, 5 seeds each)
# ===========================================================================
print("\n" + "=" * 70)
print("PART C: Full pipeline, 5% omega perturbation (10 trajectories, 5 seeds)")
print("=" * 70)

part_c_results = []
for traj_idx in TEST_TRAJS:
    traj_results = []
    for seed in range(5):
        result = run_pipeline(traj_idx, omega_perturbation_pct=5, rng_seed=seed * 100)
        traj_results.append(result)

    valid = [r for r in traj_results if 'error' not in r]
    n_conv = sum(1 for r in valid if r['converged'])
    if valid:
        best = min(valid, key=lambda r: r['q0_err_deg'])
        print(f"  Traj {traj_idx} (omega={omega_mags_arr[traj_idx]:.3f}): "
              f"{n_conv}/5 converged, best q0_err={best['q0_err_deg']:.2f} deg")
    else:
        print(f"  Traj {traj_idx}: all skipped")

    part_c_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags_arr[traj_idx]),
        'n_converged': n_conv,
        'n_valid': len(valid),
        'results': traj_results,
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nPart A: Convergence basin (5 trajectories)")
for traj_idx in TEST_TRAJS[:5]:
    traj_basin = [r for r in basin_results if r['traj_idx'] == traj_idx]
    if not traj_basin:
        continue
    max_pert_converged = max((r['omega_pert_pct'] for r in traj_basin
                              if r.get('converged', False)), default=-1)
    print(f"  Traj {traj_idx} (omega={omega_mags_arr[traj_idx]:.3f}): "
          f"max converged perturbation = {max_pert_converged}%")

valid_b = [r for r in part_b_results if 'error' not in r]
n_conv_b = sum(1 for r in valid_b if r['converged'])
n_correct_hyp = sum(1 for r in valid_b if r.get('winner_correct', False))
print(f"\nPart B: Oracle omega — {n_conv_b}/{len(valid_b)} converged, "
      f"{n_correct_hyp}/{len(valid_b)} correct hypothesis")
if valid_b:
    q0_errs = [r['q0_err_deg'] for r in valid_b if r['converged']]
    if q0_errs:
        print(f"  Converged: median q0_err={np.median(q0_errs):.2f} deg, "
              f"max={max(q0_errs):.2f} deg")

n_traj_c_converged = sum(1 for r in part_c_results if r['n_converged'] > 0)
print(f"\nPart C: 5% omega perturbation — "
      f"{n_traj_c_converged}/{len(part_c_results)} trajectories have ≥1 convergence")


# ===========================================================================
# Plot
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Micro-51: End-to-End Pipeline Results", fontsize=14,
             fontweight='bold')

# Panel 1: Basin characterisation
ax = axes[0]
for traj_idx in TEST_TRAJS[:5]:
    traj_data = [r for r in basin_results if r['traj_idx'] == traj_idx]
    if not traj_data:
        continue
    perts = [r['omega_pert_pct'] for r in traj_data]
    q0_errs = [r['q0_err_deg'] for r in traj_data]
    ax.plot(perts, q0_errs, 'o-', markersize=5,
            label=f'ω={omega_mags_arr[traj_idx]:.2f}')
ax.axhline(5.0, color='red', linestyle='--', alpha=0.5, label='5 deg threshold')
ax.set_xlabel('Omega perturbation (%)')
ax.set_ylabel('q0 error (deg)')
ax.set_title('Part A: Convergence Basin')
ax.legend(fontsize=6)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Panel 2: Oracle omega results
ax = axes[1]
valid_b2 = [r for r in valid_b]
if valid_b2:
    omegas = [r['omega_dps'] for r in valid_b2]
    q0_errs = [r['q0_err_deg'] for r in valid_b2]
    colors = ['green' if r['converged'] else 'red' for r in valid_b2]
    ax.scatter(omegas, q0_errs, c=colors, s=80, edgecolors='black', zorder=3)
    ax.axhline(5.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('q0 error (deg)')
ax.set_title(f'Part B: Oracle omega ({n_conv_b}/{len(valid_b)} converged)')
ax.grid(True, alpha=0.3)

# Panel 3: 5% perturbation
ax = axes[2]
for r in part_c_results:
    valid = [x for x in r['results'] if 'error' not in x]
    if valid:
        errs = [x['q0_err_deg'] for x in valid]
        conv = [x['converged'] for x in valid]
        ax.scatter([r['omega_dps']] * len(errs), errs,
                   c=['green' if c else 'red' for c in conv],
                   s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
ax.axhline(5.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('q0 error (deg)')
ax.set_title(f'Part C: 5% perturbation ({n_traj_c_converged}/10 have ≥1 conv)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "micro51_end_to_end.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save
# ===========================================================================
results_json = {
    'experiment': 'micro51_end_to_end',
    'part_a_basin': basin_results,
    'part_b_oracle': part_b_results,
    'part_c_perturbed': [{k: v for k, v in r.items() if k != 'results'}
                         for r in part_c_results],
    'summary': {
        'oracle_converged': f"{n_conv_b}/{len(valid_b)}",
        'perturbed_converged': f"{n_traj_c_converged}/{len(part_c_results)}",
    },
    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "micro51_end_to_end.json"
with open(str(json_path), 'w') as f:
    json.dump(results_json, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
