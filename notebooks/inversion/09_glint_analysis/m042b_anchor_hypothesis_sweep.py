#!/usr/bin/env python3
"""Micro-42b -- Anchor hypothesis sweep.

Question: If we don't know which normal is glinting at the anchor epoch, can we
try all 14 as hypotheses (each giving a 4-DOF problem) and identify the correct
one by cost gap?  Does min-over-normals at non-anchor glints work?

Part 1: Evaluate cost at true parameters for each of 14 anchor hypotheses.
Part 2: Multi-start Nelder-Mead for 3 hypotheses (correct, close-wrong, far-wrong).
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
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin
from scipy.optimize import minimize

from lib.experiment_setup import (
    setup_experiment, save_results, attitude_error_deg,
)
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ===========================================================================
# Helpers (from m042)
# ===========================================================================

def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    """Quaternion (wxyz) that aligns n_body with pab via twist phi around n_body."""
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_sparse(q_anchor, omega_anchor, anchor_time, glint_times,
                     inertia_tensor):
    """Propagate from anchor to specific glint times only.

    Parameters
    ----------
    q_anchor : ndarray (4,) wxyz
    omega_anchor : ndarray (3,) rad/s body-frame
    anchor_time : float, seconds from epoch 0
    glint_times : ndarray (M,), seconds from epoch 0
    inertia_tensor : ndarray (3,3)

    Returns
    -------
    glint_quats : ndarray (M, 4) wxyz quaternions at each glint time
    """
    dt_from_anchor = glint_times - anchor_time
    n_glints = len(glint_times)
    glint_quats = np.zeros((n_glints, 4))

    # Forward glints (dt >= 0)
    fwd_mask = dt_from_anchor >= 0
    if fwd_mask.any():
        fwd_dts = dt_from_anchor[fwd_mask]
        fwd_times = np.concatenate([[0.0], fwd_dts])
        quats_fwd, _ = propagate_attitude(
            q_anchor, omega_anchor, fwd_times, "tumbling", inertia_tensor)
        glint_quats[fwd_mask] = quats_fwd[1:]  # skip anchor (index 0)

    # Backward glints (dt < 0)
    bwd_mask = dt_from_anchor < 0
    if bwd_mask.any():
        bwd_dts = -dt_from_anchor[bwd_mask][::-1]  # make positive, ascending
        bwd_times = np.concatenate([[0.0], bwd_dts])
        quats_bwd, _ = propagate_attitude(
            q_anchor, -omega_anchor, bwd_times, "tumbling", inertia_tensor)
        glint_quats[bwd_mask] = quats_bwd[1:][::-1]  # reverse back

    return glint_quats


def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    """Compute sum of min-over-normals alignment cost at each glint.

    For each glint, finds the normal with best alignment to PAB and
    uses (1 - dot)^2 as the cost.

    Returns total cost (float) and per-glint best normal indices.
    """
    n_glints = len(glint_quats)
    n_normals = len(all_normals)
    total_cost = 0.0
    best_normals = np.zeros(n_glints, dtype=int)

    for i in range(n_glints):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        # R^T @ n rotates body normal to inertial frame
        # Then dot with PAB_inertial
        best_dot = -np.inf
        best_j = 0
        for j in range(n_normals):
            n_inertial = R.T @ all_normals[j]
            dot_val = np.dot(n_inertial, glint_pab_arr[i])
            if dot_val > best_dot:
                best_dot = dot_val
                best_j = j

        alignment_error = 1.0 - best_dot
        total_cost += alignment_error ** 2
        best_normals[i] = best_j

    return total_cost, best_normals


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("m042b -- Anchor hypothesis sweep")
print("=" * 70)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)

n_obs = CTX.n_observations
print(f"Setup complete: {n_obs} observations, dt_sampling = {CTX.dt_sampling:.2f}s")

# Get full omega history (need true omega at anchor)
_, true_omega_history = propagate_attitude(
    CTX.true_q0, CTX.true_omega0, CTX.observation_times,
    "tumbling", CTX.inertia_tensor)


# ===========================================================================
# Load oracle glint data from m034
# ===========================================================================
print("\n--- Loading oracle glint data ---")

m034_npz = np.load(str(RESULTS_DIR / "m034_pab_alignment.npz"))
frac_flux = m034_npz['frac_flux']
unique_normals = m034_npz['unique_normals']  # (14, 3)
n_normals = len(unique_normals)

with open(str(RESULTS_DIR / "m034_pab_alignment.json")) as f:
    m034_json = json.load(f)
group_info = m034_json['all_groups']

# Detect bright peaks and oracle labels
peak_indices = argrelmin(CTX.true_lc, order=5)[0]
bright_mask = CTX.true_lc[peak_indices] < 9.0
bright_peaks = peak_indices[bright_mask]

oracle_labels = np.array([int(np.argmax(frac_flux[:, p])) for p in bright_peaks])
oracle_conf = np.array([float(frac_flux[oracle_labels[i], bright_peaks[i]])
                         for i in range(len(bright_peaks))])

# Filter to high-confidence
conf_mask = oracle_conf > 0.90
confident_peaks = bright_peaks[conf_mask]
confident_labels = oracle_labels[conf_mask]

print(f"Confident glints: {len(confident_peaks)}")
for i, (ep, lab) in enumerate(zip(confident_peaks, confident_labels)):
    print(f"  [{i}] epoch={ep}, mag={CTX.true_lc[ep]:.3f}, group={lab}, "
          f"comps={group_info[lab]['components']}")


# ===========================================================================
# Compute PAB inertial at all epochs
# ===========================================================================
sun_dir = (CTX.sun_pos - CTX.sat_pos)
sun_dir = sun_dir / np.linalg.norm(sun_dir, axis=1, keepdims=True)
obs_dir = (CTX.obs_pos - CTX.sat_pos)
obs_dir = obs_dir / np.linalg.norm(obs_dir, axis=1, keepdims=True)
pab_inertial = sun_dir + obs_dir
pab_inertial = pab_inertial / np.linalg.norm(pab_inertial, axis=1, keepdims=True)


# ===========================================================================
# Choose anchor epoch (same as m042: epoch 360, best alignment)
# ===========================================================================
ANCHOR_EPOCH = 360
anchor_pab = pab_inertial[ANCHOR_EPOCH]
anchor_time = CTX.observation_times[ANCHOR_EPOCH]
q_true_anchor = CTX.true_quaternions[ANCHOR_EPOCH]
omega_true_anchor = true_omega_history[ANCHOR_EPOCH]

# Oracle anchor normal
oracle_anchor_group = int(np.argmax(frac_flux[:, ANCHOR_EPOCH]))
print(f"\nAnchor: epoch {ANCHOR_EPOCH}, oracle group = G{oracle_anchor_group}")
print(f"  omega_true_at_anchor = {omega_true_anchor}")

# Non-anchor glint epochs and their PABs
non_anchor_mask = confident_peaks != ANCHOR_EPOCH
glint_epochs = confident_peaks[non_anchor_mask]
glint_pab_arr = pab_inertial[glint_epochs]
glint_times = CTX.observation_times[glint_epochs]
n_glints = len(glint_epochs)

print(f"Non-anchor glint epochs: {n_glints}")
print(f"  Epochs: {glint_epochs.tolist()}")


# ===========================================================================
# Part 1: Cost gap — evaluate all 14 anchor hypotheses at true parameters
# ===========================================================================
print("\n" + "=" * 70)
print("PART 1: Cost at true parameters for each anchor hypothesis")
print("=" * 70)

part1_results = []

for j in range(n_normals):
    n_body = unique_normals[j]

    # Find the phi on this hypothesis circle closest to true attitude
    R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                  q_true_anchor[3], q_true_anchor[0]])
    R0_j, _ = Rotation.align_vectors([n_body], [anchor_pab])
    R_twist = R_true * R0_j.inv()
    rotvec_twist = R_twist.as_rotvec()
    phi_j = float(np.dot(rotvec_twist, n_body))

    # Reconstruct anchor attitude from this hypothesis
    q_anchor_j = anchor_q_from_phi(phi_j, n_body, anchor_pab)
    att_err_j = attitude_error_deg(q_anchor_j, q_true_anchor)

    # Propagate from this (possibly wrong) anchor with true omega
    t0 = time.time()
    glint_quats = propagate_sparse(
        q_anchor_j, omega_true_anchor, anchor_time, glint_times,
        CTX.inertia_tensor)
    prop_time = time.time() - t0

    # Evaluate min-over-normals cost
    cost, best_normals = min_over_normals_cost(
        glint_quats, glint_pab_arr, unique_normals)

    is_correct = (j == oracle_anchor_group)
    marker = " <<<< CORRECT" if is_correct else ""

    comps = ','.join(group_info[j]['components'][:2])[:20]
    print(f"  G{j:>2d} ({comps:>20s}): phi={phi_j:>7.3f}, "
          f"att_err={att_err_j:>7.3f} deg, cost={cost:.6e}, "
          f"prop={prop_time:.3f}s{marker}")

    part1_results.append({
        'group': j,
        'components': group_info[j]['components'],
        'area': group_info[j]['total_area'],
        'normal': group_info[j]['normal'],
        'phi': phi_j,
        'attitude_error_deg': att_err_j,
        'cost': cost,
        'best_normals_at_glints': best_normals.tolist(),
        'is_correct': is_correct,
    })

# Sort by cost
sorted_results = sorted(part1_results, key=lambda r: r['cost'])
print(f"\nRanking by cost:")
for rank, r in enumerate(sorted_results):
    marker = " <<<< CORRECT" if r['is_correct'] else ""
    print(f"  #{rank+1}: G{r['group']:>2d} cost={r['cost']:.6e}, "
          f"att_err={r['attitude_error_deg']:.3f} deg{marker}")

correct_rank = next(i+1 for i, r in enumerate(sorted_results) if r['is_correct'])
correct_cost = next(r['cost'] for r in sorted_results if r['is_correct'])
runner_up_cost = sorted_results[1]['cost'] if sorted_results[0]['is_correct'] else sorted_results[0]['cost']
gap_ratio = runner_up_cost / (correct_cost + 1e-30)

print(f"\nCorrect hypothesis rank: #{correct_rank}")
print(f"Correct cost: {correct_cost:.6e}")
print(f"Runner-up cost: {runner_up_cost:.6e}")
print(f"Gap ratio: {gap_ratio:.1f}x")


# ===========================================================================
# Part 2: Multi-start optimisation for 3 hypotheses
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2: Multi-start Nelder-Mead for 3 anchor hypotheses")
print("=" * 70)

# Pick 3 hypotheses
CORRECT_GROUP = oracle_anchor_group
CLOSE_WRONG_GROUP = 0   # G0, 15 deg from G1 (the correct one)
FAR_WRONG_GROUP = 6     # G6, 90 deg away

hypotheses = [
    (CORRECT_GROUP, "correct"),
    (CLOSE_WRONG_GROUP, "close-wrong (15 deg)"),
    (FAR_WRONG_GROUP, "far-wrong (90 deg)"),
]


def build_cost_fn(hypothesis_normal, anchor_pab_vec, anchor_time_val,
                  glint_times_arr, glint_pab_array, all_normals_arr,
                  inertia):
    """Build a closure for the NLP cost function."""

    def cost_fn(params):
        phi = params[0]
        omega = params[1:4]

        q_anchor = anchor_q_from_phi(phi, hypothesis_normal, anchor_pab_vec)

        try:
            gq = propagate_sparse(
                q_anchor, omega, anchor_time_val, glint_times_arr, inertia)
        except Exception:
            return 1e10

        cost, _ = min_over_normals_cost(gq, glint_pab_array, all_normals_arr)
        return cost

    return cost_fn


np.random.seed(42)

# Starting points: 10 phi × 2 omega = 20
phi_starts = np.linspace(0, 2 * np.pi, 10, endpoint=False)
omega_mag = np.linalg.norm(omega_true_anchor)

omega_perturbations = []
omega_labels = []
for _ in range(1):
    # 10% perturbation
    omega_perturbations.append(
        omega_true_anchor + np.random.randn(3) * omega_mag * 0.1)
    omega_labels.append('10%')
for _ in range(1):
    # 50% perturbation
    omega_perturbations.append(
        omega_true_anchor + np.random.randn(3) * omega_mag * 0.5)
    omega_labels.append('50%')

all_hypothesis_results = {}

for hyp_group, hyp_label in hypotheses:
    print(f"\n--- Hypothesis: G{hyp_group} ({hyp_label}) ---")
    hyp_normal = unique_normals[hyp_group]

    cost_fn = build_cost_fn(
        hyp_normal, anchor_pab, anchor_time, glint_times, glint_pab_arr,
        unique_normals, CTX.inertia_tensor)

    # Find true phi for this hypothesis (for reference)
    R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                  q_true_anchor[3], q_true_anchor[0]])
    R0_h, _ = Rotation.align_vectors([hyp_normal], [anchor_pab])
    R_twist = R_true * R0_h.inv()
    true_phi_h = float(np.dot(R_twist.as_rotvec(), hyp_normal))

    print(f"  True phi for this hypothesis: {true_phi_h:.4f} rad")
    print(f"  Cost at truth: {cost_fn(np.array([true_phi_h, *omega_true_anchor])):.6e}")

    results_this = []
    t_hyp = time.time()
    start_count = 0

    for phi_idx, phi_val in enumerate(phi_starts):
        for omega_pert, omega_lab in zip(omega_perturbations, omega_labels):
            start_count += 1
            x0 = np.array([phi_val, omega_pert[0], omega_pert[1], omega_pert[2]])

            try:
                res = minimize(
                    cost_fn, x0=x0, method='Nelder-Mead',
                    options={'maxfev': 200, 'xatol': 1e-10, 'fatol': 1e-14,
                             'adaptive': True})

                # Score the solution
                phi_found = res.x[0]
                omega_found = res.x[1:4]
                q_found = anchor_q_from_phi(phi_found, hyp_normal, anchor_pab)
                att_err = attitude_error_deg(q_found, q_true_anchor)

                omega_dot = np.dot(omega_found, omega_true_anchor)
                omega_norms = (np.linalg.norm(omega_found) *
                               np.linalg.norm(omega_true_anchor))
                omega_dir_err = np.rad2deg(np.arccos(
                    np.clip(omega_dot / (omega_norms + 1e-30), -1, 1)))
                omega_mag_err = (100.0 * abs(np.linalg.norm(omega_found) -
                                 np.linalg.norm(omega_true_anchor)) /
                                 np.linalg.norm(omega_true_anchor))

                success = (att_err < 5.0 and omega_dir_err < 5.0
                           and omega_mag_err < 5.0)

                results_this.append({
                    'phi_init': float(phi_val),
                    'omega_label': omega_lab,
                    'cost': float(res.fun),
                    'nfev': int(res.nfev),
                    'att_err': float(att_err),
                    'omega_dir_err': float(omega_dir_err),
                    'omega_mag_err': float(omega_mag_err),
                    'success': bool(success),
                    'params': res.x.tolist(),
                })
            except Exception as e:
                results_this.append({
                    'phi_init': float(phi_val),
                    'omega_label': omega_lab,
                    'cost': 1e10,
                    'error': str(e),
                    'success': False,
                })

            if start_count % 5 == 0:
                elapsed_hyp = time.time() - t_hyp
                print(f"    [{start_count}/{len(phi_starts)*len(omega_perturbations)}] "
                      f"{elapsed_hyp:.0f}s elapsed", flush=True)

    dt_hyp = time.time() - t_hyp
    n_success = sum(1 for r in results_this if r.get('success', False))
    best = min(results_this, key=lambda r: r['cost'])

    print(f"  Completed {len(results_this)} starts in {dt_hyp:.1f}s")
    print(f"  Success: {n_success}/{len(results_this)}")
    print(f"  Best cost: {best['cost']:.6e}")
    if 'att_err' in best:
        print(f"  Best att_err: {best['att_err']:.3f} deg")
        print(f"  Best omega_dir_err: {best['omega_dir_err']:.3f} deg")
        print(f"  Best omega_mag_err: {best['omega_mag_err']:.3f}%")

    all_hypothesis_results[hyp_label] = {
        'group': hyp_group,
        'label': hyp_label,
        'n_starts': len(results_this),
        'n_success': n_success,
        'time_s': dt_hyp,
        'best_cost': best['cost'],
        'best_att_err': best.get('att_err', None),
        'best_omega_dir_err': best.get('omega_dir_err', None),
        'best_omega_mag_err': best.get('omega_mag_err', None),
        'all_results': results_this,
    }


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nPart 1: Cost gap")
print(f"  Correct hypothesis (G{oracle_anchor_group}): rank #{correct_rank}, "
      f"cost={correct_cost:.6e}")
print(f"  Runner-up: cost={runner_up_cost:.6e}")
print(f"  Gap: {gap_ratio:.1f}x")

print(f"\nPart 2: Multi-start convergence")
for hyp_label, hyp_data in all_hypothesis_results.items():
    print(f"  {hyp_label}: {hyp_data['n_success']}/{hyp_data['n_starts']} success, "
          f"best_cost={hyp_data['best_cost']:.6e}, "
          f"best_att={hyp_data.get('best_att_err', 'N/A')}, "
          f"time={hyp_data['time_s']:.1f}s")


# ===========================================================================
# Plot: 3 panels
# ===========================================================================
print("\n--- Generating plot ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Micro-42b: Anchor Hypothesis Sweep", fontsize=14, fontweight='bold')

# Panel 1: Part 1 cost gap (bar chart, all 14 hypotheses)
ax1 = axes[0]
costs_sorted = [r['cost'] for r in sorted_results]
groups_sorted = [f"G{r['group']}" for r in sorted_results]
colors = ['green' if r['is_correct'] else 'steelblue' for r in sorted_results]
bars = ax1.bar(range(len(costs_sorted)), costs_sorted, color=colors,
               edgecolor='black', linewidth=0.5)
ax1.set_xticks(range(len(groups_sorted)))
ax1.set_xticklabels(groups_sorted, fontsize=7, rotation=45)
ax1.set_ylabel('Cost at true params')
ax1.set_title(f'Part 1: Cost gap (correct=green)\nGap={gap_ratio:.0f}x')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Part 2 convergence — cost vs phi_init for each hypothesis
ax2 = axes[1]
markers = {'correct': 'o', 'close-wrong (15 deg)': 's', 'far-wrong (90 deg)': '^'}
colors_hyp = {'correct': 'green', 'close-wrong (15 deg)': 'orange',
              'far-wrong (90 deg)': 'red'}
for hyp_label, hyp_data in all_hypothesis_results.items():
    phi_inits = [r['phi_init'] for r in hyp_data['all_results']]
    costs = [r['cost'] for r in hyp_data['all_results']]
    ax2.scatter(np.rad2deg(phi_inits), costs,
                marker=markers.get(hyp_label, 'o'),
                color=colors_hyp.get(hyp_label, 'gray'),
                s=40, alpha=0.7, edgecolors='black', linewidth=0.3,
                label=f"G{hyp_data['group']} ({hyp_label})")

ax2.set_xlabel('phi_init (deg)')
ax2.set_ylabel('Converged cost')
ax2.set_title('Part 2: Converged cost by start')
ax2.set_yscale('log')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# Panel 3: Part 2 attitude + omega error for correct hypothesis
ax3 = axes[2]
correct_results = all_hypothesis_results.get('correct', {}).get('all_results', [])
if correct_results:
    att_errs = [r.get('att_err', 180) for r in correct_results]
    omega_dir_errs = [r.get('omega_dir_err', 180) for r in correct_results]
    successes = [r.get('success', False) for r in correct_results]
    c = ['green' if s else 'red' for s in successes]
    ax3.scatter(att_errs, omega_dir_errs, c=c, s=40, alpha=0.7,
                edgecolors='black', linewidth=0.3)
    ax3.axvline(5.0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(5.0, color='gray', linestyle='--', alpha=0.5)

ax3.set_xlabel('Attitude error (deg)')
ax3.set_ylabel('Omega direction error (deg)')
ax3.set_title('Part 2: Convergence (correct hyp)')
ax3.grid(True, alpha=0.3)
legend_items = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markeredgecolor='black', markersize=8, label='Success'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=8, label='Failure'),
]
ax3.legend(handles=legend_items, fontsize=8)

plt.tight_layout()
plot_path = RESULTS_DIR / "m042b_anchor_hypothesis_sweep.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save results
# ===========================================================================
print("\n--- Saving results ---")

results = {
    'experiment': 'm042b_anchor_hypothesis_sweep',
    'anchor_epoch': ANCHOR_EPOCH,
    'oracle_anchor_group': oracle_anchor_group,
    'n_confident_glints': int(len(confident_peaks)),
    'n_non_anchor_glints': n_glints,

    'part1_cost_gap': {
        'correct_rank': correct_rank,
        'correct_cost': correct_cost,
        'runner_up_cost': runner_up_cost,
        'gap_ratio': gap_ratio,
        'all_hypotheses': part1_results,
    },

    'part2_multistart': {
        label: {
            'group': data['group'],
            'n_starts': data['n_starts'],
            'n_success': data['n_success'],
            'best_cost': data['best_cost'],
            'best_att_err': data.get('best_att_err'),
            'best_omega_dir_err': data.get('best_omega_dir_err'),
            'best_omega_mag_err': data.get('best_omega_mag_err'),
            'time_s': data['time_s'],
            'all_results': data['all_results'],
        }
        for label, data in all_hypothesis_results.items()
    },

    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "m042b_anchor_hypothesis_sweep.json"
save_results(str(json_path), results)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
