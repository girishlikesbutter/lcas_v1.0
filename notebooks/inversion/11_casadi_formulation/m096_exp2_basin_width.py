#!/usr/bin/env python3
"""
m096 Exp 2: Alignment Cost Basin Width at Truth.

Question: For each seed, how far can you perturb omega from truth before
the alignment cost doubles? Maps fundamental solvability.
"""

import sys, os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
CKPT = RESULTS_DIR / "m096_exp2_basin_width"
CKPT.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
pab_j2000 = master['pab_j2000']

CONSTRAINT_WEIGHT = 10.0
N_PERTURBATIONS = 200
PERTURBATION_ANGLES_DEG = np.concatenate([
    np.linspace(0.5, 5, 20),   # fine: 0.5° to 5°
    np.linspace(6, 15, 20),    # medium: 6° to 15°
    np.linspace(16, 30, 10),   # coarse: 16° to 30°
])  # 50 angles, each tested at 4 random directions = 200 perturbations

Z_NORMALS = {4, 5}

def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def propagate_delta_qs(omega_vec, dt_arr, I_tens):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]), "tumbling", I_tens)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]), "tumbling", I_tens)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs

def eval_alignment_cost(omega_vec, dt_constraints, pab_at_constraints,
                        constraint_allowed, qa_anchor_sets, normals, weight, I_tens):
    """Evaluate min-over-phi alignment cost for a single omega."""
    dqs = propagate_delta_qs(omega_vec, dt_constraints, I_tens)
    best_cost = np.inf
    for ni, qa_xyzw in qa_anchor_sets:
        n_phi = len(qa_xyzw)
        R_anchors = Rotation.from_quat(qa_xyzw)
        costs = np.zeros(n_phi)
        for ci in range(len(dqs)):
            dq = dqs[ci]
            R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
            R_all = R_anchors * R_delta
            pbs = R_all.apply(pab_at_constraints[ci])
            allowed = constraint_allowed[ci]
            bds = (pbs @ normals[allowed].T).max(axis=1)
            costs += weight * (1.0 - bds) ** 2
        bc = costs.min()
        if bc < best_cost:
            best_cost = bc
    return best_cost


ALL_SEEDS = list(range(100))
rng = np.random.default_rng(123)
summary = []

for seed in ALL_SEEDS:
    d = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)

    if not bool(d['valid']):
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
        summary.append({'seed': seed, 'error': 'invalid_stage1'})
        continue

    n_constraints = int(d['n_constraints'])
    if n_constraints < 1:
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
        summary.append({'seed': seed, 'error': 'no_constraints'})
        continue

    true_omega_anchor = d['true_omega_anchor']
    dt_constraints = d['dt_constraints']
    pab_at_constraints = d['pab_at_constraints']
    anchor_allowed = d['anchor_allowed']
    constraint_allowed_padded = d['constraint_allowed_padded']
    constraint_allowed_counts = d['constraint_allowed_counts']

    # Reconstruct constraint_allowed as list of lists
    constraint_allowed = []
    for ci in range(n_constraints):
        nc = int(constraint_allowed_counts[ci])
        constraint_allowed.append(constraint_allowed_padded[ci, :nc].tolist())

    # Reconstruct qa_anchor_sets
    n_sets = int(d['qa_anchor_n_sets'])
    qa_anchor_sets = []
    for i in range(n_sets):
        ni = int(d['qa_anchor_ni'][i])
        qa_xyzw = d[f'qa_xyzw_{i}']
        qa_anchor_sets.append((ni, qa_xyzw))

    # Cost at truth
    truth_cost = eval_alignment_cost(
        true_omega_anchor, dt_constraints, pab_at_constraints,
        constraint_allowed, qa_anchor_sets, unique_normals, CONSTRAINT_WEIGHT, I_tensor)

    # Perturbations: for each angle, 4 random directions perpendicular to truth
    true_dir = true_omega_anchor / np.linalg.norm(true_omega_anchor)
    true_mag = np.linalg.norm(true_omega_anchor)

    all_angles = []
    all_costs = []
    all_perturb_dirs = []
    all_perturb_omegas = []

    for angle_deg in PERTURBATION_ANGLES_DEG:
        angle_rad = np.deg2rad(angle_deg)
        for _ in range(4):
            # Random perpendicular direction
            rand_vec = rng.standard_normal(3)
            rand_vec -= np.dot(rand_vec, true_dir) * true_dir
            rand_vec /= np.linalg.norm(rand_vec)

            # Rotate true direction by angle around rand_vec
            R_perturb = Rotation.from_rotvec(angle_rad * rand_vec)
            perturbed_dir = R_perturb.apply(true_dir)
            perturbed_omega = perturbed_dir * true_mag

            cost = eval_alignment_cost(
                perturbed_omega, dt_constraints, pab_at_constraints,
                constraint_allowed, qa_anchor_sets, unique_normals, CONSTRAINT_WEIGHT, I_tensor)

            all_angles.append(angle_deg)
            all_costs.append(cost)
            all_perturb_dirs.append(perturbed_dir)
            all_perturb_omegas.append(perturbed_omega)

    all_angles = np.array(all_angles)
    all_costs = np.array(all_costs)
    all_perturb_dirs = np.array(all_perturb_dirs)
    all_perturb_omegas = np.array(all_perturb_omegas)

    # Basin half-width: angle where median cost reaches 2× truth
    # Group by angle, take median cost at each angle
    unique_angles = np.unique(all_angles)
    median_costs_by_angle = np.array([np.median(all_costs[all_angles == a]) for a in unique_angles])

    basin_2x = 30.0  # default: wider than our range
    threshold = max(truth_cost * 2, truth_cost + 0.01)  # handle near-zero truth cost
    for ai, a in enumerate(unique_angles):
        if median_costs_by_angle[ai] > threshold:
            basin_2x = float(a)
            break

    # Also compute basin at 5× and 10×
    basin_5x = 30.0
    threshold_5x = max(truth_cost * 5, truth_cost + 0.05)
    for ai, a in enumerate(unique_angles):
        if median_costs_by_angle[ai] > threshold_5x:
            basin_5x = float(a)
            break

    basin_10x = 30.0
    threshold_10x = max(truth_cost * 10, truth_cost + 0.1)
    for ai, a in enumerate(unique_angles):
        if median_costs_by_angle[ai] > threshold_10x:
            basin_10x = float(a)
            break

    np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
             seed=seed, valid=True,
             truth_cost=truth_cost,
             perturbation_angles=all_angles,
             perturbation_costs=all_costs,
             perturbation_dirs=all_perturb_dirs,
             perturbation_omegas=all_perturb_omegas,
             unique_angles=unique_angles,
             median_costs_by_angle=median_costs_by_angle,
             basin_2x_deg=basin_2x,
             basin_5x_deg=basin_5x,
             basin_10x_deg=basin_10x,
             true_omega_anchor=true_omega_anchor,
             n_constraints=n_constraints,
    )

    summary.append({
        'seed': seed,
        'truth_cost': float(truth_cost),
        'basin_2x': float(basin_2x),
        'basin_5x': float(basin_5x),
        'basin_10x': float(basin_10x),
        'n_constraints': n_constraints,
    })

save_results(str(CKPT / "summary.json"), summary)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP 2: ALIGNMENT COST BASIN WIDTH — 100 seeds")
print("=" * 70)

valid_s = [r for r in summary if 'error' not in r]
basins = [r['basin_2x'] for r in valid_s]

print(f"\n  Basin half-width (2× truth cost):")
print(f"    min={min(basins):.1f}°, median={np.median(basins):.1f}°, "
      f"mean={np.mean(basins):.1f}°, max={max(basins):.1f}°")
print(f"    < 1°: {sum(1 for b in basins if b < 1)} seeds (extremely narrow)")
print(f"    1-3°: {sum(1 for b in basins if 1 <= b < 3)} seeds (narrow, need fine grid)")
print(f"    3-5°: {sum(1 for b in basins if 3 <= b < 5)} seeds (moderate)")
print(f"    > 5°: {sum(1 for b in basins if b >= 5)} seeds (wide, grid-friendly)")
print(f"    ≥30°: {sum(1 for b in basins if b >= 30)} seeds (flat — cost doesn't discriminate)")

print(f"\n  Per-seed detail (sorted by basin width):")
for r in sorted(valid_s, key=lambda x: x['basin_2x'])[:15]:
    print(f"    seed {r['seed']:3d}: basin_2x={r['basin_2x']:5.1f}° "
          f"truth_cost={r['truth_cost']:.6f} cstr={r['n_constraints']}")
print(f"    ...")
for r in sorted(valid_s, key=lambda x: x['basin_2x'])[-5:]:
    print(f"    seed {r['seed']:3d}: basin_2x={r['basin_2x']:5.1f}° "
          f"truth_cost={r['truth_cost']:.6f} cstr={r['n_constraints']}")

print(f"\nSaved to {CKPT}/")
