#!/usr/bin/env python3
"""
m095 — Grid Cost Diagnosis: Alignment vs Expected-Dot in Step 2.

Question: Does expected-dot cost in the GRID SEARCH improve truth ranking
on failing seeds (12, 27, 33)?

m093 tested expected-dot in NM only (Step 3) and got same results as
m090. But the failing seeds fail at the GRID level — truth not in top-20.
If the grid can't find it, NM can't fix it.

This script runs both costs on the same grid and compares truth ranking.
Uses pre-computed calibration table from m093.

Usage:
  MICRO95_SEED=27 python3 m095_grid_cost_diagnosis.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO95_SEED', '27'))

# Pipeline parameters
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24

Z_NORMALS = {4, 5}


def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
CKPT_DIR = RESULTS_DIR / f"m095_grid_diag_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print(f"m095 — Grid Cost Diagnosis (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Load calibration table from m093
calib_path = RESULTS_DIR / f"m093_seed{TRAJ_SEED:03d}" / "calibration.npz"
if not calib_path.exists():
    # Fall back to any available seed's calibration (table is trajectory-independent)
    calib_path = RESULTS_DIR / "m093_seed093" / "calibration.npz"
calib = np.load(str(calib_path))
mag_table = calib['mag_table']
calib_angles_deg = calib['calib_angles_deg']
calib_dots = np.cos(np.deg2rad(calib_angles_deg))
print(f"Loaded calibration from {calib_path}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Constraints (same as m090/93)
# ══════════════════════════════════════════════════════════════════════
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
smooth_mags_at_spec = smoothed_lc[spec_peaks]
smooth_ranking = np.argsort(smooth_mags_at_spec)
if (len(smooth_ranking) >= 2 and
    abs(smooth_mags_at_spec[smooth_ranking[0]] - smooth_mags_at_spec[smooth_ranking[1]]) < 0.05):
    tied = smooth_ranking[:2]
    anchor_rank = tied[np.argmin(spec_peaks[tied])]
else:
    anchor_rank = smooth_ranking[0]
anchor_idx = int(spec_peaks[anchor_rank])
anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]
anchor_allowed = get_allowed_normals(anchor_mag)

non_anchor = spec_peaks[spec_peaks != anchor_idx]
constraint_epochs = non_anchor
constraint_mags = observed_lc[constraint_epochs]
constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]
n_constraints = len(constraint_epochs)

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={anchor_mag:.2f}")
print(f"  Allowed: {[group_names[i] for i in anchor_allowed]}")
print(f"Constraints: {n_constraints}")
for ci in range(n_constraints):
    ep = constraint_epochs[ci]
    allowed = [group_names[i] for i in constraint_allowed[ci]]
    print(f"  ep {ep}: mag={constraint_mags[ci]:.2f}, dt={dt_constraints[ci]:.0f}s -> {allowed}")

# Constraint geometry analysis
print(f"\nConstraint geometry analysis:")
for ci in range(n_constraints):
    for cj in range(ci+1, n_constraints):
        pab_dot = np.dot(pab_at_constraints[ci], pab_at_constraints[cj])
        dt_diff = abs(dt_constraints[ci] - dt_constraints[cj])
        print(f"  c{ci}-c{cj}: PAB dot={pab_dot:.4f}, dt_diff={dt_diff:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# SHARED PROPAGATION HELPER
# ══════════════════════════════════════════════════════════════════════
def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


# ══════════════════════════════════════════════════════════════════════
# COST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def expected_dot_for_normal(ni, obs_mag):
    ed = float(np.interp(obs_mag, mag_table[ni, :], calib_dots))
    return np.clip(ed, 0.0, 1.0)


def phi_cost_alignment(q_anchors_xyzw, delta_qs, pab_arr,
                        allowed_per_constraint, normals, w):
    """Standard alignment cost: (1 - max_dot)^2."""
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


def phi_cost_expected_dot(q_anchors_xyzw, delta_qs, pab_arr,
                           allowed_per_constraint, normals, obs_mags, w):
    """Expected-dot cost: (actual_dot - expected_dot)^2."""
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        obs_mag = obs_mags[ci]

        best_cost = np.full(n_phi, np.inf)
        for ni in allowed:
            actual_dots = pbs @ normals[ni]
            ed = expected_dot_for_normal(ni, obs_mag)
            cost_ni = (actual_dots - ed) ** 2
            best_cost = np.minimum(best_cost, cost_ni)
        costs += w * best_cost
    return costs


# ══════════════════════════════════════════════════════════════════════
# GRID SEARCH — BOTH COSTS
# ══════════════════════════════════════════════════════════════════════
omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))

# Globals for pool workers
_omega_dirs = omega_dirs
_omega_mags_s = omega_mags_search
_qa_anchor_sets = qa_anchor_sets
_constraint_allowed = constraint_allowed
_constraint_mags = constraint_mags


def eval_direction_both_costs(wi):
    """Evaluate one omega direction with both alignment and expected-dot cost."""
    wd = _omega_dirs[wi]
    best_align = np.inf
    best_expdot = np.inf
    best_omega_a = None
    best_omega_e = None

    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints)

        for ni, qa_xyzw in _qa_anchor_sets:
            # Alignment cost
            ca = phi_cost_alignment(
                qa_xyzw, dqs, pab_at_constraints,
                _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
            ba = int(np.argmin(ca))
            if ca[ba] < best_align:
                best_align = ca[ba]
                best_omega_a = omega_test.copy()

            # Expected-dot cost
            ce = phi_cost_expected_dot(
                qa_xyzw, dqs, pab_at_constraints,
                _constraint_allowed, unique_normals,
                _constraint_mags, CONSTRAINT_WEIGHT)
            be = int(np.argmin(ce))
            if ce[be] < best_expdot:
                best_expdot = ce[be]
                best_omega_e = omega_test.copy()

    return best_align, best_omega_a, best_expdot, best_omega_e


print(f"\n--- Grid search: {N_DIRS} dirs x {N_MAGS} mags, BOTH costs ---")
t_grid = time.time()

if __name__ == '__main__':
    with Pool(GRID_WORKERS) as pool:
        results = pool.map(eval_direction_both_costs, range(N_DIRS))
    grid_time = time.time() - t_grid
    print(f"Grid search done in {grid_time:.1f}s")

    align_costs = np.array([r[0] for r in results])
    align_omegas = np.array([r[1] for r in results])
    expdot_costs = np.array([r[2] for r in results])
    expdot_omegas = np.array([r[3] for r in results])

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"ANALYSIS — seed {TRAJ_SEED}")
    print(f"{'='*60}")

    # Find truth rank in each cost
    for label, costs, omegas in [("ALIGNMENT", align_costs, align_omegas),
                                  ("EXPECTED-DOT", expdot_costs, expdot_omegas)]:
        sorted_idx = np.argsort(costs)

        # Find best truth rank
        truth_rank = -1
        truth_werr = 999
        for i in range(len(sorted_idx)):
            ri = sorted_idx[i]
            werr = omega_dir_err(omegas[ri], true_omega_anchor)
            if werr < 5.0 and truth_rank < 0:
                truth_rank = i + 1
                truth_werr = werr

        print(f"\n  {label}:")
        print(f"    Truth rank: {truth_rank if truth_rank > 0 else 'NOT IN TOP-2000'} "
              f"(w_err={truth_werr:.1f}°)")
        print(f"    Top-5:")
        for i in range(min(5, len(sorted_idx))):
            ri = sorted_idx[i]
            werr = omega_dir_err(omegas[ri], true_omega_anchor)
            tag = " <-- NEAR TRUTH" if werr < 10 else ""
            print(f"      #{i+1}: cost={costs[ri]:.6f} w_err={werr:.1f}°{tag}")

        # Where does truth rank?
        if truth_rank > 0 and truth_rank <= 20:
            print(f"    ==> TRUTH IN TOP-20 ✓")
        elif truth_rank > 0:
            print(f"    ==> Truth at rank #{truth_rank} — needs larger pool")
        else:
            # Check if truth is even in the magnitude range
            true_wmag = np.linalg.norm(true_omega_anchor)
            mag_range = (omega_mags_search[0], omega_mags_search[-1])
            print(f"    True |w|={np.rad2deg(true_wmag):.3f} dps, "
                  f"grid range=[{np.rad2deg(mag_range[0]):.3f}, {np.rad2deg(mag_range[1]):.3f}]")

    # Detailed comparison: for each grid direction, compare costs
    # Focus on directions near truth
    print(f"\n  DIRECTIONS NEAR TRUTH (<15°):")
    for wi in range(N_DIRS):
        werr = omega_dir_err(align_omegas[wi], true_omega_anchor)
        if werr < 15:
            a_rank = int(np.searchsorted(np.sort(align_costs), align_costs[wi])) + 1
            e_rank = int(np.searchsorted(np.sort(expdot_costs), expdot_costs[wi])) + 1
            print(f"    dir#{wi}: w_err={werr:.1f}° | "
                  f"align: cost={align_costs[wi]:.6f} rank={a_rank} | "
                  f"expdot: cost={expdot_costs[wi]:.6f} rank={e_rank}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════
    np.savez(str(CKPT_DIR / "grid_diagnosis.npz"),
             align_costs=align_costs, align_omegas=align_omegas,
             expdot_costs=expdot_costs, expdot_omegas=expdot_omegas,
             true_omega_anchor=true_omega_anchor,
             constraint_epochs=constraint_epochs,
             constraint_mags=constraint_mags,
             dt_constraints=dt_constraints)

    result = {
        'traj_seed': TRAJ_SEED,
        'n_dirs': N_DIRS,
        'n_mags': N_MAGS,
        'n_constraints': n_constraints,
        'omega_est_dps': float(omega_est_dps),
        'true_omega_mag_dps': true_omega_mag_dps,
        'grid_time_s': grid_time,
    }
    save_results(str(CKPT_DIR / "result.json"), result)
    print(f"\nSaved to {CKPT_DIR}/")
    print(f"Total time: {time.time() - t_global:.1f}s")
