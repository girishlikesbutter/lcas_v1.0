#!/usr/bin/env python3
"""Micro-43 -- Focused single-normal PAB seeding with magnitude-based normal ID.

Question: If we identify the glinting normal from magnitude (using m039's
rule-based classifier), and concentrate ALL iso-brightness seeds on that one
PAB circle, how much better is candidate generation compared to:
  (a) m038's diluted 14-normal PAB seeding, and
  (b) m010's random SO(3) seeding?

Method:
  1. At glint epoch 183, apply m039's magnitude rule to predict the normal group.
  2. Compare predicted vs oracle (m034 fractional flux dominant).
  3. Generate 2160 seeds on the PREDICTED normal's PAB circle.
  4. Generate 2160 random SO(3) seeds for fair comparison.
  5. Optionally repeat for the ORACLE normal (if different from predicted).
  6. Run L-BFGS-B iso-brightness optimization from each seed.
  7. Cluster, compute angular distance to truth, compare all methods.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import get_context

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from lib.experiment_setup import (
    setup_experiment, save_results, attitude_error_deg,
)
from src.computation.facet_data_extractor import (
    extract_facet_arrays, apply_articulation_to_arrays,
)
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

LOG_PATH = RESULTS_DIR / "m043_stdout.txt"
_log_file = open(LOG_PATH, 'w')
sys.stdout = Tee(sys.__stdout__, _log_file)


# ===========================================================================
# Configuration
# ===========================================================================
TARGET_EPOCH = 183
N_PHI_EXACT = 360          # samples on the exact-alignment 1-DOF circle
OFFSET_DEGS = [1, 2, 3, 4, 5]  # concentric offset ring radii
N_RING_DIRS = 12           # directions around each offset ring
N_PHI_RING = 30            # circle samples per ring direction
N_WORKERS = 8
CLUSTER_THRESHOLD_DEG = 1.0
CONVERGENCE_TOL = 0.01     # |mag_converged - target| < tol

# Seed counts:
#   Exact alignment: 360 seeds
#   Offset rings: 5 offsets x 12 directions x 30 phi = 1800 seeds
#   Total per normal: 360 + 1800 = 2160 seeds
SEEDS_PER_NORMAL = N_PHI_EXACT + len(OFFSET_DEGS) * N_RING_DIRS * N_PHI_RING

# Magnitude-based classification thresholds (from m039)
MAG_THRESHOLD_BRIGHT = 7.5   # mag < 7.5 → z-faces group (large-area)
MAG_THRESHOLD_MEDIUM = 8.5   # 7.5 ≤ mag < 8.5 → antenna dishes
# mag ≥ 8.5 → bus/other

CTX = None  # module-level global (inherited by forked workers)


# ===========================================================================
# Lo-fi brightness and iso-brightness objective (same as m038)
# ===========================================================================
def lofi_single(q_wxyz, eidx):
    """Lo-fi brightness for one quaternion at one epoch."""
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    sv = CTX.sun_pos[eidx] - CTX.sat_pos[eidx]
    ov = CTX.obs_pos[eidx] - CTX.sat_pos[eidx]
    k1 = R @ sv
    k1 /= np.linalg.norm(k1)
    k2 = R @ ov
    k2 /= np.linalg.norm(k2)
    art = {c: m[eidx:eidx + 1] for c, m in CTX.art_matrices.items()}
    lit = create_no_shadow_lit_status(CTX.satellite, 1)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=k1.reshape(1, 3),
        k2_vectors_array=k2.reshape(1, 3),
        observer_distances=np.array([CTX.obs_dist[eidx]]),
        satellite=CTX.satellite,
        epochs=np.array([0.0]),
        pre_computed_matrices=art,
        generate_no_shadow=False,
        animate=False,
        show_progress=False,
    )
    return float(mags[0])


def iso_objective(rotvec, target_mag, eidx):
    """Squared brightness error for L-BFGS-B."""
    q_sci = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
    q_wxyz = np.array([q_sci[3], q_sci[0], q_sci[1], q_sci[2]])
    return (lofi_single(q_wxyz, eidx) - target_mag) ** 2


def run_one_seed(args):
    """Run one L-BFGS-B iso-brightness optimization from a seed rotvec."""
    seed_idx, rv_init, target_mag = args
    try:
        res = minimize(
            iso_objective, rv_init,
            args=(target_mag, TARGET_EPOCH),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8},
        )
        q_sci = Rotation.from_rotvec(res.x).as_quat()  # xyzw
        q_wxyz = np.array([q_sci[3], q_sci[0], q_sci[1], q_sci[2]])
        final_mag = lofi_single(q_wxyz, TARGET_EPOCH)
        return {
            'q_wxyz': q_wxyz,
            'residual': float(abs(final_mag - target_mag)),
            'nfev': res.nfev,
            'success': bool(res.success),
        }
    except Exception as e:
        return {'error': str(e)}


# ===========================================================================
# PAB circle seed generation (copied from m038)
# ===========================================================================
def generate_pab_circle_seeds(n_body, target_dir, n_phi=360):
    """Generate seeds on the 1-DOF circle that aligns n_body with target_dir.

    Returns list of rotvec (3,) arrays suitable for L-BFGS-B.
    """
    R0, _ = Rotation.align_vectors([target_dir], [n_body])

    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    seeds = []
    for phi in phis:
        R_twist = Rotation.from_rotvec(phi * target_dir)
        R_total = R_twist * R0
        seeds.append(R_total.as_rotvec())
    return seeds


def generate_offset_ring_seeds(n_body, pab_inertial, offset_deg,
                                n_phi=30, n_ring=12):
    """Seeds on rings at offset_deg from exact PAB alignment.

    For each of n_ring evenly-spaced directions around PAB, tilt the target
    by offset_deg in that direction, then sample the 1-DOF circle.
    """
    offset_rad = np.radians(offset_deg)

    if abs(pab_inertial[0]) < 0.9:
        perp = np.cross(pab_inertial, [1, 0, 0])
    else:
        perp = np.cross(pab_inertial, [0, 1, 0])
    perp /= np.linalg.norm(perp)

    seeds = []
    ring_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    for ring_phi in ring_angles:
        R_ring = Rotation.from_rotvec(ring_phi * pab_inertial)
        perp_rotated = R_ring.apply(perp)

        R_tilt = Rotation.from_rotvec(offset_rad * perp_rotated)
        target = R_tilt.apply(pab_inertial)
        target /= np.linalg.norm(target)

        circle_seeds = generate_pab_circle_seeds(n_body, target, n_phi=n_phi)
        seeds.extend(circle_seeds)

    return seeds


def generate_all_seeds_for_normal(n_body, pab_inertial):
    """Generate the full set of seeds (exact + offset rings) for one normal.

    Returns a list of rotvec arrays. Total count = SEEDS_PER_NORMAL.
    """
    seeds = []

    # Exact alignment circle
    circle_seeds = generate_pab_circle_seeds(n_body, pab_inertial, n_phi=N_PHI_EXACT)
    seeds.extend(circle_seeds)

    # Offset rings
    for offset_deg in OFFSET_DEGS:
        ring_seeds = generate_offset_ring_seeds(
            n_body, pab_inertial, offset_deg,
            n_phi=N_PHI_RING, n_ring=N_RING_DIRS,
        )
        seeds.extend(ring_seeds)

    return seeds


# ===========================================================================
# Quaternion clustering (copied from m038)
# ===========================================================================
def cluster_quaternions(q_wxyz_array, threshold_deg=1.0):
    """Cluster quaternions by geodesic distance, return labels and count."""
    n = len(q_wxyz_array)
    if n == 0:
        return np.array([], dtype=int), 0
    if n == 1:
        return np.array([1]), 1

    q_xyzw = q_wxyz_array[:, [1, 2, 3, 0]]
    gram = np.abs(q_xyzw @ q_xyzw.T)
    np.clip(gram, 0, 1, out=gram)
    np.arccos(gram, out=gram)
    gram *= 360.0 / np.pi   # geodesic in degrees
    np.fill_diagonal(gram, 0.0)

    condensed = squareform(gram, checks=False)
    del gram
    labels = fcluster(
        linkage(condensed, method='complete'),
        t=threshold_deg,
        criterion='distance',
    )
    del condensed
    return labels, len(set(labels))


def get_representative_quaternions(q_converged, residuals, labels):
    """Pick the best (lowest residual) quaternion per cluster."""
    rep_indices = []
    for cl in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == cl]
        best_member = min(members, key=lambda i: residuals[i])
        rep_indices.append(best_member)
    rep_q_wxyz = q_converged[rep_indices]
    return rep_q_wxyz


def compute_angular_distances_to_truth(rep_q_wxyz, q_truth_wxyz):
    """Geodesic distance from each representative to truth (in degrees)."""
    R_truth = Rotation.from_quat(
        [q_truth_wxyz[1], q_truth_wxyz[2], q_truth_wxyz[3], q_truth_wxyz[0]]
    )
    R_reps = Rotation.from_quat(rep_q_wxyz[:, [1, 2, 3, 0]])
    return np.degrees((R_reps.inv() * R_truth).magnitude())


def run_optimization_batch(seeds, target_mag, label):
    """Run L-BFGS-B from all seeds in parallel. Returns processed results."""
    n_seeds = len(seeds)
    print(f"\n--- Running L-BFGS-B for {label}: {n_seeds} seeds ({N_WORKERS} workers) ---")

    args_list = [(i, seeds[i], target_mag) for i in range(n_seeds)]

    t_opt = time.time()
    results = []
    with get_context('fork').Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, args_list, chunksize=32):
            results.append(r)
            if len(results) % 500 == 0:
                elapsed = time.time() - t_opt
                n_good = sum(
                    1 for x in results
                    if x.get('residual', 999) < CONVERGENCE_TOL
                )
                eta = elapsed / len(results) * (n_seeds - len(results))
                print(f"  [{len(results):>5}/{n_seeds}] "
                      f"{n_good} converged, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                      flush=True)

    opt_time = time.time() - t_opt
    print(f"  Optimization time: {opt_time:.1f}s "
          f"({opt_time / n_seeds:.3f}s/seed)")

    # Filter converged
    converged = [
        r for r in results
        if 'error' not in r and r['residual'] < CONVERGENCE_TOL
    ]
    n_errors = sum(1 for r in results if 'error' in r)
    n_converged = len(converged)
    print(f"  Converged: {n_converged}/{n_seeds} "
          f"({n_errors} errors, "
          f"{n_seeds - n_converged - n_errors} above tolerance)")

    return {
        'results': results,
        'converged': converged,
        'n_seeds': n_seeds,
        'n_converged': n_converged,
        'n_errors': n_errors,
        'opt_time_s': opt_time,
    }


def process_converged(opt_data, q_truth_wxyz, label):
    """Cluster converged quaternions and compute distances to truth."""
    converged = opt_data['converged']
    if len(converged) == 0:
        print(f"  WARNING: No converged results for {label}!")
        return {
            'n_unique': 0,
            'nearest_deg': float('inf'),
            'ang_dists': np.array([]),
            'rep_q_wxyz': np.array([]).reshape(0, 4),
        }

    q_converged = np.array([r['q_wxyz'] for r in converged])
    residuals = np.array([r['residual'] for r in converged])

    # Cluster
    labels, n_unique = cluster_quaternions(q_converged, CLUSTER_THRESHOLD_DEG)
    print(f"  {label}: {n_unique} unique candidates from {len(converged)} converged")

    # Representative quaternions
    rep_q_wxyz = get_representative_quaternions(q_converged, residuals, labels)

    # Angular distances to truth
    ang_dists = compute_angular_distances_to_truth(rep_q_wxyz, q_truth_wxyz)
    nearest_deg = float(np.min(ang_dists))
    print(f"  {label}: nearest to truth = {nearest_deg:.2f} deg")

    return {
        'n_unique': n_unique,
        'nearest_deg': nearest_deg,
        'ang_dists': ang_dists,
        'rep_q_wxyz': rep_q_wxyz,
    }


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Micro-43 -- Focused single-normal PAB seeding")
    print("           with magnitude-based normal identification")
    print("=" * 70)

    t_global = time.time()

    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    CTX = setup_experiment(
        n_observations=500,
        noise_sigma=0.05,
        random_seed=42,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    print(f"Setup complete: {CTX.n_observations} observations, "
          f"dt_sampling = {CTX.dt_sampling:.2f}s")

    # -----------------------------------------------------------------------
    # 2. Extract unique normals and group info
    # -----------------------------------------------------------------------
    print("\n--- Extracting unique facet normals ---")
    facet_arrays = extract_facet_arrays(CTX.satellite)
    art_normals, _ = apply_articulation_to_arrays(
        facet_arrays, CTX.art_matrices, 0, CTX.satellite,
    )
    rounded_normals = np.round(art_normals, 4)
    unique_normals, inverse_indices = np.unique(
        rounded_normals, axis=0, return_inverse=True,
    )
    n_groups = len(unique_normals)
    print(f"Total facets: {facet_arrays.total_facets}")
    print(f"Unique normal groups: {n_groups}")

    # Also load m034 data for oracle normal identification
    m034_npz = np.load(str(RESULTS_DIR / "m034_pab_alignment.npz"))
    frac_flux = m034_npz['frac_flux']           # (14, 500)
    m034_normals = m034_npz['unique_normals']  # (14, 3)

    with open(RESULTS_DIR / "m034_pab_alignment.json") as f:
        m034_json = json.load(f)
    group_info = m034_json['all_groups']

    # -----------------------------------------------------------------------
    # 3. Compute PAB_inertial and target magnitude at epoch 183
    # -----------------------------------------------------------------------
    print("\n--- Computing PAB_inertial at target epoch ---")
    sun_vec = CTX.sun_pos[TARGET_EPOCH] - CTX.sat_pos[TARGET_EPOCH]
    obs_vec = CTX.obs_pos[TARGET_EPOCH] - CTX.sat_pos[TARGET_EPOCH]
    k1_hat = sun_vec / np.linalg.norm(sun_vec)
    k2_hat = obs_vec / np.linalg.norm(obs_vec)
    pab_inertial = k1_hat + k2_hat
    pab_inertial /= np.linalg.norm(pab_inertial)
    print(f"PAB_inertial at epoch {TARGET_EPOCH}: {pab_inertial}")

    target_mag = CTX.observed_lc[TARGET_EPOCH]
    print(f"Target magnitude (observed, noisy): {target_mag:.4f}")
    print(f"True magnitude (noiseless):         {CTX.true_lc[TARGET_EPOCH]:.4f}")

    # -----------------------------------------------------------------------
    # 4. Magnitude-based normal classification (m039 rule)
    # -----------------------------------------------------------------------
    print("\n--- Magnitude-based classification ---")
    print(f"Observed magnitude at epoch {TARGET_EPOCH}: {target_mag:.4f}")

    if target_mag < MAG_THRESHOLD_BRIGHT:
        predicted_class = 'z-faces'
        print(f"  Rule: mag < {MAG_THRESHOLD_BRIGHT} -> z-faces (large area)")
    elif target_mag < MAG_THRESHOLD_MEDIUM:
        predicted_class = 'antenna'
        print(f"  Rule: {MAG_THRESHOLD_BRIGHT} <= mag < {MAG_THRESHOLD_MEDIUM} -> antenna dishes")
    else:
        predicted_class = 'bus/other'
        print(f"  Rule: mag >= {MAG_THRESHOLD_MEDIUM} -> bus/other")

    # Map predicted class to candidate normal groups
    # From m034 group_info:
    #   z-faces: groups with normal [0,0,+/-1] (groups 6,7 — area ~22.75 each)
    #   antenna: groups with normal containing [+-0.9659, ...] (groups 1,2,8,9,10,11)
    #   bus: everything else
    print(f"\n  Normal group mapping:")
    z_face_groups = []
    antenna_groups = []
    bus_groups = []
    for gi, info in enumerate(group_info):
        normal = np.array(info['normal'])
        components = info['components']
        area = info['total_area']
        # z-faces: normal is [0,0,+/-1]
        if abs(abs(normal[2]) - 1.0) < 0.01 and abs(normal[0]) < 0.01 and abs(normal[1]) < 0.01:
            z_face_groups.append(gi)
            cat = 'z-face'
        # antenna-like: high specular normals (contains AD_ component or has r_s=0.4, n_phong=200)
        elif any('AD_' in c for c in components) and area < 15:
            antenna_groups.append(gi)
            cat = 'antenna'
        else:
            bus_groups.append(gi)
            cat = 'bus/other'
        print(f"    G{gi:2d}: normal={info['normal']}, "
              f"components={components}, area={area:.2f} -> {cat}")

    print(f"\n  z-face groups:  {z_face_groups}")
    print(f"  antenna groups: {antenna_groups}")
    print(f"  bus groups:     {bus_groups}")

    # Select predicted group(s)
    if predicted_class == 'z-faces':
        predicted_groups = z_face_groups
    elif predicted_class == 'antenna':
        predicted_groups = antenna_groups
    else:
        predicted_groups = bus_groups

    print(f"\n  Predicted class: '{predicted_class}' -> groups {predicted_groups}")

    # -----------------------------------------------------------------------
    # 5. Oracle: dominant normal from m034 fractional flux
    # -----------------------------------------------------------------------
    print("\n--- Oracle normal identification ---")
    oracle_group = int(np.argmax(frac_flux[:, TARGET_EPOCH]))
    oracle_info = group_info[oracle_group]
    oracle_frac = float(frac_flux[oracle_group, TARGET_EPOCH])
    print(f"  Oracle dominant group at epoch {TARGET_EPOCH}: G{oracle_group}")
    print(f"  Normal: {oracle_info['normal']}")
    print(f"  Components: {oracle_info['components']}")
    print(f"  Fractional flux: {oracle_frac:.4f}")

    oracle_in_predicted = oracle_group in predicted_groups
    print(f"\n  Oracle group in predicted set? {oracle_in_predicted}")

    # -----------------------------------------------------------------------
    # 6. Generate seeds for each method
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SEED GENERATION")
    print("=" * 70)

    # Method A: Focused PAB seeding on PREDICTED normal(s)
    print(f"\n--- Method A: Focused PAB on predicted groups {predicted_groups} ---")
    seeds_predicted = []
    for gi in predicted_groups:
        n_body = np.array(group_info[gi]['normal'])
        group_seeds = generate_all_seeds_for_normal(n_body, pab_inertial)
        seeds_predicted.extend(group_seeds)
        print(f"  G{gi}: {len(group_seeds)} seeds")
    print(f"  Total predicted seeds: {len(seeds_predicted)}")

    # Method B: Focused PAB seeding on ORACLE normal only (if different)
    if oracle_in_predicted:
        print(f"\n--- Method B: Oracle group G{oracle_group} is in predicted set ---")
        print(f"  Generating oracle-only seeds for comparison anyway.")
    else:
        print(f"\n--- Method B: Oracle group G{oracle_group} NOT in predicted set ---")

    oracle_n_body = np.array(group_info[oracle_group]['normal'])
    seeds_oracle = generate_all_seeds_for_normal(oracle_n_body, pab_inertial)
    print(f"  Oracle seeds (G{oracle_group} only): {len(seeds_oracle)}")

    # Method C: Random SO(3) seeds (same total as predicted for fair comparison)
    n_random = len(seeds_predicted)
    print(f"\n--- Method C: Random SO(3) seeds ---")
    rng = np.random.RandomState(42)
    random_rotations = Rotation.random(n_random, random_state=rng)
    seeds_random = [r.as_rotvec() for r in random_rotations]
    print(f"  Random SO(3) seeds: {len(seeds_random)}")

    # -----------------------------------------------------------------------
    # 7. Run L-BFGS-B optimization for all three methods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("L-BFGS-B OPTIMIZATION")
    print("=" * 70)

    q_truth_183 = CTX.true_quaternions[TARGET_EPOCH]

    # Method A: Predicted normal(s) focused PAB
    opt_predicted = run_optimization_batch(seeds_predicted, target_mag, "Predicted-PAB")
    proc_predicted = process_converged(opt_predicted, q_truth_183, "Predicted-PAB")

    # Method B: Oracle normal focused PAB
    opt_oracle = run_optimization_batch(seeds_oracle, target_mag, "Oracle-PAB")
    proc_oracle = process_converged(opt_oracle, q_truth_183, "Oracle-PAB")

    # Method C: Random SO(3)
    opt_random = run_optimization_batch(seeds_random, target_mag, "Random-SO3")
    proc_random = process_converged(opt_random, q_truth_183, "Random-SO3")

    # -----------------------------------------------------------------------
    # 8. Load m038 results for comparison
    # -----------------------------------------------------------------------
    print("\n--- Loading m038 (14-normal diluted PAB) for comparison ---")
    m038_npz = np.load(str(RESULTS_DIR / "m038_pab_seeded_candidates.npz"))
    m038_ang_dists = m038_npz['ang_dists_to_truth']
    m038_n_unique = len(m038_ang_dists)
    m038_nearest = float(np.min(m038_ang_dists))
    print(f"  m038: {m038_n_unique} unique candidates, "
          f"nearest = {m038_nearest:.2f} deg, "
          f"from 30,240 seeds (14 normals)")

    # Load m010 results for comparison
    print("\n--- Loading m010 (random SO(3)) for comparison ---")
    m010_npz = np.load(str(RESULTS_DIR / "m010_optimized_candidates.npz"))
    m010_q_wxyz = m010_npz['candidate_q_wxyz']
    # m010's ang_dists_to_truth is relative to q0, not q_truth@183
    # Recompute relative to q_truth@183 for fair comparison
    m010_ang_dists_183 = compute_angular_distances_to_truth(m010_q_wxyz, q_truth_183)
    m010_n_unique = len(m010_q_wxyz)
    m010_nearest_183 = float(np.min(m010_ang_dists_183))
    print(f"  m010: {m010_n_unique} unique candidates (from 10,000 seeds)")
    print(f"  m010 nearest to truth@183: {m010_nearest_183:.2f} deg")
    print(f"  (Note: m010 JSON reports {float(m010_npz['ang_dists_to_truth'].min()):.2f} deg "
          f"relative to q0, not q_truth@183)")

    # -----------------------------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    methods = {
        'Predicted-PAB': {
            'n_seeds': opt_predicted['n_seeds'],
            'n_converged': opt_predicted['n_converged'],
            'n_unique': proc_predicted['n_unique'],
            'nearest_deg': proc_predicted['nearest_deg'],
            'opt_time_s': opt_predicted['opt_time_s'],
        },
        'Oracle-PAB': {
            'n_seeds': opt_oracle['n_seeds'],
            'n_converged': opt_oracle['n_converged'],
            'n_unique': proc_oracle['n_unique'],
            'nearest_deg': proc_oracle['nearest_deg'],
            'opt_time_s': opt_oracle['opt_time_s'],
        },
        'Random-SO3 (this run)': {
            'n_seeds': opt_random['n_seeds'],
            'n_converged': opt_random['n_converged'],
            'n_unique': proc_random['n_unique'],
            'nearest_deg': proc_random['nearest_deg'],
            'opt_time_s': opt_random['opt_time_s'],
        },
        'm038 (14-normal PAB)': {
            'n_seeds': 30240,
            'n_converged': 21305,
            'n_unique': m038_n_unique,
            'nearest_deg': m038_nearest,
            'opt_time_s': 5432.6,
        },
        'm010 (random SO3, 10K)': {
            'n_seeds': 10000,
            'n_converged': 6889,
            'n_unique': m010_n_unique,
            'nearest_deg': m010_nearest_183,
            'opt_time_s': 1755.2,
        },
    }

    header = f"{'Method':<30s}  {'Seeds':>6s}  {'Conv':>6s}  {'Unique':>6s}  {'Nearest':>8s}  {'Time':>8s}"
    print(header)
    print("-" * len(header))
    for method_name, stats in methods.items():
        print(f"{method_name:<30s}  "
              f"{stats['n_seeds']:>6d}  "
              f"{stats['n_converged']:>6d}  "
              f"{stats['n_unique']:>6d}  "
              f"{stats['nearest_deg']:>7.2f}°  "
              f"{stats['opt_time_s']:>7.1f}s")

    print(f"\nPrediction accuracy: oracle group G{oracle_group} "
          f"{'IS' if oracle_in_predicted else 'is NOT'} "
          f"in predicted set {predicted_groups}")

    # Percentiles for the three new methods
    print(f"\nAttitude error percentiles (unique candidates):")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"  {'%ile':>5s}  {'Pred-PAB':>10s}  {'Oracle-PAB':>10s}  {'Random':>10s}")
    for p in pcts:
        vals = []
        for proc in [proc_predicted, proc_oracle, proc_random]:
            if len(proc['ang_dists']) > 0:
                vals.append(f"{np.percentile(proc['ang_dists'], p):>9.2f}°")
            else:
                vals.append(f"{'N/A':>10s}")
        print(f"  {p:>4d}%  {'  '.join(vals)}")

    # -----------------------------------------------------------------------
    # 10. Plot: 3-panel figure
    # -----------------------------------------------------------------------
    print("\n--- Generating plot ---")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    # Panel 1: Histogram of angular distances to truth
    bins_dist = np.linspace(0, 180, 91)

    if len(proc_predicted['ang_dists']) > 0:
        axes[0].hist(
            proc_predicted['ang_dists'], bins=bins_dist, alpha=0.6,
            color='blue',
            label=f'Focused-predicted (N={proc_predicted["n_unique"]})',
        )
    if len(proc_oracle['ang_dists']) > 0:
        axes[0].hist(
            proc_oracle['ang_dists'], bins=bins_dist, alpha=0.5,
            color='green',
            label=f'Focused-oracle (N={proc_oracle["n_unique"]})',
        )
    if len(proc_random['ang_dists']) > 0:
        axes[0].hist(
            proc_random['ang_dists'], bins=bins_dist, alpha=0.4,
            color='gray',
            label=f'Random SO(3) (N={proc_random["n_unique"]})',
        )

    # Vertical lines for nearest distances
    if proc_predicted['nearest_deg'] < float('inf'):
        axes[0].axvline(proc_predicted['nearest_deg'], color='blue',
                        linestyle='--', linewidth=1.5,
                        label=f'Pred nearest = {proc_predicted["nearest_deg"]:.1f} deg')
    if proc_oracle['nearest_deg'] < float('inf'):
        axes[0].axvline(proc_oracle['nearest_deg'], color='green',
                        linestyle=':', linewidth=1.5,
                        label=f'Oracle nearest = {proc_oracle["nearest_deg"]:.1f} deg')
    if proc_random['nearest_deg'] < float('inf'):
        axes[0].axvline(proc_random['nearest_deg'], color='gray',
                        linestyle='-.', linewidth=1.5,
                        label=f'Random nearest = {proc_random["nearest_deg"]:.1f} deg')

    axes[0].set_xlabel('Angular distance to truth@183 (deg)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Candidate Distance to Truth', fontsize=12)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: CDF of angular distances for all methods
    for proc, label, color, ls in [
        (proc_predicted, 'Focused-predicted', 'blue', '-'),
        (proc_oracle, 'Focused-oracle', 'green', '--'),
        (proc_random, 'Random SO(3)', 'gray', '-.'),
    ]:
        if len(proc['ang_dists']) > 0:
            sorted_dists = np.sort(proc['ang_dists'])
            cdf_y = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
            axes[1].plot(sorted_dists, cdf_y, color=color, linestyle=ls,
                         linewidth=1.5, label=label)

    # Also add m038 CDF
    sorted_m38 = np.sort(m038_ang_dists)
    cdf_m38 = np.arange(1, len(sorted_m38) + 1) / len(sorted_m38)
    axes[1].plot(sorted_m38, cdf_m38, color='red', linestyle=':',
                 linewidth=1.2, label=f'm038 14-normal (N={m038_n_unique})')

    # And m010 CDF (recomputed to truth@183)
    sorted_m10 = np.sort(m010_ang_dists_183)
    cdf_m10 = np.arange(1, len(sorted_m10) + 1) / len(sorted_m10)
    axes[1].plot(sorted_m10, cdf_m10, color='orange', linestyle=':',
                 linewidth=1.2, label=f'm010 random (N={m010_n_unique})')

    axes[1].set_xlabel('Angular distance to truth@183 (deg)', fontsize=11)
    axes[1].set_ylabel('CDF (fraction of candidates)', fontsize=11)
    axes[1].set_title('CDF: Distance to Truth', fontsize=12)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 180)

    # Panel 3: Summary bar chart
    bar_labels = ['Pred-PAB', 'Oracle-PAB', 'Random\n(this run)', 'm038\n(14-norm)', 'm010\n(10K rand)']
    bar_nearest = [
        proc_predicted['nearest_deg'],
        proc_oracle['nearest_deg'],
        proc_random['nearest_deg'],
        m038_nearest,
        m010_nearest_183,
    ]
    bar_colors = ['blue', 'green', 'gray', 'red', 'orange']

    # Cap infinite values for display
    bar_nearest_display = [min(v, 180.0) for v in bar_nearest]

    bars = axes[2].bar(range(len(bar_labels)), bar_nearest_display,
                       color=bar_colors, alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(len(bar_labels)))
    axes[2].set_xticklabels(bar_labels, fontsize=9)
    axes[2].set_ylabel('Nearest candidate to truth (deg)', fontsize=11)
    axes[2].set_title('Best Candidate Distance', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, bar_nearest)):
        display_val = min(val, 180.0)
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, display_val + 1.5,
            f'{val:.1f}' if val < float('inf') else 'N/A',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    # Add seed count annotation
    bar_seeds = [
        opt_predicted['n_seeds'],
        opt_oracle['n_seeds'],
        opt_random['n_seeds'],
        30240,
        10000,
    ]
    for i, (bar, ns) in enumerate(zip(bars, bar_seeds)):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, 2,
            f'N={ns}', ha='center', va='bottom', fontsize=7,
            color='white', fontweight='bold',
        )

    fig.suptitle(
        f'Micro-43: Focused Single-Normal PAB Seeding '
        f'(Epoch {TARGET_EPOCH}, mag={target_mag:.2f})',
        fontsize=14, fontweight='bold',
    )

    plot_path = RESULTS_DIR / "m043_focused_pab_seeding.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # -----------------------------------------------------------------------
    # 11. Save results
    # -----------------------------------------------------------------------
    print("\n--- Saving results ---")

    # NPZ: all representative quaternions and angular distances
    npz_path = RESULTS_DIR / "m043_focused_pab_seeding.npz"
    np.savez(
        npz_path,
        # Predicted PAB results
        predicted_rep_q_wxyz=proc_predicted['rep_q_wxyz'],
        predicted_ang_dists=proc_predicted['ang_dists'],
        # Oracle PAB results
        oracle_rep_q_wxyz=proc_oracle['rep_q_wxyz'],
        oracle_ang_dists=proc_oracle['ang_dists'],
        # Random SO(3) results
        random_rep_q_wxyz=proc_random['rep_q_wxyz'],
        random_ang_dists=proc_random['ang_dists'],
        # m010 recomputed distances
        m010_ang_dists_at_183=m010_ang_dists_183,
        # Reference data
        truth_q_183=q_truth_183,
        target_epoch=TARGET_EPOCH,
        target_mag=target_mag,
        pab_inertial=pab_inertial,
        predicted_groups=np.array(predicted_groups),
        oracle_group=oracle_group,
    )
    print(f"NPZ saved: {npz_path}")

    # JSON summary
    def safe_percentile(arr, p):
        if len(arr) == 0:
            return None
        return round(float(np.percentile(arr, p)), 2)

    json_results = {
        'experiment': 'm043_focused_pab_seeding',
        'description': (
            'Focused single-normal PAB seeding with magnitude-based '
            'normal identification, compared to diluted 14-normal (m038) '
            'and random SO(3) (m010).'
        ),
        'target_epoch': TARGET_EPOCH,
        'target_mag': float(target_mag),
        'true_mag': float(CTX.true_lc[TARGET_EPOCH]),
        'classification': {
            'predicted_class': predicted_class,
            'mag_thresholds': {
                'bright': MAG_THRESHOLD_BRIGHT,
                'medium': MAG_THRESHOLD_MEDIUM,
            },
            'predicted_groups': predicted_groups,
            'oracle_group': oracle_group,
            'oracle_normal': group_info[oracle_group]['normal'],
            'oracle_components': group_info[oracle_group]['components'],
            'oracle_frac_flux': oracle_frac,
            'oracle_in_predicted': oracle_in_predicted,
        },
        'seed_config': {
            'n_phi_exact': N_PHI_EXACT,
            'offset_degs': OFFSET_DEGS,
            'n_ring_dirs': N_RING_DIRS,
            'n_phi_ring': N_PHI_RING,
            'seeds_per_normal': SEEDS_PER_NORMAL,
        },
        'methods': {},
    }

    for method_name, opt_data, proc_data in [
        ('predicted_pab', opt_predicted, proc_predicted),
        ('oracle_pab', opt_oracle, proc_oracle),
        ('random_so3', opt_random, proc_random),
    ]:
        json_results['methods'][method_name] = {
            'n_seeds': opt_data['n_seeds'],
            'n_converged': opt_data['n_converged'],
            'n_errors': opt_data['n_errors'],
            'convergence_rate': round(opt_data['n_converged'] / opt_data['n_seeds'], 4)
                if opt_data['n_seeds'] > 0 else 0.0,
            'n_unique_candidates': proc_data['n_unique'],
            'nearest_to_truth_deg': round(proc_data['nearest_deg'], 2)
                if proc_data['nearest_deg'] < float('inf') else None,
            'att_error_percentiles': {
                str(p): safe_percentile(proc_data['ang_dists'], p) for p in pcts
            },
            'opt_time_s': round(opt_data['opt_time_s'], 1),
        }

    # Add m038 and m010 comparison data
    json_results['comparison'] = {
        'm038_14normal_pab': {
            'n_seeds': 30240,
            'n_unique': m038_n_unique,
            'nearest_to_truth_deg': round(m038_nearest, 2),
        },
        'm010_random_so3': {
            'n_seeds': 10000,
            'n_unique': m010_n_unique,
            'nearest_to_truth_at_183_deg': round(m010_nearest_183, 2),
        },
    }

    json_results['total_time_s'] = round(time.time() - t_global, 1)

    json_path = RESULTS_DIR / "m043_focused_pab_seeding.json"
    save_results(json_path, json_results)
    print(f"JSON saved: {json_path}")

    elapsed = time.time() - t_global
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done.")

    # Restore stdout
    sys.stdout = sys.__stdout__
    _log_file.close()
