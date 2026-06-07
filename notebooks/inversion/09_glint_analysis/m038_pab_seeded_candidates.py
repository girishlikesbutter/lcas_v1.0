#!/usr/bin/env python3
"""Micro-38 -- PAB-seeded iso-brightness candidate generation.

Question: Does seeding L-BFGS-B iso-brightness optimization around
PAB-alignment circles (instead of random SO(3)) produce better candidates
with fewer seeds?

Method:
  At glint epoch 183 (mag ~7.15), for each of the 14 unique body-frame
  normals, construct the 1-DOF circle of quaternions that aligns that
  normal with PAB_inertial. Sample each circle at 1-deg spacing
  (360 x 14 = 5040 seeds). Then add concentric rings at 1-5 deg offset
  around PAB (~25K seeds). Run L-BFGS-B iso-brightness optimization from
  each seed, dedup converged candidates, and compare to m010's 5643
  candidates from 10K random seeds.
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
    setup_experiment, save_results, brightness_single_epoch, attitude_error_deg,
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

LOG_PATH = RESULTS_DIR / "m038_stdout.txt"
_log_file = open(LOG_PATH, 'w')
sys.stdout = Tee(sys.__stdout__, _log_file)

# ===========================================================================
# Configuration
# ===========================================================================
TARGET_EPOCH = 183
N_PHI_EXACT = 360          # samples per 1-DOF circle (exact alignment)
OFFSET_DEGS = [1, 2, 3, 4, 5]  # concentric ring offsets
N_RING_DIRS = 12           # directions around the offset ring
N_PHI_RING = 30            # circle samples per ring direction
N_WORKERS = 8
CLUSTER_THRESHOLD_DEG = 1.0
CONVERGENCE_TOL = 0.01     # |mag_converged - target| < tol

CTX = None  # module-level global (inherited by forked workers)

# ===========================================================================
# Lo-fi brightness and iso-brightness objective (same pattern as m010)
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
# PAB circle seed generation
# ===========================================================================
def generate_pab_circle_seeds(n_body, target_dir, n_phi=360):
    """Generate seeds on the 1-DOF circle that aligns n_body with target_dir.

    Returns list of rotvec (3,) arrays suitable for L-BFGS-B.
    """
    R0, _ = Rotation.align_vectors([target_dir], [n_body])

    # 1-DOF circle: compose R0 with rotations about target_dir
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

    # Arbitrary vector perpendicular to pab_inertial
    if abs(pab_inertial[0]) < 0.9:
        perp = np.cross(pab_inertial, [1, 0, 0])
    else:
        perp = np.cross(pab_inertial, [0, 1, 0])
    perp /= np.linalg.norm(perp)

    seeds = []
    ring_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    for ring_phi in ring_angles:
        # Rotate perp around pab_inertial by ring_phi
        R_ring = Rotation.from_rotvec(ring_phi * pab_inertial)
        perp_rotated = R_ring.apply(perp)

        # Tilt pab_inertial by offset_deg toward perp_rotated
        R_tilt = Rotation.from_rotvec(offset_rad * perp_rotated)
        target = R_tilt.apply(pab_inertial)
        target /= np.linalg.norm(target)

        # Sample the 1-DOF circle for this offset target
        circle_seeds = generate_pab_circle_seeds(n_body, target, n_phi=n_phi)
        seeds.extend(circle_seeds)

    return seeds


# ===========================================================================
# Dedup via hierarchical clustering (same approach as m010)
# ===========================================================================
def cluster_quaternions(q_wxyz_array, threshold_deg=1.0):
    """Cluster quaternions by geodesic distance, return unique representatives.

    Returns (rep_q_wxyz, labels, n_unique).
    """
    n = len(q_wxyz_array)
    if n == 0:
        return np.array([], dtype=int), 0
    if n == 1:
        return np.array([1]), 1

    # Pairwise geodesic distance via quaternion dot product
    q_xyzw = q_wxyz_array[:, [1, 2, 3, 0]]
    gram = np.abs(q_xyzw @ q_xyzw.T)
    np.clip(gram, 0, 1, out=gram)
    np.arccos(gram, out=gram)
    gram *= 360.0 / np.pi   # 2 * rad2deg → geodesic in degrees
    np.fill_diagonal(gram, 0.0)

    condensed = squareform(gram, checks=False)
    del gram
    labels = fcluster(
        linkage(condensed, method='complete'),
        t=threshold_deg,
        criterion='distance',
    )
    del condensed

    n_unique = len(set(labels))
    return labels, n_unique


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Micro-38 -- PAB-seeded iso-brightness candidate generation")
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
    # 2. Extract 14 unique body-frame normals
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

    # -----------------------------------------------------------------------
    # 3. Compute PAB_inertial at epoch 183
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
    print(f"Target magnitude (observed): {target_mag:.4f}")
    print(f"True magnitude: {CTX.true_lc[TARGET_EPOCH]:.4f}")

    # -----------------------------------------------------------------------
    # 4. Generate PAB-circle seeds
    # -----------------------------------------------------------------------
    print("\n--- Generating PAB-circle seeds ---")
    all_seeds = []          # list of (rotvec, normal_idx, ring_label)
    seed_metadata = []      # for diagnostics

    # (a) Exact alignment: 360 samples per normal
    for ni, n_body in enumerate(unique_normals):
        circle_seeds = generate_pab_circle_seeds(
            n_body, pab_inertial, n_phi=N_PHI_EXACT,
        )
        for rv in circle_seeds:
            all_seeds.append(rv)
            seed_metadata.append({'normal_idx': ni, 'offset_deg': 0.0})

    n_exact = len(all_seeds)
    print(f"Exact-alignment seeds: {n_exact} "
          f"({N_PHI_EXACT} phi x {n_groups} normals)")

    # (b) Offset rings at 1, 2, 3, 4, 5 deg
    for offset_deg in OFFSET_DEGS:
        n_before = len(all_seeds)
        for ni, n_body in enumerate(unique_normals):
            ring_seeds = generate_offset_ring_seeds(
                n_body, pab_inertial, offset_deg,
                n_phi=N_PHI_RING, n_ring=N_RING_DIRS,
            )
            for rv in ring_seeds:
                all_seeds.append(rv)
                seed_metadata.append({
                    'normal_idx': ni, 'offset_deg': float(offset_deg),
                })
        n_added = len(all_seeds) - n_before
        print(f"  Offset {offset_deg} deg: {n_added} seeds "
              f"({N_RING_DIRS} ring dirs x {N_PHI_RING} phi x {n_groups} normals)")

    n_total_seeds = len(all_seeds)
    print(f"\nTotal seeds: {n_total_seeds}")

    # -----------------------------------------------------------------------
    # 5. Run L-BFGS-B iso-brightness optimization from each seed
    # -----------------------------------------------------------------------
    print(f"\n--- Running L-BFGS-B from {n_total_seeds} seeds "
          f"({N_WORKERS} workers) ---")
    args_list = [
        (i, all_seeds[i], target_mag) for i in range(n_total_seeds)
    ]

    t_opt = time.time()
    results = []
    with get_context('fork').Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, args_list, chunksize=32):
            results.append(r)
            if len(results) % 5000 == 0:
                elapsed = time.time() - t_opt
                n_good = sum(
                    1 for x in results
                    if x.get('residual', 999) < CONVERGENCE_TOL
                )
                eta = elapsed / len(results) * (n_total_seeds - len(results))
                print(f"  [{len(results):>6}/{n_total_seeds}] "
                      f"{n_good} converged, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                      flush=True)

    opt_time = time.time() - t_opt
    print(f"Optimization: {opt_time:.1f}s "
          f"({opt_time / n_total_seeds:.3f}s/seed)")

    # -----------------------------------------------------------------------
    # 6. Filter converged and cluster
    # -----------------------------------------------------------------------
    print("\n--- Filtering converged results ---")
    converged = [
        r for r in results
        if 'error' not in r and r['residual'] < CONVERGENCE_TOL
    ]
    n_errors = sum(1 for r in results if 'error' in r)
    n_converged = len(converged)
    print(f"Converged: {n_converged}/{n_total_seeds} "
          f"({n_errors} errors, "
          f"{n_total_seeds - n_converged - n_errors} above tolerance)")

    if n_converged == 0:
        print("No converged results! Exiting.")
        sys.exit(1)

    q_converged = np.array([r['q_wxyz'] for r in converged])
    residuals_converged = np.array([r['residual'] for r in converged])

    print(f"\n--- Clustering within {CLUSTER_THRESHOLD_DEG} deg ---")
    t_cluster = time.time()
    labels, n_unique = cluster_quaternions(q_converged, CLUSTER_THRESHOLD_DEG)
    cluster_time = time.time() - t_cluster
    print(f"Unique candidates: {n_unique} from {n_converged} converged "
          f"({cluster_time:.1f}s)")

    # Best representative per cluster (lowest residual)
    rep_indices = []
    for cl in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == cl]
        best_member = min(members, key=lambda i: residuals_converged[i])
        rep_indices.append(best_member)
    rep_q_wxyz = q_converged[rep_indices]

    # -----------------------------------------------------------------------
    # 7. Compute angular distances to truth at epoch 183
    # -----------------------------------------------------------------------
    print("\n--- Angular distances to truth ---")
    q_truth_183 = CTX.true_quaternions[TARGET_EPOCH]
    R_truth_183 = Rotation.from_quat(
        [q_truth_183[1], q_truth_183[2], q_truth_183[3], q_truth_183[0]]
    )
    R_reps = Rotation.from_quat(rep_q_wxyz[:, [1, 2, 3, 0]])
    ang_dists_to_truth = np.degrees(
        (R_reps.inv() * R_truth_183).magnitude()
    )

    i_nearest = np.argmin(ang_dists_to_truth)
    nearest_dist = ang_dists_to_truth[i_nearest]
    print(f"Nearest candidate to truth@183: {nearest_dist:.2f} deg")

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nAttitude error percentiles (unique basins):")
    for p in pcts:
        print(f"  {p:>2}%ile: {np.percentile(ang_dists_to_truth, p):>7.2f} deg")

    # -----------------------------------------------------------------------
    # 8. Load m010 candidates and cross-compare
    # -----------------------------------------------------------------------
    print("\n--- Loading m010 candidates for comparison ---")
    m010_path = RESULTS_DIR / "m010_optimized_candidates.npz"
    m010_data = np.load(str(m010_path))
    m010_q_wxyz = m010_data['candidate_q_wxyz']   # (5643, 4) wxyz
    m010_ang_dists = m010_data['ang_dists_to_truth']
    n_micro10 = len(m010_q_wxyz)
    print(f"Loaded {n_micro10} m010 candidates")

    m010_nearest = m010_ang_dists.min()
    print(f"m010 nearest to truth@q0: {m010_nearest:.2f} deg")

    # Cross-match: for each m010 candidate, find nearest PAB-seeded candidate
    print("\n--- Cross-matching m010 vs PAB-seeded candidates ---")
    t_cross = time.time()

    # Pairwise angular distances between m010 and PAB-seeded reps
    R_micro10 = Rotation.from_quat(m010_q_wxyz[:, [1, 2, 3, 0]])

    # Process in batches to avoid memory explosion
    CROSS_MATCH_DEG = 2.0
    n_matched = 0
    min_cross_dists = np.full(n_micro10, np.inf)

    batch_size = 500
    for start in range(0, n_micro10, batch_size):
        end = min(start + batch_size, n_micro10)
        batch_R = R_micro10[start:end]

        # Compute pairwise distances: (batch, n_unique)
        for j in range(len(rep_q_wxyz)):
            dists_deg = np.degrees(
                (batch_R.inv() * R_reps[j]).magnitude()
            )
            min_cross_dists[start:end] = np.minimum(
                min_cross_dists[start:end], dists_deg,
            )

    n_matched = int(np.sum(min_cross_dists < CROSS_MATCH_DEG))
    cross_time = time.time() - t_cross

    print(f"Cross-match within {CROSS_MATCH_DEG} deg: "
          f"{n_matched}/{n_micro10} m010 candidates recovered "
          f"({100 * n_matched / n_micro10:.1f}%)")
    print(f"Cross-match time: {cross_time:.1f}s")

    # -----------------------------------------------------------------------
    # 9. Summary statistics
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"PAB-seeded approach:")
    print(f"  Total seeds:          {n_total_seeds}")
    print(f"  Converged:            {n_converged} "
          f"({100 * n_converged / n_total_seeds:.1f}%)")
    print(f"  Unique candidates:    {n_unique}")
    print(f"  Nearest to truth@183: {nearest_dist:.2f} deg")
    print()
    print(f"m010 (random SO(3)) comparison:")
    print(f"  Seeds:                10,000")
    print(f"  Unique candidates:    {n_micro10}")
    print(f"  Nearest to truth@q0:  {m010_nearest:.2f} deg")
    print()
    print(f"Cross-match ({CROSS_MATCH_DEG} deg):")
    print(f"  m010 candidates recovered by PAB-seeded: "
          f"{n_matched}/{n_micro10} ({100 * n_matched / n_micro10:.1f}%)")
    print()
    print(f"Efficiency ratio: "
          f"{n_unique}/{n_total_seeds} = "
          f"{n_unique / n_total_seeds:.4f} candidates/seed (PAB) vs "
          f"{n_micro10}/10000 = {n_micro10 / 10000:.4f} candidates/seed (random)")

    # -----------------------------------------------------------------------
    # 10. Plot: 3-panel figure
    # -----------------------------------------------------------------------
    print("\n--- Generating plot ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Panel 1: Histogram of angular distances to truth
    bins_dist = np.linspace(0, 180, 91)
    axes[0].hist(
        ang_dists_to_truth, bins=bins_dist, alpha=0.6, color='blue',
        label=f'PAB-seeded (N={n_unique})',
    )
    axes[0].hist(
        m010_ang_dists, bins=bins_dist, alpha=0.4, color='gray',
        label=f'm010 random (N={n_micro10})',
    )
    axes[0].axvline(
        nearest_dist, color='blue', linestyle='--', linewidth=1.5,
        label=f'PAB nearest = {nearest_dist:.1f} deg',
    )
    axes[0].axvline(
        m010_nearest, color='gray', linestyle=':', linewidth=1.5,
        label=f'm010 nearest = {m010_nearest:.1f} deg',
    )
    axes[0].set_xlabel('Angular distance to truth (deg)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Distance to Truth', fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Cross-match CDF
    sorted_cross = np.sort(min_cross_dists)
    cdf_y = np.arange(1, n_micro10 + 1) / n_micro10
    axes[1].plot(sorted_cross, cdf_y, 'b-', linewidth=1.5)
    axes[1].axvline(
        CROSS_MATCH_DEG, color='red', linestyle='--', linewidth=1,
        label=f'{CROSS_MATCH_DEG} deg threshold',
    )
    axes[1].axhline(
        n_matched / n_micro10, color='red', linestyle=':', linewidth=1,
        alpha=0.5,
        label=f'{100 * n_matched / n_micro10:.1f}% recovered',
    )
    axes[1].set_xlabel('Nearest PAB-seeded candidate (deg)', fontsize=11)
    axes[1].set_ylabel('Fraction of m010 candidates', fontsize=11)
    axes[1].set_title('Cross-match: m010 recovery', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 30)

    # Panel 3: Convergence rate by seed type
    # Count converged per offset category
    offset_categories = [0.0] + OFFSET_DEGS
    conv_counts = {od: 0 for od in offset_categories}
    total_counts = {od: 0 for od in offset_categories}

    # We need to match results back to seed metadata. Since imap_unordered
    # doesn't preserve order, we stored seed_idx in args but results came
    # back unordered. We don't have the mapping. Instead, just report overall.
    # Use a simpler bar: seeds per offset and total converged per offset.
    n_seeds_exact = N_PHI_EXACT * n_groups
    n_seeds_per_offset = N_RING_DIRS * N_PHI_RING * n_groups

    bar_labels = ['Exact (0 deg)'] + [f'{d} deg' for d in OFFSET_DEGS]
    bar_seeds = [n_seeds_exact] + [n_seeds_per_offset] * len(OFFSET_DEGS)

    axes[2].bar(
        range(len(bar_labels)), bar_seeds, alpha=0.6, color='steelblue',
    )
    axes[2].set_xticks(range(len(bar_labels)))
    axes[2].set_xticklabels(bar_labels, fontsize=9)
    axes[2].set_ylabel('Number of seeds', fontsize=11)
    axes[2].set_title('Seed Distribution by Offset', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')

    # Add text annotation with key stats
    stats_text = (
        f"Total seeds: {n_total_seeds}\n"
        f"Converged: {n_converged}\n"
        f"Unique: {n_unique}\n"
        f"Nearest: {nearest_dist:.1f} deg"
    )
    axes[2].text(
        0.95, 0.95, stats_text, transform=axes[2].transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    fig.suptitle(
        'Micro-38: PAB-Seeded Iso-Brightness Candidates '
        f'(Epoch {TARGET_EPOCH})',
        fontsize=14, fontweight='bold',
    )

    plot_path = RESULTS_DIR / "m038_pab_seeded_candidates.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # -----------------------------------------------------------------------
    # 11. Save results
    # -----------------------------------------------------------------------
    print("\n--- Saving results ---")

    # NPZ: converged quaternions for downstream use
    npz_path = RESULTS_DIR / "m038_pab_seeded_candidates.npz"
    np.savez(
        npz_path,
        rep_q_wxyz=rep_q_wxyz,
        ang_dists_to_truth=ang_dists_to_truth,
        all_converged_q_wxyz=q_converged,
        min_cross_dists_to_micro10=min_cross_dists,
        truth_q_183=q_truth_183,
        target_epoch=TARGET_EPOCH,
        target_mag=target_mag,
    )
    print(f"NPZ saved: {npz_path}")

    # JSON summary
    json_results = {
        'experiment': 'm038_pab_seeded_candidates',
        'target_epoch': TARGET_EPOCH,
        'target_mag': float(target_mag),
        'true_mag': float(CTX.true_lc[TARGET_EPOCH]),
        'n_unique_normals': n_groups,
        'seed_config': {
            'n_phi_exact': N_PHI_EXACT,
            'offset_degs': OFFSET_DEGS,
            'n_ring_dirs': N_RING_DIRS,
            'n_phi_ring': N_PHI_RING,
            'n_exact_seeds': n_seeds_exact,
            'n_offset_seeds': n_total_seeds - n_seeds_exact,
            'n_total_seeds': n_total_seeds,
        },
        'optimization': {
            'method': 'L-BFGS-B',
            'maxiter': 50,
            'convergence_tol': CONVERGENCE_TOL,
            'n_workers': N_WORKERS,
            'opt_time_s': round(opt_time, 1),
        },
        'results': {
            'n_converged': n_converged,
            'n_errors': n_errors,
            'convergence_rate': round(n_converged / n_total_seeds, 4),
            'n_unique_candidates': n_unique,
            'cluster_threshold_deg': CLUSTER_THRESHOLD_DEG,
            'nearest_to_truth_deg': round(float(nearest_dist), 2),
            'att_error_percentiles': {
                str(p): round(float(np.percentile(ang_dists_to_truth, p)), 2)
                for p in pcts
            },
        },
        'm010_comparison': {
            'n_micro10_candidates': n_micro10,
            'm010_nearest_to_truth_deg': round(float(m010_nearest), 2),
            'cross_match_threshold_deg': CROSS_MATCH_DEG,
            'n_micro10_recovered': n_matched,
            'recovery_fraction': round(n_matched / n_micro10, 4),
            'efficiency_pab': round(n_unique / n_total_seeds, 4),
            'efficiency_random': round(n_micro10 / 10000, 4),
        },
        'total_time_s': round(time.time() - t_global, 1),
    }

    json_path = RESULTS_DIR / "m038_pab_seeded_candidates.json"
    save_results(json_path, json_results)
    print(f"JSON saved: {json_path}")

    elapsed = time.time() - t_global
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done.")
