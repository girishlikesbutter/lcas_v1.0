#!/usr/bin/env python3
"""
m105 — Pairwise Peak Alignment Diagnostic.

Tests whether requiring a candidate omega0 to produce body-frame PAB near
lobe normals at TWO peaks from different lobe families simultaneously creates
a much tighter constraint than single-peak alignment.

Target seeds: 28 and 44 (grid-failure seeds).

Algorithm:
  1. Generate 2000-dir Fibonacci grid x 20 magnitude bins = 40,000 candidates.
  2. For each seed, select top-3 bright peak pairs from different lobe families.
  3. Propagate delta-q from identity for each candidate at peak epochs.
  4. For each peak pair, sweep psi (twist about n1) to check if any twist
     simultaneously aligns the rotated PAB at both peaks to their assigned lobes.
  5. Report survivor counts for oracle (known lobe assignments) and blind
     (all 100 lobe combos) modes, plus intersection across all 3 pairs.

Output: data/results/inversion_diagnostics/m105_pairwise/results.json
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path('/home/girish/projects/lcas_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_euler

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m105_pairwise"

TARGET_SEEDS = [28, 44]
N_DIRS = 2000
N_MAGS = 20
N_PSI = 72          # 5-degree twist steps
SURVIVAL_THRESH_DEG = 10.0
PROP_WORKERS = 24
MAG_MARGIN = 0.30   # +/- 30% around true |omega|


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def fibonacci_sphere(n):
    """Generate n approximately uniform points on the unit sphere."""
    indices = np.arange(n, dtype=float)
    phi_golden = (1 + np.sqrt(5)) / 2
    theta = np.arccos(1 - 2 * (indices + 0.5) / n)
    phi = 2 * np.pi * indices / phi_golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def wxyz_to_xyzw(q):
    """Convert scalar-first [w,x,y,z] to scipy [x,y,z,w]."""
    return np.array([q[1], q[2], q[3], q[0]])


def wxyz_to_xyzw_batch(q):
    """Convert batch of scalar-first [w,x,y,z] to scipy [x,y,z,w]."""
    return q[:, [1, 2, 3, 0]]


def lobe_family(name):
    """Map lobe name to family string for cross-family checks."""
    name_stripped = name.replace('+', '').replace('-', '')
    if name_stripped in ('X',):
        return 'X'
    elif name_stripped in ('Y',):
        return 'Y'
    elif name_stripped in ('Z',):
        return 'Z'
    else:
        return 'WD/ED'


def rotation_align(v_from, v_to):
    """
    Compute rotation matrix that maps v_from to v_to.
    Both must be unit vectors.  Returns scipy Rotation.
    Handles near-parallel and near-antiparallel cases.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = np.dot(v_from, v_to)

    if dot > 1.0 - 1e-12:
        # Already aligned
        return Rotation.identity()

    if dot < -1.0 + 1e-12:
        # Antiparallel — pick any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, perp)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)

    axis = np.cross(v_from, v_to)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return Rotation.from_rotvec(angle * axis)


def angular_distance_abs(v, n):
    """
    Angular distance between v and n, accounting for +/- sign ambiguity.
    Returns angle in degrees in [0, 90].
    """
    dot = np.clip(np.abs(np.dot(v, n)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


def angular_distance_abs_batch(vs, n):
    """
    Batch angular distance between rows of vs and a single n.
    Returns array of angles in degrees in [0, 90].
    """
    dots = np.clip(np.abs(vs @ n), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


# ---------------------------------------------------------------------------
# Propagation worker
# ---------------------------------------------------------------------------

def propagate_one(args):
    """Worker: propagate one omega0 candidate from identity quaternion."""
    omega0, times, I_tensor = args
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats, _ = propagate_euler(q_id, omega0, I_tensor, times, rtol=1e-8, atol=1e-10)
    return quats  # shape (len(times), 4)


# ---------------------------------------------------------------------------
# Peak pair selection
# ---------------------------------------------------------------------------

def select_peak_pairs(peak_epochs, peak_mags, peak_lobes, group_names, n_pairs=3):
    """
    Select top n_pairs bright peak pairs from different lobe families.

    Selection criteria:
      - Both peaks have mag < 9
      - Peaks are from different lobe families
      - |ep1 - ep2| > 50 epochs
      - Score = -(mag1 + mag2) to maximize brightness (lower mag = brighter)

    Returns list of dicts with ep1, ep2, lobe1, lobe2, mag1, mag2.
    """
    # Filter to bright peaks
    bright_mask = peak_mags < 9.0
    bright_eps = peak_epochs[bright_mask]
    bright_mags = peak_mags[bright_mask]
    bright_lobes = peak_lobes[bright_mask]
    bright_lobe_names = [group_names[g] for g in bright_lobes]

    n_bright = len(bright_eps)
    if n_bright < 2:
        return []

    # Generate all valid pairs
    candidates = []
    for i in range(n_bright):
        for j in range(i + 1, n_bright):
            fam_i = lobe_family(bright_lobe_names[i])
            fam_j = lobe_family(bright_lobe_names[j])
            if fam_i == fam_j:
                continue
            ep_sep = abs(int(bright_eps[i]) - int(bright_eps[j]))
            if ep_sep <= 50:
                continue
            score = -(bright_mags[i] + bright_mags[j])  # more negative = brighter
            candidates.append({
                'ep1': int(bright_eps[i]),
                'ep2': int(bright_eps[j]),
                'lobe1': bright_lobe_names[i],
                'lobe2': bright_lobe_names[j],
                'lobe1_idx': int(bright_lobes[i]),
                'lobe2_idx': int(bright_lobes[j]),
                'mag1': float(bright_mags[i]),
                'mag2': float(bright_mags[j]),
                'ep_sep': ep_sep,
                'score': score,
            })

    # Sort by brightness (most negative score first)
    candidates.sort(key=lambda c: c['score'])
    return candidates[:n_pairs]


# ---------------------------------------------------------------------------
# Core pairwise alignment check
# ---------------------------------------------------------------------------

def check_alignment_single_pair(delta_q_at_t1, delta_q_at_t2, pab1, pab2,
                                n1, n2, n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    For one candidate omega and one peak pair, check if any twist psi
    about n1 simultaneously aligns PAB at both peaks.

    Parameters
    ----------
    delta_q_at_t1 : (4,) wxyz quaternion at peak 1
    delta_q_at_t2 : (4,) wxyz quaternion at peak 2
    pab1 : (3,) inertial PAB at peak 1 epoch
    pab2 : (3,) inertial PAB at peak 2 epoch
    n1 : (3,) lobe normal for peak 1
    n2 : (3,) lobe normal for peak 2
    n_psi : int, number of twist angles to test
    thresh_deg : float, survival threshold in degrees

    Returns
    -------
    best_ang_dist_peak2 : float, minimum angular distance at peak 2 (degrees)
    """
    # Rotate inertial PAB through delta-q at each peak
    R_dq1 = Rotation.from_quat(wxyz_to_xyzw(delta_q_at_t1))
    R_dq2 = Rotation.from_quat(wxyz_to_xyzw(delta_q_at_t2))
    v1 = R_dq1.apply(pab1)
    v2 = R_dq2.apply(pab2)

    # Compute alignment rotation: v1 -> n1
    R_align = rotation_align(v1, n1)

    # Sweep twist psi about n1
    psi_values = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    twist_rotvecs = np.outer(psi_values, n1 / np.linalg.norm(n1))  # (N_PSI, 3)
    R_twists = Rotation.from_rotvec(twist_rotvecs)

    # R_q0 = R_twist @ R_align for each psi
    # Apply to v2
    v2_aligned = R_align.apply(v2)                    # (3,)
    v2_twisted = R_twists.apply(v2_aligned)            # (N_PSI, 3)

    # Angular distance to n2, accounting for +/- sign ambiguity
    ang_dists = angular_distance_abs_batch(v2_twisted, n2)  # (N_PSI,)
    return float(np.min(ang_dists))


def check_alignment_batch(delta_qs, t1_idx, t2_idx, pab1, pab2, n1, n2,
                          n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Vectorized alignment check for all candidates at one peak pair.

    Parameters
    ----------
    delta_qs : (n_cand, n_times, 4) wxyz quaternions
    t1_idx : int, index into the time axis for peak 1
    t2_idx : int, index into the time axis for peak 2
    pab1 : (3,) inertial PAB at peak 1
    pab2 : (3,) inertial PAB at peak 2
    n1, n2 : (3,) lobe normals

    Returns
    -------
    best_dists : (n_cand,) minimum angular distance at peak 2 across psi
    """
    n_cand = delta_qs.shape[0]
    best_dists = np.full(n_cand, 999.0)

    # Pre-compute twist rotations (shared across candidates)
    psi_values = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    n1_unit = n1 / np.linalg.norm(n1)
    twist_rotvecs = np.outer(psi_values, n1_unit)
    R_twists = Rotation.from_rotvec(twist_rotvecs)

    for i in range(n_cand):
        dq1 = delta_qs[i, t1_idx]
        dq2 = delta_qs[i, t2_idx]

        R_dq1 = Rotation.from_quat(wxyz_to_xyzw(dq1))
        R_dq2 = Rotation.from_quat(wxyz_to_xyzw(dq2))
        v1 = R_dq1.apply(pab1)
        v2 = R_dq2.apply(pab2)

        R_align = rotation_align(v1, n1)
        v2_aligned = R_align.apply(v2)
        v2_twisted = R_twists.apply(v2_aligned)

        ang_dists = angular_distance_abs_batch(v2_twisted, n2)
        best_dists[i] = np.min(ang_dists)

    return best_dists


def check_alignment_batch_blind(delta_qs, t1_idx, t2_idx, pab1, pab2,
                                unique_normals, group_names,
                                n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Blind mode: test all lobe combos (n1, n2) from different families.

    Returns
    -------
    survivor_mask : (n_cand,) bool — survives if ANY cross-family lobe
                    combo gives angular distance < threshold
    best_dists : (n_cand,) float — best angular distance across all combos
    """
    n_cand = delta_qs.shape[0]
    n_lobes = len(unique_normals)
    best_dists = np.full(n_cand, 999.0)

    for li in range(n_lobes):
        fam_i = lobe_family(group_names[li])
        for lj in range(n_lobes):
            fam_j = lobe_family(group_names[lj])
            if fam_i == fam_j:
                continue
            dists = check_alignment_batch(
                delta_qs, t1_idx, t2_idx, pab1, pab2,
                unique_normals[li], unique_normals[lj],
                n_psi=n_psi, thresh_deg=thresh_deg
            )
            best_dists = np.minimum(best_dists, dists)

    survivor_mask = best_dists < thresh_deg
    return survivor_mask, best_dists


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_seed(seed, traj_data):
    """Run pairwise alignment diagnostic for one seed."""
    t0_seed = time.time()
    print(f"\n{'='*70}")
    print(f"  Seed {seed}")
    print(f"{'='*70}")

    # Unpack trajectory data
    pab_j2000 = traj_data['pab_j2000']        # (500, 3)
    omega0s = traj_data['omega0s']              # (100, 3)
    q0s = traj_data['q0s']                      # (100, 4)
    omega_mags = traj_data['omega_mags']        # (100,) deg/s
    I_tensor = traj_data['inertia_tensor']      # (3, 3)
    peak_seeds = traj_data['peak_seeds']
    peak_epochs = traj_data['peak_epochs']
    peak_prominences = traj_data['peak_prominences']
    best_group = traj_data['best_group']        # (100, 500)
    unique_normals = traj_data['unique_normals']  # (10, 3)
    group_names = list(traj_data['group_names'])  # list of str
    mag_hifi = traj_data['mag_hifi']            # (100, 500)
    dt = float(traj_data['dt_sampling'])

    true_omega0 = omega0s[seed]                 # rad/s
    true_omega_mag_dps = omega_mags[seed]       # deg/s
    true_q0 = q0s[seed]                         # wxyz

    print(f"  True |omega| = {true_omega_mag_dps:.4f} deg/s")
    print(f"  True omega0  = {np.degrees(true_omega0)} deg/s")

    # --- Peak pair selection ---
    seed_mask = peak_seeds == seed
    s_peak_epochs = peak_epochs[seed_mask]
    s_peak_mags = mag_hifi[seed, s_peak_epochs]
    s_peak_lobes = best_group[seed, s_peak_epochs]

    pairs = select_peak_pairs(s_peak_epochs, s_peak_mags, s_peak_lobes,
                              group_names, n_pairs=3)
    if len(pairs) == 0:
        print("  WARNING: No valid peak pairs found!")
        return None

    print(f"\n  Selected {len(pairs)} peak pairs:")
    for p in pairs:
        print(f"    ep1={p['ep1']:3d} ({p['lobe1']}, mag={p['mag1']:.1f}) "
              f"ep2={p['ep2']:3d} ({p['lobe2']}, mag={p['mag2']:.1f}) "
              f"sep={p['ep_sep']} epochs")

    # --- Collect all unique epochs needed ---
    all_epochs_set = set()
    for p in pairs:
        all_epochs_set.add(p['ep1'])
        all_epochs_set.add(p['ep2'])
    all_epochs_sorted = sorted(all_epochs_set)
    epoch_to_tidx = {ep: i + 1 for i, ep in enumerate(all_epochs_sorted)}
    # times array: [0.0, ep1*dt, ep2*dt, ...]
    times = np.array([0.0] + [ep * dt for ep in all_epochs_sorted])
    n_times = len(times)

    print(f"\n  Propagation epochs: {all_epochs_sorted} ({n_times} times incl. t=0)")

    # --- Generate omega0 candidates ---
    dirs = fibonacci_sphere(N_DIRS)
    omega_mag_min = np.radians(true_omega_mag_dps * (1.0 - MAG_MARGIN))
    omega_mag_max = np.radians(true_omega_mag_dps * (1.0 + MAG_MARGIN))
    mags_rads = np.linspace(omega_mag_min, omega_mag_max, N_MAGS)

    # Build (N_DIRS * N_MAGS, 3) array of omega0 candidates
    omega_candidates = []
    for m in mags_rads:
        omega_candidates.append(dirs * m)
    omega_candidates = np.vstack(omega_candidates)  # (40000, 3)
    n_cand = len(omega_candidates)
    print(f"  Grid: {N_DIRS} dirs x {N_MAGS} mags = {n_cand} candidates")

    # --- Check truth proximity to grid ---
    true_dir = true_omega0 / np.linalg.norm(true_omega0)
    dir_dists = np.arccos(np.clip(dirs @ true_dir, -1.0, 1.0))
    closest_dir_idx = np.argmin(dir_dists)
    closest_dir_ang = np.degrees(dir_dists[closest_dir_idx])

    true_mag_rads = np.linalg.norm(true_omega0)
    mag_dists = np.abs(mags_rads - true_mag_rads)
    closest_mag_idx = np.argmin(mag_dists)
    closest_mag_err_pct = 100 * mag_dists[closest_mag_idx] / true_mag_rads

    print(f"  Truth -> nearest grid dir: {closest_dir_ang:.2f} deg "
          f"(dir idx {closest_dir_idx})")
    print(f"  Truth -> nearest grid mag: {closest_mag_err_pct:.2f}% error "
          f"(mag idx {closest_mag_idx})")

    # Find the index of the best matching candidate to truth
    cand_dists = np.linalg.norm(omega_candidates - true_omega0, axis=1)
    truth_cand_idx = np.argmin(cand_dists)
    truth_cand_dist_dps = np.degrees(cand_dists[truth_cand_idx])
    print(f"  Truth -> nearest candidate: idx {truth_cand_idx}, "
          f"dist {truth_cand_dist_dps:.4f} deg/s")

    # --- Propagate all candidates ---
    print(f"\n  Propagating {n_cand} candidates ({PROP_WORKERS} workers)...")
    t0_prop = time.time()

    args_list = [(omega_candidates[i], times, I_tensor) for i in range(n_cand)]
    with Pool(PROP_WORKERS) as pool:
        results_list = pool.map(propagate_one, args_list, chunksize=100)

    # Stack into (n_cand, n_times, 4)
    delta_qs = np.array(results_list)  # (n_cand, n_times, 4)
    dt_prop = time.time() - t0_prop
    print(f"  Propagation done in {dt_prop:.1f}s")

    # --- Check each peak pair ---
    pair_results = []
    all_oracle_survivors = np.ones(n_cand, dtype=bool)
    all_blind_survivors = np.ones(n_cand, dtype=bool)

    for pi, pair in enumerate(pairs):
        ep1, ep2 = pair['ep1'], pair['ep2']
        t1_idx = epoch_to_tidx[ep1]
        t2_idx = epoch_to_tidx[ep2]
        n1 = unique_normals[pair['lobe1_idx']]
        n2 = unique_normals[pair['lobe2_idx']]

        print(f"\n  --- Pair {pi}: ep {ep1} ({pair['lobe1']}) "
              f"<-> ep {ep2} ({pair['lobe2']}) ---")

        # Oracle mode: known lobe assignments
        t0_pair = time.time()
        oracle_dists = check_alignment_batch(
            delta_qs, t1_idx, t2_idx,
            pab_j2000[ep1], pab_j2000[ep2],
            n1, n2, n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
        )
        oracle_survivors = oracle_dists < SURVIVAL_THRESH_DEG
        n_oracle_surv = int(np.sum(oracle_survivors))
        truth_oracle_dist = float(oracle_dists[truth_cand_idx])
        truth_oracle_survives = bool(oracle_survivors[truth_cand_idx])

        # Truth rank among survivors
        if n_oracle_surv > 0 and truth_oracle_survives:
            # Rank by angular distance (lower = better)
            surv_dists = oracle_dists[oracle_survivors]
            truth_rank = int(np.sum(surv_dists <= truth_oracle_dist))
        else:
            truth_rank = -1

        dt_oracle = time.time() - t0_pair
        oracle_reduction = n_cand / n_oracle_surv if n_oracle_surv > 0 else float('inf')

        print(f"    ORACLE: {n_oracle_surv} survivors "
              f"(reduction {oracle_reduction:.0f}x), "
              f"truth dist={truth_oracle_dist:.2f} deg, "
              f"survives={truth_oracle_survives}, rank={truth_rank}, "
              f"[{dt_oracle:.1f}s]")

        # Blind mode: all cross-family lobe combos
        t0_blind = time.time()
        blind_surv_mask, blind_dists = check_alignment_batch_blind(
            delta_qs, t1_idx, t2_idx,
            pab_j2000[ep1], pab_j2000[ep2],
            unique_normals, group_names,
            n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
        )
        n_blind_surv = int(np.sum(blind_surv_mask))
        truth_blind_survives = bool(blind_surv_mask[truth_cand_idx])
        dt_blind = time.time() - t0_blind
        blind_reduction = n_cand / n_blind_surv if n_blind_surv > 0 else float('inf')

        # Count cross-family lobe combos
        n_lobes = len(unique_normals)
        n_cross_combos = 0
        for li in range(n_lobes):
            for lj in range(n_lobes):
                if lobe_family(group_names[li]) != lobe_family(group_names[lj]):
                    n_cross_combos += 1

        print(f"    BLIND:  {n_blind_surv} survivors "
              f"(reduction {blind_reduction:.0f}x), "
              f"truth survives={truth_blind_survives}, "
              f"combos={n_cross_combos}, [{dt_blind:.1f}s]")

        # Track intersection
        all_oracle_survivors &= oracle_survivors
        all_blind_survivors &= blind_surv_mask

        pair_results.append({
            'ep1': ep1,
            'ep2': ep2,
            'lobe1': pair['lobe1'],
            'lobe2': pair['lobe2'],
            'mag1': pair['mag1'],
            'mag2': pair['mag2'],
            'ep_sep': pair['ep_sep'],
            'oracle': {
                'n_survivors': n_oracle_surv,
                'truth_survives': truth_oracle_survives,
                'truth_best_ang_dist_peak2': truth_oracle_dist,
                'truth_rank': truth_rank,
                'reduction_factor': float(oracle_reduction),
            },
            'blind': {
                'n_lobe_combos': n_cross_combos,
                'n_survivors': n_blind_surv,
                'truth_survives': truth_blind_survives,
                'reduction_factor': float(blind_reduction),
            },
        })

    # --- Intersection across all pairs ---
    n_oracle_inter = int(np.sum(all_oracle_survivors))
    n_blind_inter = int(np.sum(all_blind_survivors))
    truth_oracle_inter = bool(all_oracle_survivors[truth_cand_idx])
    truth_blind_inter = bool(all_blind_survivors[truth_cand_idx])
    oracle_inter_reduction = n_cand / n_oracle_inter if n_oracle_inter > 0 else float('inf')
    blind_inter_reduction = n_cand / n_blind_inter if n_blind_inter > 0 else float('inf')

    print(f"\n  --- Intersection ({len(pairs)} pairs) ---")
    print(f"    ORACLE: {n_oracle_inter} survivors "
          f"(reduction {oracle_inter_reduction:.0f}x), "
          f"truth survives={truth_oracle_inter}")
    print(f"    BLIND:  {n_blind_inter} survivors "
          f"(reduction {blind_inter_reduction:.0f}x), "
          f"truth survives={truth_blind_inter}")

    dt_seed = time.time() - t0_seed
    print(f"\n  Seed {seed} total: {dt_seed:.1f}s")

    return {
        'seed': seed,
        'true_omega0_dps': [float(x) for x in np.degrees(true_omega0)],
        'true_omega_mag_dps': float(true_omega_mag_dps),
        'n_candidates': n_cand,
        'truth_nearest_dir_deg': float(closest_dir_ang),
        'truth_nearest_mag_pct': float(closest_mag_err_pct),
        'truth_nearest_cand_idx': int(truth_cand_idx),
        'peak_pairs': pair_results,
        'intersection': {
            'n_pairs': len(pairs),
            'oracle': {
                'n_survivors_3pair': n_oracle_inter,
                'truth_survives': truth_oracle_inter,
                'reduction_factor': float(oracle_inter_reduction),
            },
            'blind': {
                'n_survivors_3pair': n_blind_inter,
                'truth_survives': truth_blind_inter,
                'reduction_factor': float(blind_inter_reduction),
            },
        },
        'runtime_s': float(dt_seed),
    }


def print_summary(all_results):
    """Print a clear summary table."""
    print("\n")
    print("=" * 80)
    print("  MICRO105 — PAIRWISE PEAK ALIGNMENT DIAGNOSTIC — SUMMARY")
    print("=" * 80)

    for res in all_results:
        seed = res['seed']
        print(f"\n  Seed {seed}  (|omega| = {res['true_omega_mag_dps']:.4f} deg/s)")
        print(f"  {'':4s}  Truth grid proximity: "
              f"dir {res['truth_nearest_dir_deg']:.2f} deg, "
              f"mag {res['truth_nearest_mag_pct']:.2f}%")

        print(f"  {'Pair':6s}  {'Lobes':12s}  {'Sep':>5s}  "
              f"{'Oracle surv':>12s}  {'Red':>6s}  {'T?':>3s}  "
              f"{'Blind surv':>12s}  {'Red':>6s}  {'T?':>3s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*5}  "
              f"{'-'*12}  {'-'*6}  {'-'*3}  "
              f"{'-'*12}  {'-'*6}  {'-'*3}")

        for i, pp in enumerate(res['peak_pairs']):
            lobes = f"{pp['lobe1']}/{pp['lobe2']}"
            o = pp['oracle']
            b = pp['blind']
            print(f"  {i:6d}  {lobes:12s}  {pp['ep_sep']:5d}  "
                  f"{o['n_survivors']:12d}  {o['reduction_factor']:6.0f}x  "
                  f"{'Y' if o['truth_survives'] else 'N':>3s}  "
                  f"{b['n_survivors']:12d}  {b['reduction_factor']:6.0f}x  "
                  f"{'Y' if b['truth_survives'] else 'N':>3s}")

        inter = res['intersection']
        o = inter['oracle']
        b = inter['blind']
        print(f"  {'INTER':6s}  {'ALL':12s}  {'':5s}  "
              f"{o['n_survivors_3pair']:12d}  {o['reduction_factor']:6.0f}x  "
              f"{'Y' if o['truth_survives'] else 'N':>3s}  "
              f"{b['n_survivors_3pair']:12d}  {b['reduction_factor']:6.0f}x  "
              f"{'Y' if b['truth_survives'] else 'N':>3s}")

    print(f"\n{'='*80}\n")


def main():
    t0_total = time.time()

    # Load trajectory data
    traj_path = DATA_DIR / "m046_trajectories.npz"
    print(f"Loading {traj_path}")
    traj_data = dict(np.load(traj_path, allow_pickle=True))
    # Convert group_names from ndarray to list of str
    traj_data['group_names'] = list(traj_data['group_names'])

    all_results = []
    for seed in TARGET_SEEDS:
        res = run_seed(seed, traj_data)
        if res is not None:
            all_results.append(res)

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "results.json"

    # Replace inf with large sentinel for JSON compatibility
    def sanitize_for_json(obj):
        if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return 1e30
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print_summary(all_results)

    dt_total = time.time() - t0_total
    print(f"Total runtime: {dt_total:.1f}s")


if __name__ == '__main__':
    main()
