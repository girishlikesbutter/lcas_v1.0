#!/usr/bin/env python3
"""
m106 -- Vectorized Pairwise Peak Alignment at Scale.

Hypothesis: Vectorized pairwise peak alignment will achieve ~100x+ reduction
on 10K omega candidates in under 10 min per seed, and will work on all 5
grid-failure seeds (1, 11, 28, 44, 46).

Two-stage approach validated in m105 POC:
  Stage 1 -- 3-pair intersection: top 3 bright cross-family peak pairs,
             intersect survivors => ~100x reduction.
  Stage 2 -- Full-observation scoring: for each survivor, count how many of
             ALL bright peaks it can align at. Truth scores ~11/14; best FP ~6.

Key improvements over m105:
  - Pure numpy vectorization (zero Python loops over candidates)
  - 10,000 candidates (2000 dirs x 5 mag bins) instead of 40,000
  - Propagation at ALL bright peak epochs, not just pair epochs
  - Stage 2 full-observation scoring (m105 only had Stage 1)
  - Checkpoint/resume at every stage

Output: data/results/inversion_diagnostics/m106_pairwise_vec/seed_XXX/
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool

PROJECT_ROOT = Path('/home/girish/projects/lcas_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_euler

# ======================================================================
# Configuration
# ======================================================================

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_BASE = RESULTS_DIR / "m106_pairwise_vec"

# Seed 1 has only 2 bright peaks, both -X (same family) — 0 cross-family pairs.
# Included for documentation; run_seed() will return None and skip it gracefully.
TARGET_SEEDS = [1, 11, 28, 44, 46]
N_DIRS = 2000
N_MAGS = 5
N_PSI = 72            # 5-degree twist steps
SURVIVAL_THRESH_DEG = 10.0
PROP_WORKERS = 24
PROP_CHUNKSIZE = 50
MAG_MARGIN = 0.30     # +/- 30% around true |omega|
BRIGHT_MAG_THRESH = 9.0
PAIR_EPOCH_SEP = 50   # minimum epoch separation for peak pairs
N_PAIRS = 3           # number of peak pairs for Stage 1
STAGE2_N_PSI = 72     # psi resolution for Stage 2

# ======================================================================
# Dual logging
# ======================================================================

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()


# ======================================================================
# Utility functions
# ======================================================================

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


def quat_rotate_vec_batch(q_wxyz, v):
    """
    Rotate a single vector v (3,) by each of n quaternions q_wxyz (n, 4).
    Returns (n, 3).

    Uses the formula: v' = v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v))
    which avoids building rotation matrices entirely.
    """
    w = q_wxyz[:, 0:1]    # (n, 1)
    xyz = q_wxyz[:, 1:4]  # (n, 3)
    t = 2.0 * np.cross(xyz, v)           # (n, 3)
    return v + w * t + np.cross(xyz, t)   # (n, 3)


def quat_rotate_vec_batch_2d(q_wxyz, vs):
    """
    Rotate n vectors vs (n, 3) by n quaternions q_wxyz (n, 4) element-wise.
    Returns (n, 3).
    """
    w = q_wxyz[:, 0:1]    # (n, 1)
    xyz = q_wxyz[:, 1:4]  # (n, 3)
    t = 2.0 * np.cross(xyz, vs)           # (n, 3)
    return vs + w * t + np.cross(xyz, t)   # (n, 3)


def sanitize_for_json(obj):
    """Replace inf/nan with JSON-safe values."""
    if isinstance(obj, float):
        if obj == float('inf') or obj == float('-inf'):
            return 1e30
        if obj != obj:  # NaN
            return None
        return obj
    elif isinstance(obj, np.floating):
        return sanitize_for_json(float(obj))
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json_atomic(filepath, data):
    """Atomic JSON save (write to .tmp then rename)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(sanitize_for_json(data), f, indent=2)
    tmp_path.rename(filepath)


# ======================================================================
# Propagation worker
# ======================================================================

def propagate_one(args):
    """Worker: propagate one omega0 candidate from identity quaternion."""
    omega0, times, I_tensor = args
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats, _ = propagate_euler(q_id, omega0, I_tensor, times, rtol=1e-8, atol=1e-10)
    return quats  # shape (len(times), 4) wxyz


# ======================================================================
# Peak pair selection (same logic as m105)
# ======================================================================

def select_peak_pairs(peak_epochs, peak_mags, peak_lobes, group_names, n_pairs=N_PAIRS):
    """
    Select top n_pairs bright peak pairs from different lobe families.

    Selection criteria:
      - Both peaks have mag < BRIGHT_MAG_THRESH
      - Peaks are from different lobe families
      - |ep1 - ep2| > PAIR_EPOCH_SEP epochs
      - Score = -(mag1 + mag2) to maximize brightness (lower mag = brighter)

    Returns list of dicts with ep1, ep2, lobe1, lobe2, mag1, mag2.
    """
    bright_mask = peak_mags < BRIGHT_MAG_THRESH
    bright_eps = peak_epochs[bright_mask]
    bright_mags = peak_mags[bright_mask]
    bright_lobes = peak_lobes[bright_mask]
    bright_lobe_names = [group_names[g] for g in bright_lobes]

    n_bright = len(bright_eps)
    if n_bright < 2:
        return []

    candidates = []
    for i in range(n_bright):
        for j in range(i + 1, n_bright):
            fam_i = lobe_family(bright_lobe_names[i])
            fam_j = lobe_family(bright_lobe_names[j])
            if fam_i == fam_j:
                continue
            ep_sep = abs(int(bright_eps[i]) - int(bright_eps[j]))
            if ep_sep <= PAIR_EPOCH_SEP:
                continue
            score = -(bright_mags[i] + bright_mags[j])
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

    candidates.sort(key=lambda c: c['score'])
    return candidates[:n_pairs]


def get_bright_peaks(seed, traj_data):
    """Return sorted bright peak epoch indices and their lobe assignments for a seed."""
    peak_seeds = traj_data['peak_seeds']
    peak_epochs = traj_data['peak_epochs']
    best_group = traj_data['best_group']
    mag_hifi = traj_data['mag_hifi']
    group_names = traj_data['group_names']

    seed_mask = peak_seeds == seed
    s_peak_epochs = peak_epochs[seed_mask]
    s_peak_mags = mag_hifi[seed, s_peak_epochs]
    s_peak_lobes = best_group[seed, s_peak_epochs]

    bright_mask = s_peak_mags < BRIGHT_MAG_THRESH
    bright_eps = s_peak_epochs[bright_mask]
    bright_mags = s_peak_mags[bright_mask]
    bright_lobes = s_peak_lobes[bright_mask]
    bright_lobe_names = [group_names[g] for g in bright_lobes]

    # Sort by brightness (lowest mag first)
    order = np.argsort(bright_mags)
    return bright_eps[order], bright_mags[order], bright_lobes[order], \
        [bright_lobe_names[i] for i in order]


# ======================================================================
# Stage 1: Vectorized pairwise alignment check
# ======================================================================

def check_alignment_vectorized(delta_qs_t1, delta_qs_t2, pab1, pab2, n1, n2,
                               n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Fully vectorized alignment check for ALL candidates at one peak pair.

    For each candidate, we:
      1. Rotate pab1 by delta_q at t1 => v1 (n_cand, 3)
      2. Rotate pab2 by delta_q at t2 => v2 (n_cand, 3)
      3. Compute Rodrigues R_align that maps v1 -> n1, apply to v2 => v2_aligned
      4. Sweep n_psi twist angles about n1, apply to v2_aligned
      5. Measure angular distance to n2 for each psi, take min

    Zero Python loops over candidates.

    Parameters
    ----------
    delta_qs_t1 : (n_cand, 4) wxyz quaternions at peak 1 epoch
    delta_qs_t2 : (n_cand, 4) wxyz quaternions at peak 2 epoch
    pab1 : (3,) inertial PAB at peak 1
    pab2 : (3,) inertial PAB at peak 2
    n1, n2 : (3,) lobe normals
    n_psi : int, number of twist angles
    thresh_deg : float, survival threshold

    Returns
    -------
    best_dists : (n_cand,) min angular distance at peak 2 across psi values
    """
    n_cand = delta_qs_t1.shape[0]

    # Step 1-2: Rotate PABs through delta-q
    v1 = quat_rotate_vec_batch(delta_qs_t1, pab1)  # (n_cand, 3)
    v2 = quat_rotate_vec_batch(delta_qs_t2, pab2)  # (n_cand, 3)

    # Step 3: Rodrigues rotation mapping v1 -> n1, applied to v2
    # axis = cross(v1, n1), angle via cos = dot(v1, n1), sin = |cross|
    n1_unit = n1 / np.linalg.norm(n1)
    axes_raw = np.cross(v1, n1_unit)           # (n_cand, 3)
    sin_a = np.linalg.norm(axes_raw, axis=1)   # (n_cand,)
    cos_a = v1 @ n1_unit                       # (n_cand,)

    # Normalize axes (handle near-parallel: sin_a ~ 0)
    safe_sin = np.maximum(sin_a, 1e-12)
    axes = axes_raw / safe_sin[:, None]         # (n_cand, 3)

    # Rodrigues on v2: v2_aligned = v2*cos_a + (axis x v2)*sin_a + axis*(axis . v2)*(1 - cos_a)
    axv2 = np.cross(axes, v2)                  # (n_cand, 3)
    adv2 = np.sum(axes * v2, axis=1)           # (n_cand,)
    v2_aligned = (v2 * cos_a[:, None]
                  + axv2 * sin_a[:, None]
                  + axes * (adv2 * (1 - cos_a))[:, None])  # (n_cand, 3)

    # Handle near-parallel case (v1 ~ n1): v2_aligned = v2
    parallel_mask = sin_a < 1e-10
    if np.any(parallel_mask):
        v2_aligned[parallel_mask] = v2[parallel_mask]

    # Handle near-antiparallel case (v1 ~ -n1): need 180-deg rotation about any perp axis
    antiparallel_mask = cos_a < -1.0 + 1e-10
    if np.any(antiparallel_mask):
        # For antiparallel v1, R_align is 180 deg about any axis perp to n1
        # v2_aligned = -v2 + 2*n1*(n1.v2) — reflection through plane containing n1
        n1dv2_anti = v2[antiparallel_mask] @ n1_unit
        v2_aligned[antiparallel_mask] = (-v2[antiparallel_mask]
                                          + 2 * n1_unit * n1dv2_anti[:, None])

    # Step 4: Twist sweep about n1
    psi_values = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    cos_psi = np.cos(psi_values)               # (n_psi,)
    sin_psi = np.sin(psi_values)               # (n_psi,)

    # Rodrigues twist: n1 x v2_aligned, n1 . v2_aligned
    n1xv = np.cross(n1_unit, v2_aligned)       # (n_cand, 3)
    n1dv = v2_aligned @ n1_unit                # (n_cand,)

    # v_twisted[i, j, :] = v2_aligned[i]*cos_psi[j] + (n1 x v2_aligned[i])*sin_psi[j]
    #                     + n1*(n1 . v2_aligned[i])*(1 - cos_psi[j])
    # Shape: (n_cand, n_psi, 3)
    term1 = v2_aligned[:, None, :] * cos_psi[None, :, None]
    term2 = n1xv[:, None, :] * sin_psi[None, :, None]
    term3 = n1_unit[None, None, :] * (n1dv[:, None] * (1 - cos_psi[None, :]))[:, :, None]
    v_twisted = term1 + term2 + term3          # (n_cand, n_psi, 3)

    # Step 5: Angular distance to n2, with +/- sign ambiguity
    n2_unit = n2 / np.linalg.norm(n2)
    dots = np.abs(v_twisted @ n2_unit)         # (n_cand, n_psi)
    dots = np.clip(dots, 0.0, 1.0)
    ang_dists = np.degrees(np.arccos(dots))    # (n_cand, n_psi)
    best_dists = ang_dists.min(axis=1)         # (n_cand,)

    return best_dists


def check_alignment_blind_vectorized(delta_qs_t1, delta_qs_t2, pab1, pab2,
                                     unique_normals, group_names,
                                     n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Blind mode: test all cross-family lobe combos (n1, n2).
    A candidate survives if ANY cross-family combo gives dist < threshold.

    Returns
    -------
    best_dists : (n_cand,) best angular distance across all lobe combos
    """
    n_cand = delta_qs_t1.shape[0]
    n_lobes = len(unique_normals)
    best_dists = np.full(n_cand, 999.0)

    for li in range(n_lobes):
        fam_i = lobe_family(group_names[li])
        for lj in range(n_lobes):
            fam_j = lobe_family(group_names[lj])
            if fam_i == fam_j:
                continue
            dists = check_alignment_vectorized(
                delta_qs_t1, delta_qs_t2, pab1, pab2,
                unique_normals[li], unique_normals[lj],
                n_psi=n_psi, thresh_deg=thresh_deg
            )
            best_dists = np.minimum(best_dists, dists)

    return best_dists


# ======================================================================
# Stage 2: Full-observation scoring
# ======================================================================

def score_full_observation_oracle(delta_qs, survivor_indices, bright_epoch_tidxs,
                                  pab_at_epochs, lobe_normals_at_epochs,
                                  n_psi=STAGE2_N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Stage 2 oracle scoring: for each survivor, use the brightest peak as
    reference, sweep psi at that peak, and count how many other bright peaks
    align under each psi.

    Parameters
    ----------
    delta_qs : (n_cand, n_times, 4) wxyz quaternions (full candidate array)
    survivor_indices : (n_surv,) int indices into the candidate axis
    bright_epoch_tidxs : (n_peaks,) int indices into the time axis
    pab_at_epochs : (n_peaks, 3) inertial PAB at each peak epoch
    lobe_normals_at_epochs : (n_peaks, 3) assigned lobe normals per peak
    n_psi : int, psi resolution
    thresh_deg : float, alignment threshold

    Returns
    -------
    scores : (n_surv,) int, number of peaks aligned (including reference)
    best_psi_idx : (n_surv,) int, which psi gave the best score
    """
    n_surv = len(survivor_indices)
    n_peaks = len(bright_epoch_tidxs)

    if n_peaks < 2:
        return np.ones(n_surv, dtype=int), np.zeros(n_surv, dtype=int)

    # Reference peak = index 0 (brightest, since peaks are sorted by brightness)
    ref_tidx = bright_epoch_tidxs[0]
    ref_pab = pab_at_epochs[0]
    ref_normal = lobe_normals_at_epochs[0]
    ref_normal_unit = ref_normal / np.linalg.norm(ref_normal)

    # Psi values
    psi_values = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    cos_psi = np.cos(psi_values)  # (n_psi,)
    sin_psi = np.sin(psi_values)  # (n_psi,)

    scores = np.zeros(n_surv, dtype=int)
    best_psi_idx = np.zeros(n_surv, dtype=int)

    # Process survivors in batch
    # Extract survivor quaternions at all peak times
    surv_dqs = delta_qs[survivor_indices]  # (n_surv, n_times, 4)

    # For reference peak: rotate ref_pab by each survivor's delta-q
    ref_dqs = surv_dqs[:, ref_tidx, :]  # (n_surv, 4)
    v_ref = quat_rotate_vec_batch(ref_dqs, ref_pab)  # (n_surv, 3)

    # Compute R_align: v_ref -> ref_normal (Rodrigues)
    axes_raw = np.cross(v_ref, ref_normal_unit)
    sin_a = np.linalg.norm(axes_raw, axis=1)
    cos_a = v_ref @ ref_normal_unit
    safe_sin = np.maximum(sin_a, 1e-12)
    axes = axes_raw / safe_sin[:, None]

    # For each non-reference peak, compute the rotated PAB, align, twist, check
    # We accumulate a score matrix: (n_surv, n_psi) = count of aligned peaks per psi
    score_matrix = np.ones((n_surv, n_psi), dtype=int)  # reference always counts

    for pk in range(1, n_peaks):
        pk_tidx = bright_epoch_tidxs[pk]
        pk_pab = pab_at_epochs[pk]
        pk_normal = lobe_normals_at_epochs[pk]
        pk_normal_unit = pk_normal / np.linalg.norm(pk_normal)

        # Rotate pk_pab by each survivor's delta-q at this peak's epoch
        pk_dqs = surv_dqs[:, pk_tidx, :]  # (n_surv, 4)
        v_pk = quat_rotate_vec_batch(pk_dqs, pk_pab)  # (n_surv, 3)

        # Apply R_align (same alignment from reference peak) to v_pk via Rodrigues
        axv = np.cross(axes, v_pk)
        adv = np.sum(axes * v_pk, axis=1)
        v_pk_aligned = (v_pk * cos_a[:, None]
                        + axv * sin_a[:, None]
                        + axes * (adv * (1 - cos_a))[:, None])

        # Handle parallel and antiparallel
        par_mask = sin_a < 1e-10
        if np.any(par_mask):
            v_pk_aligned[par_mask] = v_pk[par_mask]
        anti_mask = cos_a < -1.0 + 1e-10
        if np.any(anti_mask):
            n_dot_v = v_pk[anti_mask] @ ref_normal_unit
            v_pk_aligned[anti_mask] = -v_pk[anti_mask] + 2 * ref_normal_unit * n_dot_v[:, None]

        # Twist about ref_normal: v_twisted = v_pk_aligned*cos + (n x v)*sin + n*(n.v)*(1-cos)
        nxv = np.cross(ref_normal_unit, v_pk_aligned)  # (n_surv, 3)
        ndv = v_pk_aligned @ ref_normal_unit            # (n_surv,)

        # (n_surv, n_psi, 3)
        t1 = v_pk_aligned[:, None, :] * cos_psi[None, :, None]
        t2 = nxv[:, None, :] * sin_psi[None, :, None]
        t3 = ref_normal_unit[None, None, :] * (ndv[:, None] * (1 - cos_psi[None, :]))[:, :, None]
        v_twisted = t1 + t2 + t3

        # Angular distance to this peak's assigned lobe normal
        dots = np.abs(v_twisted @ pk_normal_unit)  # (n_surv, n_psi)
        dots = np.clip(dots, 0.0, 1.0)
        ang_dists = np.degrees(np.arccos(dots))

        # Does this peak align under threshold?
        aligned = (ang_dists < thresh_deg).astype(int)  # (n_surv, n_psi)
        score_matrix += aligned

    # Best psi per survivor
    best_psi_idx = np.argmax(score_matrix, axis=1)   # (n_surv,)
    scores = score_matrix[np.arange(n_surv), best_psi_idx]  # (n_surv,)

    return scores, best_psi_idx


def score_full_observation_blind(delta_qs, survivor_indices, bright_epoch_tidxs,
                                 pab_at_epochs, unique_normals, group_names,
                                 n_psi=STAGE2_N_PSI, thresh_deg=SURVIVAL_THRESH_DEG):
    """
    Stage 2 blind scoring: for each survivor, try all lobes as the reference
    lobe assignment, and for each non-reference peak, try all lobes.

    For efficiency, we fix the reference peak's lobe and sweep psi. For each
    psi, we assign each other peak to whichever lobe gives the smallest
    angular distance (if < threshold, count it).

    This means we loop over reference lobe choices (n_lobes) but vectorize
    over survivors and psi.

    Returns
    -------
    scores : (n_surv,) int, best score across all reference lobe choices
    """
    n_surv = len(survivor_indices)
    n_peaks = len(bright_epoch_tidxs)
    n_lobes = len(unique_normals)

    if n_peaks < 2:
        return np.ones(n_surv, dtype=int)

    surv_dqs = delta_qs[survivor_indices]  # (n_surv, n_times, 4)

    ref_tidx = bright_epoch_tidxs[0]
    ref_pab = pab_at_epochs[0]

    psi_values = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    cos_psi = np.cos(psi_values)
    sin_psi = np.sin(psi_values)

    best_scores = np.zeros(n_surv, dtype=int)

    # Rotate ref_pab by each survivor's delta-q at reference epoch
    ref_dqs = surv_dqs[:, ref_tidx, :]  # (n_surv, 4)
    v_ref = quat_rotate_vec_batch(ref_dqs, ref_pab)  # (n_surv, 3)

    # Pre-rotate all peak PABs
    v_peaks = []  # list of (n_surv, 3)
    for pk in range(1, n_peaks):
        pk_dqs = surv_dqs[:, bright_epoch_tidxs[pk], :]
        v_peaks.append(quat_rotate_vec_batch(pk_dqs, pab_at_epochs[pk]))

    # Try each lobe as the reference assignment
    for ref_li in range(n_lobes):
        ref_normal = unique_normals[ref_li]
        ref_normal_unit = ref_normal / np.linalg.norm(ref_normal)

        # R_align: v_ref -> ref_normal
        axes_raw = np.cross(v_ref, ref_normal_unit)
        sin_a = np.linalg.norm(axes_raw, axis=1)
        cos_a = v_ref @ ref_normal_unit
        safe_sin = np.maximum(sin_a, 1e-12)
        axes = axes_raw / safe_sin[:, None]

        # Score matrix: (n_surv, n_psi), starting at 1 for the reference peak
        score_matrix = np.ones((n_surv, n_psi), dtype=int)

        for pk_i, v_pk in enumerate(v_peaks):
            # Apply R_align to v_pk
            axv = np.cross(axes, v_pk)
            adv = np.sum(axes * v_pk, axis=1)
            v_pk_aligned = (v_pk * cos_a[:, None]
                            + axv * sin_a[:, None]
                            + axes * (adv * (1 - cos_a))[:, None])

            par_mask = sin_a < 1e-10
            if np.any(par_mask):
                v_pk_aligned[par_mask] = v_pk[par_mask]
            anti_mask = cos_a < -1.0 + 1e-10
            if np.any(anti_mask):
                n_dot_v = v_pk[anti_mask] @ ref_normal_unit
                v_pk_aligned[anti_mask] = (-v_pk[anti_mask]
                                           + 2 * ref_normal_unit * n_dot_v[:, None])

            # Twist about ref_normal
            nxv = np.cross(ref_normal_unit, v_pk_aligned)
            ndv = v_pk_aligned @ ref_normal_unit

            t1 = v_pk_aligned[:, None, :] * cos_psi[None, :, None]
            t2 = nxv[:, None, :] * sin_psi[None, :, None]
            t3 = (ref_normal_unit[None, None, :]
                  * (ndv[:, None] * (1 - cos_psi[None, :]))[:, :, None])
            v_twisted = t1 + t2 + t3  # (n_surv, n_psi, 3)

            # Check against ALL lobes, take best
            min_ang = np.full((n_surv, n_psi), 999.0)
            for lj in range(n_lobes):
                nj = unique_normals[lj] / np.linalg.norm(unique_normals[lj])
                dots = np.abs(v_twisted @ nj)  # (n_surv, n_psi)
                dots = np.clip(dots, 0.0, 1.0)
                ang = np.degrees(np.arccos(dots))
                min_ang = np.minimum(min_ang, ang)

            aligned = (min_ang < thresh_deg).astype(int)
            score_matrix += aligned

        # Best psi per survivor for this reference lobe
        psi_scores = score_matrix.max(axis=1)  # (n_surv,)
        best_scores = np.maximum(best_scores, psi_scores)

    return best_scores


# ======================================================================
# Main per-seed logic
# ======================================================================

def run_seed(seed, traj_data):
    """Run full two-stage pairwise alignment for one seed."""
    t0_seed = time.time()

    # Output directory for this seed
    seed_dir = OUT_BASE / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = open(str(seed_dir / "pipeline.log"), "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"\n{'='*70}")
    print(f"  MICRO106 -- Seed {seed}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Unpack trajectory data
    # ------------------------------------------------------------------
    pab_j2000 = traj_data['pab_j2000']        # (500, 3)
    omega0s = traj_data['omega0s']             # (100, 3)
    q0s = traj_data['q0s']                     # (100, 4) wxyz
    omega_mags = traj_data['omega_mags']       # (100,) deg/s
    I_tensor = traj_data['inertia_tensor']     # (3, 3)
    best_group = traj_data['best_group']       # (100, 500)
    unique_normals = traj_data['unique_normals']  # (10, 3)
    group_names = traj_data['group_names']
    mag_hifi = traj_data['mag_hifi']           # (100, 500)
    dt = float(traj_data['dt_sampling'])

    true_omega0 = omega0s[seed]                # rad/s
    true_omega_mag_dps = omega_mags[seed]      # deg/s
    true_q0 = q0s[seed]                        # wxyz

    print(f"  True |omega| = {true_omega_mag_dps:.4f} deg/s")
    print(f"  True omega0  = [{', '.join(f'{x:.6f}' for x in np.degrees(true_omega0))}] deg/s")

    # ------------------------------------------------------------------
    # Get bright peaks and select pairs
    # ------------------------------------------------------------------
    bright_eps, bright_mags, bright_lobes, bright_lobe_names = \
        get_bright_peaks(seed, traj_data)
    n_bright = len(bright_eps)

    print(f"\n  Bright peaks (mag < {BRIGHT_MAG_THRESH}): {n_bright}")
    for i in range(min(n_bright, 20)):
        print(f"    ep {bright_eps[i]:3d}  mag {bright_mags[i]:.1f}  "
              f"lobe {bright_lobe_names[i]}")

    # Peak pair selection for Stage 1
    peak_seeds = traj_data['peak_seeds']
    peak_epochs = traj_data['peak_epochs']
    seed_mask = peak_seeds == seed
    s_peak_epochs = peak_epochs[seed_mask]
    s_peak_mags = mag_hifi[seed, s_peak_epochs]
    s_peak_lobes = best_group[seed, s_peak_epochs]

    pairs = select_peak_pairs(s_peak_epochs, s_peak_mags, s_peak_lobes,
                              group_names, n_pairs=N_PAIRS)
    if len(pairs) == 0:
        print("  WARNING: No valid peak pairs found!")
        sys.stdout = old_stdout
        log_file.close()
        return None

    print(f"\n  Selected {len(pairs)} peak pairs for Stage 1:")
    for p in pairs:
        print(f"    ep1={p['ep1']:3d} ({p['lobe1']}, mag={p['mag1']:.1f}) "
              f"ep2={p['ep2']:3d} ({p['lobe2']}, mag={p['mag2']:.1f}) "
              f"sep={p['ep_sep']} epochs")

    # ------------------------------------------------------------------
    # Collect ALL unique epochs: pair epochs + all bright peak epochs
    # ------------------------------------------------------------------
    all_epochs_set = set()
    for p in pairs:
        all_epochs_set.add(p['ep1'])
        all_epochs_set.add(p['ep2'])
    for ep in bright_eps:
        all_epochs_set.add(int(ep))

    all_epochs_sorted = sorted(all_epochs_set)
    epoch_to_tidx = {ep: i + 1 for i, ep in enumerate(all_epochs_sorted)}
    # times array: [0.0, ep1*dt, ep2*dt, ...]
    times = np.array([0.0] + [ep * dt for ep in all_epochs_sorted])
    n_times = len(times)

    print(f"\n  Propagation epochs: {len(all_epochs_sorted)} unique "
          f"({n_times} times incl. t=0)")

    # ------------------------------------------------------------------
    # Generate omega0 candidate grid
    # ------------------------------------------------------------------
    dirs = fibonacci_sphere(N_DIRS)
    omega_mag_min = np.radians(true_omega_mag_dps * (1.0 - MAG_MARGIN))
    omega_mag_max = np.radians(true_omega_mag_dps * (1.0 + MAG_MARGIN))
    mags_rads = np.linspace(omega_mag_min, omega_mag_max, N_MAGS)

    omega_candidates = np.vstack([dirs * m for m in mags_rads])  # (N_DIRS*N_MAGS, 3)
    n_cand = len(omega_candidates)
    print(f"  Grid: {N_DIRS} dirs x {N_MAGS} mags = {n_cand} candidates")

    # Truth proximity
    true_dir = true_omega0 / np.linalg.norm(true_omega0)
    dir_dists = np.arccos(np.clip(dirs @ true_dir, -1.0, 1.0))
    closest_dir_idx = np.argmin(dir_dists)
    closest_dir_ang = np.degrees(dir_dists[closest_dir_idx])

    true_mag_rads = np.linalg.norm(true_omega0)
    mag_dists = np.abs(mags_rads - true_mag_rads)
    closest_mag_idx = np.argmin(mag_dists)
    closest_mag_err_pct = 100 * mag_dists[closest_mag_idx] / true_mag_rads

    cand_dists = np.linalg.norm(omega_candidates - true_omega0, axis=1)
    truth_cand_idx = np.argmin(cand_dists)
    truth_cand_dist_dps = np.degrees(cand_dists[truth_cand_idx])

    print(f"  Truth -> nearest grid dir: {closest_dir_ang:.2f} deg")
    print(f"  Truth -> nearest grid mag: {closest_mag_err_pct:.2f}% error")
    print(f"  Truth -> nearest candidate: idx {truth_cand_idx}, "
          f"dist {truth_cand_dist_dps:.4f} deg/s")

    # ==================================================================
    # Stage 0: Propagation (with checkpoint)
    # ==================================================================
    stage0_ckpt = seed_dir / "stage0_delta_qs.npz"
    if stage0_ckpt.exists():
        print(f"\n  Stage 0: Loading checkpoint {stage0_ckpt}")
        ckpt = np.load(str(stage0_ckpt))
        delta_qs = ckpt['delta_qs']
        dt_prop = float(ckpt.get('runtime_s', 0))
        print(f"  Stage 0: Loaded {delta_qs.shape} delta_qs (was {dt_prop:.1f}s)")
    else:
        print(f"\n  Stage 0: Propagating {n_cand} candidates "
              f"({PROP_WORKERS} workers, chunksize={PROP_CHUNKSIZE})...")
        t0_prop = time.time()

        args_list = [(omega_candidates[i], times, I_tensor) for i in range(n_cand)]
        with Pool(PROP_WORKERS) as pool:
            results_list = pool.map(propagate_one, args_list, chunksize=PROP_CHUNKSIZE)

        delta_qs = np.array(results_list)  # (n_cand, n_times, 4)
        dt_prop = time.time() - t0_prop
        print(f"  Stage 0: Propagation done in {dt_prop:.1f}s, shape={delta_qs.shape}")

        # Save checkpoint
        np.savez(str(stage0_ckpt), delta_qs=delta_qs, runtime_s=dt_prop,
                 omega_candidates=omega_candidates, times=times,
                 epoch_to_tidx_keys=np.array(list(epoch_to_tidx.keys())),
                 epoch_to_tidx_vals=np.array(list(epoch_to_tidx.values())))
        print(f"  Saved: {stage0_ckpt}")

    # ==================================================================
    # Stage 1: 3-pair fast filter (VECTORIZED)
    # ==================================================================
    stage1_ckpt = seed_dir / "stage1_results.npz"
    if stage1_ckpt.exists():
        print(f"\n  Stage 1: Loading checkpoint {stage1_ckpt}")
        ckpt = np.load(str(stage1_ckpt), allow_pickle=True)
        all_oracle_survivors = ckpt['oracle_intersection']
        all_blind_survivors = ckpt['blind_intersection']
        pair_results_list = [x.item() if hasattr(x, 'item') else x
                              for x in ckpt['pair_results']]
        dt_stage1 = float(ckpt.get('runtime_s', 0))
        print(f"  Stage 1: Oracle intersection: {np.sum(all_oracle_survivors)} survivors")
        print(f"  Stage 1: Blind intersection:  {np.sum(all_blind_survivors)} survivors")
    else:
        print(f"\n  Stage 1: 3-pair fast filter (vectorized)")
        t0_stage1 = time.time()

        all_oracle_survivors = np.ones(n_cand, dtype=bool)
        all_blind_survivors = np.ones(n_cand, dtype=bool)
        pair_results_list = []

        for pi, pair in enumerate(pairs):
            ep1, ep2 = pair['ep1'], pair['ep2']
            t1_idx = epoch_to_tidx[ep1]
            t2_idx = epoch_to_tidx[ep2]
            n1 = unique_normals[pair['lobe1_idx']]
            n2 = unique_normals[pair['lobe2_idx']]

            print(f"\n  --- Pair {pi}: ep {ep1} ({pair['lobe1']}) "
                  f"<-> ep {ep2} ({pair['lobe2']}) ---")

            # Oracle mode
            t0_pair = time.time()
            oracle_dists = check_alignment_vectorized(
                delta_qs[:, t1_idx, :], delta_qs[:, t2_idx, :],
                pab_j2000[ep1], pab_j2000[ep2], n1, n2,
                n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
            )
            oracle_survivors = oracle_dists < SURVIVAL_THRESH_DEG
            n_oracle_surv = int(np.sum(oracle_survivors))
            truth_oracle_dist = float(oracle_dists[truth_cand_idx])
            truth_oracle_survives = bool(oracle_survivors[truth_cand_idx])
            dt_oracle = time.time() - t0_pair
            oracle_reduction = n_cand / n_oracle_surv if n_oracle_surv > 0 else float('inf')

            print(f"    ORACLE: {n_oracle_surv} survivors "
                  f"(reduction {oracle_reduction:.0f}x), "
                  f"truth dist={truth_oracle_dist:.2f} deg, "
                  f"survives={truth_oracle_survives} [{dt_oracle:.1f}s]")

            # Blind mode
            t0_blind = time.time()
            blind_dists = check_alignment_blind_vectorized(
                delta_qs[:, t1_idx, :], delta_qs[:, t2_idx, :],
                pab_j2000[ep1], pab_j2000[ep2],
                unique_normals, group_names,
                n_psi=N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
            )
            blind_survivors = blind_dists < SURVIVAL_THRESH_DEG
            n_blind_surv = int(np.sum(blind_survivors))
            truth_blind_survives = bool(blind_survivors[truth_cand_idx])
            dt_blind = time.time() - t0_blind
            blind_reduction = n_cand / n_blind_surv if n_blind_surv > 0 else float('inf')

            # Count cross-family combos
            n_lobes = len(unique_normals)
            n_cross_combos = sum(
                1 for li in range(n_lobes) for lj in range(n_lobes)
                if lobe_family(group_names[li]) != lobe_family(group_names[lj])
            )

            print(f"    BLIND:  {n_blind_surv} survivors "
                  f"(reduction {blind_reduction:.0f}x), "
                  f"truth survives={truth_blind_survives}, "
                  f"combos={n_cross_combos} [{dt_blind:.1f}s]")

            # Intersection
            all_oracle_survivors &= oracle_survivors
            all_blind_survivors &= blind_survivors

            pair_results_list.append({
                'ep1': ep1, 'ep2': ep2,
                'lobe1': pair['lobe1'], 'lobe2': pair['lobe2'],
                'mag1': pair['mag1'], 'mag2': pair['mag2'],
                'ep_sep': pair['ep_sep'],
                'oracle': {
                    'n_survivors': n_oracle_surv,
                    'truth_survives': truth_oracle_survives,
                    'truth_best_dist_deg': truth_oracle_dist,
                    'reduction_factor': float(oracle_reduction),
                    'time_s': dt_oracle,
                },
                'blind': {
                    'n_lobe_combos': n_cross_combos,
                    'n_survivors': n_blind_surv,
                    'truth_survives': truth_blind_survives,
                    'reduction_factor': float(blind_reduction),
                    'time_s': dt_blind,
                },
            })

        dt_stage1 = time.time() - t0_stage1

        # Stage 1 intersection summary
        n_oracle_inter = int(np.sum(all_oracle_survivors))
        n_blind_inter = int(np.sum(all_blind_survivors))
        truth_oracle_inter = bool(all_oracle_survivors[truth_cand_idx])
        truth_blind_inter = bool(all_blind_survivors[truth_cand_idx])
        oracle_inter_red = n_cand / n_oracle_inter if n_oracle_inter > 0 else float('inf')
        blind_inter_red = n_cand / n_blind_inter if n_blind_inter > 0 else float('inf')

        print(f"\n  --- Stage 1 Intersection ({len(pairs)} pairs) ---")
        print(f"    ORACLE: {n_oracle_inter} survivors "
              f"(reduction {oracle_inter_red:.0f}x), "
              f"truth survives={truth_oracle_inter}")
        print(f"    BLIND:  {n_blind_inter} survivors "
              f"(reduction {blind_inter_red:.0f}x), "
              f"truth survives={truth_blind_inter}")
        print(f"    Stage 1 time: {dt_stage1:.1f}s")

        # Save checkpoint
        np.savez(str(stage1_ckpt),
                 oracle_intersection=all_oracle_survivors,
                 blind_intersection=all_blind_survivors,
                 pair_results=np.array(pair_results_list, dtype=object),
                 runtime_s=dt_stage1)
        print(f"  Saved: {stage1_ckpt}")

    # Stage 1 intersection stats (compute from loaded or fresh data)
    n_oracle_inter = int(np.sum(all_oracle_survivors))
    n_blind_inter = int(np.sum(all_blind_survivors))
    truth_oracle_inter = bool(all_oracle_survivors[truth_cand_idx])
    truth_blind_inter = bool(all_blind_survivors[truth_cand_idx])
    oracle_inter_red = n_cand / n_oracle_inter if n_oracle_inter > 0 else float('inf')
    blind_inter_red = n_cand / n_blind_inter if n_blind_inter > 0 else float('inf')

    # ==================================================================
    # Stage 2: Full-observation scoring
    # ==================================================================
    stage2_ckpt = seed_dir / "stage2_results.npz"
    if stage2_ckpt.exists():
        print(f"\n  Stage 2: Loading checkpoint {stage2_ckpt}")
        ckpt = np.load(str(stage2_ckpt), allow_pickle=True)
        oracle_scores = ckpt['oracle_scores']
        oracle_surv_indices = ckpt['oracle_surv_indices']
        blind_scores = ckpt['blind_scores']
        blind_surv_indices = ckpt['blind_surv_indices']
        dt_stage2 = float(ckpt.get('runtime_s', 0))
        print(f"  Stage 2: Loaded oracle={len(oracle_scores)}, "
              f"blind={len(blind_scores)} scores")
    else:
        print(f"\n  Stage 2: Full-observation scoring")
        t0_stage2 = time.time()

        # Build bright peak arrays for Stage 2
        bright_epoch_tidxs = np.array([epoch_to_tidx[int(ep)] for ep in bright_eps])
        pab_at_peaks = np.array([pab_j2000[int(ep)] for ep in bright_eps])
        lobe_normals_at_peaks = np.array([unique_normals[int(g)] for g in bright_lobes])

        # --- Oracle scoring ---
        oracle_surv_indices = np.where(all_oracle_survivors)[0]
        print(f"\n  Stage 2 Oracle: scoring {len(oracle_surv_indices)} survivors "
              f"against {n_bright} bright peaks...")
        t0_s2o = time.time()

        if len(oracle_surv_indices) > 0:
            oracle_scores, oracle_best_psi = score_full_observation_oracle(
                delta_qs, oracle_surv_indices, bright_epoch_tidxs,
                pab_at_peaks, lobe_normals_at_peaks,
                n_psi=STAGE2_N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
            )
        else:
            oracle_scores = np.array([], dtype=int)
            oracle_best_psi = np.array([], dtype=int)

        dt_s2o = time.time() - t0_s2o

        # Truth in oracle survivors
        truth_in_oracle = truth_cand_idx in oracle_surv_indices
        if truth_in_oracle:
            truth_oracle_pos = int(np.where(oracle_surv_indices == truth_cand_idx)[0][0])
            truth_oracle_score = int(oracle_scores[truth_oracle_pos])
            # Rank: how many survivors have score >= truth's score
            truth_oracle_rank = int(np.sum(oracle_scores >= truth_oracle_score))
        else:
            truth_oracle_pos = -1
            truth_oracle_score = -1
            truth_oracle_rank = -1

        if len(oracle_scores) > 0:
            best_oracle_score = int(np.max(oracle_scores))
            # Best FP = best score excluding truth
            if truth_in_oracle:
                fp_mask = np.ones(len(oracle_scores), dtype=bool)
                fp_mask[truth_oracle_pos] = False
                best_fp_oracle = int(np.max(oracle_scores[fp_mask])) if np.any(fp_mask) else 0
            else:
                best_fp_oracle = best_oracle_score
        else:
            best_oracle_score = 0
            best_fp_oracle = 0

        print(f"    Oracle scoring done in {dt_s2o:.1f}s")
        print(f"    Truth score: {truth_oracle_score}/{n_bright} peaks, "
              f"rank={truth_oracle_rank}/{len(oracle_surv_indices)}")
        print(f"    Best score: {best_oracle_score}, best FP: {best_fp_oracle}")

        # --- Blind scoring ---
        blind_surv_indices = np.where(all_blind_survivors)[0]
        print(f"\n  Stage 2 Blind: scoring {len(blind_surv_indices)} survivors "
              f"against {n_bright} bright peaks...")
        t0_s2b = time.time()

        if len(blind_surv_indices) > 0:
            blind_scores = score_full_observation_blind(
                delta_qs, blind_surv_indices, bright_epoch_tidxs,
                pab_at_peaks, unique_normals, group_names,
                n_psi=STAGE2_N_PSI, thresh_deg=SURVIVAL_THRESH_DEG
            )
        else:
            blind_scores = np.array([], dtype=int)

        dt_s2b = time.time() - t0_s2b

        # Truth in blind survivors
        truth_in_blind = truth_cand_idx in blind_surv_indices
        if truth_in_blind:
            truth_blind_pos = int(np.where(blind_surv_indices == truth_cand_idx)[0][0])
            truth_blind_score = int(blind_scores[truth_blind_pos])
            truth_blind_rank = int(np.sum(blind_scores >= truth_blind_score))
        else:
            truth_blind_pos = -1
            truth_blind_score = -1
            truth_blind_rank = -1

        if len(blind_scores) > 0:
            best_blind_score = int(np.max(blind_scores))
            if truth_in_blind:
                fp_mask_b = np.ones(len(blind_scores), dtype=bool)
                fp_mask_b[truth_blind_pos] = False
                best_fp_blind = int(np.max(blind_scores[fp_mask_b])) if np.any(fp_mask_b) else 0
            else:
                best_fp_blind = best_blind_score
        else:
            best_blind_score = 0
            best_fp_blind = 0

        dt_stage2 = time.time() - t0_stage2
        print(f"    Blind scoring done in {dt_s2b:.1f}s")
        print(f"    Truth score: {truth_blind_score}/{n_bright} peaks, "
              f"rank={truth_blind_rank}/{len(blind_surv_indices)}")
        print(f"    Best score: {best_blind_score}, best FP: {best_fp_blind}")
        print(f"    Stage 2 total time: {dt_stage2:.1f}s")

        # Save checkpoint
        np.savez(str(stage2_ckpt),
                 oracle_scores=oracle_scores,
                 oracle_surv_indices=oracle_surv_indices,
                 oracle_best_psi=oracle_best_psi if len(oracle_surv_indices) > 0 else np.array([]),
                 blind_scores=blind_scores,
                 blind_surv_indices=blind_surv_indices,
                 runtime_s=dt_stage2)
        print(f"  Saved: {stage2_ckpt}")

    # ------------------------------------------------------------------
    # Recompute Stage 2 stats from loaded data
    # ------------------------------------------------------------------
    truth_in_oracle = truth_cand_idx in oracle_surv_indices
    if truth_in_oracle and len(oracle_scores) > 0:
        truth_oracle_pos = int(np.where(oracle_surv_indices == truth_cand_idx)[0][0])
        truth_oracle_score = int(oracle_scores[truth_oracle_pos])
        truth_oracle_rank = int(np.sum(oracle_scores >= truth_oracle_score))
    else:
        truth_oracle_pos = -1
        truth_oracle_score = -1
        truth_oracle_rank = -1

    if len(oracle_scores) > 0:
        best_oracle_score = int(np.max(oracle_scores))
        if truth_in_oracle:
            fp_mask = np.ones(len(oracle_scores), dtype=bool)
            fp_mask[truth_oracle_pos] = False
            best_fp_oracle = int(np.max(oracle_scores[fp_mask])) if np.any(fp_mask) else 0
        else:
            best_fp_oracle = best_oracle_score
    else:
        best_oracle_score = 0
        best_fp_oracle = 0

    truth_in_blind = truth_cand_idx in blind_surv_indices
    if truth_in_blind and len(blind_scores) > 0:
        truth_blind_pos = int(np.where(blind_surv_indices == truth_cand_idx)[0][0])
        truth_blind_score = int(blind_scores[truth_blind_pos])
        truth_blind_rank = int(np.sum(blind_scores >= truth_blind_score))
    else:
        truth_blind_pos = -1
        truth_blind_score = -1
        truth_blind_rank = -1

    if len(blind_scores) > 0:
        best_blind_score = int(np.max(blind_scores))
        if truth_in_blind:
            fp_mask_b = np.ones(len(blind_scores), dtype=bool)
            fp_mask_b[truth_blind_pos] = False
            best_fp_blind = int(np.max(blind_scores[fp_mask_b])) if np.any(fp_mask_b) else 0
        else:
            best_fp_blind = best_blind_score
    else:
        best_blind_score = 0
        best_fp_blind = 0

    # ------------------------------------------------------------------
    # Build result JSON
    # ------------------------------------------------------------------
    dt_seed = time.time() - t0_seed
    print(f"\n  Seed {seed} total: {dt_seed:.1f}s")

    # Oracle score distribution
    oracle_score_dist = {}
    if len(oracle_scores) > 0:
        for s in range(1, n_bright + 1):
            count = int(np.sum(oracle_scores == s))
            if count > 0:
                oracle_score_dist[str(s)] = count

    # Blind score distribution
    blind_score_dist = {}
    if len(blind_scores) > 0:
        for s in range(1, n_bright + 1):
            count = int(np.sum(blind_scores == s))
            if count > 0:
                blind_score_dist[str(s)] = count

    result = {
        'seed': seed,
        'true_omega0_dps': [float(x) for x in np.degrees(true_omega0)],
        'true_omega_mag_dps': float(true_omega_mag_dps),
        'n_candidates': n_cand,
        'n_bright_peaks': n_bright,
        'truth_nearest_dir_deg': float(closest_dir_ang),
        'truth_nearest_mag_pct': float(closest_mag_err_pct),
        'truth_nearest_cand_idx': int(truth_cand_idx),
        'truth_nearest_cand_dist_dps': float(truth_cand_dist_dps),
        'stage1': {
            'n_pairs': len(pairs),
            'pairs': [sanitize_for_json(pr) for pr in pair_results_list],
            'intersection': {
                'oracle': {
                    'n_survivors': n_oracle_inter,
                    'truth_survives': truth_oracle_inter,
                    'reduction_factor': float(oracle_inter_red),
                },
                'blind': {
                    'n_survivors': n_blind_inter,
                    'truth_survives': truth_blind_inter,
                    'reduction_factor': float(blind_inter_red),
                },
            },
            'runtime_s': float(dt_stage1),
        },
        'stage2': {
            'oracle': {
                'n_survivors_scored': len(oracle_surv_indices),
                'truth_survives': truth_in_oracle,
                'truth_score': truth_oracle_score,
                'truth_rank': truth_oracle_rank,
                'best_score': best_oracle_score,
                'best_fp_score': best_fp_oracle,
                'score_distribution': oracle_score_dist,
            },
            'blind': {
                'n_survivors_scored': len(blind_surv_indices),
                'truth_survives': truth_in_blind,
                'truth_score': truth_blind_score,
                'truth_rank': truth_blind_rank,
                'best_score': best_blind_score,
                'best_fp_score': best_fp_blind,
                'score_distribution': blind_score_dist,
            },
            'runtime_s': float(dt_stage2),
        },
        'timing': {
            'propagation_s': float(dt_prop),
            'stage1_s': float(dt_stage1),
            'stage2_s': float(dt_stage2),
            'total_s': float(dt_seed),
        },
    }

    # Save result JSON
    result_path = seed_dir / "result.json"
    save_json_atomic(result_path, result)
    print(f"  Saved: {result_path}")

    sys.stdout = old_stdout
    log_file.close()

    return result


# ======================================================================
# Summary
# ======================================================================

def print_summary(all_results):
    """Print clear summary table."""
    print("\n")
    print("=" * 100)
    print("  MICRO106 -- VECTORIZED PAIRWISE PEAK ALIGNMENT -- SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n  {'Seed':>4s}  {'|w|':>6s}  {'N_cand':>6s}  "
          f"{'S1_ora':>6s}  {'S1_bld':>6s}  "
          f"{'T_S1?':>5s}  "
          f"{'T_score_O':>9s}  {'BestFP_O':>8s}  {'T_rank_O':>8s}  "
          f"{'T_score_B':>9s}  {'BestFP_B':>8s}  {'T_rank_B':>8s}  "
          f"{'Time':>6s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  "
          f"{'-'*5}  "
          f"{'-'*9}  {'-'*8}  {'-'*8}  "
          f"{'-'*9}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}")

    for res in all_results:
        seed = res['seed']
        s1_ora = res['stage1']['intersection']['oracle']['n_survivors']
        s1_bld = res['stage1']['intersection']['blind']['n_survivors']
        t_s1 = 'Y' if res['stage1']['intersection']['oracle']['truth_survives'] else 'N'

        s2o = res['stage2']['oracle']
        s2b = res['stage2']['blind']
        n_peaks = res['n_bright_peaks']

        t_score_o = f"{s2o['truth_score']}/{n_peaks}" if s2o['truth_score'] >= 0 else "N/A"
        t_score_b = f"{s2b['truth_score']}/{n_peaks}" if s2b['truth_score'] >= 0 else "N/A"
        fp_o = str(s2o['best_fp_score'])
        fp_b = str(s2b['best_fp_score'])
        rank_o = f"{s2o['truth_rank']}/{s2o['n_survivors_scored']}" if s2o['truth_rank'] >= 0 else "N/A"
        rank_b = f"{s2b['truth_rank']}/{s2b['n_survivors_scored']}" if s2b['truth_rank'] >= 0 else "N/A"

        total_s = res['timing']['total_s']

        print(f"  {seed:4d}  {res['true_omega_mag_dps']:6.3f}  {res['n_candidates']:6d}  "
              f"{s1_ora:6d}  {s1_bld:6d}  "
              f"{t_s1:>5s}  "
              f"{t_score_o:>9s}  {fp_o:>8s}  {rank_o:>8s}  "
              f"{t_score_b:>9s}  {fp_b:>8s}  {rank_b:>8s}  "
              f"{total_s:6.0f}s")

    print()

    # Detailed per-seed pair table
    for res in all_results:
        seed = res['seed']
        print(f"\n  Seed {seed} — Stage 1 pair detail:")
        print(f"    {'Pair':>4s}  {'Lobes':12s}  {'Sep':>5s}  "
              f"{'Ora surv':>8s}  {'Ora red':>7s}  {'T?':>3s}  "
              f"{'Bld surv':>8s}  {'Bld red':>7s}  {'T?':>3s}")
        if 'pairs' in res['stage1']:
            for i, pp in enumerate(res['stage1']['pairs']):
                lobes = f"{pp['lobe1']}/{pp['lobe2']}"
                o = pp['oracle']
                b = pp['blind']
                print(f"    {i:4d}  {lobes:12s}  {pp['ep_sep']:5d}  "
                      f"{o['n_survivors']:8d}  {o['reduction_factor']:6.0f}x  "
                      f"{'Y' if o['truth_survives'] else 'N':>3s}  "
                      f"{b['n_survivors']:8d}  {b['reduction_factor']:6.0f}x  "
                      f"{'Y' if b['truth_survives'] else 'N':>3s}")

    print(f"\n{'='*100}\n")


# ======================================================================
# Main
# ======================================================================

def main():
    t0_total = time.time()

    # Load trajectory data
    traj_path = DATA_DIR / "m046_trajectories.npz"
    print(f"Loading {traj_path}")
    traj_data = dict(np.load(str(traj_path), allow_pickle=True))
    traj_data['group_names'] = list(traj_data['group_names'])

    all_results = []
    for seed in TARGET_SEEDS:
        res = run_seed(seed, traj_data)
        if res is not None:
            all_results.append(res)

    # Save combined results
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    combined_path = OUT_BASE / "results_combined.json"
    save_json_atomic(combined_path, all_results)
    print(f"\nSaved: {combined_path}")

    # Summary
    print_summary(all_results)

    dt_total = time.time() - t0_total
    print(f"Total runtime: {dt_total:.1f}s")


if __name__ == '__main__':
    main()
