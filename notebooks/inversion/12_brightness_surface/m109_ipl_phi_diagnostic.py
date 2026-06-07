#!/usr/bin/env python3
"""
m109 -- IPL-Based Phi Discrimination Diagnostic.

Tests whether IPL centroid metrics provide phi discrimination that lo-fi MSE
does not, for ATT_FAIL seeds (correct omega, wrong attitude). At the TRUTH
omega, sweeps 36 phi values and scores each with 8 metrics.

Metrics:
  1. Lo-fi MSE (baseline)           -- zero-phase brightness surface vs observed
  2. IPL centroid proximity (unwtd)  -- mean angular dist to nearest centroid
  3. IPL centroid proximity (wtd)    -- weighted by 1/loop_count
  4. IPL membership fraction         -- fraction of epochs within 15 deg
  5. Brightness derivative matching  -- MSE of finite-diff dL/dt
  6. Ensemble: prox * lofi_mse       -- product of normalised metrics 2 and 1
  7. Band analysis (A/B/C)           -- centroid prox for tight/med/loose epochs
  8. Topology stability              -- nearest-centroid identity changes

Seeds: 27, 46, 58 (ATT_FAIL pure), 0, 75 (borderline), 93 (OK control)

Usage:
  python3 notebooks/inversion/12_brightness_surface/m109_ipl_phi_diagnostic.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.brightness_surface import load_satellite, extract_component_data

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
IPL_DIR = RESULTS_DIR / "isoshell_viewer"
OUT_DIR = RESULTS_DIR / "m109_ipl_phi"

SEEDS = [27, 46, 58, 0, 75, 93]
SEED_STATUS = {27: "ATT_FAIL", 46: "ATT_FAIL", 58: "ATT_FAIL",
               0: "ATT_FAIL_border", 75: "ATT_FAIL_border", 93: "OK"}

N_PHI = 36                     # phi offsets: 0, 10, 20, ..., 350 deg
PHI_OFFSETS_DEG = np.arange(N_PHI) * (360.0 / N_PHI)
PHI_OFFSETS_RAD = np.deg2rad(PHI_OFFSETS_DEG)

MEMBERSHIP_THRESHOLD_DEG = 15.0  # for metric 4

# Band boundaries for metric 7 (loop_count thresholds)
BAND_A_MAX = 4    # tight: loop_count <= 4
BAND_B_MAX = 8    # medium: 4 < loop_count <= 8
                   # loose: loop_count > 8


# ── Logging ──────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# ── Brightness Surface Evaluation ────────────────────────────────────
def build_facet_groups():
    """Load satellite and extract facet groups for zero-phase brightness eval.

    Returns list of (normal, area, r_d, r_s, n_phong) tuples.
    """
    satellite = load_satellite()
    components = extract_component_data(satellite)
    facet_groups = []
    for comp in components:
        for g in comp['groups']:
            facet_groups.append((
                np.array(g['normal']),
                g['area'],
                g['r_d'],
                g['r_s'],
                g['n_phong'],
            ))
    return facet_groups


def brightness_surface(d, facet_groups):
    """Evaluate zero-phase lo-fi brightness at body-frame directions.

    Parameters
    ----------
    d : (N, 3) array
        Unit directions in body frame (treated as k1 = k2 = PAB).
    facet_groups : list of (normal, area, r_d, r_s, n_phong)

    Returns
    -------
    flux : (N,) array
        Reflected flux at each direction.
    """
    flux = np.zeros(len(d))
    for n, A, r_d, r_s, n_phong in facet_groups:
        cos_theta = np.clip(d @ n, 0, None)  # (N,)
        mask = cos_theta > 0
        diffuse = ((28 * r_d) / (23 * np.pi) * (1 - r_s)
                   * cos_theta * (1 - (1 - cos_theta / 2)**5)**2)
        specular = ((n_phong + 1) / (8 * np.pi) * r_s
                    * np.power(cos_theta, n_phong))
        flux += A * (diffuse + specular) * mask
    return flux


def flux_to_mag(flux, ref_flux):
    """Convert flux array to magnitude using a reference flux."""
    safe_flux = np.where(flux > 0, flux, 1e-30)
    return -2.5 * np.log10(safe_flux / ref_flux)


# ── Delta-Q Phi Sweep ───────────────────────────────────────────────
def compute_pab_for_phi(phi_offset_rad, delta_R, R0_truth, anchor_normal, pab_j2000):
    """Compute candidate body-frame PAB path for a given phi offset.

    Parameters
    ----------
    phi_offset_rad : float
        Twist angle offset relative to truth.
    delta_R : Rotation (N,)
        Precomputed relative rotations: R_truth(t) * R_truth(0)^{-1}.
    R0_truth : Rotation (scalar)
        Truth rotation at epoch 0.
    anchor_normal : (3,) array
        Body-frame normal for the twist axis.
    pab_j2000 : (N, 3) array
        Inertial PAB at each epoch.

    Returns
    -------
    pab_body : (N, 3) array
        Predicted body-frame PAB at each epoch.
    """
    R_twist = Rotation.from_rotvec(phi_offset_rad * anchor_normal)
    R0_cand = R_twist * R0_truth
    R_cand = delta_R * R0_cand  # broadcasting: (N,) * scalar = (N,)
    pab_body = R_cand.apply(pab_j2000)  # (N, 3)
    return pab_body


# ── Metric Functions ─────────────────────────────────────────────────
def metric_lofi_mse(pred_mag, obs_mag):
    """Metric 1: MSE of zero-phase lo-fi magnitudes vs observed lo-fi."""
    return float(np.mean((pred_mag - obs_mag)**2))


def metric_centroid_proximity_unweighted(pab_body, centroids):
    """Metric 2: Mean angular distance from PAB to nearest centroid.

    Parameters
    ----------
    pab_body : (N, 3)
    centroids : object array of length N, each element is (n_loops, 3) or None.

    Returns
    -------
    float : mean angular distance in degrees (NaN epochs skipped).
    """
    n_epochs = len(pab_body)
    ang_dists = []
    for t in range(n_epochs):
        c = centroids[t]
        if c is None or (hasattr(c, '__len__') and len(c) == 0):
            continue
        c = np.atleast_2d(c)
        dots = np.clip(c @ pab_body[t], -1, 1)
        min_dist = np.rad2deg(np.arccos(np.max(dots)))
        ang_dists.append(min_dist)
    if len(ang_dists) == 0:
        return np.nan
    return float(np.mean(ang_dists))


def metric_centroid_proximity_weighted(pab_body, centroids, loop_counts):
    """Metric 3: Weighted mean angular distance, weight = 1/loop_count.

    Emphasizes epochs with few loops (tighter constraints).
    """
    n_epochs = len(pab_body)
    ang_dists = []
    weights = []
    for t in range(n_epochs):
        c = centroids[t]
        if c is None or (hasattr(c, '__len__') and len(c) == 0):
            continue
        lc = int(loop_counts[t])
        if lc <= 0:
            continue
        c = np.atleast_2d(c)
        dots = np.clip(c @ pab_body[t], -1, 1)
        min_dist = np.rad2deg(np.arccos(np.max(dots)))
        ang_dists.append(min_dist)
        weights.append(1.0 / lc)
    if len(ang_dists) == 0:
        return np.nan
    ang_dists = np.array(ang_dists)
    weights = np.array(weights)
    return float(np.average(ang_dists, weights=weights))


def metric_membership_fraction(pab_body, centroids, threshold_deg=MEMBERSHIP_THRESHOLD_DEG):
    """Metric 4: Fraction of epochs where PAB is within threshold of any centroid."""
    n_epochs = len(pab_body)
    n_valid = 0
    n_member = 0
    for t in range(n_epochs):
        c = centroids[t]
        if c is None or (hasattr(c, '__len__') and len(c) == 0):
            continue
        n_valid += 1
        c = np.atleast_2d(c)
        dots = np.clip(c @ pab_body[t], -1, 1)
        min_dist = np.rad2deg(np.arccos(np.max(dots)))
        if min_dist <= threshold_deg:
            n_member += 1
    if n_valid == 0:
        return np.nan
    return float(n_member / n_valid)


def metric_derivative_mse(pred_mag, obs_mag):
    """Metric 5: MSE of magnitude derivative (finite differences)."""
    dpred = np.diff(pred_mag)
    dobs = np.diff(obs_mag)
    return float(np.mean((dpred - dobs)**2))


def metric_ensemble(lofi_mse, centroid_prox):
    """Metric 6: Product of normalised lo-fi MSE and centroid proximity.

    Normalisation is done at the caller level over all phi values.
    This function just returns the product.
    """
    return lofi_mse * centroid_prox


def metric_centroid_proximity_banded(pab_body, centroids, loop_counts):
    """Metric 7: Centroid proximity separated by loop-count bands.

    Returns
    -------
    dict with keys 'band_A', 'band_B', 'band_C', each a float (mean ang dist).
    """
    n_epochs = len(pab_body)
    bands = {'band_A': [], 'band_B': [], 'band_C': []}
    for t in range(n_epochs):
        c = centroids[t]
        if c is None or (hasattr(c, '__len__') and len(c) == 0):
            continue
        lc = int(loop_counts[t])
        if lc <= 0:
            continue
        c = np.atleast_2d(c)
        dots = np.clip(c @ pab_body[t], -1, 1)
        min_dist = np.rad2deg(np.arccos(np.max(dots)))
        if lc <= BAND_A_MAX:
            bands['band_A'].append(min_dist)
        elif lc <= BAND_B_MAX:
            bands['band_B'].append(min_dist)
        else:
            bands['band_C'].append(min_dist)
    result = {}
    for key, vals in bands.items():
        result[key] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return result


def metric_topology_stability(pab_body, centroids):
    """Metric 8: Count of nearest-centroid identity changes between epochs.

    For consecutive epochs with valid centroids, track the index of the
    nearest centroid. Count transitions where the nearest centroid identity
    changes. Truth phi should produce smooth evolution (low count).
    """
    n_epochs = len(pab_body)
    nearest_ids = []  # (epoch_idx, nearest_centroid_direction)
    for t in range(n_epochs):
        c = centroids[t]
        if c is None or (hasattr(c, '__len__') and len(c) == 0):
            nearest_ids.append(None)
            continue
        c = np.atleast_2d(c)
        dots = c @ pab_body[t]
        idx = np.argmax(dots)
        nearest_ids.append(c[idx].copy())

    # Count identity changes: direction of nearest centroid changes significantly
    changes = 0
    prev_dir = None
    for nid in nearest_ids:
        if nid is None:
            prev_dir = None
            continue
        if prev_dir is not None:
            dot = np.clip(np.dot(prev_dir, nid), -1, 1)
            ang = np.rad2deg(np.arccos(dot))
            if ang > 20.0:  # >20 deg jump = identity change
                changes += 1
        prev_dir = nid
    return changes


# ── Per-Seed Processing ──────────────────────────────────────────────
def process_seed(seed, traj, ipl, facet_groups):
    """Run full phi sweep for one seed, return all metric scores.

    Returns
    -------
    dict with keys:
      'phi_offsets_deg': (N_PHI,)
      'lofi_mse': (N_PHI,)
      'centroid_prox': (N_PHI,)
      'weighted_prox': (N_PHI,)
      'membership': (N_PHI,)
      'deriv_mse': (N_PHI,)
      'ensemble': (N_PHI,)
      'band_A': (N_PHI,)
      'band_B': (N_PHI,)
      'band_C': (N_PHI,)
      'stability': (N_PHI,)
      'anchor_normal': (3,)
      'anchor_epoch': int
    """
    t0 = time.time()
    prefix = f"s{seed:03d}_"

    # ── Load truth data ──────────────────────────────────────────────
    q_truth = traj['quaternions'][seed]       # (500, 4) wxyz
    pab_j2000 = traj['pab_j2000']            # (500, 3)
    pab_body_truth = traj['pab_body'][seed]   # (500, 3)
    mag_lofi_obs = traj['mag_lofi'][seed]     # (500,)
    mag_hifi = traj['mag_hifi'][seed]         # (500,)
    unique_normals = traj['unique_normals']   # (10, 3)
    n_epochs = len(q_truth)

    # ── Load IPL data ────────────────────────────────────────────────
    centroids = ipl[prefix + 'centroids']     # (500,) object
    loop_counts = ipl[prefix + 'loop_counts'] # (500,)
    ang_dists = ipl[prefix + 'ang_dists']     # (500,)

    # ── Determine anchor (brightest peak) ────────────────────────────
    peak_mask = traj['peak_seeds'] == seed
    seed_peaks = traj['peak_epochs'][peak_mask]
    brightest_peak = seed_peaks[np.argmin(mag_hifi[seed_peaks])]

    pab_at_anchor = pab_body_truth[brightest_peak]
    dots = unique_normals @ pab_at_anchor
    anchor_normal = unique_normals[np.argmax(dots)].copy()

    print(f"  Seed {seed}: anchor_epoch={brightest_peak}, "
          f"anchor_normal={anchor_normal}, "
          f"n_peaks={len(seed_peaks)}")

    # ── Precompute delta_R (independent of phi) ──────────────────────
    # scipy uses xyzw convention; our data is wxyz
    R_truth = Rotation.from_quat(q_truth[:, [1, 2, 3, 0]])  # (500,)
    R0_truth = R_truth[0]
    R0_inv = R0_truth.inv()
    delta_R = R_truth * R0_inv  # (500,)

    # ── CRITICAL CHECK: phi_offset=0 reproduces truth PAB ────────────
    pab_check = compute_pab_for_phi(0.0, delta_R, R0_truth, anchor_normal, pab_j2000)
    max_err = np.max(np.linalg.norm(pab_check - pab_body_truth, axis=1))
    print(f"    phi=0 PAB reconstruction error: {max_err:.2e} "
          f"({'OK' if max_err < 1e-10 else 'WARN'})")

    # ── Calibrate flux-to-mag reference ──────────────────────────────
    # Evaluate brightness surface at truth PAB, calibrate so mean matches
    truth_flux = brightness_surface(pab_body_truth, facet_groups)
    # Use median ratio for robust calibration
    valid = (truth_flux > 0) & np.isfinite(mag_lofi_obs)
    if valid.sum() < 10:
        print(f"    WARNING: only {valid.sum()} valid epochs for calibration")
    # ref_flux: the flux that maps to mag=0
    # mag = -2.5 * log10(flux / ref_flux) => ref_flux = flux * 10^(mag/2.5)
    # We want the zero-phase approx to match observed lo-fi ON AVERAGE
    # Compute ref_flux that minimises MSE: just match the median
    obs_flux = 10**(-mag_lofi_obs[valid] / 2.5)  # arbitrary ref=1
    pred_flux_norm = truth_flux[valid]
    scale = np.median(obs_flux / pred_flux_norm)
    ref_flux = 1.0 / scale  # so flux_to_mag(flux * scale, 1) ~ obs mag
    # Simpler: convert to mag and shift
    truth_mag_raw = -2.5 * np.log10(np.where(truth_flux > 0, truth_flux, 1e-30))
    mag_offset = np.median(mag_lofi_obs[valid] - truth_mag_raw[valid])

    # ── Phi sweep ────────────────────────────────────────────────────
    scores_lofi_mse = np.zeros(N_PHI)
    scores_centroid_prox = np.zeros(N_PHI)
    scores_weighted_prox = np.zeros(N_PHI)
    scores_membership = np.zeros(N_PHI)
    scores_deriv_mse = np.zeros(N_PHI)
    scores_band_A = np.full(N_PHI, np.nan)
    scores_band_B = np.full(N_PHI, np.nan)
    scores_band_C = np.full(N_PHI, np.nan)
    scores_stability = np.zeros(N_PHI)

    for pi, phi_rad in enumerate(PHI_OFFSETS_RAD):
        pab_body = compute_pab_for_phi(phi_rad, delta_R, R0_truth,
                                       anchor_normal, pab_j2000)

        # Metric 1: lo-fi MSE (zero-phase approximation)
        flux_pred = brightness_surface(pab_body, facet_groups)
        mag_pred = -2.5 * np.log10(np.where(flux_pred > 0, flux_pred, 1e-30))
        mag_pred += mag_offset  # calibrate to match observed lo-fi
        scores_lofi_mse[pi] = metric_lofi_mse(mag_pred, mag_lofi_obs)

        # Metric 2: centroid proximity (unweighted)
        scores_centroid_prox[pi] = metric_centroid_proximity_unweighted(
            pab_body, centroids)

        # Metric 3: centroid proximity (weighted)
        scores_weighted_prox[pi] = metric_centroid_proximity_weighted(
            pab_body, centroids, loop_counts)

        # Metric 4: membership fraction
        scores_membership[pi] = metric_membership_fraction(pab_body, centroids)

        # Metric 5: derivative MSE
        scores_deriv_mse[pi] = metric_derivative_mse(mag_pred, mag_lofi_obs)

        # Metric 7: banded proximity
        band_scores = metric_centroid_proximity_banded(
            pab_body, centroids, loop_counts)
        scores_band_A[pi] = band_scores['band_A']
        scores_band_B[pi] = band_scores['band_B']
        scores_band_C[pi] = band_scores['band_C']

        # Metric 8: topology stability
        scores_stability[pi] = metric_topology_stability(pab_body, centroids)

    # Metric 6: ensemble (normalised product)
    # Normalise each component to [0, 1] range before multiplying
    def normalise(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    scores_ensemble = normalise(scores_lofi_mse) * normalise(scores_centroid_prox)

    elapsed = time.time() - t0
    print(f"    Phi sweep done in {elapsed:.1f}s")

    return {
        'phi_offsets_deg': PHI_OFFSETS_DEG,
        'lofi_mse': scores_lofi_mse,
        'centroid_prox': scores_centroid_prox,
        'weighted_prox': scores_weighted_prox,
        'membership': scores_membership,
        'deriv_mse': scores_deriv_mse,
        'ensemble': scores_ensemble,
        'band_A': scores_band_A,
        'band_B': scores_band_B,
        'band_C': scores_band_C,
        'stability': scores_stability,
        'anchor_normal': anchor_normal,
        'anchor_epoch': int(brightest_peak),
    }


# ── Discrimination Ratio ─────────────────────────────────────────────
def discrimination_ratio(scores, truth_idx=0, lower_is_better=True):
    """Compute discrimination ratio.

    For lower-is-better metrics: best_wrong / truth (>1 = truth wins).
    For higher-is-better metrics: truth / best_wrong (>1 = truth wins).
    """
    truth_score = scores[truth_idx]
    wrong_scores = np.concatenate([scores[:truth_idx], scores[truth_idx + 1:]])

    if lower_is_better:
        best_wrong = np.nanmin(wrong_scores)
        if truth_score == 0 or np.isnan(truth_score):
            return np.nan
        return best_wrong / truth_score
    else:
        best_wrong = np.nanmax(wrong_scores)
        if best_wrong == 0 or np.isnan(best_wrong):
            return np.nan
        return truth_score / best_wrong


def print_seed_table(seed, results):
    """Print per-phi score table for one seed."""
    print(f"\n{'='*100}")
    print(f"  Seed {seed} ({SEED_STATUS[seed]})  |  "
          f"anchor_epoch={results['anchor_epoch']}, "
          f"anchor_normal={results['anchor_normal']}")
    print(f"{'='*100}")

    header = (f"{'phi':>5s} | {'lo-fi MSE':>10s} | {'cen_prox':>9s} | "
              f"{'wt_prox':>9s} | {'member%':>8s} | {'dL/dt MSE':>10s} | "
              f"{'ensemble':>9s} | {'band_A':>8s} | {'band_B':>8s} | "
              f"{'band_C':>8s} | {'stab':>5s}")
    print(header)
    print("-" * len(header))

    for pi in range(N_PHI):
        marker = " ***" if pi == 0 else ""
        row = (f"{results['phi_offsets_deg'][pi]:5.0f} | "
               f"{results['lofi_mse'][pi]:10.6f} | "
               f"{results['centroid_prox'][pi]:9.4f} | "
               f"{results['weighted_prox'][pi]:9.4f} | "
               f"{results['membership'][pi]:8.4f} | "
               f"{results['deriv_mse'][pi]:10.6f} | "
               f"{results['ensemble'][pi]:9.6f} | "
               f"{results['band_A'][pi]:8.4f} | "
               f"{results['band_B'][pi]:8.4f} | "
               f"{results['band_C'][pi]:8.4f} | "
               f"{results['stability'][pi]:5.0f}{marker}")
        print(row)

    # Discrimination ratios
    print()
    disc = {}
    disc['lofi_mse'] = discrimination_ratio(results['lofi_mse'], 0, lower_is_better=True)
    disc['centroid_prox'] = discrimination_ratio(results['centroid_prox'], 0, lower_is_better=True)
    disc['weighted_prox'] = discrimination_ratio(results['weighted_prox'], 0, lower_is_better=True)
    disc['membership'] = discrimination_ratio(results['membership'], 0, lower_is_better=False)
    disc['deriv_mse'] = discrimination_ratio(results['deriv_mse'], 0, lower_is_better=True)
    disc['ensemble'] = discrimination_ratio(results['ensemble'], 0, lower_is_better=True)
    disc['band_A'] = discrimination_ratio(results['band_A'], 0, lower_is_better=True)
    disc['band_B'] = discrimination_ratio(results['band_B'], 0, lower_is_better=True)
    disc['band_C'] = discrimination_ratio(results['band_C'], 0, lower_is_better=True)
    disc['stability'] = discrimination_ratio(results['stability'], 0, lower_is_better=True)

    print("  Discrimination ratios (>1 = truth phi is best):")
    for name, val in disc.items():
        flag = "OK" if val > 1.0 else "FAIL" if val < 1.0 else "---"
        print(f"    {name:>15s}: {val:7.3f}  [{flag}]")

    return disc


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(str(OUT_DIR / "diagnostic.log"), "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 60)
    print("m109 -- IPL-Based Phi Discrimination Diagnostic")
    print(f"  Seeds: {SEEDS}")
    print(f"  N_PHI: {N_PHI} (step = {360/N_PHI:.0f} deg)")
    print(f"  Membership threshold: {MEMBERSHIP_THRESHOLD_DEG} deg")
    print("=" * 60)
    t_global = time.time()

    # ── Load trajectory data ─────────────────────────────────────────
    print("\nLoading trajectory data...")
    traj = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    print(f"  Loaded: {traj['n_trajectories']} trajectories, "
          f"{traj['n_obs']} epochs each")

    # ── Load IPL data ────────────────────────────────────────────────
    print("Loading IPL data...")
    ipl = np.load(str(IPL_DIR / "ipl_all_epochs.npz"), allow_pickle=True)
    print(f"  Loaded: {len(ipl.files)} keys")

    # ── Load satellite model for brightness surface ──────────────────
    print("Loading satellite model for brightness surface...")
    facet_groups = build_facet_groups()
    print(f"  {len(facet_groups)} facet groups loaded")

    # ── Process each seed ────────────────────────────────────────────
    all_results = {}
    all_disc = {}

    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"# Processing seed {seed} ({SEED_STATUS[seed]})")
        print(f"{'#'*60}")
        results = process_seed(seed, traj, ipl, facet_groups)
        all_results[seed] = results
        disc = print_seed_table(seed, results)
        all_disc[seed] = disc

    # ── Save all results ─────────────────────────────────────────────
    save_dict = {}
    save_dict['seeds'] = np.array(SEEDS)
    save_dict['phi_offsets_deg'] = PHI_OFFSETS_DEG
    for seed in SEEDS:
        sp = f"s{seed:03d}_"
        r = all_results[seed]
        save_dict[sp + 'lofi_mse'] = r['lofi_mse']
        save_dict[sp + 'centroid_prox'] = r['centroid_prox']
        save_dict[sp + 'weighted_prox'] = r['weighted_prox']
        save_dict[sp + 'membership'] = r['membership']
        save_dict[sp + 'deriv_mse'] = r['deriv_mse']
        save_dict[sp + 'ensemble'] = r['ensemble']
        save_dict[sp + 'band_A'] = r['band_A']
        save_dict[sp + 'band_B'] = r['band_B']
        save_dict[sp + 'band_C'] = r['band_C']
        save_dict[sp + 'stability'] = r['stability']
        save_dict[sp + 'anchor_normal'] = r['anchor_normal']
        save_dict[sp + 'anchor_epoch'] = np.array(r['anchor_epoch'])

    npz_path = OUT_DIR / "diagnostic.npz"
    np.savez(str(npz_path), **save_dict)
    print(f"\nSaved: {npz_path}")

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n\n{'='*130}")
    print("SUMMARY: Discrimination Ratios (>1 = truth phi correctly identified)")
    print(f"{'='*130}")
    metric_names = ['lofi_mse', 'centroid_prox', 'weighted_prox', 'membership',
                    'deriv_mse', 'ensemble', 'band_A', 'band_B', 'band_C',
                    'stability']
    header = f"{'Seed':>6s} | {'Status':>15s}"
    for mn in metric_names:
        header += f" | {mn:>13s}"
    print(header)
    print("-" * len(header))

    for seed in SEEDS:
        row = f"{seed:6d} | {SEED_STATUS[seed]:>15s}"
        for mn in metric_names:
            val = all_disc[seed].get(mn, np.nan)
            if np.isnan(val):
                row += f" | {'nan':>13s}"
            else:
                flag = "*" if val > 1.0 else " "
                row += f" | {val:12.3f}{flag}"
        print(row)

    # ── Count wins per metric ────────────────────────────────────────
    print()
    att_fail_seeds = [s for s in SEEDS if 'ATT_FAIL' in SEED_STATUS[s]]
    print(f"ATT_FAIL seeds: {att_fail_seeds}")
    print("\nMetric win counts (disc > 1) across ATT_FAIL seeds:")
    for mn in metric_names:
        wins = sum(1 for s in att_fail_seeds
                   if all_disc[s].get(mn, 0) > 1.0)
        total = len(att_fail_seeds)
        bar = "#" * wins + "." * (total - wins)
        print(f"  {mn:>15s}: {wins}/{total}  [{bar}]")

    # ── VERDICT ──────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    # Find metric(s) with highest ATT_FAIL win rate
    best_rate = 0
    best_metrics = []
    for mn in metric_names:
        wins = sum(1 for s in att_fail_seeds
                   if all_disc[s].get(mn, 0) > 1.0)
        rate = wins / len(att_fail_seeds)
        if rate > best_rate:
            best_rate = rate
            best_metrics = [mn]
        elif rate == best_rate:
            best_metrics.append(mn)

    if best_rate > 0.5:
        print(f"Best metric(s) for ATT_FAIL phi discrimination: "
              f"{', '.join(best_metrics)} ({best_rate*100:.0f}% win rate)")
        # Report average discrimination ratio for best metrics
        for mn in best_metrics:
            ratios = [all_disc[s][mn] for s in att_fail_seeds
                      if not np.isnan(all_disc[s].get(mn, np.nan))]
            if ratios:
                print(f"  {mn}: mean disc = {np.mean(ratios):.3f}, "
                      f"min disc = {np.min(ratios):.3f}")
    else:
        print("NO metric reliably discriminates truth phi for ATT_FAIL seeds.")
        print("  Best win rate across ATT_FAIL seeds: "
              f"{best_rate*100:.0f}% ({', '.join(best_metrics)})")
        # Show which metric came closest
        print("\n  Per-metric mean discrimination ratio (ATT_FAIL seeds):")
        for mn in metric_names:
            ratios = [all_disc[s][mn] for s in att_fail_seeds
                      if not np.isnan(all_disc[s].get(mn, np.nan))]
            if ratios:
                print(f"    {mn:>15s}: {np.mean(ratios):.3f}")

    elapsed = time.time() - t_global
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Saved: {npz_path}")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == '__main__':
    main()
