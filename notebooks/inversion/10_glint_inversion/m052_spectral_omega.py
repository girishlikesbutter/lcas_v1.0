#!/usr/bin/env python3
"""Micro-52 -- Spectral omega estimation from LC frequency content.

The frequency spectrum of the LC depends on omega direction relative to
principal axes but is INVARIANT to q0.  This decouples omega from attitude.

Part A: For 20 trajectories, compute Lomb-Scargle periodogram of the hi-fi LC.
        Extract dominant frequencies.  Correlate with |omega| and omega direction.

Part B: For each candidate omega direction, predict the expected frequency
        content by simulating a short trajectory (any q0).  Match against
        observed spectrum.  Does the correct omega direction score best?

Part C: Build a practical omega estimator: peak count → |omega|, then
        spectral matching → omega direction.  Test on 20 trajectories.
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
from scipy.signal import find_peaks, lombscargle
from scipy.stats import spearmanr

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"


def omega_direction_error(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def fibonacci_sphere(n_points):
    pts = np.zeros((n_points, 3))
    gr = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
        phi = 2 * np.pi * i / gr
        pts[i] = [np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)]
    return pts


def compute_lomb_scargle(times, mags, freq_range=None, n_freqs=500):
    """Compute Lomb-Scargle periodogram of a light curve.

    Parameters
    ----------
    times : (N,) observation times in seconds
    mags : (N,) apparent magnitudes
    freq_range : (f_min, f_max) in Hz, or None for auto
    n_freqs : number of frequency points

    Returns
    -------
    freqs : (n_freqs,) frequency array in Hz
    power : (n_freqs,) normalised power
    """
    # Work with brightness (flux proxy), not magnitude
    # Lower mag = brighter, so negate
    signal = -mags
    signal = signal - np.mean(signal)

    dt = np.median(np.diff(times))
    if freq_range is None:
        f_min = 1.0 / (times[-1] - times[0])  # lowest resolvable
        f_max = 0.5 / dt  # Nyquist
    else:
        f_min, f_max = freq_range

    freqs = np.linspace(f_min, f_max, n_freqs)
    angular_freqs = 2 * np.pi * freqs
    power = lombscargle(times, signal, angular_freqs, normalize=True)

    return freqs, power


def extract_spectral_features(freqs, power, n_peaks=5):
    """Extract dominant frequencies and their relative powers."""
    # Find peaks in the periodogram
    peak_idx, peak_props = find_peaks(power, height=0.01, distance=5,
                                       prominence=0.01)
    if len(peak_idx) == 0:
        return {'dominant_freq': 0.0, 'dominant_power': 0.0,
                'freq_ratio_12': 0.0, 'n_sig_peaks': 0,
                'peak_freqs': [], 'peak_powers': []}

    # Sort by power
    sorted_idx = peak_idx[np.argsort(power[peak_idx])[::-1]]
    top_idx = sorted_idx[:n_peaks]

    peak_freqs = freqs[top_idx]
    peak_powers = power[top_idx]

    features = {
        'dominant_freq': float(peak_freqs[0]),
        'dominant_power': float(peak_powers[0]),
        'n_sig_peaks': len(peak_idx),
        'peak_freqs': peak_freqs.tolist(),
        'peak_powers': peak_powers.tolist(),
    }

    if len(peak_freqs) >= 2:
        features['freq_ratio_12'] = float(peak_freqs[1] / peak_freqs[0])
        features['second_freq'] = float(peak_freqs[1])
        features['power_ratio_12'] = float(peak_powers[1] / peak_powers[0])
    else:
        features['freq_ratio_12'] = 0.0

    return features


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m052 -- Spectral omega estimation")
print("=" * 70)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
n_traj = int(master['n_trajectories'])

# Principal axes of inertia
eigvals, eigvecs = np.linalg.eigh(I_tensor)
print(f"Principal moments: {eigvals}")
print(f"Asymmetry parameter: {(eigvals[1]-eigvals[0])/(eigvals[2]-eigvals[0]):.3f}")


# ===========================================================================
# Part A: Spectral features vs omega properties (all 100 trajectories)
# ===========================================================================
print("\n" + "=" * 70)
print("PART A: Spectral features across 100 trajectories")
print("=" * 70)

all_features = []
for traj_idx in range(n_traj):
    mags = mag_hifi[traj_idx]
    freqs, power = compute_lomb_scargle(obs_times, mags)
    features = extract_spectral_features(freqs, power)

    omega0 = omega0s[traj_idx]
    omega_mag = float(omega_mags_arr[traj_idx])

    # Omega direction relative to principal axes
    omega_dir = omega0 / np.linalg.norm(omega0)
    # Angle from each principal axis
    angles_to_pa = [float(np.rad2deg(np.arccos(np.clip(abs(np.dot(omega_dir, eigvecs[:, i])), 0, 1))))
                    for i in range(3)]

    # Expected tumbling frequency: |omega| / 360 Hz (one rev per 360/|omega| seconds)
    expected_freq = omega_mag / 360.0  # in Hz (omega_mag in deg/s)

    features['traj_idx'] = traj_idx
    features['omega_mag_dps'] = omega_mag
    features['omega_dir'] = omega_dir.tolist()
    features['angles_to_pa'] = angles_to_pa
    features['expected_tumble_freq'] = expected_freq

    # Count peaks from LC
    peaks_idx, _ = find_peaks(-mags, distance=5, prominence=0.3)
    features['n_lc_peaks'] = len(peaks_idx)
    features['n_bright_peaks'] = int(np.sum(mags[peaks_idx] < 9.0))

    all_features.append(features)

# Correlations
omega_mags_all = np.array([f['omega_mag_dps'] for f in all_features])
dom_freqs = np.array([f['dominant_freq'] for f in all_features])
expected_freqs = np.array([f['expected_tumble_freq'] for f in all_features])
n_peaks_all = np.array([f['n_lc_peaks'] for f in all_features])
n_bright_all = np.array([f['n_bright_peaks'] for f in all_features])

rho_domfreq, p_domfreq = spearmanr(omega_mags_all, dom_freqs)
rho_peaks, p_peaks = spearmanr(omega_mags_all, n_peaks_all)

print(f"\nCorrelations with |omega|:")
print(f"  Dominant LS frequency: rho={rho_domfreq:.3f}, p={p_domfreq:.4f}")
print(f"  Number of LC peaks:    rho={rho_peaks:.3f}, p={p_peaks:.4f}")

# Calibrate peak count → |omega|
# Fit: |omega| = a * n_peaks + b
from numpy.polynomial import polynomial as P
coeffs = P.polyfit(n_peaks_all, omega_mags_all, 1)
omega_from_peaks = P.polyval(n_peaks_all, coeffs)
peak_count_error = np.abs(omega_from_peaks - omega_mags_all) / omega_mags_all * 100

print(f"\nPeak count calibration: |omega| = {coeffs[1]:.4f} * n_peaks + {coeffs[0]:.4f}")
print(f"  Median error: {np.median(peak_count_error):.1f}%")
print(f"  Within ±20%: {np.sum(peak_count_error < 20)}/100")
print(f"  Within ±30%: {np.sum(peak_count_error < 30)}/100")

# Also try dominant frequency → |omega|
# Expected: dominant_freq ≈ k * |omega| / 360
valid_mask = dom_freqs > 0
if valid_mask.sum() > 10:
    ratio = omega_mags_all[valid_mask] / (dom_freqs[valid_mask] * 360)
    print(f"\n|omega| / (f_dom * 360): median={np.median(ratio):.2f}, "
          f"std={np.std(ratio):.2f}")
    k_fit = np.median(ratio)
    omega_from_freq = dom_freqs * 360 * k_fit
    freq_error = np.abs(omega_from_freq - omega_mags_all) / omega_mags_all * 100
    freq_error = freq_error[valid_mask]
    print(f"  Freq-based calibration: median error={np.median(freq_error):.1f}%")
    print(f"  Within ±20%: {np.sum(freq_error < 20)}/{valid_mask.sum()}")
    print(f"  Within ±30%: {np.sum(freq_error < 30)}/{valid_mask.sum()}")


# ===========================================================================
# Part B: Spectral matching for omega direction
# ===========================================================================
print("\n" + "=" * 70)
print("PART B: Spectral matching for omega direction (10 trajectories)")
print("=" * 70)

# Select 10 test trajectories
omega_sorted = np.argsort(omega_mags_arr)
candidates = [idx for idx in omega_sorted if all_features[idx]['n_bright_peaks'] >= 3]
sel = np.linspace(0, len(candidates) - 1, 10, dtype=int)
TEST_B = [candidates[i] for i in sel]

N_CANDIDATE_DIRS = 200
candidate_dirs = fibonacci_sphere(N_CANDIDATE_DIRS)

# For spectral matching, generate a synthetic LC for each candidate omega
# using a random q0, then compare power spectra.
# Key insight: power spectrum is q0-invariant, so one random q0 suffices.

part_b_results = []

for traj_idx in TEST_B:
    t0 = time.time()
    mags_obs = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    # Observed spectrum
    freqs_obs, power_obs = compute_lomb_scargle(obs_times, mags_obs)
    features_obs = extract_spectral_features(freqs_obs, power_obs)

    # Estimate |omega| from peak count
    n_peaks = all_features[traj_idx]['n_lc_peaks']
    omega_mag_est = float(P.polyval(n_peaks, coeffs))
    mag_est_err = abs(omega_mag_est - omega_mag_true) / omega_mag_true * 100

    print(f"\n  Traj {traj_idx} (|omega|={omega_mag_true:.3f} deg/s, "
          f"est={omega_mag_est:.3f}, err={mag_est_err:.0f}%)")
    print(f"    Observed dominant freq: {features_obs['dominant_freq']*1000:.3f} mHz, "
          f"n_sig_peaks={features_obs['n_sig_peaks']}")

    # For each candidate direction, simulate LC with random q0 and estimated |omega|
    rng = np.random.RandomState(42)
    q0_random = Rotation.random(random_state=rng).as_quat()  # xyzw
    q0_random_wxyz = np.array([q0_random[3], q0_random[0],
                                q0_random[1], q0_random[2]])

    # Also try with true |omega| for comparison
    spectral_scores = np.zeros(N_CANDIDATE_DIRS)
    spectral_scores_true_mag = np.zeros(N_CANDIDATE_DIRS)

    for i, d in enumerate(candidate_dirs):
        # With estimated magnitude
        omega_test = np.deg2rad(omega_mag_est) * d
        try:
            quats_test, _ = propagate_attitude(
                q0_random_wxyz, omega_test, obs_times, "tumbling", I_tensor)

            # Compute synthetic "alignment curve" as LC proxy
            # For each epoch: max alignment of any normal with PAB
            max_align = np.zeros(len(obs_times))
            for t in range(len(obs_times)):
                q = quats_test[t]
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                bd = -1.0
                for j in range(len(unique_normals)):
                    dv = np.dot(R.T @ unique_normals[j], pab_j2000[t])
                    if dv > bd:
                        bd = dv
                max_align[t] = bd

            # Compute spectrum of alignment curve
            freqs_test, power_test = compute_lomb_scargle(obs_times, -max_align)

            # Spectral similarity: cross-correlation of power spectra
            # Normalise both
            p_obs_norm = power_obs / (np.max(power_obs) + 1e-10)
            p_test_norm = power_test / (np.max(power_test) + 1e-10)
            spectral_scores[i] = -np.sum(p_obs_norm * p_test_norm)  # negative = better
        except Exception:
            spectral_scores[i] = 0

        # With true magnitude
        omega_test2 = np.deg2rad(omega_mag_true) * d
        try:
            quats_test2, _ = propagate_attitude(
                q0_random_wxyz, omega_test2, obs_times, "tumbling", I_tensor)
            max_align2 = np.zeros(len(obs_times))
            for t in range(len(obs_times)):
                q = quats_test2[t]
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                bd = -1.0
                for j in range(len(unique_normals)):
                    dv = np.dot(R.T @ unique_normals[j], pab_j2000[t])
                    if dv > bd:
                        bd = dv
                max_align2[t] = bd
            _, power_test2 = compute_lomb_scargle(obs_times, -max_align2)
            p_test2_norm = power_test2 / (np.max(power_test2) + 1e-10)
            spectral_scores_true_mag[i] = -np.sum(p_obs_norm * p_test2_norm)
        except Exception:
            spectral_scores_true_mag[i] = 0

    # Find best direction
    best_idx = np.argmin(spectral_scores)
    best_dir = candidate_dirs[best_idx]
    best_omega = np.deg2rad(omega_mag_est) * best_dir
    dir_err = omega_direction_error(best_omega, omega_true)

    best_idx_tm = np.argmin(spectral_scores_true_mag)
    best_dir_tm = candidate_dirs[best_idx_tm]
    dir_err_tm = omega_direction_error(
        np.deg2rad(omega_mag_true) * best_dir_tm, omega_true)

    # Top 5
    top5_idx = np.argsort(spectral_scores)[:5]
    top5_errs = [omega_direction_error(np.deg2rad(omega_mag_est) * candidate_dirs[i],
                                        omega_true) for i in top5_idx]

    top5_idx_tm = np.argsort(spectral_scores_true_mag)[:5]
    top5_errs_tm = [omega_direction_error(np.deg2rad(omega_mag_true) * candidate_dirs[i],
                                           omega_true) for i in top5_idx_tm]

    dt = time.time() - t0
    print(f"    Est mag: best dir_err={dir_err:.1f} deg, "
          f"top5={[f'{e:.0f}' for e in top5_errs]}")
    print(f"    True mag: best dir_err={dir_err_tm:.1f} deg, "
          f"top5={[f'{e:.0f}' for e in top5_errs_tm]}")
    print(f"    Time: {dt:.1f}s")

    part_b_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est_dps': omega_mag_est,
        'mag_est_err_pct': mag_est_err,
        'dir_err_est_mag': dir_err,
        'dir_err_true_mag': dir_err_tm,
        'top5_errs_est': top5_errs,
        'top5_errs_true': top5_errs_tm,
        'runtime_s': dt,
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nPart A: Omega magnitude estimation")
print(f"  Peak count calibration: median error {np.median(peak_count_error):.1f}%, "
      f"within ±30%: {np.sum(peak_count_error < 30)}/100")

print(f"\nPart B: Spectral direction matching (200 candidate dirs)")
if part_b_results:
    dir_errs_est = [r['dir_err_est_mag'] for r in part_b_results]
    dir_errs_true = [r['dir_err_true_mag'] for r in part_b_results]
    print(f"  With estimated |omega|: median best dir_err={np.median(dir_errs_est):.1f} deg")
    print(f"    < 20 deg: {sum(1 for e in dir_errs_est if e < 20)}/10")
    print(f"    < 45 deg: {sum(1 for e in dir_errs_est if e < 45)}/10")
    print(f"  With true |omega|: median best dir_err={np.median(dir_errs_true):.1f} deg")
    print(f"    < 20 deg: {sum(1 for e in dir_errs_true if e < 20)}/10")
    print(f"    < 45 deg: {sum(1 for e in dir_errs_true if e < 45)}/10")


# ===========================================================================
# Plots
# ===========================================================================
print("\n--- Generating plots ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Micro-52: Spectral Omega Estimation", fontsize=14, fontweight='bold')

# Panel 1: Peak count vs |omega|
ax = axes[0, 0]
ax.scatter(n_peaks_all, omega_mags_all, s=20, alpha=0.5, c='steelblue')
x_fit = np.linspace(0, 40, 100)
ax.plot(x_fit, P.polyval(x_fit, coeffs), 'r-', label='Linear fit')
ax.set_xlabel('Number of LC peaks')
ax.set_ylabel('|omega| (deg/s)')
ax.set_title(f'Peak count → |omega| (rho={rho_peaks:.3f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Peak count magnitude error
ax = axes[0, 1]
ax.hist(peak_count_error, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(20, color='green', linestyle='--', label='20%')
ax.axvline(30, color='orange', linestyle='--', label='30%')
ax.axvline(np.median(peak_count_error), color='red', linestyle='--',
           label=f'Median={np.median(peak_count_error):.0f}%')
ax.set_xlabel('|omega| error (%)')
ax.set_ylabel('Count')
ax.set_title('Peak count magnitude error')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 3: Dominant LS frequency vs expected
ax = axes[0, 2]
ax.scatter(expected_freqs * 1000, dom_freqs * 1000, s=20, alpha=0.5, c='steelblue')
ax.set_xlabel('Expected tumble freq (mHz)')
ax.set_ylabel('Dominant LS freq (mHz)')
ax.set_title(f'LS freq vs expected (rho={rho_domfreq:.3f})')
lims = [0, max(max(expected_freqs), max(dom_freqs)) * 1000 * 1.1]
ax.plot(lims, lims, 'r--', alpha=0.5)
ax.grid(True, alpha=0.3)

# Panel 4: Spectral direction errors (est mag)
ax = axes[1, 0]
if part_b_results:
    omegas = [r['omega_dps'] for r in part_b_results]
    errs = [r['dir_err_est_mag'] for r in part_b_results]
    ax.scatter(omegas, errs, s=80, c='steelblue', edgecolors='black', zorder=3)
    ax.axhline(20, color='green', linestyle='--', alpha=0.5, label='20 deg')
    ax.axhline(45, color='orange', linestyle='--', alpha=0.5, label='45 deg')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best direction error (deg)')
ax.set_title('Spectral matching (est |omega|)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: Spectral direction errors (true mag)
ax = axes[1, 1]
if part_b_results:
    errs_tm = [r['dir_err_true_mag'] for r in part_b_results]
    ax.scatter(omegas, errs_tm, s=80, c='orange', edgecolors='black', zorder=3)
    ax.axhline(20, color='green', linestyle='--', alpha=0.5, label='20 deg')
    ax.axhline(45, color='orange', linestyle='--', alpha=0.5, label='45 deg')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best direction error (deg)')
ax.set_title('Spectral matching (true |omega|)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 6: Example periodogram
ax = axes[1, 2]
example_idx = TEST_B[len(TEST_B)//2]
freqs_ex, power_ex = compute_lomb_scargle(obs_times, mag_hifi[example_idx])
ax.plot(freqs_ex * 1000, power_ex, 'b-', linewidth=0.8)
exp_f = omega_mags_arr[example_idx] / 360.0
ax.axvline(exp_f * 1000, color='red', linestyle='--',
           label=f'Expected: {exp_f*1000:.2f} mHz')
ax.set_xlabel('Frequency (mHz)')
ax.set_ylabel('LS Power')
ax.set_title(f'Example periodogram (traj {example_idx})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m052_spectral_omega.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot: {plot_path}")

# Save
results = {
    'experiment': 'm052_spectral_omega',
    'part_a': {
        'peak_count_calibration': {'coeffs': coeffs.tolist(),
                                    'median_error_pct': float(np.median(peak_count_error)),
                                    'within_20pct': int(np.sum(peak_count_error < 20)),
                                    'within_30pct': int(np.sum(peak_count_error < 30))},
        'dom_freq_correlation': {'rho': float(rho_domfreq), 'p': float(p_domfreq)},
        'peak_count_correlation': {'rho': float(rho_peaks), 'p': float(p_peaks)},
    },
    'part_b': part_b_results,
    'total_time_s': time.time() - t_global,
}
json_path = RESULTS_DIR / "m052_spectral_omega.json"
with open(str(json_path), 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)
print(f"JSON: {json_path}")
print(f"\nTotal: {time.time() - t_global:.0f}s")
