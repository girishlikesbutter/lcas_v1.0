#!/usr/bin/env python3
"""
m066 — Test: do brightness peaks coincide with normal-alignment minima?

Hypothesis: at every observed brightness peak, at least one normal group
should be at a local minimum in its angular distance to the PAB direction.
This is a necessary condition: a brightness peak means reflected light is
momentarily maximized, which requires some face sweeping through closest
alignment with the sun-observer bisector (PAB).

If this holds at truth and fails at wrong candidates, it's a constraint
that can reject wrong (q0, omega) hypotheses without computing brightness.

Method:
  For each trajectory (known q0, omega):
    1. Propagate attitude → R(t) at 500 epochs
    2. For each of 10 normal groups: compute ang_dist(R(t) @ n_g, PAB(t))
    3. Find local minima in each group's angular distance curve
    4. Find observed brightness peaks
    5. At each peak: check if any group has a local minimum within ±K epochs
    6. Count "explained" vs "unexplained" peaks

  Then perturb omega direction by [5, 10, 20, 45, 90] degrees:
    - Repeat the analysis with wrong attitude trajectory
    - Count how many peaks become "unexplained" (violations)

This tells us the discriminating power of the alignment-minimum constraint.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, argrelmin

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"


# ── Helpers ─────────────────────────────────────────────────────────────

def compute_alignment_curves(quaternions, unique_normals, pab_j2000):
    """Compute angular distance between each normal and PAB at each epoch.

    Returns: (n_normals, n_epochs) array of angular distances in degrees.
    Small values = face is near-aligned with PAB = potential glint.
    """
    n_epochs = len(quaternions)
    n_normals = len(unique_normals)
    ang_dist = np.zeros((n_normals, n_epochs))

    for t in range(n_epochs):
        q = quaternions[t]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        for g in range(n_normals):
            n_inertial = R @ unique_normals[g]
            cos_angle = np.clip(np.dot(n_inertial, pab_j2000[t]), -1, 1)
            ang_dist[g, t] = np.rad2deg(np.arccos(cos_angle))

    return ang_dist


def find_alignment_minima(ang_dist_curve, order=3):
    """Find epochs where angular distance is at a local minimum.

    order=3 means each minimum must be lower than its 3 neighbors on each side.
    """
    minima = argrelmin(ang_dist_curve, order=order)[0]
    return minima


def check_peak_alignment(brightness_peaks, alignment_minima_per_group,
                         tolerance_epochs=3):
    """For each brightness peak, check if any group has an alignment
    minimum within ±tolerance_epochs.

    Returns: (n_explained, n_total, per_peak_details)
    """
    n_explained = 0
    details = []

    for peak_ep in brightness_peaks:
        explained = False
        explaining_groups = []
        for g, minima in enumerate(alignment_minima_per_group):
            if len(minima) == 0:
                continue
            distances = np.abs(minima - peak_ep)
            closest = distances.min()
            if closest <= tolerance_epochs:
                explained = True
                closest_min = minima[distances.argmin()]
                explaining_groups.append((g, int(closest_min), int(closest)))

        if explained:
            n_explained += 1
        details.append({
            'peak_epoch': int(peak_ep),
            'explained': explained,
            'groups': explaining_groups,
        })

    return n_explained, len(brightness_peaks), details


def random_perturbation_on_sphere(direction, angle_deg, rng):
    """Perturb a unit vector by a fixed angle in a random tangent direction."""
    direction = direction / np.linalg.norm(direction)
    rand_vec = rng.standard_normal(3)
    rand_vec -= np.dot(rand_vec, direction) * direction
    norm = np.linalg.norm(rand_vec)
    if norm < 1e-10:
        rand_vec = rng.standard_normal(3)
        rand_vec -= np.dot(rand_vec, direction) * direction
        norm = np.linalg.norm(rand_vec)
    rand_vec /= norm
    angle_rad = np.deg2rad(angle_deg)
    perturbed = np.cos(angle_rad) * direction + np.sin(angle_rad) * rand_vec
    return perturbed / np.linalg.norm(perturbed)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("Loading m046 data...", flush=True)
t0 = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
mag_hifi = master['mag_hifi']
n_normals = len(unique_normals)

# Select 10 trajectories spanning omega range
omega_sorted = np.argsort(omega_mags_arr)
sel_idx = np.linspace(0, 99, 10, dtype=int)
TEST_SEEDS = [int(omega_sorted[i]) for i in sel_idx]

print(f"Loaded in {time.time() - t0:.1f}s")
print(f"Test seeds: {TEST_SEEDS}")
print(f"Omega range: {[f'{omega_mags_arr[s]:.3f}' for s in TEST_SEEDS]}")

# Peak detection parameters
PEAK_DISTANCE = 5
PEAK_PROMINENCE = 0.3
ALIGNMENT_ORDER = 3       # order for argrelmin
TOLERANCE_EPOCHS = 3      # ±3 epochs = ±21.6s window

# Perturbation angles to test
PERT_ANGLES = [0, 5, 10, 20, 45, 90]
N_PERT_TRIALS = 5
rng = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════
# PART A: Alignment at truth
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("PART A: Do brightness peaks coincide with alignment minima at truth?")
print("=" * 60)

all_truth_results = []

for seed in TEST_SEEDS:
    mags = mag_hifi[seed]
    quats = master['quaternions'][seed]

    # Brightness peaks
    peaks, props = find_peaks(-mags, distance=PEAK_DISTANCE,
                               prominence=PEAK_PROMINENCE)
    bright = peaks[mags[peaks] < 9.0]

    # Alignment curves at truth
    ang_dist = compute_alignment_curves(quats, unique_normals, pab_j2000)

    # Find alignment minima per group
    minima_per_group = []
    for g in range(n_normals):
        mins = find_alignment_minima(ang_dist[g], order=ALIGNMENT_ORDER)
        minima_per_group.append(mins)

    # Check correspondence
    n_exp_all, n_all, details_all = check_peak_alignment(
        peaks, minima_per_group, TOLERANCE_EPOCHS)
    n_exp_bright, n_bright, details_bright = check_peak_alignment(
        bright, minima_per_group, TOLERANCE_EPOCHS)

    # Min alignment angle at each peak
    peak_min_angles = []
    for ep in peaks:
        min_ang = ang_dist[:, ep].min()
        peak_min_angles.append(min_ang)

    result = {
        'seed': int(seed),
        'omega_dps': float(omega_mags_arr[seed]),
        'n_peaks_all': len(peaks),
        'n_peaks_bright': len(bright),
        'explained_all': n_exp_all,
        'explained_bright': n_exp_bright,
        'pct_explained_all': n_exp_all / max(len(peaks), 1) * 100,
        'pct_explained_bright': n_exp_bright / max(len(bright), 1) * 100,
        'median_min_angle_at_peak': float(np.median(peak_min_angles)) if peak_min_angles else 0,
        'max_min_angle_at_peak': float(np.max(peak_min_angles)) if peak_min_angles else 0,
    }
    all_truth_results.append(result)

    print(f"  Seed {seed:2d} (ω={omega_mags_arr[seed]:.3f}): "
          f"{n_exp_all}/{len(peaks)} peaks explained ({n_exp_all/max(len(peaks),1)*100:.0f}%), "
          f"bright: {n_exp_bright}/{len(bright)}, "
          f"min_ang median={np.median(peak_min_angles):.1f}° max={np.max(peak_min_angles):.1f}°",
          flush=True)


# ══════════════════════════════════════════════════════════════════════
# PART B: Does perturbation break the correspondence?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("PART B: Perturbation test — do wrong candidates violate the constraint?")
print("=" * 60)

all_pert_results = []

for seed in TEST_SEEDS:
    mags = mag_hifi[seed]
    q0 = q0s[seed]
    omega0 = omega0s[seed]
    omega_dir = omega0 / np.linalg.norm(omega0)
    omega_mag = np.linalg.norm(omega0)

    # Brightness peaks (all, not just bright)
    peaks, _ = find_peaks(-mags, distance=PEAK_DISTANCE,
                           prominence=PEAK_PROMINENCE)

    seed_results = {'seed': int(seed), 'omega_dps': float(omega_mags_arr[seed]),
                    'n_peaks': len(peaks), 'perturbations': {}}

    for pert_deg in PERT_ANGLES:
        explained_rates = []

        for trial in range(N_PERT_TRIALS if pert_deg > 0 else 1):
            if pert_deg > 0:
                pert_dir = random_perturbation_on_sphere(omega_dir, pert_deg, rng)
                pert_omega = pert_dir * omega_mag
            else:
                pert_omega = omega0

            # Propagate with perturbed omega
            quats_pert, _ = propagate_attitude(
                q0, pert_omega, obs_times, "tumbling", I_tensor)

            # Alignment curves
            ang_dist = compute_alignment_curves(
                quats_pert, unique_normals, pab_j2000)

            # Find minima per group
            minima_per_group = [
                find_alignment_minima(ang_dist[g], order=ALIGNMENT_ORDER)
                for g in range(n_normals)
            ]

            # Check
            n_exp, n_total, _ = check_peak_alignment(
                peaks, minima_per_group, TOLERANCE_EPOCHS)
            explained_rates.append(n_exp / max(n_total, 1) * 100)

        seed_results['perturbations'][str(pert_deg)] = {
            'mean_explained_pct': float(np.mean(explained_rates)),
            'min_explained_pct': float(np.min(explained_rates)),
            'max_explained_pct': float(np.max(explained_rates)),
            'n_trials': N_PERT_TRIALS if pert_deg > 0 else 1,
        }

    all_pert_results.append(seed_results)

    print(f"  Seed {seed:2d} ({len(peaks):2d} pk): ", end="", flush=True)
    for pert_deg in PERT_ANGLES:
        r = seed_results['perturbations'][str(pert_deg)]
        print(f"  {pert_deg:2d}°={r['mean_explained_pct']:4.0f}%", end="")
    print(flush=True)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("SUMMARY")
print("=" * 60)

# Part A
truth_pcts = [r['pct_explained_all'] for r in all_truth_results]
print(f"\nPart A — Peaks explained at truth:")
print(f"  Mean: {np.mean(truth_pcts):.1f}%, Min: {np.min(truth_pcts):.1f}%, "
      f"Max: {np.max(truth_pcts):.1f}%")

# Part B — average explained % at each perturbation angle
print(f"\nPart B — Mean explained % vs perturbation angle:")
print(f"{'Pert':>5s} | {'Mean':>5s} | {'Min':>5s} | {'Max':>5s} | {'Gap from truth':>14s}")
print("-" * 50)
for pert_deg in PERT_ANGLES:
    pcts = [r['perturbations'][str(pert_deg)]['mean_explained_pct']
            for r in all_pert_results]
    truth_mean = np.mean(truth_pcts)
    pert_mean = np.mean(pcts)
    gap = truth_mean - pert_mean
    print(f"{pert_deg:5d}° | {pert_mean:5.1f} | {np.min(pcts):5.1f} | "
          f"{np.max(pcts):5.1f} | {gap:+13.1f}%")


# ── Save ──────────────────────────────────────────────────────────────

results = {
    'part_a': all_truth_results,
    'part_b': all_pert_results,
    'parameters': {
        'peak_distance': PEAK_DISTANCE,
        'peak_prominence': PEAK_PROMINENCE,
        'alignment_order': ALIGNMENT_ORDER,
        'tolerance_epochs': TOLERANCE_EPOCHS,
        'n_pert_trials': N_PERT_TRIALS,
        'perturbation_angles': PERT_ANGLES,
        'test_seeds': TEST_SEEDS,
    },
    'total_time_s': time.time() - t0,
}

out_path = RESULTS_DIR / 'm066_alignment_minima.json'
with open(str(out_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Total time: {time.time() - t0:.1f}s")
