#!/usr/bin/env python3
"""
m071b — Plot the WRONG geometric winner's LC vs truth.

The geometric cost selected ω#2 φ#2 (88.9° omega error) over truth.
Propagate this wrong candidate and show why the geometric cost is
fooled but the LC clearly doesn't match.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

# Setup (skip true LC — we have it from m046)
print("Setting up...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
true_lc = master['mag_hifi'][93]
true_q0 = master['q0s'][93]
true_omega0 = master['omega0s'][93]

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Load wrong winner from m071
data = np.load(str(RESULTS_DIR / "m071_geometric_selection.npz"))
# Rank #1 by geometric cost = candidate index 0 after sorting
# But the NPZ stores by original candidate index. Read the JSON to find which.
import json
with open(str(RESULTS_DIR / "m071_geometric_selection.json")) as f:
    r71 = json.load(f)

# The candidates are sorted by geo_cost in the JSON
wrong_winner = r71['candidates'][0]
truth_cand = None
for c in r71['candidates']:
    if c['post_w0_err'] < 5:
        truth_cand = c
        break

print(f"Wrong winner: ω#{wrong_winner['omega_rank']+1} φ#{wrong_winner['phi_rank']+1} "
      f"| q0={wrong_winner['post_q0_err']:.1f}° ω={wrong_winner['post_w0_err']:.1f}°")
if truth_cand:
    print(f"Truth cand:   ω#{truth_cand['omega_rank']+1} φ#{truth_cand['phi_rank']+1} "
          f"| q0={truth_cand['post_q0_err']:.1f}° ω={truth_cand['post_w0_err']:.1f}°")

# Find the index in the NPZ arrays
# The NPZ has q0_refined_0, q0_refined_1, ... in original candidate order
# We need to map from the sorted JSON back to original index
# The JSON candidates are sorted by geo_cost. The original index is implicit.
# Actually, the NPZ stores by the ORIGINAL candidate index before sorting.
# The JSON 'candidates' list is sorted by geo_cost. We need the original index.
# Simpler: just find the candidate with the lowest geo_cost in the NPZ.

# Re-derive: the JSON has omega_rank and phi_rank for each candidate.
# Original index = omega_rank * 2 + phi_rank
wrong_idx = wrong_winner['omega_rank'] * 2 + wrong_winner['phi_rank']
wrong_q0 = data[f'q0_refined_{wrong_idx}']
wrong_w0 = data[f'w0_refined_{wrong_idx}']

print(f"\nWrong winner state:")
print(f"  q0: {wrong_q0}")
print(f"  w0: {np.rad2deg(wrong_w0)} deg/s")

# Generate hi-fi LCs
print("Generating hi-fi LCs...", flush=True)

obj = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

# Wrong winner LC
quats_wrong, _ = propagate_attitude(wrong_q0, wrong_w0, obs_times, "tumbling", I_tensor)
k1_w, k2_w = obj._compute_body_frame_vectors(quats_wrong)
wrong_lc = obj._generate_predicted_lightcurve(k1_w, k2_w)

# True LC (regenerate for consistency)
quats_true, _ = propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I_tensor)
k1_t, k2_t = obj._compute_body_frame_vectors(quats_true)
true_lc_regen = obj._generate_predicted_lightcurve(k1_t, k2_t)

residual = true_lc_regen - wrong_lc
rms = np.sqrt(np.nanmean(residual**2))
print(f"RMS residual (true vs wrong winner): {rms:.3f} mag")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                          gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax1 = axes[0]
ax1.plot(obs_times, observed_lc, 'k-', alpha=0.3, linewidth=0.6, label='Observed')
ax1.plot(obs_times, true_lc_regen, 'g-', alpha=0.7, linewidth=0.9, label='True state')
ax1.plot(obs_times, wrong_lc, 'r--', alpha=0.7, linewidth=0.9,
         label=f'Wrong geo winner (q0={wrong_winner["post_q0_err"]:.0f}° '
               f'ω={wrong_winner["post_w0_err"]:.0f}°)')
ax1.set_ylabel('Apparent Magnitude')
ax1.set_title(f'Seed 93 | Wrong geometric winner vs truth | RMS = {rms:.2f} mag')
ax1.legend(loc='upper right', fontsize=9)
ax1.invert_yaxis()

ax2 = axes[1]
valid = np.isfinite(residual)
ax2.plot(obs_times[valid], residual[valid], 'b-', alpha=0.5, linewidth=0.6)
ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('True − Wrong (mag)')
ax2.set_ylim(-8, 8)

plt.tight_layout()
out_path = RESULTS_DIR / "m071b_wrong_winner_lc.png"
plt.savefig(str(out_path), dpi=150)
print(f"Saved: {out_path}")
