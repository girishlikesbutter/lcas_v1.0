#!/usr/bin/env python3
"""
m069c — Light curve comparison: true vs estimated state from m069b.

Propagates both the true and estimated (q0, omega0), generates hi-fi
light curves, and plots them with a residual panel.
"""

import sys
import os
import time
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

# ── Setup ──
print("Setting up...", flush=True)
t0 = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
true_lc = master['mag_hifi'][93]

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Load refined result
refined = np.load(str(RESULTS_DIR / "m069b_geometric_refinement.npz"))
true_q0 = refined['true_q0']
true_omega0 = refined['true_omega0']
est_q0 = refined['q0_refined']
est_omega0 = refined['w0_refined']

print(f"Setup: {time.time() - t0:.1f}s")


# ── Generate hi-fi LC for estimated state ──
print("Generating estimated hi-fi LC...", flush=True)
t1 = time.time()

obj = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

# Estimated LC
quats_est, _ = propagate_attitude(est_q0, est_omega0, obs_times, "tumbling", I_tensor)
k1_est, k2_est = obj._compute_body_frame_vectors(quats_est)
est_lc = obj._generate_predicted_lightcurve(k1_est, k2_est)

print(f"Estimated LC: {time.time() - t1:.1f}s")

# True LC (already have true_lc from dataset, but regenerate for consistency)
print("Generating true hi-fi LC...", flush=True)
t2 = time.time()
quats_true, _ = propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I_tensor)
k1_true, k2_true = obj._compute_body_frame_vectors(quats_true)
true_lc_regen = obj._generate_predicted_lightcurve(k1_true, k2_true)
print(f"True LC: {time.time() - t2:.1f}s")

# Residual
residual = true_lc_regen - est_lc
rms = np.sqrt(np.nanmean(residual**2))
valid = np.isfinite(residual)
print(f"Residual RMS: {rms:.4f} mag")
print(f"Max |residual|: {np.nanmax(np.abs(residual)):.4f} mag")


# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                          gridspec_kw={'height_ratios': [3, 1]},
                          sharex=True)

ax1 = axes[0]
ax1.plot(obs_times, observed_lc, 'k-', alpha=0.3, linewidth=0.6,
         label='Observed (hi-fi + noise)')
ax1.plot(obs_times, true_lc_regen, 'g-', alpha=0.7, linewidth=0.9,
         label='True state')
ax1.plot(obs_times, est_lc, 'r--', alpha=0.7, linewidth=0.9,
         label='Estimated state (q0=1.94°, ω=0.14°)')
ax1.set_ylabel('Apparent Magnitude')
ax1.set_title(f'Seed 93 | Geometric refinement result | RMS residual = {rms:.4f} mag')
ax1.legend(loc='upper right', fontsize=9)
ax1.invert_yaxis()

ax2 = axes[1]
ax2.plot(obs_times[valid], residual[valid], 'b-', alpha=0.5, linewidth=0.6)
ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('True − Estimated (mag)')
ax2.set_ylim(-1, 1)

plt.tight_layout()
out_path = RESULTS_DIR / "m069c_lc_comparison.png"
plt.savefig(str(out_path), dpi=150)
print(f"\nSaved: {out_path}")

# Save data
np.savez(str(RESULTS_DIR / "m069c_lc_comparison.npz"),
         obs_times=obs_times, observed_lc=observed_lc,
         true_lc=true_lc_regen, est_lc=est_lc, residual=residual)
print(f"Saved: m069c_lc_comparison.npz")
print(f"Total: {time.time() - t0:.1f}s")
