#!/usr/bin/env python3
"""
m071e — Correct phi+180 twin: rotate q by 180 about +X,
transform the SAME inertial omega into the new body frame.
"""

import sys
import os
import numpy as np
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

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][93]
true_w0 = master['omega0s'][93]

# R(q) = J2000 → body
R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
R_180x = Rotation.from_rotvec(np.pi * np.array([1, 0, 0]))

# Body-frame rotation of 180° about +X
# R_twin = R_true @ R_180x (body-frame rotation applied first)
R_twin = R_true * R_180x
q_twin_xyzw = R_twin.as_quat()
q_twin = np.array([q_twin_xyzw[3], q_twin_xyzw[0], q_twin_xyzw[1], q_twin_xyzw[2]])

# Same inertial omega, transform to new body frame
# w_inertial = R_true.T @ w_body_true (body → inertial)
w_inertial = R_true.as_matrix().T @ true_w0
# w_body_twin = R_twin @ w_inertial (inertial → new body)
w_twin = R_twin.as_matrix() @ w_inertial

print(f"True  q0: {true_q0}")
print(f"Twin  q0: {q_twin}")
print(f"True  w0 (body, deg/s): {np.rad2deg(true_w0)}")
print(f"Twin  w0 (body, deg/s): {np.rad2deg(w_twin)}")
print(f"Inertial w (deg/s):     {np.rad2deg(w_inertial)}")
print(f"|w| true: {np.rad2deg(np.linalg.norm(true_w0)):.4f}")
print(f"|w| twin: {np.rad2deg(np.linalg.norm(w_twin)):.4f}")

# Propagate both
print("\nPropagating...", flush=True)
quats_true, _ = propagate_attitude(true_q0, true_w0, obs_times, "tumbling", I_tensor)
quats_twin, _ = propagate_attitude(q_twin, w_twin, obs_times, "tumbling", I_tensor)

# Generate hi-fi LCs
print("Generating hi-fi LCs...", flush=True)
obj = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=np.zeros(500),
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

k1t, k2t = obj._compute_body_frame_vectors(quats_true)
k1tw, k2tw = obj._compute_body_frame_vectors(quats_twin)
lc_true = obj._generate_predicted_lightcurve(k1t, k2t)
lc_twin = obj._generate_predicted_lightcurve(k1tw, k2tw)

diff = lc_true - lc_twin
rms = np.sqrt(np.nanmean(diff**2))
print(f"\nRESULT:")
print(f"  RMS:  {rms:.6f} mag")
print(f"  Max:  {np.nanmax(np.abs(diff)):.6f} mag")
print(f"  IDENTICAL (atol=0.01)?  {np.allclose(lc_true, lc_twin, atol=0.01)}")
print(f"  IDENTICAL (atol=0.001)? {np.allclose(lc_true, lc_twin, atol=0.001)}")

# Save
np.savez(str(RESULTS_DIR / "m071e_correct_twin.npz"),
         q_twin=q_twin, w_twin=w_twin,
         true_q0=true_q0, true_w0=true_w0,
         w_inertial=w_inertial,
         lc_true=lc_true, lc_twin=lc_twin, diff=diff)
print(f"Saved: m071e_correct_twin.npz")
