#!/usr/bin/env python3
"""
m072b — Compare true vs alternate (degenerate twin) attitude histories.

Produces:
  1. Hi-fi light curves for both solutions
  2. 3D animations for both solutions
  3. Comparison LC plot

Investigates the geometric degeneracy found in m072 where the pipeline
converged to a solution with ~180 deg attitude error and ~89 deg omega error.
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

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.visualization.plotly_animation_generator import create_interactive_3d_animation

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m072_twin_comparison"
OUT_DIR.mkdir(exist_ok=True)

TRAJ_SEED = 93

# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("Loading satellite model and geometry (10 min window, 500 pts)...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T10:10:00',
                       skip_true_lc=True)

obs_times = CTX.observation_times
I_tensor = CTX.inertia_tensor

# Load true and alternate solutions
res = np.load(str(RESULTS_DIR / "m072_pipeline_seed093" / "result.npz"))
true_q0 = res['true_q0']
true_omega0 = res['true_omega0']
alt_q0 = res['q0_refined']
alt_omega0 = res['w0_refined']

print(f"True q0:  {true_q0}")
print(f"True w0:  {np.rad2deg(true_omega0)} deg/s")
print(f"Alt  q0:  {alt_q0}")
print(f"Alt  w0:  {np.rad2deg(alt_omega0)} deg/s")
print(f"q0 error: {attitude_error_deg(alt_q0, true_q0):.1f} deg")

w_true_dir = true_omega0 / np.linalg.norm(true_omega0)
w_alt_dir = alt_omega0 / np.linalg.norm(alt_omega0)
w_angle = np.rad2deg(np.arccos(np.clip(np.abs(np.dot(w_true_dir, w_alt_dir)), 0, 1)))
print(f"w  error: {w_angle:.1f} deg")
print()


# ══════════════════════════════════════════════════════════════════════
# PROPAGATE BOTH ATTITUDE HISTORIES
# ══════════════════════════════════════════════════════════════════════
print("Propagating attitudes...", flush=True)
true_quats, _ = propagate_attitude(true_q0, true_omega0, obs_times,
                                    "tumbling", I_tensor)
alt_quats, _ = propagate_attitude(alt_q0, alt_omega0, obs_times,
                                   "tumbling", I_tensor)


# ══════════════════════════════════════════════════════════════════════
# HELPER: compute body-frame vectors and generate LC + animation data
# ══════════════════════════════════════════════════════════════════════

def compute_lc_and_animation(quats, label):
    """Given quaternion history, compute hi-fi LC and animation data."""
    print(f"  Computing {label}: body-frame vectors...", flush=True)
    n_obs = len(quats)
    k1 = np.zeros((n_obs, 3))
    k2 = np.zeros((n_obs, 3))

    for i in range(n_obs):
        q = quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        sun_vec = CTX.sun_pos[i] - CTX.sat_pos[i]
        obs_vec = CTX.obs_pos[i] - CTX.sat_pos[i]
        k1[i] = R @ (sun_vec / np.linalg.norm(sun_vec))
        k2[i] = R @ (obs_vec / np.linalg.norm(obs_vec))

    print(f"  Computing {label}: shadows...", flush=True)
    lit_status = compute_shadows(
        satellite=CTX.satellite, k1_vectors=k1,
        explicit_component_matrices=CTX.art_matrices,
        show_progress=False)

    print(f"  Computing {label}: light curve + animation frames...", flush=True)
    magnitudes, flux, _, _, distances, animation_data = generate_lightcurves(
        facet_lit_status_dict=lit_status,
        k1_vectors_array=k1,
        k2_vectors_array=k2,
        observer_distances=CTX.obs_dist,
        satellite=CTX.satellite,
        epochs=CTX.epochs,
        pre_computed_matrices=CTX.art_matrices,
        generate_no_shadow=False,
        animate=True,
        show_progress=False)

    # Build geometry_data dict with attitude matrices for the animation
    att_matrices = np.zeros((n_obs, 3, 3))
    for i in range(n_obs):
        q = quats[i]
        att_matrices[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    geometry_data = {
        'sat_att_matrices': att_matrices,
        'sun_positions': CTX.sun_pos,
        'obs_positions': CTX.obs_pos,
        'sat_positions': CTX.sat_pos,
    }

    return magnitudes, animation_data, geometry_data


# ══════════════════════════════════════════════════════════════════════
# COMPUTE BOTH
# ══════════════════════════════════════════════════════════════════════
print("\n--- True attitude ---")
true_mags, true_anim, true_geom = compute_lc_and_animation(true_quats, "TRUE")

print("\n--- Alternate attitude ---")
alt_mags, alt_anim, alt_geom = compute_lc_and_animation(alt_quats, "ALT")


# ══════════════════════════════════════════════════════════════════════
# SAVE LIGHT CURVES
# ══════════════════════════════════════════════════════════════════════
np.savez(str(OUT_DIR / "twin_lightcurves.npz"),
         obs_times=obs_times, true_mags=true_mags, alt_mags=alt_mags,
         true_q0=true_q0, true_omega0=true_omega0,
         alt_q0=alt_q0, alt_omega0=alt_omega0)
print(f"\nSaved light curves to {OUT_DIR}/twin_lightcurves.npz")


# ══════════════════════════════════════════════════════════════════════
# COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

time_min = obs_times / 60.0

ax1.plot(time_min, true_mags, 'b-', alpha=0.8, linewidth=0.8, label='True')
ax1.plot(time_min, alt_mags, 'r-', alpha=0.8, linewidth=0.8, label='Alternate (twin)')
ax1.set_ylabel('Magnitude')
ax1.legend()
ax1.invert_yaxis()
ax1.set_title(f'Seed {TRAJ_SEED}: True vs Degenerate Twin  |  '
              f'q0 err={attitude_error_deg(alt_q0, true_q0):.1f} deg  '
              f'w err={w_angle:.1f} deg')
ax1.grid(True, alpha=0.3)

# Residual
ax2.plot(time_min, true_mags - alt_mags, 'k-', linewidth=0.8)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('True - Alt (mag)')
ax2.set_title(f'Residual  |  RMS={np.sqrt(np.mean((true_mags - alt_mags)**2)):.4f} mag')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "twin_lc_comparison.png"), dpi=150)
print(f"Saved LC plot to {OUT_DIR}/twin_lc_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# ANIMATIONS (OBSERVER VIEW — as seen from DST telescope)
# ══════════════════════════════════════════════════════════════════════
time_hours = obs_times / 3600.0

# All 500 frames (already 10 min window, ~1.2s spacing)
anim_idx = list(range(len(obs_times)))

print(f"\nGenerating observer-view animations ({len(anim_idx)} frames each)...")


def to_observer_view(anim_data, quats, anim_idx):
    """
    Transform animation data to the observer's viewpoint.

    For each frame:
      - Compute obs_dir_j2000 = normalize(obs_pos - sat_pos)
      - Construct R_view that maps obs_dir_j2000 → +Z (observer behind camera)
      - Rotate mesh vertices: R_view @ R(q) @ body_verts
      - Rotate sun direction: R_view @ sun_dir_j2000
      - Observer direction becomes ~[0,0,1] (fixed, points at camera)
    """
    observer_anim = []
    identity_atts = np.zeros((len(anim_idx), 3, 3))

    for i, fi in enumerate(anim_idx):
        frame = anim_data[fi]
        q = quats[fi]
        R_body2j2000 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        identity_atts[i] = np.eye(3)

        # Observer and sun directions in J2000
        obs_dir_j2000 = CTX.obs_pos[fi] - CTX.sat_pos[fi]
        obs_dir_j2000 = obs_dir_j2000 / np.linalg.norm(obs_dir_j2000)
        sun_dir_j2000 = CTX.sun_pos[fi] - CTX.sat_pos[fi]
        sun_dir_j2000 = sun_dir_j2000 / np.linalg.norm(sun_dir_j2000)

        # R_view: maps obs_dir_j2000 → +Z
        R_view_rot, _ = Rotation.align_vectors([[0, 0, 1]], [obs_dir_j2000])
        R_view = R_view_rot.as_matrix()

        # Combined rotation: body → J2000 → observer view
        R_total = R_view @ R_body2j2000

        new_frame = dict(frame)
        body_verts = frame['transformed_vertices']
        new_frame['transformed_vertices'] = (R_total @ body_verts.T).T
        new_frame['sun_direction'] = R_view @ sun_dir_j2000
        new_frame['observer_direction'] = R_view @ obs_dir_j2000  # ≈ [0,0,1]

        observer_anim.append(new_frame)

    geom = {
        'sat_att_matrices': identity_atts,
    }
    return observer_anim, geom


true_obs_anim, true_obs_geom = to_observer_view(true_anim, true_quats, anim_idx)

print("  True (observer view) animation...", flush=True)
create_interactive_3d_animation(
    animation_data=true_obs_anim,
    magnitudes=true_mags[anim_idx],
    time_hours=time_hours[anim_idx],
    geometry_data=true_obs_geom,
    satellite_name="IS-901 TRUE (observer view)",
    show_j2000_frame=False,
    show_body_frame=False,
    show_sun_vector=True,
    show_observer_vector=True,
    frame_duration_ms=100,
    save=True,
    color_mode='flux',
    output_path=OUT_DIR / "animation_TRUE_observer.html",
)
print(f"  Saved: {OUT_DIR}/animation_TRUE_observer.html")

alt_obs_anim, alt_obs_geom = to_observer_view(alt_anim, alt_quats, anim_idx)

print("  Alternate (observer view) animation...", flush=True)
create_interactive_3d_animation(
    animation_data=alt_obs_anim,
    magnitudes=alt_mags[anim_idx],
    time_hours=time_hours[anim_idx],
    geometry_data=alt_obs_geom,
    satellite_name="IS-901 ALTERNATE (observer view)",
    show_j2000_frame=False,
    show_body_frame=False,
    show_sun_vector=True,
    show_observer_vector=True,
    frame_duration_ms=100,
    save=True,
    color_mode='flux',
    output_path=OUT_DIR / "animation_ALT_observer.html",
)
print(f"  Saved: {OUT_DIR}/animation_ALT_observer.html")

print(f"\nDone. All outputs in {OUT_DIR}/")
