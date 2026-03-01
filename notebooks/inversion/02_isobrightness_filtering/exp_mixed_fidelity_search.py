#!/usr/bin/env python3
"""
Mixed-fidelity iso-brightness search:
1. 10,000 lo-fi seeds → find all lo-fi iso-brightness attitudes (~5 min)
2. Cluster converged attitudes
3. Use cluster centers as hi-fi seeds → refine (~10 min)
4. Check if true attitude is in final set

Phase 1: Original true attitude
Phase 2: 10 random true attitudes (robustness)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.cluster.hierarchy import fcluster, linkage

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.computation.shadow_engine import create_no_shadow_lit_status

def log(msg):
    print(msg, flush=True)

t_start = time.time()

# ─── Setup ───
config_path = "intelsat_901/intelsat_901_config.yaml"
n_observations = 100
OBSERVER_ID = 399999
noise_sigma = 0.05

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config(config_path)
metakernel_path = config_manager.get_metakernel_path(config)
satellite_id = config.spice_config.satellite_id
start_time_utc = config.simulation_defaults.start_time
end_time_utc = config.simulation_defaults.end_time

satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(config=config, config_manager=config_manager, masses=component_masses,
                                              articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

fixed_articulation_angles = {
    'SP_North': np.full(n_observations, 0.0),
    'SP_South': np.full(n_observations, 0.0),
    'AD_East': np.full(n_observations, 15.0),
    'AD_West': np.full(n_observations, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

log(f"Setup: {time.time()-t_start:.1f}s")

# ─── Brightness evaluation ───
def brightness_at_quaternion(q_wxyz, epoch_idx, use_shadows=True):
    """Compute brightness for one quaternion at one epoch. use_shadows=False for lo-fi."""
    q = q_wxyz
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    sun_j2000 = sun_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    obs_j2000 = observer_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    k1_body = R @ sun_j2000; k1_body /= np.linalg.norm(k1_body)
    k2_body = R @ obs_j2000; k2_body /= np.linalg.norm(k2_body)
    k1 = k1_body.reshape(1, 3)
    k2 = k2_body.reshape(1, 3)
    art = {c: m[0:1] for c, m in articulation_matrices.items()}
    
    if use_shadows:
        lit = compute_shadows(satellite=satellite, k1_vectors=k1, explicit_component_matrices=art, show_progress=False)
    else:
        lit = create_no_shadow_lit_status(satellite, 1)
    
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.array([observer_distances[epoch_idx]]),
        satellite=satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art, generate_no_shadow=False, animate=False, show_progress=False)
    return mags[0]

def axis_angle_to_quat(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])

def brightness_objective(aa_params, epoch_idx, target_mag, use_shadows=True):
    q = axis_angle_to_quat(aa_params)
    pred = brightness_at_quaternion(q, epoch_idx, use_shadows=use_shadows)
    return (pred - target_mag)**2

def mixed_fidelity_search(true_q_epoch, epoch_idx, n_lofi_seeds=10000, cluster_threshold=5.0, label=""):
    """
    Mixed-fidelity iso-brightness search at one epoch.
    Returns dict with min_att_error, n_hifi_candidates, timing, etc.
    """
    R_true = Rotation.from_quat([true_q_epoch[1], true_q_epoch[2], true_q_epoch[3], true_q_epoch[0]])
    
    # True brightness (hi-fi) + noise
    true_mag = brightness_at_quaternion(true_q_epoch, epoch_idx, use_shadows=True)
    np.random.seed(42)
    target_mag = true_mag + np.random.normal(0, noise_sigma)
    
    # Also get lo-fi brightness for lo-fi target
    true_mag_lofi = brightness_at_quaternion(true_q_epoch, epoch_idx, use_shadows=False)
    target_mag_lofi = true_mag_lofi + np.random.normal(0, noise_sigma)
    
    log(f"  {label} Target mag (hi-fi): {target_mag:.4f}, (lo-fi): {target_mag_lofi:.4f}")
    
    # ── Stage 1: Lo-fi sweep ──
    log(f"  Stage 1: {n_lofi_seeds} lo-fi seeds...")
    t_lofi = time.time()
    seeds = Rotation.random(n_lofi_seeds, random_state=42)
    lofi_results = []
    
    for si in range(n_lofi_seeds):
        aa_init = seeds[si].as_rotvec()
        try:
            res = minimize(brightness_objective, aa_init, args=(epoch_idx, target_mag_lofi, False),
                          method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, epoch_idx, use_shadows=False)
            residual = abs(final_mag - target_mag_lofi)
            if residual < noise_sigma:
                R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
                att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
                lofi_results.append({'rotvec': res.x, 'att_err': att_err, 'residual': residual})
        except:
            pass
        
        if (si+1) % 500 == 0:
            log(f"    {si+1}/{n_lofi_seeds} done, {len(lofi_results)} good so far ({time.time()-t_lofi:.0f}s)")
    
    dt_lofi = time.time() - t_lofi
    log(f"  Stage 1 done: {len(lofi_results)}/{n_lofi_seeds} converged within σ, {dt_lofi:.0f}s")
    
    if len(lofi_results) == 0:
        log(f"  ERROR: No lo-fi solutions found!")
        return {'min_att_err': 999, 'n_lofi_good': 0, 'n_clusters': 0, 'n_hifi_good': 0, 
                'dt_lofi': dt_lofi, 'dt_hifi': 0}
    
    lofi_errs = [r['att_err'] for r in lofi_results]
    log(f"  Lo-fi attitude errors: min={min(lofi_errs):.2f}°, median={np.median(lofi_errs):.1f}°")
    
    # ── Stage 2: Cluster lo-fi results ──
    rotvecs = np.array([r['rotvec'] for r in lofi_results])
    if len(rotvecs) > 1:
        Z = linkage(rotvecs, method='average', metric='euclidean')
        clusters = fcluster(Z, t=np.deg2rad(cluster_threshold), criterion='distance')
        n_clusters = len(set(clusters))
        
        # Get cluster centers
        cluster_centers = []
        for ci in sorted(set(clusters)):
            members = [rotvecs[i] for i, c in enumerate(clusters) if c == ci]
            center = np.mean(members, axis=0)
            cluster_centers.append(center)
    else:
        n_clusters = 1
        cluster_centers = [rotvecs[0]]
    
    log(f"  Stage 2: {n_clusters} clusters (threshold={cluster_threshold}°)")
    
    # ── Stage 3: Hi-fi refinement of cluster centers ──
    log(f"  Stage 3: Hi-fi refinement of {n_clusters} cluster centers...")
    t_hifi = time.time()
    hifi_results = []
    
    for ci, center in enumerate(cluster_centers):
        try:
            res = minimize(brightness_objective, center, args=(epoch_idx, target_mag, True),
                          method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, epoch_idx, use_shadows=True)
            residual = abs(final_mag - target_mag)
            R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
            att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
            hifi_results.append({'cluster': ci, 'att_err': att_err, 'residual': residual})
        except:
            pass
    
    dt_hifi = time.time() - t_hifi
    
    good_hifi = [r for r in hifi_results if r['residual'] < noise_sigma]
    hifi_errs = [r['att_err'] for r in good_hifi] if good_hifi else [999]
    min_hifi_err = min(hifi_errs)
    
    log(f"  Stage 3 done: {len(good_hifi)}/{n_clusters} within σ, {dt_hifi:.0f}s")
    log(f"  Hi-fi attitude errors: min={min_hifi_err:.2f}°")
    log(f"  Truth recovered (<1°): {'YES ✓' if min_hifi_err < 1.0 else 'NO ✗'}")
    log(f"  Truth recovered (<5°): {'YES ✓' if min_hifi_err < 5.0 else 'NO ✗'}")
    log(f"  Total time: {dt_lofi + dt_hifi:.0f}s")
    
    return {
        'min_att_err': float(min_hifi_err),
        'n_lofi_good': len(lofi_results),
        'n_clusters': n_clusters,
        'n_hifi_good': len(good_hifi),
        'min_lofi_err': float(min(lofi_errs)),
        'dt_lofi': dt_lofi,
        'dt_hifi': dt_hifi,
        'hifi_top5': sorted(hifi_errs)[:5]
    }

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Original true attitude
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("PHASE 1: Original true attitude, epoch 0")
log(f"{'='*70}")

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0_orig = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])

p1 = mixed_fidelity_search(true_q0_orig, epoch_idx=0, n_lofi_seeds=2000, label="Phase1")

log(f"\n>>> PHASE 1 COMPLETE — min error: {p1['min_att_err']:.2f}° <<<\n")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 10 random true attitudes
# ═══════════════════════════════════════════════════════════════
log(f"{'='*70}")
log("PHASE 2: Robustness — 10 random true attitudes")
log(f"{'='*70}")

N_TRIALS = 10
phase2_results = []
true_attitudes = Rotation.random(N_TRIALS, random_state=777)

for ti in range(N_TRIALS):
    R_true_i = true_attitudes[ti]
    q_scipy = R_true_i.as_quat()  # x,y,z,w
    q_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    
    log(f"\n--- Trial {ti+1}/{N_TRIALS} ---")
    result = mixed_fidelity_search(q_wxyz, epoch_idx=0, n_lofi_seeds=2000, label=f"T{ti+1}")
    phase2_results.append(result)

# ── Summary ──
log(f"\n{'='*70}")
log("SUMMARY")
log(f"{'='*70}")
log(f"Phase 1: min_err={p1['min_att_err']:.2f}°, {p1['n_clusters']} clusters, "
    f"recovered(<1°): {'YES' if p1['min_att_err']<1 else 'NO'}, "
    f"recovered(<5°): {'YES' if p1['min_att_err']<5 else 'NO'}")

n_rec_1 = sum(1 for r in phase2_results if r['min_att_err'] < 1.0)
n_rec_5 = sum(1 for r in phase2_results if r['min_att_err'] < 5.0)
min_errs = [r['min_att_err'] for r in phase2_results]
log(f"\nPhase 2 ({N_TRIALS} trials):")
log(f"  Recovered <1°: {n_rec_1}/{N_TRIALS}")
log(f"  Recovered <5°: {n_rec_5}/{N_TRIALS}")
log(f"  Min errors: {[f'{e:.1f}°' for e in min_errs]}")
log(f"  Mean min error: {np.mean(min_errs):.2f}°")
log(f"\nTotal time: {time.time()-t_start:.0f}s")

# Save
out = {
    'phase1': p1,
    'phase2': phase2_results,
    'summary': {'recovered_1deg': n_rec_1, 'recovered_5deg': n_rec_5, 'n_trials': N_TRIALS}
}
outpath = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_mixed_fidelity_search.json"
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2, default=str)
log(f"Saved: {outpath}")
