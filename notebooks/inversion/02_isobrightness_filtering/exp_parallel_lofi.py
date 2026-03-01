#!/usr/bin/env python3
"""
Step 1: Validate parallel lo-fi iso-brightness optimization.
2000 random seeds → L-BFGS-B lo-fi, 8 parallel workers.
Then hi-fi refinement of cluster centers.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.cluster.hierarchy import fcluster, linkage
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

def log(msg):
    print(msg, flush=True)

t_start = time.time()

# ─── Setup ───
from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

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

# True parameters
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))

log(f"Setup: {time.time()-t_start:.1f}s")

# ─── Brightness helpers ───
def brightness_at_quaternion(q_wxyz, epoch_idx, use_shadows):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
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

# ─── Worker functions ───
def lofi_worker(args):
    aa_init, epoch_idx, target_mag, true_q_wxyz = args
    try:
        def obj(aa):
            q = axis_angle_to_quat(aa)
            pred = brightness_at_quaternion(q, epoch_idx, use_shadows=False)
            return (pred - target_mag)**2
        
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
        q_final = axis_angle_to_quat(res.x)
        final_mag = brightness_at_quaternion(q_final, epoch_idx, use_shadows=False)
        residual = abs(final_mag - target_mag)
        R_true = Rotation.from_quat([true_q_wxyz[1], true_q_wxyz[2], true_q_wxyz[3], true_q_wxyz[0]])
        R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
        att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return {'rotvec': res.x.tolist(), 'att_err': float(att_err), 'residual': float(residual), 'nfev': res.nfev}
    except Exception as e:
        return {'rotvec': aa_init.tolist(), 'att_err': 999.0, 'residual': 999.0, 'error': str(e)}

def hifi_worker(args):
    aa_init, epoch_idx, target_mag, true_q_wxyz = args
    try:
        def obj(aa):
            q = axis_angle_to_quat(aa)
            pred = brightness_at_quaternion(q, epoch_idx, use_shadows=True)
            return (pred - target_mag)**2
        
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
        q_final = axis_angle_to_quat(res.x)
        final_mag = brightness_at_quaternion(q_final, epoch_idx, use_shadows=True)
        residual = abs(final_mag - target_mag)
        R_true = Rotation.from_quat([true_q_wxyz[1], true_q_wxyz[2], true_q_wxyz[3], true_q_wxyz[0]])
        R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
        att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return {'rotvec': res.x.tolist(), 'att_err': float(att_err), 'residual': float(residual), 'nfev': res.nfev}
    except Exception as e:
        return {'rotvec': aa_init.tolist(), 'att_err': 999.0, 'residual': 999.0, 'error': str(e)}

# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Generate observed brightness
    true_mag = brightness_at_quaternion(true_q0, 0, use_shadows=True)
    np.random.seed(42)
    target_mag_hifi = true_mag + np.random.normal(0, noise_sigma)
    
    # Lo-fi target: shift by systematic bias
    true_mag_lofi = brightness_at_quaternion(true_q0, 0, use_shadows=False)
    lofi_bias = true_mag_lofi - true_mag
    target_mag_lofi = target_mag_hifi + lofi_bias
    
    log(f"Hi-fi target: {target_mag_hifi:.4f} mag")
    log(f"Lo-fi target: {target_mag_lofi:.4f} mag (bias: {lofi_bias:.4f})")
    
    R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
    
    # ── Stage 1: 2000 parallel lo-fi optimizations ──
    N_LOFI = 10000
    log(f"\n{'='*70}")
    log(f"STAGE 1: {N_LOFI} lo-fi optimizations, 8 parallel workers")
    log(f"{'='*70}")
    
    seeds = Rotation.random(N_LOFI, random_state=42)
    worker_args = [(seeds[i].as_rotvec(), 0, target_mag_lofi, true_q0) for i in range(N_LOFI)]
    
    t1 = time.time()
    with Pool(8) as pool:
        lofi_results = pool.map(lofi_worker, worker_args)
    dt1 = time.time() - t1
    
    good_lofi = [r for r in lofi_results if r['residual'] < noise_sigma]
    lofi_errs = sorted([r['att_err'] for r in good_lofi])
    avg_nfev = np.mean([r.get('nfev', 0) for r in lofi_results if 'nfev' in r])
    
    log(f"Time: {dt1:.1f}s")
    log(f"Converged within σ: {len(good_lofi)}/{N_LOFI}")
    log(f"Avg function evals: {avg_nfev:.0f}")
    log(f"Min attitude error (lo-fi): {lofi_errs[0]:.2f}°" if lofi_errs else "No good results!")
    log(f"Top-10 closest: {[f'{e:.1f}°' for e in lofi_errs[:10]]}")
    
    # ── Stage 2: Cluster lo-fi results ──
    log(f"\n{'='*70}")
    log("STAGE 2: Cluster lo-fi results")
    log(f"{'='*70}")
    
    good_rotvecs = np.array([r['rotvec'] for r in good_lofi])
    if len(good_rotvecs) > 1:
        Z = linkage(good_rotvecs, method='average', metric='euclidean')
        clusters = fcluster(Z, t=np.deg2rad(5.0), criterion='distance')
        n_clusters = len(set(clusters))
        
        cluster_centers = []
        cluster_att_errs = []
        for ci in sorted(set(clusters)):
            members_idx = [i for i, c in enumerate(clusters) if c == ci]
            members_rv = good_rotvecs[members_idx]
            center = np.mean(members_rv, axis=0)
            cluster_centers.append(center)
            # Attitude error of center
            R_center = Rotation.from_rotvec(center)
            err = np.rad2deg((R_center.inv() * R_true).magnitude())
            cluster_att_errs.append(err)
        
        sorted_errs = sorted(cluster_att_errs)
        log(f"Distinct clusters (5° threshold): {n_clusters}")
        log(f"Min cluster center error: {sorted_errs[0]:.2f}°")
        log(f"Top-10 cluster errors: {[f'{e:.1f}°' for e in sorted_errs[:10]]}")
    else:
        log(f"Only {len(good_rotvecs)} good results, skipping clustering")
        cluster_centers = [good_rotvecs[0]] if len(good_rotvecs) == 1 else []
        n_clusters = len(cluster_centers)
    
    # ── Summary ──
    dt3 = 0
    log(f"\n{'='*70}")
    log("SUMMARY (lo-fi only)")
    log(f"{'='*70}")
    log(f"Lo-fi stage: {N_LOFI} seeds, {len(good_lofi)} converged, {dt1:.1f}s")
    log(f"Clustering: {n_clusters} distinct attitudes")
    log(f"Min lo-fi attitude error: {lofi_errs[0]:.2f}°" if lofi_errs else "No good results")
    log(f"Candidates within 5° of truth: {sum(1 for e in lofi_errs if e < 5.0)}")
    log(f"Candidates within 10° of truth: {sum(1 for e in lofi_errs if e < 10.0)}")
    log(f"Total time: {time.time()-t_start:.0f}s")
    
    # Save
    out = {
        'n_lofi_seeds': N_LOFI,
        'n_lofi_good': len(good_lofi),
        'n_clusters': n_clusters,
        'min_lofi_err': float(lofi_errs[0]) if lofi_errs else 999,
        'lofi_top20': lofi_errs[:20] if lofi_errs else [],
        'n_within_5deg': sum(1 for e in lofi_errs if e < 5.0),
        'n_within_10deg': sum(1 for e in lofi_errs if e < 10.0),
        'dt_lofi': dt1,
        'dt_total': time.time() - t_start
    }
    outpath = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_parallel_lofi_10k.json"
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
