#!/usr/bin/env python3
"""
Iso-brightness attitude finding experiments.

Exp A: How many distinct attitudes match a given brightness? (200 random seeds, L-BFGS-B)
Exp B: Optimizer comparison (L-BFGS-B vs Nelder-Mead vs Powell)
Exp C: Scaling to 1000 seeds, cluster analysis
Exp D: Repeat at multiple epochs
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

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
from src.inversion.objective_function import ObjectiveFunction
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

def log(msg):
    print(msg)
    sys.stdout.flush()

log("="*70)
log("ISO-BRIGHTNESS ATTITUDE FINDING")
log("="*70)

t_start = time.time()

# ─── Setup (reused from previous experiments) ───
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

# True parameters & lightcurve
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))

true_quaternions, _ = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor)

true_k1, true_k2 = obj_temp._compute_body_frame_vectors(true_quaternions)
true_lit = compute_shadows(satellite=satellite, k1_vectors=true_k1,
                            explicit_component_matrices=articulation_matrices, show_progress=False)
true_lc, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=true_lit, k1_vectors_array=true_k1, k2_vectors_array=true_k2,
    observer_distances=observer_distances, satellite=satellite, epochs=epochs,
    pre_computed_matrices=articulation_matrices, generate_no_shadow=False, animate=False, show_progress=False)

np.random.seed(42)
observed_lc = true_lc + np.random.normal(0, noise_sigma, n_observations)

log(f"Setup time: {time.time() - t_start:.1f}s")
log(f"True LC range: [{true_lc.min():.2f}, {true_lc.max():.2f}] mag")
log(f"Noise: σ = {noise_sigma} mag")

# ─── Single-attitude brightness evaluation ───
eval_count = [0]

def brightness_at_quaternion(q_wxyz, epoch_idx):
    """Compute hi-fi brightness for one quaternion at one epoch."""
    eval_count[0] += 1
    q = q_wxyz
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    
    sun_j2000 = sun_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    obs_j2000 = observer_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    
    k1_body = R @ sun_j2000; k1_body = k1_body / np.linalg.norm(k1_body)
    k2_body = R @ obs_j2000; k2_body = k2_body / np.linalg.norm(k2_body)
    
    k1 = k1_body.reshape(1, 3)
    k2 = k2_body.reshape(1, 3)
    
    art = {}
    for comp_name, mat_full in articulation_matrices.items():
        art[comp_name] = mat_full[0:1]
    
    lit = compute_shadows(satellite=satellite, k1_vectors=k1,
                          explicit_component_matrices=art, show_progress=False)
    
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.array([observer_distances[epoch_idx]]),
        satellite=satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art, generate_no_shadow=False, animate=False, show_progress=False)
    
    return mags[0]

def axis_angle_to_quat(aa):
    """Convert axis-angle (3,) to quaternion w,x,y,z."""
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])

def brightness_objective(aa_params, epoch_idx, target_mag):
    """Objective: squared error between predicted and target brightness."""
    q = axis_angle_to_quat(aa_params)
    pred = brightness_at_quaternion(q, epoch_idx)
    return (pred - target_mag)**2

# ═══════════════════════════════════════════════════════════════
# EXP A: 200 random seeds, L-BFGS-B, epoch 0
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("EXP A: Find iso-brightness attitudes (200 seeds, L-BFGS-B, epoch 0)")
log(f"{'='*70}")

N_SEEDS_A = 200
target_mag_0 = observed_lc[0]
log(f"Target brightness: {target_mag_0:.4f} mag")
log(f"Tolerance: {noise_sigma} mag (1σ)")

seeds_A = Rotation.random(N_SEEDS_A, random_state=123)
results_A = []
t_expA = time.time()

for si in range(N_SEEDS_A):
    aa_init = seeds_A[si].as_rotvec()
    eval_count[0] = 0
    t_opt = time.time()
    
    try:
        res = minimize(brightness_objective, aa_init, args=(0, target_mag_0),
                      method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
        
        q_final = axis_angle_to_quat(res.x)
        final_mag = brightness_at_quaternion(q_final, 0)
        residual = abs(final_mag - target_mag_0)
        
        # Distance from true attitude at epoch 0
        R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
        R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
        att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
        
        results_A.append({
            'seed': si, 'residual_mag': float(residual),
            'att_error_deg': float(att_err),
            'final_rotvec': res.x.tolist(),
            'final_mag': float(final_mag),
            'n_evals': eval_count[0],
            'opt_time': time.time() - t_opt,
            'success': res.success,
        })
    except Exception as e:
        results_A.append({'seed': si, 'error': str(e)})
    
    if (si+1) % 20 == 0:
        elapsed = time.time() - t_expA
        good = sum(1 for r in results_A if r.get('residual_mag', 999) < noise_sigma)
        log(f"  {si+1}/{N_SEEDS_A}: {good} within σ, {elapsed:.1f}s elapsed, "
            f"~{elapsed/(si+1)*N_SEEDS_A:.0f}s total est")

t_expA_dur = time.time() - t_expA

# Analysis
good_A = [r for r in results_A if r.get('residual_mag', 999) < noise_sigma]
close_A = [r for r in results_A if r.get('residual_mag', 999) < 2*noise_sigma]

log(f"\nEXP A RESULTS ({t_expA_dur:.1f}s):")
log(f"  Seeds: {N_SEEDS_A}")
log(f"  Within 1σ ({noise_sigma} mag): {len(good_A)}")
log(f"  Within 2σ ({2*noise_sigma} mag): {len(close_A)}")

if good_A:
    att_errors = [r['att_error_deg'] for r in good_A]
    log(f"  Attitude errors of good fits: min={min(att_errors):.1f}°, max={max(att_errors):.1f}°, "
        f"mean={np.mean(att_errors):.1f}°")
    avg_evals = np.mean([r['n_evals'] for r in good_A])
    avg_time = np.mean([r['opt_time'] for r in good_A])
    log(f"  Avg evals per opt: {avg_evals:.0f}, avg time: {avg_time:.2f}s")
    
    # Cluster by angular distance (5° threshold)
    from scipy.cluster.hierarchy import fcluster, linkage
    rotvecs = np.array([r['final_rotvec'] for r in good_A])
    if len(rotvecs) > 1:
        # Compute pairwise angular distances
        rotations = Rotation.from_rotvec(rotvecs)
        n = len(rotations)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist_matrix[i,j] = np.rad2deg((rotations[i].inv() * rotations[j]).magnitude())
                dist_matrix[j,i] = dist_matrix[i,j]
        
        # Condensed form for linkage
        from scipy.spatial.distance import squareform
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='complete')
        clusters = fcluster(Z, t=5.0, criterion='distance')
        n_clusters = len(set(clusters))
        log(f"  Distinct attitude clusters (5° threshold): {n_clusters}")
        
        # Show cluster centers
        for ci in sorted(set(clusters)):
            members = [good_A[i] for i, c in enumerate(clusters) if c == ci]
            att_errs = [m['att_error_deg'] for m in members]
            resids = [m['residual_mag'] for m in members]
            log(f"    Cluster {ci}: {len(members)} members, att_err={np.mean(att_errs):.1f}±{np.std(att_errs):.1f}°, "
                f"resid={np.mean(resids):.4f}±{np.std(resids):.4f}")
    else:
        log(f"  Only {len(rotvecs)} good result(s), skipping clustering")

log(f"\n  CHECKPOINT: Exp A complete at {time.time()-t_start:.0f}s total")

# ═══════════════════════════════════════════════════════════════
# EXP B: Optimizer comparison (50 seeds each)
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("EXP B: Optimizer comparison (50 seeds, 3 methods)")
log(f"{'='*70}")

N_SEEDS_B = 50
seeds_B = Rotation.random(N_SEEDS_B, random_state=456)
methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
results_B = {}

for method in methods:
    log(f"\n  Method: {method}")
    method_results = []
    t_method = time.time()
    
    for si in range(N_SEEDS_B):
        aa_init = seeds_B[si].as_rotvec()
        eval_count[0] = 0
        t_opt = time.time()
        
        try:
            opts = {'maxiter': 50}
            if method == 'L-BFGS-B':
                opts['ftol'] = 1e-6
            elif method == 'Nelder-Mead':
                opts['xatol'] = 1e-4
                opts['fatol'] = 1e-6
            
            res = minimize(brightness_objective, aa_init, args=(0, target_mag_0),
                          method=method, options=opts)
            
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, 0)
            residual = abs(final_mag - target_mag_0)
            
            R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
            R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
            att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
            
            method_results.append({
                'seed': si, 'residual_mag': float(residual),
                'att_error_deg': float(att_err), 'n_evals': eval_count[0],
                'opt_time': time.time() - t_opt, 'success': bool(res.success),
            })
        except Exception as e:
            method_results.append({'seed': si, 'error': str(e)})
        
        if (si+1) % 25 == 0:
            elapsed = time.time() - t_method
            good = sum(1 for r in method_results if r.get('residual_mag', 999) < noise_sigma)
            log(f"    {si+1}/{N_SEEDS_B}: {good} within σ, {elapsed:.1f}s")
    
    results_B[method] = method_results
    good_count = sum(1 for r in method_results if r.get('residual_mag', 999) < noise_sigma)
    avg_evals = np.mean([r.get('n_evals', 0) for r in method_results if 'n_evals' in r])
    avg_time = np.mean([r.get('opt_time', 0) for r in method_results if 'opt_time' in r])
    log(f"  {method}: {good_count}/{N_SEEDS_B} within σ, avg {avg_evals:.0f} evals, avg {avg_time:.2f}s")

log(f"\n  CHECKPOINT: Exp B complete at {time.time()-t_start:.0f}s total")

# ═══════════════════════════════════════════════════════════════
# EXP D: Multiple epochs (use best optimizer from B, 200 seeds each)
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("EXP D: Multi-epoch iso-brightness (200 seeds, 4 epochs)")
log(f"{'='*70}")

# Pick best method from B
best_method = max(methods, key=lambda m: sum(1 for r in results_B[m] if r.get('residual_mag', 999) < noise_sigma))
log(f"Using best method from Exp B: {best_method}")

test_epochs = [0, 25, 50, 75]
N_SEEDS_D = 200
results_D = {}

for ei in test_epochs:
    log(f"\n  Epoch {ei} (target: {observed_lc[ei]:.3f} mag)")
    target = observed_lc[ei]
    seeds_D = Rotation.random(N_SEEDS_D, random_state=789 + ei)
    epoch_results = []
    t_epoch = time.time()
    
    # Get true attitude at this epoch
    true_q_ei = true_quaternions[ei]
    
    for si in range(N_SEEDS_D):
        aa_init = seeds_D[si].as_rotvec()
        eval_count[0] = 0
        
        try:
            opts = {'maxiter': 50}
            if best_method == 'L-BFGS-B':
                opts['ftol'] = 1e-6
            
            res = minimize(brightness_objective, aa_init, args=(ei, target),
                          method=best_method, options=opts)
            
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, ei)
            residual = abs(final_mag - target)
            
            R_true = Rotation.from_quat([true_q_ei[1], true_q_ei[2], true_q_ei[3], true_q_ei[0]])
            R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
            att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
            
            epoch_results.append({
                'seed': si, 'residual_mag': float(residual),
                'att_error_deg': float(att_err),
                'final_rotvec': res.x.tolist(),
                'n_evals': eval_count[0],
            })
        except Exception as e:
            epoch_results.append({'seed': si, 'error': str(e)})
        
        if (si+1) % 50 == 0:
            elapsed = time.time() - t_epoch
            good = sum(1 for r in epoch_results if r.get('residual_mag', 999) < noise_sigma)
            log(f"    {si+1}/{N_SEEDS_D}: {good} within σ, {elapsed:.1f}s")
    
    results_D[ei] = epoch_results
    good = [r for r in epoch_results if r.get('residual_mag', 999) < noise_sigma]
    log(f"  Epoch {ei}: {len(good)}/{N_SEEDS_D} within σ")
    
    if good:
        att_errs = [r['att_error_deg'] for r in good]
        log(f"    Attitude errors: min={min(att_errs):.1f}°, max={max(att_errs):.1f}°, mean={np.mean(att_errs):.1f}°")
        
        # Quick clustering
        if len(good) > 1:
            rotvecs = np.array([r['final_rotvec'] for r in good])
            rotations = Rotation.from_rotvec(rotvecs)
            n = len(rotations)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    dist_matrix[i,j] = np.rad2deg((rotations[i].inv() * rotations[j]).magnitude())
                    dist_matrix[j,i] = dist_matrix[i,j]
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import fcluster, linkage
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='complete')
            clusters = fcluster(Z, t=5.0, criterion='distance')
            n_clusters = len(set(clusters))
            log(f"    Distinct clusters (5°): {n_clusters}")

log(f"\n  CHECKPOINT: Exp D complete at {time.time()-t_start:.0f}s total")

# ─── Save all results ───
log(f"\n{'='*70}")
log("FINAL SUMMARY")
log(f"{'='*70}")
total_time = time.time() - t_start
log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

output = {
    'noise_sigma': noise_sigma,
    'total_time_s': total_time,
    'exp_a': {
        'n_seeds': N_SEEDS_A, 'epoch': 0, 'target_mag': float(target_mag_0),
        'within_1sigma': len(good_A), 'within_2sigma': len(close_A),
        'results': results_A,
    },
    'exp_b': {m: {
        'within_1sigma': sum(1 for r in results_B[m] if r.get('residual_mag', 999) < noise_sigma),
        'n_seeds': N_SEEDS_B,
    } for m in methods},
    'exp_d': {str(ei): {
        'target_mag': float(observed_lc[ei]),
        'within_1sigma': sum(1 for r in results_D[ei] if r.get('residual_mag', 999) < noise_sigma),
        'n_seeds': N_SEEDS_D,
    } for ei in test_epochs},
}

out_path = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_isobrightness.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
log(f"\nSaved to {out_path}")
