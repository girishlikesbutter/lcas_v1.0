#!/usr/bin/env python3
"""
Triplet-matching experiment.

Uses fitted iso-brightness attitudes (not random sampling) at 3 well-separated epochs.
For each epoch, L-BFGS-B from 200 random seeds finds ~200 attitudes matching observed brightness.
Then checks which triplets (q1, q2, q3) are connected by a consistent omega:
  - ω₁₂ implied by (q1, q2) over Δt₁₂
  - ω₂₃ implied by (q2, q3) over Δt₂₃
  - Require ||ω₁₂ - ω₂₃|| < tolerance AND |ω| < periodicity bound
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
    print(msg); sys.stdout.flush()

log("="*70)
log("TRIPLET-MATCHING INVERSION")
log("="*70)

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

# ─── Brightness helpers ───
def brightness_at_quaternion(q_wxyz, epoch_idx):
    q = q_wxyz
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    sun_j2000 = sun_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    obs_j2000 = observer_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    k1_body = R @ sun_j2000; k1_body = k1_body / np.linalg.norm(k1_body)
    k2_body = R @ obs_j2000; k2_body = k2_body / np.linalg.norm(k2_body)
    k1 = k1_body.reshape(1, 3); k2 = k2_body.reshape(1, 3)
    art = {cn: m[0:1] for cn, m in articulation_matrices.items()}
    lit = compute_shadows(satellite=satellite, k1_vectors=k1, explicit_component_matrices=art, show_progress=False)
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

def brightness_objective(aa_params, epoch_idx, target_mag):
    q = axis_angle_to_quat(aa_params)
    pred = brightness_at_quaternion(q, epoch_idx)
    return (pred - target_mag)**2

# ─── Omega bound from FFT (no margin) ───
from scipy.fft import fft, fftfreq
dt_obs = observation_times[1] - observation_times[0]
yf = fft(observed_lc - observed_lc.mean())
xf = fftfreq(n_observations, dt_obs)
positive = xf > 0
power = np.abs(yf[positive])**2
peak_freq = xf[positive][np.argmax(power)]
estimated_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
omega_upper = 2 * np.pi / estimated_period if estimated_period < np.inf else 1.0
log(f"Omega upper bound (FFT, no margin): {np.rad2deg(omega_upper):.4f}°/s")
log(f"True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f}°/s")

# ─── Step 1: Find iso-brightness attitudes at 3 epochs ───
TRIPLET_EPOCHS = [0, 33, 66]  # well-separated
N_SEEDS = 200
CLUSTER_THRESH = 5.0  # degrees

attitude_sets = {}  # epoch -> list of (rotvec, quaternion_wxyz, residual)

for ei in TRIPLET_EPOCHS:
    log(f"\n{'='*70}")
    log(f"FINDING ISO-BRIGHTNESS ATTITUDES AT EPOCH {ei} (t={observation_times[ei]:.0f}s)")
    log(f"{'='*70}")
    
    target = observed_lc[ei]
    true_q_ei = true_quaternions[ei]
    log(f"Target: {target:.4f} mag")
    
    seeds = Rotation.random(N_SEEDS, random_state=100 + ei)
    found = []
    t_ep = time.time()
    
    for si in range(N_SEEDS):
        aa_init = seeds[si].as_rotvec()
        try:
            res = minimize(brightness_objective, aa_init, args=(ei, target),
                          method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, ei)
            residual = abs(final_mag - target)
            
            if residual < noise_sigma:  # within 1σ
                R_true = Rotation.from_quat([true_q_ei[1], true_q_ei[2], true_q_ei[3], true_q_ei[0]])
                R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
                att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
                found.append({
                    'rotvec': res.x.tolist(),
                    'q_wxyz': q_final.tolist(),
                    'residual': float(residual),
                    'att_error_deg': float(att_err),
                })
        except:
            pass
        
        if (si+1) % 50 == 0:
            log(f"  {si+1}/{N_SEEDS}: {len(found)} within σ, {time.time()-t_ep:.1f}s")
    
    # Cluster to get unique attitudes
    if len(found) > 1:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        rotvecs = np.array([f['rotvec'] for f in found])
        rotations = Rotation.from_rotvec(rotvecs)
        n = len(rotations)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.rad2deg((rotations[i].inv() * rotations[j]).magnitude())
                dist[i,j] = d; dist[j,i] = d
        Z = linkage(squareform(dist), method='complete')
        clusters = fcluster(Z, t=CLUSTER_THRESH, criterion='distance')
        
        # Pick best (lowest residual) from each cluster
        unique = []
        for ci in sorted(set(clusters)):
            members = [found[i] for i, c in enumerate(clusters) if c == ci]
            best = min(members, key=lambda x: x['residual'])
            unique.append(best)
        
        attitude_sets[ei] = unique
        att_errs = [u['att_error_deg'] for u in unique]
        log(f"  Epoch {ei}: {len(found)} fits → {len(unique)} unique clusters")
        log(f"  Attitude errors: min={min(att_errs):.1f}°, max={max(att_errs):.1f}°")
    else:
        attitude_sets[ei] = found
        log(f"  Epoch {ei}: {len(found)} fits")

# ─── Step 2: Triplet matching ───
log(f"\n{'='*70}")
log("TRIPLET MATCHING")
log(f"{'='*70}")

e1, e2, e3 = TRIPLET_EPOCHS
dt12 = observation_times[e2] - observation_times[e1]
dt23 = observation_times[e3] - observation_times[e2]

set1 = attitude_sets.get(e1, [])
set2 = attitude_sets.get(e2, [])
set3 = attitude_sets.get(e3, [])

log(f"Epoch {e1}: {len(set1)} candidates")
log(f"Epoch {e2}: {len(set2)} candidates")
log(f"Epoch {e3}: {len(set3)} candidates")
log(f"Δt₁₂ = {dt12:.0f}s, Δt₂₃ = {dt23:.0f}s")
log(f"Total triplets to check: {len(set1)} × {len(set2)} × {len(set3)} = {len(set1)*len(set2)*len(set3)}")

# Omega consistency tolerance (deg/s)
# For a slow tumbler, omega should be nearly constant over the observation window
OMEGA_CONSISTENCY_TOL_DEG = 0.05  # deg/s tolerance on ||ω₁₂ - ω₂₃||

feasible_triplets = []
t_match = time.time()

for i, a1 in enumerate(set1):
    R1 = Rotation.from_quat([a1['q_wxyz'][1], a1['q_wxyz'][2], a1['q_wxyz'][3], a1['q_wxyz'][0]])
    
    for j, a2 in enumerate(set2):
        R2 = Rotation.from_quat([a2['q_wxyz'][1], a2['q_wxyz'][2], a2['q_wxyz'][3], a2['q_wxyz'][0]])
        
        # ω₁₂
        R_12 = R2 * R1.inv()
        omega_12 = R_12.as_rotvec() / dt12
        omega_12_mag = np.linalg.norm(omega_12)
        
        # Check magnitude bound
        if omega_12_mag > omega_upper:
            continue
        
        for k, a3 in enumerate(set3):
            R3 = Rotation.from_quat([a3['q_wxyz'][1], a3['q_wxyz'][2], a3['q_wxyz'][3], a3['q_wxyz'][0]])
            
            # ω₂₃
            R_23 = R3 * R2.inv()
            omega_23 = R_23.as_rotvec() / dt23
            omega_23_mag = np.linalg.norm(omega_23)
            
            if omega_23_mag > omega_upper:
                continue
            
            # Consistency check
            omega_diff = np.rad2deg(np.linalg.norm(omega_12 - omega_23))
            
            if omega_diff < OMEGA_CONSISTENCY_TOL_DEG:
                avg_omega = (omega_12 + omega_23) / 2
                
                feasible_triplets.append({
                    'i': i, 'j': j, 'k': k,
                    'omega_12_deg_s': np.rad2deg(omega_12).tolist(),
                    'omega_23_deg_s': np.rad2deg(omega_23).tolist(),
                    'avg_omega_deg_s': np.rad2deg(avg_omega).tolist(),
                    'omega_diff_deg_s': float(omega_diff),
                    'omega_12_mag_deg_s': float(np.rad2deg(omega_12_mag)),
                    'omega_23_mag_deg_s': float(np.rad2deg(omega_23_mag)),
                    'att_err_e1': a1['att_error_deg'],
                    'att_err_e2': a2['att_error_deg'],
                    'att_err_e3': a3['att_error_deg'],
                    'q1': a1['q_wxyz'],
                    'avg_omega_rad_s': avg_omega.tolist(),
                })

match_time = time.time() - t_match
log(f"Triplet matching: {match_time:.3f}s")
log(f"Feasible triplets: {len(feasible_triplets)}")

if feasible_triplets:
    # Sort by omega consistency
    feasible_triplets.sort(key=lambda x: x['omega_diff_deg_s'])
    
    log(f"\nTOP 10 BY OMEGA CONSISTENCY:")
    for ti, t in enumerate(feasible_triplets[:10]):
        log(f"  #{ti}: ω_diff={t['omega_diff_deg_s']:.6f}°/s, "
            f"att_err=[{t['att_err_e1']:.1f}°, {t['att_err_e2']:.1f}°, {t['att_err_e3']:.1f}°], "
            f"|ω₁₂|={t['omega_12_mag_deg_s']:.4f}°/s, |ω₂₃|={t['omega_23_mag_deg_s']:.4f}°/s")
    
    # ─── Step 3: Validate best triplets against full lightcurve ───
    log(f"\n{'='*70}")
    log("VALIDATION: Propagate best triplets against full lightcurve")
    log(f"{'='*70}")
    
    TOP_N = min(20, len(feasible_triplets))
    val_epochs = list(range(0, 100, 5))  # every 5th epoch
    
    val_results = []
    for ti, triplet in enumerate(feasible_triplets[:TOP_N]):
        q0_cand = np.array(triplet['q1'])
        omega_cand = np.array(triplet['avg_omega_rad_s'])
        
        try:
            val_times = observation_times[val_epochs]
            prop_quats, _ = propagate_attitude(
                q0=q0_cand, omega0=omega_cand,
                times=np.concatenate([[0], val_times]),
                mode="tumbling", inertia_tensor=inertia_tensor)
            
            val_quats = prop_quats[1:]
            val_mags = []
            for vi, ve in enumerate(val_epochs):
                m = brightness_at_quaternion(val_quats[vi], ve)
                val_mags.append(m)
            val_mags = np.array(val_mags)
            
            val_residuals = np.abs(val_mags - observed_lc[val_epochs])
            mean_resid = val_residuals.mean()
            
            R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
            R_cand = Rotation.from_quat([q0_cand[1], q0_cand[2], q0_cand[3], q0_cand[0]])
            att_err = np.rad2deg((R_cand.inv() * R_true).magnitude())
            omega_err = np.rad2deg(np.linalg.norm(omega_cand - true_omega0))
            
            val_results.append({
                'triplet_idx': ti,
                'att_error_deg': float(att_err),
                'omega_error_deg_s': float(omega_err),
                'mean_val_residual': float(mean_resid),
                'omega_diff_deg_s': triplet['omega_diff_deg_s'],
            })
            
            log(f"  Triplet {ti}: att_err={att_err:.1f}°, ω_err={omega_err:.4f}°/s, "
                f"mean_resid={mean_resid:.3f} mag, ω_consistency={triplet['omega_diff_deg_s']:.6f}°/s")
        except Exception as e:
            log(f"  Triplet {ti}: FAILED - {e}")
            val_results.append({'triplet_idx': ti, 'error': str(e)})
    
    valid = [r for r in val_results if 'mean_val_residual' in r]
    if valid:
        by_resid = sorted(valid, key=lambda x: x['mean_val_residual'])
        by_att = sorted(valid, key=lambda x: x['att_error_deg'])
        
        log(f"\nBEST BY VALIDATION RESIDUAL:")
        for r in by_resid[:3]:
            log(f"  att_err={r['att_error_deg']:.1f}°, ω_err={r['omega_error_deg_s']:.4f}°/s, resid={r['mean_val_residual']:.3f}")
        
        log(f"\nBEST BY ATTITUDE ERROR:")
        for r in by_att[:3]:
            log(f"  att_err={r['att_error_deg']:.1f}°, ω_err={r['omega_error_deg_s']:.4f}°/s, resid={r['mean_val_residual']:.3f}")
else:
    log("\nNO FEASIBLE TRIPLETS FOUND.")
    log("Trying with looser omega consistency tolerance...")
    
    # Try progressively looser tolerances
    for tol in [0.1, 0.2, 0.5, 1.0]:
        count = 0
        for i, a1 in enumerate(set1):
            R1 = Rotation.from_quat([a1['q_wxyz'][1], a1['q_wxyz'][2], a1['q_wxyz'][3], a1['q_wxyz'][0]])
            for j, a2 in enumerate(set2):
                R2 = Rotation.from_quat([a2['q_wxyz'][1], a2['q_wxyz'][2], a2['q_wxyz'][3], a2['q_wxyz'][0]])
                R_12 = R2 * R1.inv()
                omega_12 = R_12.as_rotvec() / dt12
                if np.linalg.norm(omega_12) > omega_upper:
                    continue
                for k, a3 in enumerate(set3):
                    R3 = Rotation.from_quat([a3['q_wxyz'][1], a3['q_wxyz'][2], a3['q_wxyz'][3], a3['q_wxyz'][0]])
                    R_23 = R3 * R2.inv()
                    omega_23 = R_23.as_rotvec() / dt23
                    if np.linalg.norm(omega_23) > omega_upper:
                        continue
                    omega_diff = np.rad2deg(np.linalg.norm(omega_12 - omega_23))
                    if omega_diff < tol:
                        count += 1
        log(f"  Tolerance {tol}°/s: {count} triplets")
    
    val_results = []

# ─── Summary ───
log(f"\n{'='*70}")
log("SUMMARY")
log(f"{'='*70}")
total_time = time.time() - t_start
log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
log(f"Epochs: {TRIPLET_EPOCHS}")
log(f"Attitudes per epoch: {[len(attitude_sets.get(e, [])) for e in TRIPLET_EPOCHS]}")
log(f"Feasible triplets: {len(feasible_triplets)}")
if feasible_triplets and valid:
    best = by_resid[0]
    log(f"Best by validation: att_err={best['att_error_deg']:.1f}°, ω_err={best['omega_error_deg_s']:.4f}°/s")
    best_att = by_att[0]
    log(f"Best by attitude: att_err={best_att['att_error_deg']:.1f}°, ω_err={best_att['omega_error_deg_s']:.4f}°/s")

output = {
    'triplet_epochs': TRIPLET_EPOCHS,
    'n_seeds': N_SEEDS,
    'omega_upper_deg_s': float(np.rad2deg(omega_upper)),
    'omega_consistency_tol_deg_s': OMEGA_CONSISTENCY_TOL_DEG,
    'attitudes_per_epoch': {str(e): len(attitude_sets.get(e, [])) for e in TRIPLET_EPOCHS},
    'feasible_triplets': len(feasible_triplets),
    'total_time_s': total_time,
    'results': val_results if feasible_triplets else [],
}

out_path = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_triplet_matching.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
log(f"\nSaved to {out_path}")
