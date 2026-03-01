#!/usr/bin/env python3
"""
Mixed-fidelity iso-brightness search v2:
1. Dense lo-fi grid: 100k attitudes, evaluate brightness (no optimizer)
2. Pick ~200 closest to target brightness
3. Hi-fi L-BFGS-B refinement, parallelized (8 workers)
4. Check if true attitude appears in final set

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
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

def log(msg):
    print(msg, flush=True)

t_start = time.time()

# ─── Setup (done once in main process) ───
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

log(f"Setup: {time.time()-t_start:.1f}s")
log(f"CPU cores available: {cpu_count()}")

# ─── Brightness evaluation ───
def brightness_at_quaternion(q_wxyz, epoch_idx, use_shadows=True):
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

def brightness_objective_hifi(aa_params, epoch_idx, target_mag):
    q = axis_angle_to_quat(aa_params)
    pred = brightness_at_quaternion(q, epoch_idx, use_shadows=True)
    return (pred - target_mag)**2

# Worker function for parallel hi-fi refinement
def hifi_refine_worker(args):
    """Refine one seed with hi-fi L-BFGS-B. Returns (rotvec, att_err, residual) or None."""
    aa_init, epoch_idx, target_mag, true_q_wxyz = args
    try:
        res = minimize(brightness_objective_hifi, aa_init, args=(epoch_idx, target_mag),
                      method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
        q_final = axis_angle_to_quat(res.x)
        final_mag = brightness_at_quaternion(q_final, epoch_idx, use_shadows=True)
        residual = abs(final_mag - target_mag)
        R_true = Rotation.from_quat([true_q_wxyz[1], true_q_wxyz[2], true_q_wxyz[3], true_q_wxyz[0]])
        R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
        att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return {'rotvec': res.x.tolist(), 'att_err': float(att_err), 'residual': float(residual)}
    except Exception as e:
        return {'rotvec': aa_init.tolist(), 'att_err': 999.0, 'residual': 999.0, 'error': str(e)}

def mixed_fidelity_search(true_q_epoch, epoch_idx, n_grid=100000, n_hifi_seeds=200, label=""):
    """
    Stage 1: Dense lo-fi grid eval (no optimizer)
    Stage 2: Pick closest n_hifi_seeds to target
    Stage 3: Parallel hi-fi L-BFGS-B refinement
    """
    R_true = Rotation.from_quat([true_q_epoch[1], true_q_epoch[2], true_q_epoch[3], true_q_epoch[0]])
    
    # True brightness (hi-fi) + noise
    true_mag_hifi = brightness_at_quaternion(true_q_epoch, epoch_idx, use_shadows=True)
    target_mag = true_mag_hifi + np.random.normal(0, noise_sigma)
    
    # Lo-fi target (use lo-fi brightness of true attitude + same noise offset)
    true_mag_lofi = brightness_at_quaternion(true_q_epoch, epoch_idx, use_shadows=False)
    lofi_offset = true_mag_lofi - true_mag_hifi  # systematic lo-fi bias
    target_mag_lofi = target_mag + lofi_offset  # shift target to lo-fi space
    
    log(f"  {label} Hi-fi target: {target_mag:.4f} mag, Lo-fi target: {target_mag_lofi:.4f} mag (bias: {lofi_offset:.4f})")
    
    # ── Stage 1: Dense lo-fi grid evaluation ──
    log(f"  Stage 1: Evaluating {n_grid} random attitudes in lo-fi...")
    t_lofi = time.time()
    
    grid_rots = Rotation.random(n_grid, random_state=42)
    grid_mags = np.zeros(n_grid)
    
    for i in range(n_grid):
        q_scipy = grid_rots[i].as_quat()  # x,y,z,w
        q_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        grid_mags[i] = brightness_at_quaternion(q_wxyz, epoch_idx, use_shadows=False)
        
        if (i+1) % 10000 == 0:
            elapsed = time.time() - t_lofi
            rate = (i+1) / elapsed
            eta = (n_grid - i - 1) / rate
            log(f"    {i+1}/{n_grid} ({elapsed:.0f}s elapsed, {rate:.0f}/s, ETA {eta:.0f}s)")
    
    dt_lofi = time.time() - t_lofi
    log(f"  Stage 1 done: {dt_lofi:.0f}s ({n_grid/dt_lofi:.0f} evals/s)")
    
    # ── Stage 2: Select closest to target ──
    residuals_lofi = np.abs(grid_mags - target_mag_lofi)
    sorted_idx = np.argsort(residuals_lofi)
    
    # Take top n_hifi_seeds
    selected_idx = sorted_idx[:n_hifi_seeds]
    selected_residuals = residuals_lofi[selected_idx]
    
    # Check how many are within sigma
    n_within_sigma = np.sum(residuals_lofi < noise_sigma)
    log(f"  Stage 2: {n_within_sigma} grid points within σ={noise_sigma} of lo-fi target")
    log(f"  Selected {n_hifi_seeds} closest: residual range [{selected_residuals[0]:.4f}, {selected_residuals[-1]:.4f}] mag")
    
    # Compute attitude errors of selected lo-fi candidates
    selected_att_errs = []
    for idx in selected_idx:
        R_found = grid_rots[idx]
        att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
        selected_att_errs.append(att_err)
    
    min_lofi_att_err = min(selected_att_errs)
    log(f"  Lo-fi candidate attitude errors: min={min_lofi_att_err:.2f}°, median={np.median(selected_att_errs):.1f}°")
    
    # ── Stage 3: Parallel hi-fi refinement ──
    log(f"  Stage 3: Hi-fi L-BFGS-B refinement of {n_hifi_seeds} seeds (8 parallel workers)...")
    t_hifi = time.time()
    
    # Prepare worker args
    worker_args = []
    for idx in selected_idx:
        aa_init = grid_rots[idx].as_rotvec()
        worker_args.append((aa_init, epoch_idx, target_mag, true_q_epoch))
    
    with Pool(8) as pool:
        hifi_results = pool.map(hifi_refine_worker, worker_args)
    
    dt_hifi = time.time() - t_hifi
    
    good_hifi = [r for r in hifi_results if r['residual'] < noise_sigma]
    hifi_errs = sorted([r['att_err'] for r in good_hifi]) if good_hifi else [999]
    min_hifi_err = min(hifi_errs)
    
    log(f"  Stage 3 done: {len(good_hifi)}/{n_hifi_seeds} within σ, {dt_hifi:.0f}s")
    log(f"  Hi-fi min attitude error: {min_hifi_err:.2f}°")
    log(f"  Top-5 closest: {[f'{e:.1f}°' for e in hifi_errs[:5]]}")
    log(f"  Truth recovered (<1°): {'YES ✓' if min_hifi_err < 1.0 else 'NO ✗'}")
    log(f"  Truth recovered (<5°): {'YES ✓' if min_hifi_err < 5.0 else 'NO ✗'}")
    log(f"  Total time: {dt_lofi + dt_hifi:.0f}s (lofi={dt_lofi:.0f}s, hifi={dt_hifi:.0f}s)")
    
    return {
        'min_att_err': float(min_hifi_err),
        'min_lofi_att_err': float(min_lofi_att_err),
        'n_within_sigma_lofi': int(n_within_sigma),
        'n_hifi_good': len(good_hifi),
        'dt_lofi': dt_lofi,
        'dt_hifi': dt_hifi,
        'hifi_top5': hifi_errs[:5],
        'lofi_top5_att_err': sorted(selected_att_errs)[:5]
    }

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Original true attitude
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log(f"\n{'='*70}")
    log("PHASE 1: Original true attitude, 100k lo-fi grid → 200 hi-fi seeds")
    log(f"{'='*70}")
    
    true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
    true_angle_rad = np.deg2rad(45.0)
    true_q0_orig = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
    
    np.random.seed(42)
    p1 = mixed_fidelity_search(true_q0_orig, epoch_idx=0, n_grid=100000, n_hifi_seeds=200, label="Phase1")
    
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
        q_scipy = R_true_i.as_quat()
        q_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        
        log(f"\n--- Trial {ti+1}/{N_TRIALS} ---")
        np.random.seed(42 + ti)
        result = mixed_fidelity_search(q_wxyz, epoch_idx=0, n_grid=100000, n_hifi_seeds=200, label=f"T{ti+1}")
        phase2_results.append(result)
    
    # ── Summary ──
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"Phase 1: min_err={p1['min_att_err']:.2f}°, "
        f"recovered(<1°): {'YES' if p1['min_att_err']<1 else 'NO'}, "
        f"recovered(<5°): {'YES' if p1['min_att_err']<5 else 'NO'}")
    
    n_rec_1 = sum(1 for r in phase2_results if r['min_att_err'] < 1.0)
    n_rec_5 = sum(1 for r in phase2_results if r['min_att_err'] < 5.0)
    min_errs = [r['min_att_err'] for r in phase2_results]
    log(f"\nPhase 2 ({N_TRIALS} trials):")
    log(f"  Recovered <1°: {n_rec_1}/{N_TRIALS}")
    log(f"  Recovered <5°: {n_rec_5}/{N_TRIALS}")
    for ti, r in enumerate(phase2_results):
        log(f"  Trial {ti+1}: min_err={r['min_att_err']:.1f}° (lofi={r['min_lofi_att_err']:.1f}°, "
            f"lofi_σ={r['n_within_sigma_lofi']}, hifi_good={r['n_hifi_good']}, "
            f"time={r['dt_lofi']+r['dt_hifi']:.0f}s)")
    log(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Save
    out = {
        'phase1': p1,
        'phase2': phase2_results,
        'summary': {'recovered_1deg': n_rec_1, 'recovered_5deg': n_rec_5, 'n_trials': N_TRIALS}
    }
    outpath = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_mixed_fidelity_search_v2.json"
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Saved: {outpath}")
