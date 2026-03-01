#!/usr/bin/env python3
"""
Robustness test: For 10 random true attitudes, do 10k lo-fi seeds.
Check if bottom 200 by residual contains truth (<5°).
"""
import sys, time, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.articulation import compute_rotation_matrices_from_angles

def log(msg):
    print(msg, flush=True)

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
metakernel_path = config_manager.get_metakernel_path(config)
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, 100)
geometry_data = compute_observation_geometry(epochs=epochs, satellite_id=config.spice_config.satellite_id, observer_id=399999, spice_handler=spice_handler, config=config)
sun_pos = geometry_data['sun_positions']
obs_pos = geometry_data['obs_positions']
sat_pos = geometry_data['sat_positions']
obs_dist = geometry_data['observer_distances']
art_matrices = compute_rotation_matrices_from_angles({'SP_North': np.full(100,0.), 'SP_South': np.full(100,0.), 'AD_East': np.full(100,15.), 'AD_West': np.full(100,15.)}, satellite)

def brightness_eval(q_wxyz, epoch_idx, use_shadows):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (sun_pos[epoch_idx] - sat_pos[epoch_idx]); s /= np.linalg.norm(s)
    o = R @ (obs_pos[epoch_idx] - sat_pos[epoch_idx]); o /= np.linalg.norm(o)
    a = {c: m[0:1] for c, m in art_matrices.items()}
    if use_shadows:
        lit = compute_shadows(satellite=satellite, k1_vectors=s.reshape(1,3), explicit_component_matrices=a, show_progress=False)
    else:
        lit = create_no_shadow_lit_status(satellite, 1)
    m, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3), k2_vectors_array=o.reshape(1,3), observer_distances=np.array([obs_dist[epoch_idx]]), satellite=satellite, epochs=np.array([0.]), pre_computed_matrices=a, generate_no_shadow=False, animate=False, show_progress=False)
    return m[0]

def aa2q(aa):
    a = np.linalg.norm(aa)
    if a < 1e-12: return np.array([1.,0.,0.,0.])
    ax = aa/a
    return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])

# Global state set per trial (used by worker)
_target_lofi = None
_true_q = None

def init_worker(target_lofi, true_q):
    global _target_lofi, _true_q
    _target_lofi = target_lofi
    _true_q = true_q

def lofi_worker(aa_init):
    try:
        def obj(aa):
            q = aa2q(aa)
            return (brightness_eval(q, 0, use_shadows=False) - _target_lofi)**2
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter':50,'ftol':1e-12})
        q = aa2q(res.x)
        mag = brightness_eval(q, 0, use_shadows=False)
        resid = abs(mag - _target_lofi)
        R_true = Rotation.from_quat([_true_q[1], _true_q[2], _true_q[3], _true_q[0]])
        R_found = Rotation.from_quat([q[1],q[2],q[3],q[0]])
        err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return (float(resid), float(err))
    except:
        return (999., 999.)

if __name__ == '__main__':
    N_TRIALS = 10
    N_SEEDS = 10000
    
    true_attitudes = Rotation.random(N_TRIALS, random_state=777)
    trial_results = []
    
    log(f"{'='*70}")
    log(f"ROBUSTNESS TEST: 10 random true attitudes, {N_SEEDS} lo-fi seeds each")
    log(f"{'='*70}\n")
    
    t_total = time.time()
    
    for ti in range(N_TRIALS):
        q_scipy = true_attitudes[ti].as_quat()  # x,y,z,w
        true_q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        
        # Generate target
        np.random.seed(42 + ti)
        mag_hifi = brightness_eval(true_q, 0, use_shadows=True)
        mag_lofi = brightness_eval(true_q, 0, use_shadows=False)
        target_hifi = mag_hifi + np.random.normal(0, 0.05)
        lofi_bias = mag_lofi - mag_hifi
        target_lofi = target_hifi + lofi_bias
        
        seeds = Rotation.random(N_SEEDS, random_state=1000 + ti)
        args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]
        
        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_lofi, true_q)) as pool:
            results = pool.map(lofi_worker, args)
        dt = time.time() - t0
        
        # Sort by residual
        good = [(r, e) for r, e in results if r < 0.05]
        good_sorted = sorted(good, key=lambda x: x[0])
        
        # Check buckets
        buckets = {}
        for topk in [100, 200, 500, 1000]:
            bucket = good_sorted[:topk]
            bucket_errs = [e for _, e in bucket]
            min_err = min(bucket_errs) if bucket_errs else 999
            n5 = sum(1 for e in bucket_errs if e < 5)
            n10 = sum(1 for e in bucket_errs if e < 10)
            buckets[topk] = {'min_err': min_err, 'n_within_5': n5, 'n_within_10': n10}
        
        # Overall stats
        all_errs = sorted([e for _, e in good])
        min_err_all = all_errs[0] if all_errs else 999
        
        result = {
            'trial': ti,
            'n_good': len(good),
            'min_err_all': float(min_err_all),
            'buckets': buckets,
            'time': dt,
            'lofi_bias': float(lofi_bias)
        }
        trial_results.append(result)
        
        b200 = buckets[200]
        status = "YES ✓" if b200['min_err'] < 5.0 else f"NO ✗ ({b200['min_err']:.1f}°)"
        log(f"Trial {ti+1:2d}/10: {len(good)}/{N_SEEDS} converged | "
            f"min_err_all={min_err_all:.1f}° | "
            f"bottom200: min={b200['min_err']:.1f}° n<5°={b200['n_within_5']} | "
            f"truth_in_200: {status} | {dt:.0f}s")
    
    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    
    n_success_200 = sum(1 for r in trial_results if r['buckets'][200]['min_err'] < 5.0)
    n_success_500 = sum(1 for r in trial_results if r['buckets'][500]['min_err'] < 5.0)
    n_success_1000 = sum(1 for r in trial_results if r['buckets'][1000]['min_err'] < 5.0)
    n_success_all = sum(1 for r in trial_results if r['min_err_all'] < 5.0)
    
    log(f"Truth (<5°) in bottom 100:  {sum(1 for r in trial_results if r['buckets'][100]['min_err'] < 5.0)}/10")
    log(f"Truth (<5°) in bottom 200:  {n_success_200}/10")
    log(f"Truth (<5°) in bottom 500:  {n_success_500}/10")
    log(f"Truth (<5°) in bottom 1000: {n_success_1000}/10")
    log(f"Truth (<5°) in ALL results: {n_success_all}/10")
    
    log(f"\nPer-trial min errors (all 10k):")
    for r in trial_results:
        log(f"  Trial {r['trial']+1}: {r['min_err_all']:.2f}°")
    
    log(f"\nTotal time: {time.time()-t_total:.0f}s")
    
    out = {'trials': trial_results, 'summary': {
        'n_trials': N_TRIALS, 'n_seeds': N_SEEDS,
        'success_bottom200': n_success_200,
        'success_bottom500': n_success_500,
        'success_all': n_success_all
    }}
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_residual_robustness.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
