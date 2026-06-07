#!/usr/bin/env python3
"""
10k lo-fi seeds, save residual AND attitude error for every result.
Answer: if we take bottom 200 by residual, is truth in there?
"""
import sys, time, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import create_no_shadow_lit_status
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

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])

def brightness_lofi(q_wxyz):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (sun_pos[0] - sat_pos[0]); s /= np.linalg.norm(s)
    o = R @ (obs_pos[0] - sat_pos[0]); o /= np.linalg.norm(o)
    a = {c: m[0:1] for c, m in art_matrices.items()}
    lit = create_no_shadow_lit_status(satellite, 1)
    m, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3), k2_vectors_array=o.reshape(1,3), observer_distances=np.array([obs_dist[0]]), satellite=satellite, epochs=np.array([0.]), pre_computed_matrices=a, generate_no_shadow=False, animate=False, show_progress=False)
    return m[0]

def aa2q(aa):
    a = np.linalg.norm(aa)
    if a < 1e-12: return np.array([1.,0.,0.,0.])
    ax = aa/a
    return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])

np.random.seed(42)
true_mag_lofi = brightness_lofi(true_q0)
# Use same target as the 10k run
from src.computation.shadow_engine import compute_shadows
def brightness_hifi(q_wxyz):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (sun_pos[0] - sat_pos[0]); s /= np.linalg.norm(s)
    o = R @ (obs_pos[0] - sat_pos[0]); o /= np.linalg.norm(o)
    a = {c: m[0:1] for c, m in art_matrices.items()}
    lit = compute_shadows(satellite=satellite, k1_vectors=s.reshape(1,3), explicit_component_matrices=a, show_progress=False)
    m, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3), k2_vectors_array=o.reshape(1,3), observer_distances=np.array([obs_dist[0]]), satellite=satellite, epochs=np.array([0.]), pre_computed_matrices=a, generate_no_shadow=False, animate=False, show_progress=False)
    return m[0]

true_mag_hifi = brightness_hifi(true_q0)
target_mag_hifi = true_mag_hifi + np.random.normal(0, 0.05)
lofi_bias = true_mag_lofi - true_mag_hifi
target_mag_lofi = target_mag_hifi + lofi_bias

log(f"Hi-fi target: {target_mag_hifi:.6f}, Lo-fi target: {target_mag_lofi:.6f}")
log(f"True lo-fi brightness: {true_mag_lofi:.6f}, residual from target: {abs(true_mag_lofi - target_mag_lofi):.6f}")

def worker(aa_init):
    try:
        def obj(aa):
            q = aa2q(aa)
            return (brightness_lofi(q) - target_mag_lofi)**2
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter':50,'ftol':1e-12})
        q = aa2q(res.x)
        mag = brightness_lofi(q)
        resid = abs(mag - target_mag_lofi)
        Rf = Rotation.from_quat([q[1],q[2],q[3],q[0]])
        err = np.rad2deg((Rf.inv() * R_true).magnitude())
        return {'resid': float(resid), 'att_err': float(err), 'fval': float(res.fun), 'nfev': int(res.nfev)}
    except Exception as e:
        return {'resid': 999., 'att_err': 999., 'error': str(e)}

if __name__ == '__main__':
    N = 10000
    log(f"\nRunning {N} lo-fi optimizations, 8 workers...")
    seeds = Rotation.random(N, random_state=42)
    args = [seeds[i].as_rotvec() for i in range(N)]
    
    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(worker, args)
    dt = time.time() - t0
    log(f"Time: {dt:.1f}s")
    
    good = [r for r in results if r['resid'] < 0.05]
    log(f"Converged: {len(good)}/{N}")
    
    # Sort by residual
    good_sorted = sorted(good, key=lambda r: r['resid'])
    
    resids = np.array([r['resid'] for r in good_sorted])
    errs = np.array([r['att_err'] for r in good_sorted])
    fvals = np.array([r['fval'] for r in good_sorted])
    
    log(f"\nResidual distribution:")
    log(f"  min:    {resids.min():.2e}")
    log(f"  1%:     {np.percentile(resids, 1):.2e}")
    log(f"  5%:     {np.percentile(resids, 5):.2e}")
    log(f"  10%:    {np.percentile(resids, 10):.2e}")
    log(f"  25%:    {np.percentile(resids, 25):.2e}")
    log(f"  median: {np.percentile(resids, 50):.2e}")
    log(f"  75%:    {np.percentile(resids, 75):.2e}")
    log(f"  max:    {resids.max():.2e}")
    
    log(f"\nfval (objective = residual²) distribution:")
    log(f"  min:    {fvals.min():.2e}")
    log(f"  1%:     {np.percentile(fvals, 1):.2e}")
    log(f"  median: {np.percentile(fvals, 50):.2e}")
    log(f"  max:    {fvals.max():.2e}")
    
    # Key question: bottom 200 by residual
    bottom200 = good_sorted[:200]
    bottom200_errs = [r['att_err'] for r in bottom200]
    min_err_200 = min(bottom200_errs)
    n_within5_200 = sum(1 for e in bottom200_errs if e < 5)
    n_within10_200 = sum(1 for e in bottom200_errs if e < 10)
    
    log(f"\n{'='*70}")
    log(f"BOTTOM 200 BY RESIDUAL:")
    log(f"  Residual range: [{bottom200[0]['resid']:.2e}, {bottom200[-1]['resid']:.2e}]")
    log(f"  Min att error: {min_err_200:.2f}°")
    log(f"  Within 5°: {n_within5_200}")
    log(f"  Within 10°: {n_within10_200}")
    log(f"  TRUE ATTITUDE IN BOTTOM 200? {'YES ✓' if min_err_200 < 5.0 else 'NO ✗'}")
    
    # Also check bottom 500, 1000
    for topk in [100, 200, 500, 1000, 2000]:
        topk_errs = [r['att_err'] for r in good_sorted[:topk]]
        log(f"  Bottom {topk:5d}: min_err={min(topk_errs):.2f}°, within_5°={sum(1 for e in topk_errs if e<5)}, within_10°={sum(1 for e in topk_errs if e<10)}")
    
    # Correlation on log scale
    from scipy.stats import spearmanr
    sp, sp_p = spearmanr(resids, errs)
    log_resids = np.log10(resids + 1e-20)
    sp_log, sp_log_p = spearmanr(log_resids, errs)
    log(f"\nSpearman corr(residual, att_err): {sp:.3f} (p={sp_p:.2e})")
    log(f"Spearman corr(log10_residual, att_err): {sp_log:.3f} (p={sp_log_p:.2e})")
    
    # Show some examples from bottom-10
    log(f"\nTop-10 by lowest residual:")
    for r in good_sorted[:10]:
        log(f"  resid={r['resid']:.2e}  fval={r['fval']:.2e}  att_err={r['att_err']:.1f}°")
    
    log(f"\nTop-10 by lowest att_err:")
    by_err = sorted(good, key=lambda r: r['att_err'])
    for r in by_err[:10]:
        log(f"  resid={r['resid']:.2e}  fval={r['fval']:.2e}  att_err={r['att_err']:.1f}°")
    
    # Save full results
    out = {
        'n_seeds': N,
        'n_good': len(good),
        'dt': dt,
        'residual_percentiles': {
            'min': float(resids.min()), 'p1': float(np.percentile(resids,1)),
            'p5': float(np.percentile(resids,5)), 'p10': float(np.percentile(resids,10)),
            'p25': float(np.percentile(resids,25)), 'median': float(np.median(resids)),
            'p75': float(np.percentile(resids,75)), 'max': float(resids.max())
        },
        'bottom200_min_err': float(min_err_200),
        'bottom200_n_within5': n_within5_200,
        'spearman': float(sp), 'spearman_log': float(sp_log),
        'all_results': [{'resid': r['resid'], 'att_err': r['att_err']} for r in good_sorted]
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_residual_vs_error.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved: {outpath}")
