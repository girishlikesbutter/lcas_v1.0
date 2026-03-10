#!/usr/bin/env python3
"""Quick check: does lo-fi residual correlate with attitude error?"""
import sys, time, numpy as np
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
true_q0 = np.array([np.cos(np.deg2rad(22.5)), *(np.sin(np.deg2rad(22.5)) * true_axis)])

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

R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
np.random.seed(42)
target = brightness_lofi(true_q0) + np.random.normal(0, 0.05)

def worker(aa_init):
    try:
        def obj(aa):
            q = aa2q(aa)
            return (brightness_lofi(q) - target)**2
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter':50,'ftol':1e-6})
        q = aa2q(res.x)
        mag = brightness_lofi(q)
        resid = abs(mag - target)
        Rf = Rotation.from_quat([q[1],q[2],q[3],q[0]])
        err = np.rad2deg((Rf.inv() * R_true).magnitude())
        return (float(resid), float(err))
    except:
        return (999., 999.)

if __name__ == '__main__':
    seeds = Rotation.random(500, random_state=42)
    args = [seeds[i].as_rotvec() for i in range(500)]
    
    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(worker, args)
    print(f"Time: {time.time()-t0:.1f}s", flush=True)
    
    resids = np.array([r[0] for r in results])
    errs = np.array([r[1] for r in results])
    good = resids < 0.05
    rg = resids[good]
    eg = errs[good]
    
    from scipy.stats import spearmanr, pearsonr
    sp, sp_p = spearmanr(rg, eg)
    pe, pe_p = pearsonr(rg, eg)
    print(f'Converged: {good.sum()}/500')
    print(f'Spearman corr(residual, att_err): {sp:.3f} (p={sp_p:.3e})')
    print(f'Pearson corr(residual, att_err):  {pe:.3f} (p={pe_p:.3e})')
    
    idx_by_resid = np.argsort(rg)
    idx_by_err = np.argsort(eg)
    print(f'\nTop-10 by LOWEST residual:')
    for i in idx_by_resid[:10]:
        print(f'  resid={rg[i]:.6f}  att_err={eg[i]:.1f}°')
    print(f'\nTop-10 by LOWEST att_err:')
    for i in idx_by_err[:10]:
        print(f'  resid={rg[i]:.6f}  att_err={eg[i]:.1f}°')
    print(f'\nMin residual: {rg.min():.6f} at err={eg[rg.argmin()]:.1f}°')
    print(f'Min att_err:  {eg.min():.2f}° at resid={rg[eg.argmin()]:.6f}')
