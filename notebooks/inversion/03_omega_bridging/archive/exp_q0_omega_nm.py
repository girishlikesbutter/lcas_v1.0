#!/usr/bin/env python3
"""
q0-fixed omega optimization — Nelder-Mead, 1 start, all candidates.
Quick concept validation: does multi-epoch residual correlate with truth?
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
from src.computation.shadow_engine import create_no_shadow_lit_status, compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

def log(msg):
    print(msg, flush=True)

t_start = time.time()

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
metakernel_path = config_manager.get_metakernel_path(config)
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

n_observations = 500
OBSERVER_ID = 399999
noise_sigma = 0.05

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et("2020-02-05T11:00:00")
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=OBSERVER_ID, spice_handler=spice_handler, config=config)
sun_pos = geometry_data['sun_positions']
obs_pos = geometry_data['obs_positions']
sat_pos = geometry_data['sat_positions']
obs_dist = geometry_data['observer_distances']

art_angles = {'SP_North': np.full(n_observations, 0.0), 'SP_South': np.full(n_observations, 0.0),
              'AD_East': np.full(n_observations, 15.0), 'AD_West': np.full(n_observations, 15.0)}
art_matrices = compute_rotation_matrices_from_angles(art_angles, satellite)

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.5, -0.3, 2.0]))

true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

log(f"Setup: {time.time()-t_start:.1f}s")

# Generate observed lightcurve
from src.inversion.objective_function import ObjectiveFunction
obj_temp = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations), sun_positions_j2000=sun_pos,
    observer_positions_j2000=obs_pos, satellite_positions_j2000=sat_pos,
    observer_distances=obs_dist, compute_shadows_flag=True,
    articulation_matrices=art_matrices, mode="tumbling", inertia_tensor=inertia_tensor)

true_k1, true_k2 = obj_temp._compute_body_frame_vectors(true_quaternions)
true_lit = compute_shadows(satellite=satellite, k1_vectors=true_k1,
    explicit_component_matrices=art_matrices, show_progress=False)
true_lc, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=true_lit, k1_vectors_array=true_k1,
    k2_vectors_array=true_k2, observer_distances=obs_dist, satellite=satellite, epochs=epochs,
    pre_computed_matrices=art_matrices, generate_no_shadow=False, animate=False, show_progress=False)

np.random.seed(42)
observed_lc = true_lc + np.random.normal(0, noise_sigma, n_observations)
log(f"Lightcurve: [{observed_lc.min():.2f}, {observed_lc.max():.2f}] mag")

# Check epochs and biases
check_indices = list(range(0, 10))
check_times = observation_times[check_indices]

def brightness_lofi(q_wxyz, epoch_idx):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (sun_pos[epoch_idx] - sat_pos[epoch_idx]); s /= np.linalg.norm(s)
    o = R @ (obs_pos[epoch_idx] - sat_pos[epoch_idx]); o /= np.linalg.norm(o)
    a = {c: m[epoch_idx:epoch_idx+1] for c, m in art_matrices.items()}
    lit = create_no_shadow_lit_status(satellite, 1)
    m, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3),
        k2_vectors_array=o.reshape(1,3), observer_distances=np.array([obs_dist[epoch_idx]]),
        satellite=satellite, epochs=np.array([0.]), pre_computed_matrices=a,
        generate_no_shadow=False, animate=False, show_progress=False)
    return m[0]

def brightness_hifi(q_wxyz, epoch_idx):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (sun_pos[epoch_idx] - sat_pos[epoch_idx]); s /= np.linalg.norm(s)
    o = R @ (obs_pos[epoch_idx] - sat_pos[epoch_idx]); o /= np.linalg.norm(o)
    a = {c: m[epoch_idx:epoch_idx+1] for c, m in art_matrices.items()}
    lit = compute_shadows(satellite=satellite, k1_vectors=s.reshape(1,3), explicit_component_matrices=a, show_progress=False)
    m, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3),
        k2_vectors_array=o.reshape(1,3), observer_distances=np.array([obs_dist[epoch_idx]]),
        satellite=satellite, epochs=np.array([0.]), pre_computed_matrices=a,
        generate_no_shadow=False, animate=False, show_progress=False)
    return m[0]

lofi_biases = []
for ei in check_indices:
    tq = true_quaternions[ei]
    bias = brightness_lofi(tq, ei) - brightness_hifi(tq, ei)
    lofi_biases.append(float(bias))

observed_targets = [observed_lc[ei] + lofi_biases[i] for i, ei in enumerate(check_indices)]
log(f"Check epochs: {check_indices}, biases: {[f'{b:.4f}' for b in lofi_biases]}")

# Step 1 worker
def aa2q(aa):
    a = np.linalg.norm(aa)
    if a < 1e-12: return np.array([1.,0.,0.,0.])
    ax = aa/a
    return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])

_target_lofi = None
_true_q_epoch = None
_epoch_idx = None

def init_step1(target_lofi, true_q, eidx):
    global _target_lofi, _true_q_epoch, _epoch_idx
    _target_lofi = target_lofi; _true_q_epoch = true_q; _epoch_idx = eidx

def step1_worker(aa_init):
    try:
        def obj(aa):
            q = aa2q(aa)
            return (brightness_lofi(q, _epoch_idx) - _target_lofi)**2
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter':50,'ftol':1e-12})
        q = aa2q(res.x)
        resid = abs(brightness_lofi(q, _epoch_idx) - _target_lofi)
        R_true = Rotation.from_quat([_true_q_epoch[1], _true_q_epoch[2], _true_q_epoch[3], _true_q_epoch[0]])
        R_found = Rotation.from_quat([q[1],q[2],q[3],q[0]])
        err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return {'quat': q.tolist(), 'resid': float(resid), 'att_err': float(err)}
    except:
        return None

# Step 2 worker
_ci = None; _ct = None; _tgt = None
_sp = None; _op = None; _satp = None; _od = None
_am = None; _sat = None; _it = None; _tom = None

def init_step2(ci, ct, tgt, sp, op, satp, od, am, sat, it, tom):
    global _ci, _ct, _tgt, _sp, _op, _satp, _od, _am, _sat, _it, _tom
    _ci=ci; _ct=ct; _tgt=tgt; _sp=sp; _op=op; _satp=satp; _od=od
    _am=am; _sat=sat; _it=it; _tom=tom

def eval_omega(omega_rad, q0):
    quats, _ = propagate_attitude(q0=q0, omega0=omega_rad, times=_ct, mode="tumbling", inertia_tensor=_it)
    total = 0.0
    for i, ei in enumerate(_ci):
        q = quats[i]
        R = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        s = R @ (_sp[ei] - _satp[ei]); s /= np.linalg.norm(s)
        o = R @ (_op[ei] - _satp[ei]); o /= np.linalg.norm(o)
        a = {c: m[ei:ei+1] for c, m in _am.items()}
        lit = create_no_shadow_lit_status(_sat, 1)
        mag, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3),
            k2_vectors_array=o.reshape(1,3), observer_distances=np.array([_od[ei]]),
            satellite=_sat, epochs=np.array([0.]), pre_computed_matrices=a,
            generate_no_shadow=False, animate=False, show_progress=False)
        total += (mag[0] - _tgt[i])**2
    return total

def step2_worker(args):
    q0, cand_idx, att_err = args
    q0 = np.array(q0)
    try:
        res = minimize(lambda w: eval_omega(w, q0), np.array([0.,0.,0.]),
                      method='Nelder-Mead', options={'maxiter':100, 'xatol':1e-4, 'fatol':1e-10})
        omega_deg = np.rad2deg(res.x)
        omega_err = np.linalg.norm(omega_deg - np.rad2deg(_tom))
        return {
            'cand_idx': int(cand_idx), 'att_err': float(att_err),
            'omega_deg': omega_deg.tolist(), 'omega_err': float(omega_err),
            'resid': float(res.fun), 'nfev': int(res.nfev)
        }
    except:
        return None

if __name__ == '__main__':
    N_SEEDS = 10000
    anchor_idx = 0

    log(f"\n{'='*60}")
    log(f"STEP 1: {N_SEEDS} lo-fi seeds")

    true_q_anchor = true_quaternions[anchor_idx]
    target_anchor = observed_lc[anchor_idx] + lofi_biases[0]

    seeds = Rotation.random(N_SEEDS, random_state=9000)
    args_s1 = [seeds[i].as_rotvec() for i in range(N_SEEDS)]

    t0 = time.time()
    with Pool(8, initializer=init_step1, initargs=(target_anchor, true_q_anchor, anchor_idx)) as pool:
        results_s1 = pool.map(step1_worker, args_s1)
    dt_s1 = time.time() - t0

    candidates = [r for r in results_s1 if r is not None and r['resid'] < noise_sigma]
    candidates.sort(key=lambda r: r['att_err'])
    log(f"  Converged: {len(candidates)}/{N_SEEDS} in {dt_s1:.0f}s")
    errs = [c['att_err'] for c in candidates]
    log(f"  Min err: {min(errs):.2f}°, within 5°: {sum(1 for e in errs if e<5)}, within 10°: {sum(1 for e in errs if e<10)}")

    N_OPT = len(candidates)
    log(f"\n{'='*60}")
    log(f"STEP 2: Nelder-Mead ω optimization ({N_OPT} candidates, 1 start, 10 epochs)")

    args_s2 = [(c['quat'], i, c['att_err']) for i, c in enumerate(candidates)]

    t0 = time.time()
    BATCH = 500
    all_results = []

    for batch_start in range(0, len(args_s2), BATCH):
        batch_end = min(batch_start + BATCH, len(args_s2))
        batch = args_s2[batch_start:batch_end]

        with Pool(8, initializer=init_step2,
                  initargs=(check_indices, check_times, observed_targets,
                           sun_pos, obs_pos, sat_pos, obs_dist,
                           art_matrices, satellite, inertia_tensor, true_omega0)) as pool:
            batch_results = pool.map(step2_worker, batch)

        good = [r for r in batch_results if r is not None]
        all_results.extend(good)

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 1
        eta = (len(args_s2) - batch_end) / rate if rate > 0 else 0

        if all_results:
            best = min(all_results, key=lambda r: r['resid'])
            log(f"  Batch {batch_start//BATCH+1}: {batch_end}/{len(args_s2)}, "
                f"{elapsed:.0f}s, ETA {eta:.0f}s | best: att={best['att_err']:.1f}° ω_err={best['omega_err']:.3f}°/s nfev={best['nfev']}")

    dt_s2 = time.time() - t0
    log(f"\n  Step 2: {dt_s2:.0f}s ({len(all_results)} results)")

    # Results
    log(f"\n{'='*60}")
    log("RESULTS")

    if all_results:
        all_results.sort(key=lambda r: r['resid'])

        log(f"\n  Top-20 by residual:")
        for i, r in enumerate(all_results[:20]):
            log(f"    {i+1}. resid={r['resid']:.6f} att={r['att_err']:.2f}° "
                f"ω=[{r['omega_deg'][0]:.3f},{r['omega_deg'][1]:.3f},{r['omega_deg'][2]:.3f}]°/s ω_err={r['omega_err']:.3f}°/s")

        best = all_results[0]
        log(f"\n  BEST: att={best['att_err']:.2f}°, ω_err={best['omega_err']:.3f}°/s")
        log(f"    ω=[{best['omega_deg'][0]:.4f}, {best['omega_deg'][1]:.4f}, {best['omega_deg'][2]:.4f}]°/s")
        log(f"    True=[{np.rad2deg(true_omega0[0]):.4f}, {np.rad2deg(true_omega0[1]):.4f}, {np.rad2deg(true_omega0[2]):.4f}]°/s")

        from scipy.stats import spearmanr
        resids = [r['resid'] for r in all_results]
        att_errs_r = [r['att_err'] for r in all_results]
        omega_errs_r = [r['omega_err'] for r in all_results]
        rho_att, p_att = spearmanr(resids, att_errs_r)
        rho_omega, p_omega = spearmanr(resids, omega_errs_r)
        log(f"\n  Spearman (residual vs att_err): {rho_att:.4f} (p={p_att:.2e})")
        log(f"  Spearman (residual vs omega_err): {rho_omega:.4f} (p={p_omega:.2e})")

        close = [r for r in all_results if r['att_err'] < 5 and r['omega_err'] < 0.5]
        log(f"\n  Close to truth (att<5° AND ω<0.5°/s): {len(close)}")
        for r in close:
            rank = all_results.index(r) + 1
            log(f"    rank={rank}/{len(all_results)} att={r['att_err']:.2f}° ω_err={r['omega_err']:.3f}°/s resid={r['resid']:.6f}")

    log(f"\nTotal time: {time.time()-t_start:.0f}s")

    out = {
        'n_seeds': N_SEEDS, 'n_candidates': len(candidates), 'n_results': len(all_results),
        'method': 'Nelder-Mead', 'n_starts': 1, 'n_check_epochs': len(check_indices),
        'results_top200': all_results[:200],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_q0_omega_nm.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
