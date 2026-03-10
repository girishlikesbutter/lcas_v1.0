#!/usr/bin/env python3
"""
Forward-Propagation Filter:
1. 10k lo-fi seeds at epoch 0 → ~5000 candidate attitudes
2. For each candidate q0, sample ω values on a grid, propagate forward
3. At each subsequent epoch, check brightness match → cull (q0, ω) pairs
4. After N epochs, count survivors

No N² matching. O(N_candidates × N_omega × N_epochs) with early termination.
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
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

def log(msg):
    print(msg, flush=True)

t_start = time.time()

# ─── Setup ───
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
dt_sampling = observation_times[1] - observation_times[0]

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

log(f"Setup: {time.time()-t_start:.1f}s, {n_observations} epochs, Δt={dt_sampling:.2f}s")

# ─── Generate observed lightcurve ───
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
log(f"Lightcurve range: [{observed_lc.min():.2f}, {observed_lc.max():.2f}] mag")

# FFT omega bound
from scipy.fft import rfft, rfftfreq
fft_vals = np.abs(rfft(observed_lc - np.mean(observed_lc)))
freqs = rfftfreq(len(observed_lc), d=dt_sampling)
peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
omega_bound_deg = peak_freq * 360.0
log(f"FFT omega bound: {omega_bound_deg:.3f} °/s (true |ω|: {np.linalg.norm(np.rad2deg(true_omega0)):.3f} °/s)")

# ─── Brightness helper ───
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

def aa2q(aa):
    a = np.linalg.norm(aa)
    if a < 1e-12: return np.array([1.,0.,0.,0.])
    ax = aa/a
    return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])

_target_lofi = None
_true_q_epoch = None
_epoch_idx = None

def init_worker(target_lofi, true_q, eidx):
    global _target_lofi, _true_q_epoch, _epoch_idx
    _target_lofi = target_lofi
    _true_q_epoch = true_q
    _epoch_idx = eidx

def lofi_worker(aa_init):
    try:
        def obj(aa):
            q = aa2q(aa)
            return (brightness_lofi(q, _epoch_idx) - _target_lofi)**2
        res = minimize(obj, aa_init, method='L-BFGS-B', options={'maxiter':50,'ftol':1e-12})
        q = aa2q(res.x)
        mag = brightness_lofi(q, _epoch_idx)
        resid = abs(mag - _target_lofi)
        R_true = Rotation.from_quat([_true_q_epoch[1], _true_q_epoch[2], _true_q_epoch[3], _true_q_epoch[0]])
        R_found = Rotation.from_quat([q[1],q[2],q[3],q[0]])
        err = np.rad2deg((R_found.inv() * R_true).magnitude())
        return {'quat': q.tolist(), 'resid': float(resid), 'att_err': float(err)}
    except:
        return None

# ─── Forward propagation worker ───
# Shared data set in init
_observed_lc_shared = None
_sun_pos_shared = None
_obs_pos_shared = None
_sat_pos_shared = None
_obs_dist_shared = None
_art_matrices_shared = None
_satellite_shared = None
_inertia_tensor_shared = None
_check_epochs_shared = None
_lofi_biases_shared = None

def init_prop_worker(observed_lc, sun_p, obs_p, sat_p, obs_d, art_m, sat, inertia, check_eps, biases):
    global _observed_lc_shared, _sun_pos_shared, _obs_pos_shared, _sat_pos_shared
    global _obs_dist_shared, _art_matrices_shared, _satellite_shared
    global _inertia_tensor_shared, _check_epochs_shared, _lofi_biases_shared
    _observed_lc_shared = observed_lc
    _sun_pos_shared = sun_p
    _obs_pos_shared = obs_p
    _sat_pos_shared = sat_p
    _obs_dist_shared = obs_d
    _art_matrices_shared = art_m
    _satellite_shared = sat
    _inertia_tensor_shared = inertia
    _check_epochs_shared = check_eps
    _lofi_biases_shared = biases

def prop_worker(args):
    """Test one (q0, omega) pair against all check epochs."""
    q0, omega, cand_idx, omega_idx = args
    try:
        check_epochs = _check_epochs_shared
        n_check = len(check_epochs)
        
        # Propagate attitude through all check epochs
        max_t = max(check_epochs) * 7.2144  # epoch index to time (approximate)
        # Actually use observation_times directly
        # We need the times for each check epoch
        
        # For tumbling propagation with inertia tensor
        # Use simple Euler rotation for speed: q(t) = q_rot(omega*t) * q0
        # This assumes torque-free (which propagate_attitude with inertia_tensor handles properly)
        # But for speed, assume constant omega (good approximation over short intervals)
        
        R0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
        
        n_pass = 0
        for i, ep_idx in enumerate(check_epochs):
            # Time from epoch 0 of the observation
            dt = ep_idx * 7.2144288577154308  # dt_sampling
            
            # Propagate: R(t) = R(omega*t) * R0 (constant omega approximation)
            rotvec = omega * dt  # omega in rad/s, dt in seconds
            R_t = Rotation.from_rotvec(rotvec) * R0
            q_t = R_t.as_quat()  # scipy xyzw
            q_wxyz = np.array([q_t[3], q_t[0], q_t[1], q_t[2]])
            
            # Evaluate brightness
            R = R_t.as_matrix()
            s = R @ (_sun_pos_shared[ep_idx] - _sat_pos_shared[ep_idx])
            s = s / np.linalg.norm(s)
            o = R @ (_obs_pos_shared[ep_idx] - _sat_pos_shared[ep_idx])
            o = o / np.linalg.norm(o)
            
            a = {c: m[ep_idx:ep_idx+1] for c, m in _art_matrices_shared.items()}
            lit = create_no_shadow_lit_status(_satellite_shared, 1)
            mag, _, _, _, _, _ = generate_lightcurves(
                facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3),
                k2_vectors_array=o.reshape(1,3),
                observer_distances=np.array([_obs_dist_shared[ep_idx]]),
                satellite=_satellite_shared, epochs=np.array([0.]),
                pre_computed_matrices=a,
                generate_no_shadow=False, animate=False, show_progress=False)
            
            target = _observed_lc_shared[ep_idx] + _lofi_biases_shared[i]
            if abs(mag[0] - target) > 3 * 0.05:  # 3σ tolerance
                return None  # Early termination
            n_pass += 1
        
        # All epochs passed!
        return {'cand_idx': cand_idx, 'omega_idx': omega_idx, 
                'q0': q0.tolist(), 'omega_deg': np.rad2deg(omega).tolist(),
                'n_pass': n_pass}
    except:
        return None

if __name__ == '__main__':
    N_SEEDS = 10000
    anchor_idx = 0  # Use epoch 0 as anchor
    
    # ─── Step 1: 10k lo-fi seeds at epoch 0 ───
    log(f"\n{'='*60}")
    log(f"STEP 1: Lo-fi candidates at epoch {anchor_idx} (mag={observed_lc[anchor_idx]:.3f})")
    
    true_q_anchor = true_quaternions[anchor_idx]
    lofi_bias_anchor = brightness_lofi(true_q_anchor, anchor_idx) - brightness_hifi(true_q_anchor, anchor_idx)
    target_anchor = observed_lc[anchor_idx] + lofi_bias_anchor
    
    seeds = Rotation.random(N_SEEDS, random_state=7000)
    args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]
    
    t0 = time.time()
    with Pool(8, initializer=init_worker, initargs=(target_anchor, true_q_anchor, anchor_idx)) as pool:
        results = pool.map(lofi_worker, args)
    dt_run = time.time() - t0
    
    candidates = [r for r in results if r is not None and r['resid'] < noise_sigma]
    candidates.sort(key=lambda r: r['att_err'])
    log(f"  Converged: {len(candidates)}/{N_SEEDS} in {dt_run:.0f}s")
    errs = [c['att_err'] for c in candidates]
    log(f"  Min err: {min(errs):.2f}°, within 5°: {sum(1 for e in errs if e<5)}, within 10°: {sum(1 for e in errs if e<10)}")
    
    # ─── Step 2: Build omega grid ───
    # FFT bound gives max |ω| ~ 3.2°/s. True is ~2°/s.
    # Grid: 11 points per axis from -3.5 to 3.5 °/s = 11³ = 1331 omega samples
    omega_max_deg = omega_bound_deg + 0.5  # small margin
    n_omega_per_axis = 11
    omega_1d = np.linspace(-omega_max_deg, omega_max_deg, n_omega_per_axis)
    omega_grid = np.array(np.meshgrid(omega_1d, omega_1d, omega_1d)).T.reshape(-1, 3)
    # Filter: only keep |ω| <= omega_max_deg
    omega_norms = np.linalg.norm(omega_grid, axis=1)
    omega_grid = omega_grid[omega_norms <= omega_max_deg]
    omega_grid_rad = np.deg2rad(omega_grid)
    n_omega = len(omega_grid_rad)
    log(f"\n  Omega grid: {n_omega} samples (max {omega_max_deg:.1f}°/s, {n_omega_per_axis} per axis)")
    log(f"  True omega: [{np.rad2deg(true_omega0[0]):.2f}, {np.rad2deg(true_omega0[1]):.2f}, {np.rad2deg(true_omega0[2]):.2f}] °/s")
    
    # Check closest grid point to truth
    diffs = np.linalg.norm(omega_grid - np.rad2deg(true_omega0), axis=1)
    closest_idx = np.argmin(diffs)
    log(f"  Closest grid point to truth: {omega_grid[closest_idx]} (dist={diffs[closest_idx]:.3f}°/s)")
    
    # ─── Step 3: Select check epochs ───
    # Use 20 epochs spread across the observation window for diversity
    # Skip epoch 0 (that's our anchor). Pick epochs with good brightness variation.
    n_check = 20
    # Spread evenly but skip first few (too similar to anchor)
    check_epoch_indices = np.linspace(5, n_observations - 1, n_check, dtype=int).tolist()
    log(f"\n  Check epochs ({n_check}): {check_epoch_indices}")
    log(f"  Brightness at check epochs: {[f'{observed_lc[i]:.2f}' for i in check_epoch_indices]}")
    
    # Precompute lo-fi biases at check epochs
    lofi_biases = []
    for ei in check_epoch_indices:
        true_q_ei = true_quaternions[ei]
        bias = brightness_lofi(true_q_ei, ei) - brightness_hifi(true_q_ei, ei)
        lofi_biases.append(bias)
    log(f"  Lo-fi biases: min={min(lofi_biases):.4f}, max={max(lofi_biases):.4f}")
    
    # ─── Step 4: Forward propagation ───
    log(f"\n{'='*60}")
    log(f"STEP 2: Forward propagation ({len(candidates)} candidates × {n_omega} omegas = {len(candidates)*n_omega:,} tests)")
    log(f"  Each test checks {n_check} epochs, early-terminates on first fail")
    
    # Build argument list
    prop_args = []
    for ci, cand in enumerate(candidates):
        q0 = np.array(cand['quat'])
        for oi, omega in enumerate(omega_grid_rad):
            prop_args.append((q0, omega, ci, oi))
    
    log(f"  Total jobs: {len(prop_args):,}")
    
    # Process in batches to report progress and limit memory
    BATCH = 50000
    all_survivors = []
    t_prop = time.time()
    
    for batch_start in range(0, len(prop_args), BATCH):
        batch_end = min(batch_start + BATCH, len(prop_args))
        batch = prop_args[batch_start:batch_end]
        
        with Pool(8, initializer=init_prop_worker,
                  initargs=(observed_lc, sun_pos, obs_pos, sat_pos, obs_dist,
                           art_matrices, satellite, inertia_tensor,
                           check_epoch_indices, lofi_biases)) as pool:
            results = pool.map(prop_worker, batch)
        
        survivors = [r for r in results if r is not None]
        all_survivors.extend(survivors)
        
        elapsed = time.time() - t_prop
        rate = (batch_end) / elapsed
        eta = (len(prop_args) - batch_end) / rate if rate > 0 else 0
        log(f"  Batch {batch_start//BATCH + 1}: {batch_end}/{len(prop_args)} done, "
            f"{len(all_survivors)} survivors, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    dt_prop = time.time() - t_prop
    log(f"\n  Forward propagation: {dt_prop:.0f}s")
    log(f"  Survivors: {len(all_survivors)} / {len(prop_args):,}")
    
    # ─── Step 5: Analyze survivors ───
    log(f"\n{'='*60}")
    log("RESULTS")
    
    if all_survivors:
        # Check attitude errors of surviving (q0, omega) pairs
        for s in all_survivors:
            ci = s['cand_idx']
            s['att_err'] = candidates[ci]['att_err']
            # Omega error
            s['omega_err'] = float(np.linalg.norm(np.array(s['omega_deg']) - np.rad2deg(true_omega0)))
        
        all_survivors.sort(key=lambda s: s['att_err'])
        
        log(f"\n  Total survivors: {len(all_survivors)}")
        log(f"  Unique q0 candidates: {len(set(s['cand_idx'] for s in all_survivors))}")
        log(f"  Unique omega indices: {len(set(s['omega_idx'] for s in all_survivors))}")
        
        log(f"\n  Top-20 by attitude error:")
        for i, s in enumerate(all_survivors[:20]):
            log(f"    {i+1}. att_err={s['att_err']:.2f}° ω=[{s['omega_deg'][0]:.2f},{s['omega_deg'][1]:.2f},{s['omega_deg'][2]:.2f}]°/s ω_err={s['omega_err']:.2f}°/s")
        
        # Truth recovery
        close = [s for s in all_survivors if s['att_err'] < 5 and s['omega_err'] < 1]
        log(f"\n  Close to truth (att<5° AND ω_err<1°/s): {len(close)}")
        for s in close:
            log(f"    att={s['att_err']:.2f}° ω=[{s['omega_deg'][0]:.2f},{s['omega_deg'][1]:.2f},{s['omega_deg'][2]:.2f}]°/s ω_err={s['omega_err']:.2f}°/s")
    else:
        log("  NO SURVIVORS — all (q0, ω) pairs failed brightness check")
        log("  This means either: grid too coarse, tolerance too tight, or constant-ω assumption wrong")
    
    log(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Save
    out = {
        'n_candidates': len(candidates),
        'n_omega': n_omega,
        'n_total_tests': len(prop_args),
        'n_check_epochs': n_check,
        'check_epoch_indices': check_epoch_indices,
        'omega_bound_deg': float(omega_bound_deg),
        'omega_grid_spacing_deg': float(omega_1d[1] - omega_1d[0]),
        'n_survivors': len(all_survivors),
        'survivors': all_survivors[:200],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_forward_prop.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
