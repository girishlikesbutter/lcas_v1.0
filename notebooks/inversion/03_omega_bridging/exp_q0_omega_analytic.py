#!/usr/bin/env python3
"""
q0-fixed omega optimization with ANALYTIC attitude propagation.
IS-901: I2/I3 = 0.992, near-axisymmetric → closed-form q(t).

Axisymmetric solution (I2=I3=It, I1=Is):
  ω1(t) = ω1(0)                          [constant spin]
  ω2(t) = ωp·cos(Ωt + φ)                 [precession]
  ω3(t) = ωp·sin(Ωt + φ)                 [precession]
  
  where: Ω = (Is - It)/It · ω1(0)        [body precession rate]
         ωp = sqrt(ω2(0)² + ω3(0)²)      [precession amplitude]
         φ = atan2(ω3(0), ω2(0))          [precession phase]

  q(t) = q_precession(t) * q_spin(t) * q0
  
  Actually simpler: integrate q̇ = ½ q ⊗ [0,ω(t)]
  For axisymmetric case, ω in body frame has constant ω1 and 
  rotating ω2,ω3 → decompose into spin + precession.
"""
import sys, time, json, numpy as np, traceback
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
from src.computation.shadow_engine import create_no_shadow_lit_status, compute_shadows
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
principal_moments = inertia_result.principal_moments
I1, I2, I3 = principal_moments

log(f"Principal moments: I1={I1:.1f}, I2={I2:.1f}, I3={I3:.1f}")
log(f"I2/I3 = {I2/I3:.4f} (1.0 = perfectly axisymmetric)")

# Use average of I2, I3 for axisymmetric approximation
It = (I2 + I3) / 2  # transverse moment
Is = I1              # spin (symmetry) axis moment

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

# Generate true lightcurve using NUMERICAL propagation (ground truth)
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

log(f"Setup: {time.time()-t_start:.1f}s")

# ─── Validate analytic vs numerical propagation ───
def propagate_analytic(q0_wxyz, omega0_rad, times, Is, It):
    """
    Analytic attitude propagation for axisymmetric body.
    q0_wxyz: initial quaternion [w,x,y,z]
    omega0_rad: initial angular velocity [wx,wy,wz] in body frame (rad/s)
    times: array of times (s)
    Is: spin axis moment of inertia (I1, smallest)
    It: transverse moment of inertia (average of I2,I3)
    
    Returns: quaternions array (N,4) in wxyz format
    """
    # Body-frame angular velocity components
    w1_0 = omega0_rad[0]  # spin axis component (constant)
    w2_0 = omega0_rad[1]
    w3_0 = omega0_rad[2]
    
    # Precession rate in body frame
    Omega_body = (Is - It) / It * w1_0
    
    # Precession amplitude and phase
    wp = np.sqrt(w2_0**2 + w3_0**2)
    phi = np.arctan2(w3_0, w2_0)
    
    # Body-frame omega(t):
    # w1(t) = w1_0 (constant)
    # w2(t) = wp * cos(Omega_body * t + phi)
    # w3(t) = wp * sin(Omega_body * t + phi)
    
    # The total rotation can be decomposed as:
    # In the body frame, we have spin about e1 at rate w1_0
    # plus precession of the transverse components at rate Omega_body
    
    # The inertial-frame angular velocity has two components:
    # 1. Rotation about the body's e1 axis at rate (w1_0 + Omega_body) [nutation removed]
    # Actually, let me use the direct quaternion approach.
    
    # For axisymmetric body, the motion decomposes into:
    # q(t) = q_inertial_precession(t) * q0 ... no, let me think more carefully.
    
    # The body frame angular velocity is:
    # ω(t) = [w1_0, wp*cos(Ωt+φ), wp*sin(Ωt+φ)]
    #
    # This can be decomposed as rotation about e1 at rate w1_0
    # composed with rotation of the e2-e3 plane at rate Omega_body.
    #
    # Equivalently: the angular velocity in the body frame is the sum of:
    #   - spin about body e1: [w1_0, 0, 0]
    #   - a vector rotating in the e2-e3 plane: [0, wp*cos(Ωt+φ), wp*sin(Ωt+φ)]
    #
    # The second part is equivalent to a precession. In the body frame,
    # the transverse angular velocity rotates at rate Omega_body.
    #
    # For the quaternion: think of it as q(t) = q_space(t) · q0
    # where q_space(t) represents the accumulated rotation.
    #
    # For axisymmetric case, the space-frame angular momentum L is fixed.
    # L = I · ω, and in space frame L is constant.
    # The body precesses around L.
    #
    # Direct integration approach: since ω(t) is known analytically,
    # we can integrate q̇ = ½ q ⊗ [0, ω(t)] using the fact that
    # the problem decomposes into two constant-rate rotations.
    #
    # q(t) = R_precession(-Ωt) · R_total_spin(ψ·t) · q0
    # where ψ is related to the total angular velocity magnitude
    #
    # Actually, the cleanest decomposition:
    # In a frame rotating with the body precession, the angular velocity is constant.
    # Define a frame S' that rotates about e1 at rate -Omega_body.
    # In S', the angular velocity is [w1_0 - Omega_body, wp*cos(φ), wp*sin(φ)] = constant
    # So the rotation in S' is a simple constant-rate rotation!
    # Then: q(t) = q_e1_rotation(Omega_body * t) · q_constant_rate(t) · q0
    
    # Constant angular velocity in the co-rotating frame
    w_corot = np.array([w1_0 + Omega_body, wp * np.cos(phi), wp * np.sin(phi)])
    # Wait, need to be careful with the sign.
    # If we go to a frame rotating at +Omega_body about e1,
    # the apparent angular velocity is ω - Omega_body*e1
    # = [w1_0 - Omega_body, wp*cos(Ωt+φ-Ωt), wp*sin(Ωt+φ-Ωt)]
    # = [w1_0 - Omega_body, wp*cos(φ), wp*sin(φ)]
    # This IS constant! 
    
    w_corot = np.array([w1_0 - Omega_body, wp * np.cos(phi), wp * np.sin(phi)])
    w_corot_mag = np.linalg.norm(w_corot)
    
    N = len(times)
    quats = np.zeros((N, 4))  # wxyz
    
    if w_corot_mag < 1e-15:
        # No rotation in co-rotating frame, just precession about e1
        for i, t in enumerate(times):
            angle_prec = Omega_body * t
            q_prec = np.array([np.cos(angle_prec/2), np.sin(angle_prec/2), 0, 0])
            quats[i] = quat_mult(q_prec, q0_wxyz)
    else:
        w_corot_axis = w_corot / w_corot_mag
        
        for i, t in enumerate(times):
            # Constant-rate rotation in co-rotating frame
            angle_rot = w_corot_mag * t
            ha = angle_rot / 2
            q_rot = np.array([np.cos(ha), *(np.sin(ha) * w_corot_axis)])
            
            # Precession about e1
            angle_prec = Omega_body * t
            hp = angle_prec / 2
            q_prec = np.array([np.cos(hp), np.sin(hp), 0, 0])
            
            # Total: first apply q_rot (in co-rotating frame), then q_prec (back to body frame)
            # q(t) = q_prec * q_rot * q0
            q_temp = quat_mult(q_rot, q0_wxyz)
            quats[i] = quat_mult(q_prec, q_temp)
    
    return quats

def quat_mult(q, r):
    """Multiply quaternions q * r, both in wxyz format."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Validate: compare analytic vs numerical at check times
check_indices = list(range(0, 10))
check_times = observation_times[check_indices]

q_numeric = true_quaternions[check_indices]
q_analytic = propagate_analytic(true_q0, true_omega0, check_times, Is, It)

log(f"\nAnalytic vs Numerical validation (true params):")
for i in range(len(check_indices)):
    qn = q_numeric[i]
    qa = q_analytic[i]
    # Quaternion distance
    dot = abs(np.dot(qn, qa))
    dot = min(dot, 1.0)
    angle_err = np.rad2deg(2 * np.arccos(dot))
    log(f"  t={check_times[i]:.1f}s: angle error = {angle_err:.4f}°")

# Also validate at later times
late_indices = [50, 100, 200, 300, 400, 499]
late_times = observation_times[late_indices]
q_numeric_late = true_quaternions[late_indices]
q_analytic_late = propagate_analytic(true_q0, true_omega0, late_times, Is, It)

log(f"\nLate-time validation:")
for i, li in enumerate(late_indices):
    dot = abs(np.dot(q_numeric_late[i], q_analytic_late[i]))
    dot = min(dot, 1.0)
    angle_err = np.rad2deg(2 * np.arccos(dot))
    log(f"  t={late_times[i]:.1f}s (idx={li}): angle error = {angle_err:.4f}°")

# ─── Benchmark: analytic vs propagate_attitude ───
import timeit
t0 = time.time()
for _ in range(1000):
    propagate_analytic(true_q0, true_omega0, check_times, Is, It)
t_analytic = (time.time() - t0) / 1000

t0 = time.time()
for _ in range(100):
    propagate_attitude(q0=true_q0, omega0=true_omega0, times=check_times,
        mode="tumbling", inertia_tensor=inertia_tensor)
t_numeric = (time.time() - t0) / 100

log(f"\nBenchmark:")
log(f"  Analytic: {t_analytic*1e6:.0f} µs per propagation")
log(f"  Numerical: {t_numeric*1e3:.1f} ms per propagation")
log(f"  Speedup: {t_numeric/t_analytic:.0f}×")

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
log(f"\nLightcurve range: [{observed_lc.min():.2f}, {observed_lc.max():.2f}] mag")

# ─── Check epochs and lo-fi biases ───
check_indices_full = list(range(0, 10))
check_times_full = observation_times[check_indices_full]

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
for ei in check_indices_full:
    tq = true_quaternions[ei]
    bias = brightness_lofi(tq, ei) - brightness_hifi(tq, ei)
    lofi_biases.append(float(bias))

observed_targets = [observed_lc[ei] + lofi_biases[i] for i, ei in enumerate(check_indices_full)]

log(f"Check epochs: {check_indices_full}")
log(f"Lo-fi biases: {[f'{b:.4f}' for b in lofi_biases]}")

# ─── Step 1: Lo-fi candidates ───
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
    _target_lofi = target_lofi
    _true_q_epoch = true_q
    _epoch_idx = eidx

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

# ─── Step 2: Omega optimization with analytic propagation ───
_check_idx_s = None
_check_times_s = None
_targets_s = None
_sun_s = None; _obs_s = None; _sat_s = None; _obsd_s = None
_art_s = None; _satellite_s = None; _Is_s = None; _It_s = None; _true_om_s = None

def init_step2(ci, ct, tgt, sun, obs, sat, obsd, art, satobj, Is_val, It_val, true_om):
    global _check_idx_s, _check_times_s, _targets_s, _sun_s, _obs_s, _sat_s, _obsd_s
    global _art_s, _satellite_s, _Is_s, _It_s, _true_om_s
    _check_idx_s = ci; _check_times_s = ct; _targets_s = tgt
    _sun_s = sun; _obs_s = obs; _sat_s = sat; _obsd_s = obsd
    _art_s = art; _satellite_s = satobj; _Is_s = Is_val; _It_s = It_val; _true_om_s = true_om

def eval_omega_analytic(omega_rad, q0):
    """Evaluate brightness residual using analytic propagation."""
    quats = propagate_analytic(q0, omega_rad, _check_times_s, _Is_s, _It_s)
    total_resid = 0.0
    for i, ei in enumerate(_check_idx_s):
        q = quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        s = R @ (_sun_s[ei] - _sat_s[ei]); s /= np.linalg.norm(s)
        o = R @ (_obs_s[ei] - _sat_s[ei]); o /= np.linalg.norm(o)
        a = {c: m[ei:ei+1] for c, m in _art_s.items()}
        lit = create_no_shadow_lit_status(_satellite_s, 1)
        mag, _, _, _, _, _ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=s.reshape(1,3),
            k2_vectors_array=o.reshape(1,3), observer_distances=np.array([_obsd_s[ei]]),
            satellite=_satellite_s, epochs=np.array([0.]), pre_computed_matrices=a,
            generate_no_shadow=False, animate=False, show_progress=False)
        total_resid += (mag[0] - _targets_s[i])**2
    return total_resid

def step2_worker(args):
    """Optimize omega for a fixed q0 using analytic propagation."""
    q0, cand_idx, att_err = args
    q0 = np.array(q0)
    try:
        best_resid = np.inf
        best_omega = None
        
        omega_starts = [
            np.array([0., 0., 0.]),
            np.deg2rad(np.array([1., -1., 1.])),
            np.deg2rad(np.array([-1., 1., -1.])),
        ]
        
        omega_bound = np.deg2rad(4.0)
        bounds = [(-omega_bound, omega_bound)] * 3
        
        for omega_init in omega_starts:
            try:
                res = minimize(lambda w: eval_omega_analytic(w, q0), omega_init, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter':30, 'ftol':1e-10})
                if res.fun < best_resid:
                    best_resid = res.fun
                    best_omega = res.x
            except:
                continue
        
        if best_omega is None:
            return None
        
        omega_deg = np.rad2deg(best_omega)
        omega_err = np.linalg.norm(omega_deg - np.rad2deg(_true_om_s))
        
        return {
            'cand_idx': int(cand_idx),
            'att_err': float(att_err),
            'omega_deg': omega_deg.tolist(),
            'omega_err': float(omega_err),
            'resid': float(best_resid),
        }
    except:
        return None

if __name__ == '__main__':
    N_SEEDS = 10000
    anchor_idx = 0

    log(f"\n{'='*60}")
    log(f"STEP 1: {N_SEEDS} lo-fi seeds at epoch {anchor_idx}")

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

    # Use ALL candidates
    N_OPT = len(candidates)
    log(f"  Optimizing omega for ALL {N_OPT} candidates")

    # ─── Step 2: Omega optimization with analytic propagation ───
    log(f"\n{'='*60}")
    log(f"STEP 2: Omega optimization ({N_OPT} candidates × 3 starts × 10 epochs, ANALYTIC)")

    args_s2 = [(c['quat'], i, c['att_err']) for i, c in enumerate(candidates)]

    t0 = time.time()
    BATCH = 500
    all_results = []
    
    for batch_start in range(0, len(args_s2), BATCH):
        batch_end = min(batch_start + BATCH, len(args_s2))
        batch = args_s2[batch_start:batch_end]
        
        with Pool(8, initializer=init_step2,
                  initargs=(check_indices_full, check_times_full, observed_targets,
                           sun_pos, obs_pos, sat_pos, obs_dist,
                           art_matrices, satellite, Is, It, true_omega0)) as pool:
            batch_results = pool.map(step2_worker, batch)
        
        good = [r for r in batch_results if r is not None]
        all_results.extend(good)
        
        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 1
        eta = (len(args_s2) - batch_end) / rate if rate > 0 else 0
        
        if all_results:
            best = min(all_results, key=lambda r: r['resid'])
            log(f"  Batch {batch_start//BATCH+1}: {batch_end}/{len(args_s2)} done, "
                f"{elapsed:.0f}s, ETA {eta:.0f}s | best: att={best['att_err']:.1f}° ω_err={best['omega_err']:.3f}°/s resid={best['resid']:.6f}")

    dt_s2 = time.time() - t0
    log(f"\n  Omega optimization: {dt_s2:.0f}s ({len(all_results)} results)")

    # ─── Results ───
    log(f"\n{'='*60}")
    log("RESULTS")

    if all_results:
        all_results.sort(key=lambda r: r['resid'])
        
        log(f"\n  Top-20 by residual:")
        for i, r in enumerate(all_results[:20]):
            log(f"    {i+1}. resid={r['resid']:.6f} att_err={r['att_err']:.2f}° "
                f"ω=[{r['omega_deg'][0]:.3f},{r['omega_deg'][1]:.3f},{r['omega_deg'][2]:.3f}]°/s ω_err={r['omega_err']:.3f}°/s")
        
        best = all_results[0]
        log(f"\n  BEST: att={best['att_err']:.2f}°, ω_err={best['omega_err']:.3f}°/s, resid={best['resid']:.8f}")
        log(f"    ω=[{best['omega_deg'][0]:.4f}, {best['omega_deg'][1]:.4f}, {best['omega_deg'][2]:.4f}] °/s")
        log(f"    True ω=[{np.rad2deg(true_omega0[0]):.4f}, {np.rad2deg(true_omega0[1]):.4f}, {np.rad2deg(true_omega0[2]):.4f}] °/s")
        
        from scipy.stats import spearmanr
        resids = [r['resid'] for r in all_results]
        att_errs_r = [r['att_err'] for r in all_results]
        omega_errs_r = [r['omega_err'] for r in all_results]
        rho_att, _ = spearmanr(resids, att_errs_r)
        rho_omega, _ = spearmanr(resids, omega_errs_r)
        log(f"\n  Spearman (residual vs att_err): {rho_att:.4f}")
        log(f"  Spearman (residual vs omega_err): {rho_omega:.4f}")
        
        # Truth recovery
        close = [r for r in all_results if r['att_err'] < 5 and r['omega_err'] < 0.5]
        log(f"\n  Close to truth (att<5° AND ω_err<0.5°/s): {len(close)}")
        if close:
            for r in close:
                rank = all_results.index(r) + 1
                log(f"    rank={rank} att={r['att_err']:.2f}° ω_err={r['omega_err']:.3f}°/s resid={r['resid']:.6f}")
    
    log(f"\nTotal time: {time.time()-t_start:.0f}s")

    out = {
        'n_seeds': N_SEEDS, 'n_candidates': len(candidates), 'n_optimized': N_OPT,
        'principal_moments': [float(I1), float(I2), float(I3)],
        'analytic_speedup': float(t_numeric/t_analytic),
        'n_results': len(all_results),
        'results_top100': all_results[:100],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_q0_omega_analytic.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
