#!/usr/bin/env python3
"""
3-Epoch Omega Bridge Experiment.
1. Generate lightcurve, pick 3 brightness-separated epochs
2. 10k lo-fi candidates per epoch (8 parallel workers)
3. Pairwise omega filtering (rotation angle < ω_bound × Δt)
4. Triplet consistency (ω₁₂ ≈ ω₂₃)
5. Report survivors and whether truth is among them
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
n_observations = 100
OBSERVER_ID = 399999
noise_sigma = 0.05

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
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

# True parameters
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))

# Propagate true attitude
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

log(f"Setup: {time.time()-t_start:.1f}s")

# ─── Generate observed lightcurve (hi-fi) ───
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

# ─── Step 1: Pick 3 brightness-separated epochs ───
BRIGHTNESS_SEP = 0.5  # mag

ep1_idx = 0
ep2_idx = None
for i in range(1, n_observations):
    if abs(observed_lc[i] - observed_lc[ep1_idx]) >= BRIGHTNESS_SEP:
        ep2_idx = i
        break

if ep2_idx is None:
    log(f"ERROR: Could not find epoch 2 with {BRIGHTNESS_SEP} mag separation!")
    sys.exit(1)

ep3_idx = None
for i in range(ep2_idx + 1, n_observations):
    if abs(observed_lc[i] - observed_lc[ep2_idx]) >= BRIGHTNESS_SEP:
        ep3_idx = i
        break

if ep3_idx is None:
    log(f"ERROR: Could not find epoch 3 with {BRIGHTNESS_SEP} mag separation!")
    sys.exit(1)

epoch_indices = [ep1_idx, ep2_idx, ep3_idx]
epoch_times = [observation_times[i] for i in epoch_indices]
epoch_mags = [observed_lc[i] for i in epoch_indices]

log(f"\nSelected epochs:")
for j, ei in enumerate(epoch_indices):
    log(f"  Epoch {j+1}: index={ei}, t={observation_times[ei]:.0f}s, mag={observed_lc[ei]:.3f}")
log(f"  Δt₁₂={epoch_times[1]-epoch_times[0]:.0f}s, Δt₂₃={epoch_times[2]-epoch_times[1]:.0f}s, Δt₁₃={epoch_times[2]-epoch_times[0]:.0f}s")

# True attitudes at selected epochs
true_qs = [true_quaternions[i] for i in epoch_indices]
true_Rs = [Rotation.from_quat([q[1],q[2],q[3],q[0]]) for q in true_qs]

# True omega between epochs (for reference)
for a, b, la, lb in [(0,1,"1","2"), (1,2,"2","3"), (0,2,"1","3")]:
    dR = true_Rs[b] * true_Rs[a].inv()
    rv = dR.as_rotvec()
    dt = epoch_times[b] - epoch_times[a]
    omega = np.rad2deg(rv / dt)
    log(f"  True ω_{la}{lb}: [{omega[0]:.4f}, {omega[1]:.4f}, {omega[2]:.4f}] °/s, |ω|={np.linalg.norm(omega):.4f} °/s")

# ─── Brightness evaluation helpers ───
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

# Worker globals
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
        return {'rotvec': res.x.tolist(), 'quat': q.tolist(), 'resid': float(resid), 'att_err': float(err)}
    except:
        return None

if __name__ == '__main__':
    omega_bound_deg = 0.198  # °/s from FFT
    N_SEEDS = 10000
    
    # ─── Step 2: Lo-fi candidates at each epoch ───
    all_candidates = {}
    
    for j, ei in enumerate(epoch_indices):
        log(f"\n{'='*70}")
        log(f"EPOCH {j+1} (index={ei}, mag={observed_lc[ei]:.3f}): {N_SEEDS} lo-fi seeds")
        log(f"{'='*70}")
        
        # Lo-fi target
        true_mag_hifi = brightness_hifi(true_qs[j], ei)
        true_mag_lofi = brightness_lofi(true_qs[j], ei)
        lofi_bias = true_mag_lofi - true_mag_hifi
        target_lofi = observed_lc[ei] + lofi_bias
        
        log(f"  Hi-fi observed: {observed_lc[ei]:.4f}, Lo-fi target: {target_lofi:.4f} (bias: {lofi_bias:.4f})")
        
        seeds = Rotation.random(N_SEEDS, random_state=2000 + j)
        args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]
        
        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_lofi, true_qs[j], ei)) as pool:
            results = pool.map(lofi_worker, args)
        dt = time.time() - t0
        
        good = [r for r in results if r is not None and r['resid'] < noise_sigma]
        good_errs = sorted([r['att_err'] for r in good])
        
        log(f"  Converged: {len(good)}/{N_SEEDS} in {dt:.0f}s")
        log(f"  Min att error: {good_errs[0]:.2f}°" if good_errs else "  NO GOOD RESULTS")
        log(f"  Within 5°: {sum(1 for e in good_errs if e < 5)}")
        
        all_candidates[j] = good
    
    # ─── Step 3: Pairwise omega filtering ───
    log(f"\n{'='*70}")
    log("PAIRWISE OMEGA FILTERING")
    log(f"{'='*70}")
    
    bridges = [(0, 1), (1, 2), (0, 2)]
    bridge_pairs = {}
    
    for a, b in bridges:
        dt_ab = epoch_times[b] - epoch_times[a]
        max_angle_deg = omega_bound_deg * dt_ab
        
        log(f"\n  Bridge {a+1}→{b+1}: Δt={dt_ab:.0f}s, max_rotation={max_angle_deg:.1f}°")
        
        if max_angle_deg > 180:
            log(f"  SKIP: max_angle > 180°, no culling power")
            bridge_pairs[(a, b)] = None
            continue
        
        cands_a = all_candidates[a]
        cands_b = all_candidates[b]
        
        # Vectorized: compute all pairwise rotation angles
        # q format is w,x,y,z — convert to scipy x,y,z,w
        quats_a = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_a])
        quats_b = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_b])
        
        na, nb = len(cands_a), len(cands_b)
        log(f"  Candidates: {na} × {nb} = {na*nb:,} pairs")
        
        # For rotation angle: |angle(R_b * R_a^-1)| 
        # Using quaternion: angle = 2*arccos(|q_a · q_b|) for relative rotation
        # But we need actual rotation, not just angle from quaternion dot product
        # q_rel = q_b * q_a^{-1}, angle = 2*arccos(|q_rel.w|)
        # q_a^{-1} = [w, -x, -y, -z] for unit quaternion
        # q_rel = q_b * conj(q_a)
        
        # Vectorized quaternion multiplication: q_b * conj(q_a)
        # conj(q_a) in x,y,z,w format: [-x, -y, -z, w]
        
        # Actually simpler: rotation angle between two orientations
        # angle = 2 * arccos(|q_a · q_b|)  (this is the geodesic distance on SO(3))
        # But this gives the angle of q_b * q_a^{-1} only if both are unit quaternions
        # Actually: angle(R_b R_a^{-1}) = 2 * arccos(|<q_a, q_b>|)
        
        # Compute dot products: na × nb matrix
        # dots[i,j] = |q_a[i] · q_b[j]|
        t_filter = time.time()
        
        # Process in chunks to manage memory (100M floats = 800MB)
        CHUNK = 2000
        surviving_pairs = []
        max_angle_rad = np.deg2rad(max_angle_deg)
        
        for i_start in range(0, na, CHUNK):
            i_end = min(i_start + CHUNK, na)
            chunk_a = quats_a[i_start:i_end]  # (chunk, 4)
            
            # Dot product: (chunk, 4) @ (4, nb) → (chunk, nb)
            dots = np.abs(chunk_a @ quats_b.T)
            dots = np.clip(dots, 0, 1)
            angles = 2 * np.arccos(dots)  # rotation angles in radians
            
            # Find pairs within max_angle
            valid_i, valid_j = np.where(angles < max_angle_rad)
            valid_i += i_start  # offset back to global index
            
            for ii, jj in zip(valid_i, valid_j):
                surviving_pairs.append((int(ii), int(jj)))
        
        dt_filter = time.time() - t_filter
        log(f"  Surviving pairs: {len(surviving_pairs):,} / {na*nb:,} ({100*len(surviving_pairs)/(na*nb):.2f}%)")
        log(f"  Filter time: {dt_filter:.1f}s")
        
        bridge_pairs[(a, b)] = surviving_pairs
    
    # ─── Step 4: Triplet consistency ───
    log(f"\n{'='*70}")
    log("TRIPLET CONSISTENCY (ω₁₂ ≈ ω₂₃)")
    log(f"{'='*70}")
    
    OMEGA_THRESHOLD_DEG = 0.05  # °/s
    
    pairs_12 = bridge_pairs.get((0, 1))
    pairs_23 = bridge_pairs.get((1, 2))
    pairs_13 = bridge_pairs.get((0, 2))
    
    if pairs_12 is None or pairs_23 is None:
        log("  Cannot do triplet matching — one or more bridges skipped (Δt too large)")
        log("  Trying with available bridges only...")
        # Fall through to report
        surviving_triplets = []
    else:
        # Build index: for each epoch-2 candidate, which epoch-1 and epoch-3 candidates pair with it?
        from collections import defaultdict
        ep2_to_ep1 = defaultdict(list)  # ep2_idx → list of ep1_idx
        ep2_to_ep3 = defaultdict(list)  # ep2_idx → list of ep3_idx
        
        for i1, i2 in pairs_12:
            ep2_to_ep1[i2].append(i1)
        for i2, i3 in pairs_23:
            ep2_to_ep3[i2].append(i3)
        
        # For each epoch-2 candidate that appears in both bridges
        common_ep2 = set(ep2_to_ep1.keys()) & set(ep2_to_ep3.keys())
        log(f"  Epoch-2 candidates in both bridges: {len(common_ep2)}")
        
        # Compute omega vectors for matching triplets
        cands = [all_candidates[j] for j in range(3)]
        dt_12 = epoch_times[1] - epoch_times[0]
        dt_23 = epoch_times[2] - epoch_times[1]
        
        surviving_triplets = []
        n_tested = 0
        
        for i2 in common_ep2:
            q2 = cands[1][i2]['quat']
            R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
            
            for i1 in ep2_to_ep1[i2]:
                q1 = cands[0][i1]['quat']
                R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                
                dR_12 = R2 * R1.inv()
                omega_12 = np.rad2deg(dR_12.as_rotvec() / dt_12)  # °/s
                
                for i3 in ep2_to_ep3[i2]:
                    q3 = cands[2][i3]['quat']
                    R3 = Rotation.from_quat([q3[1], q3[2], q3[3], q3[0]])
                    
                    dR_23 = R3 * R2.inv()
                    omega_23 = np.rad2deg(dR_23.as_rotvec() / dt_23)  # °/s
                    
                    omega_diff = np.linalg.norm(omega_12 - omega_23)
                    n_tested += 1
                    
                    if omega_diff < OMEGA_THRESHOLD_DEG:
                        # Also check 1→3 bridge if available
                        dR_13 = R3 * R1.inv()
                        dt_13 = epoch_times[2] - epoch_times[0]
                        omega_13 = np.rad2deg(dR_13.as_rotvec() / dt_13)
                        
                        omega_avg = (omega_12 + omega_23) / 2
                        omega_13_diff = np.linalg.norm(omega_13 - omega_avg)
                        
                        att_errs = [cands[0][i1]['att_err'], cands[1][i2]['att_err'], cands[2][i3]['att_err']]
                        
                        surviving_triplets.append({
                            'indices': (i1, i2, i3),
                            'att_errs': att_errs,
                            'omega_12': omega_12.tolist(),
                            'omega_23': omega_23.tolist(),
                            'omega_13': omega_13.tolist(),
                            'omega_diff_12_23': float(omega_diff),
                            'omega_diff_13': float(omega_13_diff),
                            'max_att_err': max(att_errs)
                        })
        
        log(f"  Triplets tested: {n_tested:,}")
        log(f"  Triplets surviving (ω threshold={OMEGA_THRESHOLD_DEG}°/s): {len(surviving_triplets)}")
    
    # ─── Step 5: Report ───
    log(f"\n{'='*70}")
    log("RESULTS")
    log(f"{'='*70}")
    
    log(f"\nCandidates per epoch:")
    for j in range(3):
        n = len(all_candidates[j])
        errs = sorted([c['att_err'] for c in all_candidates[j]])
        log(f"  Epoch {j+1}: {n} candidates, min_err={errs[0]:.2f}°, within_5°={sum(1 for e in errs if e<5)}")
    
    log(f"\nBridge filtering:")
    for a, b in bridges:
        pairs = bridge_pairs.get((a, b))
        if pairs is None:
            log(f"  {a+1}→{b+1}: SKIPPED (Δt too large)")
        else:
            log(f"  {a+1}→{b+1}: {len(pairs):,} surviving pairs")
    
    log(f"\nTriplet results:")
    log(f"  Surviving triplets: {len(surviving_triplets)}")
    
    if surviving_triplets:
        # Check if truth is among survivors
        truth_triplets = [t for t in surviving_triplets if max(t['att_errs']) < 10]
        close_triplets = [t for t in surviving_triplets if max(t['att_errs']) < 5]
        
        log(f"  Triplets with all att_err < 10°: {len(truth_triplets)}")
        log(f"  Triplets with all att_err < 5°: {len(close_triplets)}")
        
        # Sort by max att error
        surviving_triplets.sort(key=lambda t: max(t['att_errs']))
        
        log(f"\n  Top-10 triplets by max attitude error:")
        for i, t in enumerate(surviving_triplets[:10]):
            log(f"    {i+1}. att_errs=[{t['att_errs'][0]:.1f}°, {t['att_errs'][1]:.1f}°, {t['att_errs'][2]:.1f}°] "
                f"ω_diff_12_23={t['omega_diff_12_23']:.4f}°/s ω_diff_13={t['omega_diff_13']:.4f}°/s")
        
        log(f"\n  Top-10 triplets by ω consistency (lowest ω_diff):")
        by_omega = sorted(surviving_triplets, key=lambda t: t['omega_diff_12_23'])
        for i, t in enumerate(by_omega[:10]):
            log(f"    {i+1}. att_errs=[{t['att_errs'][0]:.1f}°, {t['att_errs'][1]:.1f}°, {t['att_errs'][2]:.1f}°] "
                f"ω_diff_12_23={t['omega_diff_12_23']:.4f}°/s")
    
    # Also try relaxed thresholds
    for thresh in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        n_surv = sum(1 for t in surviving_triplets if t['omega_diff_12_23'] < thresh) if surviving_triplets else 0
        log(f"  ω threshold {thresh:.2f}°/s: {n_surv} triplets")
    
    log(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Save
    out = {
        'epoch_indices': epoch_indices,
        'epoch_times': epoch_times,
        'epoch_mags': epoch_mags,
        'n_candidates': [len(all_candidates[j]) for j in range(3)],
        'n_surviving_triplets': len(surviving_triplets),
        'triplets': surviving_triplets[:100],  # save top 100
        'omega_bound_deg': omega_bound_deg,
        'omega_threshold_deg': OMEGA_THRESHOLD_DEG,
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_omega_bridge.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
