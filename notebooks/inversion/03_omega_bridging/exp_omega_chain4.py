#!/usr/bin/env python3
"""
Experiment A: 4-Epoch Chain with Adjacent Bridges
Uses 4 consecutive epochs (all Δt=36s), 3 adjacent bridges.
Chain consistency: A→B→C→D requires shared intermediate candidates.
"""
import sys, time, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from multiprocessing import Pool
from collections import defaultdict

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

# ─── Setup (same as v2) ───
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

# True parameters (realistic tumbling)
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.5, -0.3, 2.0]))

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

# ─── Step 1: Pick 4 consecutive epochs with max brightness variation ───
# Score each window of 4 consecutive epochs by brightness range
dt_sampling = observation_times[1] - observation_times[0]
log(f"Epoch spacing: {dt_sampling:.1f}s")

best_score = 0
best_start = 0
for i in range(n_observations - 3):
    window_mags = observed_lc[i:i+4]
    score = np.ptp(window_mags)  # range = max - min
    if score > best_score:
        best_score = score
        best_start = i

epoch_indices = list(range(best_start, best_start + 4))
epoch_times_sel = [observation_times[i] for i in epoch_indices]
epoch_mags = [observed_lc[i] for i in epoch_indices]

log(f"\nSelected 4 consecutive epochs (max brightness variation = {best_score:.3f} mag):")
for j, ei in enumerate(epoch_indices):
    log(f"  Epoch {j}: index={ei}, t={observation_times[ei]:.0f}s, mag={observed_lc[ei]:.3f}")

# True attitudes at selected epochs
true_qs = [true_quaternions[i] for i in epoch_indices]
true_Rs = [Rotation.from_quat([q[1],q[2],q[3],q[0]]) for q in true_qs]

# True omega between adjacent epochs
for a in range(3):
    b = a + 1
    dR = true_Rs[b] * true_Rs[a].inv()
    rv = dR.as_rotvec()
    dt = epoch_times_sel[b] - epoch_times_sel[a]
    omega = np.rad2deg(rv / dt)
    log(f"  True ω_{a}{b}: [{omega[0]:.4f}, {omega[1]:.4f}, {omega[2]:.4f}] °/s, |ω|={np.linalg.norm(omega):.4f} °/s")

# ─── Brightness helpers ───
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
    # FFT omega bound
    from scipy.fft import rfft, rfftfreq
    fft_vals = np.abs(rfft(observed_lc - np.mean(observed_lc)))
    freqs = rfftfreq(len(observed_lc), d=dt_sampling)
    peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
    omega_bound_deg = peak_freq * 360.0
    log(f"FFT omega bound: {omega_bound_deg:.3f} °/s (true |ω|: {np.linalg.norm(np.rad2deg(true_omega0)):.3f} °/s)")

    N_SEEDS = 10000

    # ─── Step 2: Lo-fi candidates at each of 4 epochs ───
    all_candidates = {}
    for j, ei in enumerate(epoch_indices):
        log(f"\n{'='*60}")
        log(f"EPOCH {j} (index={ei}, mag={observed_lc[ei]:.3f}): {N_SEEDS} lo-fi seeds")

        true_mag_hifi = brightness_hifi(true_qs[j], ei)
        true_mag_lofi = brightness_lofi(true_qs[j], ei)
        lofi_bias = true_mag_lofi - true_mag_hifi
        target_lofi = observed_lc[ei] + lofi_bias
        log(f"  Hi-fi: {observed_lc[ei]:.4f}, Lo-fi target: {target_lofi:.4f} (bias: {lofi_bias:.4f})")

        seeds = Rotation.random(N_SEEDS, random_state=3000 + j)
        args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]

        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_lofi, true_qs[j], ei)) as pool:
            results = pool.map(lofi_worker, args)
        dt = time.time() - t0

        good = [r for r in results if r is not None and r['resid'] < noise_sigma]
        good_errs = sorted([r['att_err'] for r in good])
        log(f"  Converged: {len(good)}/{N_SEEDS} in {dt:.0f}s")
        if good_errs:
            log(f"  Min att error: {good_errs[0]:.2f}°, within 5°: {sum(1 for e in good_errs if e < 5)}")
        all_candidates[j] = good

    # ─── Step 3: Adjacent bridge filtering ───
    log(f"\n{'='*60}")
    log("ADJACENT BRIDGE FILTERING")

    bridges = [(0, 1), (1, 2), (2, 3)]
    bridge_pairs = {}

    for a, b in bridges:
        dt_ab = epoch_times_sel[b] - epoch_times_sel[a]
        max_angle_deg = omega_bound_deg * dt_ab
        log(f"\n  Bridge {a}→{b}: Δt={dt_ab:.0f}s, max_rotation={max_angle_deg:.1f}°")

        if max_angle_deg > 180:
            log(f"  SKIP: max_angle > 180°")
            bridge_pairs[(a, b)] = None
            continue

        cands_a = all_candidates[a]
        cands_b = all_candidates[b]

        quats_a = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_a])
        quats_b = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_b])

        na, nb = len(cands_a), len(cands_b)
        log(f"  Candidates: {na} × {nb} = {na*nb:,} pairs")

        max_angle_rad = np.deg2rad(max_angle_deg)
        surviving = []
        CHUNK = 2000
        t_f = time.time()

        for i_start in range(0, na, CHUNK):
            i_end = min(i_start + CHUNK, na)
            chunk_a = quats_a[i_start:i_end]
            dots = np.abs(chunk_a @ quats_b.T)
            dots = np.clip(dots, 0, 1)
            angles = 2 * np.arccos(dots)
            valid_i, valid_j = np.where(angles < max_angle_rad)
            valid_i += i_start
            for ii, jj in zip(valid_i, valid_j):
                surviving.append((int(ii), int(jj)))

        log(f"  Surviving: {len(surviving):,} / {na*nb:,} ({100*len(surviving)/(na*nb):.2f}%) in {time.time()-t_f:.1f}s")
        bridge_pairs[(a, b)] = surviving

    # ─── Step 4: Chain consistency A→B→C→D ───
    log(f"\n{'='*60}")
    log("CHAIN CONSISTENCY: A→B→C→D")

    # Check all bridges exist
    all_bridges_ok = all(bridge_pairs.get((a, a+1)) is not None for a in range(3))
    if not all_bridges_ok:
        log("  FAIL: one or more bridges skipped")
        surviving_chains = []
    else:
        # Forward chain: intersect at shared intermediate nodes
        # Bridge 0→1: set of (i0, i1) pairs
        # Bridge 1→2: set of (i1, i2) pairs
        # Bridge 2→3: set of (i2, i3) pairs
        # Chain: require i1 in bridge01 matches i1 in bridge12, etc.

        pairs_01 = bridge_pairs[(0, 1)]
        pairs_12 = bridge_pairs[(1, 2)]
        pairs_23 = bridge_pairs[(2, 3)]

        # Build forward index
        log(f"  Bridge 0→1: {len(pairs_01):,} pairs")
        log(f"  Bridge 1→2: {len(pairs_12):,} pairs")
        log(f"  Bridge 2→3: {len(pairs_23):,} pairs")

        # i1→[i0s] from bridge 0→1
        i1_to_i0 = defaultdict(list)
        for i0, i1 in pairs_01:
            i1_to_i0[i1].append(i0)

        # i1→[i2s] from bridge 1→2
        i1_to_i2 = defaultdict(list)
        for i1, i2 in pairs_12:
            i1_to_i2[i1].append(i2)

        # i2→[i3s] from bridge 2→3
        i2_to_i3 = defaultdict(list)
        for i2, i3 in pairs_23:
            i2_to_i3[i2].append(i3)

        # Chain: for each i1 present in both bridge01 and bridge12
        common_i1 = set(i1_to_i0.keys()) & set(i1_to_i2.keys())
        log(f"  Epoch-1 candidates in both bridges 0→1 and 1→2: {len(common_i1)}")

        # Now additionally require omega consistency along the chain
        # For each chain (i0, i1, i2, i3), compute ω₀₁, ω₁₂, ω₂₃ and require consistency
        OMEGA_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        cands = [all_candidates[j] for j in range(4)]
        dt_01 = epoch_times_sel[1] - epoch_times_sel[0]
        dt_12 = epoch_times_sel[2] - epoch_times_sel[1]
        dt_23 = epoch_times_sel[3] - epoch_times_sel[2]

        # First: build chains without omega filtering (just topology)
        t_chain = time.time()
        
        # Precompute quaternion arrays for fast access
        quats = []
        for j in range(4):
            qs = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands[j]])
            quats.append(qs)

        # Count total chains via topology
        total_chains = 0
        chain_i1_i2_counts = 0
        for i1 in common_i1:
            i0_list = i1_to_i0[i1]
            i2_list = i1_to_i2[i1]
            for i2 in i2_list:
                i3_list = i2_to_i3.get(i2, [])
                if i3_list:
                    total_chains += len(i0_list) * len(i3_list)
                    chain_i1_i2_counts += 1

        log(f"  Topological chains (i0,i1,i2,i3): {total_chains:,}")
        log(f"  (i1,i2) pairs linking to both sides: {chain_i1_i2_counts:,}")

        # If too many chains, sample or use omega filtering directly
        # Let's do omega filtering on the chain
        surviving_chains = []
        n_checked = 0
        
        for i1 in common_i1:
            i0_list = i1_to_i0[i1]
            i2_list = i1_to_i2[i1]
            
            R1 = Rotation.from_quat(quats[1][i1])
            
            # Precompute ω₀₁ for all i0
            omega_01_list = []
            for i0 in i0_list:
                R0 = Rotation.from_quat(quats[0][i0])
                dR = R1 * R0.inv()
                omega_01_list.append(np.rad2deg(dR.as_rotvec() / dt_01))
            omega_01_arr = np.array(omega_01_list) if omega_01_list else np.empty((0,3))
            
            for i2 in i2_list:
                R2 = Rotation.from_quat(quats[2][i2])
                omega_12 = np.rad2deg((R2 * R1.inv()).as_rotvec() / dt_12)
                
                i3_list = i2_to_i3.get(i2, [])
                if not i3_list:
                    continue
                
                # Precompute ω₂₃ for all i3
                omega_23_list = []
                for i3 in i3_list:
                    R3 = Rotation.from_quat(quats[3][i3])
                    omega_23_list.append(np.rad2deg((R3 * R2.inv()).as_rotvec() / dt_23))
                omega_23_arr = np.array(omega_23_list)
                
                # Check ω₁₂ vs ω₀₁ (for each i0) and ω₁₂ vs ω₂₃ (for each i3)
                # Use largest threshold for initial collection
                max_thresh = max(OMEGA_THRESHOLDS)
                
                # Filter i0 by |ω₀₁ - ω₁₂| < max_thresh
                if len(omega_01_arr) > 0:
                    diff_01_12 = np.linalg.norm(omega_01_arr - omega_12, axis=1)
                    good_i0_mask = diff_01_12 < max_thresh
                else:
                    continue
                
                # Filter i3 by |ω₂₃ - ω₁₂| < max_thresh
                diff_23_12 = np.linalg.norm(omega_23_arr - omega_12, axis=1)
                good_i3_mask = diff_23_12 < max_thresh
                
                for idx0 in np.where(good_i0_mask)[0]:
                    i0 = i0_list[idx0]
                    omega_01 = omega_01_arr[idx0]
                    d_01_12 = diff_01_12[idx0]
                    
                    for idx3 in np.where(good_i3_mask)[0]:
                        i3 = i3_list[idx3]
                        omega_23 = omega_23_arr[idx3]
                        d_23_12 = diff_23_12[idx3]
                        d_01_23 = np.linalg.norm(omega_01 - omega_23)
                        max_diff = max(d_01_12, d_23_12, d_01_23)
                        
                        att_errs = [cands[k][idx]['att_err'] for k, idx in enumerate([i0, i1, i2, i3])]
                        
                        surviving_chains.append({
                            'indices': [i0, int(i1), i2, i3],
                            'att_errs': att_errs,
                            'omega_01': omega_01.tolist(),
                            'omega_12': omega_12.tolist(),
                            'omega_23': omega_23.tolist(),
                            'diff_01_12': float(d_01_12),
                            'diff_12_23': float(d_23_12),
                            'diff_01_23': float(d_01_23),
                            'max_omega_diff': float(max_diff),
                            'max_att_err': float(max(att_errs))
                        })
                
                n_checked += 1
                if n_checked % 5000 == 0:
                    log(f"    Checked {n_checked} (i1,i2) pairs, {len(surviving_chains)} chains so far ({time.time()-t_chain:.0f}s)")

        dt_chain = time.time() - t_chain
        log(f"  Chain matching time: {dt_chain:.1f}s")
        log(f"  Total chains (max ω threshold {max(OMEGA_THRESHOLDS)}°/s): {len(surviving_chains)}")

    # ─── Step 5: Results ───
    log(f"\n{'='*60}")
    log("RESULTS")
    log(f"{'='*60}")

    log(f"\nCandidates per epoch:")
    for j in range(4):
        n = len(all_candidates[j])
        errs = sorted([c['att_err'] for c in all_candidates[j]])
        within5 = sum(1 for e in errs if e < 5)
        log(f"  Epoch {j}: {n} candidates, min_err={errs[0]:.2f}°, within_5°={within5}")

    log(f"\nBridge filtering:")
    for a in range(3):
        pairs = bridge_pairs.get((a, a+1))
        if pairs is None:
            log(f"  {a}→{a+1}: SKIPPED")
        else:
            na = len(all_candidates[a])
            nb = len(all_candidates[a+1])
            log(f"  {a}→{a+1}: {len(pairs):,} / {na*nb:,} ({100*len(pairs)/(na*nb):.2f}%)")

    log(f"\nChain results at different ω thresholds:")
    for thresh in OMEGA_THRESHOLDS:
        chains_t = [c for c in surviving_chains if c['max_omega_diff'] < thresh]
        n_t = len(chains_t)
        truth_t = sum(1 for c in chains_t if c['max_att_err'] < 10)
        close_t = sum(1 for c in chains_t if c['max_att_err'] < 5)
        log(f"  ω < {thresh:.2f}°/s: {n_t:,} chains, truth(<10°)={truth_t}, close(<5°)={close_t}")

    if surviving_chains:
        surviving_chains.sort(key=lambda c: c['max_omega_diff'])
        log(f"\n  Top-10 chains by ω consistency:")
        for i, c in enumerate(surviving_chains[:10]):
            errs = c['att_errs']
            log(f"    {i+1}. att=[{errs[0]:.1f}°,{errs[1]:.1f}°,{errs[2]:.1f}°,{errs[3]:.1f}°] "
                f"max_ω_diff={c['max_omega_diff']:.4f}°/s")

        surviving_chains.sort(key=lambda c: c['max_att_err'])
        log(f"\n  Top-10 chains by max attitude error:")
        for i, c in enumerate(surviving_chains[:10]):
            errs = c['att_errs']
            log(f"    {i+1}. att=[{errs[0]:.1f}°,{errs[1]:.1f}°,{errs[2]:.1f}°,{errs[3]:.1f}°] "
                f"max_ω_diff={c['max_omega_diff']:.4f}°/s")

    # Unique epoch-0 attitudes in surviving chains at different thresholds
    log(f"\nUnique epoch-0 candidates at different ω thresholds:")
    for thresh in OMEGA_THRESHOLDS:
        chains_t = [c for c in surviving_chains if c['max_omega_diff'] < thresh]
        unique_i0 = set(c['indices'][0] for c in chains_t)
        log(f"  ω < {thresh:.2f}°/s: {len(unique_i0)} unique epoch-0 attitudes")

    log(f"\nTotal time: {time.time()-t_start:.0f}s")

    # Save
    out = {
        'epoch_indices': epoch_indices,
        'epoch_times': epoch_times_sel,
        'epoch_mags': epoch_mags,
        'n_candidates': [len(all_candidates[j]) for j in range(4)],
        'omega_bound_deg': float(omega_bound_deg),
        'n_bridges': {f"{a}_{a+1}": len(bridge_pairs[(a,a+1)]) if bridge_pairs.get((a,a+1)) else 0 for a in range(3)},
        'n_surviving_chains': len(surviving_chains),
        'chains_top100': surviving_chains[:100] if surviving_chains else [],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_omega_chain4.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
