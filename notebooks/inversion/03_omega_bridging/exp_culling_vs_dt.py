#!/usr/bin/env python3
"""
Quick test: how does bridge culling power scale with Δt (epoch spacing)?
Use existing 10k candidates from 2 epochs, vary the simulated Δt,
measure what fraction of pairs survive the rotation-angle filter.
Also measure: at each Δt, does tighter culling preferentially retain truth?
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

# Generate high-cadence lightcurve: 500 epochs over 1 hour
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

# FFT omega bound
from scipy.fft import rfft, rfftfreq
fft_vals = np.abs(rfft(observed_lc - np.mean(observed_lc)))
freqs = rfftfreq(len(observed_lc), d=dt_sampling)
peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
omega_bound_deg = peak_freq * 360.0
log(f"FFT omega bound: {omega_bound_deg:.3f} °/s")

# ─── Run lo-fi candidates at 2 epochs with different separations ───
# Pick epoch 0 as anchor. Test separations: 1, 2, 5, 10, 20, 50 epochs apart
# (Δt = 7.2s, 14.4s, 36s, 72s, 144s, 360s)

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

if __name__ == '__main__':
    N_SEEDS = 5000  # 5k per epoch to keep fast
    
    # Epoch pairs to test: anchor=0, second epoch at various offsets
    offsets = [1, 2, 5, 10, 20, 50]
    anchor_idx = 0
    
    # First: get candidates at anchor epoch
    log(f"\n{'='*60}")
    log(f"ANCHOR EPOCH (index={anchor_idx}, mag={observed_lc[anchor_idx]:.3f}): {N_SEEDS} seeds")
    
    true_q_anchor = true_quaternions[anchor_idx]
    lofi_bias_anchor = brightness_lofi(true_q_anchor, anchor_idx) - brightness_hifi(true_q_anchor, anchor_idx)
    target_anchor = observed_lc[anchor_idx] + lofi_bias_anchor
    
    seeds = Rotation.random(N_SEEDS, random_state=4000)
    args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]
    
    t0 = time.time()
    with Pool(8, initializer=init_worker, initargs=(target_anchor, true_q_anchor, anchor_idx)) as pool:
        results_anchor = pool.map(lofi_worker, args)
    dt = time.time() - t0
    
    cands_anchor = [r for r in results_anchor if r is not None and r['resid'] < noise_sigma]
    log(f"  Converged: {len(cands_anchor)}/{N_SEEDS} in {dt:.0f}s")
    errs_a = sorted([c['att_err'] for c in cands_anchor])
    log(f"  Min err: {errs_a[0]:.2f}°, within 5°: {sum(1 for e in errs_a if e<5)}")
    
    quats_anchor = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_anchor])
    
    # Now for each offset, get candidates and measure culling
    results_by_offset = []
    
    for offset in offsets:
        second_idx = anchor_idx + offset
        if second_idx >= n_observations:
            log(f"\nOffset {offset}: out of range, skip")
            continue
        
        dt_gap = observation_times[second_idx] - observation_times[anchor_idx]
        max_rot_deg = omega_bound_deg * dt_gap
        
        log(f"\n{'='*60}")
        log(f"OFFSET={offset} (epoch {second_idx}, Δt={dt_gap:.1f}s, max_rot={max_rot_deg:.1f}°, mag={observed_lc[second_idx]:.3f})")
        
        if max_rot_deg > 180:
            log(f"  SKIP: exceeds 180°")
            results_by_offset.append({
                'offset': offset, 'dt': float(dt_gap), 'max_rot_deg': float(max_rot_deg),
                'status': 'skip_180', 'cull_pct': 0, 'truth_survival': None
            })
            continue
        
        true_q_second = true_quaternions[second_idx]
        lofi_bias = brightness_lofi(true_q_second, second_idx) - brightness_hifi(true_q_second, second_idx)
        target_second = observed_lc[second_idx] + lofi_bias
        
        seeds2 = Rotation.random(N_SEEDS, random_state=5000 + offset)
        args2 = [seeds2[i].as_rotvec() for i in range(N_SEEDS)]
        
        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_second, true_q_second, second_idx)) as pool:
            results_second = pool.map(lofi_worker, args2)
        dt = time.time() - t0
        
        cands_second = [r for r in results_second if r is not None and r['resid'] < noise_sigma]
        log(f"  Converged: {len(cands_second)}/{N_SEEDS} in {dt:.0f}s")
        errs_s = sorted([c['att_err'] for c in cands_second])
        log(f"  Min err: {errs_s[0]:.2f}°, within 5°: {sum(1 for e in errs_s if e<5)}")
        
        quats_second = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in cands_second])
        
        na, nb = len(cands_anchor), len(cands_second)
        max_angle_rad = np.deg2rad(max_rot_deg)
        
        # Count surviving pairs (chunked to save memory)
        CHUNK = 1000
        total_surviving = 0
        # Also track: for anchor candidates near truth (<5°), how many second-epoch partners survive?
        truth_anchor_indices = [i for i, c in enumerate(cands_anchor) if c['att_err'] < 5]
        truth_second_indices = [i for i, c in enumerate(cands_second) if c['att_err'] < 5]
        
        # Per-anchor survival counts
        anchor_survival = np.zeros(na, dtype=int)
        
        for i_start in range(0, na, CHUNK):
            i_end = min(i_start + CHUNK, na)
            chunk = quats_anchor[i_start:i_end]
            dots = np.abs(chunk @ quats_second.T)
            dots = np.clip(dots, 0, 1)
            angles = 2 * np.arccos(dots)
            valid = angles < max_angle_rad
            anchor_survival[i_start:i_end] = valid.sum(axis=1)
            total_surviving += valid.sum()
        
        total_pairs = na * nb
        cull_pct = 100 * (1 - total_surviving / total_pairs)
        
        # How many partners do truth-anchor candidates have?
        truth_anchor_partners = [int(anchor_survival[i]) for i in truth_anchor_indices[:10]]
        avg_partners = np.mean(anchor_survival)
        truth_avg_partners = np.mean(truth_anchor_partners) if truth_anchor_partners else 0
        
        # Also: among truth-anchor's partners, how many are truth-second?
        # Check for first truth-anchor candidate
        truth_pair_count = 0
        if truth_anchor_indices and truth_second_indices:
            i_truth = truth_anchor_indices[0]
            q_ta = quats_anchor[i_truth]
            dots_t = np.abs(quats_second @ q_ta)
            dots_t = np.clip(dots_t, 0, 1)
            angles_t = 2 * np.arccos(dots_t)
            partners_t = np.where(angles_t < max_angle_rad)[0]
            truth_pair_count = sum(1 for j in partners_t if cands_second[j]['att_err'] < 5)
        
        log(f"  Pairs: {total_surviving:,} / {total_pairs:,} survive ({cull_pct:.1f}% culled)")
        log(f"  Avg partners per anchor candidate: {avg_partners:.0f}")
        log(f"  Truth-anchor avg partners: {truth_avg_partners:.0f}")
        log(f"  Truth-anchor → truth-second pairs: {truth_pair_count}")
        
        results_by_offset.append({
            'offset': offset,
            'dt': float(dt_gap),
            'max_rot_deg': float(max_rot_deg),
            'n_anchor': na, 'n_second': nb,
            'total_pairs': int(total_pairs),
            'surviving_pairs': int(total_surviving),
            'cull_pct': float(cull_pct),
            'avg_partners': float(avg_partners),
            'truth_anchor_partners': float(truth_avg_partners),
            'truth_truth_pairs': truth_pair_count,
            'n_truth_anchor': len(truth_anchor_indices),
            'n_truth_second': len(truth_second_indices),
            'status': 'ok'
        })
    
    # ─── Summary ───
    log(f"\n{'='*60}")
    log("SUMMARY: Culling power vs epoch spacing")
    log(f"{'='*60}")
    log(f"{'Offset':>6} {'Δt(s)':>7} {'MaxRot°':>8} {'Culled%':>8} {'AvgPart':>8} {'TruthPart':>10} {'T→T pairs':>10}")
    for r in results_by_offset:
        if r['status'] == 'skip_180':
            log(f"{r['offset']:>6} {r['dt']:>7.1f} {r['max_rot_deg']:>8.1f}    SKIP (>180°)")
        else:
            log(f"{r['offset']:>6} {r['dt']:>7.1f} {r['max_rot_deg']:>8.1f} {r['cull_pct']:>8.1f} {r['avg_partners']:>8.0f} {r['truth_anchor_partners']:>10.0f} {r['truth_truth_pairs']:>10}")
    
    log(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_culling_vs_dt.json'
    with open(outpath, 'w') as f:
        json.dump({'omega_bound_deg': float(omega_bound_deg), 'n_seeds': N_SEEDS, 'results': results_by_offset,
                   'total_time': time.time()-t_start}, f, indent=2)
    log(f"Saved: {outpath}")
