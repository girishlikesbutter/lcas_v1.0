#!/usr/bin/env python3
"""
7-Epoch Chain with 7.2s spacing (99% culling per bridge).
Forward-propagation chain: at each step, only keep candidates
consistent with at least one survivor from the previous epoch.
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
max_angle_rad = np.deg2rad(omega_bound_deg * dt_sampling)
max_angle_deg = np.rad2deg(max_angle_rad)
log(f"FFT omega bound: {omega_bound_deg:.3f} °/s, max rotation per step: {max_angle_deg:.1f}°")

# ─── Pick 7 consecutive epochs with good brightness variation ───
N_CHAIN = 10  # try 10 epochs for stronger filtering
best_score = 0
best_start = 0
for i in range(n_observations - N_CHAIN + 1):
    score = np.ptp(observed_lc[i:i+N_CHAIN])
    if score > best_score:
        best_score = score
        best_start = i

epoch_indices = list(range(best_start, best_start + N_CHAIN))
log(f"\nSelected {N_CHAIN} consecutive epochs starting at index {best_start} (brightness range={best_score:.3f} mag):")
for j, ei in enumerate(epoch_indices):
    log(f"  Epoch {j}: idx={ei}, t={observation_times[ei]:.1f}s, mag={observed_lc[ei]:.3f}")

true_qs = [true_quaternions[i] for i in epoch_indices]

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
    N_SEEDS = 5000

    # ─── Step 1: Lo-fi candidates at each epoch ───
    all_candidates = []
    all_quats = []  # scipy xyzw format

    for j, ei in enumerate(epoch_indices):
        log(f"\n{'='*60}")
        log(f"EPOCH {j} (idx={ei}, mag={observed_lc[ei]:.3f}): {N_SEEDS} seeds")

        true_mag_lofi = brightness_lofi(true_qs[j], ei)
        true_mag_hifi = brightness_hifi(true_qs[j], ei)
        lofi_bias = true_mag_lofi - true_mag_hifi
        target_lofi = observed_lc[ei] + lofi_bias
        log(f"  Lo-fi target: {target_lofi:.4f} (bias: {lofi_bias:.4f})")

        seeds = Rotation.random(N_SEEDS, random_state=6000 + j)
        args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]

        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_lofi, true_qs[j], ei)) as pool:
            results = pool.map(lofi_worker, args)
        dt_run = time.time() - t0

        good = [r for r in results if r is not None and r['resid'] < noise_sigma]
        good.sort(key=lambda r: r['att_err'])
        log(f"  Converged: {len(good)}/{N_SEEDS} in {dt_run:.0f}s")
        if good:
            log(f"  Min att err: {good[0]['att_err']:.2f}°, within 5°: {sum(1 for r in good if r['att_err'] < 5)}")

        all_candidates.append(good)
        qs = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in good])
        all_quats.append(qs)

    # ─── Step 2: Forward chain filtering ───
    log(f"\n{'='*60}")
    log("FORWARD CHAIN FILTERING")
    log(f"Max rotation per step: {max_angle_deg:.1f}°")

    # Track which candidates survive at each epoch
    # Start: all epoch-0 candidates are alive
    # At each step: candidate at epoch j+1 survives if it has ≥1 partner at epoch j among survivors

    survivor_indices = [set(range(len(all_candidates[0])))]  # all epoch-0 candidates
    survivor_partner_count = [{}]  # epoch 0: no partners needed

    for step in range(N_CHAIN - 1):
        j_prev = step
        j_next = step + 1
        prev_survivors = survivor_indices[j_prev]

        if not prev_survivors:
            log(f"\n  Step {j_prev}→{j_next}: NO survivors from previous epoch! Chain broken.")
            survivor_indices.append(set())
            survivor_partner_count.append({})
            continue

        quats_prev = all_quats[j_prev]
        quats_next = all_quats[j_next]
        n_prev_surv = len(prev_survivors)
        n_next = len(quats_next)

        # Build subset of previous survivors' quaternions
        prev_surv_list = sorted(prev_survivors)
        prev_surv_quats = quats_prev[prev_surv_list]  # (n_prev_surv, 4)

        # For each next candidate, check if any previous survivor is within max_angle
        next_survivors = set()
        partner_counts = {}

        CHUNK = 1000
        for i_start in range(0, n_next, CHUNK):
            i_end = min(i_start + CHUNK, n_next)
            chunk_next = quats_next[i_start:i_end]  # (chunk, 4)

            # Rotation angles: (chunk, n_prev_surv)
            dots = np.abs(chunk_next @ prev_surv_quats.T)
            dots = np.clip(dots, 0, 1)
            angles = 2 * np.arccos(dots)

            # For each next candidate, count how many previous survivors are compatible
            compatible = angles < max_angle_rad  # (chunk, n_prev_surv)
            n_partners = compatible.sum(axis=1)

            for k in range(i_end - i_start):
                idx = i_start + k
                if n_partners[k] > 0:
                    next_survivors.add(idx)
                    partner_counts[idx] = int(n_partners[k])

        survivor_indices.append(next_survivors)
        survivor_partner_count.append(partner_counts)

        # Stats
        n_surv = len(next_survivors)
        truth_in = [i for i in next_survivors if all_candidates[j_next][i]['att_err'] < 5]
        truth_total = sum(1 for c in all_candidates[j_next] if c['att_err'] < 5)
        log(f"\n  Step {j_prev}→{j_next}: {n_prev_surv} prev survivors → {n_surv}/{n_next} next survive "
            f"({100*(1-n_surv/n_next):.1f}% culled)")
        log(f"    Truth candidates: {len(truth_in)}/{truth_total} survived")
        if partner_counts:
            pcs = list(partner_counts.values())
            log(f"    Partners: min={min(pcs)}, median={np.median(pcs):.0f}, max={max(pcs)}")

    # ─── Step 3: Results ───
    log(f"\n{'='*60}")
    log("CHAIN SURVIVAL SUMMARY")
    log(f"{'='*60}")
    log(f"{'Epoch':>6} {'Total':>7} {'Survived':>9} {'Culled%':>8} {'Truth(<5°)':>11} {'TruthSurv':>10}")

    for j in range(N_CHAIN):
        n_total = len(all_candidates[j])
        n_surv = len(survivor_indices[j])
        cull = 100 * (1 - n_surv / n_total) if n_total > 0 else 0
        truth_total = sum(1 for c in all_candidates[j] if c['att_err'] < 5)
        truth_surv = sum(1 for i in survivor_indices[j] if all_candidates[j][i]['att_err'] < 5)
        log(f"{j:>6} {n_total:>7} {n_surv:>9} {cull:>8.1f} {truth_total:>11} {truth_surv:>10}")

    # Final survivors at last epoch — trace back their attitude errors
    final_surv = survivor_indices[-1]
    if final_surv:
        final_errs = [all_candidates[-1][i]['att_err'] for i in final_surv]
        log(f"\nFinal epoch survivors: {len(final_surv)}")
        log(f"  Attitude errors: min={min(final_errs):.2f}°, median={np.median(final_errs):.1f}°, max={max(final_errs):.1f}°")
        log(f"  Within 5°: {sum(1 for e in final_errs if e < 5)}")
        log(f"  Within 10°: {sum(1 for e in final_errs if e < 10)}")

    # Epoch-0 survivors (traced forward through entire chain)
    # The real question: which epoch-0 candidates have a viable path through all epochs?
    # We need backward tracing for this — but forward filtering already constrains.
    # Let's just report epoch-0 survivors from the forward pass perspective.
    ep0_surv = survivor_indices[0]  # all of them (no culling at epoch 0 in forward pass)
    
    # Actually do backward pass too: epoch 0 candidate survives only if it connects to a survivor at epoch 1
    log(f"\n{'='*60}")
    log("BACKWARD PASS (propagate survival constraints back to epoch 0)")
    
    backward_survivors = [set() for _ in range(N_CHAIN)]
    backward_survivors[-1] = survivor_indices[-1].copy()
    
    for step in range(N_CHAIN - 2, -1, -1):
        j_curr = step
        j_next = step + 1
        next_surv = backward_survivors[j_next]
        
        if not next_surv:
            log(f"  Backward {j_next}→{j_curr}: NO next survivors, chain broken")
            continue
        
        curr_candidates = survivor_indices[j_curr]  # only consider forward-survivors
        quats_curr = all_quats[j_curr]
        quats_next = all_quats[j_next]
        
        next_surv_list = sorted(next_surv)
        next_surv_quats = quats_next[next_surv_list]
        
        curr_surv = set()
        for i in curr_candidates:
            q = quats_curr[i]
            dots = np.abs(next_surv_quats @ q)
            dots = np.clip(dots, 0, 1)
            angles = 2 * np.arccos(dots)
            if np.any(angles < max_angle_rad):
                curr_surv.add(i)
        
        backward_survivors[j_curr] = curr_surv
        truth_surv = sum(1 for i in curr_surv if all_candidates[j_curr][i]['att_err'] < 5)
        log(f"  Backward {j_next}→{j_curr}: {len(curr_surv)} survive (truth: {truth_surv})")
    
    log(f"\nFINAL BIDIRECTIONAL SURVIVAL:")
    log(f"{'Epoch':>6} {'Survived':>9} {'Truth(<5°)':>11}")
    for j in range(N_CHAIN):
        n_surv = len(backward_survivors[j])
        truth_surv = sum(1 for i in backward_survivors[j] if all_candidates[j][i]['att_err'] < 5)
        log(f"{j:>6} {n_surv:>9} {truth_surv:>11}")
    
    # Epoch 0 final survivors
    ep0_final = backward_survivors[0]
    if ep0_final:
        ep0_errs = sorted([all_candidates[0][i]['att_err'] for i in ep0_final])
        log(f"\nEpoch-0 final survivors: {len(ep0_final)}")
        log(f"  Errors: {[f'{e:.1f}°' for e in ep0_errs[:20]]}")
    
    log(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Save
    out = {
        'n_chain': N_CHAIN,
        'dt_sampling': float(dt_sampling),
        'max_angle_deg': float(max_angle_deg),
        'omega_bound_deg': float(omega_bound_deg),
        'epoch_indices': epoch_indices,
        'epoch_mags': [float(observed_lc[i]) for i in epoch_indices],
        'n_candidates': [len(all_candidates[j]) for j in range(N_CHAIN)],
        'forward_survivors': [len(survivor_indices[j]) for j in range(N_CHAIN)],
        'bidirectional_survivors': [len(backward_survivors[j]) for j in range(N_CHAIN)],
        'ep0_final_errors': [all_candidates[0][i]['att_err'] for i in sorted(backward_survivors[0])] if backward_survivors[0] else [],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_chain7_tight.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
