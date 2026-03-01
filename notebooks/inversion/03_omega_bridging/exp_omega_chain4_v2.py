#!/usr/bin/env python3
"""
Experiment A v2: 4-Epoch Chain with Adjacent Bridges (memory-safe)
Key fix: Don't store all bridge pairs. Instead:
1. For each epoch-1 candidate, find compatible epoch-0 and epoch-2 candidates via rotation angle
2. For each epoch-2 candidate in step 1, find compatible epoch-3 candidates
3. Check omega consistency along the chain
All done in streaming fashion — no giant pair lists.
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

# ─── Setup (same as before) ───
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

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.5, -0.3, 2.0]))

true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

log(f"Setup: {time.time()-t_start:.1f}s")

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

# ─── Pick 4 consecutive epochs with max brightness variation ───
dt_sampling = observation_times[1] - observation_times[0]
best_score = 0
best_start = 0
for i in range(n_observations - 3):
    score = np.ptp(observed_lc[i:i+4])
    if score > best_score:
        best_score = score
        best_start = i

epoch_indices = list(range(best_start, best_start + 4))
epoch_times_sel = [observation_times[i] for i in epoch_indices]

log(f"\nSelected epochs (max brightness variation = {best_score:.3f} mag):")
for j, ei in enumerate(epoch_indices):
    log(f"  Epoch {j}: index={ei}, t={observation_times[ei]:.0f}s, mag={observed_lc[ei]:.3f}")

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
    from scipy.fft import rfft, rfftfreq

    fft_vals = np.abs(rfft(observed_lc - np.mean(observed_lc)))
    freqs = rfftfreq(len(observed_lc), d=dt_sampling)
    peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
    omega_bound_deg = peak_freq * 360.0
    log(f"FFT omega bound: {omega_bound_deg:.3f} °/s (true |ω|: {np.linalg.norm(np.rad2deg(true_omega0)):.3f} °/s)")

    N_SEEDS = 10000

    # ─── Lo-fi candidates at each of 4 epochs ───
    all_candidates = {}
    for j, ei in enumerate(epoch_indices):
        log(f"\n{'='*60}")
        log(f"EPOCH {j} (index={ei}, mag={observed_lc[ei]:.3f}): {N_SEEDS} lo-fi seeds")

        true_mag_lofi = brightness_lofi(true_qs[j], ei)
        true_mag_hifi = brightness_hifi(true_qs[j], ei)
        lofi_bias = true_mag_lofi - true_mag_hifi
        target_lofi = observed_lc[ei] + lofi_bias
        log(f"  Lo-fi target: {target_lofi:.4f} (bias: {lofi_bias:.4f})")

        seeds = Rotation.random(N_SEEDS, random_state=3000 + j)
        args = [seeds[i].as_rotvec() for i in range(N_SEEDS)]

        t0 = time.time()
        with Pool(8, initializer=init_worker, initargs=(target_lofi, true_qs[j], ei)) as pool:
            results = pool.map(lofi_worker, args)
        dt = time.time() - t0

        good = [r for r in results if r is not None and r['resid'] < noise_sigma]
        good.sort(key=lambda r: r['att_err'])
        log(f"  Converged: {len(good)}/{N_SEEDS} in {dt:.0f}s")
        if good:
            log(f"  Min att error: {good[0]['att_err']:.2f}°, within 5°: {sum(1 for r in good if r['att_err'] < 5)}")
        all_candidates[j] = good

    # ─── Chain matching (streaming, memory-safe) ───
    log(f"\n{'='*60}")
    log("CHAIN MATCHING (streaming)")

    # Precompute quaternion arrays (scipy xyzw format)
    quats = []
    for j in range(4):
        qs = np.array([[c['quat'][1], c['quat'][2], c['quat'][3], c['quat'][0]] for c in all_candidates[j]])
        quats.append(qs)
        log(f"  Epoch {j}: {len(qs)} candidates")

    dt_adj = epoch_times_sel[1] - epoch_times_sel[0]  # all adjacent gaps equal
    max_angle_rad = np.deg2rad(omega_bound_deg * dt_adj)
    max_angle_deg = np.rad2deg(max_angle_rad)
    log(f"  Adjacent Δt: {dt_adj:.0f}s, max rotation: {max_angle_deg:.1f}°")

    if max_angle_deg > 180:
        log("  ABORT: max rotation > 180° even for adjacent epochs!")
        surviving_chains = []
    else:
        # Strategy: iterate over epoch-1 candidates (pivot).
        # For each i1: find compatible i0 (bridge 0→1) and i2 (bridge 1→2).
        # For each compatible i2: find compatible i3 (bridge 2→3).
        # Then check omega consistency across the chain.

        OMEGA_THRESH = 1.0  # °/s — collect all, filter later

        n0, n1, n2, n3 = [len(quats[j]) for j in range(4)]
        surviving_chains = []
        t_chain = time.time()
        
        for idx1 in range(n1):
            if idx1 % 1000 == 0 and idx1 > 0:
                log(f"    i1={idx1}/{n1}, chains so far: {len(surviving_chains)} ({time.time()-t_chain:.0f}s)")

            q1 = quats[1][idx1]
            R1 = Rotation.from_quat(q1)

            # Find compatible i0 candidates (rotation angle < max_angle_rad)
            dots_01 = np.abs(quats[0] @ q1)
            dots_01 = np.clip(dots_01, 0, 1)
            compat_i0 = np.where(2 * np.arccos(dots_01) < max_angle_rad)[0]

            if len(compat_i0) == 0:
                continue

            # Find compatible i2 candidates
            dots_12 = np.abs(quats[2] @ q1)
            dots_12 = np.clip(dots_12, 0, 1)
            compat_i2 = np.where(2 * np.arccos(dots_12) < max_angle_rad)[0]

            if len(compat_i2) == 0:
                continue

            # Compute ω₀₁ for all compatible i0
            omega_01_arr = np.zeros((len(compat_i0), 3))
            for k, i0 in enumerate(compat_i0):
                R0 = Rotation.from_quat(quats[0][i0])
                omega_01_arr[k] = np.rad2deg((R1 * R0.inv()).as_rotvec() / dt_adj)

            # Compute ω₁₂ for all compatible i2
            omega_12_arr = np.zeros((len(compat_i2), 3))
            for k, i2 in enumerate(compat_i2):
                R2 = Rotation.from_quat(quats[2][i2])
                omega_12_arr[k] = np.rad2deg((R2 * R1.inv()).as_rotvec() / dt_adj)

            # For each i2: find compatible i3 and compute ω₂₃
            for k2, i2 in enumerate(compat_i2):
                omega_12 = omega_12_arr[k2]

                # Filter i0 by |ω₀₁ - ω₁₂| < OMEGA_THRESH
                diff_01 = np.linalg.norm(omega_01_arr - omega_12, axis=1)
                good_i0_idx = np.where(diff_01 < OMEGA_THRESH)[0]
                if len(good_i0_idx) == 0:
                    continue

                # Find compatible i3
                q2 = quats[2][i2]
                dots_23 = np.abs(quats[3] @ q2)
                dots_23 = np.clip(dots_23, 0, 1)
                compat_i3 = np.where(2 * np.arccos(dots_23) < max_angle_rad)[0]

                if len(compat_i3) == 0:
                    continue

                R2 = Rotation.from_quat(q2)
                # Compute ω₂₃ for compatible i3
                for i3 in compat_i3:
                    R3 = Rotation.from_quat(quats[3][i3])
                    omega_23 = np.rad2deg((R3 * R2.inv()).as_rotvec() / dt_adj)

                    diff_12_23 = np.linalg.norm(omega_12 - omega_23)
                    if diff_12_23 > OMEGA_THRESH:
                        continue

                    # This i3 is consistent with i2. Now pair with good i0s.
                    for ki0 in good_i0_idx:
                        i0 = compat_i0[ki0]
                        omega_01 = omega_01_arr[ki0]
                        d_01_12 = diff_01[ki0]
                        d_01_23 = np.linalg.norm(omega_01 - omega_23)
                        max_diff = max(d_01_12, diff_12_23, d_01_23)

                        if max_diff > OMEGA_THRESH:
                            continue

                        att_errs = [
                            all_candidates[0][i0]['att_err'],
                            all_candidates[1][idx1]['att_err'],
                            all_candidates[2][i2]['att_err'],
                            all_candidates[3][i3]['att_err']
                        ]

                        surviving_chains.append({
                            'indices': [int(i0), int(idx1), int(i2), int(i3)],
                            'att_errs': att_errs,
                            'omega_01': omega_01.tolist(),
                            'omega_12': omega_12.tolist(),
                            'omega_23': omega_23.tolist(),
                            'max_omega_diff': float(max_diff),
                            'max_att_err': float(max(att_errs))
                        })

                        # Cap at 100k chains to prevent OOM
                        if len(surviving_chains) >= 100000:
                            break
                    if len(surviving_chains) >= 100000:
                        break
                if len(surviving_chains) >= 100000:
                    break
            if len(surviving_chains) >= 100000:
                log(f"    HIT 100k chain cap at i1={idx1}")
                break

        dt_chain = time.time() - t_chain
        log(f"  Chain matching: {dt_chain:.0f}s, {len(surviving_chains)} chains found")

    # ─── Results ───
    log(f"\n{'='*60}")
    log("RESULTS")

    log(f"\nCandidates per epoch:")
    for j in range(4):
        n = len(all_candidates[j])
        errs = [c['att_err'] for c in all_candidates[j]]
        log(f"  Epoch {j}: {n}, min_err={min(errs):.2f}°, within_5°={sum(1 for e in errs if e<5)}")

    THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    log(f"\nChains at different ω thresholds:")
    for thresh in THRESHOLDS:
        ct = [c for c in surviving_chains if c['max_omega_diff'] < thresh]
        n_t = len(ct)
        truth = sum(1 for c in ct if c['max_att_err'] < 10)
        close = sum(1 for c in ct if c['max_att_err'] < 5)
        uniq0 = len(set(c['indices'][0] for c in ct))
        log(f"  ω<{thresh:.2f}°/s: {n_t:,} chains, truth(<10°)={truth}, close(<5°)={close}, unique_ep0={uniq0}")

    if surviving_chains:
        surviving_chains.sort(key=lambda c: c['max_omega_diff'])
        log(f"\nTop-10 by ω consistency:")
        for i, c in enumerate(surviving_chains[:10]):
            e = c['att_errs']
            log(f"  {i+1}. att=[{e[0]:.1f}°,{e[1]:.1f}°,{e[2]:.1f}°,{e[3]:.1f}°] max_ω_diff={c['max_omega_diff']:.4f}°/s")

        surviving_chains.sort(key=lambda c: c['max_att_err'])
        log(f"\nTop-10 by attitude error:")
        for i, c in enumerate(surviving_chains[:10]):
            e = c['att_errs']
            log(f"  {i+1}. att=[{e[0]:.1f}°,{e[1]:.1f}°,{e[2]:.1f}°,{e[3]:.1f}°] max_ω_diff={c['max_omega_diff']:.4f}°/s")

    log(f"\nTotal time: {time.time()-t_start:.0f}s")

    # Save (only top 500 chains by omega consistency)
    surviving_chains.sort(key=lambda c: c['max_omega_diff'])
    out = {
        'epoch_indices': epoch_indices,
        'epoch_times': epoch_times_sel,
        'epoch_mags': [float(observed_lc[i]) for i in epoch_indices],
        'n_candidates': [len(all_candidates[j]) for j in range(4)],
        'omega_bound_deg': float(omega_bound_deg),
        'max_rotation_deg': float(max_angle_deg),
        'n_surviving_chains': len(surviving_chains),
        'chains_top500': surviving_chains[:500],
        'total_time': time.time() - t_start
    }
    outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/exp_omega_chain4.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {outpath}")
