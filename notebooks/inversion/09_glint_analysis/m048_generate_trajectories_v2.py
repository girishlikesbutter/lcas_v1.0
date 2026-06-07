#!/usr/bin/env python3
"""Micro-48 -- Generate 100 realistic trajectories with randomised start times.

Same as m046 but:
  - Start time randomised between 08:35 and 15:35 UTC (2020-02-05)
  - 1 hour duration per trajectory
  - Equatorial and 3D phase angles stored
  - Equatorial PAB stored
  - Geometry computed per trajectory (no longer shared)

Omega magnitude: uniform [0.1, 1.5] deg/s.
Random q0 (uniform SO(3)), random omega direction.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, peak_prominences

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR = RESULTS_DIR / "per_trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

# ===== Configuration =====
N_TRAJECTORIES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_OBS = 500
DURATION_S = 3600.0  # 1 hour
OMEGA_MAG_RANGE_DPS = (0.1, 1.5)
START_UTC_EARLIEST = '2020-02-05T08:35:00'
START_UTC_LATEST = '2020-02-05T15:35:00'
GROUP_ORDER = [13, 0, 8, 5, 7, 6, 11, 2, 12, 1]
GROUP_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z', '+WD', '-WD', '+ED', '-ED']
N_WORKERS = int(os.environ.get('MICRO48_N_WORKERS', '4'))

# ===========================================================================
# One-time setup (satellite model + normals — geometry done per trajectory)
# ===========================================================================
print("=" * 70, flush=True)
print(f"m048 -- Generate {N_TRAJECTORIES} trajectories (randomised start times)", flush=True)
print(f"  omega range: {OMEGA_MAG_RANGE_DPS} deg/s", flush=True)
print(f"  start time range: {START_UTC_EARLIEST} to {START_UTC_LATEST}", flush=True)
print("=" * 70, flush=True)
t_global = time.time()

print("Setting up satellite model...", flush=True)
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
satellite = STLLoader.create_satellite_from_stl_config(
    config=config, config_manager=config_manager)
BRDFCalculator().update_satellite_brdf_with_manager(satellite, BRDFManager(config))

masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0,
          'AD_East': 50.0, 'AD_West': 50.0}
inertia_tensor = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses=masses, articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}
).inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(
    str(config_manager.get_metakernel_path(config)))

# ET range for start time randomisation
et_earliest = spice_handler.utc_to_et(START_UTC_EARLIEST)
et_latest = spice_handler.utc_to_et(START_UTC_LATEST)

# Extract normal families
facet_arrays = extract_facet_arrays(satellite)
# Use a dummy art_matrices for normal extraction (just need epoch 0)
dummy_art = compute_rotation_matrices_from_angles(
    {'SP_North': np.full(1, 0.), 'SP_South': np.full(1, 0.),
     'AD_East': np.full(1, 15.), 'AD_West': np.full(1, 15.)}, satellite)
art_normals, _ = apply_articulation_to_arrays(facet_arrays, dummy_art, 0, satellite)
all_unique_normals, inverse_indices = np.unique(
    np.round(art_normals, 4), axis=0, return_inverse=True)
unique_normals = all_unique_normals[GROUP_ORDER]
n_groups = len(GROUP_ORDER)

# Group-to-facet mapping for flux aggregation
group_facet_masks = []
for g_idx, g_orig in enumerate(GROUP_ORDER):
    group_facet_masks.append(inverse_indices == g_orig)

group_areas = np.array([float(facet_arrays.areas[m].sum()) for m in group_facet_masks])

print(f"  {N_OBS} epochs per trajectory, {n_groups} normal groups", flush=True)


# ===========================================================================
# Per-trajectory worker
# ===========================================================================
def process_trajectory(seed):
    """Generate all data for one trajectory. Saves to individual NPZ."""
    t0 = time.time()

    rng = np.random.RandomState(seed)

    # Random q0
    q0_scipy = Rotation.random(random_state=rng)
    q0_xyzw = q0_scipy.as_quat()
    q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

    # Random omega
    omega_dir = rng.randn(3)
    omega_dir /= np.linalg.norm(omega_dir)
    omega_mag_dps = rng.uniform(*OMEGA_MAG_RANGE_DPS)
    omega0_dps = omega_mag_dps * omega_dir
    omega0_rad = np.deg2rad(omega0_dps)
    omega_theta = np.arccos(np.clip(omega_dir[2], -1, 1))
    omega_phi = np.arctan2(omega_dir[1], omega_dir[0])

    # Random start time
    start_et = rng.uniform(et_earliest, et_latest)
    end_et = start_et + DURATION_S
    epochs = np.linspace(start_et, end_et, N_OBS)
    observation_times = epochs - epochs[0]
    dt_sampling = observation_times[-1] / (N_OBS - 1)
    start_utc = spice_handler.et_to_utc(start_et, 'C', 0)

    # SPICE geometry for this time window
    geometry_data = compute_observation_geometry(
        epochs=epochs, satellite_id=config.spice_config.satellite_id,
        observer_id=399999, spice_handler=spice_handler, config=config)
    sun_pos = geometry_data['sun_positions']
    obs_pos = geometry_data['obs_positions']
    sat_pos = geometry_data['sat_positions']
    obs_dist = geometry_data['observer_distances']

    # Articulation matrices
    art_matrices = compute_rotation_matrices_from_angles(
        {'SP_North': np.full(N_OBS, 0.), 'SP_South': np.full(N_OBS, 0.),
         'AD_East': np.full(N_OBS, 15.), 'AD_West': np.full(N_OBS, 15.)}, satellite)

    # Propagate attitude
    quaternions, _ = propagate_attitude(
        q0=q0_wxyz, omega0=omega0_rad,
        times=observation_times, mode="tumbling",
        inertia_tensor=inertia_tensor)

    # Body-frame vectors and R^T
    k1_body = np.zeros((N_OBS, 3))
    k2_body = np.zeros((N_OBS, 3))
    R_body_to_inertial = np.zeros((N_OBS, 3, 3))
    for i in range(N_OBS):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_body_to_inertial[i] = R.T
        sv = sun_pos[i] - sat_pos[i]
        k1_body[i] = R @ sv / np.linalg.norm(sv)
        ov = obs_pos[i] - sat_pos[i]
        k2_body[i] = R @ ov / np.linalg.norm(ov)

    # PAB in body frame
    pab_body_unnorm = k1_body + k2_body
    pab_body = pab_body_unnorm / np.linalg.norm(pab_body_unnorm, axis=1, keepdims=True)

    # PAB in J2000
    k1_j2000 = sun_pos - sat_pos
    k1_j2000 /= np.linalg.norm(k1_j2000, axis=1, keepdims=True)
    k2_j2000 = obs_pos - sat_pos
    k2_j2000 /= np.linalg.norm(k2_j2000, axis=1, keepdims=True)
    pab_j2000_unnorm = k1_j2000 + k2_j2000
    pab_j2000 = pab_j2000_unnorm / np.linalg.norm(pab_j2000_unnorm, axis=1, keepdims=True)

    # 3D phase angle
    phase_angle_3d = np.degrees(np.arccos(np.clip(
        np.sum(k1_j2000 * k2_j2000, axis=1), -1, 1)))

    # Equatorial phase angle (project k1, k2 onto J2000 XY plane)
    k1_eq = k1_j2000[:, :2]
    k2_eq = k2_j2000[:, :2]
    k1_eq_n = k1_eq / np.linalg.norm(k1_eq, axis=1, keepdims=True)
    k2_eq_n = k2_eq / np.linalg.norm(k2_eq, axis=1, keepdims=True)
    phase_angle_equatorial = np.degrees(np.arccos(np.clip(
        np.sum(k1_eq_n * k2_eq_n, axis=1), -1, 1)))

    # Equatorial PAB (PAB projected onto J2000 XY plane, normalised)
    pab_eq_unnorm = k1_eq_n + k2_eq_n
    pab_eq_norms = np.linalg.norm(pab_eq_unnorm, axis=1, keepdims=True)
    # Handle degenerate case (phase_eq ≈ 180°)
    pab_eq_norms = np.where(pab_eq_norms < 1e-10, 1.0, pab_eq_norms)
    pab_equatorial = pab_eq_unnorm / pab_eq_norms

    # Hi-fi lightcurve with per-facet flux
    lit_hifi = compute_shadows(
        satellite=satellite, k1_vectors=k1_body,
        explicit_component_matrices=art_matrices, show_progress=False)
    mag_hifi, flux_hifi, _, _, _, anim_data = generate_lightcurves(
        facet_lit_status_dict=lit_hifi, k1_vectors_array=k1_body,
        k2_vectors_array=k2_body, observer_distances=obs_dist,
        satellite=satellite, epochs=epochs,
        pre_computed_matrices=art_matrices,
        generate_no_shadow=False, animate=True, show_progress=False)

    # Lo-fi lightcurve
    lit_lofi = create_no_shadow_lit_status(satellite, N_OBS)
    mag_lofi, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_lofi, k1_vectors_array=k1_body,
        k2_vectors_array=k2_body, observer_distances=obs_dist,
        satellite=satellite, epochs=epochs,
        pre_computed_matrices=art_matrices,
        generate_no_shadow=False, animate=False, show_progress=False)

    # Aggregate flux by normal group
    group_flux = np.zeros((n_groups, N_OBS))
    for i in range(N_OBS):
        flat_flux = np.zeros(facet_arrays.total_facets)
        flat_idx = 0
        for component in satellite.components:
            for fj in range(len(component.facets)):
                facet_key = f"{component.name}_{fj}"
                flat_flux[flat_idx] = anim_data[i]['facet_flux'][facet_key]
                flat_idx += 1
        for g in range(n_groups):
            group_flux[g, i] = flat_flux[group_facet_masks[g]].sum()

    total_flux = group_flux.sum(axis=0)
    safe_total = np.where(total_flux > 1e-30, total_flux, 1.0)
    group_frac_flux = group_flux / safe_total[np.newaxis, :]

    # Alignment: angular distance of each group to PAB (inertial frame)
    ang_dist = np.zeros((n_groups, N_OBS))
    for g in range(n_groups):
        n_j2000 = R_body_to_inertial @ unique_normals[g]
        ndot = np.sum(n_j2000 * pab_j2000, axis=1)
        ang_dist[g] = np.degrees(np.arccos(np.clip(ndot, -1, 1)))

    min_ang_dist = ang_dist.min(axis=0)
    best_group = ang_dist.argmin(axis=0)

    # Peak detection on hi-fi LC
    peak_idx, _ = find_peaks(-mag_hifi, distance=3)
    if len(peak_idx) > 0:
        peak_proms, _, _ = peak_prominences(-mag_hifi, peak_idx)
    else:
        peak_proms = np.array([])

    elapsed = time.time() - t0

    # Save
    out_path = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    np.savez_compressed(str(out_path),
        seed=seed,
        q0_wxyz=q0_wxyz,
        omega0_rad=omega0_rad,
        omega_mag_dps=omega_mag_dps,
        omega_theta_deg=np.degrees(omega_theta),
        omega_phi_deg=np.degrees(omega_phi),
        start_et=start_et,
        dt_sampling=dt_sampling,
        observation_times=observation_times,
        quaternions=quaternions,
        sun_pos=sun_pos,
        obs_pos=obs_pos,
        sat_pos=sat_pos,
        obs_dist=obs_dist,
        k1_body=k1_body,
        k2_body=k2_body,
        pab_body=pab_body,
        pab_j2000=pab_j2000,
        pab_equatorial=pab_equatorial,
        phase_angle_3d=phase_angle_3d,
        phase_angle_equatorial=phase_angle_equatorial,
        mag_hifi=mag_hifi,
        mag_lofi=mag_lofi,
        group_flux=group_flux,
        group_frac_flux=group_frac_flux,
        ang_dist=ang_dist,
        min_ang_dist=min_ang_dist,
        best_group=best_group,
        hifi_peak_epochs=peak_idx,
        hifi_peak_prominences=peak_proms,
        runtime_s=elapsed,
    )

    return seed, omega_mag_dps, len(peak_idx), elapsed, start_utc, \
           phase_angle_equatorial.min(), phase_angle_equatorial.max()


# ===========================================================================
# Check if this is a worker subprocess
# ===========================================================================
if os.environ.get('MICRO48_WORKER_SEED') is not None:
    worker_seed = int(os.environ['MICRO48_WORKER_SEED'])
    out_path = TRAJ_DIR / f"traj_seed{worker_seed:03d}.npz"
    if out_path.exists():
        print(f"  [skip] seed={worker_seed:3d}", flush=True)
    else:
        s, om, npk, rt, sutc, epa_min, epa_max = process_trajectory(worker_seed)
        print(f"  [done] seed={s:3d}  |w|={om:.2f}  peaks={npk:3d}  "
              f"epa=[{epa_min:.1f},{epa_max:.1f}]°  {sutc}  {rt:.0f}s", flush=True)
    sys.exit(0)


# ===========================================================================
# Main: dispatch workers
# ===========================================================================
import subprocess

seeds_todo = []
for seed in range(N_TRAJECTORIES):
    out_path = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    if out_path.exists():
        print(f"  [skip] seed={seed:3d}", flush=True)
    else:
        seeds_todo.append(seed)

print(f"\n{len(seeds_todo)} to compute, {N_TRAJECTORIES - len(seeds_todo)} already done", flush=True)
print(f"Using {N_WORKERS} parallel workers\n", flush=True)

BATCH_SIZE = N_WORKERS
for batch_start in range(0, len(seeds_todo), BATCH_SIZE):
    batch = seeds_todo[batch_start:batch_start + BATCH_SIZE]
    procs = []
    for seed in batch:
        env = os.environ.copy()
        env['MICRO48_WORKER_SEED'] = str(seed)
        env['PYTHONUNBUFFERED'] = '1'
        p = subprocess.Popen(
            [sys.executable, __file__],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((seed, p))

    for seed, p in procs:
        stdout, _ = p.communicate()
        output = stdout.decode().strip()
        for line in output.split('\n'):
            if '[done]' in line or '[skip]' in line:
                print(line, flush=True)
                break
        else:
            if p.returncode != 0:
                print(f"  [FAIL] seed={seed:3d}  rc={p.returncode}", flush=True)

    done_so_far = min(batch_start + BATCH_SIZE, len(seeds_todo))
    print(f"  --- {done_so_far}/{len(seeds_todo)} complete ---", flush=True)


# ===========================================================================
# Collect into master NPZ
# ===========================================================================
print("\nCollecting per-trajectory files into master NPZ...", flush=True)

# Load all and stack
all_data = []
for seed in range(N_TRAJECTORIES):
    path = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    all_data.append(dict(np.load(path, allow_pickle=True)))

# Stack arrays
def stack(key):
    return np.array([d[key] for d in all_data])

# Scalar/1D per trajectory
seeds = stack('seed')
q0s = stack('q0_wxyz')
omega0s = stack('omega0_rad')
omega_mags = stack('omega_mag_dps')
omega_thetas = stack('omega_theta_deg')
omega_phis = stack('omega_phi_deg')
start_ets = stack('start_et')
runtimes = stack('runtime_s')

# Per-trajectory arrays (T, N, ...) or (T, G, N)
all_obs_times = stack('observation_times')
all_quats = stack('quaternions')
all_sun_pos = stack('sun_pos')
all_obs_pos = stack('obs_pos')
all_sat_pos = stack('sat_pos')
all_obs_dist = stack('obs_dist')
all_k1 = stack('k1_body')
all_k2 = stack('k2_body')
all_pab_body = stack('pab_body')
all_pab_j2000 = stack('pab_j2000')
all_pab_eq = stack('pab_equatorial')
all_phase_3d = stack('phase_angle_3d')
all_phase_eq = stack('phase_angle_equatorial')
all_mag_hifi = stack('mag_hifi')
all_mag_lofi = stack('mag_lofi')
all_group_flux = stack('group_flux')
all_group_frac = stack('group_frac_flux')
all_ang_dist = stack('ang_dist')
all_min_ang = stack('min_ang_dist')
all_best_grp = stack('best_group')

# Peaks (ragged — flatten with seed index)
all_peak_seeds = []
all_peak_epochs = []
all_peak_proms = []
for d in all_data:
    s = int(d['seed'])
    pe = d['hifi_peak_epochs']
    pp = d['hifi_peak_prominences']
    all_peak_seeds.extend([s] * len(pe))
    all_peak_epochs.extend(pe.tolist())
    all_peak_proms.extend(pp.tolist())

npz_path = RESULTS_DIR / "m048_trajectories.npz"
np.savez_compressed(str(npz_path),
    # Metadata
    n_trajectories=N_TRAJECTORIES,
    n_obs=N_OBS,
    duration_s=DURATION_S,
    unique_normals=unique_normals,
    group_names=np.array(GROUP_NAMES),
    group_order=np.array(GROUP_ORDER),
    group_areas=group_areas,
    inertia_tensor=inertia_tensor,
    # Per-trajectory scalars
    seeds=seeds,
    q0s=q0s,
    omega0s=omega0s,
    omega_mags=omega_mags,
    omega_thetas=omega_thetas,
    omega_phis=omega_phis,
    start_ets=start_ets,
    runtimes=runtimes,
    # Per-trajectory arrays
    observation_times=all_obs_times,
    quaternions=all_quats,
    sun_pos=all_sun_pos,
    obs_pos=all_obs_pos,
    sat_pos=all_sat_pos,
    obs_dist=all_obs_dist,
    k1_body=all_k1,
    k2_body=all_k2,
    pab_body=all_pab_body,
    pab_j2000=all_pab_j2000,
    pab_equatorial=all_pab_eq,
    phase_angle_3d=all_phase_3d,
    phase_angle_equatorial=all_phase_eq,
    mag_hifi=all_mag_hifi,
    mag_lofi=all_mag_lofi,
    group_flux=all_group_flux,
    group_frac_flux=all_group_frac,
    ang_dist=all_ang_dist,
    min_ang_dist=all_min_ang,
    best_group=all_best_grp,
    # Peaks
    peak_seeds=np.array(all_peak_seeds, dtype=int),
    peak_epochs=np.array(all_peak_epochs, dtype=int),
    peak_prominences=np.array(all_peak_proms, dtype=float),
)

file_size_mb = npz_path.stat().st_size / 1e6
print(f"Saved: {npz_path} ({file_size_mb:.1f} MB)", flush=True)

# Summary
elapsed = time.time() - t_global
print(f"\n{'=' * 70}", flush=True)
print(f"m048 complete: {N_TRAJECTORIES} trajectories in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
print(f"  omega range: [{omega_mags.min():.2f}, {omega_mags.max():.2f}] deg/s", flush=True)
print(f"  eq phase range: [{all_phase_eq.min():.1f}°, {all_phase_eq.max():.1f}°]", flush=True)
print(f"  total peaks: {len(all_peak_seeds)}", flush=True)
print(f"  mean peaks per trajectory: {len(all_peak_seeds)/N_TRAJECTORIES:.1f}", flush=True)
print(f"{'=' * 70}", flush=True)
