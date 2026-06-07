"""
Experiment: Omega-first estimation → attitude search → joint refinement

KEY INSIGHT: The true omega has a dominant z-component (0.05°/s) that causes
~180° rotation over 3600s. The lightcurve's periodicity directly encodes omega.

Strategy:
  1. Estimate omega MAGNITUDE from lightcurve periodicity (Lomb-Scargle)
  2. Grid search over omega DIRECTION (unit sphere) at fixed magnitude
  3. For each omega candidate, run attitude-only L-BFGS-B (3 params, fast)
  4. Best combined → joint hi-fi L-BFGS-B refinement

The omega magnitude is well-constrained by the dominant frequency.
The omega direction has only 2 DOF (unit sphere), so a coarse grid is feasible.
"""
import sys, os, time, json, itertools
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
from scipy.signal import lombscargle
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.dynamics import propagate_attitude
from src.inversion import (
    ObjectiveFunction, axis_angle_to_quaternion, quaternion_to_axis_angle, normalize_quaternion,
)
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp_omega_first.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70); print("OMEGA-FIRST INVERSION"); print("=" * 70)

# SETUP
print("[1/5] Setup...", flush=True)
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
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
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]
geometry_data = compute_observation_geometry(epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice_handler, config=config)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']
fixed_articulation_angles = {
    'SP_North': np.full(N_OBS, 0.0), 'SP_South': np.full(N_OBS, 0.0),
    'AD_East': np.full(N_OBS, 15.0), 'AD_West': np.full(N_OBS, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

# TRUE PARAMS & LIGHTCURVE
print("[2/5] Generating lightcurve...", flush=True)
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0],
                     np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)
obj_gen = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS), sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor)
k1_vectors, k2_vectors = obj_gen._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False)
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)
print(f"  Done in {elapsed():.1f}s", flush=True)

# OBJECTIVES
obj_lofi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=False,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor)
obj_hifi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor)

# STEP 1: OMEGA MAGNITUDE FROM PERIOD
print("[3/5] Estimating omega magnitude from periodicity...", flush=True)
dt = np.median(np.diff(observation_times))
f_nyquist = 0.5 / dt
freqs = np.linspace(0.0001, f_nyquist, 10000)
angular_freqs = 2 * np.pi * freqs
lc_centered = observed_lightcurve - np.mean(observed_lightcurve)
power = lombscargle(observation_times, lc_centered, angular_freqs, normalize=True)

# Get top 5 peaks
peak_indices = []
for i in range(1, len(power)-1):
    if power[i] > power[i-1] and power[i] > power[i+1] and power[i] > 0.1 * max(power):
        peak_indices.append(i)
peak_indices.sort(key=lambda i: -power[i])
peak_freqs = [freqs[i] for i in peak_indices[:5]]

f_dominant = peak_freqs[0] if peak_freqs else freqs[np.argmax(power)]
# For tumbling satellite, lightcurve period ≈ half the rotation period
# (because the satellite looks the same after 180° rotation, roughly)
# So omega ≈ 4*pi*f or omega ≈ 2*pi*f depending on symmetry
omega_est_1 = 2 * np.pi * f_dominant  # one-to-one
omega_est_2 = 4 * np.pi * f_dominant  # factor-of-2 harmonic

true_omega_mag = np.linalg.norm(true_omega0)
print(f"  Dominant frequency: {f_dominant:.6f} Hz (T={1/f_dominant:.1f}s)", flush=True)
print(f"  Peak freqs: {[f'{f:.6f}' for f in peak_freqs[:5]]}", flush=True)
print(f"  Omega estimate 1 (2πf): {np.rad2deg(omega_est_1):.4f}°/s ({omega_est_1:.6f} rad/s)", flush=True)
print(f"  Omega estimate 2 (4πf): {np.rad2deg(omega_est_2):.4f}°/s ({omega_est_2:.6f} rad/s)", flush=True)
print(f"  True omega magnitude:   {np.rad2deg(true_omega_mag):.4f}°/s ({true_omega_mag:.6f} rad/s)", flush=True)

# Use multiple magnitude candidates
omega_mags = []
for f in peak_freqs[:3]:
    omega_mags.append(2 * np.pi * f)
    omega_mags.append(4 * np.pi * f)
# Add a range around the dominant
for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    omega_mags.append(factor * omega_est_1)
omega_mags = sorted(set([round(om, 8) for om in omega_mags if om > 0]))
print(f"  Testing {len(omega_mags)} omega magnitudes", flush=True)

# STEP 2: GRID OVER OMEGA DIRECTION + ATTITUDE
print(f"\n[4/5] Grid search: omega direction × attitude...", flush=True)

# Omega direction grid: icosahedron-like sampling
# Use spherical coordinates with coarse grid
n_theta = 8  # polar
n_phi = 12   # azimuthal
directions = []
for i in range(n_theta + 1):
    theta = np.pi * i / n_theta
    if i == 0 or i == n_theta:
        directions.append(np.array([0, 0, 1 if i == 0 else -1]))
    else:
        n_phi_i = max(4, int(n_phi * np.sin(theta)))
        for j in range(n_phi_i):
            phi = 2 * np.pi * j / n_phi_i
            directions.append(np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))

print(f"  {len(directions)} directions × {len(omega_mags)} magnitudes = {len(directions)*len(omega_mags)} omega candidates", flush=True)

# For each omega candidate, do a QUICK attitude-only L-BFGS-B from a few starts
best_candidates = []
total_candidates = len(directions) * len(omega_mags)
n_tested = 0

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

for omega_mag in omega_mags:
    for d in directions:
        omega_candidate = omega_mag * d
        n_tested += 1
        
        # Quick attitude-only optimization with this omega fixed
        def att_only_obj(aa):
            x = np.concatenate([aa, omega_candidate])
            return obj_lofi.evaluate(x)
        
        # Try from 3 random starts
        rng = np.random.default_rng(SEED + n_tested)
        best_f_this = float('inf')
        best_x_this = None
        
        for start_i in range(3):
            aa0 = rng.uniform(-np.pi, np.pi, 3)
            try:
                res = minimize(att_only_obj, aa0, method='L-BFGS-B',
                             bounds=[(-np.pi, np.pi)]*3,
                             options={'maxiter': 20, 'maxfun': 50})
                if res.fun < best_f_this:
                    best_f_this = res.fun
                    best_x_this = np.concatenate([res.x, omega_candidate])
            except:
                pass
        
        if best_x_this is not None:
            best_candidates.append((best_f_this, best_x_this.copy()))
        
        if n_tested % 100 == 0:
            best_candidates.sort(key=lambda x: x[0])
            top = best_candidates[0] if best_candidates else (float('inf'), None)
            att_e, om_e = compute_errors(top[1]) if top[1] is not None else (999, 999)
            print(f"  [{n_tested}/{total_candidates}] {elapsed():.0f}s | best f={top[0]:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s", flush=True)

# Sort and take top 20
best_candidates.sort(key=lambda x: x[0])
top_candidates = best_candidates[:20]

print(f"\n  Grid search done in {elapsed():.0f}s", flush=True)
print(f"  Top 5 candidates:", flush=True)
for i, (f, x) in enumerate(top_candidates[:5]):
    att_e, om_e = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s omega_mag={np.rad2deg(np.linalg.norm(x[3:])):.4f}°/s", flush=True)

# STEP 3: REFINE TOP CANDIDATES (lo-fi joint L-BFGS-B)
print(f"\n[5/5] Refining top candidates (lo-fi → hi-fi)...", flush=True)

results_list = []
for i, (f_init, x_init) in enumerate(top_candidates[:10]):
    att0, om0 = compute_errors(x_init)
    
    # Lo-fi joint refinement
    bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
    n_ev = [0]
    def obj_count(x): n_ev[0] += 1; return obj_lofi.evaluate(x)
    res_lofi = minimize(obj_count, x_init, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 50, 'maxfun': 200})
    att_lofi, om_lofi = compute_errors(res_lofi.x)
    
    # Hi-fi refinement (small budget)
    n_hifi = [0]
    def obj_hifi_count(x): n_hifi[0] += 1; return obj_hifi.evaluate(x)
    t_hifi_start = time.perf_counter()
    res_hifi = minimize(obj_hifi_count, res_lofi.x, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 10, 'maxfun': 30})
    t_hifi_elapsed = time.perf_counter() - t_hifi_start
    att_hifi, om_hifi = compute_errors(res_hifi.x)
    
    success = att_hifi < 5.0 and om_hifi < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} grid:{att0:.1f}°/{om0:.4f} → lofi:{att_lofi:.1f}°/{om_lofi:.4f} → hifi:{att_hifi:.1f}°/{om_hifi:.4f} ({n_ev[0]}+{n_hifi[0]}ev, {t_hifi_elapsed:.1f}s hifi)", flush=True)
    
    results_list.append({
        "rank": i+1, "grid_f": round(f_init, 4),
        "grid_att": round(att0, 2), "grid_omega": round(om0, 4),
        "lofi_att": round(att_lofi, 2), "lofi_omega": round(om_lofi, 4),
        "hifi_att": round(att_hifi, 2), "hifi_omega": round(om_hifi, 4),
        "success": success, "lofi_evals": n_ev[0], "hifi_evals": n_hifi[0],
        "x_best": res_hifi.x.tolist(),
    })

# SUMMARY
print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
print(f"Total time: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)
print(f"Omega estimates tested: {len(omega_mags)} magnitudes × {len(directions)} directions", flush=True)

s = sum(1 for r in results_list if r["success"])
print(f"Successes: {s}/{len(results_list)}", flush=True)

best = min(results_list, key=lambda r: r["hifi_att"] + r["hifi_omega"] * 10)
print(f"Best: att={best['hifi_att']:.2f}° ω={best['hifi_omega']:.4f}°/s", flush=True)
print(f"  Grid start: att={best['grid_att']:.1f}° ω={best['grid_omega']:.4f}°/s", flush=True)

results = {
    "omega_estimates": {
        "dominant_freq_hz": f_dominant,
        "omega_est_1_dps": round(np.rad2deg(omega_est_1), 4),
        "omega_est_2_dps": round(np.rad2deg(omega_est_2), 4),
        "true_omega_mag_dps": round(np.rad2deg(true_omega_mag), 4),
        "n_magnitudes": len(omega_mags),
        "n_directions": len(directions),
    },
    "candidates": results_list,
    "best": best,
    "n_successes": s,
}
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1), "results": results}, indent=2, default=str))
print(f"\nResults: {RESULTS_FILE}", flush=True)
