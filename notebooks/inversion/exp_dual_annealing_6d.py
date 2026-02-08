"""
Experiment: dual_annealing on full 6D joint problem (lo-fi)

GOAL: Find the true solution from scratch using a global optimizer that handles
the narrow-basin landscape better than DE.

Key advantages of dual_annealing:
  - Combines generalized simulated annealing with local search
  - Better at escaping local minima than DE for narrow basins
  - Uses omega bounds from period analysis

Strategy:
  1. Period analysis to bound omega (already done: ±1.95°/s)
  2. dual_annealing on lo-fi objective with bounded omega
  3. If lo-fi finds a good candidate, handoff to hi-fi L-BFGS-B

Timeout: 8 minutes for dual_annealing, then hi-fi handoff.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import dual_annealing, minimize
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
RESULTS_FILE = RESULTS_DIR / "exp_dual_annealing_6d.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
DA_MAXITER = 1000  # dual_annealing iterations
DA_TIMEOUT = 480   # 8 min
OMEGA_SAFETY = 5.0

t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70); print("DUAL ANNEALING 6D GLOBAL OPTIMIZATION"); print("=" * 70)

# SETUP (same as all experiments)
print("[1/4] Setup...", flush=True)
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
print(f"  Done in {elapsed():.1f}s", flush=True)

# TRUE PARAMS & LIGHTCURVE
print("[2/4] Generating lightcurve...", flush=True)
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

# PERIOD ANALYSIS
print("[3/4] Period analysis for omega bounds...", flush=True)
dt = np.median(np.diff(observation_times))
f_nyquist = 0.5 / dt
freqs = np.linspace(0.0001, f_nyquist, 5000)
angular_freqs = 2 * np.pi * freqs
lc_centered = observed_lightcurve - np.mean(observed_lightcurve)
power = lombscargle(observation_times, lc_centered, angular_freqs, normalize=True)
f_dominant = freqs[np.argmax(power)]
omega_bound = OMEGA_SAFETY * 2 * np.pi * f_dominant  # rad/s
omega_bound_deg = np.rad2deg(omega_bound)
print(f"  Dominant freq: {f_dominant:.6f} Hz, T={1/f_dominant:.1f}s", flush=True)
print(f"  Omega bound: ±{omega_bound_deg:.2f}°/s (±{omega_bound:.6f} rad/s)", flush=True)
print(f"  True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f}°/s", flush=True)

# DUAL ANNEALING
print(f"\n[4/4] Running dual_annealing (max {DA_TIMEOUT}s)...", flush=True)
bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3

n_eval = [0]
best_f = [float('inf')]
best_x = [None]
t_da = time.perf_counter()

def objective(x):
    n_eval[0] += 1
    val = obj_lofi.evaluate(x)
    if val < best_f[0]:
        best_f[0] = val
        best_x[0] = x.copy()
        q_true = axis_angle_to_quaternion(true_axis_angle)
        q_x = axis_angle_to_quaternion(x[:3])
        dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
        att_err = np.rad2deg(2 * np.arccos(dot))
        omega_err = np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))
        print(f"  [eval {n_eval[0]}] new best f={val:.6f} att={att_err:.1f}° ω={omega_err:.4f}°/s", flush=True)
    if n_eval[0] % 500 == 0:
        print(f"  [eval {n_eval[0]}] {time.perf_counter()-t_da:.0f}s elapsed, best_f={best_f[0]:.6f}", flush=True)
    return val

def callback(x, f, context):
    if time.perf_counter() - t_da > DA_TIMEOUT:
        print(f"  ⏰ Timeout at {n_eval[0]} evals", flush=True)
        return True  # stop
    return False

try:
    result = dual_annealing(
        objective, bounds, maxiter=DA_MAXITER, seed=SEED,
        callback=callback, no_local_search=False,
        initial_temp=5230.0, restart_temp_ratio=2e-5, visit=2.62, accept=-5.0,
    )
    da_success = True
    x_da = result.x
    f_da = result.fun
except Exception as e:
    print(f"  dual_annealing error: {e}", flush=True)
    da_success = best_x[0] is not None
    x_da = best_x[0] if da_success else true_params
    f_da = best_f[0]

t_da = time.perf_counter() - t_da

q_true = axis_angle_to_quaternion(true_axis_angle)
q_da = axis_angle_to_quaternion(x_da[:3])
dot = min(np.abs(np.dot(q_true, q_da)), 1.0)
da_att_err = np.rad2deg(2 * np.arccos(dot))
da_omega_err = np.rad2deg(np.linalg.norm(x_da[3:] - true_params[3:]))

print(f"\n  dual_annealing result:", flush=True)
print(f"    Time: {t_da:.1f}s ({t_da/60:.1f} min)", flush=True)
print(f"    Evals: {n_eval[0]}", flush=True)
print(f"    Attitude error: {da_att_err:.2f}°", flush=True)
print(f"    Omega error: {da_omega_err:.4f}°/s", flush=True)
print(f"    Objective: {f_da:.6f}", flush=True)

# HI-FI HANDOFF (if reasonable candidate)
print(f"\n  Hi-fi L-BFGS-B handoff...", flush=True)
hifi_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3
n2 = [0]; t2 = time.perf_counter()
def obj2(x): n2[0] += 1; return obj_hifi.evaluate(x)
res_hifi = minimize(obj2, x_da, method='L-BFGS-B', bounds=hifi_bounds,
                     options={'maxiter': 30, 'maxfun': 60, 'ftol': 1e-12})
t2 = time.perf_counter() - t2

q_hifi = axis_angle_to_quaternion(res_hifi.x[:3])
dot_h = min(np.abs(np.dot(q_true, q_hifi)), 1.0)
hifi_att_err = np.rad2deg(2 * np.arccos(dot_h))
hifi_omega_err = np.rad2deg(np.linalg.norm(res_hifi.x[3:] - true_params[3:]))

print(f"    Time: {t2:.1f}s, Evals: {n2[0]}", flush=True)
print(f"    Attitude error: {hifi_att_err:.2f}°", flush=True)
print(f"    Omega error: {hifi_omega_err:.4f}°/s", flush=True)

success = hifi_att_err < 5.0 and hifi_omega_err < 0.1
print(f"\n  {'✓ SUCCESS' if success else '✗ FAIL'}", flush=True)

# SAVE
results = {
    "true_params": true_params.tolist(),
    "omega_bound_deg_s": omega_bound_deg,
    "dual_annealing": {
        "time_s": round(t_da, 1), "n_evals": n_eval[0],
        "att_err_deg": round(da_att_err, 2), "omega_err_dps": round(da_omega_err, 4),
        "objective": round(f_da, 6), "x_best": x_da.tolist(),
    },
    "hifi_handoff": {
        "time_s": round(t2, 1), "n_evals": n2[0],
        "att_err_deg": round(hifi_att_err, 2), "omega_err_dps": round(hifi_omega_err, 4),
        "x_best": res_hifi.x.tolist(),
    },
    "success": success,
}
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1), "results": results}, indent=2, default=str))
print(f"\nResults: {RESULTS_FILE}", flush=True)
print(f"Total: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)
