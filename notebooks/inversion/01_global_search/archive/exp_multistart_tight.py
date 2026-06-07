"""
Multi-start L-BFGS-B with VERY tight omega bounds.

Key insight: true omega magnitude is 0.05°/s. Period analysis gives ~0.2°/s (4x off).
But we can try multiple omega bound levels: ±0.1, ±0.05, ±0.02°/s.
At ±0.02°/s, the omega search space is tiny enough that random starts should hit.

500 random starts per bound level, lo-fi, then hi-fi handoff of best.
Each L-BFGS-B: maxfun=100 → ~4s. 500 starts → ~33 min per level.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
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
RESULTS_FILE = RESULTS_DIR / "exp_multistart_tight.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("MULTI-START WITH TIGHT OMEGA BOUNDS", flush=True)
print("=" * 70, flush=True)

# SETUP (identical to all other scripts)
print("[1/3] Setup...", flush=True)
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

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0],
                     np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

print("[2/3] Generating lightcurve...", flush=True)
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

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# MULTI-START AT DIFFERENT OMEGA BOUND LEVELS
# ============================================================================
print(f"\n[3/3] Multi-start search...", flush=True)

omega_bound_levels = [
    (np.deg2rad(0.1), 500, "±0.1°/s"),
    (np.deg2rad(0.05), 500, "±0.05°/s"),
    (np.deg2rad(0.02), 500, "±0.02°/s"),
]

all_results = []
overall_best = None
overall_best_f = float('inf')

for omega_bnd, n_starts, label in omega_bound_levels:
    print(f"\n--- Omega bound: {label}, {n_starts} starts ---", flush=True)
    bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bnd, omega_bnd)] * 3
    
    rng = np.random.default_rng(SEED)
    best_f = float('inf')
    best_x = None
    n_success_lofi = 0
    t_level = time.perf_counter()
    
    for i in range(n_starts):
        x0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, 3),
            rng.uniform(-omega_bnd, omega_bnd, 3)
        ])
        
        try:
            res = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 30, 'maxfun': 100})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                att_e, om_e = compute_errors(best_x)
                if att_e < 5 and om_e < 0.1:
                    n_success_lofi += 1
        except:
            pass
        
        if (i+1) % 100 == 0:
            att_e, om_e = compute_errors(best_x) if best_x is not None else (999, 999)
            rate = (i+1) / (time.perf_counter() - t_level)
            eta = (n_starts - i - 1) / rate / 60
            print(f"    [{i+1}/{n_starts}] best f={best_f:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s | {rate:.1f} starts/s ETA {eta:.1f}min", flush=True)
    
    t_level = time.perf_counter() - t_level
    att_e, om_e = compute_errors(best_x) if best_x is not None else (999, 999)
    print(f"  Result: f={best_f:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s ({t_level:.0f}s, {n_success_lofi} lo-fi successes)", flush=True)
    
    all_results.append({
        "omega_bound": label, "n_starts": n_starts,
        "best_f": round(best_f, 6), "att_err": round(att_e, 2), "omega_err": round(om_e, 4),
        "time_s": round(t_level, 1), "n_lofi_success": n_success_lofi,
        "x_best": best_x.tolist() if best_x is not None else None,
    })
    
    if best_f < overall_best_f:
        overall_best_f = best_f
        overall_best = best_x.copy()

# HI-FI HANDOFF
print(f"\n--- Hi-fi handoff from overall best ---", flush=True)
att0, om0 = compute_errors(overall_best)
print(f"  Lo-fi best: att={att0:.1f}° ω={om0:.4f}°/s f={overall_best_f:.4f}", flush=True)

n_hev = [0]
def hf(x): n_hev[0] += 1; return obj_hifi.evaluate(x)
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
res_hifi = minimize(hf, overall_best, method='L-BFGS-B', bounds=joint_bounds,
                     options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12})
att_h, om_h = compute_errors(res_hifi.x)
success = att_h < 5 and om_h < 0.1
print(f"  Hi-fi: att={att_h:.2f}° ω={om_h:.4f}°/s ({n_hev[0]} evals)", flush=True)
print(f"  {'✓ SUCCESS!' if success else '✗ FAIL'}", flush=True)

if success:
    print(f"  Recovered: {[round(np.rad2deg(v),4) for v in res_hifi.x[:3]]}° att, {[round(np.rad2deg(v),4) for v in res_hifi.x[3:]]}°/s ω", flush=True)
    print(f"  True:      {[round(v,4) for v in np.rad2deg(true_axis_angle).tolist()]}° att, {[round(np.rad2deg(v),4) for v in true_omega0.tolist()]}°/s ω", flush=True)

# Also try hi-fi handoff from each level's best
print(f"\n  Per-level hi-fi handoffs:", flush=True)
for r in all_results:
    if r['x_best'] is None: continue
    x = np.array(r['x_best'])
    n2 = [0]
    def hf2(x2): n2[0] += 1; return obj_hifi.evaluate(x2)
    res2 = minimize(hf2, x, method='L-BFGS-B', bounds=joint_bounds,
                     options={'maxiter': 30, 'maxfun': 60})
    a, o = compute_errors(res2.x)
    s = a < 5 and o < 0.1
    sym = "✓" if s else "✗"
    print(f"    {sym} {r['omega_bound']}: att={a:.2f}° ω={o:.4f}°/s", flush=True)
    r['hifi_att'] = round(a, 2)
    r['hifi_omega'] = round(o, 4)
    r['hifi_success'] = s

print(f"\nTotal time: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)

RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1),
    "results": {"levels": all_results, "overall_success": success,
                "hifi_att": round(att_h, 2), "hifi_omega": round(om_h, 4)}
}, indent=2, default=str))
print(f"Results: {RESULTS_FILE}", flush=True)
