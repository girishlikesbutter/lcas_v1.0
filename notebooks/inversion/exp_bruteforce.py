"""
Brute-force grid evaluation + refinement.

At 0.03s per eval, we can do 20,000 evals in 10 min.
Strategy: 
  - 20³ attitude grid × 5³ omega grid = 1,000,000 points (too many)
  - Instead: 10³ attitude × 5³ omega = 125,000 in ~62 min (too long)
  - Better: 8³ attitude × 3³ omega = 13,824 in ~7 min ← DO THIS
  - Then refine best 20 with L-BFGS-B
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
from src.inversion import ObjectiveFunction, axis_angle_to_quaternion, quaternion_to_axis_angle
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp_bruteforce.json"
N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("BRUTE-FORCE GRID + REFINEMENT", flush=True)
print("=" * 70, flush=True)

# SETUP
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

print("[2/4] Generating lightcurve...", flush=True)
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)
obj_gen = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS), sun_positions_j2000=geometry_data['sun_positions'],
    observer_positions_j2000=geometry_data['obs_positions'],
    satellite_positions_j2000=geometry_data['sat_positions'],
    observer_distances=geometry_data['observer_distances'], compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)
k1_vectors, k2_vectors = obj_gen._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=geometry_data['observer_distances'],
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False)
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)
print(f"  Done in {elapsed():.1f}s", flush=True)

obj_lofi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=geometry_data['sun_positions'],
    observer_positions_j2000=geometry_data['obs_positions'],
    satellite_positions_j2000=geometry_data['sat_positions'],
    observer_distances=geometry_data['observer_distances'], compute_shadows_flag=False,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)
obj_hifi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=geometry_data['sun_positions'],
    observer_positions_j2000=geometry_data['obs_positions'],
    satellite_positions_j2000=geometry_data['sat_positions'],
    observer_distances=geometry_data['observer_distances'], compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# STAGE 1: BRUTE-FORCE 6D GRID
# ============================================================================
print(f"\n[3/4] Stage 1: Brute-force 6D grid...", flush=True)
N_ATT = 10   # 10 per axis → 1000 attitude points
N_OMG = 5    # 5 per axis → 125 omega points  
# Total: 125,000 evals × 0.03s = 62 min. Too slow.
# Compromise: N_ATT=8, N_OMG=3 → 13,824 evals × 0.03s = 7 min
N_ATT = 8
N_OMG = 5   # we can afford 5 since omega space is small

omega_max = np.deg2rad(0.15)  # ±0.15°/s covers true omega with margin

att_vals = np.linspace(-np.pi + np.pi/N_ATT, np.pi - np.pi/N_ATT, N_ATT)
omg_vals = np.linspace(-omega_max, omega_max, N_OMG)

total = N_ATT**3 * N_OMG**3
print(f"  Grid: {N_ATT}³ × {N_OMG}³ = {total} points", flush=True)
print(f"  Estimated: {total * 0.03 / 60:.1f} min", flush=True)

t_s1 = time.perf_counter()
best_f = float('inf')
best_x = None
n = 0
top_k = []  # keep top 50

for a1 in att_vals:
    for a2 in att_vals:
        for a3 in att_vals:
            for w1 in omg_vals:
                for w2 in omg_vals:
                    for w3 in omg_vals:
                        x = np.array([a1, a2, a3, w1, w2, w3])
                        f = obj_lofi.evaluate(x)
                        n += 1
                        
                        if len(top_k) < 50:
                            top_k.append((f, x.copy()))
                            if len(top_k) == 50:
                                top_k.sort(key=lambda c: c[0])
                        elif f < top_k[-1][0]:
                            top_k[-1] = (f, x.copy())
                            top_k.sort(key=lambda c: c[0])
                        
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
            
            # Progress every 125 omega combos (1 attitude point done)
            if n % (N_OMG**3 * 10) == 0:
                ae, oe = compute_errors(best_x)
                rate = n / (time.perf_counter() - t_s1)
                eta = (total - n) / rate
                print(f"  [{n}/{total}] {elapsed():.0f}s | best f={best_f:.4f} att={ae:.1f}° ω={oe:.4f}°/s | {rate:.0f} ev/s ETA {eta:.0f}s", flush=True)

t_s1 = time.perf_counter() - t_s1
ae, oe = compute_errors(best_x)
print(f"\n  Stage 1 done: {t_s1:.0f}s ({t_s1/60:.1f} min), {n} evals", flush=True)
print(f"  Best grid: f={best_f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)
print(f"  Top 5:", flush=True)
for i, (f, x) in enumerate(top_k[:5]):
    ae, oe = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 2: REFINE TOP 20 WITH LO-FI L-BFGS-B → HI-FI HANDOFF
# ============================================================================
print(f"\n[4/4] Stage 2: Refine top 20 (lo-fi → hi-fi)...", flush=True)
t_s2 = time.perf_counter()
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_max*2, omega_max*2)] * 3

hifi_results = []
for i, (f0, x0) in enumerate(top_k[:20]):
    ae0, oe0 = compute_errors(x0)
    
    # Lo-fi
    rl = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    ae_l, oe_l = compute_errors(rl.x)
    
    # Hi-fi
    t_h = time.perf_counter()
    rh = minimize(lambda x: obj_hifi.evaluate(x), rl.x, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 30, 'maxfun': 60, 'ftol': 1e-12})
    t_h = time.perf_counter() - t_h
    ae_h, oe_h = compute_errors(rh.x)
    
    success = ae_h < 5 and oe_h < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} grid:{ae0:.1f}° → lofi:{ae_l:.1f}° → hifi:{ae_h:.2f}°/{oe_h:.4f}°/s ({t_h:.1f}s)", flush=True)
    
    hifi_results.append({
        "rank": i+1, "grid_att": round(ae0, 1), "lofi_att": round(ae_l, 2),
        "hifi_att": round(ae_h, 2), "hifi_omega": round(oe_h, 4),
        "success": success, "x_best": rh.x.tolist(), "f_hifi": round(float(rh.fun), 6),
    })

t_s2 = time.perf_counter() - t_s2

# ============================================================================
# SUMMARY
# ============================================================================
total_time = elapsed()
n_success = sum(1 for r in hifi_results if r["success"])
print(f"\n{'='*70}", flush=True)
print(f"TOTAL: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
print(f"Stage 1 (grid): {t_s1:.0f}s | Stage 2 (refine): {t_s2:.0f}s", flush=True)
print(f"\nSUCCESS: {n_success}/{len(hifi_results)}", flush=True)

best = min(hifi_results, key=lambda r: r['hifi_att'] + r['hifi_omega']*10)
print(f"\nBest: att={best['hifi_att']:.2f}° ω={best['hifi_omega']:.4f}°/s — {'✓ SUCCESS!' if best['success'] else '✗ FAIL'}", flush=True)
if best['success']:
    print(f"  Recovered: att={[round(np.rad2deg(v),3) for v in best['x_best'][:3]]}°", flush=True)
    print(f"  True:      att={[round(v,3) for v in np.rad2deg(true_axis_angle).tolist()]}°", flush=True)
    print(f"  Recovered: ω={[round(np.rad2deg(v),5) for v in best['x_best'][3:]]}°/s", flush=True)
    print(f"  True:      ω={[round(np.rad2deg(v),5) for v in true_omega0.tolist()]}°/s", flush=True)

RESULTS_FILE.write_text(json.dumps({
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(total_time, 1),
    "results": {
        "grid": {"n_att": N_ATT, "n_omg": N_OMG, "total_evals": n, "time_s": round(t_s1,1)},
        "refine_time_s": round(t_s2, 1),
        "hifi_results": hifi_results, "n_success": n_success, "best": best,
        "true_params_deg": {"att": np.rad2deg(true_axis_angle).tolist(), "omega": np.rad2deg(true_omega0).tolist()},
    }
}, indent=2, default=str))
print(f"\nSaved: {RESULTS_FILE}", flush=True)
