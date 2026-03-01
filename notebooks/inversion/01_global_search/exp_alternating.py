"""
Alternating Estimation — exploits basin analysis insights directly.

KEY INSIGHT FROM BASIN DATA:
  - Attitude-only basin: ~5° (90% at 5°, with exact ω)
  - Omega-only basin: ~0.02°/s (with exact attitude)
  - Joint basin: <1° (effectively unsolvable directly)

STRATEGY:
  1. Multi-start attitude-only search (ω=0), keep top candidates
  2. For each candidate: fix attitude, optimize omega (3-param, fast)
  3. Fix omega, re-optimize attitude (3-param, fast)
  4. Iterate 2-3 until convergence
  5. Joint 6-param lo-fi polish
  6. Hi-fi handoff

This directly mirrors the basin finding: each subproblem has a wide basin,
the joint problem doesn't. So solve them alternately.

The attitude-only search at ω=0 won't give us correct attitude (satellite 
rotates ~180° during observation). BUT it will give us SOME candidates that
partially match the lightcurve pattern. Among ~50 random starts, the 
attitude-only with ω=0 search found 9/18 successes up to 10° in Phase 1.

Wait — Phase 1 attitude-only used TRUE omega, not ω=0.
So we need a different approach for the initial attitude screen.

NEW STRATEGY:
  1. Generate 200 random attitude starts (axis-angle)
  2. For each: do alternating att/omega optimization (3 iters)
  3. Each subproblem uses L-BFGS-B (3 params, maxfun=50)
  4. Top 20 → joint lo-fi polish → hi-fi handoff
  
  Time: 200 starts × 3 iters × 2 subproblems × 50 evals × 0.03s = 1800s = 30 min
  But each L-BFGS-B has overhead... let's use maxfun=30 → ~1s each
  200 × 3 × 2 × 1s = 1200s = 20 min. Manageable.
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
RESULTS_FILE = RESULTS_DIR / "exp_alternating.json"
N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("ALTERNATING ESTIMATION (BASIN-INFORMED)", flush=True)
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

omega_bound = np.deg2rad(0.15)  # ±0.15°/s — true omega mag is 0.05°/s

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# STAGE 1: ALTERNATING ESTIMATION FROM 300 RANDOM STARTS
# ============================================================================
print(f"\n[3/4] Alternating estimation from 300 random starts...", flush=True)
N_STARTS = 300
N_ITERS = 5  # alternating iterations
att_bounds = [(-np.pi, np.pi)] * 3
omg_bounds = [(-omega_bound, omega_bound)] * 3

rng = np.random.default_rng(SEED)
results = []
t_s1 = time.perf_counter()

for s in range(N_STARTS):
    # Random initial attitude, omega starts at 0
    att = rng.uniform(-np.pi, np.pi, 3)
    omega = np.zeros(3)
    
    for it in range(N_ITERS):
        # Fix attitude, optimize omega
        def omg_obj(w, _att=att.copy()):
            return obj_lofi.evaluate(np.concatenate([_att, w]))
        try:
            r = minimize(omg_obj, omega, method='L-BFGS-B', bounds=omg_bounds,
                        options={'maxiter': 15, 'maxfun': 40})
            omega = r.x.copy()
        except: pass
        
        # Fix omega, optimize attitude
        def att_obj(a, _omg=omega.copy()):
            return obj_lofi.evaluate(np.concatenate([a, _omg]))
        try:
            r = minimize(att_obj, att, method='L-BFGS-B', bounds=att_bounds,
                        options={'maxiter': 15, 'maxfun': 40})
            att = r.x.copy()
        except: pass
    
    x_final = np.concatenate([att, omega])
    f_final = obj_lofi.evaluate(x_final)
    results.append((f_final, x_final.copy()))
    
    if (s+1) % 50 == 0:
        results.sort(key=lambda c: c[0])
        bf, bx = results[0]
        ae, oe = compute_errors(bx)
        rate = (s+1) / (time.perf_counter() - t_s1)
        eta = (N_STARTS - s - 1) / rate
        print(f"  [{s+1}/{N_STARTS}] best f={bf:.4f} att={ae:.1f}° ω={oe:.4f}°/s | {rate:.1f}/s ETA {eta:.0f}s", flush=True)

t_s1 = time.perf_counter() - t_s1
results.sort(key=lambda c: c[0])
print(f"\n  Stage 1 done: {t_s1:.0f}s", flush=True)
print(f"  Top 10:", flush=True)
for i, (f, x) in enumerate(results[:10]):
    ae, oe = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 2: JOINT LO-FI POLISH → HI-FI HANDOFF (TOP 20)
# ============================================================================
print(f"\n[4/4] Joint lo-fi → hi-fi (top 20)...", flush=True)
t_s2 = time.perf_counter()
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound*2, omega_bound*2)] * 3

hifi_results = []
for i, (f0, x0) in enumerate(results[:20]):
    ae0, oe0 = compute_errors(x0)
    
    # Lo-fi joint polish
    rl = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    ae_l, oe_l = compute_errors(rl.x)
    
    # Hi-fi handoff
    t_h = time.perf_counter()
    rh = minimize(lambda x: obj_hifi.evaluate(x), rl.x, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12})
    t_h = time.perf_counter() - t_h
    ae_h, oe_h = compute_errors(rh.x)
    
    success = ae_h < 5 and oe_h < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} alt:{ae0:.1f}°/{oe0:.4f} → lofi:{ae_l:.1f}°/{oe_l:.4f} → hifi:{ae_h:.2f}°/{oe_h:.4f}°/s ({t_h:.1f}s)", flush=True)
    
    hifi_results.append({
        "rank": i+1, "alt_att": round(ae0, 1), "alt_omega": round(oe0, 4),
        "lofi_att": round(ae_l, 2), "lofi_omega": round(oe_l, 4),
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
print(f"Stage 1 (alternating): {t_s1:.0f}s | Stage 2 (refine): {t_s2:.0f}s", flush=True)
print(f"\n*** SUCCESS: {n_success}/{len(hifi_results)} ***", flush=True)

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
        "n_starts": N_STARTS, "n_iters": N_ITERS,
        "stage1_time_s": round(t_s1, 1), "stage2_time_s": round(t_s2, 1),
        "hifi_results": hifi_results, "n_success": n_success, "best": best,
        "true_params_deg": {"att": np.rad2deg(true_axis_angle).tolist(), "omega": np.rad2deg(true_omega0).tolist()},
    }
}, indent=2, default=str))
print(f"\nSaved: {RESULTS_FILE}", flush=True)
