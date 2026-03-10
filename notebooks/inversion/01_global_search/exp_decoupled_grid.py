"""
Decoupled Grid Search: Attitude grid + per-candidate omega optimization

Strategy (based on basin data):
  - Attitude-only basin is ~5° (with exact ω)
  - Omega-only basin is ~0.02°/s (with exact attitude)
  - Joint basin is <1° (impossibly narrow for direct search)
  
  Solution: For each attitude candidate on a grid, find the best-fit omega
  via L-BFGS-B (3 params, fast). This decouples the problem.
  
  Grid: 30° spacing in axis-angle space → 1728 candidates
  Per candidate: omega-only L-BFGS-B (maxfun=30, ~1.2s at 0.04s/eval)
  Total Stage 1: ~35 min
  
  Then: top 20 → joint lo-fi refinement → top 5 → hi-fi handoff
  Total: ~45 min
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize, differential_evolution
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
RESULTS_FILE = RESULTS_DIR / "exp_decoupled_grid.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("DECOUPLED GRID SEARCH: ATTITUDE GRID + OMEGA FIT", flush=True)
print("=" * 70, flush=True)

# SETUP
print("[1/6] Setup...", flush=True)
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

# TRUE PARAMS
print("[2/6] Generating observed lightcurve...", flush=True)
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

# Omega bounds from period analysis
dt = np.median(np.diff(observation_times))
f_nyquist = 0.5 / dt
freqs = np.linspace(0.0001, f_nyquist, 10000)
angular_freqs = 2 * np.pi * freqs
lc_centered = observed_lightcurve - np.mean(observed_lightcurve)
power = lombscargle(observation_times, lc_centered, angular_freqs, normalize=True)
f_dominant = freqs[np.argmax(power)]
omega_bound = 5.0 * 2 * np.pi * f_dominant  # generous bound
omega_bound_deg = np.rad2deg(omega_bound)
print(f"  Omega bound: ±{omega_bound_deg:.2f}°/s (±{omega_bound:.6f} rad/s)", flush=True)

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# STAGE 1: ATTITUDE GRID + OMEGA FIT
# ============================================================================
print(f"\n[3/6] Stage 1: Attitude grid with per-candidate omega fit...", flush=True)

# Generate attitude grid: 30° spacing in axis-angle
N_GRID = 12  # points per axis → 12³ = 1728
grid_vals = np.linspace(-np.pi + np.pi/N_GRID, np.pi - np.pi/N_GRID, N_GRID)
total_candidates = N_GRID ** 3
print(f"  Grid: {N_GRID}³ = {total_candidates} attitude candidates", flush=True)
print(f"  Estimated time: {total_candidates * 1.5 / 60:.0f} min", flush=True)

omega_bounds = [(-omega_bound, omega_bound)] * 3
candidates = []
n_done = 0
t_stage1 = time.perf_counter()

for i, aa1 in enumerate(grid_vals):
    for j, aa2 in enumerate(grid_vals):
        for k, aa3 in enumerate(grid_vals):
            aa_fixed = np.array([aa1, aa2, aa3])
            
            # Omega-only L-BFGS-B
            def omega_obj(omega):
                x = np.concatenate([aa_fixed, omega])
                return obj_lofi.evaluate(x)
            
            # Try from omega=0 and one random start
            best_f = float('inf')
            best_omega = np.zeros(3)
            
            for omega_start in [np.zeros(3), np.random.uniform(-omega_bound*0.3, omega_bound*0.3, 3)]:
                try:
                    res = minimize(omega_obj, omega_start, method='L-BFGS-B',
                                 bounds=omega_bounds, options={'maxiter': 10, 'maxfun': 30})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_omega = res.x.copy()
                except:
                    pass
            
            x_best = np.concatenate([aa_fixed, best_omega])
            candidates.append((best_f, x_best))
            n_done += 1
            
            if n_done % 200 == 0:
                # Sort and peek at best
                candidates.sort(key=lambda c: c[0])
                top_f, top_x = candidates[0]
                att_e, om_e = compute_errors(top_x)
                rate = n_done / (time.perf_counter() - t_stage1)
                eta = (total_candidates - n_done) / rate / 60
                print(f"  [{n_done}/{total_candidates}] {elapsed():.0f}s | best f={top_f:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s | ETA {eta:.0f}min", flush=True)

t_stage1 = time.perf_counter() - t_stage1
candidates.sort(key=lambda c: c[0])
print(f"\n  Stage 1 done in {t_stage1:.0f}s ({t_stage1/60:.1f} min)", flush=True)
print(f"  Top 10 candidates:", flush=True)
for i, (f, x) in enumerate(candidates[:10]):
    att_e, om_e = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={att_e:.1f}° ω={om_e:.4f}°/s aa=[{np.rad2deg(x[0]):.0f},{np.rad2deg(x[1]):.0f},{np.rad2deg(x[2]):.0f}]° ω=[{np.rad2deg(x[3]):.3f},{np.rad2deg(x[4]):.3f},{np.rad2deg(x[5]):.3f}]°/s", flush=True)

# Save intermediate
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1), "stage": "1_complete",
    "stage1_time_s": round(t_stage1, 1), "n_candidates": total_candidates,
    "top10": [{"f": round(f,4), "att_err": round(compute_errors(x)[0],2),
               "omega_err": round(compute_errors(x)[1],4), "x": x.tolist()} 
              for f, x in candidates[:10]]
}, indent=2, default=str))

# ============================================================================
# STAGE 2: JOINT LO-FI REFINEMENT OF TOP 20
# ============================================================================
print(f"\n[4/6] Stage 2: Joint lo-fi refinement (top 20)...", flush=True)
t_stage2 = time.perf_counter()

joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3
refined = []

for i, (f_init, x_init) in enumerate(candidates[:20]):
    att0, om0 = compute_errors(x_init)
    
    n_ev = [0]
    def obj_count(x): n_ev[0] += 1; return obj_lofi.evaluate(x)
    
    res = minimize(obj_count, x_init, method='L-BFGS-B', bounds=joint_bounds,
                   options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    
    att_f, om_f = compute_errors(res.x)
    sym = "✓" if att_f < 5 and om_f < 0.1 else "✗"
    print(f"  [{i+1}] {sym} grid:{att0:.1f}°/{om0:.4f} → refined:{att_f:.2f}°/{om_f:.4f}°/s | {n_ev[0]}ev f={res.fun:.4f}", flush=True)
    
    refined.append((res.fun, res.x.copy(), att_f, om_f))

t_stage2 = time.perf_counter() - t_stage2
refined.sort(key=lambda r: r[0])
print(f"  Stage 2 done in {t_stage2:.0f}s", flush=True)

# ============================================================================
# STAGE 3: HI-FI REFINEMENT OF TOP 5
# ============================================================================
print(f"\n[5/6] Stage 3: Hi-fi L-BFGS-B refinement (top 5)...", flush=True)
t_stage3 = time.perf_counter()

hifi_results = []
for i, (f_lofi, x_lofi, att_lofi, om_lofi) in enumerate(refined[:5]):
    n_ev = [0]
    t_h = time.perf_counter()
    def obj_hifi_count(x): n_ev[0] += 1; return obj_hifi.evaluate(x)
    
    res = minimize(obj_hifi_count, x_lofi, method='L-BFGS-B', bounds=joint_bounds,
                   options={'maxiter': 30, 'maxfun': 60, 'ftol': 1e-12})
    t_h = time.perf_counter() - t_h
    
    att_h, om_h = compute_errors(res.x)
    success = att_h < 5 and om_h < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} lofi:{att_lofi:.1f}°/{om_lofi:.4f} → hifi:{att_h:.2f}°/{om_h:.4f}°/s | {n_ev[0]}ev {t_h:.1f}s", flush=True)
    
    hifi_results.append({
        "rank": i+1, "lofi_att": round(att_lofi, 2), "lofi_omega": round(om_lofi, 4),
        "hifi_att": round(att_h, 2), "hifi_omega": round(om_h, 4),
        "success": success, "hifi_evals": n_ev[0], "hifi_time": round(t_h, 1),
        "x_best": res.x.tolist(), "f_hifi": round(float(res.fun), 6),
    })

t_stage3 = time.perf_counter() - t_stage3

# ============================================================================
# STAGE 4: FINER GRID AROUND BEST CANDIDATE
# ============================================================================
print(f"\n[6/6] Stage 4: Fine grid around best candidate...", flush=True)
best_x = refined[0][1]
best_aa = best_x[:3]

# 5° grid around best attitude, ±0.02 rad/s around best omega
fine_spacing = np.deg2rad(5)
fine_points = np.linspace(-2*fine_spacing, 2*fine_spacing, 5)  # ±10° in 5° steps = 5 points each axis
n_fine = 0
fine_candidates = []

for da1 in fine_points:
    for da2 in fine_points:
        for da3 in fine_points:
            aa_fine = best_aa + np.array([da1, da2, da3])
            
            def omega_obj_fine(omega):
                x = np.concatenate([aa_fine, omega])
                return obj_lofi.evaluate(x)
            
            best_f_fine = float('inf')
            best_omega_fine = best_x[3:].copy()
            
            for omega_start in [best_x[3:], np.zeros(3)]:
                try:
                    res = minimize(omega_obj_fine, omega_start, method='L-BFGS-B',
                                 bounds=omega_bounds, options={'maxiter': 15, 'maxfun': 40})
                    if res.fun < best_f_fine:
                        best_f_fine = res.fun
                        best_omega_fine = res.x.copy()
                except:
                    pass
            
            x_fine = np.concatenate([aa_fine, best_omega_fine])
            fine_candidates.append((best_f_fine, x_fine))
            n_fine += 1

fine_candidates.sort(key=lambda c: c[0])
print(f"  {n_fine} fine grid points evaluated", flush=True)

# Joint refinement of top 5 fine candidates
print(f"  Refining top 5 fine candidates (lo-fi → hi-fi)...", flush=True)
for i, (f_fine, x_fine) in enumerate(fine_candidates[:5]):
    att0, om0 = compute_errors(x_fine)
    
    # Lo-fi joint
    res_lofi = minimize(lambda x: obj_lofi.evaluate(x), x_fine, method='L-BFGS-B',
                        bounds=joint_bounds, options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    att_l, om_l = compute_errors(res_lofi.x)
    
    # Hi-fi
    n_hev = [0]
    def hf(x): n_hev[0] += 1; return obj_hifi.evaluate(x)
    res_hifi = minimize(hf, res_lofi.x, method='L-BFGS-B', bounds=joint_bounds,
                        options={'maxiter': 30, 'maxfun': 60, 'ftol': 1e-12})
    att_h, om_h = compute_errors(res_hifi.x)
    
    success = att_h < 5 and om_h < 0.1
    sym = "✓" if success else "✗"
    print(f"    [{i+1}] {sym} fine:{att0:.1f}°/{om0:.4f} → lofi:{att_l:.1f}°/{om_l:.4f} → hifi:{att_h:.2f}°/{om_h:.4f}°/s ({n_hev[0]} hifi ev)", flush=True)
    
    hifi_results.append({
        "rank": len(hifi_results)+1, "source": "fine_grid",
        "lofi_att": round(att_l, 2), "lofi_omega": round(om_l, 4),
        "hifi_att": round(att_h, 2), "hifi_omega": round(om_h, 4),
        "success": success, "hifi_evals": n_hev[0],
        "x_best": res_hifi.x.tolist(),
    })

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"Total time: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)
print(f"Stage 1 (grid + omega fit): {t_stage1:.0f}s", flush=True)
print(f"Stage 2 (joint lo-fi refine): {t_stage2:.0f}s", flush=True)
print(f"Stage 3 (hi-fi handoff): {t_stage3:.0f}s", flush=True)

n_success = sum(1 for r in hifi_results if r["success"])
print(f"\nHi-fi results: {n_success}/{len(hifi_results)} success", flush=True)
for r in hifi_results:
    sym = "✓" if r["success"] else "✗"
    src = r.get("source", "coarse_grid")
    print(f"  {sym} [{src}] att={r['hifi_att']:.2f}° ω={r['hifi_omega']:.4f}°/s", flush=True)

best_result = min(hifi_results, key=lambda r: r['hifi_att'] + r['hifi_omega']*10)
print(f"\nBest: att={best_result['hifi_att']:.2f}° ω={best_result['hifi_omega']:.4f}°/s", flush=True)
print(f"  {'✓ SUCCESS!' if best_result['success'] else '✗ FAIL'}", flush=True)
if best_result['success']:
    print(f"  Recovered params: {[round(np.rad2deg(v),4) for v in best_result['x_best'][:3]]}° attitude", flush=True)
    print(f"  Recovered omega:  {[round(np.rad2deg(v),4) for v in best_result['x_best'][3:]]}°/s", flush=True)
    print(f"  True attitude:    {[round(v,4) for v in np.rad2deg(true_axis_angle).tolist()]}°", flush=True)
    print(f"  True omega:       {[round(np.rad2deg(v),4) for v in true_omega0.tolist()]}°/s", flush=True)

# Save
results = {
    "stage1": {"time_s": round(t_stage1, 1), "n_candidates": total_candidates},
    "stage2": {"time_s": round(t_stage2, 1)},
    "stage3": {"time_s": round(t_stage3, 1)},
    "hifi_results": hifi_results,
    "best": best_result,
    "n_success": n_success,
    "true_params_deg": {
        "attitude": np.rad2deg(true_axis_angle).tolist(),
        "omega": np.rad2deg(true_omega0).tolist(),
    },
}
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1), "results": results}, indent=2, default=str))
print(f"\nResults: {RESULTS_FILE}", flush=True)
print("Done.", flush=True)
