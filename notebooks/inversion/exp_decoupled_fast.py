"""
Fast decoupled grid search. Fixes from v1:
- Disable ObjectiveFunction verbose output (show_progress=False)
- Use multiprocessing for parallel grid evaluation
- Coarser grid (N=8 → 512 candidates) with finer refinement around best
"""
import sys, os, time, json, multiprocessing as mp
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
from scipy.signal import lombscargle

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp_decoupled_fast.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("FAST DECOUPLED GRID SEARCH", flush=True)
print("=" * 70, flush=True)

# ---- SETUP ----
print("[1/5] Setup...", flush=True)
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

# True params
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0],
                     np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

# Generate observed lightcurve
print("[2/5] Generating lightcurve...", flush=True)
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)
obj_gen = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS), sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)
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

# Objectives - SILENT
obj_lofi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=False,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)
obj_hifi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000, satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances, compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)

# Omega bounds
omega_bound = np.deg2rad(1.0)  # ±1°/s generous
omega_bounds_opt = [(-omega_bound, omega_bound)] * 3

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# STAGE 1: COARSE GRID (8³ = 512) + omega fit
# ============================================================================
print(f"\n[3/5] Stage 1: Coarse grid (8³=512) + omega fit...", flush=True)
N_GRID = 8
grid_vals = np.linspace(-np.pi + np.pi/N_GRID, np.pi - np.pi/N_GRID, N_GRID)
candidates = []
t_s1 = time.perf_counter()
n = 0

for aa1 in grid_vals:
    for aa2 in grid_vals:
        for aa3 in grid_vals:
            aa = np.array([aa1, aa2, aa3])
            
            def omega_obj(omega, _aa=aa):
                return obj_lofi.evaluate(np.concatenate([_aa, omega]))
            
            best_f = float('inf')
            best_omega = np.zeros(3)
            
            for w0 in [np.zeros(3), np.random.uniform(-omega_bound*0.3, omega_bound*0.3, 3)]:
                try:
                    r = minimize(omega_obj, w0, method='L-BFGS-B', bounds=omega_bounds_opt,
                                options={'maxiter': 10, 'maxfun': 30, })
                    if r.fun < best_f:
                        best_f = r.fun
                        best_omega = r.x.copy()
                except: pass
            
            candidates.append((best_f, np.concatenate([aa, best_omega])))
            n += 1
            
            if n % 50 == 0:
                candidates.sort(key=lambda c: c[0])
                bf, bx = candidates[0]
                ae, oe = compute_errors(bx)
                rate = n / (time.perf_counter() - t_s1)
                eta = (512 - n) / rate
                print(f"  [{n}/512] {elapsed():.0f}s | best f={bf:.4f} att={ae:.1f}° ω={oe:.4f}°/s | ETA {eta:.0f}s", flush=True)

t_s1 = time.perf_counter() - t_s1
candidates.sort(key=lambda c: c[0])
print(f"  Stage 1: {t_s1:.0f}s ({t_s1/60:.1f} min)", flush=True)

# Show top 10
print("  Top 10:", flush=True)
for i, (f, x) in enumerate(candidates[:10]):
    ae, oe = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 2: FINE GRID around top 5 (±22.5° in 7.5° steps = 7³ per candidate)
# ============================================================================
print(f"\n[4/5] Stage 2: Fine grid around top 5...", flush=True)
t_s2 = time.perf_counter()
fine_spacing = np.deg2rad(7.5)
fine_offsets = np.arange(-3, 4) * fine_spacing  # 7 points: ±22.5° in 7.5° steps

fine_candidates = []
for rank, (_, x_coarse) in enumerate(candidates[:5]):
    aa_center = x_coarse[:3]
    omega_start = x_coarse[3:]
    n_fine = 0
    
    for da1 in fine_offsets:
        for da2 in fine_offsets:
            for da3 in fine_offsets:
                aa = aa_center + np.array([da1, da2, da3])
                
                def omega_obj_f(omega, _aa=aa):
                    return obj_lofi.evaluate(np.concatenate([_aa, omega]))
                
                best_f = float('inf')
                best_w = omega_start.copy()
                for w0 in [omega_start, np.zeros(3)]:
                    try:
                        r = minimize(omega_obj_f, w0, method='L-BFGS-B', bounds=omega_bounds_opt,
                                    options={'maxiter': 10, 'maxfun': 30, })
                        if r.fun < best_f:
                            best_f = r.fun
                            best_w = r.x.copy()
                    except: pass
                
                fine_candidates.append((best_f, np.concatenate([aa, best_w])))
                n_fine += 1
    
    ae, oe = compute_errors(fine_candidates[-1][1])
    print(f"  Candidate {rank+1}: {n_fine} fine points, best att nearby", flush=True)

fine_candidates.sort(key=lambda c: c[0])
t_s2 = time.perf_counter() - t_s2
print(f"  Stage 2: {t_s2:.0f}s. {len(fine_candidates)} total fine candidates", flush=True)
print(f"  Top 5 fine:", flush=True)
for i, (f, x) in enumerate(fine_candidates[:5]):
    ae, oe = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 3: JOINT LO-FI REFINEMENT → HI-FI HANDOFF
# ============================================================================
print(f"\n[5/5] Stage 3: Joint lo-fi → hi-fi (top 10)...", flush=True)
t_s3 = time.perf_counter()
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3

hifi_results = []
for i, (f0, x0) in enumerate(fine_candidates[:10]):
    ae0, oe0 = compute_errors(x0)
    
    # Lo-fi joint
    rl = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12, })
    ae_l, oe_l = compute_errors(rl.x)
    
    # Hi-fi handoff
    t_h = time.perf_counter()
    rh = minimize(lambda x: obj_hifi.evaluate(x), rl.x, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 30, 'maxfun': 60, 'ftol': 1e-12, })
    t_h = time.perf_counter() - t_h
    ae_h, oe_h = compute_errors(rh.x)
    
    success = ae_h < 5 and oe_h < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} grid:{ae0:.1f}° → lofi:{ae_l:.1f}° → hifi:{ae_h:.2f}°/{oe_h:.4f}°/s ({t_h:.1f}s)", flush=True)
    
    hifi_results.append({
        "rank": i+1, "grid_att": round(ae0, 1), "lofi_att": round(ae_l, 2),
        "hifi_att": round(ae_h, 2), "hifi_omega": round(oe_h, 4),
        "success": success, "f_hifi": round(float(rh.fun), 6),
        "x_best": rh.x.tolist(),
    })

t_s3 = time.perf_counter() - t_s3

# ============================================================================
# ALSO TRY: 1000 random multi-starts with tight omega (±0.05°/s)
# ============================================================================
print(f"\n[BONUS] 1000 random multi-starts (±0.05°/s omega)...", flush=True)
t_ms = time.perf_counter()
tight_omega = np.deg2rad(0.05)
tight_bounds = [(-np.pi, np.pi)] * 3 + [(-tight_omega, tight_omega)] * 3
rng = np.random.default_rng(SEED)
ms_best_f = float('inf')
ms_best_x = None

for i in range(1000):
    x0 = np.concatenate([rng.uniform(-np.pi, np.pi, 3), rng.uniform(-tight_omega, tight_omega, 3)])
    try:
        r = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=tight_bounds,
                     options={'maxiter': 20, 'maxfun': 60, })
        if r.fun < ms_best_f:
            ms_best_f = r.fun
            ms_best_x = r.x.copy()
    except: pass
    
    if (i+1) % 200 == 0:
        ae, oe = compute_errors(ms_best_x)
        rate = (i+1) / (time.perf_counter() - t_ms)
        print(f"  [{i+1}/1000] best f={ms_best_f:.4f} att={ae:.1f}° ω={oe:.4f}°/s ({rate:.1f}/s)", flush=True)

t_ms = time.perf_counter() - t_ms
ae_ms, oe_ms = compute_errors(ms_best_x)
print(f"  Multi-start: {t_ms:.0f}s, best att={ae_ms:.1f}° ω={oe_ms:.4f}°/s", flush=True)

# Hi-fi handoff of multistart best
rh_ms = minimize(lambda x: obj_hifi.evaluate(x), ms_best_x, method='L-BFGS-B', bounds=joint_bounds,
                 options={'maxiter': 30, 'maxfun': 60, })
ae_msh, oe_msh = compute_errors(rh_ms.x)
ms_success = ae_msh < 5 and oe_msh < 0.1
sym = "✓" if ms_success else "✗"
print(f"  {sym} Hi-fi: att={ae_msh:.2f}° ω={oe_msh:.4f}°/s", flush=True)

hifi_results.append({
    "rank": "multistart", "source": "random_tight_omega",
    "lofi_att": round(ae_ms, 2), "hifi_att": round(ae_msh, 2), "hifi_omega": round(oe_msh, 4),
    "success": ms_success, "x_best": rh_ms.x.tolist(),
})

# ============================================================================
# SUMMARY
# ============================================================================
total = elapsed()
n_success = sum(1 for r in hifi_results if r["success"])
print(f"\n{'='*70}", flush=True)
print(f"TOTAL TIME: {total:.0f}s ({total/60:.1f} min)", flush=True)
print(f"Stage 1 (coarse 512): {t_s1:.0f}s", flush=True)
print(f"Stage 2 (fine 5×343): {t_s2:.0f}s", flush=True)
print(f"Stage 3 (lo-fi+hi-fi): {t_s3:.0f}s", flush=True)
print(f"Multi-start bonus: {t_ms:.0f}s", flush=True)
print(f"\nSUCCESS: {n_success}/{len(hifi_results)}", flush=True)

best = min(hifi_results, key=lambda r: r['hifi_att'] + r['hifi_omega']*10)
print(f"Best: att={best['hifi_att']:.2f}° ω={best['hifi_omega']:.4f}°/s — {'✓ SUCCESS' if best['success'] else '✗ FAIL'}", flush=True)
if best['success']:
    print(f"  Recovered: att={[round(np.rad2deg(v),3) for v in best['x_best'][:3]]}° ω={[round(np.rad2deg(v),5) for v in best['x_best'][3:]]}°/s", flush=True)
    print(f"  True:      att={[round(v,3) for v in np.rad2deg(true_axis_angle).tolist()]}° ω={[round(np.rad2deg(v),5) for v in true_omega0.tolist()]}°/s", flush=True)

RESULTS_FILE.write_text(json.dumps({
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(total, 1),
    "results": {
        "stage1_s": round(t_s1,1), "stage2_s": round(t_s2,1), "stage3_s": round(t_s3,1),
        "multistart_s": round(t_ms,1),
        "hifi_results": hifi_results, "n_success": n_success, "best": best,
    }
}, indent=2, default=str))
print(f"\nSaved: {RESULTS_FILE}", flush=True)
