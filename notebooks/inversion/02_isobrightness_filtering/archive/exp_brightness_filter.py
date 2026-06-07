"""
Brightness-Constrained Search (Girish's insight)

Instead of blindly searching 6D, use individual brightness measurements
as constraints to filter attitude candidates:

1. Generate 50,000 random attitudes
2. Evaluate brightness at t=0 (single epoch, no propagation needed)
3. Keep only attitudes within ε of first observed brightness → ~few hundred
4. For survivors, try omega candidates, propagate 2-3 steps, check match
5. Best candidates → full lo-fi → hi-fi

This is similar to Burton's "viewing sphere" (2021) but simpler.

Key advantage: evaluating brightness at ONE epoch is much cheaper than
a full lightcurve (~50 epochs). We can screen 50k candidates in seconds.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
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
RESULTS_FILE = RESULTS_DIR / "exp_brightness_filter.json"
N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("BRIGHTNESS-CONSTRAINED SEARCH", flush=True)
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

print("[2/6] Generating observed lightcurve...", flush=True)
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
print(f"  True first brightness: {true_lightcurve[0]:.4f}", flush=True)
print(f"  Observed first brightness: {observed_lightcurve[0]:.4f}", flush=True)
print(f"  Done in {elapsed():.1f}s", flush=True)

# Create objective functions
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

omega_bound = np.deg2rad(0.15)

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot)), np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))

# ============================================================================
# STAGE 1: SINGLE-EPOCH BRIGHTNESS SCREENING
# ============================================================================
# We need a way to evaluate brightness at a single epoch for a given attitude.
# The ObjectiveFunction evaluates the full lightcurve. 
# Instead, we create a 1-epoch objective for screening.
print(f"\n[3/6] Stage 1: Single-epoch brightness screening (50,000 attitudes)...", flush=True)

# Create single-epoch objective for first observation only
N_SCREEN1 = 2  # minimum for propagator
obj_single = ObjectiveFunction(
    satellite=satellite, 
    observation_times=observation_times[:N_SCREEN1],
    observed_lightcurve=observed_lightcurve[:N_SCREEN1],
    sun_positions_j2000=geometry_data['sun_positions'][:N_SCREEN1],
    observer_positions_j2000=geometry_data['obs_positions'][:N_SCREEN1],
    satellite_positions_j2000=geometry_data['sat_positions'][:N_SCREEN1],
    observer_distances=geometry_data['observer_distances'][:N_SCREEN1],
    compute_shadows_flag=False,
    articulation_matrices={k: v[:N_SCREEN1] for k, v in articulation_matrices.items()},
    mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)

# Also create 3-epoch objective for secondary screening  
N_SCREEN = 3
obj_3ep = ObjectiveFunction(
    satellite=satellite,
    observation_times=observation_times[:N_SCREEN],
    observed_lightcurve=observed_lightcurve[:N_SCREEN],
    sun_positions_j2000=geometry_data['sun_positions'][:N_SCREEN],
    observer_positions_j2000=geometry_data['obs_positions'][:N_SCREEN],
    satellite_positions_j2000=geometry_data['sat_positions'][:N_SCREEN],
    observer_distances=geometry_data['observer_distances'][:N_SCREEN],
    compute_shadows_flag=False,
    articulation_matrices={k: v[:N_SCREEN] for k, v in articulation_matrices.items()},
    mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)

N_CANDIDATES = 50000
rng = np.random.default_rng(SEED)

# Generate uniform random quaternions, convert to axis-angle
# Uniform on SO(3) via random quaternions
rand_quats = rng.normal(size=(N_CANDIDATES, 4))
rand_quats /= np.linalg.norm(rand_quats, axis=1, keepdims=True)
# Ensure positive scalar part for consistency
rand_quats[rand_quats[:, 0] < 0] *= -1

t_s1 = time.perf_counter()
single_scores = np.zeros(N_CANDIDATES)

for i in range(N_CANDIDATES):
    aa = quaternion_to_axis_angle(rand_quats[i])
    # For single epoch, omega doesn't matter (no propagation over 0 time)
    x = np.concatenate([aa, np.zeros(3)])
    single_scores[i] = obj_single.evaluate(x)
    
    if (i+1) % 10000 == 0:
        n_good = np.sum(single_scores[:i+1] < np.percentile(single_scores[:i+1][single_scores[:i+1] > 0], 5))
        rate = (i+1) / (time.perf_counter() - t_s1)
        print(f"  [{i+1}/{N_CANDIDATES}] {rate:.0f}/s, min_score={np.min(single_scores[:i+1]):.4f}", flush=True)

t_s1 = time.perf_counter() - t_s1

# Keep top 1% (500 candidates)
threshold_pct = 1.0
threshold = np.percentile(single_scores, threshold_pct)
mask = single_scores <= threshold
n_survivors = np.sum(mask)
survivor_indices = np.where(mask)[0]
survivor_quats = rand_quats[survivor_indices]
survivor_scores = single_scores[survivor_indices]

print(f"  Screened {N_CANDIDATES} in {t_s1:.1f}s ({N_CANDIDATES/t_s1:.0f}/s)", flush=True)
print(f"  Threshold (top {threshold_pct}%): {threshold:.4f}", flush=True)
print(f"  Survivors: {n_survivors}", flush=True)

# Check how many survivors are actually close to truth
n_close = 0
for idx in survivor_indices:
    aa = quaternion_to_axis_angle(rand_quats[idx])
    q_x = axis_angle_to_quaternion(aa)
    dot = min(np.abs(np.dot(axis_angle_to_quaternion(true_axis_angle), q_x)), 1.0)
    err = np.rad2deg(2 * np.arccos(dot))
    if err < 30:
        n_close += 1
print(f"  Survivors within 30° of truth: {n_close}/{n_survivors}", flush=True)

# ============================================================================
# STAGE 2: 3-EPOCH SCREENING WITH OMEGA CANDIDATES
# ============================================================================
print(f"\n[4/6] Stage 2: 3-epoch screening with omega candidates...", flush=True)
t_s2 = time.perf_counter()

# For each survivor attitude, try 27 omega candidates (3³ grid over ±omega_bound)
omg_grid = np.linspace(-omega_bound, omega_bound, 3)
omg_candidates = np.array(np.meshgrid(omg_grid, omg_grid, omg_grid)).T.reshape(-1, 3)
print(f"  {n_survivors} attitudes × {len(omg_candidates)} omegas = {n_survivors * len(omg_candidates)} combos", flush=True)

combo_results = []
for i, idx in enumerate(survivor_indices):
    aa = quaternion_to_axis_angle(rand_quats[idx])
    
    for omg in omg_candidates:
        x = np.concatenate([aa, omg])
        score = obj_3ep.evaluate(x)
        combo_results.append((score, x.copy()))
    
    if (i+1) % 100 == 0:
        combo_results.sort(key=lambda c: c[0])
        bf, bx = combo_results[0]
        ae, oe = compute_errors(bx)
        print(f"  [{i+1}/{n_survivors}] best 3ep={bf:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

t_s2 = time.perf_counter() - t_s2
combo_results.sort(key=lambda c: c[0])
print(f"  Stage 2: {t_s2:.1f}s, {len(combo_results)} combos evaluated", flush=True)
print(f"  Top 10:", flush=True)
for i, (f, x) in enumerate(combo_results[:10]):
    ae, oe = compute_errors(x)
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 3: FULL LO-FI OPTIMIZATION (TOP 50)
# ============================================================================
print(f"\n[5/6] Stage 3: Full lo-fi L-BFGS-B (top 50 combos)...", flush=True)
t_s3 = time.perf_counter()
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3

lofi_results = []
for i, (f0, x0) in enumerate(combo_results[:50]):
    ae0, oe0 = compute_errors(x0)
    
    rl = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    ae_l, oe_l = compute_errors(rl.x)
    lofi_results.append((rl.fun, rl.x.copy(), ae_l, oe_l))
    
    if (i+1) % 10 == 0:
        lofi_results.sort(key=lambda r: r[0])
        print(f"  [{i+1}/50] best lofi: att={lofi_results[0][2]:.1f}° ω={lofi_results[0][3]:.4f}°/s", flush=True)

t_s3 = time.perf_counter() - t_s3
lofi_results.sort(key=lambda r: r[0])
print(f"  Stage 3: {t_s3:.1f}s", flush=True)
print(f"  Top 5 after lo-fi:", flush=True)
for i, (f, x, ae, oe) in enumerate(lofi_results[:5]):
    print(f"    {i+1}. f={f:.4f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# STAGE 4: HI-FI HANDOFF (TOP 10)
# ============================================================================
print(f"\n[6/6] Stage 4: Hi-fi handoff (top 10)...", flush=True)
t_s4 = time.perf_counter()

hifi_results = []
for i, (f_lofi, x_lofi, ae_l, oe_l) in enumerate(lofi_results[:10]):
    t_h = time.perf_counter()
    rh = minimize(lambda x: obj_hifi.evaluate(x), x_lofi, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12})
    t_h = time.perf_counter() - t_h
    ae_h, oe_h = compute_errors(rh.x)
    
    success = ae_h < 5 and oe_h < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} lofi:{ae_l:.1f}°/{oe_l:.4f} → hifi:{ae_h:.2f}°/{oe_h:.4f}°/s ({t_h:.1f}s)", flush=True)
    
    hifi_results.append({
        "rank": i+1, "lofi_att": round(ae_l, 2), "lofi_omega": round(oe_l, 4),
        "hifi_att": round(ae_h, 2), "hifi_omega": round(oe_h, 4),
        "success": success, "x_best": rh.x.tolist(), "f_hifi": round(float(rh.fun), 6),
    })

t_s4 = time.perf_counter() - t_s4

# ============================================================================
# SUMMARY
# ============================================================================
total_time = elapsed()
n_success = sum(1 for r in hifi_results if r["success"])
print(f"\n{'='*70}", flush=True)
print(f"TOTAL: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
print(f"  Stage 1 (single-epoch screen): {t_s1:.0f}s ({N_CANDIDATES} candidates)", flush=True)
print(f"  Stage 2 (3-epoch + omega): {t_s2:.0f}s ({len(combo_results)} combos)", flush=True)
print(f"  Stage 3 (full lo-fi): {t_s3:.0f}s (50 candidates)", flush=True)
print(f"  Stage 4 (hi-fi handoff): {t_s4:.0f}s (10 candidates)", flush=True)
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
        "stage1": {"n_candidates": N_CANDIDATES, "n_survivors": n_survivors, 
                   "n_close_30deg": n_close, "time_s": round(t_s1, 1)},
        "stage2": {"n_combos": len(combo_results), "time_s": round(t_s2, 1)},
        "stage3": {"time_s": round(t_s3, 1)},
        "stage4": {"time_s": round(t_s4, 1)},
        "hifi_results": hifi_results, "n_success": n_success, "best": best,
        "true_params_deg": {"att": np.rad2deg(true_axis_angle).tolist(), 
                           "omega": np.rad2deg(true_omega0).tolist()},
    }
}, indent=2, default=str))
print(f"\nSaved: {RESULTS_FILE}", flush=True)
