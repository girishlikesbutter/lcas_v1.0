"""
Exp 3 - Convergence Basin Mapping (RESUME)

Picks up where the killed run left off. Runs remaining tests, then does
hi-fi verification on key results.

Plan:
  Phase 1: Complete lo-fi basin mapping (remaining attitude, omega, combined)
  Phase 2: Finer grid around 5-10° boundary
  Phase 3: Multiple random directions per perturbation level (statistical)
  Phase 4: Hi-fi verification on successful lo-fi convergence cases
"""
import sys
import os
from pathlib import Path
import time
import json
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
from src.inversion import (
    ObjectiveFunction,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    normalize_quaternion,
)
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp3_convergence_basin_full.json"

N_OBS = 50
NOISE_SIGMA = 0.05
SEED = 42
DEADLINE_HOURS = 12  # run until 8am

t0_script = time.perf_counter()
def elapsed():
    return time.perf_counter() - t0_script

def hours_left():
    return DEADLINE_HOURS - elapsed() / 3600

def check_deadline():
    if hours_left() < 0.1:
        print(f"\n⏰ DEADLINE approaching. Saving.", flush=True)
        return True
    return False

# ============================================================================
# SETUP
# ============================================================================
print("=" * 70, flush=True)
print("EXP 3 - CONVERGENCE BASIN (FULL RUN)", flush=True)
print(f"Deadline: {DEADLINE_HOURS}h from now", flush=True)
print("=" * 70, flush=True)

print("[1/3] Loading config...", flush=True)
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

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice_handler, config=config
)
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

# ============================================================================
# TRUE PARAMS & LIGHTCURVE
# ============================================================================
print("[2/3] Generating observed lightcurve...", flush=True)

true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([
    np.cos(true_angle_rad / 2),
    np.sin(true_angle_rad / 2) * true_axis[0],
    np.sin(true_angle_rad / 2) * true_axis[1],
    np.sin(true_angle_rad / 2) * true_axis[2],
])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

true_quaternions, _ = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor
)

# Hi-fi observed lightcurve (generated once)
obj_hifi_gen = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS),
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

k1_vectors, k2_vectors = obj_hifi_gen._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(
    satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False
)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False,
)
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)

print(f"  Done in {elapsed():.1f}s", flush=True)

# ============================================================================
# OBJECTIVES
# ============================================================================
obj_lofi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

obj_hifi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

# Time both
t0 = time.perf_counter()
_ = obj_lofi.evaluate(true_params)
t_lofi = time.perf_counter() - t0

t0 = time.perf_counter()
_ = obj_hifi.evaluate(true_params)
t_hifi = time.perf_counter() - t0

print(f"  Lo-fi: {t_lofi:.4f}s | Hi-fi: {t_hifi:.2f}s | Speedup: {t_hifi/t_lofi:.0f}x", flush=True)

# ============================================================================
# HELPERS
# ============================================================================
results = {
    "true_params": true_params.tolist(),
    "true_axis_angle_deg": np.rad2deg(true_axis_angle).tolist(),
    "true_omega_deg_s": np.rad2deg(true_omega0).tolist(),
    "lofi_eval_s": t_lofi,
    "hifi_eval_s": t_hifi,
    "speedup": t_hifi / t_lofi,
    "phase1_attitude_only": [],
    "phase1_omega_only": [],
    "phase1_combined": [],
    "phase2_fine_grid": [],
    "phase3_statistical": [],
    "phase4_hifi_verification": [],
}

def perturb_attitude(q0, pert_deg, rng):
    pert_axis = rng.standard_normal(3)
    pert_axis /= np.linalg.norm(pert_axis)
    pert_rad = np.deg2rad(pert_deg)
    q_pert = np.array([np.cos(pert_rad/2), np.sin(pert_rad/2)*pert_axis[0],
                        np.sin(pert_rad/2)*pert_axis[1], np.sin(pert_rad/2)*pert_axis[2]])
    w = q_pert[0]*q0[0] - np.dot(q_pert[1:], q0[1:])
    v = q_pert[0]*q0[1:] + q0[0]*q_pert[1:] + np.cross(q_pert[1:], q0[1:])
    q_new = normalize_quaternion(np.array([w, v[0], v[1], v[2]]))
    return quaternion_to_axis_angle(q_new)

def run_opt(x0, label, obj, maxfun=500):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x0 = axis_angle_to_quaternion(x0[:3])
    dot = min(np.abs(np.dot(q_true, q_x0)), 1.0)
    att_dist = np.rad2deg(2 * np.arccos(dot))
    omega_dist = np.rad2deg(np.linalg.norm(x0[3:] - true_params[3:]))

    bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
    n_eval = [0]
    t0 = time.perf_counter()

    def objective(x):
        n_eval[0] += 1
        return obj.evaluate(x)

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 100, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-8})

    dt = time.perf_counter() - t0
    q_best = axis_angle_to_quaternion(res.x[:3])
    dot_f = min(np.abs(np.dot(q_true, q_best)), 1.0)
    att_err = np.rad2deg(2 * np.arccos(dot_f))
    omega_err = np.rad2deg(np.linalg.norm(res.x[3:] - true_params[3:]))
    rms = np.sqrt(res.fun) if res.fun >= 0 else float('inf')
    success = att_err < 5.0 and omega_err < 0.1

    sym = "✓" if success else "✗"
    print(f"  [{label}] {sym} {att_dist:.1f}°→{att_err:.2f}° ω:{omega_dist:.4f}→{omega_err:.4f}°/s | {n_eval[0]}ev {dt:.1f}s", flush=True)

    return {
        "label": label, "initial_att_deg": round(att_dist, 2), "initial_omega_dps": round(omega_dist, 4),
        "final_att_deg": round(att_err, 2), "final_omega_dps": round(omega_err, 4),
        "rms": round(rms, 6), "n_evals": n_eval[0], "time_s": round(dt, 1),
        "success": success, "x_best": res.x.tolist(),
    }

def save():
    RESULTS_FILE.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed(), 1),
        "hours_remaining": round(hours_left(), 2),
        "results": results,
    }, indent=2, default=str))

# ============================================================================
# PHASE 1: COMPLETE BASIN MAP (LO-FI)
# ============================================================================
print(f"\n[3/3] PHASE 1: Lo-fi basin mapping", flush=True)
print("=" * 70, flush=True)

# 1a: Attitude only
print("\n--- Attitude only (omega exact) ---", flush=True)
att_perts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 45, 60, 90, 120, 180]
rng = np.random.default_rng(SEED)
for deg in att_perts:
    if check_deadline(): break
    aa = perturb_attitude(true_q0, deg, rng)
    x0 = np.concatenate([aa, true_omega0])
    r = run_opt(x0, f"att_{deg}", obj_lofi)
    results["phase1_attitude_only"].append(r)
    save()

# 1b: Omega only
print("\n--- Omega only (attitude exact) ---", flush=True)
omega_perts = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
rng2 = np.random.default_rng(SEED + 1)
for dps in omega_perts:
    if check_deadline(): break
    d = rng2.standard_normal(3); d /= np.linalg.norm(d)
    omega_new = true_omega0 + np.deg2rad(dps) * d
    x0 = np.concatenate([true_axis_angle, omega_new])
    r = run_opt(x0, f"omega_{dps}", obj_lofi)
    results["phase1_omega_only"].append(r)
    save()

# 1c: Combined
print("\n--- Combined perturbations ---", flush=True)
combined = [
    (1, 0.001), (2, 0.005), (3, 0.01), (5, 0.01), (5, 0.02),
    (5, 0.05), (7, 0.02), (7, 0.05), (10, 0.05), (10, 0.1),
    (15, 0.05), (15, 0.1), (20, 0.1), (30, 0.2), (45, 0.5),
]
rng3 = np.random.default_rng(SEED + 2)
for att_d, om_d in combined:
    if check_deadline(): break
    aa = perturb_attitude(true_q0, att_d, rng3)
    d = rng3.standard_normal(3); d /= np.linalg.norm(d)
    omega_new = true_omega0 + np.deg2rad(om_d) * d
    x0 = np.concatenate([aa, omega_new])
    r = run_opt(x0, f"comb_{att_d}d_{om_d}dps", obj_lofi)
    results["phase1_combined"].append(r)
    save()

# ============================================================================
# PHASE 2: FINE GRID AROUND BOUNDARY (5-10°)
# ============================================================================
if not check_deadline():
    print(f"\nPHASE 2: Fine grid around boundary", flush=True)
    print("=" * 70, flush=True)
    fine_degs = [5, 6, 7, 8, 9, 10]
    N_TRIALS = 10  # multiple random directions
    rng4 = np.random.default_rng(SEED + 100)
    for deg in fine_degs:
        if check_deadline(): break
        for trial in range(N_TRIALS):
            if check_deadline(): break
            aa = perturb_attitude(true_q0, deg, rng4)
            x0 = np.concatenate([aa, true_omega0])
            r = run_opt(x0, f"fine_{deg}d_t{trial}", obj_lofi)
            results["phase2_fine_grid"].append(r)
        save()

# ============================================================================
# PHASE 3: STATISTICAL TESTS (multiple directions at key distances)
# ============================================================================
if not check_deadline():
    print(f"\nPHASE 3: Statistical convergence tests", flush=True)
    print("=" * 70, flush=True)
    stat_degs = [3, 5, 7, 10]
    stat_omegas = [0.01, 0.05, 0.1]
    N_STAT = 20
    rng5 = np.random.default_rng(SEED + 200)
    for deg in stat_degs:
        for om in stat_omegas:
            if check_deadline(): break
            successes = 0
            for trial in range(N_STAT):
                if check_deadline(): break
                aa = perturb_attitude(true_q0, deg, rng5)
                d = rng5.standard_normal(3); d /= np.linalg.norm(d)
                omega_new = true_omega0 + np.deg2rad(om) * d
                x0 = np.concatenate([aa, omega_new])
                r = run_opt(x0, f"stat_{deg}d_{om}dps_t{trial}", obj_lofi)
                results["phase3_statistical"].append(r)
                if r["success"]: successes += 1
            rate = successes / N_STAT * 100 if N_STAT > 0 else 0
            print(f"  >> {deg}° + {om}°/s: {successes}/{N_STAT} = {rate:.0f}% success", flush=True)
            save()

# ============================================================================
# PHASE 4: HI-FI VERIFICATION
# ============================================================================
if not check_deadline() and hours_left() > 1.0:
    print(f"\nPHASE 4: Hi-fi verification (selective)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Hi-fi eval time: {t_hifi:.2f}s → maxfun=30 → ~{30*t_hifi/60:.1f} min/test", flush=True)
    
    # Take lo-fi successes from phase 1 attitude tests, verify with hi-fi
    lofi_successes = [t for t in results["phase1_attitude_only"] if t["success"]]
    # Also run from true params as baseline
    hifi_tests = [(true_params.copy(), "hifi_sanity")]
    for t in lofi_successes[:5]:  # max 5 hi-fi tests
        # Start from lo-fi optimum (handoff)
        x_lofi = np.array(t["x_best"])
        hifi_tests.append((x_lofi, f"hifi_handoff_{t['label']}"))
    
    for x0, label in hifi_tests:
        if check_deadline(): break
        remaining_mins = hours_left() * 60
        est_mins = 30 * t_hifi / 60
        if remaining_mins < est_mins + 5:
            print(f"  Skipping {label}: need {est_mins:.0f}min, {remaining_mins:.0f}min left", flush=True)
            break
        r = run_opt(x0, label, obj_hifi, maxfun=30)
        results["phase4_hifi_verification"].append(r)
        save()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'=' * 70}", flush=True)
print("FINAL SUMMARY", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"Total time: {elapsed():.0f}s ({elapsed()/3600:.1f}h)", flush=True)

for phase_name in ["phase1_attitude_only", "phase1_omega_only", "phase1_combined",
                     "phase2_fine_grid", "phase3_statistical", "phase4_hifi_verification"]:
    tests = results[phase_name]
    if not tests: continue
    s = sum(1 for t in tests if t["success"])
    print(f"  {phase_name}: {s}/{len(tests)} converged", flush=True)

# Convergence boundaries
att_tests = results["phase1_attitude_only"]
if att_tests:
    att_s = [t for t in att_tests if t["success"]]
    att_f = [t for t in att_tests if not t["success"]]
    if att_s:
        print(f"\n  Lo-fi attitude convergence radius: ≤{max(t['initial_att_deg'] for t in att_s):.0f}°", flush=True)
    if att_f:
        print(f"  Lo-fi attitude divergence from:    ≥{min(t['initial_att_deg'] for t in att_f):.0f}°", flush=True)

om_tests = results["phase1_omega_only"]
if om_tests:
    om_s = [t for t in om_tests if t["success"]]
    om_f = [t for t in om_tests if not t["success"]]
    if om_s:
        print(f"  Lo-fi omega convergence radius:    ≤{max(t['initial_omega_dps'] for t in om_s):.4f}°/s", flush=True)
    if om_f:
        print(f"  Lo-fi omega divergence from:       ≥{min(t['initial_omega_dps'] for t in om_f):.4f}°/s", flush=True)

# Phase 2 fine grid summary
fine = results["phase2_fine_grid"]
if fine:
    print(f"\n  Fine grid (5-10°) success rates:", flush=True)
    from collections import defaultdict
    by_deg = defaultdict(list)
    for t in fine:
        deg = int(t["label"].split("_")[1].replace("d", ""))
        by_deg[deg].append(t["success"])
    for deg in sorted(by_deg.keys()):
        vals = by_deg[deg]
        print(f"    {deg}°: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals)*100:.0f}%", flush=True)

hifi = results["phase4_hifi_verification"]
if hifi:
    print(f"\n  Hi-fi verification:", flush=True)
    for t in hifi:
        sym = "✓" if t["success"] else "✗"
        print(f"    {sym} {t['label']}: att={t['final_att_deg']:.2f}° ω={t['final_omega_dps']:.4f}°/s", flush=True)

results["complete"] = True
save()
print(f"\nResults: {RESULTS_FILE}", flush=True)
print("Done.", flush=True)
