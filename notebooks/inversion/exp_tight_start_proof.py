"""
Experiment: Tight-start proof-of-concept

GOAL: Prove that "given a good initial guess, mixed-fidelity inversion works."
This is the minimum viable result for Roberto.

Approach:
  1. Start from true + small perturbation (0.5°, 1°, 2°, 3° attitude + 0.005°/s omega)
  2. Lo-fi L-BFGS-B → handoff to hi-fi L-BFGS-B
  3. Show that the two-stage approach recovers truth

10 trials at each level, hard 5-min total timeout.
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
from src.inversion import (
    ObjectiveFunction, axis_angle_to_quaternion, quaternion_to_axis_angle, normalize_quaternion,
)
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp_tight_start_proof.json"

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70); print("TIGHT-START MIXED-FIDELITY PROOF OF CONCEPT"); print("=" * 70)

# SETUP
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

# TRUE PARAMS
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0],
                     np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

# OBSERVED LIGHTCURVE
print("[2/3] Generating observed lightcurve...", flush=True)
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

print(f"  Setup done in {elapsed():.1f}s", flush=True)

# HELPERS
def perturb_attitude(q0, pert_deg, rng):
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    rad = np.deg2rad(pert_deg)
    qp = np.array([np.cos(rad/2), np.sin(rad/2)*axis[0], np.sin(rad/2)*axis[1], np.sin(rad/2)*axis[2]])
    w = qp[0]*q0[0] - np.dot(qp[1:], q0[1:])
    v = qp[0]*q0[1:] + q0[0]*qp[1:] + np.cross(qp[1:], q0[1:])
    return normalize_quaternion(np.array([w, v[0], v[1], v[2]]))

def compute_errors(x):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(x[:3])
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    att_err = np.rad2deg(2 * np.arccos(dot))
    omega_err = np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))
    return att_err, omega_err

def run_mixed_fidelity(x0, label):
    """Lo-fi L-BFGS-B → Hi-fi L-BFGS-B handoff."""
    att0, om0 = compute_errors(x0)
    bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
    
    # Stage 1: Lo-fi
    n1 = [0]; t1 = time.perf_counter()
    def obj1(x): n1[0] += 1; return obj_lofi.evaluate(x)
    res1 = minimize(obj1, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    t1 = time.perf_counter() - t1
    att1, om1 = compute_errors(res1.x)
    
    # Stage 2: Hi-fi (from lo-fi result)
    n2 = [0]; t2 = time.perf_counter()
    def obj2(x): n2[0] += 1; return obj_hifi.evaluate(x)
    res2 = minimize(obj2, res1.x, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12})
    t2 = time.perf_counter() - t2
    att2, om2 = compute_errors(res2.x)
    
    success = att2 < 5.0 and om2 < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{label}] {sym} init:{att0:.1f}°/{om0:.4f} → lofi:{att1:.2f}°/{om1:.4f} → hifi:{att2:.2f}°/{om2:.4f} | {n1[0]}+{n2[0]}ev {t1:.1f}+{t2:.1f}s", flush=True)
    
    return {
        "label": label, "init_att": round(att0, 2), "init_omega": round(om0, 4),
        "lofi_att": round(att1, 2), "lofi_omega": round(om1, 4), "lofi_evals": n1[0], "lofi_time": round(t1, 1),
        "hifi_att": round(att2, 2), "hifi_omega": round(om2, 4), "hifi_evals": n2[0], "hifi_time": round(t2, 1),
        "success": success, "x_best": res2.x.tolist(),
    }

# RUN TESTS
print("[3/3] Running mixed-fidelity tests...\n", flush=True)
results = {"true_params": true_params.tolist(), "tests": []}

test_configs = [
    (0.5, 0.005, 10),
    (1.0, 0.005, 10),
    (2.0, 0.01, 10),
    (3.0, 0.01, 10),
    (5.0, 0.02, 10),
]

for att_pert, om_pert, n_trials in test_configs:
    print(f"\n--- {att_pert}° attitude + {om_pert}°/s omega ({n_trials} trials) ---", flush=True)
    rng = np.random.default_rng(SEED + int(att_pert * 100))
    successes = 0
    for trial in range(n_trials):
        q_new = perturb_attitude(true_q0, att_pert, rng)
        aa_new = quaternion_to_axis_angle(q_new)
        d = rng.standard_normal(3); d /= np.linalg.norm(d)
        omega_new = true_omega0 + np.deg2rad(om_pert) * d
        x0 = np.concatenate([aa_new, omega_new])
        r = run_mixed_fidelity(x0, f"t{att_pert}d_{om_pert}dps_r{trial}")
        results["tests"].append(r)
        if r["success"]: successes += 1
    
    rate = successes / n_trials * 100
    print(f"  >> {att_pert}° + {om_pert}°/s: {successes}/{n_trials} = {rate:.0f}%", flush=True)

# SUMMARY
print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
print(f"Total time: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)

from collections import defaultdict
by_level = defaultdict(list)
for t in results["tests"]:
    level = f"{t['init_att']:.1f}°+{t['init_omega']:.3f}°/s"
    by_level[level].append(t["success"])

for level, vals in by_level.items():
    print(f"  {level}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals)*100:.0f}%", flush=True)

# Highlight best result
best = min(results["tests"], key=lambda t: t["hifi_att"] + t["hifi_omega"]*10)
print(f"\n  Best result: {best['label']}", flush=True)
print(f"    Init: {best['init_att']:.1f}° att, {best['init_omega']:.4f}°/s ω", flush=True)
print(f"    Lo-fi: {best['lofi_att']:.2f}° att, {best['lofi_omega']:.4f}°/s ω ({best['lofi_evals']} evals, {best['lofi_time']}s)", flush=True)
print(f"    Hi-fi: {best['hifi_att']:.2f}° att, {best['hifi_omega']:.4f}°/s ω ({best['hifi_evals']} evals, {best['hifi_time']}s)", flush=True)

results["complete"] = True
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": round(elapsed(), 1), "results": results}, indent=2, default=str))
print(f"\nResults: {RESULTS_FILE}", flush=True)
print("Done.", flush=True)
